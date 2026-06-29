"""Vision grounding / blinding layer -- Gemma sees only camera-derived positions.

This module is the anti-cheat layer: it prevents Gemma from reading ground-truth
object positions out of the simulator. Instead, everything Gemma perceives comes
from camera pixels (instance segmentation -> depth back-projection), giving the
model a realistic "vision-only" view of the world.

The vision system has three stages:
  1. Instance-segmentation perceiver (precise, needs seg+depths obs)
  2. Color-contour fallback (needs only RGB + depth)
  3. Ground-truth error computation (judge only, never shown to Gemma)

Key invariant: no string in the prompt should ever say "ground truth".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.web.lib.perception import (
    Detection,
    StateMap,
    perceive_instances,
    backproject,
    ColorContourPerceptor,
    _get_table_z,
)
from src.web.lib.sim import Snapshot
from src.web.lib.tasks import TaskSpec


# ---------------------------------------------------------------------------
# GroundedBelief -- the vision-only world state shown to Gemma
# ---------------------------------------------------------------------------

@dataclass
class GroundedBelief:
    """Vision-derived belief about object positions in the scene.

    This is the ONLY object-position data Gemma ever sees. It is constructed
    purely from camera pixels + depth, never from simulator ground truth.

    Fields:
        detections: list of vision Perception.Detection objects
        camera: camera name used for detection
        gt_positions: dict of {label: (x,y,z)} from ground truth (JUDGE ONLY)
        errors: dict of {label: L2_error_m} computed against ground truth
    """
    detections: list[Detection] = field(default_factory=list)
    camera: str = "birdview"
    gt_positions: dict | None = None
    errors: dict | None = None

    def as_prompt_block(self) -> str:
        """Build the text block Gemma sees in its prompt.

        This is the replacement for _objects_text() in brain.py. It NEVER
        mentions ground truth. Coordinates come from camera depth back-projection.
        """
        if not self.detections:
            return (
                "PERCEIVED OBJECTS (from camera vision, not ground truth): "
                "none detected.\n"
            )
        lines = ["PERCEIVED OBJECTS (from camera vision, not ground truth):"]
        for i, d in enumerate(self.detections):
            name = d.label or f"{d.color_name()} object #{i}"
            if d.world_xyz is not None:
                x, y, z = d.world_xyz
                lines.append(
                    f"  {name}: world=({x:+.3f}, {y:+.3f}, {z:+.3f}) "
                    f"[{d.source} conf={d.confidence:.2f}, area={d.area_px}px]"
                )
            else:
                lines.append(
                    f"  {name}: pixel=({d.cx},{d.cy}) "
                    f"[{d.source} conf={d.confidence:.2f}, area={d.area_px}px]"
                )
        return "\n".join(lines) + "\n"

    def as_object_dict(self) -> dict[str, np.ndarray]:
        """Return detections as {label: np.array([x, y, z])} for snapshot compat.

        Only includes detections that have both a label and world_xyz set.
        """
        out: dict[str, np.ndarray] = {}
        for d in self.detections:
            if d.label is not None and d.world_xyz is not None:
                out[d.label] = np.array(d.world_xyz, dtype=float)
        return out

    def num_detected(self) -> int:
        return len([d for d in self.detections if d.world_xyz is not None])


# ---------------------------------------------------------------------------
# VisionGroundingModule -- orchestrates the perception pipeline
# ---------------------------------------------------------------------------

class VisionGroundingModule:
    """Orchestrates vision-only perception: instance seg -> depth -> belief.

    Usage:
        vgm = VisionGroundingModule(sim, env_model, task_spec)
        obs = sim.env()._get_observations(force_update=True)
        belief = vgm.perceive(obs, gt_snapshot=some_snapshot)
        prompt_block = belief.as_prompt_block()
    """

    def __init__(
        self,
        sim: Any,
        env_model: Any,
        task_spec: TaskSpec,
        height: int = 384,
        width: int = 384,
    ) -> None:
        self._sim = sim
        self._env_model = env_model
        self._task_spec = task_spec
        self._height = height
        self._width = width
        # Lazy-init fallback perceptor
        self._contour_perceptor: ColorContourPerceptor | None = None

    def _fallback_detect(
        self, obs: dict, camera: str
    ) -> tuple[list[Detection], Optional[list[tuple[float, float, float]]]]:
        """Run color-contour detection with depth back-projection."""
        rgb = obs.get(f"{camera}_image")
        depth_obs = obs.get(f"{camera}_depth")
        if rgb is None:
            return [], None

        if self._contour_perceptor is None:
            self._contour_perceptor = ColorContourPerceptor()
        dets = self._contour_perceptor.detect(rgb)

        worlds: Optional[list[tuple[float, float, float]]] = None
        if depth_obs is not None and dets:
            pixels = [(d.cx, d.cy) for d in dets]
            worlds = backproject(
                self._sim, depth_obs, camera,
                self._height, self._width, pixels,
            )
        return dets, worlds

    def perceive(
        self,
        obs: dict,
        gt_snapshot: Snapshot | None = None,
        camera: str = "birdview",
    ) -> GroundedBelief:
        """Run the full vision pipeline and return a GroundedBelief.

        Args:
            obs: Raw observation dict from robosuite (includes _image, _depth,
                 _segmentation_instance).
            gt_snapshot: Optional ground-truth Snapshot for error computation
                         (judge only, never shown to Gemma).
            camera: Camera name to use (default "birdview").

        Returns:
            GroundedBelief with detections derived purely from camera pixels.
        """
        # Step 1: Try instance-segmentation perceiver (precise, needs seg obs)
        state_map = perceive_instances(
            self._sim, self._env_model, obs, camera,
            height=self._height, width=self._width,
        )

        # Step 2: Fallback to color-contour if instance seg failed
        if not state_map.detections:
            fallback_dets, fallback_worlds = self._fallback_detect(obs, camera)
            if fallback_dets and fallback_worlds:
                for d, w in zip(fallback_dets, fallback_worlds):
                    d.world_xyz = w
                    # Label by color for the fallback path
                    d.label = f"object_{d.color_name()}"
                state_map = StateMap(detections=fallback_dets, camera=camera)

        # Step 3: Compute L2 errors against ground truth (judge only)
        errors: dict[str, float] | None = None
        gt_positions: dict[str, tuple[float, float, float]] | None = None

        if gt_snapshot is not None and state_map.detections:
            gt_positions = {}
            errors = {}
            for d in state_map.detections:
                label = d.label
                if label is None or d.world_xyz is None:
                    continue
                gt_arr = gt_snapshot.objects.get(label)
                if gt_arr is not None:
                    gt_xyz = (float(gt_arr[0]), float(gt_arr[1]), float(gt_arr[2]))
                    gt_positions[label] = gt_xyz
                    dx = d.world_xyz[0] - gt_xyz[0]
                    dy = d.world_xyz[1] - gt_xyz[1]
                    dz = d.world_xyz[2] - gt_xyz[2]
                    errors[label] = round(float(np.sqrt(dx**2 + dy**2 + dz**2)), 4)

        return GroundedBelief(
            detections=state_map.detections,
            camera=state_map.camera,
            gt_positions=gt_positions,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# resolve_vision_target -- look up an object in vision belief
# ---------------------------------------------------------------------------

def resolve_vision_target(
    target_name: str,
    belief: GroundedBelief,
) -> tuple[float, float, float]:
    """Look up an object's (x, y, z) from vision-derived belief.

    Exact match first, then case-insensitive substring fallback.

    Args:
        target_name: Object label to find (e.g. "cube", "Can", "object_red").
        belief: GroundedBelief from the vision pipeline.

    Returns:
        (x, y, z) world coordinates.

    Raises:
        ValueError if the target is not found in the belief.
    """
    # Exact match
    for d in belief.detections:
        if d.label == target_name and d.world_xyz is not None:
            return (
                float(d.world_xyz[0]),
                float(d.world_xyz[1]),
                float(d.world_xyz[2]),
            )

    # Case-insensitive substring match
    target_lower = target_name.lower()
    for d in belief.detections:
        if d.label is not None and target_lower in d.label.lower():
            if d.world_xyz is not None:
                return (
                    float(d.world_xyz[0]),
                    float(d.world_xyz[1]),
                    float(d.world_xyz[2]),
                )

    raise ValueError(
        f"Object {target_name!r} not found in vision belief. "
        f"Detected labels: {[d.label for d in belief.detections if d.label is not None]}"
    )
