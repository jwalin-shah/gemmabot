"""Per-step verification — does ground-truth physics back up the agent?

The verifier is intentionally dumb: it reads object positions out of the
snapshot, applies the thresholds from tasks.py, and emits a structured
Verdict. It does NOT consult Gemma's self-reported "stage" — that would make
the agent its own judge. The verdict is the demo's claim of correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.web.lib.sim import Snapshot
from src.web.lib.tasks import (
    DOOR_OPEN_ANGLE,
    GRASP_TOL,
    LIFT_HEIGHT,
    NUT_ON_PEG_Z,
    PLACE_TOL,
    REACH_TOL,
    TABLE_Z,
    WIPE_THRESHOLD,
    TaskSpec,
)


@dataclass
class Verdict:
    task: str
    mode: str
    reached: bool
    grasped: bool
    lifted: bool
    placed: bool
    success: bool
    distance_to_target: float
    object_height: float
    distance_to_goal_xy: float | None
    distance_xy: float
    distance_z: float
    notes: str

    def to_json(self) -> dict:
        return asdict(self)


def verify(snap_before: Snapshot, snap_after: Snapshot, spec: TaskSpec) -> Verdict:
    """Compute a Verdict from the post-motion snapshot."""

    # Door mode: measure hinge angle
    if spec.mode == "door":
        hinge = snap_after.state.get("hinge_qpos", 0.0)
        reached = False
        grasped = False
        lifted = hinge > DOOR_OPEN_ANGLE
        placed = False
        success = lifted
        return Verdict(
            task=spec.key, mode=spec.mode,
            reached=reached, grasped=grasped, lifted=lifted, placed=placed,
            success=success,
            distance_to_target=float(hinge),
            object_height=0.0, distance_to_goal_xy=None,
            distance_xy=0.0, distance_z=0.0,
            notes=f"Door opened {hinge:.2f} rad{' (enough!)' if success else ' (keep pulling)'}",
        )

    # Wipe mode: measure proportion_wiped
    if spec.mode == "wipe":
        wiped = snap_after.state.get("proportion_wiped", 0.0)
        success = wiped >= WIPE_THRESHOLD
        return Verdict(
            task=spec.key, mode=spec.mode,
            reached=False, grasped=False, lifted=False, placed=False,
            success=success,
            distance_to_target=float(wiped),
            object_height=0.0, distance_to_goal_xy=None,
            distance_xy=0.0, distance_z=0.0,
            notes=f"Wipe coverage: {wiped*100:.0f}%{' (done!)' if success else ' (keep sweeping)'}",
        )

    # Object-based modes (pick_place, lift, stack, nut_assembly, clear_table)
    target_pos = snap_after.objects.get(spec.target_object or "")
    spawn_pos = snap_before.objects.get(spec.target_object or "")
    if target_pos is None:
        return Verdict(
            task=spec.key, mode=spec.mode,
            reached=False, grasped=False, lifted=False, placed=False,
            success=False,
            distance_to_target=float("inf"),
            object_height=0.0, distance_to_goal_xy=None,
            distance_xy=float("inf"), distance_z=float("inf"),
            notes=f"target object {spec.target_object!r} missing from snapshot",
        )

    ee = np.asarray(snap_after.ee_pos, dtype=float)
    delta = ee - target_pos
    dxy = float(np.linalg.norm(delta[:2]))
    dz = float(delta[2])
    dxyz = float(np.linalg.norm(delta))
    reached = dxyz < REACH_TOL
    xy_reached = dxy < REACH_TOL  # within 5cm in XY plane

    obj_z = float(target_pos[2])
    spawn_z = float(spawn_pos[2]) if spawn_pos is not None else obj_z
    z_delta = obj_z - spawn_z
    grasped = (dxyz < GRASP_TOL) and (not snap_after.gripper_open) and (z_delta > -0.01)

    lifted = z_delta > LIFT_HEIGHT

    # Nut assembly: lifted means on the peg
    if spec.mode == "nut_assembly":
        lifted = obj_z > NUT_ON_PEG_Z
        placed = lifted
        success = lifted
        distance_xy = None
        return Verdict(
            task=spec.key, mode=spec.mode,
            reached=reached, grasped=grasped, lifted=lifted, placed=placed,
            success=success,
            distance_to_target=round(dxyz, 4),
            object_height=round(obj_z, 4), distance_to_goal_xy=None,
            distance_xy=round(dxy, 4), distance_z=round(dz, 4),
            notes=f"nut z={obj_z:.3f} {'on peg!' if success else 'keep lifting'}",
        )

    distance_xy: float | None = None
    placed = False

    if spec.mode in ("pick_place", "clear_table") and spec.place_xy is not None:
        gx, gy = spec.place_xy
        distance_xy = float(np.linalg.norm(target_pos[:2] - np.array([gx, gy])))
        placed = distance_xy < PLACE_TOL and obj_z < spawn_z + 0.06

    # Clear table: check how many are in the bin
    if spec.mode == "clear_table" and spec.compound_objects:
        px, py = spec.place_xy or (0.10, 0.28)
        placed_count = sum(
            1 for obj in spec.compound_objects
            if snap_after.objects.get(obj) is not None
            and float(np.linalg.norm(
                snap_after.objects[obj][:2] - np.array([px, py])
            )) < PLACE_TOL
            and float(snap_after.objects[obj][2]) < TABLE_Z + 0.06
        )
        # Build actionable feedback: XY distance to target + progress
        progress_bits = [f"{placed_count}/{len(spec.compound_objects)} in bin"]
        if not reached:
            if xy_reached:
                progress_bits.append(
                    f"to {spec.target_object}: XY {dxy*100:.1f}cm \u2705 | "
                    f"Z {dz*100:.1f}cm \u2b07 -- DESCEND to grasp"
                )
            else:
                progress_bits.append(
                    f"to {spec.target_object}: XY {dxy*100:.1f}cm | "
                    f"Z {dz*100:.1f}cm -- move CLOSER in XY"
                )
        if placed_count > 0:
            progress_bits.append("keep going -- pick up remaining objects")
        notes = " | ".join(progress_bits)
        success = placed_count >= len(spec.compound_objects)
        return Verdict(
            task=spec.key, mode=spec.mode,
            reached=reached, grasped=grasped,
            lifted=placed_count > 0, placed=success,
            success=success,
            distance_to_target=round(dxyz, 4),
            object_height=round(obj_z, 4),
            distance_to_goal_xy=round(distance_xy, 4) if distance_xy is not None else None,
            distance_xy=round(dxy, 4), distance_z=round(dz, 4),
            notes=notes,
        )

    success = _success(spec, reached=reached, grasped=grasped, lifted=lifted, placed=placed)

    notes_bits = []
    if not reached:
        if xy_reached:
            notes_bits.append(
                f"XY: {dxy*100:.1f}cm ✅ | Z: {dz*100:.1f}cm ⬇ "
                "hovering above object -- DESCEND to grasp"
            )
        else:
            notes_bits.append(
                f"XY: {dxy*100:.1f}cm | Z: {dz*100:.1f}cm -- "
                "move CLOSER in XY to the object"
            )
    if reached and not grasped:
        notes_bits.append("gripper not closed around object" if snap_after.gripper_open else
                          f"closed but object did not rise ({z_delta*100:+.1f}cm)")
    if grasped and not lifted and spec.mode != "pick_place":
        notes_bits.append(f"lift more — object is {z_delta*100:.1f}cm up, need >{LIFT_HEIGHT*100:.0f}cm")
    if lifted and spec.mode == "pick_place" and not placed:
        notes_bits.append(f"object {distance_xy*100:.1f}cm from drop zone" if distance_xy is not None else "no drop zone")
    if not notes_bits:
        notes_bits.append("looking good")

    return Verdict(
        task=spec.key, mode=spec.mode,
        reached=reached, grasped=grasped, lifted=lifted, placed=placed,
        success=success,
        distance_to_target=round(dxyz, 4),
        object_height=round(obj_z, 4),
        distance_to_goal_xy=None if distance_xy is None else round(distance_xy, 4),
        distance_xy=round(dxy, 4), distance_z=round(dz, 4),
        notes=" | ".join(notes_bits),
    )


def _success(spec: TaskSpec, *, reached: bool, grasped: bool, lifted: bool, placed: bool) -> bool:
    if spec.mode == "pick_place":
        return placed
    if spec.mode == "lift":
        return lifted
    if spec.mode == "stack":
        return lifted  # refined by env_success
    return False


def env_success(snap: Snapshot, spec: TaskSpec) -> bool:
    """Mode-specific success override.

    - door: hinge_qpos > threshold
    - wipe: proportion_wiped > threshold
    - nut_assembly: nut z > peg height
    - clear_table: all compound objects in bin2
    - stack: cubeA above cubeB, xy-aligned
    """
    if spec.mode == "stack":
        return stack_success(snap, spec)
    if spec.mode == "door":
        return snap.state.get("hinge_qpos", 0.0) > DOOR_OPEN_ANGLE
    if spec.mode == "wipe":
        return snap.state.get("proportion_wiped", 0.0) >= WIPE_THRESHOLD
    if spec.mode == "nut_assembly" and snap.objects.get(spec.target_object or "") is not None:
        return float(snap.objects[spec.target_object][2]) > NUT_ON_PEG_Z
    if spec.mode == "clear_table" and spec.compound_objects:
        px, py = spec.place_xy or (0.10, 0.28)
        placed_count = sum(
            1 for obj in spec.compound_objects
            if snap.objects.get(obj) is not None
            and float(np.linalg.norm(snap.objects[obj][:2] - np.array([px, py]))) < PLACE_TOL
            and float(snap.objects[obj][2]) < TABLE_Z + 0.06
        )
        return placed_count >= len(spec.compound_objects)
    return False


def stack_success(snap: Snapshot, spec: TaskSpec) -> bool:
    if spec.mode != "stack" or spec.secondary_object is None:
        return False
    a = snap.objects.get(spec.target_object or "")
    b = snap.objects.get(spec.secondary_object)
    if a is None or b is None:
        return False
    xy_dist = float(np.linalg.norm(a[:2] - b[:2]))
    return xy_dist < 0.03 and (a[2] - b[2]) > 0.025
