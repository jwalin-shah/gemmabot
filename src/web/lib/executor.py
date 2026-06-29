"""Motion executor with tool-calling + disturbance support.

Two execution modes:
  1. Legacy mode: XYZ + gripper targets (for backward compat / fine control)
  2. Tool mode: Gemma calls named tools with parameters

Each tool is a self-contained handler that knows how to accomplish one
high-level skill in physics. This is the primary interface Gemma uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.web.lib.grounding import GroundedBelief, resolve_vision_target
from src.web.lib.imaging import img_to_b64
from src.web.lib.sim import PandaSim, Snapshot
from src.config import (
    MOTION_MAX_STEPS as MAX_MOTION_STEPS,
    MOTION_REACH_TOLERANCE as REACH_TOLERANCE,
    MOTION_ACTION_GAIN as ACTION_GAIN,
    MOTION_FRAME_EVERY as FRAME_EVERY,
    MOTION_GRIPPER_CONFIRM_STEPS as GRIPPER_CONFIRM_STEPS,
    GRIPPER_CONFIRM_TOL,
    GRIPPER_CLOSE_CMD,
    GRIPPER_OPEN_CMD,
)


@dataclass
class ExecuteResult:
    frames: list[dict]
    final_snapshot: Snapshot


@dataclass
class ToolDef:
    """Definition of one tool Gemma can call."""
    name: str
    params: dict[str, str]
    description: str
    handler: Callable = field(repr=False)


# ═══════════════════════════════════════════════════════════════
# Disturbance — move objects mid-task to test adaptation
# ═══════════════════════════════════════════════════════════════

def resolve_target(target_name: str, snap: "Snapshot") -> tuple[float, float, float]:
    """Look up an object's (x, y, z) from the snapshot by name.
    Exact match first, then case-insensitive substring fallback."""
    target_pos = snap.objects.get(target_name)
    if target_pos is not None:
        return (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
    for k, v in snap.objects.items():
        if target_name.lower() in k.lower():
            return (float(v[0]), float(v[1]), float(v[2]))
    raise ValueError(f"Object {target_name!r} not found in {list(snap.objects)}")


# ═══════════════════════════════════════════════════════════════
# Tool Handlers
# ═══════════════════════════════════════════════════════════════

class MoveToHandler:
    """Move end-effector to an absolute XYZ position."""

    def __init__(self, executor: "MotionExecutor") -> None:
        self.executor = executor

    def __call__(self, snap: Snapshot, x: float | None = None, y: float | None = None, z: float | None = None, target: str | None = None, **kwargs) -> ExecuteResult:
        if target is not None:
            tx, ty, tz = resolve_target(target, snap)
            return self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.10}, "hold")
        return self.executor._execute_raw(snap, {"target_x": x, "target_y": y, "target_z": z}, "hold")


class GraspHandler:
    """Grasp a named object: approach, center, close, lift."""

    def __init__(self, executor: "MotionExecutor") -> None:
        self.executor = executor
        self._obj_name: str = ""

    def __call__(self, snap: Snapshot, object_name: str, **kwargs) -> ExecuteResult:
        self._obj_name = object_name
        try:
            tx, ty, tz = resolve_target(object_name, snap)
        except ValueError:
            return ExecuteResult(frames=[], final_snapshot=snap)

        # Step 1: approach from above (10cm over object)
        r1 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.10}, "open")
        snap = r1.final_snapshot

        # Step 2: descend to same height as object (not below - prevent floor collision)
        r2 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz - 0.02}, "open")
        snap = r2.final_snapshot

        # Step 3: close gripper around object
        r3 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz - 0.02}, "close")
        snap = r3.final_snapshot

        # Step 4: hold closed to settle fingers
        r4 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz - 0.02}, "close")
        snap = r4.final_snapshot

        frames = r1.frames + r2.frames + r3.frames + r4.frames
        return ExecuteResult(frames=frames, final_snapshot=snap)


class LiftHandler:
    """Lift the currently grasped object upward."""

    def __init__(self, executor: "MotionExecutor") -> None:
        self.executor = executor

    def __call__(self, snap: Snapshot, height: float = 0.15, **kwargs) -> ExecuteResult:
        ee = snap.ee_pos
        target_z = float(ee[2]) + height
        r = self.executor._execute_raw(snap, {"target_x": float(ee[0]), "target_y": float(ee[1]), "target_z": target_z}, "close")
        return r


class PlaceHandler:
    """Place held object at (x, y) on the table surface."""

    def __init__(self, executor: "MotionExecutor") -> None:
        self.executor = executor

    def __call__(self, snap: Snapshot, x: float, y: float, z: float | None = None, **kwargs) -> ExecuteResult:
        table_z = z or 0.85
        # Descend to placement position
        r1 = self.executor._execute_raw(snap, {"target_x": x, "target_y": y, "target_z": table_z + 0.05}, "close")
        snap = r1.final_snapshot
        # Lower to surface
        r2 = self.executor._execute_raw(snap, {"target_x": x, "target_y": y, "target_z": table_z + 0.01}, "close")
        snap = r2.final_snapshot
        # Open gripper
        r3 = self.executor._execute_raw(snap, {"target_x": x, "target_y": y, "target_z": table_z + 0.01}, "open")

        frames = r1.frames + r2.frames + r3.frames
        return ExecuteResult(frames=frames, final_snapshot=r3.final_snapshot)


class OrientedGraspHandler:
    """Grasp a named object from a specified side direction.

    Directions: above (top-down), left, right, front, back.
    Used for flat or wide objects where a standard top-down grasp won't work.
    """

    _APPROACH_OFFSETS: dict[str, tuple[float, float]] = {
        "above": (0.0, 0.0),
        "left": (-0.08, 0.0),
        "right": (0.08, 0.0),
        "front": (0.0, -0.08),
        "back": (0.0, 0.08),
    }

    def __init__(self, executor: "MotionExecutor") -> None:
        self.executor = executor
        self._obj_name: str = ""

    def __call__(self, snap: Snapshot, object_name: str, direction: str = "above", **kwargs) -> ExecuteResult:
        self._obj_name = object_name
        try:
            tx, ty, tz = resolve_target(object_name, snap)
        except ValueError:
            return ExecuteResult(frames=[], final_snapshot=snap)

        ox, oy = self._APPROACH_OFFSETS.get(direction, (0.0, 0.0))

        # Step 1: approach from the side at 10cm over the object
        r1 = self.executor._execute_raw(
            snap,
            {"target_x": tx + ox, "target_y": ty + oy, "target_z": tz + 0.10},
            "open",
        )
        snap = r1.final_snapshot

        # Step 2: move laterally to object center at grasp height
        r2 = self.executor._execute_raw(
            snap,
            {"target_x": tx, "target_y": ty, "target_z": tz + 0.02},
            "open",
        )
        snap = r2.final_snapshot

        # Step 3: descend slightly and close gripper
        r3 = self.executor._execute_raw(
            snap,
            {"target_x": tx, "target_y": ty, "target_z": tz - 0.01},
            "close",
        )
        snap = r3.final_snapshot

        # Step 4: hold closed to settle fingers
        r4 = self.executor._execute_raw(
            snap,
            {"target_x": tx, "target_y": ty, "target_z": tz - 0.01},
            "close",
        )
        snap = r4.final_snapshot

        frames = r1.frames + r2.frames + r3.frames + r4.frames
        return ExecuteResult(frames=frames, final_snapshot=snap)


# ═══════════════════════════════════════════════════════════════
# Tool Dispatcher
# ═══════════════════════════════════════════════════════════════

def build_tools(executor: "MotionExecutor") -> dict[str, ToolDef]:
    """Build the tool registry Gemma can call."""
    mh = MoveToHandler(executor)
    gh = GraspHandler(executor)
    oh = OrientedGraspHandler(executor)
    lh = LiftHandler(executor)
    ph = PlaceHandler(executor)

    return {
        tool.name: tool for tool in [
            ToolDef("move_to", {"x": "float", "y": "float", "z": "float"},
                    "Move end-effector to absolute position (x, y, z). Gripper stays as-is.", mh),
            ToolDef("grasp", {"object_name": "string"},
                    "Grasp the named object (Can, Milk, Bread, Cereal, cube, cubeA, SquareNut, etc). Handles approach, close, and lift.", gh),
            ToolDef("grasp_side", {"object_name": "string", "direction": "string"},
                    "Grasp object from the side. Directions: above, left, right, front, back. Use for flat or wide objects where top-down grasp won't work.", oh),
            ToolDef("lift", {"height": "float (optional, default 0.15)"},
                    "Lift the currently grasped object upward by <height> meters. Keep gripper closed.", lh),
            ToolDef("place", {"x": "float", "y": "float"},
                    "Place held object at table position (x, y). Descends and opens gripper.", ph),
        ]
    }




# ═══════════════════════════════════════════════════════════════
# Motion Executor (legacy + tool dispatch)
# ═══════════════════════════════════════════════════════════════

class MotionExecutor:
    """Stateful executor with tool-calling and raw motion support."""

    def __init__(self, sim: PandaSim) -> None:
        self._sim = sim
        self._desired_closed: bool = False

    @property
    def desired_closed(self) -> bool:
        return self._desired_closed

    def seed_from(self, snap: Snapshot) -> None:
        self._desired_closed = not snap.gripper_open

    def _gripper_cmd(self) -> float:
        return GRIPPER_CLOSE_CMD if self._desired_closed else GRIPPER_OPEN_CMD

    def _resolve_intent(self, gripper_action: str) -> bool:
        prev = self._desired_closed
        if gripper_action == "close":
            self._desired_closed = True
        elif gripper_action == "open":
            self._desired_closed = False
        return prev != self._desired_closed

    # ── Raw motion (used internally by tool handlers) ──

    def _execute_raw(self, snap: Snapshot, intent_target: dict, gripper_action: str, ori_delta: tuple[float, float, float] | None = None) -> ExecuteResult:
        gripper_changed = self._resolve_intent(gripper_action)
        action_dim = self._sim.env().action_spec[0].shape[0]

        tx = float(intent_target.get("target_x", snap.ee_pos[0]))
        ty = float(intent_target.get("target_y", snap.ee_pos[1]))
        tz = float(intent_target.get("target_z", snap.ee_pos[2]))

        frames: list[dict] = []

        for i in range(MAX_MOTION_STEPS):
            ee = snap.ee_pos
            dx, dy, dz = tx - float(ee[0]), ty - float(ee[1]), tz - float(ee[2])
            dist = float(np.linalg.norm([dx, dy, dz]))
            if dist < REACH_TOLERANCE:
                break

            direction = np.array([dx, dy, dz]) / max(dist, 0.001)
            speed = ACTION_GAIN if dist > 0.03 else max(dist / 0.03 * ACTION_GAIN, 0.05)
            delta = direction * speed
            action = np.zeros(action_dim, dtype=np.float32)
            action[0:3] = delta[:3]
            if ori_delta is not None:
                action[3:6] = ori_delta
            if action_dim >= 7:
                action[6] = self._gripper_cmd()
            action = np.clip(action, -ACTION_GAIN, ACTION_GAIN)
            snap = self._sim.step(action)
            if i % FRAME_EVERY == 0:
                frames.append(_frame_from_snap(snap))

        confirm_steps = GRIPPER_CONFIRM_STEPS if gripper_changed else 3
        prev_qpos = snap.gripper_qpos
        for _ in range(confirm_steps):
            action = np.zeros(action_dim, dtype=np.float32)
            if action_dim >= 7:
                action[6] = self._gripper_cmd()
            snap = self._sim.step(action)
            if abs(snap.gripper_qpos - prev_qpos) < GRIPPER_CONFIRM_TOL:
                break
            prev_qpos = snap.gripper_qpos

        frames.append(_frame_from_snap(snap, final=True))
        return ExecuteResult(frames=frames, final_snapshot=snap)

    # ── Tool dispatch ──

    def execute_tool(self, snap: Snapshot, tool_name: str, params: dict) -> ExecuteResult:
        """Execute a named tool with parameters."""
        tools = build_tools(self)
        if tool_name not in tools:
            raise ValueError(f"Unknown tool: {tool_name}, known: {list(tools)}")
        td = tools[tool_name]
        return td.handler(snap, **params)

    def execute_tool_vision(self, snap: Snapshot, tool_name: str, params: dict, belief: GroundedBelief) -> ExecuteResult:
        """Execute a named tool using vision-derived positions instead of ground truth.

        Creates a snapshot copy with object positions from the vision belief,
        then dispatches through the normal execute_tool path. This keeps the
        tool handlers unchanged - they see vision-derived coordinates via
        resolve_target() on the modified snapshot's objects dict.

        Args:
            snap: Current ground-truth snapshot (used for EE pose, cameras).
            tool_name: Tool name (move_to, grasp, grasp_side, lift, place).
            params: Tool parameters dictionary.
            belief: GroundedBelief with vision-derived object positions.

        Returns:
            ExecuteResult with frames and final vision-stepped snapshot.
            On ValueError (target not found in vision), returns empty frames
            and the original snapshot.
        """
        # Build a snapshot with vision-derived object positions
        from dataclasses import replace

        vision_objects = belief.as_object_dict()
        vision_snap = replace(snap, objects=vision_objects)

        try:
            return self.execute_tool(vision_snap, tool_name, params)
        except ValueError:
            # Target not found in vision belief -- return gracefully
            return ExecuteResult(frames=[], final_snapshot=snap)

    # ── Legacy execute (for backward compat with non-tool mode) ──

    def execute(self, snap: Snapshot, intent_target: dict, gripper_action: str) -> ExecuteResult:
        return self._execute_raw(snap, intent_target, gripper_action)


def _frame_from_snap(snap: Snapshot, final: bool = False) -> dict:
    payload = {
        "birdview": img_to_b64(snap.birdview),
        "frontview": img_to_b64(snap.frontview),
        "ee": [round(float(snap.ee_pos[i]), 3) for i in range(3)],
        "gripper_open": snap.gripper_open,
    }
    if final:
        payload["final"] = True
    return payload
