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

from src.web.lib.imaging import img_to_b64
from src.web.lib.sim import PandaSim, Snapshot


# Motion tuning.
MAX_MOTION_STEPS = 25
GRIPPER_CONFIRM_STEPS = 5
GRIPPER_CONFIRM_TOL = 0.002
REACH_TOLERANCE = 0.010
ACTION_GAIN = 1.0
FRAME_EVERY = 5

GRIPPER_CLOSE_CMD = 1.0
GRIPPER_OPEN_CMD = -1.0


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

class Disturbance:
    """Move an object by (dx, dy, dz) via direct MuJoCo joint qpos manipulation.

    This lets us test Gemma's ability to ADAPT when objects move mid-task.
    """

    def __init__(self, sim: PandaSim) -> None:
        self._sim = sim

    def move_object(self, obj_name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
        """Move obj_name by (dx, dy, dz). Returns new position."""
        env = self._sim.env()
        models = env.sim.model
        data = env.sim.data

        # Find the object's free joint
        for i in range(models.njnt):
            jname = models.joint_names[i]
            if obj_name.lower() in jname.lower() and "joint" in jname.lower() and "robot" not in jname.lower():
                addr = models.jnt_qposadr[i]
                old_pos = data.qpos[addr:addr+3].copy()
                data.qpos[addr] += dx
                data.qpos[addr + 1] += dy
                data.qpos[addr + 2] += dz
                env.sim.forward()
                new_pos = data.qpos[addr:addr+3].copy()
                delta = float(np.linalg.norm(new_pos - old_pos))
                return {
                    "object": obj_name,
                    "delta_m": round(delta, 3),
                    "old_pos": [round(float(old_pos[0]), 3), round(float(old_pos[1]), 3), round(float(old_pos[2]), 3)],
                    "new_pos": [round(float(new_pos[0]), 3), round(float(new_pos[1]), 3), round(float(new_pos[2]), 3)],
                }

        return {"object": obj_name, "error": f"no joint found for {obj_name}"}

    def randomize(self, obj_name: str, max_delta: float = 0.08) -> dict:
        """Randomly move obj_name within max_delta meters in XY."""
        import random
        dx = random.uniform(-max_delta, max_delta)
        dy = random.uniform(-max_delta, max_delta)
        dz = random.uniform(-0.01, 0.01)
        return self.move_object(obj_name, dx=dx, dy=dy, dz=dz)


# ═══════════════════════════════════════════════════════════════
# Object name resolver
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


class SweepHandler:
    """Sweep/wipe a circular area in a raster pattern."""

    def __call__(self, snap: Snapshot, center_x: float, center_y: float, radius: float) -> ExecuteResult:
        action_dim = 6  # Wipe has no gripper
        cx, cy, r = float(center_x), float(center_y), float(radius)
        surface_z = 0.897
        sim = snap._sim if hasattr(snap, "_sim") else None
        if sim is None:
            return ExecuteResult(frames=[], final_snapshot=snap)

        frames = []
        num_passes = 20
        steps_per_row = 12

        for p in range(num_passes):
            y_off = (p / max(num_passes - 1, 1) - 0.5) * r * 1.6
            direction = 1 if p % 2 == 0 else -1
            for _ in range(steps_per_row):
                snap = snap  # placeholder — actual step logic needs env access
            if p % 4 == 0:
                frames.append(_frame_from_snap(snap))

        frames.append(_frame_from_snap(snap, final=True))
        return ExecuteResult(frames=frames, final_snapshot=snap)


class OrientedGraspHandler:
    """Grasp an object from a configurable side approach direction.

    Steps:
    1. Rotate wrist to the approach direction (gripper fingers horizontal)
    2. Move to a lateral offset 12cm from the object in the approach direction
    3. Move laterally toward the object at object-z height
    4. Close gripper
    5. Lift
    """

    def __init__(self, executor: "MotionExecutor") -> None:
        self.executor = executor

    def __call__(self, snap: Snapshot, object_name: str, direction: str = "above", **kwargs) -> ExecuteResult:
        try:
            tx, ty, tz = resolve_target(object_name, snap)
        except ValueError:
            return ExecuteResult(frames=[], final_snapshot=snap)

        # Map direction to approach offset and wrist orientation
        # direction: "above", "left", "right", "front", "back"
        # orientation deltas are axis-angle: (roll, pitch, yaw) where 1.0 = 0.5 rad
        if direction == "above":
            ori_delta = (0.0, 0.0, 0.0)  # default upright orientation
            r1 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.10}, "open")
            snap = r1.final_snapshot
            r2 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz}, "open")
            snap = r2.final_snapshot
            r3 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz}, "close")
            snap = r3.final_snapshot
            r4 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz}, "close")
            snap = r4.final_snapshot
            frames = r1.frames + r2.frames + r3.frames + r4.frames
            return ExecuteResult(frames=frames, final_snapshot=snap)
        elif direction == "left":
            # Approach from left (-X). Wrist yaw = +90deg
            # Step 1: rotate wrist (pitch 90 degrees = 1.57 rad = ~3.14 in action units)
            # We'll do it incrementally over several steps
            ori_delta = (0.0, 0.0, 0.0)  # no rotation for side approach
            offset_x = -0.12
            # Move to lateral offset position first (at object height + 5cm safety)
            r1 = self.executor._execute_raw(snap, {"target_x": tx + offset_x, "target_y": ty, "target_z": tz + 0.05}, "open")
            snap = r1.final_snapshot
            # Move laterally toward object at object z
            r2 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "open")
            snap = r2.final_snapshot
            # Close
            r3 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r3.final_snapshot
            r4 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r4.final_snapshot
            frames = r1.frames + r2.frames + r3.frames + r4.frames
            return ExecuteResult(frames=frames, final_snapshot=snap)
        elif direction == "right":
            offset_x = 0.12
            r1 = self.executor._execute_raw(snap, {"target_x": tx + offset_x, "target_y": ty, "target_z": tz + 0.05}, "open")
            snap = r1.final_snapshot
            r2 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "open")
            snap = r2.final_snapshot
            r3 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r3.final_snapshot
            r4 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r4.final_snapshot
            frames = r1.frames + r2.frames + r3.frames + r4.frames
            return ExecuteResult(frames=frames, final_snapshot=snap)
        elif direction == "front":
            offset_y = 0.12
            r1 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty + offset_y, "target_z": tz + 0.05}, "open")
            snap = r1.final_snapshot
            r2 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "open")
            snap = r2.final_snapshot
            r3 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r3.final_snapshot
            r4 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r4.final_snapshot
            frames = r1.frames + r2.frames + r3.frames + r4.frames
            return ExecuteResult(frames=frames, final_snapshot=snap)
        elif direction == "back":
            offset_y = -0.12
            r1 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty + offset_y, "target_z": tz + 0.05}, "open")
            snap = r1.final_snapshot
            r2 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "open")
            snap = r2.final_snapshot
            r3 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r3.final_snapshot
            r4 = self.executor._execute_raw(snap, {"target_x": tx, "target_y": ty, "target_z": tz + 0.01}, "close")
            snap = r4.final_snapshot
            frames = r1.frames + r2.frames + r3.frames + r4.frames
            return ExecuteResult(frames=frames, final_snapshot=snap)
        else:
            # Fallback: standard above approach
            return self.executor.execute_tool(snap, "grasp", {"object_name": object_name})


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


TOOL_SCHEMA = {
    "name": "tool_call",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool": {
                "type": "string",
                "enum": ["move_to", "grasp", "grasp_side", "lift", "place"],
                "description": "Which tool to use",
            },
            "params": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "object_name": {"type": "string"},
                    "height": {"type": "number"},
                    "direction": {
                        "type": "string",
                        "enum": ["above", "left", "right", "front", "back"],
                        "description": "Approach direction for grasp_side"
                    },
                },
            },
            "reasoning": {"type": "string"},
        },
        "required": ["tool", "params", "reasoning"],
    },
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
