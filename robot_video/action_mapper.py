"""Map between human-interpretable intents and robot action vectors.

Provides heuristic intent inference from consecutive frames, and bidirectional
conversion between Gemma 4 structured intents and LeRobot action spaces.

Task types:
  - "aloha_mobile_cabinet"  — 7-DOF (6dof + gripper)
  - "pusht"                 — 2D XY pusher position
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Action space metadata
# ---------------------------------------------------------------------------

_TASK_SPECS: dict[str, dict[str, Any]] = {
    "aloha_mobile_cabinet": {
        "dim": 7,
        "gripper_idx": 6,
        "position_indices": (0, 1, 2),      # x, y, z
        "orientation_indices": (3, 4, 5),   # roll, pitch, yaw
        "gripper_threshold": 0.3,            # change above this = toggling grasp
        "movement_threshold": 0.02,          # position delta above this = intentional move
    },
    "pusht": {
        "dim": 2,
        "gripper_idx": None,                  # no gripper
        "position_indices": (0, 1),           # x, y
        "orientation_indices": None,
        "gripper_threshold": 0.5,
        "movement_threshold": 5.0,            # pixels
    },
    "aloha_sim_transfer_cube": {
        "dim": 7,
        "gripper_idx": 6,
        "position_indices": (0, 1, 2),
        "orientation_indices": (3, 4, 5),
        "gripper_threshold": 0.3,
        "movement_threshold": 0.02,
    },
}


def task_specific_action_space(task_type: str) -> dict:
    """Return action-space metadata for a given task type.

    Returns dict with keys:
        dim, gripper_idx, position_indices, orientation_indices,
        gripper_threshold, movement_threshold
    """
    spec = _TASK_SPECS.get(task_type)
    if spec is not None:
        return dict(spec)
    # Default fallback for unknown task types
    return {
        "dim": 7,
        "gripper_idx": 6,
        "position_indices": (0, 1, 2),
        "orientation_indices": (3, 4, 5),
        "gripper_threshold": 0.3,
        "movement_threshold": 0.02,
    }


# ---------------------------------------------------------------------------
# Intent inference (heuristic from raw state deltas)
# ---------------------------------------------------------------------------

@dataclass
class InferredIntent:
    """Human-readable interpretation of a frame-to-frame action."""
    action_type: str               # grasp | release | move_to | pull | push | stop | orient
    target_position: list[float] | None = None  # approximate 3D position
    target_object: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


def infer_human_intent(
    action: list[float],
    prev_action: list[float] | None,
    state: list[float],
    prev_state: list[float] | None,
    task_type: str = "aloha_mobile_cabinet",
) -> tuple[str, dict]:
    """Heuristic intent inference from raw action and state deltas.

    Returns
        (action_type: str, metadata: dict)

    Action types:
        "grasp"      — gripper closing
        "release"    — gripper opening
        "move_to"    — significant position change
        "pull"       — movement toward the robot base (decreasing x in ALOHA)
        "push"       — movement away from base
        "stop"       — negligible action (still)
        "orient"     — orientation-only change
    """
    spec = task_specific_action_space(task_type)
    dim = spec["dim"]
    pos_idxs = spec["position_indices"]
    grip_idx = spec["gripper_idx"]
    grip_thresh = spec["gripper_threshold"]
    move_thresh = spec["movement_threshold"]

    # Pad actions if too short
    act = list(action)
    while len(act) < dim:
        act.append(0.0)

    if prev_action is not None:
        prev = list(prev_action)
        while len(prev) < dim:
            prev.append(0.0)
    else:
        prev = [0.0] * dim

    # -- Gripper change ----------------------------------------------------
    grip_delta = 0.0
    if grip_idx is not None and grip_idx < len(act) and grip_idx < len(prev):
        grip_delta = abs(act[grip_idx] - prev[grip_idx])

    if grip_delta > grip_thresh:
        grip_val = act[grip_idx]
        if grip_val < 0.5:
            action_type = "grasp"
        else:
            action_type = "release"
        meta = {
            "gripper_delta": round(grip_delta, 4),
            "gripper_value": round(grip_val, 4),
        }
        return action_type, meta

    # -- Position change ---------------------------------------------------
    pos_delta = 0.0
    for idx in pos_idxs:
        if idx < len(act) and idx < len(prev):
            pos_delta += (act[idx] - prev[idx]) ** 2
    pos_delta = math.sqrt(pos_delta)

    pos_values = [act[i] for i in pos_idxs if i < len(act)]

    if pos_delta < move_thresh:
        return "stop", {"position_delta": round(pos_delta, 4)}

    # Determine direction
    # For ALOHA cabinet: negative x = toward robot (pull)
    if task_type in ("aloha_mobile_cabinet", "aloha_sim_transfer_cube"):
        x_idx = pos_idxs[0]
        if x_idx < len(act) and x_idx < len(prev):
            x_change = act[x_idx] - prev[x_idx]
            if x_change < -move_thresh:
                action_type = "pull"
            elif x_change > move_thresh:
                action_type = "push"
            else:
                action_type = "move_to"
        else:
            action_type = "move_to"
    else:
        action_type = "move_to"

    meta = {
        "position_delta": round(pos_delta, 4),
        "position": [round(v, 4) for v in pos_values],
    }
    return action_type, meta


# ---------------------------------------------------------------------------
# Intent -> action vector
# ---------------------------------------------------------------------------

@dataclass
class GemmaIntent:
    """A structured intent produced by Gemma 4, encoding a desired robot action."""
    tool: str = "gripper"
    target: str = ""          # object name or "position"
    params: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "target": self.target,
            "params": dict(self.params),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GemmaIntent":
        return cls(
            tool=d.get("tool", "gripper"),
            target=d.get("target", ""),
            params=d.get("params", {}),
            reasoning=d.get("reasoning", ""),
        )


def gemma_intent_to_action_vector(
    intent: GemmaIntent | dict,
    current_state: list[float],
    object_positions: dict[str, list[float]] | None = None,
    task_type: str = "aloha_mobile_cabinet",
) -> list[float]:
    """Map a GemmaIntent (or dict) to a concrete action vector.

    Args:
        intent: GemmaIntent dataclass or dict with tool/target/params
        current_state: Current robot state (used for delta-based actions)
        object_positions: Map of object name -> [x, y, z] for target resolution
        task_type: Task type key for action space metadata

    Returns:
        Action vector as list[float] matching the task action space.
    """
    if isinstance(intent, dict):
        intent = GemmaIntent.from_dict(intent)

    spec = task_specific_action_space(task_type)
    dim = spec["dim"]
    pos_idxs = spec["position_indices"]
    grip_idx = spec["gripper_idx"]

    # Start from current state's position
    action = [0.0] * dim
    for i, v in enumerate(current_state):
        if i < dim:
            action[i] = v

    tool = intent.tool.lower()

    # -- Position target ---------------------------------------------------
    if tool in ("gripper", "arm", "pusher"):
        target_pos: list[float] | None = None

        # Resolve by object name
        if intent.target and object_positions:
            obj_pos = object_positions.get(intent.target)
            if obj_pos is not None:
                target_pos = list(obj_pos)

        # Resolve from explicit params
        if target_pos is None:
            target_pos = []
            for i in pos_idxs:
                key = ["x", "y", "z"][i] if i < 3 else f"axis_{i}"
                val = intent.params.get(key)
                if val is not None:
                    target_pos.append(val)
                elif i < len(current_state):
                    target_pos.append(current_state[i])

        if target_pos:
            for i, idx in enumerate(pos_idxs):
                if i < len(target_pos) and idx < dim:
                    action[idx] = target_pos[i]

        # Orientation
        ori_idxs = spec.get("orientation_indices")
        if ori_idxs:
            for i, idx in enumerate(ori_idxs):
                key = ["roll", "pitch", "yaw"][i] if i < 3 else f"orient_{i}"
                val = intent.params.get(key)
                if val is not None and idx < dim:
                    action[idx] = val

    # -- Gripper -----------------------------------------------------------
    if grip_idx is not None and grip_idx < dim:
        grip_cmd = intent.params.get("gripper_open")
        if grip_cmd is not None:
            action[grip_idx] = 1.0 if grip_cmd else 0.0
        elif intent.tool.lower() == "grasp":
            action[grip_idx] = 0.0
        elif intent.tool.lower() == "release":
            action[grip_idx] = 1.0

    return action
