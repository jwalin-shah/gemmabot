"""Map between human-interpretable intents and robot action vectors.

Provides a dataset registry for 10+ LeRobot datasets, heuristic intent inference
from consecutive frames, and bidirectional conversion between Gemma 4 structured
intents and LeRobot action spaces.

The DATASET_REGISTRY at module level maps repo_id -> config so the replay system
works with ANY supported LeRobot dataset without hardcoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataset Registry — one entry per supported LeRobot dataset
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "lerobot/pusht": {
        "task_type": "pusht",
        "action_dim": 2,
        "gripper_idx": None,
        "position_indices": [0, 1],
        "description": "Push the T-shaped block to the green target zone",
        "tools": ["move_to", "stop"],
        "camera_keys": ["top"],
        "task_family": "pushing",
    },
    "lerobot/aloha_sim_transfer_cube_human": {
        "task_type": "aloha_transfer_cube",
        "action_dim": 14,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Pick up the cube and transfer it to the other arm",
        "tools": ["move_to", "grasp", "release", "stop"],
        "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "task_family": "bi_manipulation",
    },
    "lerobot/aloha_mobile_cabinet": {
        "task_type": "aloha_cabinet",
        "action_dim": 14,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Open the cabinet door using the handle",
        "tools": ["move_to", "grasp", "pull", "release", "stop"],
        "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "task_family": "mobile_manipulation",
    },
    "lerobot/droid": {
        "task_type": "droid",
        "action_dim": 7,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Pick and place objects on a tabletop",
        "tools": ["move_to", "grasp", "lift", "place", "stop"],
        "camera_keys": ["cam_high", "cam_wrist"],
        "task_family": "pick_place",
    },
    "lerobot/bridge_orig": {
        "task_type": "bridge",
        "action_dim": 7,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Push, slide, or pick up objects on a tabletop",
        "tools": ["move_to", "grasp", "release", "stop"],
        "camera_keys": ["cam_high", "cam_wrist"],
        "task_family": "pick_place",
    },
    "lerobot/kuka": {
        "task_type": "kuka",
        "action_dim": 6,
        "gripper_idx": None,
        "position_indices": [0, 1, 2],
        "description": "Reach toward and interact with objects on a table",
        "tools": ["move_to", "stop"],
        "camera_keys": ["cam_high"],
        "task_family": "reaching",
    },
    "lerobot/libero_object_no_noops": {
        "task_type": "libero",
        "action_dim": 7,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Pick up the object and place it at the target location",
        "tools": ["move_to", "grasp", "lift", "place", "stop"],
        "camera_keys": ["agentview", "robot0_eye_in_hand"],
        "task_family": "pick_place",
    },
    "lerobot/libero_spatial_no_noops": {
        "task_type": "libero_spatial",
        "action_dim": 7,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Pick an object from one location and place at another",
        "tools": ["move_to", "grasp", "lift", "place", "stop"],
        "camera_keys": ["agentview", "robot0_eye_in_hand"],
        "task_family": "pick_place",
    },
    "lerobot/taco_play": {
        "task_type": "taco_play",
        "action_dim": 7,
        "gripper_idx": 6,
        "position_indices": [0, 1, 2],
        "description": "Pick and place objects on a tabletop",
        "tools": ["move_to", "grasp", "release", "stop"],
        "camera_keys": ["cam_high", "cam_wrist"],
        "task_family": "pick_place",
    },
    "lerobot/nyu_door_opening": {
        "task_type": "door_opening",
        "action_dim": 8,
        "gripper_idx": 7,
        "position_indices": [0, 1, 2],
        "description": "Open a door by grasping the handle and pulling",
        "tools": ["move_to", "grasp", "pull", "release", "stop"],
        "camera_keys": ["cam_high", "cam_wrist"],
        "task_family": "mobile_manipulation",
    },
    "lerobot/berkeley_autolab_ur5": {
        "task_type": "ur5_reach",
        "action_dim": 6,
        "gripper_idx": None,
        "position_indices": [0, 1, 2],
        "description": "Reach toward objects on a tabletop using a UR5 arm",
        "tools": ["move_to", "stop"],
        "camera_keys": ["cam_high"],
        "task_family": "reaching",
    },
}

_TASK_TYPE_TO_REPO: dict[str, str] = {
    "pusht": "lerobot/pusht",
    "aloha_mobile_cabinet": "lerobot/aloha_mobile_cabinet",
    "aloha_transfer_cube": "lerobot/aloha_sim_transfer_cube_human",
    "droid": "lerobot/droid",
    "bridge": "lerobot/bridge_orig",
    "kuka": "lerobot/kuka",
    "libero": "lerobot/libero_object_no_noops",
    "libero_spatial": "lerobot/libero_spatial_no_noops",
    "taco_play": "lerobot/taco_play",
    "door_opening": "lerobot/nyu_door_opening",
    "ur5_reach": "lerobot/berkeley_autolab_ur5",
}


def get_task_config(repo_id: str) -> dict[str, Any] | None:
    """Look up a dataset config by repo_id (e.g. ``lerobot/pusht``).

    Returns the registry entry dict, or ``None`` if unknown.
    """
    return DATASET_REGISTRY.get(repo_id)


def available_datasets() -> dict[str, dict[str, Any]]:
    """Return metadata about all registered datasets.

    Returns a copy of the full DATASET_REGISTRY.
    """
    return dict(DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# Action space metadata (built from registry)
# ---------------------------------------------------------------------------

def _build_task_specs() -> dict[str, dict[str, Any]]:
    """Build the internal _TASK_SPECS dict from DATASET_REGISTRY entries.

    Each entry maps task_type -> action-space metadata including dimension,
    indices for position / orientation / gripper, and delta thresholds.
    """
    specs: dict[str, dict[str, Any]] = {}

    for repo_id, config in DATASET_REGISTRY.items():
        task_type = config["task_type"]
        dim = config["action_dim"]
        grip_idx = config["gripper_idx"]
        pos_idxs = tuple(config["position_indices"])

        # Derive orientation indices: if dim >= 6 and first three are pos,
        # assume axes 3,4,5 are orientation (roll, pitch, yaw).
        ori_idxs: tuple[int, ...] | None = None
        if dim >= 6 and list(pos_idxs[:3]) == [0, 1, 2]:
            ori_idxs = (3, 4, 5)

        # Threshold defaults
        grip_thresh: float = 0.3
        move_thresh: float = 0.02

        # Special-case overrides
        family = config.get("task_family", "")
        if family == "pushing":
            move_thresh = 5.0  # pixels for PushT
            grip_thresh = 0.5
        elif task_type == "door_opening":
            move_thresh = 0.015

        specs[task_type] = {
            "dim": dim,
            "gripper_idx": grip_idx,
            "position_indices": pos_idxs,
            "orientation_indices": ori_idxs,
            "gripper_threshold": grip_thresh,
            "movement_threshold": move_thresh,
        }

    # Also keep the original aloha_mobile_cabinet shorthand for backward compat
    specs["aloha_mobile_cabinet"] = specs.get("aloha_cabinet", {
        "dim": 7,
        "gripper_idx": 6,
        "position_indices": (0, 1, 2),
        "orientation_indices": (3, 4, 5),
        "gripper_threshold": 0.3,
        "movement_threshold": 0.02,
    })
    # Backward compat: the original aloha_sim_transfer_cube key (dim=7)
    specs["aloha_sim_transfer_cube"] = {
        "dim": 7,
        "gripper_idx": 6,
        "position_indices": (0, 1, 2),
        "orientation_indices": (3, 4, 5),
        "gripper_threshold": 0.3,
        "movement_threshold": 0.02,
    }

    return specs


_TASK_SPECS: dict[str, dict[str, Any]] = _build_task_specs()


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
    action_type: str               # grasp | release | move_to | pull | push | stop | orient | lift | place
    target_position: list[float] | None = None  # approximate 3D position
    target_object: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


def _task_family_for(task_type: str) -> str:
    """Return the task_family (pushing, pick_place, bi_manipulation, etc.) for a task_type."""
    # Try registry first
    for config in DATASET_REGISTRY.values():
        if config["task_type"] == task_type:
            return config.get("task_family", "pick_place")
    # Fallback by name
    t = task_type.lower()
    if "push" in t:
        return "pushing"
    if "cabinet" in t or "door" in t:
        return "mobile_manipulation"
    if "transfer_cube" in t:
        return "bi_manipulation"
    if "kuka" in t or "reach" in t or "ur5" in t:
        return "reaching"
    return "pick_place"


def infer_human_intent(
    action: list[float],
    prev_action: list[float] | None,
    state: list[float],
    prev_state: list[float] | None,
    task_type: str = "aloha_mobile_cabinet",
) -> tuple[str, dict]:
    """Heuristic intent inference from raw action and state deltas.

    Handles all task families from the DATASET_REGISTRY:

      - **pushing**:          XY deltas only  -> move_to / stop
      - **pick_place**:       gripper + Z changes -> grasp / release / move_to / lift / place
      - **bi_manipulation**:  dual-arm state analysis
      - **mobile_manipulation**: pull actions for door/cabinet opening
      - **reaching**:         XY movement only

    Returns
        (action_type: str, metadata: dict)

    Action types:
        "grasp"      -- gripper closing
        "release"    -- gripper opening
        "move_to"    -- significant position change
        "pull"       -- movement toward the robot base (decreasing x)
        "push"       -- movement away from base
        "lift"       -- significant positive Z change
        "place"      -- significant negative Z change with gripper open
        "stop"       -- negligible action (still)
        "orient"     -- orientation-only change
    """
    spec = task_specific_action_space(task_type)
    dim = spec["dim"]
    pos_idxs = spec["position_indices"]
    grip_idx = spec["gripper_idx"]
    grip_thresh = spec["gripper_threshold"]
    move_thresh = spec["movement_threshold"]
    family = _task_family_for(task_type)

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

    # -- Per-family logic --------------------------------------------------
    if family == "pushing":
        # Pushing: XY movement only, no gripper
        action_type = "move_to"
        meta = {
            "position_delta": round(pos_delta, 4),
            "position": [round(v, 4) for v in pos_values],
            "task_family": "pushing",
        }
        return action_type, meta

    elif family == "reaching":
        # Reaching: just XY/Z movement, no gripper interactions
        action_type = "move_to"
        meta = {
            "position_delta": round(pos_delta, 4),
            "position": [round(v, 4) for v in pos_values],
            "task_family": "reaching",
        }
        return action_type, meta

    elif family in ("pick_place", "bi_manipulation"):
        # Pick and place: look for Z changes as lift/place signals
        z_idx = pos_idxs[2] if len(pos_idxs) > 2 else None
        z_delta = 0.0
        if z_idx is not None and z_idx < len(act) and z_idx < len(prev):
            z_delta = act[z_idx] - prev[z_idx]

        z_threshold = move_thresh * 3

        if abs(z_delta) > z_threshold and z_delta > 0:
            action_type = "lift"
        elif abs(z_delta) > z_threshold and z_delta < 0:
            action_type = "place"
        else:
            if family == "bi_manipulation":
                x_idx = pos_idxs[0]
                if x_idx < len(act) and x_idx < len(prev):
                    x_change = act[x_idx] - prev[x_idx]
                    if x_change < -move_thresh * 2:
                        action_type = "pull"
                    elif x_change > move_thresh * 2:
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
            "z_delta": round(z_delta, 4),
            "task_family": family,
        }
        return action_type, meta

    elif family == "mobile_manipulation":
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

        meta = {
            "position_delta": round(pos_delta, 4),
            "position": [round(v, 4) for v in pos_values],
            "task_family": "mobile_manipulation",
        }
        return action_type, meta

    # Fallback: generic direction-based heuristics
    if task_type in ("aloha_mobile_cabinet", "aloha_sim_transfer_cube", "aloha_cabinet", "aloha_transfer_cube"):
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
    task_type: str = ""       # which dataset task type this intent targets
    confidence: float = 1.0   # how confident Gemma is in this prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "target": self.target,
            "params": dict(self.params),
            "reasoning": self.reasoning,
            "task_type": self.task_type,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GemmaIntent":
        return cls(
            tool=d.get("tool", "gripper"),
            target=d.get("target", ""),
            params=d.get("params", {}),
            reasoning=d.get("reasoning", ""),
            task_type=d.get("task_type", ""),
            confidence=float(d.get("confidence", 1.0)),
        )


def gemma_intent_to_action_vector(
    intent: GemmaIntent | dict,
    current_state: list[float],
    object_positions: dict[str, list[float]] | None = None,
    task_type: str = "aloha_mobile_cabinet",
) -> list[float]:
    """Map a GemmaIntent (or dict) to a concrete action vector.

    Args:
        intent: GemmaIntent dataclass or dict with tool/target/params.
        current_state: Current robot state (used for delta-based actions).
        object_positions: Map of object name -> [x, y, z] for target resolution.
        task_type: Task type key for action space metadata.

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
    if tool in ("gripper", "arm", "pusher", "move_to", "grasp", "lift", "place", "pull", "push"):
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
        elif intent.tool.lower() in ("grasp", "grasp_side", "lift"):
            action[grip_idx] = 0.0  # closed
        elif intent.tool.lower() in ("release", "place"):
            action[grip_idx] = 1.0  # open

    # -- Add small positional offsets for lift/place if Z in params --------
    if tool == "lift" and len(pos_idxs) >= 3:
        z_idx = pos_idxs[2]
        z_val = intent.params.get("z")
        if z_val is not None and z_idx < dim:
            action[z_idx] = z_val

    if tool == "place" and len(pos_idxs) >= 3:
        z_idx = pos_idxs[2]
        z_val = intent.params.get("z")
        if z_val is not None and z_idx < dim:
            action[z_idx] = z_val

    return action
