"""Compare Gemma 4 predicted actions against ground-truth dataset actions.

No heuristics, no "inferred intent" guessing. Direct comparison:
  - Position: L2 distance between Gemma's target and human's actual position
  - Gripper: direct open/closed comparison
  - Tool: derived from the action vector itself (gripper changed? position moved?)

The ground truth IS the dataset action vector. Gemma's prediction IS her
structured output. We compare them directly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from robot_video.action_mapper import (
    GemmaIntent,
    gemma_intent_to_action_vector,
    task_specific_action_space,
)


@dataclass
class ComparisonScore:
    """Per-frame comparison -- no inferred intents, just direct measurement."""

    tool_match: bool = False
    tool_accuracy: float = 0.0
    tool_predicted: str = ""
    tool_ground_truth: str = ""

    position_distance: float = 0.0
    position_error_normalised: float = 0.0
    action_distance: float = 0.0

    gripper_match: bool = False
    gripper_accuracy: float = 0.0
    gripper_predicted: float = 0.0
    gripper_ground_truth: float = 0.0

    per_axis_errors: list[float] = field(default_factory=list)
    latency_ms: float = 0.0

    frame_index: int = -1
    episode_index: int = -1
    ground_truth_action: list[float] = field(default_factory=list)
    predicted_action: list[float] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def success(self) -> bool:
        return self.position_error_normalised >= 0.9 and self.gripper_match


def _derive_action_type(
    action: list[float],
    prev_action: list[float] | None,
    spec: dict,
) -> str:
    """Derive action type from raw action vector -- purely from the numbers."""
    dim = spec["dim"]
    grip_idx = spec["gripper_idx"]
    pos_idxs = spec["position_indices"]
    move_thresh = spec["movement_threshold"]

    act = list(action)
    while len(act) < dim:
        act.append(0.0)

    if prev_action is not None:
        prev = list(prev_action)
        while len(prev) < dim:
            prev.append(0.0)
    else:
        prev = [0.0] * dim

    if grip_idx is not None and grip_idx < len(act) and grip_idx < len(prev):
        grip_delta = abs(act[grip_idx] - prev[grip_idx])
        grip_thresh = spec.get("gripper_threshold", 0.3)
        if grip_delta > grip_thresh:
            return "grasp" if act[grip_idx] < 0.5 else "release"

    pos_delta = 0.0
    for idx in pos_idxs:
        if idx < len(act) and idx < len(prev):
            pos_delta += (act[idx] - prev[idx]) ** 2
    pos_delta = math.sqrt(pos_delta)

    return "move_to" if pos_delta > move_thresh else "stop"


class ActionComparator:
    """Compare dataset GT actions against Gemma predictions. Pure measurement."""

    _MAX_POS_DISTANCE: float = 0.5

    def __init__(self, task_type: str = "aloha_mobile_cabinet") -> None:
        self.task_type = task_type
        self.spec = task_specific_action_space(task_type)
        if task_type == "pusht":
            self._MAX_POS_DISTANCE = 512.0 * math.sqrt(2)
        self._results: list[ComparisonScore] = []

    def compare(
        self,
        ground_truth_action: list[float],
        predicted_intent: GemmaIntent | dict | None,
        current_state: list[float],
        prev_action: list[float] | None = None,
        object_positions: dict[str, list[float]] | None = None,
        *,
        frame_index: int = -1,
        episode_index: int = -1,
        latency_ms: float = 0.0,
    ) -> ComparisonScore:
        gt = list(ground_truth_action)
        spec = self.spec
        grip_idx = spec["gripper_idx"]
        pos_idxs = spec["position_indices"]

        gt_type = _derive_action_type(gt, prev_action, spec)

        if predicted_intent is None:
            pred = [0.0] * spec["dim"]
            tool_match = False
            tool_acc = 0.0
            pred_type = "none"
        else:
            pred = gemma_intent_to_action_vector(
                predicted_intent, current_state, object_positions, self.task_type,
            )
            pred_type = _derive_action_type(pred, prev_action, spec)

            if isinstance(predicted_intent, dict):
                stated_tool = predicted_intent.get("tool", "")
            else:
                stated_tool = predicted_intent.tool
            stated_tool = stated_tool.lower()

            tool_to_type = {
                "move_to": "move_to", "grasp": "grasp", "grasp_side": "grasp",
                "lift": "move_to", "place": "move_to", "pull": "move_to",
                "push": "move_to", "release": "release", "stop": "stop", "done": "stop",
            }
            tool_match = tool_to_type.get(stated_tool, "move_to") == gt_type
            tool_acc = 1.0 if tool_match else 0.0

        pos_dist = 0.0
        gt_pos = [gt[i] for i in pos_idxs if i < len(gt)]
        pred_pos = [pred[i] for i in pos_idxs if i < len(pred)]
        for i in range(min(len(gt_pos), len(pred_pos))):
            pos_dist += (gt_pos[i] - pred_pos[i]) ** 2
        pos_dist = math.sqrt(pos_dist)

        action_dist = 0.0
        per_axis = []
        n = min(len(gt), len(pred))
        for i in range(n):
            err = gt[i] - pred[i]
            action_dist += err ** 2
            per_axis.append(round(abs(err), 6))
        action_dist = math.sqrt(action_dist)

        grip_pred = 0.0
        grip_gt = 0.0
        if grip_idx is not None and grip_idx < len(pred) and grip_idx < len(gt):
            grip_pred = float(pred[grip_idx])
            grip_gt = float(gt[grip_idx])
            grip_match = (grip_pred >= 0.5) == (grip_gt >= 0.5)
            grip_acc = 1.0 if grip_match else 0.0
        else:
            grip_match = True
            grip_acc = 1.0

        max_dist = self._MAX_POS_DISTANCE
        pos_err_norm = max(0.0, 1.0 - (pos_dist / max_dist)) if max_dist > 0 else 0.0

        score = ComparisonScore(
            tool_match=tool_match,
            tool_accuracy=tool_acc,
            tool_predicted=pred_type if predicted_intent is not None else "none",
            tool_ground_truth=gt_type,
            position_distance=round(pos_dist, 6),
            position_error_normalised=round(pos_err_norm, 4),
            action_distance=round(action_dist, 6),
            gripper_match=grip_match,
            gripper_accuracy=grip_acc,
            gripper_predicted=grip_pred,
            gripper_ground_truth=grip_gt if grip_idx is not None and grip_idx < len(gt) else 0.0,
            per_axis_errors=per_axis,
            latency_ms=latency_ms,
            frame_index=frame_index,
            episode_index=episode_index,
            ground_truth_action=gt,
            predicted_action=pred if predicted_intent is not None else [],
            timestamp=time.time(),
        )
        self._results.append(score)
        return score

    @property
    def results(self) -> list[ComparisonScore]:
        return list(self._results)

    def clear(self) -> None:
        self._results.clear()

    @staticmethod
    def aggregate(scores: list[ComparisonScore]) -> dict[str, Any]:
        n = len(scores)
        if n == 0:
            return {"num_frames": 0, "tool_accuracy": 0.0,
                    "avg_position_distance": 0.0, "avg_action_distance": 0.0,
                    "avg_gripper_accuracy": 0.0, "avg_latency_ms": 0.0,
                    "success_rate": 0.0, "position_error_p50": 0.0, "position_error_p95": 0.0}
        tool_acc = sum(s.tool_accuracy for s in scores) / n
        pos_dist = sum(s.position_distance for s in scores) / n
        act_dist = sum(s.action_distance for s in scores) / n
        grip_acc = sum(s.gripper_accuracy for s in scores) / n
        latency = sum(s.latency_ms for s in scores) / n
        success_rate = sum(1 for s in scores if s.success) / n
        pos_dists = sorted(s.position_distance for s in scores)
        p50 = pos_dists[len(pos_dists) // 2] if pos_dists else 0.0
        p95 = pos_dists[min(int(len(pos_dists) * 0.95), len(pos_dists) - 1)] if pos_dists else 0.0
        return {"num_frames": n, "tool_accuracy": round(tool_acc, 4),
                "avg_position_distance": round(pos_dist, 6),
                "avg_action_distance": round(act_dist, 6),
                "avg_gripper_accuracy": round(grip_acc, 4),
                "avg_latency_ms": round(latency, 2), "success_rate": round(success_rate, 4),
                "position_error_p50": round(p50, 6), "position_error_p95": round(p95, 6)}

    @staticmethod
    def compute_benchmark(scores: list[ComparisonScore], task_type: str = "") -> dict[str, Any]:
        if not scores:
            return {"task_type": task_type, "num_frames": 0, "verdict": "NO_DATA",
                    "aggregate": ActionComparator.aggregate(scores)}
        agg = ActionComparator.aggregate(scores)
        ta, p95 = agg["tool_accuracy"], agg["position_error_p95"]
        if ta >= 0.8 and p95 < 0.05:
            verdict = "PASS"
        elif ta >= 0.5 or p95 < 0.15:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"
        return {"task_type": task_type, "num_frames": agg["num_frames"],
                "verdict": verdict, "aggregate": agg}
