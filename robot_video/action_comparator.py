"""Compare ground-truth dataset actions with Gemma 4 predicted intents.

Provides scoring primitives (ComparisonScore, ActionComparator) that quantify
how well a model's predicted action matches the dataset's human-demonstrated
action at the level of tool use, position, and gripper state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from robot_video.action_mapper import (
    GemmaIntent,
    gemma_intent_to_action_vector,
    infer_human_intent,
    task_specific_action_space,
)


# ---------------------------------------------------------------------------
# Comparison data model
# ---------------------------------------------------------------------------

@dataclass
class ComparisonScore:
    """Per-frame comparison between dataset ground truth and Gemma prediction.

    All distance/accuracy fields are normalised so that 1.0 = perfect match
    and 0.0 = maximum plausible error.
    """
    # -- Tool usage (grasp/release vs ground-truth gripper change) --------
    tool_match: bool = False            # did the predicted tool match the ground-truth action type?
    tool_accuracy: float = 0.0          # 0.0-1.0 (1.0 = same action type)
    tool_predicted: str = ""            # e.g. "grasp" / "release" / "move_to"
    tool_ground_truth: str = ""         # from heuristic on ground-truth action

    # -- Position error ---------------------------------------------------
    action_distance: float = 0.0        # Euclidean distance in action space (raw units)
    position_distance: float = 0.0      # Euclidean distance in position subspace (raw units)
    position_error_normalised: float = 0.0  # 0.0-1.0 (1.0 = perfect)

    # -- Gripper state ----------------------------------------------------
    gripper_match: bool = False         # open-vs-closed matches ground truth
    gripper_accuracy: float = 0.0       # 0.0 or 1.0 (binary for open/close)
    gripper_predicted: float = 0.0      # predicted gripper value
    gripper_ground_truth: float = 0.0   # ground truth gripper value

    # -- Per-axis breakdown -----------------------------------------------
    per_axis_errors: list[float] = field(default_factory=list)

    # -- Latency -----------------------------------------------------------
    latency_ms: float = 0.0             # how long the inference took (ms)

    # -- Metadata ---------------------------------------------------------
    frame_index: int = -1
    episode_index: int = -1
    ground_truth_action: list[float] = field(default_factory=list)
    predicted_action: list[float] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def success(self) -> bool:
        """Heuristic pass/fail: tool matches AND position error < 20% of workspace."""
        return self.tool_match and self.tool_accuracy >= 0.5 and self.position_error_normalised >= 0.5


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

class ActionComparator:
    """Compare ground-truth dataset actions with Gemma-predicted intents.

    Example::

        comp = ActionComparator(task_type="aloha_mobile_cabinet")
        score = comp.compare(
            ground_truth_action=[0.1, -0.2, 0.3, ...],
            predicted_intent={"tool": "gripper", "target": "handle", ...},
            current_state=[0.0, 0.0, 0.0, ...],
            prev_action=[0.0, 0.0, 0.0, ...],
        )
        print(score.tool_match, score.position_distance)

    For batch results use ``aggregate(scores)`` and ``compute_benchmark(scores)``.
    """

    # Maximum plausible position distance for normalisation (meters / pixels)
    _MAX_POS_DISTANCE: float = 0.5       # 50 cm for ALOHA; overridden for PushT

    def __init__(self, task_type: str = "aloha_mobile_cabinet") -> None:
        self.task_type = task_type
        self.spec = task_specific_action_space(task_type)
        # PushT operates in pixel space (512x512), so normalise differently
        if task_type == "pusht":
            self._MAX_POS_DISTANCE = 512.0 * math.sqrt(2)  # ~724 px diagonal
        self._results: list[ComparisonScore] = []

    # ------------------------------------------------------------------
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
        """Compare one frame's ground truth action against the Gemma prediction.

        Args:
            ground_truth_action: The action from the dataset (list of floats).
            predicted_intent: GemmaIntent dataclass, dict, or None (missed prediction).
            current_state: Robot state at this frame.
            prev_action: Previous frame's action (for heuristic type inference).
            object_positions: Object name -> [x, y, z] for target resolution.
            frame_index: Dataset frame index (for provenance).
            episode_index: Dataset episode index.
            latency_ms: Inference latency in milliseconds.

        Returns:
            ComparisonScore with all metrics populated.
        """
        gt = list(ground_truth_action)
        spec = self.spec
        grip_idx = spec["gripper_idx"]
        pos_idxs = spec["position_indices"]

        # -- Ground-truth action type (heuristic) ---------------------------
        gt_type, _ = infer_human_intent(gt, prev_action, current_state, None, self.task_type)

        # -- Predicted action (convert intent -> action vector) -------------
        if predicted_intent is None:
            # Null prediction — score as a complete miss
            pred = [0.0] * spec["dim"]
            tool_match = False
            tool_acc = 0.0
            tool_pred_type = "none"
            pos_dist = self._MAX_POS_DISTANCE * 1.5
            pos_err_norm = 0.0
            grip_match = False
            grip_acc = 0.0
            grip_pred = 0.0
        else:
            pred = gemma_intent_to_action_vector(
                predicted_intent, current_state, object_positions, self.task_type,
            )
            pred_type, _ = infer_human_intent(pred, prev_action, current_state, None, self.task_type)

            # Tool accuracy
            tool_match = pred_type == gt_type
            tool_acc = 1.0 if tool_match else 0.0
            tool_pred_type = pred_type

            # Position distance (position subspace only)
            pos_dist = 0.0
            gt_pos = [gt[i] for i in pos_idxs if i < len(gt)]
            pred_pos = [pred[i] for i in pos_idxs if i < len(pred)]
            n_pos = min(len(gt_pos), len(pred_pos))
            for i in range(n_pos):
                pos_dist += (gt_pos[i] - pred_pos[i]) ** 2
            pos_dist = math.sqrt(pos_dist)

            # Gripper match
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

        # -- Action distance (full action space) ----------------------------
        action_dist = 0.0
        per_axis = []
        n = min(len(gt), len(pred))
        for i in range(n):
            err = gt[i] - pred[i]
            action_dist += err ** 2
            per_axis.append(round(abs(err), 6))
        action_dist = math.sqrt(action_dist)

        # Normalised position error
        max_dist = self._MAX_POS_DISTANCE
        pos_err_norm = max(0.0, 1.0 - (pos_dist / max_dist)) if max_dist > 0 else 0.0

        score = ComparisonScore(
            tool_match=tool_match,
            tool_accuracy=tool_acc,
            tool_predicted=tool_pred_type if predicted_intent is not None else "none",
            tool_ground_truth=gt_type,
            action_distance=round(action_dist, 6),
            position_distance=round(pos_dist, 6),
            position_error_normalised=round(pos_err_norm, 4),
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

    # ------------------------------------------------------------------
    @property
    def results(self) -> list[ComparisonScore]:
        return list(self._results)

    def clear(self) -> None:
        self._results.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def aggregate(scores: list[ComparisonScore]) -> dict[str, Any]:
        """Compute aggregate statistics over a list of ComparisonScore objects.

        Returns dict with:
            num_frames, tool_accuracy, avg_position_distance,
            avg_action_distance, avg_gripper_accuracy, avg_latency_ms,
            success_rate, axis_errors (per-axis mean absolute error).
        """
        n = len(scores)
        if n == 0:
            return {
                "num_frames": 0, "tool_accuracy": 0.0,
                "avg_position_distance": 0.0, "avg_action_distance": 0.0,
                "avg_gripper_accuracy": 0.0, "avg_latency_ms": 0.0,
                "success_rate": 0.0, "axis_errors": [],
            }

        tool_acc = sum(s.tool_accuracy for s in scores) / n
        pos_dist = sum(s.position_distance for s in scores) / n
        act_dist = sum(s.action_distance for s in scores) / n
        grip_acc = sum(s.gripper_accuracy for s in scores) / n
        latency = sum(s.latency_ms for s in scores) / n
        success_rate = sum(1 for s in scores if s.success) / n

        # Per-axis aggregate
        max_axes = max((len(s.per_axis_errors) for s in scores), default=0)
        axis_errors = [0.0] * max_axes
        for s in scores:
            for i, err in enumerate(s.per_axis_errors):
                if i < max_axes:
                    axis_errors[i] += err
        axis_errors = [round(e / n, 6) for e in axis_errors]

        return {
            "num_frames": n,
            "tool_accuracy": round(tool_acc, 4),
            "avg_position_distance": round(pos_dist, 6),
            "avg_action_distance": round(act_dist, 6),
            "avg_gripper_accuracy": round(grip_acc, 4),
            "avg_latency_ms": round(latency, 2),
            "success_rate": round(success_rate, 4),
            "axis_errors": axis_errors,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def compute_benchmark(scores: list[ComparisonScore], task_type: str = "aloha_mobile_cabinet") -> dict[str, Any]:
        """Full benchmark report from a batch of comparisons.

        Includes aggregate stats plus task-specific conclusions.
        """
        if not scores:
            return {
                "task_type": task_type,
                "num_frames": 0,
                "verdict": "NO_DATA",
                "aggregate": ActionComparator.aggregate(scores),
            }

        agg = ActionComparator.aggregate(scores)

        # Determine verdict
        if agg["tool_accuracy"] >= 0.8 and agg["avg_position_distance"] < 0.05:
            verdict = "PASS"
        elif agg["tool_accuracy"] >= 0.5 or agg["avg_position_distance"] < 0.15:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        report = {
            "task_type": task_type,
            "num_frames": agg["num_frames"],
            "verdict": verdict,
            "aggregate": agg,
            "conclusion": _verdict_message(verdict, agg, task_type),
        }
        return report


def _verdict_message(verdict: str, agg: dict, task_type: str) -> str:
    if verdict == "PASS":
        return (
            f"Gemma 4 matches dataset actions well (tool accuracy {agg['tool_accuracy']:.0%}, "
            f"position error {agg['avg_position_distance']:.4f}). Suitable for "
            f"closed-loop control on {task_type}."
        )
    elif verdict == "PARTIAL":
        return (
            f"Partial match on {task_type}: tool accuracy {agg['tool_accuracy']:.0%}, "
            f"position error {agg['avg_position_distance']:.4f}. "
            f"Check gripper logic and position scaling."
        )
    else:
        return (
            f"Poor alignment (tool accuracy {agg['tool_accuracy']:.0%}, "
            f"position error {agg['avg_position_distance']:.4f}). "
            f"The intent-to-action mapping needs retuning for {task_type}."
        )
