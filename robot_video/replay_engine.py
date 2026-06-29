"""Dataset Replay Engine — replay LeRobot datasets through Gemma 4 perception.

Allows stepping through real robot dataset episodes frame by frame, recording
Gemma 4 predictions for each frame, and comparing them against ground-truth
actions for offline benchmarking.

Key classes:
    DatasetReplayEngine    — loads a LeRobot dataset, yields frames, records intents
    FrameComparison        — one frame's worth of data (image, GT action, Gemma intent, score)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from robot_video.frame_loader import LeRobotFrameSource, VideoFrame
from robot_video.action_comparator import ActionComparator, ComparisonScore
from robot_video.action_mapper import GemmaIntent

# ---------------------------------------------------------------------------
# Known datasets (public LeRobot datasets that work with this system)
# ---------------------------------------------------------------------------

_KNOWN_DATASETS: dict[str, dict[str, Any]] = {
    "lerobot/aloha_mobile_cabinet": {
        "task_type": "aloha_mobile_cabinet",
        "description": "ALOHA Mobile Cabinet Opening — 7-DOF teleoperated cabinet opening",
        "camera": "cam_high",
    },
    "lerobot/aloha_sim_transfer_cube": {
        "task_type": "aloha_sim_transfer_cube",
        "description": "ALOHA Sim Transfer Cube — pick-and-place in simulation",
        "camera": "cam_high",
    },
    "lerobot/pusht": {
        "task_type": "pusht",
        "description": "PushT — 2D block pushing task",
        "camera": None,
    },
}

KNOWN_DATASET_IDS = list(_KNOWN_DATASETS.keys())


def available_datasets() -> dict[str, dict[str, Any]]:
    """Return metadata about all known datasets."""
    return dict(_KNOWN_DATASETS)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FrameComparison:
    """One frame from a dataset episode with the Gemma 4 prediction and score."""
    frame_index: int
    episode_index: int
    image_uri: str
    ground_truth_action: list[float]
    ground_truth_state: list[float]
    predicted_intent: dict[str, Any] | None  # GemmaIntent serialised
    predicted_action: list[float]
    score: ComparisonScore | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.score is not None:
            d["score"] = self.score.to_dict()
        return d


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DatasetReplayEngine:
    """Step through a LeRobot dataset frame by frame, recording Gemma intents.

    Usage::

        engine = DatasetReplayEngine()
        engine.load_dataset("lerobot/aloha_mobile_cabinet")
        engine.select_episode(0)
        frame = engine.get_frame(10)          # frame 10 of episode 0
        engine.record_gemma_intent(
            {"tool": "gripper", "target": "handle", ...},
            latency_ms=450,
        )
        history = engine.frame_log()
    """

    def __init__(self) -> None:
        self._source: LeRobotFrameSource | None = None
        self._task_type: str = "aloha_mobile_cabinet"
        self._episode_idx: int = -1
        self._episode_frames: int = 0
        self._current_frame: VideoFrame | None = None
        self._prev_action: list[float] | None = None
        self._comparator: ActionComparator | None = None
        self._log: list[FrameComparison] = []
        self._session_id: str = uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._source is not None

    @property
    def dataset_info(self) -> dict[str, Any]:
        if self._source is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "repo_id": self._source.repo_id,
            "task_type": self._task_type,
            "num_episodes": self._source.num_episodes,
            "num_frames": self._source.num_frames,
            "fps": self._source.fps,
            "cameras": self._source.camera_keys,
        }

    def load_dataset(self, repo_id: str) -> dict[str, Any]:
        """Load a LeRobot dataset by repo ID.

        Returns dataset info dict (same as dataset_info).
        """
        # Determine task type from known datasets, or infer from repo_id
        known = _KNOWN_DATASETS.get(repo_id)
        if known is not None:
            self._task_type = known["task_type"]
        elif "pusht" in repo_id.lower():
            self._task_type = "pusht"
        else:
            self._task_type = "aloha_mobile_cabinet"

        camera_key = known["camera"] if known else None
        self._source = LeRobotFrameSource(repo_id, camera_key=camera_key)
        self._comparator = ActionComparator(task_type=self._task_type)
        self._log.clear()
        self._episode_idx = -1
        self._episode_frames = 0
        return self.dataset_info

    def select_episode(self, episode_idx: int) -> dict[str, Any]:
        """Select an episode for replay. Returns episode metadata.

        Resets the frame log and comparator for this episode.
        """
        if self._source is None:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")
        if self._source.num_episodes == 0:
            raise RuntimeError("Dataset has no episodes.")

        if episode_idx < 0 or episode_idx >= self._source.num_episodes:
            raise IndexError(
                f"Episode {episode_idx} out of range "
                f"[0, {self._source.num_episodes})"
            )

        self._episode_idx = episode_idx
        self._episode_frames = self._source.episode_frames(episode_idx)
        self._prev_action = None
        self._log.clear()
        if self._comparator is not None:
            self._comparator.clear()

        return {
            "episode_index": episode_idx,
            "num_frames": self._episode_frames,
            "task_type": self._task_type,
        }

    @property
    def current_episode(self) -> dict[str, Any]:
        if self._episode_idx < 0:
            return {"selected": False}
        return {
            "selected": True,
            "episode_index": self._episode_idx,
            "num_frames": self._episode_frames,
        }

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, frame_idx: int) -> VideoFrame:
        """Get a VideoFrame at the given index within the selected episode.

        This updates _current_frame and _prev_action for state tracking.
        Does NOT record the frame in the log — use record_gemma_intent() for that.
        """
        if self._source is None:
            raise RuntimeError("No dataset loaded.")
        if self._episode_idx < 0:
            raise RuntimeError("No episode selected.")

        max_idx = self._episode_frames - 1
        if frame_idx < 0 or frame_idx > max_idx:
            raise IndexError(f"Frame {frame_idx} out of range [0, {max_idx}]")

        frame = self._source.get_frame(episode=self._episode_idx, frame_idx=frame_idx)
        self._current_frame = frame
        return frame

    # ------------------------------------------------------------------
    # Intent recording
    # ------------------------------------------------------------------

    def record_gemma_intent(
        self,
        intent: dict[str, Any] | None,
        latency_ms: float = 0.0,
        object_positions: dict[str, list[float]] | None = None,
    ) -> FrameComparison:
        """Record a Gemma 4 intent prediction for the current frame.

        Args:
            intent: Gemma structured intent dict, or None (missed prediction).
            latency_ms: Inference latency in milliseconds.
            object_positions: Object name -> [x, y, z] for target resolution.

        Returns:
            A FrameComparison containing the prediction, ground truth, and score.
        """
        if self._current_frame is None:
            raise RuntimeError("No current frame. Call get_frame() first.")
        if self._comparator is None:
            raise RuntimeError("No comparator. Dataset must be loaded.")

        frame = self._current_frame
        gt_action = frame.action
        state = frame.state

        # Convert intent to GemmaIntent if needed
        gemma_intent = None
        pred_action: list[float] = []
        if intent is not None:
            gemma_intent = GemmaIntent.from_dict(intent)
            from robot_video.action_mapper import gemma_intent_to_action_vector
            pred_action = gemma_intent_to_action_vector(
                gemma_intent, state, object_positions, self._task_type,
            )

        # Compare
        score = self._comparator.compare(
            ground_truth_action=gt_action,
            predicted_intent=gemma_intent,
            current_state=state,
            prev_action=self._prev_action,
            object_positions=object_positions,
            frame_index=int(frame.frame_index),
            episode_index=int(frame.episode_index),
            latency_ms=latency_ms,
        )

        # Update prev_action for next frame
        self._prev_action = list(gt_action)

        fc = FrameComparison(
            frame_index=int(frame.frame_index),
            episode_index=int(frame.episode_index),
            image_uri=frame.image_uri,
            ground_truth_action=gt_action,
            ground_truth_state=state,
            predicted_intent=intent,
            predicted_action=pred_action,
            score=score,
            timestamp=time.time(),
        )
        self._log.append(fc)
        return fc

    def frame_log(self) -> list[dict]:
        """Return the full log of recorded FrameComparisons as dicts."""
        return [fc.to_dict() for fc in self._log]

    def last_comparison(self) -> FrameComparison | None:
        if not self._log:
            return None
        return self._log[-1]

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        provider: Callable[[str, list[float], list[float]], dict[str, Any] | None],
        episode_idx: int = 0,
        max_frames: int = -1,
        object_positions: dict[str, list[float]] | None = None,
        on_frame: Callable[[int, FrameComparison], None] | None = None,
        sleep_between_frames: float = 0.0,
    ) -> dict[str, Any]:
        """Run a full benchmark over an episode.

        Args:
            provider: A callable that takes (image_b64, state, action) and returns
                      a Gemma intent dict (or None on failure).
            episode_idx: Which episode to benchmark.
            max_frames: Maximum frames to process (-1 = all).
            object_positions: Object positions for intent resolution.
            on_frame: Optional callback after each frame is processed.
            sleep_between_frames: Seconds to sleep between frames (for rate limiting).

        Returns:
            Benchmark report dict from ActionComparator.compute_benchmark().
        """
        if self._source is None:
            raise RuntimeError("No dataset loaded.")

        self.select_episode(episode_idx)
        total = self._episode_frames if max_frames < 0 else min(max_frames, self._episode_frames)

        for i in range(total):
            frame = self.get_frame(i)
            image_uri = frame.image_uri
            state = frame.state
            action = frame.action

            # Call the provider to get a Gemma intent
            import time as _time
            t0 = _time.time()
            intent = provider(image_uri, state, action)
            elapsed_ms = (_time.time() - t0) * 1000.0

            fc = self.record_gemma_intent(intent, latency_ms=elapsed_ms, object_positions=object_positions)

            if on_frame is not None:
                on_frame(i, fc)

            if sleep_between_frames > 0:
                import time as _time
                _time.sleep(sleep_between_frames)

        if self._comparator is None:
            return {"task_type": self._task_type, "num_frames": 0, "verdict": "ERROR", "aggregate": {}}
        return self._comparator.compute_benchmark(
            self._comparator.results, task_type=self._task_type,
        )

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialise the current engine state (dataset, episode, log)."""
        return {
            "session_id": self._session_id,
            "dataset": self.dataset_info,
            "episode": self.current_episode,
            "log_length": len(self._log),
        }
