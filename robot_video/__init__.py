"""LeRobot dataset helpers used by the live demos and ZTP replay system."""

from robot_video.frame_loader import LeRobotFrameSource, VideoFrame
from robot_video.action_mapper import (
    GemmaIntent,
    infer_human_intent,
    gemma_intent_to_action_vector,
    task_specific_action_space,
    DATASET_REGISTRY,
    get_task_config,
    InferredIntent,
)
from robot_video.action_comparator import ComparisonScore, ActionComparator
from robot_video.replay_engine import DatasetReplayEngine, FrameComparison


def available_datasets():
    """Return combined dataset info from both the registry and the engine."""
    from robot_video.action_mapper import available_datasets as registry_datasets
    from robot_video.replay_engine import available_datasets as engine_datasets
    merged = dict(registry_datasets())
    merged.update(engine_datasets())
    return merged


__all__ = [
    "LeRobotFrameSource",
    "VideoFrame",
    "GemmaIntent",
    "InferredIntent",
    "infer_human_intent",
    "gemma_intent_to_action_vector",
    "task_specific_action_space",
    "DATASET_REGISTRY",
    "get_task_config",
    "ComparisonScore",
    "ActionComparator",
    "DatasetReplayEngine",
    "FrameComparison",
    "available_datasets",
]
