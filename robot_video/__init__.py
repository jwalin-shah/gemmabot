"""LeRobot video integration for the Cerebras × Gemma 4 robotics hackathon.

Components:
    frame_loader  — extract video frames from LeRobot datasets as data URIs
    run_pipeline  — feed frames through the Vision → Action → Safety pipeline
    compare_speed — Cerebras vs GPU speed race on LeRobot video data
"""

from robot_video.frame_loader import LeRobotFrameSource, VideoFrame

__all__ = [
    "LeRobotFrameSource",
    "VideoFrame",
]
