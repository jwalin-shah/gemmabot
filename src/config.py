"""Project configuration — load from .env or environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CEREBRAS_API_KEY: str = os.environ.get("CEREBRAS_API_KEY", "")
"""Cerebras Inference API key."""

CEREBRAS_BASE_URL: str = "https://api.cerebras.ai"
"""Cerebras API base URL."""

GEMMA_MODEL: str = "gemma-4-31b"
"""Gemma 4 31B model ID on Cerebras."""

REASONING_EFFORT: str | None = os.environ.get("REASONING_EFFORT") or None
"""Reasoning level: None (off), ""low"", ""medium"", ""high""."""

MAX_TOKENS: int = 1024
"""Max completion tokens per agent call."""

TEMPERATURE: float = 0.2
"""Sampling temperature — low for deterministic agent outputs."""

# --- Provider configuration ---
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "nvidia_nim")
"""Default LLM provider name: cerebras | openrouter | (custom)."""

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
"""OpenRouter API key."""

OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
"""OpenRouter API base URL."""

OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b")
"""Model ID on OpenRouter (e.g. google/gemma-4-31b)."""

DEFAULT_IMAGE: str = os.environ.get(
    "DEMO_IMAGE_PATH",
    str(PROJECT_ROOT / "examples" / "images" / "workspace.jpg"),
)
"""Default demo image path."""

# --- Optional comparison provider ---
COMPARISON_API_KEY: str = os.environ.get("COMPARISON_API_KEY", "")
COMPARISON_BASE_URL: str = os.environ.get("COMPARISON_BASE_URL", "")
COMPARISON_MODEL: str = os.environ.get("COMPARISON_MODEL", "")

# --- Motion tuning ---
MOTION_MAX_STEPS: int = int(os.environ.get("MOTION_MAX_STEPS", "25"))
"""Max motion-planning iterations per tool call."""

MOTION_REACH_TOLERANCE: float = float(os.environ.get("MOTION_REACH_TOLERANCE", "0.010"))
"""Euclidean distance (m) below which we consider a position reached."""

MOTION_ACTION_GAIN: float = float(os.environ.get("MOTION_ACTION_GAIN", "1.0"))
"""Gain applied to the action delta each step."""

MOTION_FRAME_EVERY: int = int(os.environ.get("MOTION_FRAME_EVERY", "5"))
"""Record a video frame every N steps during motion."""

MOTION_GRIPPER_CONFIRM_STEPS: int = int(os.environ.get("MOTION_GRIPPER_CONFIRM_STEPS", "5"))
"""Extra steps after a gripper action to confirm the fingers have settled."""

GRIPPER_CONFIRM_TOL: float = float(os.environ.get("GRIPPER_CONFIRM_TOL", "0.002"))
"""Gripper qpos change tolerance (m) below which we consider the grip settled."""

GRIPPER_CLOSE_CMD: float = float(os.environ.get("GRIPPER_CLOSE_CMD", "1.0"))
"""Action value sent to close the gripper."""

GRIPPER_OPEN_CMD: float = float(os.environ.get("GRIPPER_OPEN_CMD", "-1.0"))
"""Action value sent to open the gripper."""

# --- Camera ---
CAMERA_HEIGHT: int = int(os.environ.get("CAMERA_HEIGHT", "384"))
"""Offscreen render height in pixels."""

CAMERA_WIDTH: int = int(os.environ.get("CAMERA_WIDTH", "384"))
"""Offscreen render width in pixels."""

# --- Server ---
SERVER_PORT: int = int(os.environ.get("SERVER_PORT", "8002"))
"""HTTP server listen port."""

# --- Vision mode ---
ENABLE_VISION_MODE: bool = os.environ.get("ENABLE_VISION_MODE", "False").lower() in ("1", "true", "yes")
"""When True, include images in LLM calls (slower but enables vision)."""


__all__: list[str] = []