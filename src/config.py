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

DEFAULT_IMAGE: str = os.environ.get(
    "DEMO_IMAGE_PATH",
    str(PROJECT_ROOT / "examples" / "images" / "workspace.jpg"),
)
"""Default demo image path."""

# --- Optional comparison provider ---
COMPARISON_API_KEY: str = os.environ.get("COMPARISON_API_KEY", "")
COMPARISON_BASE_URL: str = os.environ.get("COMPARISON_BASE_URL", "")
COMPARISON_MODEL: str = os.environ.get("COMPARISON_MODEL", "")


__all__: list[str] = []