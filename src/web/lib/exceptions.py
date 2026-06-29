"""Application error hierarchy.

Base exception ``DemoError`` with typed subclasses so callers can catch
specific failure domains without inspecting string messages.
"""

from __future__ import annotations

from typing import Any


class DemoError(Exception):
    """Base exception for all demo-related errors."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        self.message = message
        self.detail = detail or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} | {self.detail}"
        return self.message


class BrainError(DemoError):
    """Error from the LLM brain (prompting, parsing, schema mismatch)."""


class ExecutorError(DemoError):
    """Error from the motion executor (tools, kinematics, simulation step)."""


class SimError(DemoError):
    """Error from the simulation environment (robosuite, MuJoCo)."""


class VerificationError(DemoError):
    """Error from the task verifier (success/failure judgment)."""


class PerceptionError(DemoError):
    """Error from vision/perception (image encoding, object detection)."""


class ProviderError(DemoError):
    """Error from the LLM provider (API key, network, rate limit)."""
