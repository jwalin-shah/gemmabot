"""Vision Agent — analyzes robot camera images using Gemma 4 multimodal."""

from __future__ import annotations

from src.client import CerebrasClient, InferenceResult

SYSTEM_PROMPT = """You are a robot vision analyst. You inspect camera images from a robots
environment and produce a structured description of what you see.

Return your analysis as a JSON object with these fields:
- "objects": list of visible objects / obstacles with approximate positions
- "layout": brief description of the spatial layout
- "hazards": any visible hazards (clutter, edges, people, pets, etc.)
- "grasp_targets": items the robot could potentially pick up or interact with
- "confidence": 0-1 score for how confident you are in the analysis

Be concise. Use cardinal directions (left, right, center, top, bottom) for positions."""


class VisionAgent:
    """Analyzes a camera image from a robot's environment."""

    def __init__(self, client: CerebrasClient) -> None:
        self._client = client

    def analyze(self, image_b64: str, prompt: str | None = None) -> InferenceResult:
        """Analyze a camera image."""
        user_prompt = prompt or "Analyze this scene for a robot. What do you see?"
        return self._client.image_chat(
            prompt=user_prompt,
            image_b64=image_b64,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )