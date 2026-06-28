"""Action Agent — translates scene (or image) into robot action commands.

Can operate in two modes:
1. Text mode (original): receives a text scene description from the VisionAgent
2. Multimodal mode (new): sees the camera image directly, like VisionAgent

In multimodal mode, Vision and Action run in parallel — both look at the raw image
simultaneously, no information lost in translation.
"""

from __future__ import annotations

from src.client import CerebrasClient, InferenceResult

SYSTEM_PROMPT = """You are a robot action planner. Given the scene and a task, you output
a sequence of robot commands to accomplish the task.

Think step by step, then return a JSON array of actions. Each action has:
- "action": one of "move_to", "pick_up", "place", "push", "pause", "navigate_home"
- "target": the object or location to interact with
- "parameters": dict of extra params (e.g. {"distance_m": 0.5, "direction": "left"})
- "reason": one-line explanation

Assume the robot has a robotic arm with gripper, wheeled base, and basic sensors.
Output ONLY valid JSON."""

# Shorter system prompt for the multimodal path — the model has the image,
# so it doesn't need a pre-digested scene description.
MULTIMODAL_SYSTEM_PROMPT = """You are a robot action planner. Look at the camera image and plan
a sequence of robot commands to accomplish the given task.

Return a JSON array of actions. Each action has:
- "action": one of "move_to", "pick_up", "place", "push", "pause", "navigate_home"
- "target": the object or location to interact with
- "parameters": dict of extra params (e.g. {"distance_m": 0.5})
- "reason": one-line explanation

Assume a robotic arm with gripper, wheeled base, and basic sensors.
Output ONLY valid JSON — no markdown, no explanation."""


class ActionAgent:
    """Plans robot actions from a scene description OR directly from an image."""

    def __init__(self, client: CerebrasClient) -> None:
        self._client = client

    def plan(
        self,
        scene_analysis: str,
        task: str = "Explore the environment and report findings.",
        image_b64: str | None = None,
    ) -> InferenceResult:
        """Generate action plan.

        If *image_b64* is provided the agent looks at the raw camera image
        directly (multimodal mode). Otherwise it plans from the text description.
        """
        if image_b64:
            return self._plan_from_image(image_b64, task)
        return self._plan_from_text(scene_analysis, task)

    # ------------------------------------------------------------------
    # Text mode — original, works from VisionAgent's description
    # ------------------------------------------------------------------
    def _plan_from_text(self, scene_analysis: str, task: str) -> InferenceResult:
        prompt = f"Task: {task}\n\nScene analysis:\n{scene_analysis}"
        return self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
        )

    # ------------------------------------------------------------------
    # Multimodal mode — sees the camera image directly
    # ------------------------------------------------------------------
    def _plan_from_image(self, image_b64: str, task: str) -> InferenceResult:
        user_content = (
            f"Task: {task}\n\n"
            "Look at the camera image and plan the actions this robot should take. "
            "Return a JSON array of actions with the schema you were given."
        )
        return self._client.image_chat(
            prompt=user_content,
            image_b64=image_b64,
            system_prompt=MULTIMODAL_SYSTEM_PROMPT,
            temperature=0.2,
        )
