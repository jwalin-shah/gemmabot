"""Model router for smart model selection across task types.

Provides a SmartRouter class that picks the right model (vision_reasoning,
json_structuring, object_labeling, text_reasoning, fast_text) for the job,
with automatic fallback on failure.

Each task type has a primary and fallback model. The router generates
dataset-specific system prompts for each task_family in the DATASET_REGISTRY.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Model Tier Definitions
# ---------------------------------------------------------------------------

MODEL_TIERS: dict[str, dict[str, str]] = {
    "vision_reasoning": {
        "primary": "google/gemma-4-31b-it:free",
        "fallback": "google/gemma-4-26b-a4b-it:free",
        "paid": "google/gemma-4-31b-it",
        "modality": "vision",
    },
    "json_structuring": {
        "primary": "qwen/qwen3-coder:free",
        "fallback": "meta-llama/llama-3.3-70b-instruct:free",
        "modality": "text",
    },
    "object_labeling": {
        "primary": "nvidia/nemotron-nano-12b-v2-vl:free",
        "fallback": "google/gemma-4-31b-it:free",
        "modality": "vision",
    },
    "text_reasoning": {
        "primary": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback": "qwen/qwen3-coder:free",
        "modality": "text",
    },
    "fast_text": {
        "primary": "liquid/lfm-2.5-1.2b-instruct:free",
        "modality": "text",
    },
}

# ---------------------------------------------------------------------------
# Dataset-specific system prompts
# ---------------------------------------------------------------------------

_TASK_FAMILY_PROMPTS: dict[str, str] = {
    "pushing": (
        "You are controlling a 2D pusher on a tabletop. "
        "Your goal is to push the T-shaped block to the green target zone. "
        "You can output move_to(x, y) to push the block in a direction, "
        "or stop to wait. No gripper is available. "
        "The image shows the top-down view of the workspace at 512x512 pixels. "
        "Output your intent as structured JSON with tool, target, params, and reasoning."
    ),
    "pick_place": (
        "You see a tabletop scene with objects to manipulate. "
        "You control a robot arm with a parallel-jaw gripper. "
        "Output tool calls to pick up objects and place them at targets. "
        "Available tools: move_to (arm motion with x,y,z target), "
        "grasp (close gripper on object), lift (vertical raise), "
        "place (lower and release), release (open gripper), stop. "
        "Use target='ObjectName' in move_to instead of raw coordinates when possible. "
        "Always: approach from above, descend, grasp, lift slightly, "
        "move to destination, descend, release, retreat."
    ),
    "bi_manipulation": (
        "You control a dual-arm robot (left and right arms, each 7-DOF). "
        "Both arms need to coordinate: one picks up the cube and transfers "
        "it to the other arm. "
        "Available tools: move_to, grasp, release, stop. "
        "Pay attention to the camera views: cam_high shows the full scene, "
        "cam_left_wrist and cam_right_wrist show each gripper's perspective. "
        "Coordinate both arms by reasoning about their current positions."
    ),
    "mobile_manipulation": (
        "You are controlling a robot arm on a mobile base. "
        "Your task is to open a cabinet door by grasping the handle and pulling. "
        "Available tools: move_to (position arm near handle), "
        "grasp (close gripper on handle), pull (move arm backward to open door), "
        "release (let go of handle), stop. "
        "The handle position is shown in the image. Grasp it firmly, "
        "then pull consistently until the door opens."
    ),
    "reaching": (
        "You are controlling a robot arm in a reaching task. "
        "Your goal is to reach toward objects on a table. "
        "No gripper is needed. Output move_to(x, y, z) to position "
        "the end effector near the target object, or stop to wait. "
        "Focus on accurate positioning from the camera view."
    ),
}

_DEFAULT_PROMPT: str = (
    "You control a robot arm. Analyze the image and output "
    "the next action as structured JSON with tool, params, target, "
    "and reasoning fields. Choose from: move_to, grasp, release, "
    "lift, place, pull, push, or stop."
)

# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

class SmartRouter:
    """Smart model router that picks the right model for each task type.

    Features:
      - Task-appropriate model selection from MODEL_TIERS
      - Automatic fallback on primary model failure
      - Dataset-specific system prompts
      - Full call pipeline with structured output parsing
      - Convenience method to create benchmark provider callables

    Usage::

        router = SmartRouter()
        model = router.get_model_for_task("vision_reasoning")
        prompt = router.get_system_prompt(task_family="pick_place")
        result = router.call_reasoning(
            image_b64="data:image/jpeg;base64,...",
            task_prompt="What action should the robot take?",
            tools_schema={"name": "tool_call", ...},
            task_type="vision_reasoning",
        )
    """

    def __init__(self, default_provider: str | None = None) -> None:
        """Initialize the SmartRouter.

        Args:
            default_provider: Provider name to use (e.g. "openrouter", "cerebras").
                              If None, reads from LLM_PROVIDER env var (default "openrouter").
        """
        self._default_provider = default_provider or os.environ.get(
            "LLM_PROVIDER", "openrouter"
        ).strip().lower()
        self._provider_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def get_model_for_task(
        self, task_type: str, prefer_free: bool = True
    ) -> str:
        """Return the best model name for a given task type.

        Args:
            task_type: One of the MODEL_TIERS keys (vision_reasoning,
                       json_structuring, object_labeling, text_reasoning, fast_text).
            prefer_free: If True, prefer free tier models.

        Returns:
            Model name string (e.g. "google/gemma-4-31b-it:free").
        """
        tier = MODEL_TIERS.get(task_type)
        if tier is None:
            # Default to text_reasoning for unknown task types
            tier = MODEL_TIERS["text_reasoning"]

        if prefer_free and "primary" in tier:
            return tier["primary"]
        return tier.get("paid", tier.get("primary", "google/gemma-4-31b-it"))

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    def get_system_prompt(
        self,
        task_family: str | None = None,
        dataset_config: dict[str, Any] | None = None,
    ) -> str:
        """Return the system prompt for a given task family or dataset config.

        Args:
            task_family: One of "pushing", "pick_place", "bi_manipulation",
                         "mobile_manipulation", "reaching", or None.
            dataset_config: Optional dataset config dict (from DATASET_REGISTRY).
                            If provided, task_family is extracted from it.

        Returns:
            System prompt string.
        """
        if dataset_config is not None:
            task_family = dataset_config.get("task_family", task_family)
            task_type = dataset_config.get("task_type", "")
            description = dataset_config.get("description", "")
            tools = dataset_config.get("tools", ["move_to", "stop"])
            camera_keys = dataset_config.get("camera_keys", [])

            base_prompt = _TASK_FAMILY_PROMPTS.get(
                task_family or "",
                _DEFAULT_PROMPT,
            )

            # Append dataset-specific details
            extra = (
                f"\n\nDataset: {task_type} ({description})\n"
                f"Available cameras: {', '.join(camera_keys) if camera_keys else 'N/A'}\n"
                f"Tool set: {', '.join(tools)}\n"
            )
            return base_prompt + extra

        if task_family is None:
            return _DEFAULT_PROMPT

        return _TASK_FAMILY_PROMPTS.get(task_family, _DEFAULT_PROMPT)

    # ------------------------------------------------------------------
    # LLM provider factory
    # ------------------------------------------------------------------

    def _get_provider(self, model: str) -> Any:
        """Get or create an LLM provider for the given model name.

        Uses model-specific OpenRouter providers with a cache.
        """
        if model in self._provider_cache:
            return self._provider_cache[model]

        from src.provider import OpenRouterProvider

        provider = OpenRouterProvider(model=model)
        self._provider_cache[model] = provider
        return provider

    def _call_with_retry(
        self,
        provider: Any,
        image_b64: str | None,
        task_prompt: str,
        system_prompt: str | None,
        tools_schema: dict[str, Any] | None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> dict[str, Any] | str | None:
        """Call the provider with the given parameters, handling JSON output.

        Returns:
            Parsed response: dict if tools_schema provided and JSON parsed,
            str otherwise, or None on failure.
        """
        try:
            if image_b64:
                result = provider.image_chat(
                    prompt=task_prompt,
                    image_b64=image_b64,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=tools_schema,
                )
            else:
                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": task_prompt})
                result = provider.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=tools_schema,
                )

            content = result.content.strip()
            if not content:
                return None

            # If tools_schema was provided, try to parse as JSON
            if tools_schema is not None:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to fix truncated JSON
                    import re
                    fixed = content
                    if fixed.count('"') % 2 == 1:
                        fixed += '"'
                    open_braces = fixed.count("{") - fixed.count("}")
                    if open_braces > 0:
                        fixed += "}" * open_braces
                    open_brackets = fixed.count("[") - fixed.count("]")
                    if open_brackets > 0:
                        fixed += "]" * open_brackets
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        # Extract tool name as last resort
                        m = re.search(r'"tool"\s*:\s*"(\w+)"', raw)
                        if m:
                            return {"tool": m.group(1), "params": {}, "reasoning": "parse_fallback"}
                        return None

            return content

        except Exception as exc:
            return None

    # ------------------------------------------------------------------
    # Full reasoning pipeline
    # ------------------------------------------------------------------

    def call_reasoning(
        self,
        image_b64: str | None,
        task_prompt: str,
        tools_schema: dict[str, Any] | None = None,
        task_type: str = "vision_reasoning",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        prefer_free: bool = True,
    ) -> dict[str, Any] | str | None:
        """Full call pipeline with model selection and automatic fallback.

        Args:
            image_b64: Base64 image URI (or None for text-only).
            task_prompt: The task instruction/description.
            tools_schema: JSON schema for structured output (or None for free text).
            task_type: The MODEL_TIERS key.
            system_prompt: Optional system prompt (auto-generated if None).
            temperature: LLM temperature.
            max_tokens: Max tokens in response.
            prefer_free: Prefer free tier models.

        Returns:
            Parsed JSON dict (if tools_schema provided) or string, or None if all models fail.
        """
        tier = MODEL_TIERS.get(task_type, MODEL_TIERS["text_reasoning"])
        modality = tier.get("modality", "text")

        # If there's no image and the tier requires vision, switch to text
        if image_b64 is None and modality == "vision":
            task_type = "text_reasoning"
            tier = MODEL_TIERS["text_reasoning"]

        # Build fallback chain
        models_to_try: list[str] = []
        if prefer_free and "primary" in tier:
            models_to_try.append(tier["primary"])
        if "paid" in tier and tier["paid"] != tier.get("primary"):
            models_to_try.append(tier["paid"])
        if "fallback" in tier:
            fb = tier["fallback"]
            if fb not in models_to_try:
                models_to_try.append(fb)

        # Ensure at least one model
        if not models_to_try:
            models_to_try = ["google/gemma-4-31b-it:free"]

        last_error: str | None = None

        for model in models_to_try:
            try:
                provider = self._get_provider(model)

                # Auto-generate system prompt if not provided and task has a family
                effective_system = system_prompt

                result = self._call_with_retry(
                    provider=provider,
                    image_b64=image_b64,
                    task_prompt=task_prompt,
                    system_prompt=effective_system,
                    tools_schema=tools_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if result is not None:
                    return result

                last_error = f"{model}: returned None"
            except Exception as exc:
                last_error = f"{model}: {exc}"
                continue

        # All models failed
        return None

    # ------------------------------------------------------------------
    # Benchmark provider factory
    # ------------------------------------------------------------------

    def create_benchmark_provider(
        self,
        task_type: str = "vision_reasoning",
        task_family: str | None = None,
        dataset_config: dict[str, Any] | None = None,
        prefer_free: bool = True,
    ) -> Callable[[str, list[float], list[float]], dict[str, Any] | None]:
        """Create a provider callable suitable for DatasetReplayEngine.run_benchmark().

        The returned callable matches the expected signature:
            (image_b64: str, state: list[float], action: list[float]) -> dict | None

        Args:
            task_type: MODEL_TIERS key for model selection.
            task_family: Task family for system prompt generation.
            dataset_config: Dataset config for prompt generation.
            prefer_free: Prefer free tier models.

        Returns:
            Callable that can be passed as the ``provider`` argument to
            ``DatasetReplayEngine.run_benchmark()``.
        """
        system_prompt = self.get_system_prompt(
            task_family=task_family,
            dataset_config=dataset_config,
        )

        # Build the tools schema for the intent output
        tools_schema = {
            "name": "tool_call",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "Which tool to use",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target object name or position",
                    },
                    "params": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {},
                        "description": "Tool parameters (x, y, z, gripper_open, etc.)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind this action choice",
                    },
                },
                "required": ["tool", "params", "reasoning"],
            },
        }

        def _provider(image_b64: str, state: list[float], action: list[float]) -> dict[str, Any] | None:
            """Call the smart router with the benchmark frame data."""
            result = self.call_reasoning(
                image_b64=image_b64 or None,
                task_prompt=(
                    f"Robot state: {[round(v, 3) for v in state[:6]]}...\n"
                    f"Previous action: {[round(v, 3) for v in action[:6]]}...\n\n"
                    f"{system_prompt}\n\n"
                    "Analyze the current scene and output the next robot action as structured JSON."
                ),
                tools_schema=tools_schema,
                task_type=task_type,
                system_prompt=system_prompt,
                prefer_free=prefer_free,
            )

            # Normalize the result: ensure it has the fields GemmaIntent expects
            if isinstance(result, dict):
                if "tool" not in result:
                    return None
                if "reasoning" not in result:
                    result["reasoning"] = ""
                result.setdefault("target", "")
                result.setdefault("params", {})
                result.setdefault("confidence", 1.0)
                result.setdefault("task_type", "")
                return result

            return None

        return _provider

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def available_tiers(self) -> list[str]:
        """Return sorted list of all available model tier names."""
        return sorted(MODEL_TIERS.keys())

    def tier_info(self, task_type: str) -> dict[str, str] | None:
        """Return tier metadata for a given task type."""
        tier = MODEL_TIERS.get(task_type)
        if tier is None:
            return None
        return dict(tier)

    def best_vision_model(self) -> str:
        """Shortcut: return the best vision model name."""
        return self.get_model_for_task("vision_reasoning")

    def best_text_model(self) -> str:
        """Shortcut: return the best text reasoning model name."""
        return self.get_model_for_task("text_reasoning")


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

_default_router: SmartRouter | None = None


def default_router() -> SmartRouter:
    """Return the module-level SmartRouter singleton."""
    global _default_router
    if _default_router is None:
        _default_router = SmartRouter()
    return _default_router
