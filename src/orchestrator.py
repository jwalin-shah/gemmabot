"""Orchestrator — coordinates multi-agent pipeline with parallel dispatch.

Pipeline modes:
  1. ``parallel`` (default): Vision + Action run in parallel, both seeing the image.
     Safety then reviews both outputs. ~500-600ms total.
  2. ``single_shot``: One multimodal call returns scene + plan + safety in a single
     Gemini 4 31B call. ~350-400ms total. Fastest option.
  3. ``sequential``: Original mode — Vision → Action → Safety in series. ~900ms total.
     Kept as a fallback for compatibility.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src import encode_image
from src.agents import ActionAgent, SafetyAgent, VisionAgent
from src.client import CerebrasClient, InferenceResult
from src.robot_controller import ActionResult, RobotController


class PipelineMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SINGLE_SHOT = "single_shot"


@dataclass
class OrchestrationResult:
    scene_analysis: str = ""
    action_plan: str = ""
    safety_review: str = ""
    executed_actions: list[ActionResult] = field(default_factory=list)
    total_time_s: float = 0.0
    pipeline: dict[str, float] = field(default_factory=dict)
    mode: str = "parallel"


# ---------------------------------------------------------------------------
# Single-shot structured output schema
# ---------------------------------------------------------------------------
SINGLE_SHOT_SCHEMA = {
    "name": "robot_pipeline",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scene_analysis": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Objects visible in the scene",
                    },
                    "layout": {"type": "string"},
                    "hazards": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "grasp_targets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["objects", "layout", "hazards", "grasp_targets"],
                "additionalProperties": False,
            },
            "action_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "move_to", "pick_up", "place", "push",
                                "pause", "navigate_home",
                            ],
                        },
                        "target": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "distance_m": {"type": "number"},
                                "direction": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["action", "target", "reason"],
                    "additionalProperties": False,
                },
                "description": "Sequence of robot actions",
            },
            "safety_review": {
                "type": "object",
                "properties": {
                    "safe": {"type": "boolean"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recommendation": {"type": "string"},
                    "severity": {"type": "number"},
                },
                "required": ["safe", "risk_level", "issues", "recommendation", "severity"],
                "additionalProperties": False,
            },
        },
        "required": ["scene_analysis", "action_plan", "safety_review"],
        "additionalProperties": False,
    },
}

SINGLE_SHOT_SYSTEM_PROMPT = """You are a robot command center. Look at the camera image and produce
three outputs in one response:

1. scene_analysis — What objects do you see? Layout? Hazards? Grasp targets?
2. action_plan — A sequence of robot commands to accomplish the task.
   Commands: move_to, pick_up, place, push, pause, navigate_home.
3. safety_review — Is this plan safe? What are the risks?

Assume a robotic arm with gripper, wheeled base, and basic sensors.
Be concise. Output ONLY valid JSON matching the schema."""


class AgentOrchestrator:
    """Runs the multi-agent pipeline with configurable parallelism.

    Usage:
        orch = AgentOrchestrator(client)
        result = orch.run("image.jpg", mode="parallel")   # default
        result = orch.run("image.jpg", mode="single_shot") # fastest
        result = orch.run("image.jpg", mode="sequential")  # original
    """

    def __init__(
        self,
        client: CerebrasClient,
        robot: RobotController | None = None,
    ) -> None:
        self._client = client
        self._vision = VisionAgent(client)
        self._action = ActionAgent(client)
        self._safety = SafetyAgent(client)
        self._robot = robot or RobotController()

        # Shared thread pool for parallel dispatch
        self._pool = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str,
        task: str = "Identify objects, pick up anything graspable, report findings.",
        mode: str = "parallel",
    ) -> OrchestrationResult:
        """Run the multi-agent pipeline.

        Args:
            image_path: Path to the robot camera image.
            task: Task description for the robot.
            mode: One of "sequential", "parallel" (default), "single_shot".

        Returns:
            OrchestrationResult with all pipeline outputs and timing.
        """
        mode_enum = PipelineMode(mode)

        if mode_enum == PipelineMode.SINGLE_SHOT:
            return self._run_single_shot(image_path, task)
        elif mode_enum == PipelineMode.PARALLEL:
            return self._run_parallel(image_path, task)
        else:
            return self._run_sequential(image_path, task)

    # ------------------------------------------------------------------
    # Mode 1: Sequential — original, one agent after another
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        image_path: str,
        task: str,
    ) -> OrchestrationResult:
        result = OrchestrationResult(mode="sequential")
        t0 = time.perf_counter()

        image_b64 = self._encode(image_path, result)

        t1 = time.perf_counter()
        result.scene_analysis = self._vision.analyze(image_b64).content
        result.pipeline["vision_agent"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        result.action_plan = self._action.plan(result.scene_analysis, task).content
        result.pipeline["action_agent"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        result.safety_review = self._safety.review(result.scene_analysis, result.action_plan).content
        result.pipeline["safety_agent"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        result.executed_actions = self._execute_plan(result.action_plan, result.safety_review)
        result.pipeline["execute"] = time.perf_counter() - t4

        result.total_time_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Mode 2: Parallel — Vision + Action concurrently, both see image
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        image_path: str,
        task: str,
    ) -> OrchestrationResult:
        result = OrchestrationResult(mode="parallel")
        t0 = time.perf_counter()

        image_b64 = self._encode(image_path, result)

        # Stage 1: Vision + Action in parallel — BOTH see the image
        t1 = time.perf_counter()

        vision_future = self._pool.submit(self._vision.analyze, image_b64)
        action_future = self._pool.submit(
            self._action.plan, "", task, image_b64
        )

        vision_res = vision_future.result()
        action_res = action_future.result()

        result.scene_analysis = vision_res.content
        result.action_plan = action_res.content
        result.pipeline["vision_agent"] = vision_res.latency_s
        result.pipeline["action_agent"] = action_res.latency_s
        stage_1_wall = time.perf_counter() - t1
        result.pipeline["stage_1_parallel_wall"] = stage_1_wall

        # Stage 2: Safety reviews both outputs (text only — fast)
        t2 = time.perf_counter()
        result.safety_review = self._safety.review(
            result.scene_analysis, result.action_plan
        ).content
        result.pipeline["safety_agent"] = time.perf_counter() - t2

        # Stage 3: Execute
        t3 = time.perf_counter()
        result.executed_actions = self._execute_plan(result.action_plan, result.safety_review)
        result.pipeline["execute"] = time.perf_counter() - t3

        result.total_time_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Mode 3: Single-shot — one multimodal call, structured output
    # ------------------------------------------------------------------

    def _run_single_shot(
        self,
        image_path: str,
        task: str,
    ) -> OrchestrationResult:
        result = OrchestrationResult(mode="single_shot")
        t0 = time.perf_counter()

        image_b64 = self._encode(image_path, result)

        t1 = time.perf_counter()
        try:
            # Use structured outputs (JSON schema) for deterministic parsing
            resp = self._client._client.chat.completions.create(
                model="gemma-4-31b",
                messages=[
                    {
                        "role": "system",
                        "content": SINGLE_SHOT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Task: {task}\n\n"
                                    "Analyze the image and return: scene_analysis (objects,"
                                    " layout, hazards, grasp_targets), action_plan (array of"
                                    " actions), and safety_review (safe, risk_level, issues)."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_b64},
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                response_format={
                    "type": "json_schema",
                    "json_schema": SINGLE_SHOT_SCHEMA,
                },
            )

            content = resp.choices[0].message.content
            elapsed = time.perf_counter() - t1
            result.pipeline["single_shot_call"] = elapsed

            data = json.loads(content) if content else {}

            # Unpack the three outputs
            scene = data.get("scene_analysis", {})
            result.scene_analysis = json.dumps(scene, indent=2)

            actions = data.get("action_plan", [])
            result.action_plan = json.dumps(actions, indent=2)

            safety = data.get("safety_review", {})
            result.safety_review = json.dumps(safety, indent=2)

        except Exception as exc:
            result.pipeline["single_shot_error"] = str(exc)
            # Fall back to parallel if single-shot fails
            result.pipeline["single_shot_fallback"] = True
            fallback = self._run_parallel(image_path, task)
            result.scene_analysis = fallback.scene_analysis
            result.action_plan = fallback.action_plan
            result.safety_review = fallback.safety_review
            result.executed_actions = fallback.executed_actions
            result.pipeline = fallback.pipeline
            result.total_time_s = time.perf_counter() - t0
            return result

        t2 = time.perf_counter()
        result.executed_actions = self._execute_plan(result.action_plan, result.safety_review)
        result.pipeline["execute"] = time.perf_counter() - t2

        result.total_time_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(image_path: str, result: OrchestrationResult) -> str:
        t = time.perf_counter()
        b64 = encode_image(image_path)
        result.pipeline["encode_image"] = time.perf_counter() - t
        return b64

    @staticmethod
    def _parse_actions(text: str) -> list[dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.splitlines(True)
            cleaned = "".join(cleaned[1:])
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed.get("actions", [parsed])
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    def _execute_plan(self, action_plan: str, safety_review: str) -> list[ActionResult]:
        try:
            review_text = safety_review.strip()
            if review_text.startswith("```"):
                review_text = review_text.splitlines(True)
                review_text = "".join(review_text[1:]).rsplit("```", 1)[0]
            review = json.loads(review_text)
        except json.JSONDecodeError:
            review = {"safe": False, "risk_level": "unknown"}

        if not review.get("safe", False):
            msg = review.get("recommendation", "risk too high")
            return [ActionResult("safety_block", "all", "skipped", message=msg)]

        actions = self._parse_actions(action_plan)
        if not actions:
            return [
                ActionResult(
                    "parse_error", "none", "failed",
                    message="Could not parse action plan",
                )
            ]

        results: list[ActionResult] = []
        for act in actions:
            r = self._robot.execute(
                action=act.get("action", "pause"),
                target=act.get("target", "unknown"),
                **act.get("parameters", {}),
            )
            results.append(r)
        return results
