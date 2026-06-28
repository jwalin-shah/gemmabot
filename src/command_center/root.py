"""Treehouse Root — Gemma 4 31B as the always-on watcher and router.

The Root continuously watches incoming signals (images, text, voice), decides
what to route and to whom, and dispatches to branch agents — all within a
single synchronous loop that runs at Cerebras speed.

Key architectural insight:
  On GPU (1-5s/call): Router is too slow, so you pipeline linearly.
  On Cerebras (50-150ms/call): Router runs in the hot path at 5-10 Hz,
  enabling true reactive multi-agent coordination.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src import encode_image
from src.client import CerebrasClient, InferenceResult
from src.command_center.branches import BranchRegistry
from src.command_center.types import (
    Branch,
    BranchOutput,
    RoutingDecision,
    SignalType,
    CommandCenterCommand,
    CommandCenterLoopResult,
    CommandCenterObservation,
    CommandCenterSignal,
    Urgency,
)

# JSON Schema for structured routing output — forces Gemma 4 to return
# a parseable decision every time, deterministically.
ROUTING_SCHEMA = {
    "name": "routing_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "observed": {
                "type": "string",
                "description": "One-line summary of what was observed",
            },
            "route_to": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [b.value for b in Branch],
                },
                "description": "Which branches to delegate to",
            },
            "parallel": {
                "type": "boolean",
                "description": "Dispatch to all branches simultaneously?",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
            },
            "instruction": {
                "type": "string",
                "description": "Specific instruction for the branch agents",
            },
            "hazards": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any hazards detected",
            },
            "command": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["action", "target", "reasoning"],
                "additionalProperties": False,
            },
            "requires": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional context required before acting",
            },
        },
        "required": [
            "observed",
            "route_to",
            "parallel",
            "priority",
            "instruction",
            "hazards",
            "command",
            "requires",
        ],
        "additionalProperties": False,
    },
}

ROUTER_SYSTEM_PROMPT = """You are the root watcher of a robot Command Center. You watch incoming signals from the world and decide what to do.

Your job:
1. OBSERVE — quickly assess what's in front of you (image, text command, sensor data)
2. ROUTE — decide which specialist branches should handle this
3. COMMAND — if urgent, output an immediate action command

Available branches:
- vision — analyze camera images for objects, layout, hazards
- action_planner — generate step-by-step robot action sequences
- safety — review plans for hazards and constraint violations
- coordinator — manage multi-agent coordination tasks
- summarizer — condense information
- oracle — general reasoning and问答

Rules:
- Be fast. This runs at 5-10 Hz. Prioritize speed over perfection.
- If you see a hazard, route to safety immediately.
- For images, always route to vision.
- For text commands, route to the relevant branch.
- You can route to MULTIPLE branches in parallel.
- If the situation is urgent (fire, collision risk), set priority to high or critical.

Output ONLY valid JSON matching the schema. No explanations."""


class CommandCenterRoot:
    """The root watcher — Gemma 4 31B continuously watching and routing.

    Usage:
        root = CommandCenterRoot(client, registry)
        result = root.watch(signal)         # single observation
        async for result in root.loop():    # continuous watch loop
            print(result.commands)
    """

    def __init__(
        self,
        client: CerebrasClient,
        registry: BranchRegistry,
    ) -> None:
        self._client = client
        self._registry = registry
        self._registry.register_all()

        # State
        self._last_observation: CommandCenterObservation | None = None
        self._context: dict[str, Any] = {}
        self._loop_count = 0
        self._total_router_time_ms = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(
        self,
        signal: CommandCenterSignal,
    ) -> CommandCenterLoopResult:
        """Process ONE signal through the Command Center: observe → route → execute.

        This is the core loop of the Command Center. On Cerebras this completes
        in 150-500ms total. On GPU it would take 3-15s.
        """
        result = CommandCenterLoopResult(signal=signal)
        t_start = time.perf_counter()

        # Step 1: Observe — send signal to Gemma 4 for routing decision
        t_router = time.perf_counter()
        decision = self._route(signal)
        result.router_latency_ms = (time.perf_counter() - t_router) * 1000

        if decision is None:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return result

        result.decision = decision
        result.observation = CommandCenterObservation(
            signal=signal,
            summary=decision.context_hint or decision.instruction,
            hazards=decision.route_to if Branch.SAFETY in decision.route_to else [],
        )

        # Step 2: Route — dispatch to branch agents
        branch_prompts: dict[Branch, str] = {}
        system_prompts: dict[Branch, str] = {}
        context: dict[str, Any] = {}

        # Build context from signal
        if signal.type == SignalType.IMAGE:
            context["image_b64"] = signal.payload
        context["signal_type"] = signal.type.value
        context["signal_source"] = signal.source

        for b in decision.route_to:
            branch_prompts[b] = decision.instruction
            # Set specific context per branch
            if b == Branch.VISION and signal.type == SignalType.IMAGE:
                branch_prompts[b] = f"{decision.instruction}\n\nAnalyze this scene."
                context["image_b64"] = signal.payload

        if decision.route_to and decision.parallel:
            # Parallel dispatch — all branches at once
            result.branch_outputs = self._registry.run_parallel(
                decision.route_to,
                branch_prompts,
                context=context,
                system_prompts=system_prompts,
            )
        elif decision.route_to:
            # Sequential dispatch
            for b in decision.route_to:
                out = self._registry.run_branch(
                    b, branch_prompts.get(b, ""),
                    context=context,
                    system_prompt=system_prompts.get(b),
                )
                result.branch_outputs.append(out)

        # Step 3: Synthesize commands
        result.commands = [CommandCenterCommand(
            action=decision.command.get("action", "wait") if decision.command else "wait",
            target=decision.command.get("target", "") if decision.command else "",
            reasoning=decision.command.get("reasoning", "") if decision.command else "",
        )]

        # Store context for next iteration
        self._last_observation = result.observation
        self._context["last_scene"] = result.observation.summary
        for out in result.branch_outputs:
            self._context[f"last_{out.branch.value}"] = out.content

        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        self._loop_count += 1
        self._total_router_time_ms += result.router_latency_ms

        return result

    def watch_image(
        self,
        image_path: str,
        task: str = "Explore the environment.",
    ) -> CommandCenterLoopResult:
        """Convenience: watch an image file with a text task."""
        image_b64 = encode_image(image_path)
        signal = CommandCenterSignal(
            type=SignalType.IMAGE,
            payload=image_b64,
            source="camera",
            metadata={"task": task, "path": image_path},
        )
        return self.watch(signal)

    def watch_text(self, text: str) -> CommandCenterLoopResult:
        """Convenience: watch a text command."""
        signal = CommandCenterSignal(
            type=SignalType.TEXT,
            payload=text,
            source="user",
        )
        return self.watch(signal)

    @property
    def stats(self) -> dict[str, Any]:
        """Command Center performance statistics."""
        return {
            "loops": self._loop_count,
            "avg_router_ms": (
                self._total_router_time_ms / self._loop_count
                if self._loop_count else 0
            ),
            "branches_registered": len(self._registry._branches),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _route(self, signal: CommandCenterSignal) -> RoutingDecision | None:
        """Send a signal to Gemma 4 31B and parse the routing decision.

        Uses structured outputs (JSON schema) to force a deterministic
        parseable response every time.
        """
        # Build the observation prompt
        if signal.type == SignalType.IMAGE:
            user_content = (
                f"Signal: Camera image from '{signal.source}'."
            )
            extra = signal.metadata.get("task", "")
            if extra:
                user_content += f"\nTask: {extra}"
            user_content += "\n\nWhat do you see? What should the robot do?"

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {"type": "image_url", "image_url": {"url": signal.payload}},
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Signal ({signal.type.value}): {signal.payload}\n\n"
                        "What should the robot do?"
                    ),
                },
            ]

        # Call Gemma 4 with structured output schema
        try:
            resp = self._client._client.chat.completions.create(
                model="gemma-4-31b",
                messages=messages,
                temperature=0.1,
                max_completion_tokens=512,
                response_format={
                    "type": "json_schema",
                    "json_schema": ROUTING_SCHEMA,
                },
            )
        except Exception as e:
            return None

        content = resp.choices[0].message.content
        if not content:
            return None

        # Parse
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        # Map to RoutingDecision
        branches = []
        for b_name in data.get("route_to", []):
            try:
                branches.append(Branch(b_name))
            except ValueError:
                pass

        urgency_map = {
            "low": Urgency.LOW,
            "medium": Urgency.MEDIUM,
            "high": Urgency.HIGH,
            "critical": Urgency.CRITICAL,
        }

        return RoutingDecision(
            route_to=branches,
            priority=urgency_map.get(data.get("priority", "low"), Urgency.LOW),
            instruction=data.get("instruction", ""),
            parallel=data.get("parallel", True),
            context_hint=data.get("observed", ""),
            command=data.get("command", {}),
        )
