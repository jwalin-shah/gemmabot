"""Tool-calling brain using Gemma 4 native function calling.

Instead of asking the model to output a JSON string (which must be parsed), this
brain defines robot actions as OpenAI-compatible tool definitions and lets Gemma 4
call them directly via the Cerebras tools API.
"""

from __future__ import annotations

import json
import time

from PIL import Image

from src.client import CerebrasClient
from src.sim.brain import Decision
from src.sim.world import image_to_data_uri

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Approach and grasp an object by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "The exact object ID to pick up",
                    },
                    "observed_zone": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D", "E", "F", "none"],
                        "description": "Which zone you see the object in",
                    },
                },
                "required": ["object_id", "observed_zone"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": "Place the held object into a bin",
            "parameters": {
                "type": "object",
                "properties": {
                    "bin_name": {
                        "type": "string",
                        "description": "The bin to place the object into",
                    },
                },
                "required": ["bin_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move the gripper to a location without grasping",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Object ID or bin name to move toward",
                    },
                },
                "required": ["target"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the instruction is fully satisfied",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished",
                    },
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are the reasoning core of a tabletop robot arm. You see a camera image "
    "with a labelled zone grid (Zone A-F). Each tick you choose the SINGLE next "
    "micro-action by calling one of the available tool functions.\n\n"
    "Rules:\n"
    "- LOOK at the image. Identify objects visually.\n"
    "- To grasp something, call `pick` with the exact object_id and the zone "
    "you see it in.\n"
    "- Once you are holding the target, call `place` with the bin name.\n"
    "- When the task is complete, call `done`.\n"
    "- The world can change between ticks. Always decide from the CURRENT image.\n"
    "- Respect constraints in the instruction (e.g. \"don't touch the blue cup\").\n"
    "- Call ONE tool per tick."
)


class ToolCallingBrain:
    """Brain that uses Gemma 4's native tool/function calling API.

    Instead of parsing a JSON blob from the model's text output, this brain
    defines robot actions as OpenAI-compatible tool definitions. Gemma 4
    natively picks the next action by calling the appropriate tool.
    """

    def __init__(self, client: CerebrasClient) -> None:
        self._client = client

    def decide(
        self,
        instruction: str,
        image: Image.Image,
        labels: dict[str, str],
        bins: list[str],
        proprioception: str = "",
    ) -> Decision:
        """Given the current image + instruction, return the next action as a
        native tool call from Gemma 4.

        Args:
            instruction: The natural-language task instruction.
            image: The current camera frame with zone grid overlay.
            labels: Mapping of object_id -> semantic label (e.g. "red cup").
            bins: List of bin names.
            proprioception: Robot state description (holding what, gripper zone).

        Returns:
            A Decision dataclass populated from the model's tool call.
        """
        # Build the vocabulary block
        obj_lines = "\n".join(
            f"  - {oid}  (looks like: {lbl})" for oid, lbl in labels.items()
        )
        vocab = (
            f"Object ids you may reference:\n{obj_lines}\n"
            f"Bins: {', '.join(bins) or '(none)'}"
        )

        prompt = (
            f"Instruction: {instruction}\n\n"
            f"{vocab}\n\n"
            f"Robot state: {proprioception or 'gripper empty'}\n\n"
            "Look at the image and call the appropriate tool."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_uri(image)},
                    },
                ],
            },
        ]

        t0 = time.perf_counter()
        result = self._client.chat(
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
            temperature=0.1,
            max_tokens=500,
        )
        latency = (time.perf_counter() - t0) * 1000

        # --- Tool call path ---
        if result.tool_calls:
            tc = result.tool_calls[0]
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            raw = json.dumps({"tool": name, "arguments": args})
            return self._tool_to_decision(name, args, latency, raw)

        # --- Text fallback path ---
        content = result.content or ""
        return self._text_fallback(content, latency)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tool_to_decision(
        self,
        name: str,
        args: dict,
        latency_ms: float,
        raw: str,
    ) -> Decision:
        """Map a tool call from the model into a ``Decision`` dataclass."""
        if name == "pick":
            oid = args.get("object_id", "")
            zone = args.get("observed_zone", "none")
            return Decision(
                skill="pick",
                target=oid,
                target_zone=zone,
                observed=f"Picking {oid} in zone {zone}",
                reasoning=f"Grasping object {oid}",
                latency_ms=latency_ms,
                raw=raw,
            )

        if name == "place":
            bin_name = args.get("bin_name", "")
            return Decision(
                skill="place",
                target=bin_name,
                target_zone="none",
                observed=f"Placing into bin {bin_name}",
                reasoning=f"Dropping held object into {bin_name}",
                latency_ms=latency_ms,
                raw=raw,
            )

        if name == "move_to":
            target = args.get("target", "")
            return Decision(
                skill="move_to",
                target=target,
                target_zone="none",
                observed=f"Moving toward {target}",
                reasoning=f"Repositioning to {target}",
                latency_ms=latency_ms,
                raw=raw,
            )

        if name == "done":
            return Decision(
                skill="done",
                target="",
                target_zone="none",
                observed="Task complete",
                reasoning=args.get("summary", "Instruction satisfied"),
                latency_ms=latency_ms,
                raw=raw,
            )

        return Decision(
            skill="stop",
            target="",
            target_zone="none",
            observed="",
            reasoning=f"Unknown tool call: {name}",
            latency_ms=latency_ms,
            raw=raw,
        )

    def _text_fallback(self, content: str, latency_ms: float) -> Decision:
        """Handle a text-only response (no tool call) as a fallback.

        With ``tool_choice="required"`` this should rarely happen, but we handle
        it gracefully by scanning for known action keywords.
        """
        lowered = content.lower().strip()

        # Check for completion sentinels
        if any(word in lowered for word in ("done", "complete", "finished")):
            return Decision(
                skill="done",
                target="",
                reasoning="Model signalled completion in text",
                latency_ms=latency_ms,
                raw=content,
            )

        # Check for action keywords in the text
        for word in ("pick", "place", "move_to", "stop"):
            if word in lowered:
                return Decision(
                    skill=word,
                    target="",
                    observed=content[:200],
                    reasoning="Parsed from text fallback",
                    latency_ms=latency_ms,
                    raw=content,
                )

        return Decision(
            skill="stop",
            target="",
            observed=content[:200],
            reasoning="No tool call returned; stopping",
            latency_ms=latency_ms,
            raw=content,
        )
