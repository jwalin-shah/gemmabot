"""The brain — Gemma 4 perceives the rendered scene and picks the next action.

The model is given the image (with the zone grid), the instruction, the robot's
own proprioception — but NO object IDs or text labels. It must identify objects
visually by their color, shape, marks, and position in the grid.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from PIL import Image

from src.client import CerebrasClient
from src.sim.world import World, image_to_data_uri

DECISION_SCHEMA = {
    "name": "robot_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "observed": {"type": "string", "description": "What you see in the image right now"},
            "reasoning": {"type": "string", "description": "One short sentence: why this action"},
            "skill": {"type": "string", "enum": ["pick", "place", "move_to", "stop", "done"]},
            "target": {
                "type": "string",
                "description": "VISUAL description of the target: color, marks, zone. E.g. 'the cracked tan cup in Zone B' or 'the blue cup'. Empty for stop/done.",
            },
            "target_zone": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "E", "F", "none"],
                "description": "The grid zone you SEE the target in",
            },
        },
        "required": ["observed", "reasoning", "skill", "target", "target_zone"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are the reasoning core of a tabletop robot arm. You see a camera image \
with a zone grid (Zone A-F). The image has NO text labels - you must identify objects \
by their VISUAL APPEARANCE only.

Skills:
- pick <visual description> : approach AND grasp. Describe the target by how it looks.
- place <bin_name>          : carry held object to a bin (rectangle on the table).
- move_to <target>          : reposition (rarely needed).
- stop                      : halt immediately.
- done                      : task is fully satisfied.

Rules:
- LOOK at each new image. Objects are colored circles. Some have marks (cracked = two diagonal black lines through a tan circle).
- Identify objects by COLOR, MARKS, POSITION, and ZONE. Never use text labels.
- Describe the target VISUALLY in the target field. The system resolves your description.
- To grasp something, issue 'pick' each tick until robot state says you are holding it.
- Once holding, 'place' it where instructed, then output 'done'.
- The world changes between ticks. Always decide from the CURRENT image.
- Respect constraints (e.g. "don't touch the blue cup").
- Output one action only, matching the schema."""


@dataclass
class Decision:
    skill: str
    target: str
    target_zone: str = "none"
    observed: str = ""
    reasoning: str = ""
    latency_ms: float = 0.0
    raw: str = ""


def _describe_objects(world=None) -> str:
    """Visual description - no IDs, just appearance hints."""
    return "Objects on the table: colored circles. One tan cup has a crack mark (two diagonal black lines). Others are solid colored circles."


class RobotBrain:
    """Wraps a CerebrasClient - perceives scene and picks action visually."""

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
        prompt = (
            f"Instruction: {instruction}\n\n"
            f"Bins (rectangles on table for dropping objects): {', '.join(bins) if bins else 'none'}\n\n"
            f"Robot state: {proprioception or 'gripper empty and open'}\n\n"
            "LOOK at the image. Identify objects by their VISUAL APPEARANCE - color, marks, zone position. "
            "Describe the target object visually (e.g. 'the cracked tan cup in Zone B'). "
            "The system matches your visual description to the right object."
        )
        t0 = time.perf_counter()
        result = self._client.image_chat(
            prompt=prompt,
            image_b64=image_to_data_uri(image),
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=350,
            response_format={"type": "json_schema", "json_schema": DECISION_SCHEMA},
        )
        latency = (time.perf_counter() - t0) * 1000
        return _parse(result.content, latency)


def _parse(content: str, latency_ms: float) -> Decision:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return Decision(
            skill="stop", target="", reasoning="unparseable response",
            latency_ms=latency_ms, raw=content or ""
        )
    return Decision(
        skill=data.get("skill", "stop"),
        target=data.get("target", ""),
        target_zone=data.get("target_zone", "none"),
        observed=data.get("observed", ""),
        reasoning=data.get("reasoning", ""),
        latency_ms=latency_ms,
        raw=content,
    )


class MockBrain:
    """Offline brain - no API calls, uses world state directly."""

    def __init__(self, world: World | None = None) -> None:
        self._world = world

    def decide(self, instruction, image, labels, bins, proprioception="") -> Decision:
        if self._world is None:
            return Decision(skill="done", target="", reasoning="no world")
        g = self._world.gripper
        bin_name = bins[0] if bins else ""
        if g.holding is None:
            target = "cracked_cup" if "cracked_cup" in labels else next(iter(labels), "")
            return Decision(
                skill="pick", target=target, target_zone="B",
                observed="mock view", reasoning="grab the cracked cup"
            )
        return Decision(
            skill="place", target=bin_name, target_zone="D",
            observed="mock view", reasoning="drop it in the bin"
        )
