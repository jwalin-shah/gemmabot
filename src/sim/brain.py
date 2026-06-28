"""The brain — Gemma 4 perceives the rendered scene and picks the next action.

The model is given the image (with the zone grid), the instruction, the robot's
own proprioception, and the *vocabulary* of object ids/labels it may reference —
but never object positions. It must look at the image to decide which object and
which zone. Coordinate precision comes from the skill layer (the id bridge).
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
            "target": {"type": "string", "description": "Exact object id or bin name; '' for stop/done"},
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
with a labelled zone grid (Zone A-F). Each tick you choose the SINGLE next micro-action.

Skills:
- pick <object_id>   : approach AND grasp an object. This automatically moves the gripper \
to the object over several ticks and closes when it arrives. To pick something up, issue \
`pick` directly and keep issuing it until your robot state shows you are holding it. Do NOT \
`move_to` an object first.
- place <bin_name>   : carry the held object to a bin and release it.
- move_to <id|bin>   : reposition the gripper WITHOUT grasping (rarely needed).
- stop               : halt immediately (use if told to stop or a hazard appears).
- done               : the instruction is fully satisfied.

Rules:
- LOOK at the image. Identify objects visually and report the zone you SEE the target in.
- To grasp something, use `pick` and repeat it until your robot state says you are holding \
it; the skill handles the approach automatically.
- Once you are holding the target, `place` it where instructed, then output `done`.
- Reference objects only by the exact ids you are given. Never invent coordinates.
- The world can change between ticks (objects move, new commands arrive). Always decide \
from the CURRENT image, not memory.
- Respect constraints in the instruction (e.g. "don't touch the blue cup").
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


def _vocab_block(labels: dict[str, str], bins: list[str]) -> str:
    obj_lines = "\n".join(f"  - {oid}  (looks like: {lbl})" for oid, lbl in labels.items())
    return f"Object ids you may reference:\n{obj_lines}\nBins: {', '.join(bins) or '(none)'}"


class RobotBrain:
    """Wraps a CerebrasClient (or any client exposing ``image_chat``)."""

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
            f"{_vocab_block(labels, bins)}\n\n"
            f"Robot state: {proprioception or 'gripper empty'}\n\n"
            "Look at the image and output the single next action."
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
        return Decision(skill="stop", target="", reasoning="unparseable response",
                        latency_ms=latency_ms, raw=content or "")
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
    """Offline brain for testing the physics spine without API calls.

    With a world reference it drives a simple pick-the-cracked-cup-then-place
    behaviour so you can verify the sim/skills before wiring in Gemma."""

    def __init__(self, world: World | None = None) -> None:
        self._world = world

    def decide(self, instruction, image, labels, bins, proprioception="") -> Decision:
        if self._world is None:
            return Decision(skill="done", target="", reasoning="no world")
        g = self._world.gripper
        bin_name = bins[0] if bins else ""
        if g.holding is None:
            target = "cracked_cup" if "cracked_cup" in labels else next(iter(labels), "")
            return Decision(skill="pick", target=target, target_zone="B",
                            observed="mock view", reasoning="grab the cracked cup")
        return Decision(skill="place", target=bin_name, target_zone="D",
                        observed="mock view", reasoning="drop it in the bin")
