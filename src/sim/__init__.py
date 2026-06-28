"""Reasoning-in-the-loop tabletop robot simulation.

Spine: a pure-Python sim (`World`), a scripted skill layer (`skills`), a Gemma 4
brain that perceives the rendered scene (`RobotBrain`), and the reactive control
loop that ties them together (`ReactiveLoop`).
"""

from __future__ import annotations

from src.sim.brain import Decision, MockBrain, RobotBrain
from src.sim.loop import ReactiveLoop, TickResult
from src.sim.skills import SKILLS, execute
from src.sim.world import Gripper, SimObject, World, image_to_data_uri

__all__ = [
    "World",
    "SimObject",
    "Gripper",
    "image_to_data_uri",
    "RobotBrain",
    "MockBrain",
    "Decision",
    "ReactiveLoop",
    "TickResult",
    "SKILLS",
    "execute",
]
