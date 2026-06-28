"""The reactive control loop — perceive -> decide -> act, once per tick.

This is the spine. Each tick renders the current scene, asks the brain for the
next action, executes one step of it, and applies physics. Because it re-decides
every tick from the current image, moving an object mid-task makes the gripper
track it — and a slow brain (GPU) acts on a stale picture.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.sim.brain import Decision
from src.sim.skills import execute, resolve_target
from src.sim.world import World


@dataclass
class TickResult:
    tick: int
    decision: Decision
    status: str  # running | done | error


class ReactiveLoop:
    def __init__(self, world: World, brain) -> None:
        self.world = world
        self.brain = brain
        self.instruction = ""
        self.history: list[TickResult] = []

    def set_instruction(self, text: str) -> None:
        self.instruction = text

    def _proprioception(self) -> str:
        g = self.world.gripper
        gz = self.world.gripper_zone()
        if g.holding:
            held = self.world.get(g.holding)
            return f"holding {g.holding} ({held.label if held else ''}), gripper in Zone {gz}"
        return f"gripper empty and open, currently in Zone {gz}"

    def tick(self) -> TickResult:
        image = self.world.render()
        labels = {o.id: o.label for o in self.world.objects.values()}
        bins = list(self.world.bins)
        decision = self.brain.decide(
            self.instruction, image, labels, bins, proprioception=self._proprioception()
        )
        resolved_target = resolve_target(decision.target, self.world) if decision.skill in ('pick', 'move_to', 'place') else decision.target
        status = execute(self.world, decision.skill, resolved_target or decision.target)
        self.world.physics()
        self.world.tick += 1
        result = TickResult(tick=self.world.tick, decision=decision, status=status)
        self.history.append(result)
        return result
