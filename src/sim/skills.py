"""Skill layer — the low-level controller.

Each skill is a small scripted motion the model can invoke by name. The model
decides *which* skill on *which* object (semantic); these functions resolve the
object id to ground-truth coordinates and move the gripper (geometric). One
``execute`` call advances the motion by a single step, so the loop re-decides
every tick and the gripper tracks targets that move.
"""

from __future__ import annotations

import math

from src.sim.world import World

STEP = 20.0  # gripper travel per tick, in pixels
REACH = 14.0  # distance at which the gripper is "at" the target


def _step_toward(world: World, tx: float, ty: float) -> bool:
    g = world.gripper
    dx, dy = tx - g.x, ty - g.y
    dist = math.hypot(dx, dy)
    if dist <= STEP:
        g.x, g.y = tx, ty
        return dist <= REACH
    g.x += STEP * dx / dist
    g.y += STEP * dy / dist
    return False


def pick(world: World, target: str) -> str:
    obj = world.get(target)
    if obj is None:
        return "error"
    if world.gripper.holding == target:
        return "done"
    arrived = _step_toward(world, obj.x, obj.y)
    if arrived:
        world.gripper.holding = target
        world.gripper.closed = True
        return "done"
    return "running"


def place(world: World, target: str) -> str:
    dest = world.resolve(target)
    if dest is None:
        return "error"
    arrived = _step_toward(world, dest[0], dest[1])
    if arrived:
        held_id = world.gripper.holding
        if held_id and (held := world.get(held_id)) is not None:
            held.x, held.y = dest
        world.gripper.holding = None
        world.gripper.closed = False
        return "done"
    return "running"


def move_to(world: World, target: str) -> str:
    dest = world.resolve(target)
    if dest is None:
        return "error"
    return "done" if _step_toward(world, dest[0], dest[1]) else "running"


def stop(world: World, target: str) -> str:
    # Halt in place — the gripper holds position this tick.
    return "done"


SKILLS = {
    "pick": pick,
    "place": place,
    "move_to": move_to,
    "stop": stop,
}


def execute(world: World, skill: str, target: str) -> str:
    """Run one step of a skill. ``done`` for unknown skills (e.g. the model's
    ``done`` sentinel) so the loop can recognise task completion."""
    fn = SKILLS.get(skill)
    if fn is None:
        return "done" if skill == "done" else "error"
    return fn(world, target)
