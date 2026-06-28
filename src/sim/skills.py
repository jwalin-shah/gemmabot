"""Skill layer — the low-level controller.

Each skill is a small scripted motion the model can invoke by name. The model
decides *which* skill on *which* object (semantic); these functions resolve the
object id to ground-truth coordinates and move the gripper (geometric). One
``execute`` call advances the motion by a single step, so the loop re-decides
every tick and the gripper tracks targets that move.
"""

from __future__ import annotations

import math

import re
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



def resolve_target(description: str, world: World) -> str | None:
    """Fuzzy-match a visual description to the best object ID.
    
    Gemma says "the cracked tan cup in Zone B" -> we match by color + attribute + zone.
    """
    desc_lower = description.lower().strip()
    if not desc_lower:
        return None
    
    # Check bins first
    for bin_name in world.bins:
        if bin_name in desc_lower:
            return bin_name
    
    best_match = None
    best_score = 0
    
    for oid, obj in world.objects.items():
        score = 0
        # Color match
        color_map = {
            (210, 60, 60): "red",
            (60, 90, 210): "blue", 
            (200, 175, 120): "tan",
        }
        obj_color_name = color_map.get(obj.color, "")
        if obj_color_name and obj_color_name in desc_lower:
            score += 3
        
        # Attribute match
        if obj.attribute == "cracked" and ("crack" in desc_lower or "broken" in desc_lower or "mark" in desc_lower):
            score += 3
        
        # Zone match
        zone = world.zone_of(oid)
        if zone and zone.lower() in desc_lower:
            score += 2
        
        # Label match (last resort partial)
        for word in obj.label.lower().split():
            if word in desc_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = oid
    
    # Only return if we have some confidence
    return best_match if best_score >= 2 else None



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
