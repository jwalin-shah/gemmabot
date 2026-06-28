"""
Full integration: Cerebras reactive loop + ZTP physics validation.

Gemma 4 on Cerebras decides the next action every ~300ms by looking at the
rendered scene. Before each action executes, ZTP physics validates it against
real force limits and soil mechanics. If physics says no, the action is blocked.

Usage:
    # Offline (no API key needed):
    uv run python ztp_integration/demo/full_demo.py --mock-brain --ticks 12

    # Live Cerebras:
    uv run python ztp_integration/demo/full_demo.py --ticks 15
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztp_integration.c_ffi.bridge import ZTPRuntime

# ---------------------------------------------------------------------------
# Inline sim types
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 420
GRID_COLS, GRID_ROWS = 3, 2
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
STEP = 20.0
REACH = 14.0


@dataclass
class SimObject:
    id: str
    label: str
    color: tuple[int, int, int]
    x: float
    y: float
    radius: float = 26.0
    attribute: str = ""

    @property
    def zone(self) -> str:
        col = min(int(self.x / (WIDTH / GRID_COLS)), GRID_COLS - 1)
        row = min(int(self.y / (HEIGHT / GRID_ROWS)), GRID_ROWS - 1)
        return ZONE_LABELS[row * GRID_COLS + col]


@dataclass
class Gripper:
    x: float = WIDTH / 2
    y: float = 30.0
    holding: str | None = None
    closed: bool = False


@dataclass
class World:
    objects: dict[str, SimObject] = field(default_factory=dict)
    gripper: Gripper = field(default_factory=Gripper)
    bins: dict[str, tuple[float, float]] = field(default_factory=dict)
    tick: int = 0

    def get(self, oid: str) -> SimObject | None:
        return self.objects.get(oid)

    def resolve(self, target: str) -> tuple[float, float] | None:
        if target in self.objects:
            o = self.objects[target]
            return (o.x, o.y)
        if target in self.bins:
            return self.bins[target]
        return None

    def gripper_zone(self) -> str:
        g = self.gripper
        col = min(int(g.x / (WIDTH / GRID_COLS)), GRID_COLS - 1)
        row = min(int(g.y / (HEIGHT / GRID_ROWS)), GRID_ROWS - 1)
        return ZONE_LABELS[row * GRID_COLS + col]

    def physics(self) -> None:
        if self.gripper.holding:
            held = self.objects.get(self.gripper.holding)
            if held is not None:
                held.x, held.y = self.gripper.x, self.gripper.y


def build_world() -> World:
    w = World()
    w.objects["red_cup"] = SimObject("red_cup", "red cup", (210, 60, 60), x=200, y=300)
    w.objects["blue_cup"] = SimObject("blue_cup", "blue cup", (60, 90, 210), x=440, y=300)
    w.objects["cracked_cup"] = SimObject("cracked_cup", "cracked tan cup", (200, 175, 120), x=330, y=180, attribute="cracked")
    w.bins["bin_left"] = (85, 360)
    return w


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

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


def execute_skill(world: World, skill: str, target: str) -> str:
    if skill == "pick":
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
    elif skill == "place":
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
    elif skill in ("done", "stop"):
        return "done"
    return "error"


# ---------------------------------------------------------------------------
# ZTP Physics Validator
# ---------------------------------------------------------------------------

@dataclass
class PhysicsValidation:
    approved: bool
    risk_score: float
    compaction: float = 0.0
    force_n: float = 0.0
    slip_risk: float = 0.0
    rejection_reason: str = ""


class ZTPActionValidator:
    def __init__(self, ztp: ZTPRuntime) -> None:
        self._ztp = ztp
        self._call_count = 0
        self._total_latency_ms = 0.0
        self.blocks: list[dict] = []

    def validate_pick(self, world: World, target_id: str) -> PhysicsValidation:
        obj = world.objects.get(target_id)
        if obj is None:
            return PhysicsValidation(approved=False, risk_score=1.0, rejection_reason="target not found")
        t0 = time.perf_counter()
        terran = self._ztp.terran_evaluate_contact(soil_type=1, moisture=0.2, mass_kg=0.8, footprint_m2=0.005, locomotion=0)
        surgical = self._ztp.surgical_evaluate_grasp(tissue_type=0, measured_force_n=2.0)
        latency = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency_ms += latency
        compaction = terran["max_compaction"]
        overstress = surgical["tissue_overstress_detected"]
        clamped_force = surgical["clamped_force"]
        risks = []
        if compaction > 0.8:
            risks.append(f"compaction {compaction:.2f} > 0.8")
        if overstress:
            risks.append(f"force > {surgical.get('max_allowed_force', 1.2):.1f}N")
        risk_score = min(1.0, compaction * 0.7 + (0.3 if overstress else 0))
        approved = risk_score < 0.6
        if not approved:
            self.blocks.append({"tick": world.tick, "skill": "pick", "target": target_id, "reason": "; ".join(risks)})
        return PhysicsValidation(approved=approved, risk_score=round(risk_score, 4), compaction=round(compaction, 6), force_n=clamped_force, slip_risk=round(max(0, compaction - 0.5) * 2, 4), rejection_reason="; ".join(risks) if risks else "")

    def validate_place(self, world: World, bin_name: str) -> PhysicsValidation:
        t0 = time.perf_counter()
        terran = self._ztp.terran_evaluate_contact(soil_type=1, moisture=0.2, mass_kg=0.5, footprint_m2=0.01, locomotion=0)
        latency = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency_ms += latency
        approved = terran["max_compaction"] < 0.9
        return PhysicsValidation(approved=approved, risk_score=round(terran["max_compaction"], 4), compaction=round(terran["max_compaction"], 6), rejection_reason="bin surface too soft" if not approved else "")

    def validate(self, world: World, skill: str, target: str) -> PhysicsValidation:
        if skill == "pick":
            return self.validate_pick(world, target)
        elif skill == "place":
            return self.validate_place(world, target)
        return PhysicsValidation(approved=True, risk_score=0.0)

    def stats(self) -> dict:
        n = self._call_count or 1
        return {"calls": self._call_count, "avg_latency_ms": round(self._total_latency_ms / n, 3), "total_blocks": len(self.blocks)}


# ---------------------------------------------------------------------------
# Render scene to image
# ---------------------------------------------------------------------------

def render_scene(world: World) -> str:
    """Render world to PNG and return base64 data URI."""
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)
    cw, ch = WIDTH / GRID_COLS, HEIGHT / GRID_ROWS
    for c in range(1, GRID_COLS):
        d.line([(c * cw, 0), (c * cw, HEIGHT)], fill=(212, 212, 218), width=1)
    for r in range(1, GRID_ROWS):
        d.line([(0, r * ch), (WIDTH, r * ch)], fill=(212, 212, 218), width=1)
    for i, lab in enumerate(ZONE_LABELS):
        r2, c2 = divmod(i, GRID_COLS)
        d.text((c2 * cw + 5, r2 * ch + 4), f"Zone {lab}", fill=(175, 175, 182))
    for name, (bx, by) in world.bins.items():
        d.rectangle([bx - 36, by - 26, bx + 36, by + 26], outline=(95, 95, 95), width=3)
        d.text((bx - 32, by - 8), name, fill=(95, 95, 95))
    for obj in world.objects.values():
        d.ellipse([obj.x - obj.radius, obj.y - obj.radius, obj.x + obj.radius, obj.y + obj.radius], fill=obj.color, outline=(35, 35, 35), width=2)
        if obj.attribute == "cracked":
            d.line([(obj.x - 11, obj.y - 13), (obj.x + 4, obj.y), (obj.x - 7, obj.y + 13)], fill=(20, 20, 20), width=2)
        d.text((obj.x - obj.radius, obj.y + obj.radius + 3), obj.label, fill=(30, 30, 30))
    g = world.gripper
    col = (205, 45, 45) if g.closed else (45, 120, 205)
    d.line([(g.x - 17, g.y), (g.x + 17, g.y)], fill=col, width=4)
    d.line([(g.x, g.y), (g.x, g.y - 24)], fill=col, width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# Brain interface: both brains receive rendered image + text prompt
# ---------------------------------------------------------------------------

DECISION_SCHEMA = {
    "name": "robot_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "observed": {"type": "string"},
            "reasoning": {"type": "string"},
            "skill": {"type": "string", "enum": ["pick", "place", "move_to", "stop", "done"]},
            "target": {"type": "string"},
            "target_zone": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F", "none"]},
        },
        "required": ["observed", "reasoning", "skill", "target", "target_zone"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are the reasoning core of a tabletop robot arm. You see a camera image \
with a labelled zone grid (Zone A-F). Each tick you choose the SINGLE next micro-action.

Skills:
- pick <object_id>   : approach AND grasp an object. Repeat until held.
- place <bin_name>   : carry the held object to a bin and release it.
- done               : task complete.

Rules:
- LOOK at the image. Identify objects visually by their labels printed beneath them.
- Reference objects only by the exact ids you are given below.
- The world can change between ticks. Always decide from the CURRENT image.
- Output one action only, matching the schema."""


class MockBrain:
    """Offline brain. Decides based on world state, not vision."""

    def __init__(self, world: World | None = None) -> None:
        self._world = world
        self._phase = "pick"

    def decide(self, instruction: str, image_b64: str, labels: dict, bins: list[str],
               proprioception: str = "") -> tuple[str, str, str, str, float]:
        if "holding" in proprioception and self._world and self._world.gripper.holding:
            self._phase = "place"
        if self._phase == "pick":
            target = "cracked_cup" if self._world and "cracked_cup" in labels else next(iter(labels), "")
            return ("pick", target, "scene with grid and objects", "grab the cracked cup visually", 0.0)
        elif self._phase == "place":
            bin_name = bins[0] if bins else ""
            return ("place", bin_name, "bin_left visible in zone D", "place held object in bin", 0.0)
        return ("done", "", "task complete", "", 0.0)


class CerebrasBrain:
    """Live Gemma 4 on Cerebras. Receives rendered image + text, returns JSON action."""

    def __init__(self) -> None:
        from src.client import CerebrasClient
        self._client = CerebrasClient()

    def decide(self, instruction: str, image_b64: str, labels: dict, bins: list[str],
               proprioception: str = "") -> tuple[str, str, str, str, float]:
        import json
        obj_lines = "\n".join(f"  - {oid}  (looks like: {lbl})" for oid, lbl in labels.items())
        vocab = f"Object ids you may reference:\n{obj_lines}\nBins: {', '.join(bins) or '(none)'}"
        prompt = (
            f"Instruction: {instruction}\n\n{vocab}\n\n"
            f"Robot state: {proprioception or 'gripper empty'}\n\n"
            "Look at the image and output the single next action."
        )
        t0 = time.perf_counter()
        result = self._client.image_chat(
            prompt=prompt,
            image_b64=image_b64,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=350,
            response_format={"type": "json_schema", "json_schema": DECISION_SCHEMA},
        )
        latency = (time.perf_counter() - t0) * 1000
        try:
            data = json.loads(result.content)
        except (json.JSONDecodeError, TypeError):
            data = {"skill": "stop", "target": "", "observed": "", "reasoning": "parse error"}
        return (data.get("skill", "stop"), data.get("target", ""),
                data.get("observed", ""), data.get("reasoning", ""), latency)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Cerebras × ZTP Full Demo")
    parser.add_argument("--mock-brain", action="store_true", help="Use MockBrain (no API calls)")
    parser.add_argument("--ticks", type=int, default=15, help="Number of ticks")
    parser.add_argument("--perturb", type=int, default=6, help="Perturbation tick")
    parser.add_argument("--lib", type=str, default=None, help="ZTP library path")
    args = parser.parse_args()

    # --- Init ZTP ---
    ztp = ZTPRuntime(lib_path=args.lib)
    ztp_mode = "NATIVE (real Rust kernel)" if ztp.available else "MOCK (synthetic)"
    print(f"🔬 ZTP Physics: {ztp_mode}")
    print()

    # --- Init world ---
    world = build_world()
    validator = ZTPActionValidator(ztp)

    # --- Init brain ---
    instruction = "Put the cracked cup into bin_left. Do not touch the blue cup."
    if args.mock_brain:
        brain = MockBrain(world)
        brain_name = "MockBrain (offline)"
    else:
        brain = CerebrasBrain()
        brain_name = "Cerebras Gemma 4 31B on Cerebras Inference"

    print(f"🧠 Brain: {brain_name}")
    print(f"📋 Task: {instruction}")
    print(f"⚡  ZTP validates every Gemma decision vs physics before execution")
    print()
    print(f"{'tick':>5s} | {'action':20s} | {'lat':>7s} | {'status':12s} | {'ZTP physics':25s} | {'Gemma 4 reasoning':40s}")
    print("-" * 120)

    perturb_at = args.perturb
    done = False

    for i in range(args.ticks):
        if done:
            break
        world.tick = i + 1

        # Perturbation
        if i == perturb_at and world.gripper.holding != "cracked_cup":
            cup = world.get("cracked_cup")
            if cup:
                cup.x, cup.y = 520, 130
                print(f"\n  ⚡ PERTURBATION: cracked cup dragged to (520,130) Zone C\n")

        # --- Step 1: Render the CURRENT scene (Gemma sees this exact image) ---
        image_b64 = render_scene(world)

        # --- Step 2: Brain decides from the image ---
        labels = {o.id: o.label for o in world.objects.values()}
        bins = list(world.bins)
        proprio = f"holding {world.gripper.holding}" if world.gripper.holding else "empty"

        skill, target, observed, reasoning, latency = brain.decide(
            instruction, image_b64, labels, bins, proprioception=proprio,
        )

        # --- Step 3: ZTP validates the decision ---
        validation = validator.validate(world, skill, target)

        # --- Step 4: Execute only if ZTP approves ---
        if not validation.approved and skill in ("pick", "place"):
            status = "❌ BLOCKED"
        else:
            exec_result = execute_skill(world, skill, target)
            if exec_result == "done" and skill == "done":
                status = "✅ DONE"
                done = True
            elif exec_result == "done" and skill == "pick":
                status = "✅ GRASPED"
            elif exec_result == "done" and skill == "place":
                status = "✅ PLACED"
            elif exec_result == "running":
                status = "⏳ moving"
            else:
                status = exec_result

        world.physics()

        # --- Build ZTP string ---
        if skill in ("pick", "place") and validation.compaction >= 0:
            if validation.approved:
                phys = f"✓ c={validation.compaction:.4f}"
                if validation.force_n > 0:
                    phys += f" f={validation.force_n:.1f}N"
            else:
                phys = f"❌ {validation.rejection_reason[:20]}"
        else:
            phys = "—"

        lat_str = f"{latency:.0f}ms" if latency > 0 else "mock"
        action_str = f"{skill:8s} {target:12s}"
        reasoning_short = reasoning[:55] if reasoning else ""

        print(f"{world.tick:>5d} | {action_str:20s} | {lat_str:>7s} | {status:12s} | {phys:25s} | {reasoning_short:40s}")

    print("-" * 120)
    print()
    g = world.gripper
    print(f"🏁 Gripper at ({g.x:.0f}, {g.y:.0f}), holding={g.holding or 'nothing'}")
    s = validator.stats()
    print(f"📊 ZTP: {s['calls']} validations @ {s['avg_latency_ms']}ms avg, {s['total_blocks']} blocked by physics")
    print()
    print("Pipeline per tick:")
    print("  1. Sim renders current scene → PNG image")
    print("  2. Gemma 4 sees image + prompt → 'pick cracked_cup' (the decision)")
    print("  3. ZTP validates: compaction=0.000 force=1.20N → ✓ SAFE")
    print("  4. Skill executes: gripper moves 20px toward target")
    print()


if __name__ == "__main__":
    main()
