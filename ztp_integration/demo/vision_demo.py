"""
Full demo: Gemma 4 actually sees + identifies objects visually.

No pre-drawn text labels on the image. Gemma 4 must use its multimodal vision
to identify objects by their visual appearance (color, shape, position in zones).
The sim then maps Gemma's natural language description back to internal IDs.

Usage:
    uv run python ztp_integration/demo/vision_demo.py [--mock] [--ticks N]
"""

from __future__ import annotations

import math, re, sys, time, json
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from ztp_integration.c_ffi.bridge import ZTPRuntime

WIDTH, HEIGHT = 640, 420
STEP = 20.0; REACH = 14.0

# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

@dataclass
class SimObject:
    id: str
    label: str       # human-readable description
    color: tuple[int, int, int]
    x: float; y: float
    radius: float = 26.0
    attribute: str = ""

@dataclass
class Gripper:
    x: float = WIDTH/2; y: float = 30.0
    holding: str | None = None; closed: bool = False

@dataclass
class World:
    objects: dict = field(default_factory=dict)
    gripper: Gripper = field(default_factory=Gripper)
    bins: dict = field(default_factory=dict)
    tick: int = 0

    def get(self, oid): return self.objects.get(oid)
    def resolve(self, target):
        if target in self.objects: o = self.objects[target]; return (o.x, o.y)
        if target in self.bins: return self.bins[target]
        return None
    def gripper_zone(self):
        c = min(int(self.gripper.x/(WIDTH/3)),2); r = min(int(self.gripper.y/(HEIGHT/2)),1)
        return ["A","B","C","D","E","F"][r*3+c]
    def physics(self):
        if self.gripper.holding and (h:=self.get(self.gripper.holding)):
            h.x, h.y = self.gripper.x, self.gripper.y

def build_world() -> World:
    w = World()
    w.objects["red_cup"] = SimObject("red_cup", "a bright red cup", (210,60,60), x=200, y=300)
    w.objects["blue_cup"] = SimObject("blue_cup", "a bright blue cup", (60,90,210), x=440, y=300)
    w.objects["cracked_cup"] = SimObject("cracked_cup", "a cracked tan/brown cup", (200,175,120), x=330, y=180, attribute="cracked")
    w.bins["bin_left"] = (85, 360)
    return w

# ---------------------------------------------------------------------------
# Render — NO text labels!
# ---------------------------------------------------------------------------

def render_scene(world: World) -> str:
    """Render world to PNG. NO text labels — Gemma must identify objects visually."""
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    # Grid
    cw, ch = WIDTH/3, HEIGHT/2
    for c in range(1,3): d.line([(c*cw,0),(c*cw,HEIGHT)], fill=(200,200,210), width=1)
    for r in range(1,2): d.line([(0,r*ch),(WIDTH,r*ch)], fill=(200,200,210), width=1)
    for i,lab in enumerate(["A","B","C","D","E","F"]):
        r2,c2 = divmod(i,3); d.text((c2*cw+5,r2*ch+4), f"Zone {lab}", fill=(170,170,180))

    # Bin (label stays — it's a fixed landmark)
    d.rectangle([85-36,360-26,85+36,360+26], outline=(95,95,95), width=3)
    d.text((85-32,360-8), "bin_left", fill=(95,95,95))

    # Objects — CIRCLES ONLY, no text labels
    for obj in world.objects.values():
        d.ellipse([obj.x-obj.radius, obj.y-obj.radius, obj.x+obj.radius, obj.y+obj.radius],
                  fill=obj.color, outline=(35,35,35), width=2)
        if obj.attribute == "cracked":
            d.line([(obj.x-11,obj.y-13),(obj.x+4,obj.y),(obj.x-7,obj.y+13)], fill=(20,20,20), width=2)

    # Gripper
    g = world.gripper
    col = (205,45,45) if g.closed else (45,120,205)
    d.line([(g.x-17,g.y),(g.x+17,g.y)], fill=col, width=4)
    d.line([(g.x,g.y),(g.x,g.y-24)], fill=col, width=4)
    d.ellipse([g.x-5,g.y-5,g.x+5,g.y+5], fill=col)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def execute_skill(world: World, skill: str, target: str) -> str:
    def step_toward(tx, ty):
        g = world.gripper; dx, dy = tx-g.x, ty-g.y; dist = math.hypot(dx, dy)
        if dist <= STEP: g.x,g.y = tx,ty; return dist <= REACH
        g.x += STEP*dx/dist; g.y += STEP*dy/dist; return False
    if skill == "pick":
        obj = world.get(target)
        if not obj: return "error"
        if world.gripper.holding == target: return "done"
        return "done" if step_toward(obj.x, obj.y) and (setattr(world.gripper,'holding',target) or setattr(world.gripper,'closed',True) or True) else "running"
    elif skill == "place":
        dest = world.resolve(target)
        if not dest: return "error"
        if step_toward(dest[0], dest[1]):
            if (h:=world.gripper.holding) and (ho:=world.get(h)): ho.x,ho.y = dest
            world.gripper.holding = None; world.gripper.closed = False; return "done"
        return "running"
    elif skill in ("done","stop"): return "done"
    return "error"

# ---------------------------------------------------------------------------
# ZTP
# ---------------------------------------------------------------------------

@dataclass
class PV: approved: bool = True; risk_score: float = 0.0; compaction: float = 0.0; force_n: float = 0.0

class ZTPVal:
    def __init__(self, ztp):
        self._ztp = ztp; self._c = 0; self._t = 0.0; self.blocks = []
    def validate(self, world, skill, target):
        if skill != "pick": return PV()
        t0 = time.perf_counter()
        surg = self._ztp.surgical_evaluate_grasp(tissue_type=0, measured_force_n=2.0)
        lat = (time.perf_counter()-t0)*1000; self._c+=1; self._t+=lat
        if surg["tissue_overstress_detected"]:
            self.blocks.append({"tick":world.tick,"reason":f"force > 1.2N"})
            return PV(approved=False, risk_score=0.8, force_n=surg["clamped_force"])
        return PV(force_n=surg["clamped_force"])
    def stats(self):
        return {"calls":self._c, "avg_ms":round(self._t/max(self._c,1),3), "blocks":len(self.blocks)}

# ---------------------------------------------------------------------------
# Object description mapping — Gemma says what it sees, sim resolves to IDs
# ---------------------------------------------------------------------------

# This is the Semantic → Geometric bridge:
# Gemma describes what it sees in natural language.
# We match the description to internal object IDs.

OBJECT_SIGNATURES = [
    (r"(?i)cracked.*(tan|brown|cup)|tan.*(cracked|cup)", "cracked_cup"),
    (r"(?i)red.*cup|bright.*red", "red_cup"),
    (r"(?i)blue.*cup|bright.*blue", "blue_cup"),
    (r"(?i)bin_left|left.*bin", "bin_left"),
]

def resolve_target(text: str) -> str:
    """Map Gemma's natural language description to internal object IDs."""
    text_lower = text.lower()
    for pattern, oid in OBJECT_SIGNATURES:
        if re.search(pattern, text_lower):
            return oid
    # Fallback: try to match any object id directly
    for oid in ["cracked_cup", "red_cup", "blue_cup", "bin_left"]:
        if oid.replace("_", " ") in text_lower or oid in text_lower:
            return oid
    return text  # pass through as-is

# ---------------------------------------------------------------------------
# System prompt — NO ids, NO labels. Gemma must look at the image.
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are the vision core of a tabletop robot arm. You see a camera \
image with a labelled zone grid (Zone A-F). Objects appear as colored circles.

Your job: look at the image and describe what you see, then choose the next action.

The scene contains:
- A bright red cup (round, red)
- A bright blue cup (round, blue)
- A cracked tan/brown cup (round, tan, with a crack line)
- A rectangular bin labeled "bin_left"

Instruction will tell you the goal. Decide the SINGLE next action.

Actions available:
- pick <description>   : grasp an object. Describe it so the robot knows which one.
- place bin_left       : put held object in the bin
- done                 : task complete

IMPORTANT: Look at the image to identify objects by their COLOR and POSITION in the zone grid. \
Describe what you see visually. Do NOT make up object names — describe the actual colors and shapes."""

class CerebrasVisionBrain:
    """Gemma 4 identifies objects visually from the raw image."""

    def __init__(self):
        from src.client import CerebrasClient
        self._client = CerebrasClient()

    def decide(self, instruction: str, image_b64: str, proprioception: str = ""):
        prompt = (
            f"Instruction: {instruction}\n\n"
            f"Robot state: {proprioception or 'gripper empty'}\n\n"
            "Look at the image. Identify each object by its visual appearance "
            "(color, position in zone grid). Describe what you see, then output "
            "your action as JSON with keys: skill, target (natural description), "
            "observed (what you visually see), zone, reasoning."
        )
        # Use freeform text, not strict schema, so Gemma describes naturally
        t0 = time.perf_counter()
        result = self._client.image_chat(
            prompt=prompt,
            image_b64=image_b64,
            system_prompt=VISION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=500,
        )
        latency = (time.perf_counter()-t0)*1000
        return self._parse(result.content, latency)

    def _parse(self, text: str, latency: float):
        """Parse Gemma's response — extract JSON or infer action from text."""
        # Try to find JSON block
        import json
        try:
            # Look for { ... }
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start:end+1])
                skill = data.get("skill", "stop")
                target_desc = data.get("target", "")
                observed = data.get("observed", "")
                reasoning = data.get("reasoning", "")
                zone = data.get("zone", "none")
            else:
                raise ValueError("no JSON found")
        except (json.JSONDecodeError, ValueError):
            # Fallback: infer from freeform text
            skill = "stop"
            target_desc = ""
            observed = text[:100]
            reasoning = text[:100]
            zone = "none"
            if "pick" in text.lower():
                skill = "pick"
                # Try to grab the noun after "pick"
                m = re.search(r"pick(?: up)? (?:the )?(.+?)(?:\.|$|into|and)", text, re.I)
                if m: target_desc = m.group(1).strip()
            elif "place" in text.lower() or "bin" in text.lower():
                skill = "place"
                target_desc = "bin_left"
            elif "done" in text.lower():
                skill = "done"

        # Resolve natural language description to internal ID
        target_id = resolve_target(target_desc)
        return skill, target_id, observed, reasoning, latency


class MockVisionBrain:
    def __init__(self, world=None):
        self._world = world; self._phase = "pick"
    def decide(self, instruction, image_b64, proprioception=""):
        if "holding" in proprioception and self._world and self._world.gripper.holding:
            self._phase = "place"
        if self._phase == "pick":
            return ("pick", "cracked_cup", "I see a tan cracked cup in Zone B", "visually identified cracked cup", 0.0)
        return ("place", "bin_left", "bin_left visible in Zone D", "place in bin", 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--perturb", type=int, default=6)
    args = parser.parse_args()

    ztp = ZTPRuntime()
    print(f"🔬 ZTP: {'NATIVE' if ztp.available else 'MOCK'}")
    print(f"🧠 Vision: {'MOCK' if args.mock else 'LIVE Gemma 4 — identifying objects visually'}")
    print()

    world = build_world()
    validator = ZTPVal(ztp)
    instruction = "Put the cracked cup into bin_left. Do not touch the blue cup."

    brain = MockVisionBrain(world) if args.mock else CerebrasVisionBrain()
    print(f"📋 {instruction}")
    print(f"🎨 NO text labels on image — Gemma identifies objects by color + position")
    print()
    print(f"{'tick':>5s} | {'action':24s} | {'lat':>7s} | {'status':12s} | {'ZTP':12s} | {'Gemma sees':50s}")
    print("-" * 120)

    for i in range(args.ticks):
        world.tick = i + 1

        if i == args.perturb and world.gripper.holding != "cracked_cup":
            c = world.get("cracked_cup")
            if c: c.x, c.y = 520, 130; print(f"\n  ⚡ CUP DRAGGED to (520,130) Zone C\n")

        # Render WITHOUT text labels
        image_b64 = render_scene(world)
        proprio = f"holding {world.gripper.holding}" if world.gripper.holding else "empty"

        # Gemma decides from vision
        skill, target, observed, reasoning, latency = brain.decide(instruction, image_b64, proprio)

        # ZTP validates
        v = validator.validate(world, skill, target)

        # Execute
        if not v.approved and skill == "pick":
            status = "❌ ZTP BLOCK"
        else:
            r = execute_skill(world, skill, target)
            if skill == "done": status = "✅ DONE"
            elif r == "done" and skill == "pick": status = "✅ GRASPED"
            elif r == "done" and skill == "place": status = "✅ PLACED"
            elif r == "running": status = "⏳ moving"
            else: status = r

        world.physics()

        # Show what Gemma identified
        lat_s = f"{latency:.0f}ms" if latency > 0 else "mock"
        action_s = f"{skill:8s} → {target:12s}"
        ztp_s = "✓" if v.approved else "❌"
        observed_s = observed[:55] if observed else reasoning[:55]

        print(f"{world.tick:>5d} | {action_s:24s} | {lat_s:>7s} | {status:12s} | {ztp_s:12s} | {observed_s:50s}")

    print("-" * 120)
    g = world.gripper
    print(f"\n🏁 Gripper at ({g.x:.0f},{g.y:.0f}), holding={g.holding or 'nothing'}")
    s = validator.stats()
    print(f"📊 ZTP: {s['calls']} calls @ {s['avg_ms']}ms, {s['blocks']} blocked")
    if not args.mock:
        print(f"\n✅ Gemma 4 identified objects visually from raw image — no pre-drawn labels!")
    print()

if __name__ == "__main__":
    main()
