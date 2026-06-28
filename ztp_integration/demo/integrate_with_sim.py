"""
Integration demo: John Kruze G^G physics meets the Gemma 4 multi-agent pipeline.

This demo runs the existing hackathon tabletop sim (World, objects, reactive loop)
alongside ZTP physics validation, showing how the Cerebras-powered AI decisions
get grounded in real physics at 1000Hz.

Usage:
    # Auto-detect native library (ZTP_LIB_PATH env or vendor/ztp-runtime)
    uv run python ztp_integration/demo/integrate_with_sim.py

    # Force mock
    uv run python ztp_integration/demo/integrate_with_sim.py --mock

    # Point to specific library
    ZTP_LIB_PATH=/path/to/libztp_runtime.dylib uv run python ztp_integration/demo/integrate_with_sim.py

No API keys needed — the brain uses MockBrain so the demo is fully offline.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make the project root importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztp_integration.c_ffi.bridge import ZTPRuntime


# ---------------------------------------------------------------------------
# Inline the hackathon sim types (self-contained, no src/ dependency)
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


def build_world() -> World:
    w = World()
    w.objects["red_cup"] = SimObject("red_cup", "red cup", (210, 60, 60), x=200, y=300)
    w.objects["blue_cup"] = SimObject("blue_cup", "blue cup", (60, 90, 210), x=440, y=300)
    w.objects["cracked_cup"] = SimObject("cracked_cup", "cracked tan cup", (200, 175, 120), x=330, y=180, attribute="cracked")
    w.bins["bin_left"] = (85, 360)
    return w


# ---------------------------------------------------------------------------
# ZTP Physics Validation Layer
# ---------------------------------------------------------------------------

@dataclass
class PhysicsValidation:
    approved: bool
    risk_score: float  # 0.0 = safe → 1.0 = max risk
    compaction: float = 0.0
    force_n: float = 0.0
    rejection_reason: str = ""


class ZTPActionValidator:
    """Validates robot actions against ZTP physics before execution."""

    def __init__(self, ztp: ZTPRuntime) -> None:
        self._ztp = ztp
        self._call_count = 0
        self._total_latency_ms = 0.0

    def validate_pick(self, world: World, target_id: str) -> PhysicsValidation:
        obj = world.objects.get(target_id)
        if obj is None:
            return PhysicsValidation(approved=False, risk_score=1.0, rejection_reason="target not found")

        t0 = time.perf_counter()
        terran = self._ztp.terran_evaluate_contact(
            soil_type=1, moisture=0.2, mass_kg=0.8,
            footprint_m2=0.005, locomotion=0,
        )
        surgical = self._ztp.surgical_evaluate_grasp(
            tissue_type=0, measured_force_n=2.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency_ms += latency

        compaction = terran["max_compaction"]
        overstress = surgical["tissue_overstress_detected"]
        clamped_force = surgical["clamped_force"]

        risks = []
        if compaction > 0.8:
            risks.append(f"soil compaction {compaction:.2f} > 0.8")
        if overstress:
            risks.append(f"force limit exceeded (clamped at {clamped_force:.2f}N)")

        risk_score = min(1.0, compaction * 0.7 + (0.3 if overstress else 0))
        approved = risk_score < 0.6

        return PhysicsValidation(
            approved=approved,
            risk_score=round(risk_score, 4),
            compaction=round(compaction, 6),
            force_n=clamped_force,
            rejection_reason="; ".join(risks) if risks else "",
        )

    def validate_place(self, world: World, bin_name: str) -> PhysicsValidation:
        t0 = time.perf_counter()
        terran = self._ztp.terran_evaluate_contact(
            soil_type=1, moisture=0.2, mass_kg=0.5,
            footprint_m2=0.01, locomotion=0,
        )
        latency = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency_ms += latency

        approved = terran["max_compaction"] < 0.9
        return PhysicsValidation(
            approved=approved,
            risk_score=round(terran["max_compaction"], 4),
            compaction=round(terran["max_compaction"], 6),
            rejection_reason="bin surface too soft" if not approved else "",
        )

    def stats(self) -> dict:
        n = self._call_count or 1
        return {"calls": self._call_count, "avg_latency_ms": round(self._total_latency_ms / n, 3)}


# ---------------------------------------------------------------------------
# Execution with ZTP validation
# ---------------------------------------------------------------------------

def execute_with_validation(world: World, skill: str, target: str, validator: ZTPActionValidator) -> dict:
    if skill == "pick":
        validation = validator.validate_pick(world, target)
    elif skill == "place":
        validation = validator.validate_place(world, target)
    else:
        validation = PhysicsValidation(approved=True, risk_score=0.0)

    if not validation.approved:
        return {
            "executed": False, "status": "physically_infeasible",
            "reason": validation.rejection_reason, "risk_score": validation.risk_score,
            "compaction": validation.compaction, "force_n": validation.force_n,
        }

    result = _execute_skill(world, skill, target)
    return {
        "executed": True, "status": result,
        "reason": "", "risk_score": validation.risk_score,
        "compaction": validation.compaction, "force_n": validation.force_n,
    }


def _execute_skill(world: World, skill: str, target: str) -> str:
    def step_toward(tx: float, ty: float) -> bool:
        g = world.gripper
        dx, dy = tx - g.x, ty - g.y
        dist = math.hypot(dx, dy)
        if dist <= STEP:
            g.x, g.y = tx, ty
            return dist <= REACH
        g.x += STEP * dx / dist
        g.y += STEP * dy / dist
        return False

    if skill == "pick":
        obj = world.objects.get(target)
        if obj is None:
            return "error"
        if world.gripper.holding == target:
            return "done"
        if step_toward(obj.x, obj.y):
            world.gripper.holding = target
            world.gripper.closed = True
            return "done"
        return "running"

    elif skill == "place":
        dest = world.bins.get(target)
        if dest is None:
            return "error"
        if step_toward(dest[0], dest[1]):
            held_id = world.gripper.holding
            if held_id and (held := world.objects.get(held_id)):
                held.x, held.y = dest
            world.gripper.holding = None
            world.gripper.closed = False
            return "done"
        return "running"

    elif skill == "done":
        return "done"

    return "error"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cerebras x ZTP Integration Demo")
    parser.add_argument("--mock", action="store_true", default=None,
                        help="Force mock ZTP (no native library)")
    parser.add_argument("--lib", type=str, default=None,
                        help="Path to native libztp_runtime.dylib")
    parser.add_argument("--ticks", type=int, default=8, help="Number of sim ticks")
    args = parser.parse_args()

    ztp = ZTPRuntime(lib_path=args.lib)

    # Determine mode
    if args.mock:
        mode = "MOCK (--mock flag set)"
    elif ztp.available:
        mode = "NATIVE (real Rust physics kernel)"
    else:
        mode = "MOCK (no native library found)"

    print(f"ZTP Runtime: {mode}")
    if not ztp.available and not args.mock:
        print("  Build it: bash ztp_integration/scripts/build_ztp.sh")
    print()

    world = build_world()
    validator = ZTPActionValidator(ztp)
    instruction = "Put the cracked cup into bin_left."

    print("=" * 74)
    print("  Cerebras × John Kruze G^G Physics Integration Demo")
    print(f"  Task: {instruction}")
    print("=" * 74)
    print()
    print(f"{'tick':>5s} | {'skill':8s} {'target':12s} | {'status':22s} | {'physics':35s}")
    print("-" * 74)

    actions = [
        ("pick", "cracked_cup"),
        ("pick", "cracked_cup"),
        ("pick", "cracked_cup"),
        ("pick", "cracked_cup"),
        ("pick", "cracked_cup"),
        ("place", "bin_left"),
        ("place", "bin_left"),
        ("done", ""),
    ]

    for i in range(min(args.ticks, len(actions))):
        skill, target = actions[i]
        report = execute_with_validation(world, skill, target, validator)

        phys_str = f"compact={report['compaction']:.4f} risk={report['risk_score']:.2f}"
        if report.get("force_n", 0) > 0:
            phys_str += f" force={report['force_n']:.2f}N"

        if report["executed"]:
            status = report["status"]
        else:
            status = f"❌ BLOCKED"

        print(f"{i + 1:>5d} | {skill:8s} {target:12s} | {status:22s} | {phys_str:35s}")
        if report.get("reason"):
            print(f"       ⚠️  {report['reason']}")

    print("-" * 74)

    g = world.gripper
    print(f"\nFinal: gripper at ({g.x:.0f}, {g.y:.0f}), "
          f"holding={g.holding or 'nothing'}")
    print(f"ZTP: {validator.stats()['calls']} calls @ {validator.stats()['avg_latency_ms']}ms avg")


if __name__ == "__main__":
    main()
