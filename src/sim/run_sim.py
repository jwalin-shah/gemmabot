"""Stage 1 runnable demo of the reactive robot loop.

Builds a tabletop scene, runs the perceive->decide->act loop, and saves a frame
per tick so you can watch the gripper reason and move. A perturbation (the
cracked cup is dragged to a new spot mid-task) demonstrates live tracking.

Usage:
    uv run python -m src.sim.run_sim --mock     # offline, no API calls
    uv run python -m src.sim.run_sim            # live Gemma 4 on Cerebras
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT
from src.sim.brain import MockBrain, RobotBrain
from src.sim.loop import ReactiveLoop
from src.sim.world import SimObject, World

INSTRUCTION = "Put the cracked cup into bin_left. Do not touch the blue cup."
MAX_TICKS = 16
PERTURB_AT = 5  # drag the cracked cup to a new spot at this tick (if not held)


def build_world() -> World:
    w = World()
    w.add(SimObject("red_cup", "red cup", (210, 60, 60), x=200, y=300))
    w.add(SimObject("blue_cup", "blue cup", (60, 90, 210), x=440, y=300))
    w.add(SimObject("cracked_cup", "cracked tan cup", (200, 175, 120), x=330, y=180, attribute="cracked"))
    w.add_bin("bin_left", 85, 360)
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="offline brain, no API")
    parser.add_argument("--ticks", type=int, default=MAX_TICKS)
    args = parser.parse_args()

    world = build_world()
    if args.mock:
        brain = MockBrain(world)
    else:
        from src.client import CerebrasClient

        brain = RobotBrain(CerebrasClient())

    loop = ReactiveLoop(world, brain)
    loop.set_instruction(INSTRUCTION)

    out_dir = Path(PROJECT_ROOT) / "runs"
    out_dir.mkdir(exist_ok=True)
    print(f"Instruction: {INSTRUCTION}\n{'-' * 72}")

    for i in range(args.ticks):
        if i == PERTURB_AT and world.gripper.holding != "cracked_cup":
            world.get("cracked_cup").x = 520
            world.get("cracked_cup").y = 130
            print(f"  >> PERTURBATION: cracked cup dragged to Zone C (520,130)")

        result = loop.tick()
        d = result.decision
        world.render().save(out_dir / f"frame_{i:02d}.png")
        print(
            f"t{result.tick:02d} | {d.skill:8s} {d.target:12s} zone={d.target_zone} "
            f"| {result.status:7s} | {d.latency_ms:6.0f}ms | {d.reasoning}"
        )
        if d.skill == "done":
            print("  >> task reported complete")
            break

    print(f"{'-' * 72}\nFrames saved to {out_dir}")


if __name__ == "__main__":
    main()
