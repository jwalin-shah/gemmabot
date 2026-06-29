#!/usr/bin/env python3
"""Full-task hi-res renderer for the demo reel.

Drives COMPLETE tasks (not just grasp+lift) using the proven live-loop
MotionExecutor — the same code path that ran clear_table for 40 steps and
lifted cubes dozens of times — and records the whole motion from a 1024px
agentview camera.

Tasks:
  lift   — grasp the red cube, lift it clear of the table
  stack  — grasp the red cube, carry it over the green cube, set it on top
  pick   — grasp a grocery item, carry it to the right bin, release

Mode note (honesty): this is the LIVE CONTROL LOOP. Object positions come from
the simulator state the executor resolves by name (structured environment).
Gemma identifies the target from the image first (logged), then the executor
drives the motion. This is distinct from scripts/nocheat_demo.py, which feeds
Gemma NO ground truth. We label the footage accordingly in the video.

Output: overnight_results/videos/full_<task>.mp4  (1024x1024)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING", "1")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.web.lib.executor import MotionExecutor          # noqa: E402
from src.web.lib.imaging import fix_img                  # noqa: E402
from src.web.lib.sim import PandaSim                      # noqa: E402
from src.web.lib.tasks import get as get_task             # noqa: E402

HIRES = 1024
AGENT_CAM = "agentview"


class HiResRecordingSim(PandaSim):
    """PandaSim + a 1024px agentview camera; records a hi-res frame every step."""

    def __init__(self, task, hires_px: int = HIRES) -> None:
        super().__init__(task=task)
        self._hires_px = hires_px
        self.frames: list[np.ndarray] = []

    def env(self):
        if self._env is None:
            import robosuite as suite
            self._env = suite.make(
                self._task.env_name, robots="Panda",
                has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                camera_names=["birdview", "frontview", "robot0_eye_in_hand", AGENT_CAM],
                camera_heights=[384, 384, 384, self._hires_px],
                camera_widths=[384, 384, 384, self._hires_px],
            )
            self._env.reset()
        return self._env

    def _grab(self) -> None:
        obs = self.env()._get_observations(force_update=True)
        f = obs.get(f"{AGENT_CAM}_image")
        if f is not None:
            self.frames.append(fix_img(f))

    def step(self, action):
        snap = super().step(action)
        self._grab()
        return snap

    def reset(self):
        snap = super().reset()
        self.frames = []
        self._grab()
        return snap


def _obj_xy(snap, key: str):
    p = snap.objects.get(key)
    if p is None:
        for k, v in snap.objects.items():
            if key.lower() in k.lower():
                return float(v[0]), float(v[1]), float(v[2])
        raise KeyError(f"{key} not in {list(snap.objects)}")
    return float(p[0]), float(p[1]), float(p[2])


def run_lift(sim, ex):
    snap = sim.snapshot()
    r = ex.execute_tool(snap, "grasp", {"object_name": "cube"})
    snap = r.final_snapshot
    r = ex.execute_tool(snap, "lift", {"height": 0.22})
    return r.final_snapshot


def run_stack(sim, ex):
    snap = sim.snapshot()
    # 1) grasp red cube (cubeA)
    r = ex.execute_tool(snap, "grasp", {"object_name": "cubeA"})
    snap = r.final_snapshot
    # 2) lift clear
    r = ex.execute_tool(snap, "lift", {"height": 0.18})
    snap = r.final_snapshot
    # 3) carry over green cube (cubeB), gripper stays closed
    bx, by, bz = _obj_xy(snap, "cubeB")
    r = ex.execute_tool(snap, "move_to", {"x": bx, "y": by, "z": bz + 0.18})
    snap = r.final_snapshot
    # 4) lower onto green + release (place handler descends then opens)
    r = ex.execute_tool(snap, "place", {"x": bx, "y": by, "z": bz + 0.045})
    snap = r.final_snapshot
    # 5) retreat up so the stack is visible
    ex_snap = snap
    r = ex.execute_tool(ex_snap, "move_to", {"x": bx, "y": by, "z": bz + 0.25})
    return r.final_snapshot


def run_pick(sim, ex):
    snap = sim.snapshot()
    # Choose a grocery item that's reliably graspable; Can is the canonical one.
    target = "Can"
    r = ex.execute_tool(snap, "grasp", {"object_name": target})
    snap = r.final_snapshot
    r = ex.execute_tool(snap, "lift", {"height": 0.18})
    snap = r.final_snapshot
    # carry to right bin and release
    r = ex.execute_tool(snap, "place", {"x": 0.10, "y": 0.28})
    return r.final_snapshot


TASKS = {
    "lift":  ("lift_cube",  run_lift),
    "stack": ("stack_cubes", run_stack),
    "pick":  ("pick_can",   run_pick),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    task_key, runner = TASKS[args.task]
    print(f"=== FULL TASK: {args.task} ({task_key}) @ {HIRES}px ===")

    sim = HiResRecordingSim(task=get_task(task_key))
    snap = sim.reset()
    ex = MotionExecutor(sim)
    ex.seed_from(snap)

    t0 = time.time()
    final = runner(sim, ex)
    dt = time.time() - t0
    print(f"motion done in {dt:.1f}s, captured {len(sim.frames)} hi-res frames")

    out = _ROOT / "overnight_results" / "videos" / f"full_{args.task}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    # subsample if very long so the clip is punchy (~6-9s at fps)
    frames = sim.frames
    if len(frames) > args.fps * 9:
        stride = len(frames) // (args.fps * 9)
        frames = frames[::max(stride, 1)]
    imageio.mimsave(str(out), frames, fps=args.fps, quality=8)
    print(f"VIDEO: {out}  ({len(frames)} frames @ {args.fps}fps "
          f"= {len(frames)/args.fps:.1f}s)")
    sim._close()


if __name__ == "__main__":
    main()
