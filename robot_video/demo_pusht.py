"""Gemma 4 watches PushT — a live reasoning demo on LeRobot's PushT simulation.

Every step of a PushT episode, Gemma 4 (via Cerebras) analyzes the camera frame
and describes what it sees and what action it would take. Meanwhile the
simulation runs with a simple heuristic policy so you can watch the T-block
move while Cerebras keeps up in real time (~4 fps).

Usage:
    uv run python -m robot_video.demo_pusht                    # live Gemma 4
    uv run python -m robot_video.demo_pusht --steps 10          # 10 frames only
    uv run python -m robot_video.demo_pusht --mock              # no API calls
    uv run python -m robot_video.demo_pusht --throttle          # GPU stand-in
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import gym_pusht.envs  # noqa: F401 — registers env
import gymnasium as gym
import numpy as np
from PIL import Image

from src.client import CerebrasClient, InferenceResult
from src.config import CEREBRAS_API_KEY
from src.sim.compare import ThrottledClient

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

VISION_PROMPT = (
    "You are a robot watching a PushT simulation. The goal is to push the "
    "T-shaped block into the green target area.\n\n"
    "Describe:\n"
    "1. Where is the T-shaped block right now?\n"
    "2. Where is the green target area?\n"
    "3. What direction should the pusher move to push the T toward the target?\n"
    "4. Is the T-block inside the target area yet?\n\n"
    "Be concise — one sentence per answer."
)

# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def _render_to_uri(frame: np.ndarray, quality: int = 90) -> str:
    """Convert a (H,W,3) numpy frame to a JPEG data URI for Gemma 4."""
    pil_img = Image.fromarray(frame)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Expert-like heuristic policy for PushT
# ---------------------------------------------------------------------------


def heuristic_action(obs: np.ndarray, step: int) -> np.ndarray:
    """Simple PushT policy: spiral-ish push to move the block.

    The actual policy doesn't matter much — the point is the env keeps stepping
    while Gemma 4 watches and reasons about each frame.
    """
    # Alternate between pushing in different directions
    phase = (step // 15) % 4
    if phase == 0:
        return np.array([0.8, 0.2], dtype=np.float32)
    elif phase == 1:
        return np.array([-0.3, 0.9], dtype=np.float32)
    elif phase == 2:
        return np.array([-0.7, -0.4], dtype=np.float32)
    else:
        return np.array([0.4, -0.6], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 watches LeRobot PushT simulation"
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to run")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Offline mode — no API calls, just the sim",
    )
    parser.add_argument(
        "--throttle",
        action="store_true",
        help="Wrap GPU in simulated latency for comparison",
    )
    parser.add_argument(
        "--gif", action="store_true", help="Save frames as GIF to runs/"
    )
    args = parser.parse_args()

    # API key check
    if not args.mock and not CEREBRAS_API_KEY:
        print("❌  CEREBRAS_API_KEY not set. Run with --mock for offline mode.")
        print("   Or add it to your .env file.")
        return

    # ---- Banner ----
    if args.mock:
        print("=" * 60)
        print("  🤖  Gemma 4 watches PushT [MOCK MODE — no API calls]")
        print("=" * 60)
    elif args.throttle:
        print("=" * 60)
        print("  🤖  Gemma 4 watches PushT [THROTTLE MODE — GPU stand-in]")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  🤖  Gemma 4 watches PushT [LIVE — Cerebras Inference]")
        print("=" * 60)
    print()

    # ---- Clients ----
    fast_client = CerebrasClient()
    if args.throttle:
        gpu_client = ThrottledClient(CerebrasClient(), extra_latency_s=1.7)
    else:
        gpu_client = None

    # ---- Build env ----
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")

    # ---- Run ----
    obs, info = env.reset()
    frames = []
    timings_fast: list[float] = []
    timings_gpu: list[float] = []

    print(f" Step | Cerebras   {'| GPU' if args.throttle else ' '}     | Observation")
    print("-" * 60)

    for i in range(args.steps):
        # Render current state (680x680)
        render_frame = env.render()
        uri = _render_to_uri(render_frame)
        if args.gif:
            frames.append(render_frame)

        # --- Cerebras reasoning ---
        if not args.mock:
            t0 = time.perf_counter()
            try:
                result = fast_client.image_chat(
                    VISION_PROMPT,
                    uri,
                    temperature=0.1,
                    max_tokens=200,
                )
                fast_ms = (time.perf_counter() - t0) * 1000
                cerebras_text = result.content.strip().replace("\n", " | ")
            except Exception as e:
                fast_ms = 0.0
                cerebras_text = f"[error: {e}]"
            timings_fast.append(fast_ms)

            # --- GPU (throttled) ---
            if gpu_client:
                t1 = time.perf_counter()
                try:
                    gpu_result = gpu_client.image_chat(
                        VISION_PROMPT,
                        uri,
                        temperature=0.1,
                        max_tokens=200,
                    )
                    gpu_ms = (time.perf_counter() - t1) * 1000
                except Exception as e:
                    gpu_ms = 0.0
                timings_gpu.append(gpu_ms)
            else:
                gpu_ms = 0.0

            # Print row
            cerebras_str = f"{fast_ms:7.0f}ms"
            if args.throttle:
                gpu_str = f" | {gpu_ms:7.0f}ms"
            else:
                gpu_str = "  "
            obs_short = cerebras_text[:55]
            print(f" {i:3d}/{(i):3d} | {cerebras_str}{gpu_str}  | {obs_short}")
        else:
            # Mock mode — just print step info
            print(f" {i:3d}/{(i):3d} |  — mock —   | step {i}")

        # Step the environment with heuristic
        action = heuristic_action(obs, i)
        obs, reward, term, trunc, _ = env.step(action)

        if term or trunc:
            print(f"\n{'=' * 60}")
            print(f"  ✅ Episode complete at step {i+1}!")
            break

    # ---- Summary ----
    env.close()
    print()
    print("=" * 60)
    print("  📊  PERFORMANCE SUMMARY")
    print("=" * 60)

    if timings_fast:
        avg_fast = sum(timings_fast) / len(timings_fast)
        print(f"\n  Cerebras avg:  {avg_fast:.0f}ms per frame  ({1000/avg_fast:.1f} fps)")

    if timings_gpu:
        avg_gpu = sum(timings_gpu) / len(timings_gpu)
        total_fast = sum(timings_fast)
        total_gpu = sum(timings_gpu)
        speedup = total_gpu / total_fast if total_fast > 0 else 0
        print(f"  GPU (simulated) avg: {avg_gpu:.0f}ms per frame  ({1000/avg_gpu:.1f} fps)")
        print(f"\n  ⚡ Speedup: {speedup:.1f}x")
        print(f"  Total: Cerebras {total_fast/1000:.2f}s | GPU {total_gpu/1000:.2f}s")

    # ---- Save GIF ----
    if args.gif and frames:
        out_dir = Path(__file__).resolve().parent.parent / "runs"
        out_dir.mkdir(exist_ok=True)
        gif_path = out_dir / "pusht_demo.gif"
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,
            loop=0,
        )
        size_kb = gif_path.stat().st_size / 1024
        print(f"\n  🎬  GIF saved: {gif_path} ({size_kb:.0f} KB, {len(frames)} frames)")

    print("\nDone.")


if __name__ == "__main__":
    main()
