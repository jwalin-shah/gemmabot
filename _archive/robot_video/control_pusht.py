"""Gemma 4 *controls* the PushT simulation — close-the-loop demo.

Gemma 4 sees the PushT camera frame, decides which direction to push,
and the simulation moves. Every step is a Cerebras call. We measure
success rate, steps-to-goal, and timing.

Usage:
    uv run python -m robot_video.control_pusht                  # 1 episode, live Gemma 4
    uv run python -m robot_video.control_pusht --episodes 3      # Run multiple episodes
    uv run python -m robot_video.control_pusht --baseline        # Random vs heuristic baseline
    uv run python -m robot_video.control_pusht --mock             # No API calls
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import gym_pusht.envs  # noqa: F401
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.client import CerebrasClient
from src.config import CEREBRAS_API_KEY

# ---------------------------------------------------------------------------
# 8 compass directions mapped to continuous PushT actions
# PushT action space: Box(0.0, 512.0, (2,)) — absolute XY position of the pusher
# We output directional nudges relative to current position
# ---------------------------------------------------------------------------

DIRECTIONS: dict[str, np.ndarray] = {
    "up":        np.array([0.0, -80.0], dtype=np.float32),
    "down":      np.array([0.0, 80.0], dtype=np.float32),
    "left":      np.array([-80.0, 0.0], dtype=np.float32),
    "right":     np.array([80.0, 0.0], dtype=np.float32),
    "up_left":   np.array([-56.0, -56.0], dtype=np.float32),
    "up_right":  np.array([56.0, -56.0], dtype=np.float32),
    "down_left": np.array([-56.0, 56.0], dtype=np.float32),
    "down_right":np.array([56.0, 56.0], dtype=np.float32),
    "wait":      np.array([0.0, 0.0], dtype=np.float32),
}

DIRECTION_NAMES = list(DIRECTIONS.keys())

# Structured output schema for Gemma 4
ACTION_SCHEMA: dict = {
    "name": "push_action",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": DIRECTION_NAMES,
                "description": "Which direction to push the T-block toward the target",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence: where is the T-block, where is the target",
            },
            "goal_reached": {
                "type": "boolean",
                "description": "Is the T-block already inside the green target area?",
            },
        },
        "required": ["direction", "reasoning", "goal_reached"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are a robot controlling a pusher in the PushT simulation.\n"
    "Your goal: push the T-shaped block into the green target zone.\n\n"
    "Look at the camera image and output:\n"
    '1. "direction": one of ' + str(DIRECTION_NAMES) + ' — which way to push\n'
    "2. \"reasoning\": where the T-block is, where the target is, what you'll do\n"
    '3. "goal_reached": true ONLY if the block is clearly inside the green area\n\n'
    "Rules:\n"
    "- The T-block is a dark T-shaped object on a light background\n"
    "- The target is a green highlighted area\n"
    "- Push the T-block toward the target by moving the pusher disk\n"
    "- If goal_reached=true, output 'wait' as direction\n"
    "- Be decisive. Keep pushing in one direction for several steps before changing.\n"
    "- If you see the T-block is in the green zone, set goal_reached=true."
)


def _render_to_uri(frame: np.ndarray, quality: int = 85) -> str:
    pil_img = Image.fromarray(frame)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _overlay_text(frame: np.ndarray, text: str, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Add timestamp text overlay to a frame."""
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil)
    d.text((8, 8), text, fill=color)
    return np.array(pil)


def heuristic_policy(obs: np.ndarray, step: int) -> np.ndarray:
    """Simple spiral pusher for baseline comparison."""
    phase = (step // 12) % 4
    if phase == 0:
        return np.array([256 + 30, 256 + 0], dtype=np.float32)
    elif phase == 1:
        return np.array([256 + 0, 256 + 30], dtype=np.float32)
    elif phase == 2:
        return np.array([256 - 30, 256 + 0], dtype=np.float32)
    else:
        return np.array([256 + 0, 256 - 30], dtype=np.float32)


def random_policy() -> np.ndarray:
    return np.random.uniform(0, 512, size=(2,)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="run heuristic+random baseline instead")
    parser.add_argument("--gif", action="store_true", help="save GIF")
    args = parser.parse_args()

    if not args.mock and not CEREBRAS_API_KEY and not args.baseline:
        print("❌  CEREBRAS_API_KEY not set. Use --mock for offline test.")
        return

    if args.baseline:
        _run_baseline(args.episodes, args.max_steps, args.gif)
        return

    client = CerebrasClient() if not args.mock else None

    print("=" * 62)
    print("  🎮  Gemma 4 Controls PushT — Closed-Loop Demo")
    print("  " + ("MOCK MODE" if args.mock else "Cerebras Inference"))
    print("=" * 62)

    total_steps = 0
    total_success = 0
    all_timings: list[float] = []
    all_gifs: list[list[np.ndarray]] = []

    for ep in range(args.episodes):
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
        obs, info = env.reset()
        frames: list[np.ndarray] = []
        ep_timings: list[float] = []
        last_dir = "wait"

        print(f"\n--- Episode {ep + 1} ---")
        print(f"{'Step':>5} {'Cerebras':>10} {'Direction':>12} {'Reward':>8} {'Goal?':>6}")
        print("-" * 55)

        for step in range(args.max_steps):
            render = env.render()
            gif_frame = render.copy()

            if args.mock:
                direction = np.random.choice(DIRECTION_NAMES)
                cerebras_ms = 0.0
                reasoning = "mock"
                goal = False
            else:
                uri = _render_to_uri(render)
                t0 = time.perf_counter()
                try:
                    result = client.image_chat(
                        SYSTEM_PROMPT if step == 0 else f"You are at step {step}. Last action was: {last_dir}. LOOK at the image and decide the NEXT action. Where is the block now? Where is the pusher? What should you do NEXT?",
                        uri,
                        temperature=0.1,
                        max_tokens=200,
                        response_format={"type": "json_schema", "json_schema": ACTION_SCHEMA},
                    )
                    cerebras_ms = (time.perf_counter() - t0) * 1000
                    data = json.loads(result.content)
                    direction = data.get("direction", "wait")
                    reasoning = data.get("reasoning", "")[:40]
                    goal = data.get("goal_reached", False)
                except Exception as e:
                    cerebras_ms = (time.perf_counter() - t0) * 1000
                    direction = "wait"
                    reasoning = f"[err: {e}]"
                    goal = False

            all_timings.append(cerebras_ms)
            ep_timings.append(cerebras_ms)
            last_dir = direction

            # Convert direction to action
            if direction in DIRECTIONS:
                # Get current position as last action or center
                if step == 0:
                    current_pos = np.array([256.0, 256.0], dtype=np.float32)
                else:
                    current_pos = last_raw_action if step > 0 else np.array([256.0, 256.0], dtype=np.float32)

                delta = DIRECTIONS[direction]
                action = np.clip(current_pos + delta, 0.0, 512.0).astype(np.float32)
            else:
                action = np.array([256.0, 256.0], dtype=np.float32)
            last_raw_action = action

            # Step env
            obs, reward, term, trunc, _ = env.step(action)

            # Overlay frame with timing
            if args.gif:
                info_text = f"t={cerebras_ms:.0f}ms dir={direction} rew={reward:.2f}"
                gif_frame = _overlay_text(gif_frame, info_text)
                frames.append(gif_frame)

            goal_str = "✅" if goal else "—"
            print(f"{step:>4}/{step:>3} {cerebras_ms:>8.0f}ms {direction:>12} {reward:>8.3f} {goal_str:>6}")

            if goal or term or trunc:
                total_success += 1
                print(f"  🎯 Episode complete! ({'goal reached' if goal else 'max steps'})")
                break

        env.close()
        total_steps += step + 1

        if args.gif and frames:
            all_gifs.append(frames)

        avg_ep = np.mean(ep_timings) if ep_timings else 0
        print(f"  Episode avg: {avg_ep:.0f}ms/frame | Steps: {step+1}")

    # Summary
    print()
    print("=" * 62)
    print("  📊  RESULTS")
    print("=" * 62)
    avg_all = np.mean(all_timings) if all_timings else 0
    print(f"  Avg Cerebras:    {avg_all:.0f}ms  ({1000/max(avg_all,1):.1f} fps)")
    print(f"  Total steps:     {total_steps}")
    print(f"  Episodes done:   {total_success}/{args.episodes}")

    # Save GIF
    if args.gif and all_gifs:
        gif_path = Path(__file__).resolve().parent.parent / "runs" / "control_pusht.gif"
        flat = [f for ep_frames in all_gifs for f in ep_frames]
        pil_frames = [Image.fromarray(f) for f in flat]
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=100, loop=0,
        )
        print(f"\n  🎬  GIF: {gif_path} ({len(flat)} frames)")


def _run_baseline(episodes: int, max_steps: int, save_gif: bool) -> None:
    """Run heuristic and random baselines for comparison."""
    for label, policy_fn in [("Heuristic", heuristic_policy), ("Random", random_policy)]:
        rewards = []
        steps_taken = []
        print(f"\n--- Baseline: {label} ---")
        for ep in range(episodes):
            env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
            obs, _ = env.reset()
            ep_reward = 0.0
            for s in range(max_steps):
                if label == "Heuristic":
                    action = policy_fn(obs, s)
                else:
                    action = policy_fn()
                obs, reward, term, trunc, _ = env.step(action)
                ep_reward += reward
                if term or trunc:
                    break
            env.close()
            rewards.append(ep_reward)
            steps_taken.append(s + 1)
            print(f"  Ep {ep+1}: reward={ep_reward:.2f}, steps={s+1}")
        print(f"  Avg reward: {np.mean(rewards):.2f} | Avg steps: {np.mean(steps_taken):.0f}")


if __name__ == "__main__":
    main()
