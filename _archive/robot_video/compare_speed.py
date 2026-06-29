"""Compare Cerebras vs GPU inference speed on LeRobot video frames.

Processes the same N frames from a LeRobot episode through both backends'
vision agents and prints a side-by-side timing comparison.

Usage:
    uv run python -m robot_video.compare_speed
    uv run python -m robot_video.compare_speed --dataset lerobot/pusht --episode 0 --frames 5 --step 1
    uv run python -m robot_video.compare_speed --throttle                           # GPU stand-in
"""

from __future__ import annotations

import argparse
import sys
import time

from src.client import CerebrasClient
from src.config import COMPARISON_API_KEY, COMPARISON_BASE_URL, COMPARISON_MODEL
from src.sim.compare import ComparisonClient, ThrottledClient
from robot_video.frame_loader import LeRobotFrameSource


VISION_PROMPT = (
    "You are a robot vision analyst. Describe the scene — what objects do you see, "
    "their approximate positions, and any hazards. Be concise."
)

TABLE_HEADER = " frame | Cerebras  | GPU       | Speedup"
TABLE_SEP = "-------+-----------+-----------+--------"

THROTTLE_LATENCY_S = 1.7


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:7.1f} ms"


def _fmt_speedup(slow_s: float, fast_s: float) -> str:
    if slow_s <= 0 or fast_s <= 0:
        return "   N/A  "
    return f"{slow_s / fast_s:6.1f}x"


def _build_gpu_client(throttle: bool) -> tuple[CerebrasClient | ComparisonClient | ThrottledClient, str]:
    """Create the GPU-side client and return it along with a display label."""
    if throttle:
        return ThrottledClient(CerebrasClient(), extra_latency_s=THROTTLE_LATENCY_S), "GPU"

    if not COMPARISON_API_KEY or not COMPARISON_BASE_URL or not COMPARISON_MODEL:
        print(
            "Error: GPU comparison not configured.\n"
            "Set COMPARISON_API_KEY, COMPARISON_BASE_URL, and COMPARISON_MODEL in your .env file,\n"
            "or use --throttle for a simulated GPU (no API key required).",
            file=sys.stderr,
        )
        sys.exit(1)

    return ComparisonClient(), "GPU"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Cerebras vs GPU inference speed on LeRobot video frames"
    )
    parser.add_argument(
        "--dataset",
        default="lerobot/pusht",
        help="LeRobot dataset identifier (default: lerobot/pusht)",
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index (default: 0)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to process per backend (default: 5)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step within the episode (default: 1, every frame)",
    )
    parser.add_argument(
        "--throttle",
        action="store_true",
        help="Use ThrottledClient as GPU stand-in (no GPU API key needed)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    try:
        source = LeRobotFrameSource(args.dataset)
    except Exception as exc:
        print(f"Error loading dataset '{args.dataset}': {exc}", file=sys.stderr)
        sys.exit(1)

    ep_start, ep_end = source.episode_range(args.episode)
    total_ep_frames = ep_end - ep_start

    # ------------------------------------------------------------------
    # Print header info
    # ------------------------------------------------------------------
    print(f"Cerebras vs GPU — Vision Agent on {args.dataset}")
    actual_frames = min(args.frames, (total_ep_frames + args.step - 1) // args.step)
    last_global_idx = ep_start + (actual_frames - 1) * args.step
    if last_global_idx >= ep_end:
        last_global_idx = ep_end - 1
    last_frame_idx = last_global_idx - ep_start
    print(
        f"Frames: episode {args.episode}, frames {0}-{last_frame_idx} (step {args.step})"
    )
    print()

    # ------------------------------------------------------------------
    # Create backends
    # ------------------------------------------------------------------
    cerebras_client = CerebrasClient()
    gpu_client, gpu_label = _build_gpu_client(args.throttle)

    # ------------------------------------------------------------------
    # Collect frames
    # ------------------------------------------------------------------
    frames = list(source.iter_episode(args.episode, step=args.step))
    frames = frames[: args.frames]

    if not frames:
        print(
            f"No frames found for episode {args.episode} with step {args.step}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run comparison
    # ------------------------------------------------------------------
    cerebras_latencies: list[float] = []
    gpu_latencies: list[float] = []

    print(TABLE_HEADER)
    print(TABLE_SEP)

    for i, frame in enumerate(frames):
        # --- Cerebras ---
        try:
            t0 = time.perf_counter()
            result = cerebras_client.image_chat(VISION_PROMPT, frame.image_uri)
            c_sec = time.perf_counter() - t0  # wall-clock
        except Exception as exc:
            print(f"  [Cerebras] error on frame {i}: {exc}", file=sys.stderr)
            c_sec = 0.0
        cerebras_latencies.append(c_sec)

        # --- GPU ---
        try:
            t0 = time.perf_counter()
            result = gpu_client.image_chat(VISION_PROMPT, frame.image_uri)
            g_sec = time.perf_counter() - t0  # wall-clock for fair comparison (includes throttle)
        except Exception as exc:
            print(f"  [GPU] error on frame {i}: {exc}", file=sys.stderr)
            g_sec = 0.0
        gpu_latencies.append(g_sec)

        row_frame = frame.frame_index if hasattr(frame, "frame_index") else i
        print(
            f"{row_frame:6d} | {_fmt_ms(c_sec)} | {_fmt_ms(g_sec)} | {_fmt_speedup(g_sec, c_sec)}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(TABLE_SEP)

    n = len(cerebras_latencies)
    avg_c = sum(cerebras_latencies) / n if n else 0.0
    avg_g = sum(gpu_latencies) / n if n else 0.0
    print(f"  avg  | {_fmt_ms(avg_c)} | {_fmt_ms(avg_g)} | {_fmt_speedup(avg_g, avg_c)}")

    total_c = sum(cerebras_latencies)
    total_g = sum(gpu_latencies)
    if total_c > 0:
        speedup = total_g / total_c
        print(f"Total: Cerebras {total_c:.2f}s | {gpu_label} {total_g:.2f}s | {speedup:.1f}x faster")
    else:
        print(f"Total: Cerebras {total_c:.2f}s | {gpu_label} {total_g:.2f}s")


if __name__ == "__main__":
    main()
