"""Run the multi-agent pipeline (Vision -> Action -> Safety) on LeRobot video frames.

Usage:
    uv run python -m robot_video.run_pipeline --dataset lerobot/pusht --episode 0 --frames 5
    uv run python -m robot_video.run_pipeline --dataset lerobot/pusht --task "Pick up the red block"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from robot_video.frame_loader import LeRobotFrameSource, VideoFrame
from src.agents import SafetyAgent, VisionAgent
from src.client import CerebrasClient, InferenceResult
from src.config import CEREBRAS_API_KEY, GEMMA_MODEL, REASONING_EFFORT

# ---------------------------------------------------------------------------
# Structured output schemas (Cerebras-compatible json_schema format)
# ---------------------------------------------------------------------------

ACTION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_plan",
        "schema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "move_to",
                                    "pick_up",
                                    "place",
                                    "push",
                                    "pause",
                                    "navigate_home",
                                ],
                            },
                            "target": {"type": "string"},
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "distance_m": {"type": "number"},
                                    "direction": {"type": "string"},
                                    "force_N": {"type": "number"},
                                    "height_m": {"type": "number"},
                                },
                                "additionalProperties": True,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["action", "target", "reason"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["actions", "summary"],
            "additionalProperties": False,
        },
    },
}

SAFETY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "safety_review",
        "schema": {
            "type": "object",
            "properties": {
                "safe": {"type": "boolean"},
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "recommendation": {"type": "string"},
                "severity": {"type": "number"},
            },
            "required": [
                "safe",
                "risk_level",
                "issues",
                "recommendation",
                "severity",
            ],
            "additionalProperties": False,
        },
    },
}

ACTION_SYSTEM_PROMPT = """You are a robot action planner. Given a scene description, you output
a sequence of robot commands to accomplish a task.

Think step by step, then return a JSON object with:
- "actions": list of action objects, each with "action", "target", "parameters", "reason"
- "summary": one-line summary of the plan

Valid actions: move_to, pick_up, place, push, pause, navigate_home.

Assume the robot has a robotic arm with gripper, wheeled base, and basic sensors."""

SAFETY_SYSTEM_PROMPT = """You are a robot safety monitor. You review scene descriptions and proposed
action plans to identify risks.

Return a JSON object with:
- "safe": boolean
- "risk_level": "low" | "medium" | "high"
- "issues": list of specific safety concerns
- "recommendation": what to do instead (if unsafe)
- "severity": 0-1 score

Be conservative — flag anything ambiguous as a risk."""


# ---------------------------------------------------------------------------
# Data container for per-frame pipeline results
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """Holds the output of all three agents for a single frame."""

    frame_index: int
    episode_index: int
    ground_truth_action: list[float]
    vision_content: str
    action_content: str
    safety_content: str
    vision_ms: float
    action_ms: float
    safety_ms: float
    total_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_action(vals: list[float], max_items: int = 6) -> str:
    """Format an action vector for display, truncating if too long."""
    shown = [f"{v:.3f}" for v in vals[:max_items]]
    if len(vals) > max_items:
        shown.append("...")
    return "[" + ", ".join(shown) + "]"


def _parse_json_or_fallback(text: str) -> dict[str, Any]:
    """Try to parse JSON from agent output, with fallback extraction."""
    text = text.strip()
    # Attempt direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block between triple backticks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass
    # Try to find JSON block between curly braces
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Vision -> Action -> Safety pipeline on LeRobot frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dataset lerobot/pusht --episode 0 --frames 5\n"
            "  %(prog)s --dataset lerobot/pusht --task \"Pick up the red block\"\n"
            "  %(prog)s --dataset lerobot/pusht --frames 10 --no-safety\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="lerobot/pusht",
        help="LeRobot dataset repo ID (default: lerobot/pusht)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to process (default: 0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to process (default: 5)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Analyze the scene and identify any objects that can be manipulated.",
        help="Task description for the action planner (default: scene analysis)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera key to use (default: first available)",
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Skip the safety agent step",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Check API key
    # ------------------------------------------------------------------
    if not CEREBRAS_API_KEY:
        print("=" * 60)
        print("  ERROR: Cerebras API key not found.")
        print()
        print("  Set CEREBRAS_API_KEY in your .env file or environment:")
        print("    export CEREBRAS_API_KEY=csk-...")
        print()
        print("  Get a key at: https://cloud.cerebras.ai/")
        print("=" * 60)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Initialise clients and agents
    # ------------------------------------------------------------------
    client = CerebrasClient()
    vision_agent = VisionAgent(client)
    safety_agent = SafetyAgent(client)

    print(f"Model: {GEMMA_MODEL}")
    if REASONING_EFFORT:
        print(f"Reasoning: {REASONING_EFFORT}")
    print()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    source = LeRobotFrameSource(args.dataset, camera_key=args.camera)
    print(source.info())
    print()

    # Determine frame range for the requested episode
    ep_start, ep_end = source.episode_range(args.episode)
    n_available = ep_end - ep_start
    n_frames = min(args.frames, n_available)

    if n_frames == 0:
        print(f"Episode {args.episode} has no frames. Exiting.")
        sys.exit(0)

    print(
        f"Processing {n_frames} frame(s) from episode {args.episode} "
        f"(frames {ep_start}-{ep_start + n_frames - 1})"
    )
    print(f"Task: {args.task}")
    if args.no_safety:
        print("Safety agent: disabled")
    print()
    print("-" * 80)

    # ------------------------------------------------------------------
    # Pipeline loop
    # ------------------------------------------------------------------
    results: list[FrameResult] = []

    for offset in range(n_frames):
        global_idx = ep_start + offset

        # -- Load frame -------------------------------------------------
        frame: VideoFrame = source.get_frame(global_idx=global_idx)
        gt_action = frame.action

        total_start = time.perf_counter()
        error: str | None = None
        vision_content = ""
        action_content = ""
        safety_content = ""
        vision_ms = 0.0
        action_ms = 0.0
        safety_ms = 0.0

        # -- Step 1: Vision Agent --------------------------------------
        try:
            t0 = time.perf_counter()
            vision_result = vision_agent.analyze(
                frame.image_uri,
                prompt=(
                    "Analyze this robot workspace scene. "
                    "List all visible objects, describe the spatial layout, "
                    "and identify any graspable items the robot could interact with."
                ),
            )
            t1 = time.perf_counter()
            vision_ms = (t1 - t0) * 1000
            vision_content = vision_result.content
        except Exception as exc:
            error = f"Vision agent failed: {exc}"

        # -- Step 2: Action Agent (structured output) -------------------
        if error is None:
            try:
                action_prompt = (
                    f"Task: {args.task}\n\n"
                    f"Scene analysis:\n{vision_content}"
                )
                t0 = time.perf_counter()
                action_result = client.chat(
                    messages=[{"role": "user", "content": action_prompt}],
                    system_prompt=ACTION_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=2048,
                    response_format=ACTION_RESPONSE_FORMAT,
                )
                t1 = time.perf_counter()
                action_ms = (t1 - t0) * 1000
                action_content = action_result.content
            except Exception as exc:
                error = f"Action agent failed: {exc}"

        # -- Step 3: Safety Agent (structured output) -------------------
        if error is None and not args.no_safety:
            try:
                safety_prompt = (
                    f"Scene analysis:\n{vision_content}\n\n"
                    f"Proposed action plan:\n{action_content}"
                )
                t0 = time.perf_counter()
                safety_result = client.chat(
                    messages=[{"role": "user", "content": safety_prompt}],
                    system_prompt=SAFETY_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=1024,
                    response_format=SAFETY_RESPONSE_FORMAT,
                )
                t1 = time.perf_counter()
                safety_ms = (t1 - t0) * 1000
                safety_content = safety_result.content
            except Exception as exc:
                error = f"Safety agent failed: {exc}"

        total_ms = (time.perf_counter() - total_start) * 1000

        # -- Store result -----------------------------------------------
        result = FrameResult(
            frame_index=frame.frame_index,
            episode_index=frame.episode_index,
            ground_truth_action=gt_action,
            vision_content=vision_content,
            action_content=action_content,
            safety_content=safety_content,
            vision_ms=vision_ms,
            action_ms=action_ms,
            safety_ms=safety_ms,
            total_ms=total_ms,
            error=error,
        )
        results.append(result)

        # -- Print per-frame output -------------------------------------
        if error:
            print(f"  Frame {frame.frame_index:>4}  ERROR: {error}")
        else:
            # Parse action JSON for display
            action_parsed = _parse_json_or_fallback(action_content)
            action_summary = action_parsed.get("summary", "") if action_parsed else ""

            # Parse safety JSON for display
            safety_parsed = _parse_json_or_fallback(safety_content) if safety_content else {}
            safety_verdict = ""
            if safety_parsed:
                safe_flag = safety_parsed.get("safe", "?")
                risk = safety_parsed.get("risk_level", "?")
                safety_verdict = f"safe={safe_flag} risk={risk}"

            print(
                f"  Frame {frame.frame_index:>4}  "
                f"Vision: {vision_ms:>7.1f}ms  "
                f"Action: {action_ms:>7.1f}ms  "
                + (
                    f"Safety: {safety_ms:>7.1f}ms  "
                    if not args.no_safety
                    else ""
                )
                + f"Total: {total_ms:>7.1f}ms"
            )
            print(f"    GT action:   {_fmt_action(gt_action)}")
            print(f"    Plan summary: {action_summary or '(see JSON below)'}")
            if safety_verdict:
                print(f"    Safety:       {safety_verdict}")
            print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("-" * 80)
    print()
    print("=== Timing Summary ===\n")
    header = (
        f"  {'Frame':>6}  {'Vision(ms)':>10}  {'Action(ms)':>10}  "
        f"{'Safety(ms)':>10}  {'Total(ms)':>10}  {'Cumulative(ms)':>14}"
    )
    sep = "  " + "-" * (len(header) - 4)
    print(header)
    print(sep)

    cumulative = 0.0
    for r in results:
        cumulative += r.total_ms
        vision_str = f"{r.vision_ms:>8.1f}" if r.vision_ms else "     N/A"
        action_str = f"{r.action_ms:>8.1f}" if r.action_ms else "     N/A"
        safety_str = (
            f"{r.safety_ms:>8.1f}" if r.safety_ms and not args.no_safety else "     N/A"
        )
        total_str = f"{r.total_ms:>8.1f}"
        cum_str = f"{cumulative:>12.1f}"
        print(
            f"  {r.frame_index:>6}  {vision_str:>10}  {action_str:>10}  "
            f"{safety_str:>10}  {total_str:>10}  {cum_str:>14}"
        )

    print(sep)
    n_ok = sum(1 for r in results if r.error is None)
    total_time = sum(r.total_ms for r in results)
    avg_time = total_time / len(results) if results else 0.0
    print(
        f"\n  Total frames: {len(results)}  |  "
        f"Successful: {n_ok}  |  "
        f"Total time: {total_time:.1f}ms  |  "
        f"Avg: {avg_time:.1f}ms/frame"
    )
    print()

    # ------------------------------------------------------------------
    # Comparison summary (ground truth vs planned)
    # ------------------------------------------------------------------
    print("=== Action Comparison (first action dimension per frame) ===\n")
    print(f"  {'Frame':>6}  {'GT[0]':>8}  {'Planned[0]':>12}  {'Match?':>8}")
    print("  " + "-" * 42)
    for r in results:
        if r.error:
            continue
        gt_first = r.ground_truth_action[0] if r.ground_truth_action else 0.0
        # Try to extract the first action parameter from the planned output
        parsed = _parse_json_or_fallback(r.action_content)
        planned_first = 0.0
        match_str = "N/A"
        if parsed and "actions" in parsed and len(parsed["actions"]) > 0:
            first_action = parsed["actions"][0]
            params = first_action.get("parameters", {})
            # Heuristic: pick the first numeric parameter value
            if isinstance(params, dict):
                for val in params.values():
                    if isinstance(val, (int, float)):
                        planned_first = float(val)
                        break
            # Check if the action target gives a rough match signal
            match_str = "see below"
        print(
            f"  {r.frame_index:>6}  {gt_first:>8.3f}  {planned_first:>12.3f}  {match_str:>8}"
        )
    print()

    # Print a few representative planned action JSONs for inspection
    print("=== Sample Planned Actions ===\n")
    shown = 0
    for r in results:
        if r.error or shown >= 3:
            continue
        try:
            parsed = json.loads(r.action_content)
            print(f"  Frame {r.frame_index}:")
            print(f"    {json.dumps(parsed, indent=4)}")
            print()
            shown += 1
        except (json.JSONDecodeError, TypeError):
            print(f"  Frame {r.frame_index}: (raw, not valid JSON)")
            print(f"    {r.action_content[:200]}")
            print()
            shown += 1

    print("=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
