#!/usr/bin/env python3
"""Demo entry point for the Cerebras × Gemma 4 robotics multi-agent hackathon."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.client import CerebrasClient
from src.config import DEFAULT_IMAGE, PROJECT_ROOT
from src.orchestrator import AgentOrchestrator
from src.robot_controller import RobotController


def _fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GemmaBot — multi-agent robotics demo powered by Cerebras × Gemma 4 31B",
    )
    parser.add_argument(
        "--image", "-i",
        default=DEFAULT_IMAGE,
        help="Path to robot camera image (PNG/JPEG)",
    )
    parser.add_argument(
        "--task", "-t",
        default="Identify objects in the scene, pick up anything graspable, and report findings.",
        help="Task description for the robot",
    )
    parser.add_argument(
        "--robot-name",
        default="GemmaBot-1",
        help="Name for the robot controller",
    )
    args = parser.parse_args()

    # ---- Banner ----
    print("=" * 62)
    print("  🤖  GemmaBot — Multi-Agent Robotics Demo")
    print("  Powered by Gemma 4 31B on Cerebras Inference")
    print("=" * 62)
    print()

    # ---- Init ----
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌  Image not found: {image_path}")
        print(f"    Place a workspace image at {PROJECT_ROOT / "examples" / "images"}")
        sys.exit(1)

    print(f"📷  Camera image: {image_path.name}")
    print(f"🎯  Task: {args.task}")
    print(f"🤖  Robot: {args.robot_name}")
    print()

    client = CerebrasClient()
    robot = RobotController(name=args.robot_name)
    orchestrator = AgentOrchestrator(client, robot)

    # ---- Run pipeline ----
    print("🚀  Launching multi-agent pipeline (Vision → Action → Safety → Execute)...")
    print("-" * 62)

    result = orchestrator.run(str(image_path), task=args.task)

    # ---- Results ----
    print()
    print("📋  RESULTS")
    print("-" * 62)

    print(f"\n🔍  VISION ANALYSIS  [{_fmt_time(result.pipeline.get('vision_agent', 0))}]")
    print(f"     {result.scene_analysis[:400]}...")

    print(f"\n🗺️   ACTION PLAN  [{_fmt_time(result.pipeline.get('action_agent', 0))}]")
    print(f"     {result.action_plan[:400]}...")

    print(f"\n🛡️   SAFETY REVIEW  [{_fmt_time(result.pipeline.get('safety_agent', 0))}]")
    print(f"     {result.safety_review[:300]}...")

    print(f"\n⚙️   EXECUTION  [{_fmt_time(result.pipeline.get('execute', 0))}]")
    for act in result.executed_actions:
        icon = {"executed": "✅", "skipped": "⏭️", "failed": "❌"}.get(act.status, "❓")
        print(f"     {icon}  {act.action}({act.target}) — {act.message}")

    # ---- Performance ----
    print()
    print("⏱️   PERFORMANCE")
    print("-" * 62)
    print(f"     Total pipeline:    {_fmt_time(result.total_time_s)}")

    # Try to extract Cerebras time_info
    if hasattr(client, "_client") and hasattr(client._client, "time_info"):
        pass  # time_info is per-request, not stored on client

    print()
    print("=" * 62)


if __name__ == "__main__":
    main()