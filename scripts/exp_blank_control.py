"""Experiment: Blank Image Control.

Show Gemma a completely blank workspace image with no objects.
This establishes the baseline hallucination rate.

Usage:
    python scripts/exp_blank_control.py --runs 50 --output overnight_results/blank/r5_blank.jsonl --show-grid --temperature 0.0
    python scripts/exp_blank_control.py --runs 50 --output overnight_results/blank/r5_blank_no_grid.jsonl --no-grid --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient

WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2

IDENTIFY_SCHEMA = {
    "name": "identify_objects",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "observed_objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "color": {"type": "string"},
                        "zone": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]},
                        "shape": {"type": "string"},
                    },
                    "required": ["description", "color", "zone", "shape"],
                    "additionalProperties": False,
                },
            },
            "total_objects_visible": {"type": "integer"},
            "gripper_status": {"type": "string"},
        },
        "required": ["observed_objects", "total_objects_visible", "gripper_status"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system. You see a camera image showing a tabletop workspace with a Zone A-F grid overlaid.

Zones:
  A (top-left)   B (top-center)   C (top-right)
  D (bottom-left) E (bottom-center) F (bottom-right)

Identify every object you can see. For each, describe its color, shape, and which zone it occupies.
Be specific. Only describe what you actually see."""


def render_blank_image(show_grid: bool = True) -> str:
    """Render a completely blank workspace image (no objects, no gripper)."""
    from PIL import Image, ImageDraw
    import base64
    import io

    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    if show_grid:
        for c in range(1, 3):
            d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=(200, 200, 210), width=1)
        d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)
        for i, lab in enumerate(ZONE_LABELS):
            r_idx, c_idx = divmod(i, 3)
            d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)), f"Zone {lab}", fill=(170, 170, 180))

    # No objects, no gripper — intentionally blank

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-grid", action="store_true", default=True)
    parser.add_argument("--no-grid", action="store_true", dest="no_grid")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="blank_control",
                        help="Label for this variation")
    args = parser.parse_args()

    show_grid = not args.no_grid
    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results = []
    errors = 0
    start_time = time.time()

    # Determine the sub-variation label
    variation = f"{args.variation}_{'grid' if show_grid else 'no_grid'}"

    for run_num in range(1, args.runs + 1):
        image_b64 = render_blank_image(show_grid=show_grid)

        prompt = (
            "Look at this camera image of a tabletop workspace with a Zone A-F grid. "
            "Identify every object you can see by its color, shape, and which zone it occupies. "
            "Be specific about what zone each object is in."
        )

        t0 = time.perf_counter()
        try:
            result = client.image_chat(
                prompt=prompt,
                image_b64=image_b64,
                system_prompt=SYSTEM_PROMPT,
                temperature=args.temperature,
                max_tokens=600,
                response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = json.loads(result.content)

            observed = parsed.get("observed_objects", [])
            reported_count = parsed.get("total_objects_visible", 0)
            gripper_status = parsed.get("gripper_status", "")

            # On a blank image, EVERY reported object is a hallucination
            n_hallucinated = len(observed)

            entry = {
                "run": run_num,
                "experiment": "blank_control",
                "variation": variation,
                "show_grid": show_grid,
                "n_objects_reported": reported_count,
                "n_hallucinated": n_hallucinated,
                "hallucinated_objects": [
                    {
                        "description": o.get("description", ""),
                        "color": o.get("color", ""),
                        "zone": o.get("zone", ""),
                        "shape": o.get("shape", ""),
                    }
                    for o in observed
                ],
                "gripper_status": gripper_status,
                "latency_ms": round(latency_ms, 1),
                "temperature": args.temperature,
                "success": n_hallucinated == 0,
                "error": None,
                "prompt_sent": prompt,
                "raw_response": result.content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)

            if run_num % 5 == 0 or run_num == 1:
                print(f"  [{run_num:>4d}/{args.runs}] hallucinated: {n_hallucinated} | "
                      f"lat: {latency_ms:.0f}ms | objects: {reported_count}")

        except Exception as e:
            errors += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            entry = {
                "run": run_num,
                "experiment": "blank_control",
                "variation": variation,
                "show_grid": show_grid,
                "n_objects_reported": 0,
                "n_hallucinated": 0,
                "hallucinated_objects": [],
                "gripper_status": "",
                "error": str(e)[:200],
                "latency_ms": round(latency_ms, 1),
                "success": False,
                "prompt_sent": prompt,
                "raw_response": "",
                "temperature": args.temperature,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)
            print(f"  [{run_num:>4d}] ERROR: {e}")
            if errors >= 20:
                print(f"  Aborting after {errors} errors")
                break

        if output_path:
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start_time

    if not results:
        result_summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        n_halls = [r["n_hallucinated"] for r in results if "n_hallucinated" in r]
        n_reported = [r["n_objects_reported"] for r in results if "n_objects_reported" in r]
        latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)

        # Collect all hallucinated object descriptions to check consistency
        hall_descriptions = {}
        for r in results:
            for h in r.get("hallucinated_objects", []):
                key = f"{h.get('color','?')}_{h.get('shape','?')}_{h.get('zone','?')}"
                hall_descriptions.setdefault(key, {"seen": 0, "color": h.get("color",""), "shape": h.get("shape",""), "zone": h.get("zone",""), "description": h.get("description","")})
                hall_descriptions[key]["seen"] += 1

        # Most common hallucinations
        sorted_halls = sorted(hall_descriptions.values(), key=lambda x: -x["seen"])

        result_summary = {
            "runs": len(results),
            "experiment": "blank_control",
            "variation": variation,
            "show_grid": show_grid,
            "mean_hallucinations": round(sum(n_halls) / len(n_halls), 3) if n_halls else 0,
            "median_hallucinations": sorted(n_halls)[len(n_halls) // 2] if n_halls else 0,
            "max_hallucinations": max(n_halls) if n_halls else 0,
            "zero_hallucination_runs": sum(1 for h in n_halls if h == 0) if n_halls else 0,
            "zero_hallucination_rate": round(sum(1 for h in n_halls if h == 0) / len(n_halls), 3) if n_halls else 0,
            "mean_reported_count": round(sum(n_reported) / len(n_reported), 1) if n_reported else 0,
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "unique_hallucination_types": len(hall_descriptions),
            "most_common_hallucinations": sorted_halls[:10],
            "gripper_statuses": {},
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(results) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "parameters": {
                "temperature": args.temperature,
                "show_grid": show_grid,
            },
        }

    print(f"\n{'='*60}")
    print(f"BLANK IMAGE CONTROL ({variation})")
    print(f"{'='*60}")
    print(f"Runs: {result_summary['runs']}")
    print(f"Mean hallucinations per image: {result_summary['mean_hallucinations']}")
    print(f"Median hallucinations: {result_summary['median_hallucinations']}")
    print(f"Max hallucinations (single image): {result_summary['max_hallucinations']}")
    print(f"Zero-hallucination rate: {result_summary['zero_hallucination_rate']:.1%}")
    print(f"Mean objects reported (total_objects_visible): {result_summary['mean_reported_count']}")
    print(f"Unique hallucination 'types' (color+shape+zone): {result_summary['unique_hallucination_types']}")
    print(f"Elapsed: {result_summary['elapsed_s']}s")

    if sorted_halls:
        print(f"\nMost common hallucinated objects:")
        for h in sorted_halls[:10]:
            print(f"  [{h['seen']:>3d}x] {h['color']:>8s} {h['shape']:>7s} in Zone {h['zone']}  "
                  f"({h['description'][:50]})")

    print(f"RESULT:{json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
