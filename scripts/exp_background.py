"""Experiment: Background Color Variation.

Structured output at temperature 0.0.
Cycles through 6 tabletop background colors to test whether Gemma 4's
spatial reasoning degrades when background color changes.

Usage:
    python scripts/exp_background.py --runs 200 \\
        --output overnight_results/background/r6_background.jsonl \\
        --variation background_change --no-grid --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient

WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2

ZONE_CENTERS = {}
for i, lab in enumerate(ZONE_LABELS):
    r, c_idx = divmod(i, 3)
    ZONE_CENTERS[lab] = (int(c_idx * cw + cw / 2), int(r * ch + ch / 2))

OBJECT_TEMPLATES = [
    {"id": "red_cup",    "color": (210, 60, 60),   "shape": "round",   "attr": ""},
    {"id": "blue_cup",   "color": (60, 90, 210),   "shape": "round",   "attr": ""},
    {"id": "green_cup",  "color": (60, 180, 60),   "shape": "round",   "attr": ""},
    {"id": "cracked_cup","color": (200, 175, 120), "shape": "round",   "attr": "cracked"},
    {"id": "yellow_cube","color": (220, 200, 40),  "shape": "square",  "attr": ""},
    {"id": "orange_star","color": (230, 140, 40),  "shape": "star",    "attr": ""},
    {"id": "pink_cup",   "color": (210, 120, 160), "shape": "round",   "attr": ""},
    {"id": "purple_cube","color": (140, 60, 180),  "shape": "square",  "attr": ""},
]

SCENE_LAYOUTS = [
    {"red_cup": "D", "blue_cup": "E", "cracked_cup": "B"},
    {"red_cup": "A", "blue_cup": "C", "green_cup": "E"},
    {"red_cup": "E", "cracked_cup": "C", "yellow_cube": "D"},
    {"blue_cup": "F", "green_cup": "A", "cracked_cup": "D"},
    {"red_cup": "D", "blue_cup": "B", "green_cup": "F", "cracked_cup": "C"},
    {"yellow_cube": "A", "green_cup": "B", "cracked_cup": "F"},
    {"blue_cup": "C", "cracked_cup": "E", "yellow_cube": "A", "green_cup": "D"},
    {"red_cup": "F", "blue_cup": "A", "green_cup": "C"},
    {"red_cup": "B", "cracked_cup": "D", "yellow_cube": "F"},
    {"red_cup": "D", "blue_cup": "F", "cracked_cup": "B", "green_cup": "A", "yellow_cube": "C"},
    {"red_cup": "E", "cracked_cup": "A", "blue_cup": "F"},
    {"green_cup": "C", "yellow_cube": "D", "red_cup": "A", "blue_cup": "B"},
    {"orange_star": "E", "pink_cup": "B", "purple_cube": "D"},
    {"red_cup": "C", "blue_cup": "E", "green_cup": "A", "yellow_cube": "F", "orange_star": "B"},
    {"pink_cup": "D", "purple_cube": "C", "cracked_cup": "A"},
]

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

# Background color definitions
BACKGROUND_COLORS = {
    "light_gray": (238, 238, 240),
    "white":      (255, 255, 255),
    "dark_gray":  (80, 80, 85),
    "wood":       (200, 170, 120),
    "black":      (30, 30, 32),
    "green_felt": (50, 120, 70),
}


def render_scene(objects: list[dict], show_grid: bool = True,
                 bg_color: tuple = (238, 238, 240),
                 gripper_override: dict | None = None) -> str:
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
    d = ImageDraw.Draw(img)

    if show_grid:
        grid_color = _grid_line_color(bg_color)
        for c in range(1, 3):
            d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=grid_color, width=1)
        d.line([(0, int(ch)), (WIDTH, int(ch))], fill=grid_color, width=1)
        text_color = _text_color(bg_color)
        for i, lab in enumerate(ZONE_LABELS):
            r_idx, c_idx = divmod(i, 3)
            d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)), f"Zone {lab}", fill=text_color)

    for obj in objects:
        x, y = obj["x"], obj["y"]
        r = obj.get("radius", 26)
        color = obj["color"]
        shape = obj.get("shape", "round")

        outline = _object_outline(bg_color)
        if shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=2)
        elif shape == "star":
            pts = []
            for k in range(10):
                angle = k * 36 - 90
                rad = r if k % 2 == 0 else r * 0.45
                pts.append((x + rad * __import__('math').cos(angle * __import__('math').pi / 180),
                            y + rad * __import__('math').sin(angle * __import__('math').pi / 180)))
            d.polygon(pts, fill=color, outline=outline)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=2)

        if obj.get("attr") == "cracked":
            d.line([(x - 11, y - 13), (x + 4, y), (x - 7, y + 13)], fill=(20, 20, 20), width=2)

    # Gripper
    gx, gy = WIDTH // 2, 20
    g_color = gripper_override.get("color", (45, 120, 205)) if gripper_override else (45, 120, 205)
    g_width = gripper_override.get("width", 4) if gripper_override else 4
    d.line([(gx - 16, gy), (gx + 16, gy)], fill=g_color, width=g_width)
    d.line([(gx, gy), (gx, gy - 24)], fill=g_color, width=g_width)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _grid_line_color(bg: tuple) -> tuple:
    """Pick a visible grid line color against the given background."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if luminance < 60:
        return (140, 140, 150)  # light lines on dark bg
    elif luminance > 200:
        return (180, 180, 190)  # slightly darker on bright bg
    else:
        return (200, 200, 210)  # default


def _text_color(bg: tuple) -> tuple:
    """Pick readable text color for zone labels."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if luminance < 80:
        return (200, 200, 210)
    else:
        return (140, 140, 155)


def _object_outline(bg: tuple) -> tuple:
    """Pick outline color that contrasts with the background."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if luminance < 60:
        return (180, 180, 180)  # lighter outline on dark bg
    else:
        return (35, 35, 35)     # dark outline on light bg


def place_objects(layout: dict) -> list[dict]:
    objects = []
    for oid, zone in layout.items():
        t = next(o for o in OBJECT_TEMPLATES if o["id"] == oid)
        zx, zy = ZONE_CENTERS[zone]
        objects.append({
            "id": oid,
            "x": zx + random.randint(-35, 35),
            "y": zy + random.randint(-25, 25),
            "color": t["color"],
            "radius": 26,
            "shape": t["shape"],
            "attr": t["attr"],
            "expected_zone": zone,
        })
    return objects


def evaluate(parsed: dict, ground_truth: list[dict]) -> dict:
    observed = parsed.get("observed_objects", [])
    reported_count = parsed.get("total_objects_visible", 0)
    zone_matches = 0
    zone_total = 0
    matched_truth = set()
    matched_obs = set()

    for oi, obs in enumerate(observed):
        obs_zone = obs.get("zone", "")
        obs_desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()

        best_dist = float("inf")
        best_gi = None
        for gi, gt in enumerate(ground_truth):
            if gi in matched_truth:
                continue
            gt_id = gt["id"].lower().replace("_", " ")
            gt_words = set(gt_id.split())
            obs_words = set(w for w in obs_desc.split() if len(w) > 2)
            overlap = len(gt_words & obs_words)
            color_map = {"red": (210,60,60), "blue": (60,90,210), "green": (60,180,60),
                         "yellow": (220,200,40), "orange": (230,140,40), "pink": (210,120,160),
                         "purple": (140,60,180), "tan": (200,175,120), "brown": (200,175,120),
                         "gray": (120,120,120), "grey": (120,120,120)}
            for cname, crgb in color_map.items():
                if cname in obs_desc and gt["color"] == crgb:
                    overlap += 1
            if overlap > 0:
                d = 1.0 / overlap
                if d < best_dist:
                    best_dist = d
                    best_gi = gi

        if best_gi is not None:
            matched_truth.add(best_gi)
            matched_obs.add(oi)
            zone_total += 1
            if obs_zone == ground_truth[best_gi]["expected_zone"]:
                zone_matches += 1

    hallucinations = len(observed) - len(matched_obs)
    misses = len(ground_truth) - len(matched_truth)

    return {
        "zone_accuracy": zone_matches / max(zone_total, 1),
        "zone_matches": zone_matches,
        "zone_total": zone_total,
        "hallucinations": hallucinations,
        "misses": misses,
        "count_error": abs(reported_count - len(ground_truth)),
        "reported_count": reported_count,
        "true_count": len(ground_truth),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Background color variation experiment for Gemma 4 spatial reasoning"
    )
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-grid", action="store_true", default=True)
    parser.add_argument("--no-grid", action="store_true", dest="no_grid")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="background_change",
                        help="Label for this variation (default: background_change)")
    parser.add_argument(
        "--background-color",
        type=str,
        default=None,
        choices=list(BACKGROUND_COLORS.keys()),
        help="If set, use a single background color for all runs. "
             "If omitted, cycle through all 6 backgrounds."
    )
    args = parser.parse_args()

    show_grid = not args.no_grid
    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine background sequence
    bg_names = list(BACKGROUND_COLORS.keys())
    if args.background_color:
        # Single background for all runs
        bg_sequence = [args.background_color] * args.runs
    else:
        # Cycle through all backgrounds
        bg_sequence = [bg_names[i % len(bg_names)] for i in range(args.runs)]

    client = CerebrasClient()
    results = []
    errors = 0
    start_time = time.time()

    n_scenes = len(SCENE_LAYOUTS)
    reps = max(1, args.runs // n_scenes)

    for si, layout in enumerate(SCENE_LAYOUTS):
        if si * reps >= args.runs:
            break
        for rep in range(reps):
            run_num = si * reps + rep + 1
            if run_num > args.runs:
                break

            bg_name = bg_sequence[run_num - 1]
            bg_rgb = BACKGROUND_COLORS[bg_name]

            objects = place_objects(layout)
            image_b64 = render_scene(
                objects,
                show_grid=show_grid,
                bg_color=bg_rgb,
            )

            prompt = (
                "Look at this camera image of a tabletop workspace with a Zone A-F grid. "
                "Identify every object you can see by its color, shape, and which zone it occupies. "
                "Be specific about what zone each object is in."
            )

            t0 = time.perf_counter()
            entry = None
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
                score = evaluate(parsed, objects)

                entry = {
                    "run": run_num,
                    "experiment": "background_change",
                    "variation": args.variation,
                    "scene": si,
                    "n_objects": len(objects),
                    "background": bg_name,
                    "background_rgb": str(bg_rgb),
                    "reported_count": score["reported_count"],
                    "count_error": score["count_error"],
                    "zone_accuracy": round(score["zone_accuracy"], 4),
                    "zone_matches": score["zone_matches"],
                    "zone_total": score["zone_total"],
                    "hallucinations": score["hallucinations"],
                    "misses": score["misses"],
                    "latency_ms": round(latency_ms, 1),
                    "temperature": args.temperature,
                    "show_grid": show_grid,
                    "success": score["zone_accuracy"] >= 0.5 and score["count_error"] <= 1,
                    "error": None,
                    "prompt_sent": prompt,
                    "raw_response": result.content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)

                if run_num % 25 == 0 or run_num == 1:
                    print(f"  [{run_num:>4d}/{args.runs}] {len(objects)} objects | "
                          f"bg: {bg_name:12s} | zone: {score['zone_accuracy']:.0%} | "
                          f"lat: {latency_ms:.0f}ms")

            except Exception as e:
                errors += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                entry = {
                    "run": run_num, "experiment": "background_change",
                    "variation": args.variation,
                    "scene": si, "n_objects": len(objects),
                    "background": bg_name,
                    "background_rgb": str(bg_rgb),
                    "error": str(e)[:200], "latency_ms": round(latency_ms, 1),
                    "success": False, "prompt_sent": prompt, "raw_response": "",
                    "temperature": args.temperature,
                    "show_grid": show_grid,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)
                if errors >= 20:
                    print(f"  [ERR] {errors} errors — aborting")
                    break

            if output_path and entry:
                with open(output_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        if errors >= 20:
            break

    elapsed = time.time() - start_time

    if not results:
        result_summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        successes = [r for r in results if r.get("success")]
        accs = [r["zone_accuracy"] for r in results if "zone_accuracy" in r]
        halls = [r["hallucinations"] for r in results if "hallucinations" in r]
        misses = [r["misses"] for r in results if "misses" in r]

        # Per-background statistics
        bg_stats = defaultdict(lambda: {"runs": 0, "accs": [], "halls": [], "misses": [],
                                         "zone_matches": 0, "zone_total": 0, "count_errors": []})
        for r in results:
            bg = r.get("background", "unknown")
            bg_stats[bg]["runs"] += 1
            if "zone_accuracy" in r:
                bg_stats[bg]["accs"].append(r["zone_accuracy"])
                bg_stats[bg]["halls"].append(r["hallucinations"])
                bg_stats[bg]["misses"].append(r["misses"])
                bg_stats[bg]["zone_matches"] += r.get("zone_matches", 0)
                bg_stats[bg]["zone_total"] += r.get("zone_total", 0)
                bg_stats[bg]["count_errors"].append(r.get("count_error", 0))

        per_background = {}
        for bg_name in bg_names:
            s = bg_stats[bg_name]
            n = len(s["accs"])
            per_background[bg_name] = {
                "runs": s["runs"],
                "mean_zone_accuracy": round(sum(s["accs"]) / n, 4) if n > 0 else 0,
                "mean_hallucinations": round(sum(s["halls"]) / n, 2) if n > 0 else 0,
                "mean_misses": round(sum(s["misses"]) / n, 2) if n > 0 else 0,
                "mean_count_error": round(sum(s["count_errors"]) / n, 2) if n > 0 else 0,
                "zone_match_rate": round(
                    s["zone_matches"] / max(s["zone_total"], 1), 4
                ),
            }

        latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)

        # Find worst background
        bg_accs = {bg: per_background[bg]["mean_zone_accuracy"] for bg in bg_names}
        worst_bg = min(bg_accs, key=bg_accs.get)
        best_bg = max(bg_accs, key=bg_accs.get)
        baseline = per_background["light_gray"]["mean_zone_accuracy"]

        result_summary = {
            "runs": len(results),
            "experiment": "background_change",
            "variation": args.variation,
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 4),
            "overall_mean_zone_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
            "overall_mean_hallucinations": round(sum(halls) / len(halls), 2) if halls else 0,
            "overall_mean_misses": round(sum(misses) / len(misses), 2) if misses else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(results) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "per_background": per_background,
            "baseline_background": "light_gray",
            "baseline_zone_accuracy": round(baseline, 4),
            "worst_background": worst_bg,
            "worst_zone_accuracy": round(bg_accs[worst_bg], 4),
            "worst_drop_vs_baseline": round(baseline - bg_accs[worst_bg], 4),
            "best_background": best_bg,
            "best_zone_accuracy": round(bg_accs[best_bg], 4),
            "parameters": {
                "temperature": args.temperature,
                "show_grid": show_grid,
                "background_color_arg": args.background_color or "(cycled)",
            },
        }

    print(f"RESULT:{json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
