"""Experiment: High Object Count — Zone Identification with 7+ Objects.

Same as the standard vision experiment, but each scene has 7 objects rather
than 3-5. Tests whether Gemma 4's zone accuracy degrades as object count
increases.

Usage:
    python scripts/exp_high_count.py --runs 150 --output overnight_results/high_count/r6_7objects.jsonl --variation objects_7 --no-grid --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

# 7 objects each, spread across all 6 zones (some zones get 2 objects)
HIGH_COUNT_SCENES = [
    {"red_cup": "A", "blue_cup": "B", "green_cup": "C", "cracked_cup": "D", "yellow_cube": "E", "orange_star": "F", "pink_cup": "A"},
    {"red_cup": "B", "blue_cup": "E", "green_cup": "F", "cracked_cup": "D", "yellow_cube": "C", "orange_star": "A", "purple_cube": "B"},
    {"red_cup": "D", "blue_cup": "E", "green_cup": "F", "cracked_cup": "C", "yellow_cube": "B", "orange_star": "A", "pink_cup": "F"},
    {"red_cup": "A", "blue_cup": "C", "green_cup": "F", "yellow_cube": "D", "orange_star": "B", "pink_cup": "E", "purple_cube": "A"},
    {"red_cup": "F", "blue_cup": "E", "green_cup": "D", "cracked_cup": "C", "yellow_cube": "B", "orange_star": "A", "pink_cup": "D"},
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


def render_scene(objects: list[dict], show_grid: bool = True, monochrome: bool = False,
                 jpeg_quality: int = 50, gripper_override: dict | None = None) -> str:
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    if show_grid:
        for c in range(1, 3):
            d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=(200, 200, 210), width=1)
        d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)
        for i, lab in enumerate(ZONE_LABELS):
            r_idx, c_idx = divmod(i, 3)
            d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)), f"Zone {lab}", fill=(170, 170, 180))

    for obj in objects:
        x, y = obj["x"], obj["y"]
        r = obj.get("radius", 26)
        if monochrome:
            color = (120, 120, 120)
        else:
            color = obj["color"]

        shape = obj.get("shape", "round")
        if shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)
        elif shape == "star":
            # Simple 5-point star approximation
            pts = []
            for k in range(10):
                angle = k * 36 - 90
                rad = r if k % 2 == 0 else r * 0.45
                pts.append((x + rad * __import__('math').cos(angle * __import__('math').pi / 180),
                            y + rad * __import__('math').sin(angle * __import__('math').pi / 180)))
            d.polygon(pts, fill=color, outline=(35, 35, 35))
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)

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


def place_objects(layout: dict, monochrome: bool = False) -> list[dict]:
    objects = []
    for oid, zone in layout.items():
        t = next(o for o in OBJECT_TEMPLATES if o["id"] == oid)
        zx, zy = ZONE_CENTERS[zone]
        objects.append({
            "id": oid,
            "x": zx + random.randint(-35, 35),
            "y": zy + random.randint(-25, 25),
            "color": (120, 120, 120) if monochrome else t["color"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--monochrome", action="store_true")
    parser.add_argument("--show-grid", action="store_true", default=True)
    parser.add_argument("--no-grid", action="store_true", dest="no_grid")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="objects_7",
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

    # Use HIGH_COUNT_SCENES instead of SCENE_LAYOUTS
    n_scenes = len(HIGH_COUNT_SCENES)
    reps = max(1, args.runs // n_scenes)

    for si, layout in enumerate(HIGH_COUNT_SCENES):
        if si * reps >= args.runs:
            break
        for rep in range(reps):
            run_num = si * reps + rep + 1
            if run_num > args.runs:
                break

            objects = place_objects(layout, monochrome=args.monochrome)
            image_b64 = render_scene(
                objects,
                show_grid=show_grid,
                monochrome=args.monochrome,
                jpeg_quality=args.jpeg_quality,
            )

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
                score = evaluate(parsed, objects)

                entry = {
                    "run": run_num,
                    "experiment": "vision",
                    "variation": args.variation,
                    "scene": si,
                    "n_objects": len(objects),
                    "reported_count": score["reported_count"],
                    "count_error": score["count_error"],
                    "zone_accuracy": round(score["zone_accuracy"], 4),
                    "zone_matches": score["zone_matches"],
                    "zone_total": score["zone_total"],
                    "hallucinations": score["hallucinations"],
                    "misses": score["misses"],
                    "latency_ms": round(latency_ms, 1),
                    "temperature": args.temperature,
                    "jpeg_quality": args.jpeg_quality,
                    "monochrome": args.monochrome,
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
                          f"zone: {score['zone_accuracy']:.0%} | lat: {latency_ms:.0f}ms")

            except Exception as e:
                errors += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                entry = {
                    "run": run_num, "experiment": "vision", "variation": args.variation,
                    "scene": si, "n_objects": len(objects),
                    "error": str(e)[:200], "latency_ms": round(latency_ms, 1),
                    "success": False, "prompt_sent": prompt, "raw_response": "",
                    "temperature": args.temperature, "jpeg_quality": args.jpeg_quality,
                    "monochrome": args.monochrome, "show_grid": show_grid,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)
                if errors >= 20:
                    print(f"  [{errors} errors — aborting")
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
        latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
        accs = [r["zone_accuracy"] for r in results if "zone_accuracy" in r]
        halls = [r["hallucinations"] for r in results if "hallucinations" in r]
        misses = [r["misses"] for r in results if "misses" in r]

        result_summary = {
            "runs": len(results),
            "experiment": "vision",
            "variation": args.variation,
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 4),
            "mean_zone_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
            "mean_hallucinations": round(sum(halls) / len(halls), 2) if halls else 0,
            "mean_misses": round(sum(misses) / len(misses), 2) if misses else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(results) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "parameters": {
                "temperature": args.temperature,
                "jpeg_quality": args.jpeg_quality,
                "monochrome": args.monochrome,
                "show_grid": show_grid,
            },
        }

    print(f"RESULT:{json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
