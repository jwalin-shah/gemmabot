"""Experiment: Pure Vision Zone Identification.

Structured output at temperature 0.0. NO coordinates, NO labels, NO cheating.
Gemma must identify objects from the rendered image alone.

Saves every prompt and raw response for post-hoc analysis.

Usage:
    python scripts/exp_vision.py --runs 200 --output results.jsonl [--temperature 0.0] [--jpeg-quality 50] [--monochrome]
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


def render_scene(objects: list[dict], show_grid: bool = True, monochrome: bool = False,
                 jpeg_quality: int = 95, gripper_override: dict | None = None) -> str:
    from PIL import Image, ImageDraw
    import base64, io, math
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
            pts = []
            for k in range(10):
                angle = k * 36 - 90
                rad = r if k % 2 == 0 else r * 0.45
                pts.append((x + rad * math.cos(angle * math.pi / 180),
                            y + rad * math.sin(angle * math.pi / 180)))
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
    # FIX: actually apply JPEG compression when quality < 95
    if jpeg_quality < 95:
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
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


def random_layout(rng: random.Random) -> dict:
    """Generate a fresh random scene layout: 1-6 distinct objects, distinct zones."""
    n_objects = rng.randint(1, 6)
    templates = rng.sample(OBJECT_TEMPLATES, n_objects)
    zones = rng.sample(ZONE_LABELS, n_objects)
    return {t["id"]: z for t, z in zip(templates, zones)}



# Color name -> RGB for identity-aware secondary metric
_COLOR_NAME_MAP = {
    "red":    (210, 60,  60),
    "blue":   (60,  90,  210),
    "green":  (60,  180, 60),
    "yellow": (220, 200, 40),
    "orange": (230, 140, 40),
    "pink":   (210, 120, 160),
    "purple": (140, 60,  180),
    "tan":    (200, 175, 120),
    "brown":  (200, 175, 120),
    "beige":  (200, 175, 120),
}
# Precompute reverse mapping (color tuple -> list of names)
_RGB_TO_NAMES: dict[tuple, list[str]] = {}
for _cn, _crgb in _COLOR_NAME_MAP.items():
    _RGB_TO_NAMES.setdefault(_crgb, []).append(_cn)


def evaluate(parsed: dict, ground_truth: list[dict]) -> dict:
    """Identity-agnostic honest zone metric using Counter intersection.

    Primary metric: zone_occupancy_accuracy = intersection / n_gt
      - denominator is always n_gt, so every miss hurts
    Secondary (colored scenes only): identity_zone_accuracy
    """
    from collections import Counter
    observed = parsed.get("observed_objects", [])
    reported_count = parsed.get("total_objects_visible", 0)

    gt_zones = [g["expected_zone"] for g in ground_truth]
    obs_zones = [o.get("zone", "") for o in observed if o.get("zone")]

    inter = sum((Counter(gt_zones) & Counter(obs_zones)).values())

    zone_occupancy_accuracy = inter / max(len(gt_zones), 1)
    zone_precision = inter / max(len(obs_zones), 1)

    if zone_occupancy_accuracy + zone_precision > 0:
        zone_f1 = (2 * zone_occupancy_accuracy * zone_precision
                   / (zone_occupancy_accuracy + zone_precision))
    else:
        zone_f1 = 0.0

    hallucinations = max(0, len(obs_zones) - inter)
    misses = max(0, len(gt_zones) - inter)
    count_error = abs(reported_count - len(gt_zones))

    # Identity-aware secondary metric -- only for non-monochrome scenes
    mono_color = (120, 120, 120)
    is_monochrome = all(g.get("color") == mono_color for g in ground_truth)

    identity_zone_accuracy = None
    if not is_monochrome:
        id_correct = 0
        for gt in ground_truth:
            gt_color = tuple(gt.get("color", (0, 0, 0)))
            gt_zone = gt["expected_zone"]
            gt_color_names = _RGB_TO_NAMES.get(gt_color, [])
            found = False
            for obs in observed:
                obs_desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()
                obs_zone = obs.get("zone", "")
                if obs_zone == gt_zone and any(cn in obs_desc for cn in gt_color_names):
                    found = True
                    break
            if found:
                id_correct += 1
        identity_zone_accuracy = id_correct / max(len(ground_truth), 1)

    return {
        # Primary headline metric (alias kept for downstream compatibility)
        "zone_accuracy": zone_occupancy_accuracy,
        "zone_occupancy_accuracy": zone_occupancy_accuracy,
        "zone_precision": zone_precision,
        "zone_f1": zone_f1,
        "zone_matches": inter,
        "zone_total": len(gt_zones),
        "hallucinations": hallucinations,
        "misses": misses,
        "count_error": count_error,
        "reported_count": reported_count,
        "true_count": len(gt_zones),
        "identity_zone_accuracy": identity_zone_accuracy,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--monochrome", action="store_true")
    parser.add_argument("--show-grid", action="store_true", default=True)
    parser.add_argument("--no-grid", action="store_true", dest="no_grid")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="standard",
                        help="Label for this variation (e.g. temp_03, jpeg_25, monochrome)")
    parser.add_argument("--randomize", action="store_true", default=True,
                        help="Generate fresh random scenes per run (default: enabled)")
    parser.add_argument("--no-randomize", action="store_false", dest="randomize",
                        help="Use fixed SCENE_LAYOUTS (for reproducibility)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for scene generation")
    args = parser.parse_args()

    show_grid = not args.no_grid
    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    scene_rng = random.Random(args.seed)

    client = CerebrasClient()
    results = []
    errors = 0
    start_time = time.time()

    for run_num in range(1, args.runs + 1):
        if args.randomize:
            layout = random_layout(scene_rng)
            si = run_num
        else:
            n_scenes = len(SCENE_LAYOUTS)
            si = (run_num - 1) % n_scenes
            layout = SCENE_LAYOUTS[si]

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

            zone_occ = score["zone_occupancy_accuracy"]
            zone_f1_val = score["zone_f1"]
            hall_val = score["hallucinations"]
            miss_val = score["misses"]

            entry = {
                "run": run_num,
                "experiment": "vision",
                "variation": args.variation,
                "scene": si,
                "n_objects": len(objects),
                "reported_count": score["reported_count"],
                "count_error": score["count_error"],
                "zone_occupancy_accuracy": round(zone_occ, 4),
                "zone_precision": round(score["zone_precision"], 4),
                "zone_f1": round(zone_f1_val, 4),
                "zone_accuracy": round(zone_occ, 4),
                "zone_matches": score["zone_matches"],
                "zone_total": score["zone_total"],
                "hallucinations": hall_val,
                "misses": miss_val,
                "identity_zone_accuracy": (round(score["identity_zone_accuracy"], 4)
                                           if score["identity_zone_accuracy"] is not None else None),
                "latency_ms": round(latency_ms, 1),
                "temperature": args.temperature,
                "jpeg_quality": args.jpeg_quality,
                "monochrome": args.monochrome,
                "show_grid": show_grid,
                "randomize": args.randomize,
                "seed": args.seed,
                "success": zone_occ >= 0.5 and score["count_error"] <= 1,
                "error": None,
                "prompt_sent": prompt,
                "raw_response": result.content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)

            if run_num % 25 == 0 or run_num == 1:
                print(f"  [{run_num:>4d}/{args.runs}] {len(objects)} objects | "
                      f"zone_occ: {zone_occ:.0%} | f1: {zone_f1_val:.0%} | "
                      f"hall: {hall_val} | miss: {miss_val} | lat: {latency_ms:.0f}ms")

        except Exception as e:
            errors += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            entry = {
                "run": run_num, "experiment": "vision", "variation": args.variation,
                "scene": si, "n_objects": len(objects) if objects else 0,
                "error": str(e)[:200], "latency_ms": round(latency_ms, 1),
                "success": False, "prompt_sent": prompt, "raw_response": "",
                "temperature": args.temperature, "jpeg_quality": args.jpeg_quality,
                "monochrome": args.monochrome, "show_grid": show_grid,
                "randomize": args.randomize, "seed": args.seed,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)
            print(f"  ERROR run {run_num}: {str(e)[:100]}")
            if errors >= 20:
                print(f"  {errors} errors -- aborting")
                break

        if output_path and entry:
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start_time

    if not results:
        result_summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        successes = [r for r in results if r.get("success")]
        latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
        accs = [r["zone_occupancy_accuracy"] for r in results if "zone_occupancy_accuracy" in r]
        f1s = [r["zone_f1"] for r in results if "zone_f1" in r]
        halls = [r["hallucinations"] for r in results if "hallucinations" in r]
        misses_list = [r["misses"] for r in results if "misses" in r]

        result_summary = {
            "runs": len(results),
            "experiment": "vision",
            "variation": args.variation,
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 4),
            "mean_zone_occupancy_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "mean_zone_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "mean_zone_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0,
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
            "mean_hallucinations": round(sum(halls) / len(halls), 2) if halls else 0,
            "mean_misses": round(sum(misses_list) / len(misses_list), 2) if misses_list else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(results) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "parameters": {
                "temperature": args.temperature,
                "jpeg_quality": args.jpeg_quality,
                "monochrome": args.monochrome,
                "show_grid": show_grid,
                "randomize": args.randomize,
                "seed": args.seed,
            },
        }

    print(f"RESULT:{json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
