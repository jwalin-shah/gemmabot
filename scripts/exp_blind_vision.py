"""Experiment: TRUE BLIND VISION — no zone priming whatsoever.

The model is NOT told about zone labels, zone layout, or zone names.
It must describe object positions using free-form spatial language.

Grid lines are drawn on the image but NO text labels (no "Zone A" text).
The model must PERCEIVE spatial layout from the image alone.

Evaluation maps free-form spatial descriptions back to canonical zones A-F.

Usage:
    python scripts/exp_blind_vision.py --runs 100 --output overnight_results/blind_vision/r9_blind.jsonl --temperature 0.0
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

# Mapping from free-form spatial descriptions to canonical zones
POSITION_TO_ZONE = {
    # Top row
    "top-left": "A", "top left": "A", "upper-left": "A", "upper left": "A",
    "top-center": "B", "top center": "B", "top-middle": "B", "top middle": "B",
    "upper-center": "B", "upper center": "B", "upper-middle": "B", "upper middle": "B",
    "top-right": "C", "top right": "C", "upper-right": "C", "upper right": "C",
    # Bottom row
    "bottom-left": "D", "bottom left": "D", "lower-left": "D", "lower left": "D",
    "bottom-center": "E", "bottom center": "E", "bottom-middle": "E", "bottom middle": "E",
    "lower-center": "E", "lower center": "E", "lower-middle": "E", "lower middle": "E",
    "bottom-right": "F", "bottom right": "F", "lower-right": "F", "lower right": "F",
    # Left column (generic)
    "left": "A", "left side": "A",
    # Right column (generic)
    "right": "F", "right side": "F",
    # Center column (generic — ambiguous, but maps to B/E)
    "center": "B", "middle": "B",
}

# Also handle prefixed forms like "the top-left", "near top-left"
# These are stripped during matching

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
                        "position": {
                            "type": "string",
                            "description": "Spatial position like top-left, top-center, bottom-right, etc.",
                        },
                        "shape": {"type": "string"},
                    },
                    "required": ["description", "color", "position", "shape"],
                    "additionalProperties": False,
                },
            },
            "total_objects_visible": {"type": "integer"},
            "gripper_status": {"type": "string"},
            "are_you_uncertain_about_any_positions": {
                "type": "boolean",
                "description": "Are you uncertain about the position of any object?",
            },
        },
        "required": [
            "observed_objects",
            "total_objects_visible",
            "gripper_status",
            "are_you_uncertain_about_any_positions",
        ],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system. You see a camera image showing a tabletop workspace."""


def render_scene(objects: list[dict], show_grid: bool = True, monochrome: bool = False,
                 jpeg_quality: int = 50, gripper_override: dict | None = None) -> str:
    """Render scene with grid lines but NO text labels on the image."""
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    if show_grid:
        # Draw grid lines only — NO text labels
        for c in range(1, 3):
            d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=(200, 200, 210), width=1)
        d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)

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


def extract_zone_from_position(position_str: str) -> str | None:
    """Map free-form position description back to a canonical zone label.

    Handles variations like 'top-left', 'the top left corner', 'upper center', etc.
    """
    if not position_str or not isinstance(position_str, str):
        return None

    pos_lower = position_str.strip().lower()

    # Try direct lookup first
    if pos_lower in POSITION_TO_ZONE:
        return POSITION_TO_ZONE[pos_lower]

    # Try stripping common prefixes/suffixes
    cleaned = pos_lower
    for prefix in ["the ", "near ", "in the ", "at the ", "on the "]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    for suffix in [" area", " region", " corner", " section", " quadrant", " part", " side",
                    " of the image", " of the workspace", " of the table", " of the scene"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break

    # Try lookup again after cleaning
    if cleaned in POSITION_TO_ZONE:
        return POSITION_TO_ZONE[cleaned]

    # Try compound patterns like "top left corner" → search for known substrings
    known_phrases = [
        ("top-left", "A"), ("top left", "A"),
        ("upper-left", "A"), ("upper left", "A"),
        ("top-center", "B"), ("top center", "B"),
        ("top-middle", "B"), ("top middle", "B"),
        ("upper-center", "B"), ("upper center", "B"),
        ("top-right", "C"), ("top right", "C"),
        ("upper-right", "C"), ("upper right", "C"),
        ("bottom-left", "D"), ("bottom left", "D"),
        ("lower-left", "D"), ("lower left", "D"),
        ("bottom-center", "E"), ("bottom center", "E"),
        ("bottom-middle", "E"), ("bottom middle", "E"),
        ("lower-center", "E"), ("lower center", "E"),
        ("bottom-right", "F"), ("bottom right", "F"),
        ("lower-right", "F"), ("lower right", "F"),
    ]

    for phrase, zone in known_phrases:
        if phrase in cleaned:
            return zone

    return None


def evaluate(parsed: dict, ground_truth: list[dict]) -> dict:
    observed = parsed.get("observed_objects", [])
    reported_count = parsed.get("total_objects_visible", 0)
    uncertain = parsed.get("are_you_uncertain_about_any_positions", False)
    position_matches = 0
    position_total = 0
    matched_truth = set()
    matched_obs = set()
    position_map_results = []  # Track mapping for analysis

    for oi, obs in enumerate(observed):
        obs_position = obs.get("position", "")
        obs_desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()

        # Map free-form position to zone
        mapped_zone = extract_zone_from_position(obs_position)

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
            position_total += 1
            expected_zone = ground_truth[best_gi]["expected_zone"]
            is_correct = mapped_zone == expected_zone
            if is_correct:
                position_matches += 1

            position_map_results.append({
                "object_id": ground_truth[best_gi]["id"],
                "raw_position": obs_position,
                "mapped_zone": mapped_zone,
                "expected_zone": expected_zone,
                "correct": is_correct,
            })
        else:
            position_map_results.append({
                "object_id": None,
                "raw_position": obs_position,
                "mapped_zone": mapped_zone,
                "expected_zone": None,
                "correct": False,
            })

    hallucinations = len(observed) - len(matched_obs)
    misses = len(ground_truth) - len(matched_truth)

    return {
        "zone_accuracy": position_matches / max(position_total, 1),
        "zone_matches": position_matches,
        "zone_total": position_total,
        "hallucinations": hallucinations,
        "misses": misses,
        "count_error": abs(reported_count - len(ground_truth)),
        "reported_count": reported_count,
        "true_count": len(ground_truth),
        "uncertain": uncertain,
        "position_mapping": position_map_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--monochrome", action="store_true")
    parser.add_argument("--show-grid", action="store_true", default=True)
    parser.add_argument("--no-grid", action="store_true", dest="no_grid")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="blind_vision",
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

    n_scenes = len(SCENE_LAYOUTS)
    reps = max(1, args.runs // n_scenes)

    for si, layout in enumerate(SCENE_LAYOUTS):
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

            # User prompt: NO zone mentions, ask for spatial language
            prompt = (
                "What objects do you see in this image? "
                "Describe each object's color, shape, and where it is positioned in the image. "
                "Use spatial terms like 'top-left', 'top-center', 'bottom-right', etc."
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
                    "experiment": "blind_vision",
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
                    "uncertain": score["uncertain"],
                    "latency_ms": round(latency_ms, 1),
                    "temperature": args.temperature,
                    "jpeg_quality": args.jpeg_quality,
                    "monochrome": args.monochrome,
                    "show_grid": show_grid,
                    "success": score["zone_accuracy"] >= 0.5 and score["count_error"] <= 1,
                    "error": None,
                    "prompt_sent": prompt,
                    "system_prompt_sent": SYSTEM_PROMPT,
                    "raw_response": result.content,
                    "position_mapping": score["position_mapping"],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)

                if run_num % 25 == 0 or run_num == 1:
                    print(f"  [{run_num:>4d}/{args.runs}] {len(objects)} objects | "
                          f"zone: {score['zone_accuracy']:.0%} | uncertain={score['uncertain']} "
                          f"| lat: {latency_ms:.0f}ms")

            except Exception as e:
                errors += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                entry = {
                    "run": run_num, "experiment": "blind_vision", "variation": args.variation,
                    "scene": si, "n_objects": len(objects),
                    "error": str(e)[:200], "latency_ms": round(latency_ms, 1),
                    "success": False, "prompt_sent": prompt,
                    "system_prompt_sent": SYSTEM_PROMPT,
                    "raw_response": "",
                    "temperature": args.temperature, "jpeg_quality": args.jpeg_quality,
                    "monochrome": args.monochrome, "show_grid": show_grid,
                    "uncertain": None,
                    "position_mapping": [],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)
                if errors >= 20:
                    print(f"  [{errors} errors — aborting]")
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
        uncertain_runs = [r for r in results if r.get("uncertain") is True]

        # Collect all position mappings for analysis
        all_mappings = []
        for r in results:
            for m in r.get("position_mapping", []):
                all_mappings.append(m)

        result_summary = {
            "runs": len(results),
            "experiment": "blind_vision",
            "variation": args.variation,
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 4),
            "mean_zone_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "uncertain_count": len(uncertain_runs),
            "uncertain_rate": round(len(uncertain_runs) / len(results), 4) if results else 0,
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
            # Analysis of spatial language used
            "position_language_analysis": {
                "total_mappings": len(all_mappings),
                "mappable_positions": len([m for m in all_mappings if m["mapped_zone"] is not None]),
                "unmappable_positions": len([m for m in all_mappings if m["mapped_zone"] is None]),
                "unique_raw_positions": len(set(m["raw_position"] for m in all_mappings if m["raw_position"])),
            },
        }

    print(f"RESULT:{json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
