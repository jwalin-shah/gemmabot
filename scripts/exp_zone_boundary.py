"""Experiment: Zone Boundary Sensitivity.

Places objects ON grid lines to test whether Gemma correctly identifies zones
for objects straddling zone boundaries.

Key questions:
  a. Does the model pick ONE zone confidently (never says "between zones")?
  b. Does it hallucinate TWO objects (one in each adjacent zone)?
  c. Which side of the boundary does it prefer?

Usage:
    python scripts/exp_zone_boundary.py --runs 200 --output overnight_results/zone_boundary/r2_zone_boundary.jsonl --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient

WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2  # 128, 192

# Grid lines
VERT_LINES = [128, 256]    # boundaries at x=128 and x=256
HORIZ_LINE = 192           # boundary at y=192

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


def get_zones_at(x: int, y: int) -> list[str]:
    """Return the zone(s) that contain this (x,y) coordinate.

    For points exactly on grid lines, return all adjacent zones.
    """
    col = []
    if x < VERT_LINES[0]:
        col.append("left")
    elif x == VERT_LINES[0]:
        col.append("left")
        col.append("center")
    elif x < VERT_LINES[1]:
        col.append("center")
    elif x == VERT_LINES[1]:
        col.append("center")
        col.append("right")
    else:
        col.append("right")

    row = []
    if y < HORIZ_LINE:
        row.append("top")
    elif y == HORIZ_LINE:
        row.append("top")
        row.append("bottom")
    else:
        row.append("bottom")

    zone_map = {
        ("left", "top"): "A",
        ("center", "top"): "B",
        ("right", "top"): "C",
        ("left", "bottom"): "D",
        ("center", "bottom"): "E",
        ("right", "bottom"): "F",
    }

    zones = []
    for c in col:
        for r in row:
            z = zone_map.get((c, r))
            if z:
                zones.append(z)
    return sorted(set(zones))


# Each boundary scene: list of dicts with explicit (x,y) placement
# boundary_type describes which lines the object is on
BOUNDARY_SCENES = [
    {
        "scene_id": 0,
        "objects": [
            # --- Boundary objects ---
            {"id": "red_cup",    "x": 192, "y": 192, "boundary": "horiz_center", "adjacent": ["B", "E"]},    # Image center, on horizontal line B/E
            {"id": "blue_cup",   "x": 128, "y": 96,  "boundary": "vert_left",    "adjacent": ["A", "B"]},     # On vertical line A/B
            {"id": "green_cup",  "x": 256, "y": 288, "boundary": "vert_right",   "adjacent": ["C", "F"]},     # On vertical line C/F
            # --- Near-boundary (5px offset) ---
            {"id": "yellow_cube","x": 133, "y": 96,  "boundary": "near_vert",    "adjacent": ["B"]},           # 5px right of x=128 → zone B
            {"id": "cracked_cup","x": 192, "y": 187, "boundary": "near_horiz",   "adjacent": ["B"]},           # 5px above y=192 → zone B
            # --- Clearly in zone ---
            {"id": "pink_cup",   "x": 64,  "y": 64,  "boundary": "none",         "adjacent": ["A"]},            # Deep in A
            {"id": "purple_cube","x": 320, "y": 320, "boundary": "none",         "adjacent": ["F"]},            # Deep in F
        ],
    },
    {
        "scene_id": 1,
        "objects": [
            # Intersection boundary: corner of A/B/D/E
            {"id": "orange_star","x": 128, "y": 192, "boundary": "intersection", "adjacent": ["A", "B", "D", "E"]},
            # Intersection boundary: corner of B/C/E/F
            {"id": "red_cup",    "x": 256, "y": 192, "boundary": "intersection", "adjacent": ["B", "C", "E", "F"]},
            # On horizontal boundary
            {"id": "blue_cup",   "x": 64,  "y": 192, "boundary": "horiz_left",   "adjacent": ["A", "D"]},       # On horizontal line A/D
            {"id": "green_cup",  "x": 320, "y": 192, "boundary": "horiz_right",  "adjacent": ["C", "F"]},       # On horizontal line C/F
            # Near-boundary
            {"id": "cracked_cup","x": 131, "y": 192, "boundary": "near_horiz_vert","adjacent": ["B", "E"]},     # 3px right of x=128, on horiz line
            # Controls
            {"id": "yellow_cube","x": 64,  "y": 320, "boundary": "none",         "adjacent": ["D"]},
            {"id": "purple_cube","x": 320, "y": 64,  "boundary": "none",         "adjacent": ["C"]},
        ],
    },
    {
        "scene_id": 2,
        "objects": [
            # Center again (reproducibility check)
            {"id": "red_cup",    "x": 192, "y": 192, "boundary": "horiz_center", "adjacent": ["B", "E"]},
            # On vertical boundary in bottom half
            {"id": "blue_cup",   "x": 128, "y": 288, "boundary": "vert_bot",     "adjacent": ["D", "E"]},       # On vertical line D/E
            {"id": "green_cup",  "x": 256, "y": 96,  "boundary": "vert_top",     "adjacent": ["B", "C"]},       # On vertical line B/C
            # Near-boundary
            {"id": "orange_star","x": 251, "y": 96,  "boundary": "near_left",    "adjacent": ["B"]},             # 5px left of x=256 → zone B
            {"id": "cracked_cup","x": 256, "y": 197, "boundary": "near_below",    "adjacent": ["E", "F"]},       # on x=256, 5px below y=192 → between E/F
            # Controls
            {"id": "yellow_cube","x": 192, "y": 64,  "boundary": "none",         "adjacent": ["B"]},
            {"id": "pink_cup",   "x": 64,  "y": 288, "boundary": "none",         "adjacent": ["D"]},
            {"id": "purple_cube","x": 320, "y": 288, "boundary": "none",         "adjacent": ["F"]},
        ],
    },
    {
        "scene_id": 3,
        "objects": [
            # Two objects on same boundary line (to test if model sees them as separate)
            {"id": "red_cup",    "x": 128, "y": 48,  "boundary": "vert_left_top","adjacent": ["A", "B"]},
            {"id": "blue_cup",   "x": 128, "y": 336, "boundary": "vert_left_bot","adjacent": ["D", "E"]},
            # One on horizontal
            {"id": "green_cup",  "x": 160, "y": 192, "boundary": "horiz_mid",    "adjacent": ["B", "E"]},
            # Near-boundary (just 2px into zone)
            {"id": "cracked_cup","x": 130, "y": 48,  "boundary": "barely_right", "adjacent": ["B"]},
            {"id": "yellow_cube","x": 126, "y": 336, "boundary": "barely_left",  "adjacent": ["D"]},
            # Controls
            {"id": "pink_cup",   "x": 64,  "y": 96,  "boundary": "none",         "adjacent": ["A"]},
            {"id": "purple_cube","x": 300, "y": 300, "boundary": "none",         "adjacent": ["F"]},
            {"id": "orange_star","x": 300, "y": 96,  "boundary": "none",         "adjacent": ["C"]},
        ],
    },
    {
        "scene_id": 4,
        "objects": [
            # All objects on / near boundaries (stress test)
            {"id": "red_cup",    "x": 192, "y": 192, "boundary": "horiz_center", "adjacent": ["B", "E"]},
            {"id": "blue_cup",   "x": 128, "y": 192, "boundary": "intersection_left","adjacent": ["A","B","D","E"]},
            {"id": "green_cup",  "x": 256, "y": 192, "boundary": "intersection_right","adjacent": ["B","C","E","F"]},
            {"id": "yellow_cube","x": 128, "y": 132, "boundary": "vert_mid_top",   "adjacent": ["A", "B"]},     # upper half vert line
            {"id": "orange_star","x": 256, "y": 256, "boundary": "vert_mid_bot",   "adjacent": ["E", "F"]},     # lower half vert line
            {"id": "pink_cup",   "x": 64,  "y": 192, "boundary": "horiz_left",     "adjacent": ["A", "D"]},
            {"id": "purple_cube","x": 320, "y": 192, "boundary": "horiz_right",    "adjacent": ["C", "F"]},
            # Near-boundary controls
            {"id": "cracked_cup","x": 133, "y": 132, "boundary": "near_control",   "adjacent": ["B"]},
        ],
    },
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
    import math
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
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def prepare_objects(scene: dict, monochrome: bool = False) -> list[dict]:
    """Convert scene object definitions to render-ready objects with computed zones."""
    objects = []
    for obj_def in scene["objects"]:
        t = next(o for o in OBJECT_TEMPLATES if o["id"] == obj_def["id"])
        color = (120, 120, 120) if monochrome else t["color"]
        computed_zones = get_zones_at(obj_def["x"], obj_def["y"])
        objects.append({
            "id": obj_def["id"],
            "x": obj_def["x"],
            "y": obj_def["y"],
            "color": color,
            "radius": 26,
            "shape": t["shape"],
            "attr": t["attr"],
            "boundary": obj_def.get("boundary", "none"),
            "adjacent_zones": computed_zones,
            "expected_zones": obj_def.get("adjacent", computed_zones),
        })
    return objects


def evaluate_boundary(parsed: dict, ground_truth: list[dict]) -> dict:
    """Evaluate with special handling for boundary objects.

    For non-boundary objects: standard zone accuracy.
    For boundary objects: measure which zone is assigned, whether two objects
    are hallucinated for one boundary object, and confidence level.
    """
    observed = parsed.get("observed_objects", [])
    reported_count = parsed.get("total_objects_visible", 0)
    n_truth = len(ground_truth)

    # Separate boundary vs non-boundary ground truth
    boundary_gts = [gt for gt in ground_truth if gt["boundary"] != "none"]
    non_boundary_gts = [gt for gt in ground_truth if gt["boundary"] == "none"]

    # Match observed to ground truth (fuzzy matching based on description)
    matched_truth: set[int] = set()
    matched_obs: set[int] = set()
    zone_matches = 0
    zone_total = 0

    # Detailed boundary analysis
    boundary_assignments: dict[str, dict] = {}  # gt_id -> {"assigned_zone", "correct_adjacent", "count_assigned"}

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
            gt = ground_truth[best_gi]
            matched_truth.add(best_gi)
            matched_obs.add(oi)

            # Track boundary assignments
            gt_id = gt["id"]
            if gt_id not in boundary_assignments:
                boundary_assignments[gt_id] = {
                    "expected_zones": gt["expected_zones"],
                    "adjacent_zones": gt["adjacent_zones"],
                    "boundary": gt["boundary"],
                    "assigned_zones": [],
                    "times_matched": 0,
                }
            boundary_assignments[gt_id]["assigned_zones"].append(obs_zone)
            boundary_assignments[gt_id]["times_matched"] += 1

            # Zone correctness: for non-boundary objects, check exact match
            if gt["boundary"] == "none":
                zone_total += 1
                if obs_zone in gt["expected_zones"]:
                    zone_matches += 1

    # Count hallucinations and misses
    hallucinations = len(observed) - len(matched_obs)
    misses = len(ground_truth) - len(matched_truth)

    # Analyze boundary assignments
    boundary_results = {}
    for gt in boundary_gts:
        gt_id = gt["id"]
        ba = boundary_assignments.get(gt_id, {
            "expected_zones": gt["expected_zones"],
            "adjacent_zones": gt["adjacent_zones"],
            "boundary": gt["boundary"],
            "assigned_zones": [],
            "times_matched": 0,
        })

        # Determine if hallucinated (model reported 2 objects matching same GT)
        # or just one assignment
        assigned = ba["assigned_zones"]
        zone_counter = Counter(assigned)

        # Check if any assignment matches an adjacent zone
        adjacent_hits = sum(1 for z in assigned if z in gt["expected_zones"])

        boundary_results[gt_id] = {
            "boundary_type": gt["boundary"],
            "expected_zones": gt["expected_zones"],
            "adjacent_zones": gt["adjacent_zones"],
            "assigned_zones": assigned,
            "unique_zones_assigned": sorted(set(assigned)),
            "n_assignments": len(assigned),
            "adjacent_hits": adjacent_hits,
            "double_hallucination": len(set(assigned)) > 1 and gt["boundary"] not in ("none", "near_", "barely_"),
            "preferred_zone": zone_counter.most_common(1)[0][0] if assigned else None,
        }

    # Count double hallucinations (model assigned 2+ different zones to one boundary object)
    double_hallucinations = sum(
        1 for br in boundary_results.values()
        if br["double_hallucination"]
    )

    return {
        "zone_accuracy": zone_matches / max(zone_total, 1) if zone_total > 0 else 1.0,
        "zone_matches": zone_matches,
        "zone_total": zone_total,
        "hallucinations": hallucinations,
        "misses": misses,
        "count_error": abs(reported_count - n_truth),
        "reported_count": reported_count,
        "true_count": n_truth,
        "n_boundary_objects": len(boundary_gts),
        "n_control_objects": len(non_boundary_gts),
        "boundary_results": boundary_results,
        "double_hallucinations": double_hallucinations,
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
    parser.add_argument("--variation", type=str, default="zone_boundary",
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

    n_scenes = len(BOUNDARY_SCENES)
    reps = max(1, args.runs // n_scenes)

    # Statistics across all boundary results
    all_boundary_preferences: dict[str, list[str]] = {}  # gt_id -> list of assigned zones
    all_double_hallucinations = 0
    total_boundary_instances = 0

    for si, scene in enumerate(BOUNDARY_SCENES):
        if si * reps >= args.runs:
            break
        for rep in range(reps):
            run_num = si * reps + rep + 1
            if run_num > args.runs:
                break

            # No random jitter — fixed positions for reproducibility
            objects = prepare_objects(scene, monochrome=args.monochrome)
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
                score = evaluate_boundary(parsed, objects)

                entry = {
                    "run": run_num,
                    "experiment": "zone_boundary",
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
                    "n_boundary_objects": score["n_boundary_objects"],
                    "double_hallucinations": score["double_hallucinations"],
                    "boundary_results": score["boundary_results"],
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

                # Accumulate boundary statistics
                for gt_id, br in score["boundary_results"].items():
                    if gt_id not in all_boundary_preferences:
                        all_boundary_preferences[gt_id] = []
                    all_boundary_preferences[gt_id].extend(br["assigned_zones"])
                all_double_hallucinations += score["double_hallucinations"]
                total_boundary_instances += score["n_boundary_objects"]

                if run_num % 25 == 0 or run_num == 1:
                    bd = score["double_hallucinations"]
                    print(f"  [{run_num:>4d}/{args.runs}] scene={si} | "
                          f"control_acc: {score['zone_accuracy']:.0%} | "
                          f"boundary_double_halluc: {bd}/{score['n_boundary_objects']} | "
                          f"lat: {latency_ms:.0f}ms")

            except Exception as e:
                errors += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                entry = {
                    "run": run_num, "experiment": "zone_boundary", "variation": args.variation,
                    "scene": si, "n_objects": len(objects),
                    "error": str(e)[:200], "latency_ms": round(latency_ms, 1),
                    "success": False, "prompt_sent": prompt, "raw_response": "",
                    "temperature": args.temperature, "jpeg_quality": args.jpeg_quality,
                    "monochrome": args.monochrome, "show_grid": show_grid,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)
                if errors >= 20:
                    print(f"  ERROR: {errors} errors — aborting")
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
        double_halls = [r["double_hallucinations"] for r in results if "double_hallucinations" in r]

        # Compute boundary preference stats
        boundary_summary = {}
        for gt_id, zone_list in sorted(all_boundary_preferences.items()):
            if not zone_list:
                continue
            counter = Counter(zone_list)
            # Find the object definition for this gt_id
            obj_def = None
            for scene in BOUNDARY_SCENES:
                for od in scene["objects"]:
                    if od["id"] == gt_id:
                        obj_def = od
                        break
                if obj_def:
                    break
            boundary_summary[gt_id] = {
                "boundary_type": obj_def.get("boundary", "unknown") if obj_def else "unknown",
                "expected_zones": obj_def.get("adjacent", []) if obj_def else [],
                "n_observations": len(zone_list),
                "zone_distribution": dict(counter.most_common()),
                "preferred_zone": counter.most_common(1)[0][0],
                "preferred_pct": round(counter.most_common(1)[0][1] / len(zone_list), 4),
            }

        result_summary = {
            "runs": len(results),
            "experiment": "zone_boundary",
            "variation": args.variation,
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 4),
            "mean_control_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "total_double_hallucinations": all_double_hallucinations,
            "double_hallucination_rate": round(all_double_hallucinations / max(total_boundary_instances, 1), 4),
            "mean_boundary_hallucinations": round(sum(halls) / len(halls), 2) if halls else 0,
            "mean_misses": round(sum(misses) / len(misses), 2) if misses else 0,
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(results) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "boundary_preferences": boundary_summary,
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
