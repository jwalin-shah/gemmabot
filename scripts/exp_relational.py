#!/usr/bin/env python3
"""Experiment: Side-by-Side Change Detection (Relational / Temporal Reasoning).

Renders two versions of a tabletop scene side-by-side (left = "before",
right = "after") and asks Gemma what changed.  This is the FIRST test of
true temporal reasoning (not just repeated single-image analysis).

6 experiment types (30 runs each = 180 total):
  1. move          — one object moves to a different zone
  2. color_change  — one object changes color (e.g. red → blue)
  3. appear        — a new object appears in the after image
  4. disappear     — an object vanishes in the after image
  5. no_change     — BOTH images are identical (control)
  6. swap          — two objects swap colors

Evaluation:
  - Change detection:  exact match on `change_type`
  - Object indent:     zone accuracy for each side independently

Usage:
    python scripts/exp_relational.py --runs 180 --output overnight_results/relational/r4_relational.jsonl --temperature 0.0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient


# ─── Constants ────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 384, 384
HALF_W = WIDTH // 2  # 192 — width of each side

# Zone grid (conceptual — no grid lines drawn) for each half
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw_half = HALF_W / 3   # 64
ch_half = HEIGHT / 2    # 192

# Zone centers for the LEFT half (before)
LEFT_ZONE_CENTERS: dict[str, tuple[int, int]] = {}
for i, lab in enumerate(ZONE_LABELS):
    r, c = divmod(i, 3)
    LEFT_ZONE_CENTERS[lab] = (int(c * cw_half + cw_half / 2),
                              int(r * ch_half + ch_half / 2))

# Zone centers for the RIGHT half (after) — shifted by HALF_W
RIGHT_ZONE_CENTERS: dict[str, tuple[int, int]] = {}
for lab, (zx, zy) in LEFT_ZONE_CENTERS.items():
    RIGHT_ZONE_CENTERS[lab] = (zx + HALF_W, zy)


OBJECT_TEMPLATES = [
    {"id": "red_cup",     "color": (210, 60, 60),   "shape": "round",   "attr": ""},
    {"id": "blue_cup",    "color": (60, 90, 210),   "shape": "round",   "attr": ""},
    {"id": "green_cup",   "color": (60, 180, 60),   "shape": "round",   "attr": ""},
    {"id": "cracked_cup", "color": (200, 175, 120), "shape": "round",   "attr": "cracked"},
    {"id": "yellow_cube", "color": (220, 200, 40),  "shape": "square",  "attr": ""},
    {"id": "orange_star", "color": (230, 140, 40),  "shape": "star",    "attr": ""},
    {"id": "pink_cup",    "color": (210, 120, 160), "shape": "round",   "attr": ""},
    {"id": "purple_cube", "color": (140, 60, 180),  "shape": "square",  "attr": ""},
]

SCENE_LAYOUTS = [
    {"red_cup": "D", "blue_cup": "E", "cracked_cup": "B"},
    {"red_cup": "A", "blue_cup": "C", "green_cup": "E"},
    {"blue_cup": "F", "green_cup": "A", "cracked_cup": "D"},
    {"red_cup": "D", "blue_cup": "B", "green_cup": "F", "cracked_cup": "C"},
    {"yellow_cube": "A", "green_cup": "B", "cracked_cup": "F"},
    {"red_cup": "F", "blue_cup": "A", "green_cup": "C"},
    {"orange_star": "E", "pink_cup": "B", "purple_cube": "D"},
    {"cracked_cup": "A", "blue_cup": "E", "green_cup": "C", "yellow_cube": "D"},
    {"blue_cup": "C", "cracked_cup": "E", "yellow_cube": "A", "green_cup": "D"},
    {"red_cup": "B", "cracked_cup": "D", "yellow_cube": "F"},
    {"pink_cup": "D", "purple_cube": "C", "cracked_cup": "A"},
    {"red_cup": "E", "cracked_cup": "A", "blue_cup": "F"},
]

EXPERIMENT_TYPES = ["move", "color_change", "appear", "disappear", "no_change", "swap"]

# ─── Schema ───────────────────────────────────────────────────────────────
CHANGE_DETECTION_SCHEMA = {
    "name": "analyze_scene",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "change_detected": {"type": "boolean"},
            "change_type": {
                "type": "string",
                "enum": ["moved", "color_changed", "appeared", "disappeared", "swapped", "no_change"],
            },
            "description": {"type": "string"},
            "objects_in_left_image": {
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
            "objects_in_right_image": {
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
        },
        "required": [
            "change_detected", "change_type", "description",
            "objects_in_left_image", "objects_in_right_image",
        ],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system analyzing a pair of tabletop workspace images side by side.

The LEFT image shows the workspace BEFORE any changes.
The RIGHT image shows the workspace AFTER changes may have occurred.

Zones (not drawn on image):
  A (top-left)     B (top-center)     C (top-right)
  D (bottom-left)  E (bottom-center)  F (bottom-right)

Your job:
1. Compare the left and right images and determine what changed.
2. Identify every object in BOTH images by color, shape, and zone.
3. Be specific. Only describe what you actually see."""


# ─── Object helpers ───────────────────────────────────────────────────────
def _lookup_template(oid: str) -> dict:
    for t in OBJECT_TEMPLATES:
        if t["id"] == oid:
            return t
    raise ValueError(f"Unknown object template: {oid}")


def place_objects(layout: dict, zone_centers: dict) -> list[dict]:
    """Return placed objects for one side, with small jitter."""
    objects = []
    for oid, zone in layout.items():
        t = _lookup_template(oid)
        zx, zy = zone_centers[zone]
        objects.append({
            "id": oid,
            "x": zx + random.randint(-35, 35),
            "y": zy + random.randint(-25, 25),
            "color": t["color"],
            "radius": 26,
            "shape": t["shape"],
            "attr": t["attr"],
            "zone": zone,
        })
    return objects


def deep_copy_objects(objects: list[dict]) -> list[dict]:
    import copy
    return copy.deepcopy(objects)


# ─── Scene mutations ──────────────────────────────────────────────────────
def apply_move(objects: list[dict], zone_centers: dict) -> list[dict]:
    """Move one object to a different zone."""
    objs = deep_copy_objects(objects)
    idx = random.randrange(len(objs))
    current_zone = objs[idx]["zone"]
    possible = [z for z in ZONE_LABELS if z != current_zone]
    new_zone = random.choice(possible)
    zx, zy = zone_centers[new_zone]
    objs[idx]["x"] = zx + random.randint(-35, 35)
    objs[idx]["y"] = zy + random.randint(-25, 25)
    objs[idx]["zone"] = new_zone
    return objs


def apply_color_change(objects: list[dict]) -> list[dict]:
    """Change one object's color to a different color."""
    objs = deep_copy_objects(objects)
    idx = random.randrange(len(objs))
    original_color = objs[idx]["color"]
    candidates = [
        (210, 60, 60), (60, 90, 210), (60, 180, 60),
        (220, 200, 40), (230, 140, 40), (210, 120, 160),
        (140, 60, 180),
    ]
    candidates = [c for c in candidates if c != original_color]
    objs[idx]["color"] = random.choice(candidates)
    return objs


def apply_appear(objects: list[dict], zone_centers: dict) -> list[dict]:
    """Add a new object that wasn't in the before scene."""
    objs = deep_copy_objects(objects)
    existing_ids = {o["id"] for o in objs}
    available = [t for t in OBJECT_TEMPLATES if t["id"] not in existing_ids]
    if not available:
        available = OBJECT_TEMPLATES
    new_t = random.choice(available)
    occupied_zones = {o["zone"] for o in objs}
    free_zones = [z for z in ZONE_LABELS if z not in occupied_zones]
    if not free_zones:
        free_zones = ZONE_LABELS
    zone = random.choice(free_zones)
    zx, zy = zone_centers[zone]
    objs.append({
        "id": new_t["id"],
        "x": zx + random.randint(-35, 35),
        "y": zy + random.randint(-25, 25),
        "color": new_t["color"],
        "radius": 26,
        "shape": new_t["shape"],
        "attr": new_t["attr"],
        "zone": zone,
    })
    return objs


def apply_disappear(objects: list[dict]) -> list[dict]:
    """Remove one object."""
    if len(objects) <= 1:
        return objects  # safety: need at least 1 to remove
    objs = deep_copy_objects(objects)
    idx = random.randrange(len(objs))
    objs.pop(idx)
    return objs


def apply_swap(objects: list[dict]) -> list[dict]:
    """Swap colors of two objects."""
    if len(objects) < 2:
        return deep_copy_objects(objects)
    objs = deep_copy_objects(objects)
    i1, i2 = random.sample(range(len(objs)), 2)
    objs[i1]["color"], objs[i2]["color"] = objs[i2]["color"], objs[i1]["color"]
    return objs


# ─── Rendering ────────────────────────────────────────────────────────────
def render_side_by_side(before_objects: list[dict], after_objects: list[dict]) -> str:
    """Render a 384x384 side-by-side image. Left = before, Right = after.
    NO grid overlay (Round 3 finding: grid hurts accuracy).
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    # Draw a subtle dividing line between the two halves
    d.line([(HALF_W, 0), (HALF_W, HEIGHT)], fill=(180, 180, 190), width=1)

    # Draw "BEFORE" and "AFTER" labels at the top
    d.text((8, 4), "BEFORE", fill=(120, 120, 130))
    d.text((HALF_W + 8, 4), "AFTER", fill=(120, 120, 130))

    def _draw_objects(objects: list[dict], draw: ImageDraw) -> None:
        for obj in objects:
            x, y = obj["x"], obj["y"]
            r = obj.get("radius", 26)
            color = obj["color"]
            shape = obj.get("shape", "round")

            if shape == "square":
                draw.rectangle([x - r, y - r, x + r, y + r], fill=color,
                               outline=(35, 35, 35), width=2)
            elif shape == "star":
                pts = []
                for k in range(10):
                    angle = k * 36 - 90
                    rad = r if k % 2 == 0 else r * 0.45
                    pts.append((
                        x + rad * math.cos(angle * math.pi / 180),
                        y + rad * math.sin(angle * math.pi / 180),
                    ))
                draw.polygon(pts, fill=color, outline=(35, 35, 35))
            else:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color,
                             outline=(35, 35, 35), width=2)

            if obj.get("attr") == "cracked":
                draw.line([(x - 11, y - 13), (x + 4, y), (x - 7, y + 13)],
                          fill=(20, 20, 20), width=2)

    _draw_objects(before_objects, d)
    _draw_objects(after_objects, d)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ─── Experiment run logic ─────────────────────────────────────────────────
def generate_run(exp_type: str, layout: dict) -> tuple[list[dict], list[dict], str]:
    """Generate (before_objects, after_objects, change_description) for one run."""
    before = place_objects(layout, LEFT_ZONE_CENTERS)

    if exp_type == "move":
        after = apply_move(before, RIGHT_ZONE_CENTERS)
        desc = "one object moved to a different zone"
    elif exp_type == "color_change":
        after = apply_color_change(before)
        desc = "one object changed color"
    elif exp_type == "appear":
        after = apply_appear(before, RIGHT_ZONE_CENTERS)
        desc = "a new object appeared"
    elif exp_type == "disappear":
        after = apply_disappear(before)
        desc = "an object disappeared"
    elif exp_type == "no_change":
        # Deep copy before so zone centers are still on the right side
        after = deep_copy_objects(before)
        # Re-position with right-side zone centers but same layout
        after = place_objects(
            {o["id"]: o["zone"] for o in after},
            RIGHT_ZONE_CENTERS,
        )
        desc = "nothing changed"
    elif exp_type == "swap":
        after = apply_swap(before)
        desc = "two objects swapped colors"
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    return before, after, desc


# ─── Evaluation ───────────────────────────────────────────────────────────
def evaluate_object_identification(
    reported: list[dict],
    ground_truth: list[dict],
) -> dict:
    """Score object identification for one side.

    Returns dict with zone_accuracy, matches, total, hallucinations, misses.
    """
    matched_truth: set[int] = set()
    matched_obs: set[int] = set()
    zone_matches = 0
    zone_total = 0

    for oi, obs in enumerate(reported):
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

            # Color heuristic
            color_map = {
                "red": (210, 60, 60), "blue": (60, 90, 210),
                "green": (60, 180, 60), "yellow": (220, 200, 40),
                "orange": (230, 140, 40), "pink": (210, 120, 160),
                "purple": (140, 60, 180), "tan": (200, 175, 120),
                "brown": (200, 175, 120),
            }
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
            if obs_zone == ground_truth[best_gi]["zone"]:
                zone_matches += 1

    hallucinations = len(reported) - len(matched_obs)
    misses = len(ground_truth) - len(matched_truth)

    return {
        "zone_accuracy": zone_matches / max(zone_total, 1),
        "zone_matches": zone_matches,
        "zone_total": zone_total,
        "hallucinations": hallucinations,
        "misses": misses,
        "count_matched": len(matched_truth),
        "true_count": len(ground_truth),
    }


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side change detection experiment",
    )
    parser.add_argument("--runs", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="standard",
                        help="Label for this variation (e.g. no_border, with_grid)")
    args = parser.parse_args()

    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results = []
    errors = 0
    start_time = time.time()

    # Distribute runs evenly across experiment types
    n_types = len(EXPERIMENT_TYPES)
    runs_per_type = max(1, args.runs // n_types)
    total_planned = runs_per_type * n_types

    print(f"Running {total_planned} runs ({runs_per_type} per experiment type)")
    print(f"  Types: {', '.join(EXPERIMENT_TYPES)}")
    print(f"  Output: {output_path or '(stdout only)'}")
    print()

    # Per-type counters for cycling through layouts
    type_counters: dict[str, int] = {et: 0 for et in EXPERIMENT_TYPES}

    for run_num in range(1, total_planned + 1):
        exp_type = EXPERIMENT_TYPES[(run_num - 1) % n_types]
        type_idx = type_counters[exp_type]

        # Cycle through layouts deterministically per type
        layout = SCENE_LAYOUTS[type_idx % len(SCENE_LAYOUTS)]
        type_counters[exp_type] += 1

        before_objs, after_objs, expected_change = generate_run(exp_type, layout)

        image_b64 = render_side_by_side(before_objs, after_objs)

        # ── Query ─────────────────────────────────────────────────────
        prompt = (
            "Compare these two images. The LEFT image shows the workspace BEFORE "
            "and the RIGHT image shows the workspace AFTER. "
            "Describe what changed between them. Identify all objects in both images."
        )

        t0 = time.perf_counter()
        try:
            result = client.image_chat(
                prompt=prompt,
                image_b64=image_b64,
                system_prompt=SYSTEM_PROMPT,
                temperature=args.temperature,
                max_tokens=800,
                response_format={"type": "json_schema", "json_schema": CHANGE_DETECTION_SCHEMA},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = json.loads(result.content)

            # ── Evaluate ──────────────────────────────────────────────
            change_detected = parsed.get("change_detected", False)
            reported_type = parsed.get("change_type", "unknown")

            # Change detection accuracy
            change_correct = reported_type == exp_type

            # Object identification accuracy for each side
            left_reported = parsed.get("objects_in_left_image", [])
            right_reported = parsed.get("objects_in_right_image", [])

            left_eval = evaluate_object_identification(left_reported, before_objs)
            right_eval = evaluate_object_identification(right_reported, after_objs)

            # Is the change detection direction correct?
            # For move/color_change/appear/disappear/swap, change_detected should be True
            # For no_change, change_detected should be False
            if exp_type == "no_change":
                detection_correct = not change_detected
            else:
                detection_correct = change_detected

            entry = {
                "run": run_num,
                "experiment": "relational",
                "variation": args.variation,
                "exp_type": exp_type,
                "scene_layout_idx": SCENE_LAYOUTS.index(layout),
                "n_objects_before": len(before_objs),
                "n_objects_after": len(after_objs),
                "expected_change": expected_change,
                "reported_change_type": reported_type,
                "change_detected": change_detected,
                "change_correct": change_correct,
                "detection_correct": detection_correct,
                "description": parsed.get("description", ""),
                "left_zone_accuracy": round(left_eval["zone_accuracy"], 4),
                "left_zone_matches": left_eval["zone_matches"],
                "left_zone_total": left_eval["zone_total"],
                "left_hallucinations": left_eval["hallucinations"],
                "left_misses": left_eval["misses"],
                "right_zone_accuracy": round(right_eval["zone_accuracy"], 4),
                "right_zone_matches": right_eval["zone_matches"],
                "right_zone_total": right_eval["zone_total"],
                "right_hallucinations": right_eval["hallucinations"],
                "right_misses": right_eval["misses"],
                "latency_ms": round(latency_ms, 1),
                "temperature": args.temperature,
                "success": change_correct,
                "error": None,
                "raw_response": result.content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)

            # Periodic progress
            if run_num % 15 == 0 or run_num == 1:
                print(
                    f"  [{run_num:>4d}/{total_planned}] {exp_type:>14s} | "
                    f"change={'OK' if change_correct else 'XX'} | "
                    f"left_z={left_eval['zone_accuracy']:.0%} "
                    f"right_z={right_eval['zone_accuracy']:.0%} | "
                    f"{latency_ms:.0f}ms"
                )

        except Exception as e:
            errors += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            entry = {
                "run": run_num, "experiment": "relational",
                "variation": args.variation, "exp_type": exp_type,
                "n_objects_before": len(before_objs),
                "n_objects_after": len(after_objs),
                "error": str(e)[:300], "latency_ms": round(latency_ms, 1),
                "success": False, "raw_response": "",
                "temperature": args.temperature,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)
            print(f"  [{run_num:>4d}/{total_planned}] {exp_type:>14s} | ERROR: {str(e)[:80]}")
            if errors >= 20:
                print(f"  ❌ {errors} errors — aborting")
                break

        # Save partial results after each run
        if output_path and entry:
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s ({errors} errors)")

    # ─── Aggregate ────────────────────────────────────────────────────
    if not results:
        summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        valid = [r for r in results if r.get("success") is not None]

        per_type: dict[str, list[dict]] = defaultdict(list)
        for r in valid:
            per_type[r.get("exp_type", "unknown")].append(r)

        type_summaries = {}
        for et in EXPERIMENT_TYPES:
            entries = per_type.get(et, [])
            if not entries:
                type_summaries[et] = {"runs": 0, "change_accuracy": 0.0}
                continue
            change_correct = sum(1 for e in entries if e.get("change_correct"))
            detection_correct = sum(1 for e in entries if e.get("detection_correct"))
            left_accs = [e.get("left_zone_accuracy", 0) for e in entries]
            right_accs = [e.get("right_zone_accuracy", 0) for e in entries]

            # For no_change, we care about false positive rate
            false_positives = sum(
                1 for e in entries
                if e.get("exp_type") == "no_change" and e.get("change_detected", True)
            )

            type_summaries[et] = {
                "runs": len(entries),
                "change_accuracy": round(change_correct / len(entries), 4),
                "detection_accuracy": round(detection_correct / len(entries), 4),
                "mean_left_zone_accuracy": round(sum(left_accs) / len(left_accs), 4),
                "mean_right_zone_accuracy": round(sum(right_accs) / len(right_accs), 4),
                "false_positives": false_positives if et == "no_change" else None,
                "false_positive_rate": round(false_positives / len(entries), 4) if et == "no_change" else None,
            }

            # Sample raw responses for the report
            sample = None
            for e in entries:
                if e.get("raw_response") and sample is None:
                    sample = {"run": e["run"], "response": e["raw_response"]}
                    break
            type_summaries[et]["sample_raw"] = sample

        all_left = [r.get("left_zone_accuracy", 0) for r in valid]
        all_right = [r.get("right_zone_accuracy", 0) for r in valid]
        all_lat = [r.get("latency_ms", 0) for r in valid if "latency_ms" in r]
        sorted_lat = sorted(all_lat)

        overall_change_acc = sum(
            1 for r in valid if r.get("change_correct")
        ) / max(len(valid), 1)

        summary = {
            "runs": len(valid),
            "experiment": "relational",
            "variation": args.variation,
            "overall_change_accuracy": round(overall_change_acc, 4),
            "mean_left_zone_accuracy": round(sum(all_left) / len(all_left), 4) if all_left else 0,
            "mean_right_zone_accuracy": round(sum(all_right) / len(all_right), 4) if all_right else 0,
            "p50_latency_ms": round(sorted_lat[len(sorted_lat) // 2], 1) if sorted_lat else 0,
            "p95_latency_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 1) if sorted_lat else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(valid) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "per_experiment_type": type_summaries,
            "parameters": {
                "temperature": args.temperature,
                "runs_requested": args.runs,
            },
        }

        # Print a nice table
        print()
        print(f"{'Type':<16} {'Runs':>5} {'ChangeAcc':>10} {'DetectAcc':>10} {'LeftZ':>7} {'RightZ':>7} {'FP':>6}")
        print("-" * 65)
        for et in EXPERIMENT_TYPES:
            ts = type_summaries.get(et, {})
            fp_str = f"{ts.get('false_positive_rate', 0):.0%}" if et == "no_change" else ""
            print(
                f"{et:<16} {ts.get('runs', 0):>5} "
                f"{ts.get('change_accuracy', 0):>10.0%} "
                f"{ts.get('detection_accuracy', 0):>10.0%} "
                f"{ts.get('mean_left_zone_accuracy', 0):>7.0%} "
                f"{ts.get('mean_right_zone_accuracy', 0):>7.0%} "
                f"{fp_str:>6}"
            )
        print("-" * 65)
        print(f"{'OVERALL':<16} {summary['runs']:>5} {summary['overall_change_accuracy']:>10.0%}          {summary['mean_left_zone_accuracy']:>7.0%} {summary['mean_right_zone_accuracy']:>7.0%}")
        print()

    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
