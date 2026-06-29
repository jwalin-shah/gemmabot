"""Experiment: Proximity Detection — Objects at Varying Distances.

Tests how Gemma 4 handles objects placed DIRECTLY NEXT TO each other
(touching but not overlapping), and at various distances.

Distance levels: 0px (touching), 5px, 15px, 30px, 60px (baseline)
3 visual combos per distance: same-color-diff-shape, diff-color-same-shape, diff-color-diff-shape
10 trials per cell => 150 runs total (5 distances x 3 combos x 10 reps)

Questions:
  - At what distance does detection of "two objects" break?
  - Does the model merge touching objects?
  - Does same-color touching cause more confusion than different-color?
  - Zone accuracy at each distance

Usage:
    python scripts/exp_proximity.py --runs 150 --output overnight_results/proximity/r8_proximity.jsonl --temperature 0.0
"""

from __future__ import annotations

import argparse
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
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2

ZONE_CENTERS: dict[str, tuple[int, int]] = {}
for i, lab in enumerate(ZONE_LABELS):
    r, c_idx = divmod(i, 3)
    ZONE_CENTERS[lab] = (int(c_idx * cw + cw / 2), int(r * ch + ch / 2))

# Object radius (matching other experiments)
RADIUS = 26
DIAMETER = RADIUS * 2  # 52

# ─── Distance levels (gap in pixels between object edges) ────────────────
DISTANCE_LEVELS = [0, 5, 15, 30, 60]

# ─── Visual combos ───────────────────────────────────────────────────────
# Each combo: (label, obj1_color, obj1_shape, obj2_color, obj2_shape)
# Same base: red round object is always object 1
VISUAL_COMBOS = [
    # (name, obj1_color, obj1_shape, obj2_color, obj2_shape)
    ("same_color_diff_shape", (210, 60, 60), "round", (210, 60, 60), "square"),
    ("diff_color_same_shape", (210, 60, 60), "round", (60, 90, 210), "round"),
    ("diff_color_diff_shape", (210, 60, 60), "round", (60, 90, 210), "square"),
]

COLOR_NAMES = {
    (210, 60, 60): "red",
    (60, 90, 210): "blue",
}

SHAPE_NAMES = {
    "round": "round",
    "square": "square",
    "star": "star",
}

# ─── Schema ──────────────────────────────────────────────────────────────
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
            "are_objects_touching_or_overlapping": {"type": "boolean"},
        },
        "required": [
            "observed_objects",
            "total_objects_visible",
            "are_objects_touching_or_overlapping",
        ],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are a robot vision system analyzing a tabletop workspace with a Zone A-F grid overlaid. "
    "Identify every object: its color, shape, and zone. "
    "Check carefully if any objects are touching or overlapping — "
    "objects that are touching (edges just meet) or overlapping should be reported as touching_or_overlapping."
)

# ─── Rendering ────────────────────────────────────────────────────────────
def render_scene(
    obj1: dict,
    obj2: dict,
    zone_label: str,
) -> str:
    """Render a 384x384 scene with two objects in the same zone at a given distance."""
    from PIL import Image, ImageDraw
    import base64, io

    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    # Draw zone grid
    for c in range(1, 3):
        d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)],
               fill=(200, 200, 210), width=1)
    d.line([(0, int(ch)), (WIDTH, int(ch))],
           fill=(200, 200, 210), width=1)
    for i, lab in enumerate(ZONE_LABELS):
        r_idx, c_idx = divmod(i, 3)
        d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)),
               f"Zone {lab}", fill=(170, 170, 180))

    # Draw objects
    for obj_data in [obj1, obj2]:
        x, y, r = obj_data["x"], obj_data["y"], RADIUS
        color = obj_data["color"]
        shape = obj_data["shape"]

        if shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r],
                        fill=color, outline=(35, 35, 35), width=2)
        elif shape == "star":
            pts = []
            for k in range(10):
                angle = k * 36 - 90
                rad = r if k % 2 == 0 else r * 0.45
                pts.append((
                    x + rad * math.cos(angle * math.pi / 180),
                    y + rad * math.sin(angle * math.pi / 180),
                ))
            d.polygon(pts, fill=color, outline=(35, 35, 35))
        else:
            d.ellipse([x - r, y - r, x + r, y + r],
                      fill=color, outline=(35, 35, 35), width=2)

    # Draw gripper (always present, standard position)
    gx, gy = WIDTH // 2, 20
    d.line([(gx - 16, gy), (gx + 16, gy)], fill=(45, 120, 205), width=4)
    d.line([(gx, gy), (gx, gy - 24)], fill=(45, 120, 205), width=4)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def generate_objects(
    zone: str,
    combo_label: str,
    distance_px: int,
    vertical_jitter: int = 8,
) -> tuple[dict, dict, str]:
    """Generate two objects in the same zone at a given edge-to-edge distance.

    The objects are placed side by side horizontally within the zone.
    distance_px = gap between object edges (0 = touching, >0 = separated).

    Returns (obj1_dict, obj2_dict, combo_name).
    """
    zx, zy = ZONE_CENTERS[zone]

    # Center of the zone is the midpoint between the two objects
    # Total width from left edge of left object to right edge of right object
    # = DIAMETER + distance_px (gap), measured from centers = DIAMETER + distance_px
    center_separation = DIAMETER + distance_px  # distance between centers

    # Jitter vertically slightly (same for both so they stay aligned)
    jy = random.randint(-vertical_jitter, vertical_jitter)

    # Find the combo config
    combo_info = None
    for c in VISUAL_COMBOS:
        if c[0] == combo_label:
            combo_info = c
            break
    assert combo_info is not None, f"Unknown combo: {combo_label}"
    _, c1_color, c1_shape, c2_color, c2_shape = combo_info

    obj1 = {
        "x": zx - center_separation // 2,
        "y": zy + jy,
        "color": c1_color,
        "shape": c1_shape,
        "expected_color": COLOR_NAMES.get(c1_color, "unknown"),
        "expected_shape": c1_shape,
        "expected_zone": zone,
    }
    obj2 = {
        "x": zx + center_separation // 2,
        "y": zy + jy,
        "color": c2_color,
        "shape": c2_shape,
        "expected_color": COLOR_NAMES.get(c2_color, "unknown"),
        "expected_shape": c2_shape,
        "expected_zone": zone,
    }

    return obj1, obj2, combo_label


def choose_random_zone() -> str:
    """Pick a random zone, avoiding edge zones to prevent objects crossing grid lines."""
    # Use all zones—objects are small enough to stay within zone bounds
    return random.choice(ZONE_LABELS)


# ─── Evaluation ──────────────────────────────────────────────────────────
def evaluate(parsed: dict, obj1: dict, obj2: dict, distance_px: int, combo_label: str) -> dict:
    """Score a model response against ground truth."""
    observed = parsed.get("observed_objects", [])
    reported_count = parsed.get("total_objects_visible", 0)
    reported_touching = parsed.get("are_objects_touching_or_overlapping", False)

    # Ground truth: 0px gap = touching, everything else = not touching
    actually_touching = distance_px == 0

    # Count accuracy
    count_correct = reported_count == 2

    # Touching detection accuracy
    touching_correct = reported_touching == actually_touching

    # Zone accuracy for each reported object
    # We need to match observed objects to ground truth
    matched_truth: set[int] = set()
    matched_obs: set[int] = set()
    zone_matches = 0
    zone_total = 0

    ground_truth = [obj1, obj2]
    gt_ids = ["obj1", "obj2"]
    gt_colors = [obj1["expected_color"], obj2["expected_color"]]
    gt_shapes = [obj1["expected_shape"], obj2["expected_shape"]]

    for oi, obs in enumerate(observed):
        obs_zone = obs.get("zone", "")
        obs_color = obs.get("color", "").lower().strip()
        obs_shape = obs.get("shape", "").lower().strip()

        best_score = 0
        best_gi = None
        for gi in range(2):
            if gi in matched_truth:
                continue
            score = 0
            # Color match
            if obs_color == gt_colors[gi]:
                score += 1
            # Shape match
            if obs_shape == gt_shapes[gi]:
                score += 1
            if score > best_score:
                best_score = score
                best_gi = gi

        if best_gi is not None and best_score > 0:
            matched_truth.add(best_gi)
            matched_obs.add(oi)
            zone_total += 1
            if obs_zone == ground_truth[best_gi]["expected_zone"]:
                zone_matches += 1

    hallucinations = len(observed) - len(matched_obs)
    misses = len(ground_truth) - len(matched_truth)

    # Did the model describe both distinct objects or merge them?
    # Merged = reported 1 object OR reported 2 objects but they have identical descriptions
    merged = False
    if reported_count == 1:
        merged = True
    elif len(observed) == 2:
        # Check if both observed objects have the same color/shape description
        if (observed[0].get("color", "").lower().strip() ==
                observed[1].get("color", "").lower().strip() and
            observed[0].get("shape", "").lower().strip() ==
                observed[1].get("shape", "").lower().strip()):
            merged = True

    return {
        "count_correct": count_correct,
        "touching_correct": touching_correct,
        "reported_touching": reported_touching,
        "actually_touching": actually_touching,
        "zone_accuracy": zone_matches / max(zone_total, 1),
        "zone_matches": zone_matches,
        "zone_total": zone_total,
        "hallucinations": hallucinations,
        "misses": misses,
        "merged": merged,
        "reported_count": reported_count,
        "true_count": 2,
    }


# ─── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proximity detection experiment — objects at varying distances",
    )
    parser.add_argument("--runs", type=int, default=150,
                        help="Total runs (default 150 = 5 distances x 3 combos x 10 reps)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="",
                        help="Path to output JSONL file")
    parser.add_argument("--variation", type=str, default="standard",
                        help="Label for this variation")
    parser.add_argument("--no-grid", action="store_true", dest="no_grid",
                        help="Skip grid overlay")
    args = parser.parse_args()

    show_grid = not args.no_grid
    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results: list[dict] = []
    errors = 0
    start_time = time.time()

    # Build trial plan: 5 distances x 3 combos x N_reps
    # We want exactly args.runs total, evenly distributed
    n_distances = len(DISTANCE_LEVELS)
    n_combos = len(VISUAL_COMBOS)
    total_cells = n_distances * n_combos
    reps_per_cell = max(1, args.runs // total_cells)
    total_planned = total_cells * reps_per_cell

    combo_labels = [c[0] for c in VISUAL_COMBOS]

    print(f"Proximity Experiment")
    print(f"  {total_planned} runs ({reps_per_cell} reps x {n_distances} distances x {n_combos} combos)")
    print(f"  Distances (gap px): {DISTANCE_LEVELS}")
    print(f"  Combos: {combo_labels}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output: {output_path or '(stdout only)'}")
    print()

    run_num = 0
    for di, distance in enumerate(DISTANCE_LEVELS):
        for ci, combo_label in enumerate(combo_labels):
            for rep in range(reps_per_cell):
                run_num += 1
                if run_num > args.runs:
                    break

                zone = choose_random_zone()
                obj1, obj2, _ = generate_objects(zone, combo_label, distance)

                image_b64 = render_scene(obj1, obj2, zone)

                prompt = (
                    "Look at this camera image of a tabletop workspace with a Zone A-F grid. "
                    "Identify every object you can see by its color, shape, and which zone it occupies. "
                    "Check carefully: are the objects touching or overlapping, or are they separated?"
                )

                t0 = time.perf_counter()
                try:
                    result = client.image_chat(
                        prompt=prompt,
                        image_b64=image_b64,
                        system_prompt=SYSTEM_PROMPT,
                        temperature=args.temperature,
                        max_tokens=600,
                        response_format={
                            "type": "json_schema",
                            "json_schema": IDENTIFY_SCHEMA,
                        },
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    parsed = json.loads(result.content)
                    score = evaluate(parsed, obj1, obj2, distance, combo_label)

                    entry: dict = {
                        "run": run_num,
                        "experiment": "proximity",
                        "variation": args.variation,
                        "distance_px": distance,
                        "combo": combo_label,
                        "zone": zone,
                        "n_expected": 2,
                        "n_reported": score["reported_count"],
                        "count_correct": score["count_correct"],
                        "touching_correct": score["touching_correct"],
                        "reported_touching": score["reported_touching"],
                        "actually_touching": score["actually_touching"],
                        "zone_accuracy": round(score["zone_accuracy"], 4),
                        "zone_matches": score["zone_matches"],
                        "zone_total": score["zone_total"],
                        "hallucinations": score["hallucinations"],
                        "misses": score["misses"],
                        "merged": score["merged"],
                        "latency_ms": round(latency_ms, 1),
                        "temperature": args.temperature,
                        "show_grid": show_grid,
                        "success": score["count_correct"] and score["zone_accuracy"] >= 0.5,
                        "error": None,
                        "raw_response": result.content,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    results.append(entry)

                    if run_num % 15 == 0 or run_num == 1:
                        touch_str = "T" if score["reported_touching"] else "."
                        print(
                            f"  [{run_num:>4d}/{total_planned}] "
                            f"gap={distance:>2d}px "
                            f"{combo_label:<24s} "
                            f"zone={zone} "
                            f"count={'OK' if score['count_correct'] else str(score['reported_count'])} "
                            f"touch={touch_str} "
                            f"zone_acc={score['zone_accuracy']:.0%} "
                            f"{latency_ms:.0f}ms"
                        )

                except Exception as e:
                    errors += 1
                    latency_ms = (time.perf_counter() - t0) * 1000
                    entry = {
                        "run": run_num,
                        "experiment": "proximity",
                        "variation": args.variation,
                        "distance_px": distance,
                        "combo": combo_label,
                        "zone": zone,
                        "error": str(e)[:300],
                        "latency_ms": round(latency_ms, 1),
                        "success": False,
                        "raw_response": "",
                        "temperature": args.temperature,
                        "show_grid": show_grid,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    results.append(entry)
                    print(f"  [{run_num:>4d}/{total_planned}] ERROR: {str(e)[:80]}")
                    if errors >= 20:
                        print(f"  D: {errors} errors — aborting")
                        break

                # Save after each run
                if output_path and entry:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")

        if errors >= 20:
            break

    elapsed = time.time() - start_time

    # ── Aggregate ────────────────────────────────────────────────────
    if not results:
        summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        valid = [r for r in results if r.get("success") is not None]

        # Per-distance aggregation
        per_distance: dict[int, list[dict]] = defaultdict(list)
        per_combo: dict[str, list[dict]] = defaultdict(list)
        per_cell: dict[str, list[dict]] = defaultdict(list)

        for r in valid:
            per_distance[r["distance_px"]].append(r)
            per_combo[r["combo"]].append(r)
            cell_key = f"{r['distance_px']}px_{r['combo']}"
            per_cell[cell_key].append(r)

        dist_summaries = {}
        for d in DISTANCE_LEVELS:
            entries = per_distance.get(d, [])
            if not entries:
                continue
            count_acc = sum(1 for e in entries if e["count_correct"]) / len(entries)
            touch_acc = sum(1 for e in entries if e["touching_correct"]) / len(entries)
            merged_rate = sum(1 for e in entries if e["merged"]) / len(entries)
            zone_accs = [e["zone_accuracy"] for e in entries]
            mean_zone = sum(zone_accs) / len(zone_accs)
            reported_touching_rate = sum(
                1 for e in entries if e["reported_touching"]
            ) / max(len(entries), 1)

            dist_summaries[f"{d}px"] = {
                "runs": len(entries),
                "count_accuracy": round(count_acc, 4),
                "touching_detection_accuracy": round(touch_acc, 4),
                "merged_rate": round(merged_rate, 4),
                "reported_touching_rate": round(reported_touching_rate, 4),
                "mean_zone_accuracy": round(mean_zone, 4),
            }

        combo_summaries = {}
        for c in combo_labels:
            entries = per_combo.get(c, [])
            if not entries:
                continue
            count_acc = sum(1 for e in entries if e["count_correct"]) / len(entries)
            touch_acc = sum(1 for e in entries if e["touching_correct"]) / len(entries)
            merged_rate = sum(1 for e in entries if e["merged"]) / len(entries)
            zone_accs = [e["zone_accuracy"] for e in entries]
            mean_zone = sum(zone_accs) / len(zone_accs)

            combo_summaries[c] = {
                "runs": len(entries),
                "count_accuracy": round(count_acc, 4),
                "touching_detection_accuracy": round(touch_acc, 4),
                "merged_rate": round(merged_rate, 4),
                "mean_zone_accuracy": round(mean_zone, 4),
            }

        # Overall
        all_count = sum(1 for r in valid if r["count_correct"]) / max(len(valid), 1)
        all_touch = sum(1 for r in valid if r["touching_correct"]) / max(len(valid), 1)
        all_merged = sum(1 for r in valid if r["merged"]) / max(len(valid), 1)
        all_zone_accs = [r["zone_accuracy"] for r in valid]
        all_lat = [r["latency_ms"] for r in valid if "latency_ms" in r]
        sorted_lat = sorted(all_lat)

        summary = {
            "runs": len(valid),
            "experiment": "proximity",
            "variation": args.variation,
            "overall_count_accuracy": round(all_count, 4),
            "overall_touching_detection_accuracy": round(all_touch, 4),
            "overall_merged_rate": round(all_merged, 4),
            "overall_mean_zone_accuracy": round(
                sum(all_zone_accs) / len(all_zone_accs), 4
            ) if all_zone_accs else 0,
            "p50_latency_ms": round(sorted_lat[len(sorted_lat) // 2], 1) if sorted_lat else 0,
            "p95_latency_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 1) if sorted_lat else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(
                len(valid) / (elapsed / 60), 1
            ) if elapsed > 0 else 0,
            "completed": errors < 20,
            "distances": dist_summaries,
            "combos": combo_summaries,
            "parameters": {
                "temperature": args.temperature,
                "show_grid": show_grid,
                "runs_requested": args.runs,
            },
        }

        # Print results table
        print()
        print("=" * 70)
        print("PROXIMITY EXPERIMENT RESULTS")
        print("=" * 70)
        print()
        print("By Distance:")
        print(f"{'Gap':>6} {'Runs':>6} {'CountAcc':>9} {'TouchDetect':>11} {'Merged':>7} {'ZoneAcc':>8} {'TouchRate':>10}")
        print("-" * 60)
        for d in DISTANCE_LEVELS:
            ds = dist_summaries.get(f"{d}px", {})
            print(
                f"{d:>4}px {ds.get('runs', 0):>6} "
                f"{ds.get('count_accuracy', 0):>9.0%} "
                f"{ds.get('touching_detection_accuracy', 0):>11.0%} "
                f"{ds.get('merged_rate', 0):>7.0%} "
                f"{ds.get('mean_zone_accuracy', 0):>8.0%} "
                f"{ds.get('reported_touching_rate', 0):>10.0%}"
            )
        print()
        print("By Visual Combo:")
        print(f"{'Combo':<26} {'Runs':>6} {'CountAcc':>9} {'TouchDetect':>11} {'Merged':>7} {'ZoneAcc':>8}")
        print("-" * 70)
        for c in combo_labels:
            cs = combo_summaries.get(c, {})
            print(
                f"{c:<26} {cs.get('runs', 0):>6} "
                f"{cs.get('count_accuracy', 0):>9.0%} "
                f"{cs.get('touching_detection_accuracy', 0):>11.0%} "
                f"{cs.get('merged_rate', 0):>7.0%} "
                f"{cs.get('mean_zone_accuracy', 0):>8.0%}"
            )
        print()

    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
