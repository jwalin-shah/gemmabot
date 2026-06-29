"""Experiment: THE GAUNTLET — Combined Stress Test.

Combines ALL stressors simultaneously:
- No zone grid (no grid lines or labels)
- 7 objects per scene
- Mixed sizes (small, medium, large radii)
- Background variation (random background color)
- Multi-step planning instruction

This is the FINAL synthesis experiment to find the breaking point.

Usage:
    python scripts/exp_gauntlet.py --runs 100 \\
        --output overnight_results/gauntlet/r7_gauntlet.jsonl \\
        --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import math
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

# 7-object scene layouts (from high_count experiment)
HIGH_COUNT_SCENES = [
    {"red_cup": "A", "blue_cup": "B", "green_cup": "C", "cracked_cup": "D", "yellow_cube": "E", "orange_star": "F", "pink_cup": "A"},
    {"red_cup": "B", "blue_cup": "E", "green_cup": "F", "cracked_cup": "D", "yellow_cube": "C", "orange_star": "A", "purple_cube": "B"},
    {"red_cup": "D", "blue_cup": "E", "green_cup": "F", "cracked_cup": "C", "yellow_cube": "B", "orange_star": "A", "pink_cup": "F"},
    {"red_cup": "A", "blue_cup": "C", "green_cup": "F", "yellow_cube": "D", "orange_star": "B", "pink_cup": "E", "purple_cube": "A"},
    {"red_cup": "F", "blue_cup": "E", "green_cup": "D", "cracked_cup": "C", "yellow_cube": "B", "orange_star": "A", "pink_cup": "D"},
]

# Mixed radii (from mixed_sizes experiment)
RADII = [15, 20, 26, 35, 42]

def radius_to_size(r: int) -> str:
    if r <= 18:
        return "small"
    elif r <= 30:
        return "medium"
    else:
        return "large"

# Background colors (from background experiment)
BACKGROUND_COLORS = {
    "light_gray": (238, 238, 240),
    "white":      (255, 255, 255),
    "dark_gray":  (80, 80, 85),
    "wood":       (200, 170, 120),
    "black":      (30, 30, 32),
    "green_felt": (50, 120, 70),
}

# Multi-step instructions (from multistep experiment)
INSTRUCTIONS = [
    ("Plan the steps to move the red cup to Zone C and the blue cup to Zone A",
     "red_cup", "C", "blue_cup", "A"),
    ("Plan the steps to put the green cup in Zone F and the yellow cube in Zone B",
     "green_cup", "F", "yellow_cube", "B"),
    ("Plan how to pick up the blue cup and place it in Zone D, then move the red cup to Zone E",
     "blue_cup", "D", "red_cup", "E"),
    ("Plan how to move the yellow cube to Zone A and the green cup to Zone C",
     "yellow_cube", "A", "green_cup", "C"),
    ("Plan the steps to take the red cup to Zone B, then take the blue cup to Zone D",
     "red_cup", "B", "blue_cup", "D"),
    ("Plan how to put the blue cup in Zone F and move the cracked cup to Zone A",
     "blue_cup", "F", "cracked_cup", "A"),
    ("Plan how to move the orange star to Zone B and the pink cup to Zone D",
     "orange_star", "B", "pink_cup", "D"),
    ("Plan the steps to move the purple cube to Zone C and the green cup to Zone E",
     "purple_cube", "C", "green_cup", "E"),
]

# Combined schema: object identification + plan steps
GAUNTLET_SCHEMA = {
    "name": "plan_actions",
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
                        "current_zone": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]},
                        "shape": {"type": "string"},
                        "size": {"type": "string", "enum": ["small", "medium", "large"]},
                    },
                    "required": ["description", "color", "current_zone", "shape", "size"],
                    "additionalProperties": False,
                },
            },
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["pick", "place", "move"]},
                        "target_object": {"type": "string"},
                        "destination_zone": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F", "bin"]},
                        "why": {"type": "string"},
                    },
                    "required": ["action", "target_object", "destination_zone", "why"],
                    "additionalProperties": False,
                },
            },
            "total_objects_visible": {"type": "integer"},
            "reasoning": {"type": "string"},
        },
        "required": ["observed_objects", "plan", "total_objects_visible", "reasoning"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot planning system viewing a tabletop workspace.

Your workspace has 6 zones arranged in a 3x2 grid:
  A (top-left)   B (top-center)   C (top-right)
  D (bottom-left) E (bottom-center) F (bottom-right)

First, identify all objects visible in the image — their color, shape, size category, and current zone.
Then, based on the instruction, output a step-by-step plan.
Each step should be one action: pick an object, then place/move it to a destination zone.
Output the plan as an ordered list of steps. The order matters."""


def render_scene(objects: list[dict], bg_color: tuple = (238, 238, 240)) -> str:
    """Render scene with NO grid lines or labels (the gauntlet condition)."""
    from PIL import Image, ImageDraw
    import base64, io

    img = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
    d = ImageDraw.Draw(img)

    # NO grid rendering — deliberate omission

    for obj in objects:
        x, y = obj["x"], obj["y"]
        r = obj.get("radius", 26)
        color = obj["color"]
        shape = obj.get("shape", "round")

        # Adaptive outline based on background
        luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        outline = (180, 180, 180) if luminance < 60 else (35, 35, 35)

        if shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=2)
        elif shape == "star":
            pts = []
            for k in range(10):
                angle = k * 36 - 90
                rad = r if k % 2 == 0 else r * 0.45
                pts.append((x + rad * math.cos(angle * math.pi / 180),
                            y + rad * math.sin(angle * math.pi / 180)))
            d.polygon(pts, fill=color, outline=outline)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=outline, width=2)

        if obj.get("attr") == "cracked":
            d.line([(x - 11, y - 13), (x + 4, y), (x - 7, y + 13)], fill=(20, 20, 20), width=2)

    # Gripper bar at top center
    gx, gy = WIDTH // 2, 20
    gripper_lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    g_color = (180, 200, 240) if gripper_lum < 80 else (45, 120, 205)
    d.line([(gx - 16, gy), (gx + 16, gy)], fill=g_color, width=4)
    d.line([(gx, gy), (gx, gy - 24)], fill=g_color, width=4)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def place_objects(layout: dict) -> list[dict]:
    """Place objects with random sizes."""
    objects = []
    for oid, zone in layout.items():
        t = next(o for o in OBJECT_TEMPLATES if o["id"] == oid)
        zx, zy = ZONE_CENTERS[zone]
        radius = random.choice(RADII)
        objects.append({
            "id": oid,
            "x": zx + random.randint(-35, 35),
            "y": zy + random.randint(-25, 25),
            "color": t["color"],
            "radius": radius,
            "shape": t["shape"],
            "attr": t["attr"],
            "expected_zone": zone,
            "expected_size": radius_to_size(radius),
        })
    return objects


def evaluate_gauntlet(parsed: dict, ground_truth: list[dict],
                      instruction_info: tuple) -> dict:
    """Evaluate both zone accuracy and plan completion."""
    observed = parsed.get("observed_objects", [])
    plan = parsed.get("plan", [])
    reported_count = parsed.get("total_objects_visible", 0)

    # --- Zone accuracy ---
    zone_matches = 0
    zone_total = 0
    size_matches = 0
    size_total = 0
    matched_truth = set()
    matched_obs = set()

    for oi, obs in enumerate(observed):
        obs_zone = obs.get("current_zone", "")
        obs_size = obs.get("size", "")
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
            color_map = {"red": (210, 60, 60), "blue": (60, 90, 210), "green": (60, 180, 60),
                         "yellow": (220, 200, 40), "orange": (230, 140, 40), "pink": (210, 120, 160),
                         "purple": (140, 60, 180), "tan": (200, 175, 120), "brown": (200, 175, 120),
                         "gray": (120, 120, 120), "grey": (120, 120, 120)}
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
            if obs_size:
                size_total += 1
                if obs_size == ground_truth[best_gi]["expected_size"]:
                    size_matches += 1

    hallucinations = len(observed) - len(matched_obs)
    misses = len(ground_truth) - len(matched_truth)

    zone_accuracy = zone_matches / max(zone_total, 1)
    size_accuracy = size_matches / max(size_total, 1)

    # --- Plan accuracy ---
    _, exp_first_target, exp_first_zone, exp_second_target, exp_second_zone = instruction_info

    first_step_correct = False
    second_step_correct = False

    if plan and len(plan) >= 1:
        s1 = plan[0]
        s1_desc = (s1.get("target_object", "") + " " +
                   " ".join([s1.get("action", ""), s1.get("destination_zone", "")])).lower()
        exp1_words = exp_first_target.replace("_", " ").lower()
        if exp1_words.split()[0] in s1_desc and exp_first_zone.lower() in s1_desc:
            first_step_correct = True

    if exp_second_target and len(plan) >= 2:
        s2 = plan[1]
        s2_desc = (s2.get("target_object", "") + " " +
                   " ".join([s2.get("action", ""), s2.get("destination_zone", "")])).lower()
        exp2_words = exp_second_target.replace("_", " ").lower()
        if exp2_words.split()[0] in s2_desc and exp_second_zone.lower() in s2_desc:
            second_step_correct = True
    elif exp_second_target is None:
        second_step_correct = True

    plan_complete = first_step_correct and second_step_correct

    return {
        "zone_accuracy": zone_accuracy,
        "zone_matches": zone_matches,
        "zone_total": zone_total,
        "size_accuracy": size_accuracy,
        "size_matches": size_matches,
        "size_total": size_total,
        "hallucinations": hallucinations,
        "misses": misses,
        "count_error": abs(reported_count - len(ground_truth)),
        "reported_count": reported_count,
        "true_count": len(ground_truth),
        "first_step_correct": first_step_correct,
        "second_step_correct": second_step_correct,
        "plan_complete": plan_complete,
        "plan_length": len(plan),
    }


def main():
    parser = argparse.ArgumentParser(
        description="THE GAUNTLET — Combined stress test for Gemma 4 spatial reasoning"
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default="gauntlet",
                        help="Label for this variation")
    args = parser.parse_args()

    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results = []
    errors = 0
    start_time = time.time()

    n_scenes = len(HIGH_COUNT_SCENES)
    n_instrs = len(INSTRUCTIONS)
    reps = max(1, args.runs // n_scenes)

    bg_names = list(BACKGROUND_COLORS.keys())

    for si, layout in enumerate(HIGH_COUNT_SCENES):
        if si * reps >= args.runs:
            break
        for rep in range(reps):
            run_num = si * reps + rep + 1
            if run_num > args.runs:
                break

            # Random background color
            bg_name = random.choice(bg_names)
            bg_rgb = BACKGROUND_COLORS[bg_name]

            # Pick instruction for this run
            instr_idx = run_num % n_instrs
            instruction, *expected = INSTRUCTIONS[instr_idx]

            # Place 7 objects with mixed sizes
            objects = place_objects(layout)

            # Render without grid
            image_b64 = render_scene(objects, bg_color=bg_rgb)

            prompt = (
                f"Instruction: {instruction}\n\n"
                "Look at this camera image of a tabletop workspace. "
                "The workspace has 6 zones: A (top-left), B (top-center), C (top-right), "
                "D (bottom-left), E (bottom-center), F (bottom-right). "
                "Identify every object you can see by its color, shape, size (small/medium/large), "
                "and which zone it occupies. Then plan the steps needed to fulfill the instruction."
            )

            t0 = time.perf_counter()
            entry = None
            try:
                result = client.image_chat(
                    prompt=prompt,
                    image_b64=image_b64,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=args.temperature,
                    max_tokens=800,
                    response_format={"type": "json_schema", "json_schema": GAUNTLET_SCHEMA},
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                parsed = json.loads(result.content)
                score = evaluate_gauntlet(parsed, objects, INSTRUCTIONS[instr_idx])

                entry = {
                    "run": run_num,
                    "experiment": "gauntlet",
                    "variation": args.variation,
                    "scene": si,
                    "instruction": instruction,
                    "n_objects": len(objects),
                    "background": bg_name,
                    "background_rgb": str(bg_rgb),
                    "reported_count": score["reported_count"],
                    "count_error": score["count_error"],
                    "zone_accuracy": round(score["zone_accuracy"], 4),
                    "zone_matches": score["zone_matches"],
                    "zone_total": score["zone_total"],
                    "size_accuracy": round(score["size_accuracy"], 4),
                    "size_matches": score["size_matches"],
                    "size_total": score["size_total"],
                    "first_step_correct": score["first_step_correct"],
                    "second_step_correct": score["second_step_correct"],
                    "plan_complete": score["plan_complete"],
                    "plan_length": score["plan_length"],
                    "hallucinations": score["hallucinations"],
                    "misses": score["misses"],
                    "latency_ms": round(latency_ms, 1),
                    "temperature": args.temperature,
                    "show_grid": False,
                    "success": score["zone_accuracy"] >= 0.5 and score["count_error"] <= 1,
                    "info": "gauntlet_combined_stress",
                    "error": None,
                    "prompt_sent": prompt,
                    "raw_response": result.content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                results.append(entry)

                if run_num % 10 == 0 or run_num == 1:
                    print(f"  [{run_num:>4d}/{args.runs}] bg={bg_name:12s} | "
                          f"zone={score['zone_accuracy']:.0%} | size={score['size_accuracy']:.0%} | "
                          f"plan={score['plan_complete']} | lat={latency_ms:.0f}ms")

            except Exception as e:
                errors += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                entry = {
                    "run": run_num, "experiment": "gauntlet",
                    "variation": args.variation, "scene": si,
                    "instruction": instruction,
                    "background": bg_name, "background_rgb": str(bg_rgb),
                    "n_objects": len(objects),
                    "error": str(e)[:200], "latency_ms": round(latency_ms, 1),
                    "success": False,
                    "info": "gauntlet_combined_stress",
                    "prompt_sent": prompt, "raw_response": "",
                    "temperature": args.temperature, "show_grid": False,
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
        latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
        accs = [r["zone_accuracy"] for r in results if "zone_accuracy" in r]
        size_accs = [r["size_accuracy"] for r in results if "size_accuracy" in r]
        halls = [r["hallucinations"] for r in results if "hallucinations" in r]
        misses = [r["misses"] for r in results if "misses" in r]
        plans_complete = [r for r in results if r.get("plan_complete")]
        first_step_oks = [r for r in results if r.get("first_step_correct")]

        # Per-background breakdown
        bg_stats = defaultdict(lambda: {"runs": 0, "accs": [], "plans": 0})
        for r in results:
            bg = r.get("background", "unknown")
            bg_stats[bg]["runs"] += 1
            if "zone_accuracy" in r:
                bg_stats[bg]["accs"].append(r["zone_accuracy"])
            if r.get("plan_complete"):
                bg_stats[bg]["plans"] += 1

        per_background = {}
        for bg_name in bg_names:
            s = bg_stats[bg_name]
            n = len(s["accs"])
            per_background[bg_name] = {
                "runs": s["runs"],
                "mean_zone_accuracy": round(sum(s["accs"]) / n, 4) if n > 0 else 0,
                "plan_complete_rate": round(s["plans"] / max(s["runs"], 1), 4),
            }

        result_summary = {
            "runs": len(results),
            "experiment": "gauntlet",
            "variation": args.variation,
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 4) if results else 0,
            "mean_zone_accuracy": round(sum(accs) / len(accs), 4) if accs else 0,
            "mean_size_accuracy": round(sum(size_accs) / len(size_accs), 4) if size_accs else 0,
            "plan_complete_rate": round(len(plans_complete) / len(results), 4) if results else 0,
            "first_step_accuracy": round(len(first_step_oks) / len(results), 4) if results else 0,
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else 0,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0,
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 1) if latencies else 0,
            "mean_hallucinations": round(sum(halls) / len(halls), 2) if halls else 0,
            "mean_misses": round(sum(misses) / len(misses), 2) if misses else 0,
            "mean_count_error": round(
                sum(r.get("count_error", 0) for r in results if "count_error" in r) /
                max(len([r for r in results if "count_error" in r]), 1), 2
            ),
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "runs_per_minute": round(len(results) / (elapsed / 60), 1) if elapsed > 0 else 0,
            "completed": errors < 20,
            "per_background": per_background,
            "stressor_summary": {
                "no_grid": True,
                "n_objects": 7,
                "mixed_radii": True,
                "radii_used": RADII,
                "background_variation": True,
                "n_backgrounds": len(bg_names),
                "multi_step_instruction": True,
                "n_instructions": n_instrs,
            },
            "benchmarks": {
                "baseline_no_grid": 99.83,
                "baseline_monochrome": 100.0,
                "baseline_7objects": 99.24,
                "baseline_multistep_plan": 100.0,
                "gauntlet_zone_accuracy": round(sum(accs) / len(accs) * 100, 2) if accs else 0,
                "gauntlet_plan_rate": round(len(plans_complete) / len(results) * 100, 2) if results else 0,
                "zone_vs_baseline_no_grid_drop": round(99.83 - (sum(accs) / len(accs) * 100), 2) if accs else None,
                "plan_vs_baseline_drop": round(100.0 - (len(plans_complete) / len(results) * 100), 2) if results else None,
            },
            "parameters": {
                "temperature": args.temperature,
                "show_grid": False,
                "object_count": "7",
            },
        }

    print(f"RESULT:{json.dumps(result_summary)}")


if __name__ == "__main__":
    main()
