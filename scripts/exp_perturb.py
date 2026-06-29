"""Experiment: Perturbation Resilience — object moves, does Gemma re-acquire?

Simulates 10 ticks of a pick-and-place. On tick 5, an object moves zones.
Measures how many ticks it takes Gemma to re-identify the new zone.

Usage:
    python scripts/exp_perturb.py --runs 50 --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
                    },
                    "required": ["description", "color", "zone"],
                    "additionalProperties": False,
                },
            },
            "total_objects_visible": {"type": "integer"},
        },
        "required": ["observed_objects", "total_objects_visible"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system. You see a tabletop workspace with a Zone A-F grid.
Identify every object by color and which zone (A-F) it occupies."""


def render(objects, gripper_y=20):
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)
    for c in range(1, 3):
        d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=(200, 200, 210), width=1)
    d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)
    for i, lab in enumerate(ZONE_LABELS):
        r_idx, c_idx = divmod(i, 3)
        d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)), f"Zone {lab}", fill=(170, 170, 180))
    for obj in objects:
        x, y = obj["x"], obj["y"]
        r = 26
        d.ellipse([x - r, y - r, x + r, y + r], fill=obj["color"], outline=(35, 35, 35), width=2)
    d.line([(WIDTH // 2 - 16, gripper_y), (WIDTH // 2 + 16, gripper_y)], fill=(45, 120, 205), width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


SCENES = [
    {"red_cup": {"zone": "D", "color": (210, 60, 60)}, "blue_cup": {"zone": "E", "color": (60, 90, 210)}},
    {"red_cup": {"zone": "A", "color": (210, 60, 60)}, "green_cup": {"zone": "C", "color": (60, 180, 60)}},
    {"blue_cup": {"zone": "F", "color": (60, 90, 210)}, "yellow_cube": {"zone": "B", "color": (220, 200, 40)}},
    {"red_cup": {"zone": "E", "color": (210, 60, 60)}, "cracked_cup": {"zone": "D", "color": (200, 175, 120)}},
]

PERTURBATIONS = [
    {"target": "red_cup", "from_zone": "D", "to_zone": "A"},
    {"target": "red_cup", "from_zone": "A", "to_zone": "F"},
    {"target": "blue_cup", "from_zone": "E", "to_zone": "B"},
    {"target": "blue_cup", "from_zone": "F", "to_zone": "C"},
    {"target": "green_cup", "from_zone": "C", "to_zone": "E"},
]


def run_sequence(scene_objects, perturb_at_tick=5, n_ticks=10, perturb_idx=0):
    """Run N ticks of visual identification, perturbing at tick 5."""
    results = []
    for tick in range(n_ticks):
        # Apply perturbation at specified tick
        if tick == perturb_at_tick:
            perturb = PERTURBATIONS[perturb_idx % len(PERTURBATIONS)]
            for oid, odata in scene_objects.items():
                if oid == perturb["target"]:
                    old_zone = odata["zone"]
                    odata["zone"] = perturb["to_zone"]
                    odata["x"] = ZONE_CENTERS[perturb["to_zone"]][0] + (hash(str(tick)) % 30 - 15)
                    odata["y"] = ZONE_CENTERS[perturb["to_zone"]][1] + (hash(str(tick+1)) % 20 - 10)

        # Convert scene_objects to render list
        render_objects = []
        for oid, odata in scene_objects.items():
            zx, zy = ZONE_CENTERS[odata["zone"]]
            render_objects.append({
                "id": oid, "x": zx, "y": zy,
                "color": odata["color"], "zone": odata["zone"],
            })

        image_b64 = render(render_objects, gripper_y=20 + tick * 3)
        yield tick, render_objects, image_b64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    all_results = []
    errors = 0
    start = time.time()

    for run_num in range(1, args.runs + 1):
        scene = SCENES[run_num % len(SCENES)]
        scene_objects = {}
        for oid, odata in scene.items():
            zx, zy = ZONE_CENTERS[odata["zone"]]
            scene_objects[oid] = {
                "zone": odata["zone"],
                "color": odata["color"],
                "x": zx,
                "y": zy,
            }

        sequence = list(run_sequence(scene_objects, perturb_at_tick=5, n_ticks=10, perturb_idx=run_num % len(PERTURBATIONS)))
        detection_delay = None
        pre_correct = 0
        post_correct = 0
        tick_results = []

        for tick, render_objects, image_b64 in sequence:
            t0 = time.perf_counter()
            try:
                result = client.image_chat(
                    prompt=f"Tick {tick}. Identify every object and its zone.",
                    image_b64=image_b64,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=400,
                    response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA},
                )
                latency = (time.perf_counter() - t0) * 1000
                parsed = json.loads(result.content)
                observed = parsed.get("observed_objects", [])

                # Check if perturbation target is correctly zoned
                perturb_target = PERTURBATIONS[run_num % len(PERTURBATIONS)]["target"]
                expected_zone_after = PERTURBATIONS[run_num % len(PERTURBATIONS)]["to_zone"]
                correct = False
                for obs in observed:
                    desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()
                    target_words = perturb_target.replace("_", " ").lower()
                    if target_words.split()[0] in desc:
                        if obs.get("zone", "") == expected_zone_after:
                            correct = True
                            if detection_delay is None and tick >= 5:
                                detection_delay = tick - 5

                if tick < 5:
                    pre_correct += int(correct)
                else:
                    post_correct += int(correct)

                tick_results.append({
                    "tick": tick, "latency_ms": round(latency, 1),
                    "correct": correct,
                })

            except Exception as e:
                errors += 1
                tick_results.append({"tick": tick, "error": str(e)[:100], "correct": False})
                if errors > 20:
                    break

        entry = {
            "run": run_num,
            "pre_perturb_accuracy": pre_correct / 5,
            "post_perturb_accuracy": post_correct / 5,
            "detection_delay": detection_delay,
            "ticks": tick_results,
        }
        all_results.append(entry)

        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(entry) + "\n")

        if run_num % 10 == 0:
            print(f"  [{run_num:>4d}/{args.runs}] delay={detection_delay}")

    delay_values = [r["detection_delay"] for r in all_results if r["detection_delay"] is not None]
    summary = {
        "runs": len(all_results),
        "mean_pre_accuracy": round(sum(r["pre_perturb_accuracy"] for r in all_results) / len(all_results), 4),
        "mean_post_accuracy": round(sum(r["post_perturb_accuracy"] for r in all_results) / len(all_results), 4),
        "mean_detection_delay": round(sum(delay_values) / len(delay_values), 1) if delay_values else None,
        "p50_detection_delay": round(sorted(delay_values)[len(delay_values)//2], 1) if delay_values else None,
        "completed": errors < 20,
        "elapsed_s": round(time.time() - start, 1),
    }
    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
