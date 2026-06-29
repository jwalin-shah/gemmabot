"""Experiment: Multi-step Reasoning -- can Gemma plan sequences from one image?

AUDIT NOTE: The old evaluate_plan measured INSTRUCTION-PARSING, not vision.
first_ok passed if the plan step 0 contained the colour word and destination
zone -- both copied verbatim from the instruction the model just read.
A model that ignores the image scores 100%.

This rewrite separates three honest signals:
  1. visual_grounding_accuracy  -- image-dependent: did the model correctly
     report each object's CURRENT zone?  Matched by COLOR word (visible in
     image -> valid matching key, not present in the prompt text).
  2. plan_ordering_accuracy     -- instruction-parsing only: did the model
     produce the right object->destination ordering?  Does NOT require image.
  3. plan_complete              -- BOTH signals correct for the targeted
     objects.  The headline metric that actually depends on the image.

Additional fix: build_objects now jitters positions (previously exact zone
centers -> bit-identical images at temp=0).  --seed and random scene+
instruction selection give real N unique combos (old code: only 12).

Usage:
    python scripts/exp_multistep.py --runs 120 --seed 1 \\
        --output overnight_results/multistep/clean_multistep.jsonl
"""

from __future__ import annotations

import argparse
import json
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
    {"id": "red_cup",     "color": (210, 60, 60),   "shape": "round",  "color_name": "red"},
    {"id": "blue_cup",    "color": (60, 90, 210),   "shape": "round",  "color_name": "blue"},
    {"id": "green_cup",   "color": (60, 180, 60),   "shape": "round",  "color_name": "green"},
    {"id": "yellow_cube", "color": (220, 200, 40),  "shape": "square", "color_name": "yellow"},
]

INSTRUCTIONS = [
    # (instruction, exp_first_target, exp_first_zone, exp_second_target, exp_second_zone)
    ("Move the red cup to Zone C and the blue cup to Zone A",
     "red_cup", "C", "blue_cup", "A"),
    ("Put the green cup in Zone F and the yellow cube in Zone B",
     "green_cup", "F", "yellow_cube", "B"),
    ("Pick up the blue cup and place it in Zone D, then move the red cup to Zone E",
     "blue_cup", "D", "red_cup", "E"),
    ("Move the yellow cube to Zone A and the green cup to Zone C",
     "yellow_cube", "A", "green_cup", "C"),
    ("Take the red cup to Zone B, then take the blue cup to Zone D",
     "red_cup", "B", "blue_cup", "D"),
    ("Put the blue cup in Zone F", "blue_cup", "F", None, None),
    ("Move the green cup to Zone A and the red cup to Zone F",
     "green_cup", "A", "red_cup", "F"),
    ("Place the yellow cube in Zone E, then the green cup in Zone B",
     "yellow_cube", "E", "green_cup", "B"),
]

SCENE_SETUPS = [
    {"red_cup": "D", "blue_cup": "E", "green_cup": "F", "yellow_cube": "A"},
    {"red_cup": "A", "blue_cup": "C", "green_cup": "E", "yellow_cube": "B"},
    {"red_cup": "B", "blue_cup": "F", "green_cup": "D", "yellow_cube": "C"},
    {"red_cup": "E", "blue_cup": "A", "green_cup": "C", "yellow_cube": "D"},
    {"red_cup": "C", "blue_cup": "D", "green_cup": "A", "yellow_cube": "F"},
    {"red_cup": "F", "blue_cup": "B", "green_cup": "E", "yellow_cube": "D"},
    {"red_cup": "D", "blue_cup": "F", "green_cup": "B", "yellow_cube": "E"},
    {"red_cup": "A", "blue_cup": "E", "green_cup": "C", "yellow_cube": "F"},
]

PLAN_SCHEMA = {
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
                        "current_zone": {"type": "string",
                                         "enum": ["A", "B", "C", "D", "E", "F"]},
                    },
                    "required": ["description", "color", "current_zone"],
                    "additionalProperties": False,
                },
            },
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string",
                                   "enum": ["pick", "place", "move"]},
                        "target_object": {"type": "string"},
                        "destination_zone": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E", "F", "bin"],
                        },
                        "why": {"type": "string"},
                    },
                    "required": ["action", "target_object",
                                 "destination_zone", "why"],
                    "additionalProperties": False,
                },
            },
            "reasoning": {"type": "string"},
        },
        "required": ["observed_objects", "plan", "reasoning"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot planning system. You see a camera image with a Zone A-F grid.

First, identify all objects visible in the image -- their color and current zone.

Then, based on the instruction you receive, output a step-by-step plan.
Each step should be one action: pick an object, then place/move it to a destination zone.

Output the plan as an ordered list of steps. The order matters."""


def render(objects, scene_layout):
    from PIL import Image, ImageDraw
    import base64
    import io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)
    for c in range(1, 3):
        d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=(200, 200, 210), width=1)
    d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)
    for i, lab in enumerate(ZONE_LABELS):
        r_idx, c_idx = divmod(i, 3)
        d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)),
               f"Zone {lab}", fill=(170, 170, 180))
    for obj in objects:
        x, y = obj["x"], obj["y"]
        r = 26
        shape = obj.get("shape", "round")
        color = obj["color"]
        if shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r],
                        fill=color, outline=(35, 35, 35), width=2)
        else:
            d.ellipse([x - r, y - r, x + r, y + r],
                      fill=color, outline=(35, 35, 35), width=2)
    d.line([(WIDTH // 2 - 16, 20), (WIDTH // 2 + 16, 20)],
           fill=(45, 120, 205), width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def build_objects(scene_layout, rng=None):
    """Return placed objects with optional position jitter.

    Previously used exact ZONE_CENTERS for every run, producing bit-identical
    images at temperature 0.  rng=Random(seed) gives distinct images.
    """
    objects = []
    for oid, zone in scene_layout.items():
        t = next(o for o in OBJECT_TEMPLATES if o["id"] == oid)
        zx, zy = ZONE_CENTERS[zone]
        jx = rng.randint(-28, 28) if rng else 0
        jy = rng.randint(-22, 22) if rng else 0
        objects.append({
            "id": oid,
            "x": zx + jx,
            "y": zy + jy,
            "color": t["color"],
            "shape": t["shape"],
            "color_name": t["color_name"],
            "current_zone": zone,
        })
    return objects


def evaluate_plan(parsed, instruction_info, scene_layout):
    """Score the plan with honest, separated signals.

    Signal 1  visual_grounding_accuracy  -- image-dependent.
    Signal 2  plan_ordering_accuracy     -- instruction-parsing only.
    Signal 3  plan_complete              -- requires BOTH.
    """
    _, exp_first_target, exp_first_zone, exp_second_target, exp_second_zone = (
        instruction_info
    )
    plan = parsed.get("plan", [])
    observed = parsed.get("observed_objects", [])

    # -- Signal 1: visual_grounding_accuracy --------------------------------
    # Match observed objects to ground-truth by COLOR WORD.
    # Color is visible in the image; color words are NOT in the instruction
    # text (instructions name objects like "red cup" but the model must
    # determine WHICH ZONE that cup currently occupies from the image).
    vg_correct = 0
    vg_total = len(scene_layout)
    matched_gt_ids: set = set()
    for obs in observed:
        obs_color = obs.get("color", "").lower()
        obs_zone = obs.get("current_zone", "")
        for tmpl in OBJECT_TEMPLATES:
            oid = tmpl["id"]
            if oid not in scene_layout or oid in matched_gt_ids:
                continue
            if tmpl["color_name"] in obs_color:
                matched_gt_ids.add(oid)
                if obs_zone == scene_layout[oid]:
                    vg_correct += 1
                break
    visual_grounding_accuracy = vg_correct / max(vg_total, 1)

    # Visual grounding for the TARGETED objects only
    targeted_ids = [exp_first_target] + (
        [exp_second_target] if exp_second_target else []
    )
    targeted_ids = [t for t in targeted_ids if t in scene_layout]
    tgt_correct = 0
    matched_tgt: set = set()
    for obs in observed:
        obs_color = obs.get("color", "").lower()
        obs_zone = obs.get("current_zone", "")
        for oid in targeted_ids:
            if oid in matched_tgt:
                continue
            color_name = next(
                t["color_name"] for t in OBJECT_TEMPLATES if t["id"] == oid
            )
            if color_name in obs_color:
                matched_tgt.add(oid)
                if obs_zone == scene_layout[oid]:
                    tgt_correct += 1
                break
    target_vg_accuracy = tgt_correct / max(len(targeted_ids), 1)

    # -- Signal 2: plan_ordering_accuracy (instruction-parsing only) --------
    # Destination zone AND object name appear verbatim in the instruction text.
    # A model that ignores the image can score 100% here.
    first_ok = False
    second_ok = False

    if len(plan) >= 1:
        s1 = plan[0]
        s1_text = (
            s1.get("target_object", "") + " " + s1.get("destination_zone", "")
        ).lower()
        exp1_color = next(
            t["color_name"] for t in OBJECT_TEMPLATES if t["id"] == exp_first_target
        )
        if exp1_color in s1_text and exp_first_zone.lower() in s1_text:
            first_ok = True

    if len(plan) >= 2 and exp_second_target:
        s2 = plan[1]
        s2_text = (
            s2.get("target_object", "") + " " + s2.get("destination_zone", "")
        ).lower()
        exp2_color = next(
            t["color_name"] for t in OBJECT_TEMPLATES if t["id"] == exp_second_target
        )
        if exp2_color in s2_text and exp_second_zone.lower() in s2_text:
            second_ok = True

    ordering_ok = first_ok and (second_ok if exp_second_target else True)
    n_steps = 2 if exp_second_target else 1
    plan_ordering_accuracy = (
        int(first_ok) + int(second_ok if exp_second_target else True)
    ) / n_steps

    # -- Signal 3: plan_complete --------------------------------------------
    # Requires BOTH: visual grounding of targeted objects correct AND ordering.
    plan_complete = (target_vg_accuracy >= 1.0) and ordering_ok

    return {
        "visual_grounding_accuracy": round(visual_grounding_accuracy, 4),
        "target_vg_accuracy": round(target_vg_accuracy, 4),
        "plan_ordering_accuracy": round(plan_ordering_accuracy, 4),
        "first_step_correct": first_ok,
        "second_step_correct": second_ok if exp_second_target else True,
        "plan_complete": plan_complete,
        "plan_length": len(plan),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for scene/instruction selection and position jitter"
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results = []
    errors = 0
    start = time.time()

    print(f"Running {args.runs} runs (seed={args.seed})")
    print(f"  Output: {args.output or '(stdout only)'}")
    print()

    for run_num in range(1, args.runs + 1):
        # Seeded random selection -> genuinely N unique combos
        scene = rng.choice(SCENE_SETUPS)
        instr_idx = rng.randrange(len(INSTRUCTIONS))
        instr_tuple = INSTRUCTIONS[instr_idx]
        instruction = instr_tuple[0]

        # Jittered positions -> distinct images even at temperature 0
        objects = build_objects(scene, rng=rng)
        image_b64 = render(objects, scene)

        prompt = (
            f"Instruction: {instruction}\n\n"
            "Look at the camera image and plan the steps needed."
        )

        t0 = time.perf_counter()
        entry = None
        try:
            result = client.image_chat(
                prompt=prompt,
                image_b64=image_b64,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = json.loads(result.content)
            score = evaluate_plan(parsed, instr_tuple, scene)

            entry = {
                "run": run_num,
                "experiment": "multistep_clean",
                "seed": args.seed,
                "instruction": instruction,
                "scene_layout": scene,
                "n_objects": len(objects),
                # Honest separated metrics
                "visual_grounding_accuracy": score["visual_grounding_accuracy"],
                "target_vg_accuracy": score["target_vg_accuracy"],
                "plan_ordering_accuracy": score["plan_ordering_accuracy"],
                "first_step_correct": score["first_step_correct"],
                "second_step_correct": score["second_step_correct"],
                "plan_complete": score["plan_complete"],
                "plan_length": score["plan_length"],
                "latency_ms": round(latency_ms, 1),
                "success": score["plan_complete"],
                "error": None,
                "prompt_sent": prompt,
                "raw_response": result.content,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)

            if run_num % 20 == 0 or run_num == 1:
                print(
                    f"  [{run_num:>4d}/{args.runs}] "
                    f"vg={score['visual_grounding_accuracy']:.0%} "
                    f"tgt_vg={score['target_vg_accuracy']:.0%} "
                    f"order={score['plan_ordering_accuracy']:.0%} "
                    f"complete={score['plan_complete']} | "
                    f"lat={latency_ms:.0f}ms"
                )

        except Exception as e:
            errors += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            entry = {
                "run": run_num,
                "experiment": "multistep_clean",
                "seed": args.seed,
                "error": str(e)[:200],
                "latency_ms": round(latency_ms, 1),
                "success": False,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)
            print(f"  [{run_num:>4d}/{args.runs}] ERROR: {str(e)[:80]}")
            if errors >= 20:
                print(f"  {errors} errors -- aborting")
                break

        if args.output and entry:
            with open(args.output, "a") as f:
                f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start

    valid = [r for r in results if r.get("error") is None]
    if not valid:
        summary = {"runs": 0, "error": "no valid results", "completed": False}
        print(f"RESULT:{json.dumps(summary)}")
        return

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return 0.0, 0.0, 0.0
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return round(center, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)

    n = len(valid)
    vg_vals = [r["visual_grounding_accuracy"] for r in valid]
    order_vals = [r["plan_ordering_accuracy"] for r in valid]
    complete_k = sum(1 for r in valid if r.get("plan_complete"))
    order_k = sum(1 for r in valid if r.get("plan_ordering_accuracy", 0) >= 1.0)
    vg_k = sum(1 for r in valid if r.get("visual_grounding_accuracy", 0) >= 1.0)

    vg_c, vg_lo, vg_hi = wilson_ci(vg_k, n)
    ord_c, ord_lo, ord_hi = wilson_ci(order_k, n)
    cmp_c, cmp_lo, cmp_hi = wilson_ci(complete_k, n)
    latencies = sorted(r["latency_ms"] for r in valid if "latency_ms" in r)

    summary = {
        "runs": n,
        "seed": args.seed,
        "experiment": "multistep_clean",
        "visual_grounding_accuracy": {
            "mean": round(sum(vg_vals) / n, 4),
            "all_correct_rate": vg_c,
            "wilson_ci_95": [vg_lo, vg_hi],
            "n": n,
            "note": "image-dependent: checks observed_objects.current_zone vs scene_layout",
        },
        "plan_ordering_accuracy": {
            "mean": round(sum(order_vals) / n, 4),
            "all_correct_rate": ord_c,
            "wilson_ci_95": [ord_lo, ord_hi],
            "n": n,
            "note": "instruction-parsing only -- does NOT require looking at the image",
        },
        "plan_complete": {
            "rate": cmp_c,
            "wilson_ci_95": [cmp_lo, cmp_hi],
            "n": n,
            "note": "requires visual grounding of moved objects + correct ordering",
        },
        "p50_latency_ms": round(latencies[n // 2], 1) if latencies else 0,
        "p95_latency_ms": round(latencies[int(n * 0.95)], 1) if latencies else 0,
        "error_count": errors,
        "elapsed_s": round(elapsed, 1),
        "completed": errors < 20,
    }

    print()
    print("=" * 65)
    print("HONEST RESULTS -- multistep_clean")
    print("=" * 65)
    print(f"  Runs (valid): {n}  (seed={args.seed})")
    print()
    print(f"  visual_grounding_accuracy (mean): {sum(vg_vals)/n:.1%}")
    print(f"    all-correct rate: {vg_c:.1%}  95% CI [{vg_lo:.1%}, {vg_hi:.1%}]")
    print("    (image-dependent: observed_objects.current_zone vs scene_layout)")
    print()
    print(f"  plan_ordering_accuracy (mean): {sum(order_vals)/n:.1%}")
    print(f"    fully-correct rate: {ord_c:.1%}  95% CI [{ord_lo:.1%}, {ord_hi:.1%}]")
    print("    (instruction-parsing only -- answer is in the prompt text)")
    print()
    print(f"  plan_complete: {cmp_c:.1%}  95% CI [{cmp_lo:.1%}, {cmp_hi:.1%}]")
    print("    (requires visual grounding + correct ordering -- image-dependent)")
    print()
    print(f"  OLD headline (instruction-parsing only): ~{ord_c:.1%}")
    print(f"  TRUE image-dependent plan_complete: {cmp_c:.1%}")
    print("=" * 65)
    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
