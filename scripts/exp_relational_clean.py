"""Experiment: Relational/Spatial Reasoning -- un-gameable vision questions.

Renders a SINGLE tabletop scene and asks a question whose answer is determined
ONLY by the rendered image geometry, never by text in the prompt.

4 question types (cycled evenly across runs):
  closest_to_gripper  Which single zone holds the object nearest to the gripper?
                      (gripper drawn at top-center ~(192, 20); ground truth =
                       argmin euclidean distance from each object's rendered (x,y))
  top_row_count       How many objects are in the top row (zones A, B, or C)?
                      (count from scene; strict integer match)
  color_in_zone       What is the color of the object in zone {Z}?
                      (Z chosen randomly from occupied zones; exact color-word match)
  leftmost_zone       Which zone holds the leftmost object?
                      (min rendered x -> zone; strict zone-letter match)

Key design properties:
  - Ground truth is computed in Python from the generated scene, not from any
    text in the prompt.
  - The prompt NEVER names the answer.  The model must inspect the image.
  - Seeded RNG gives reproducible but genuinely varied scenes.
  - Strict exact matching; no lenient word-overlap scoring.

Usage:
    python scripts/exp_relational_clean.py --runs 160 --seed 7 \\
        --output overnight_results/relational/clean_relational.jsonl
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
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2

# Top-center of each zone (no jitter yet)
ZONE_CENTERS: dict[str, tuple[int, int]] = {}
for _i, _lab in enumerate(ZONE_LABELS):
    _r, _c = divmod(_i, 3)
    ZONE_CENTERS[_lab] = (int(_c * cw + cw / 2), int(_r * ch + ch / 2))

TOP_ROW_ZONES = {"A", "B", "C"}

# Gripper rendered as a horizontal bar at top-center
GRIPPER_X, GRIPPER_Y = WIDTH // 2, 20  # (192, 20)

OBJECTS = [
    {"color_name": "red",    "color_rgb": (210, 60,  60),  "shape": "round"},
    {"color_name": "blue",   "color_rgb": (60,  90,  210), "shape": "round"},
    {"color_name": "green",  "color_rgb": (60,  180, 60),  "shape": "round"},
    {"color_name": "yellow", "color_rgb": (220, 200, 40),  "shape": "square"},
    {"color_name": "orange", "color_rgb": (230, 140, 40),  "shape": "round"},
    {"color_name": "pink",   "color_rgb": (210, 120, 160), "shape": "round"},
    {"color_name": "purple", "color_rgb": (140, 60,  180), "shape": "square"},
]

QUESTION_TYPES = [
    "closest_to_gripper",
    "top_row_count",
    "color_in_zone",
    "leftmost_zone",
]

# ---------------------------------------------------------------------------
# JSON schema: single 'answer' string field
# ---------------------------------------------------------------------------
ANSWER_SCHEMA = {
    "name": "spatial_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Your answer: a zone letter (A-F), an integer as digits, "
                    "or a color word."
                ),
            },
            "reasoning": {"type": "string"},
        },
        "required": ["answer", "reasoning"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system analyzing a tabletop workspace.

The workspace is divided into a 3x2 grid of zones:
  A (top-left)      B (top-center)      C (top-right)
  D (bottom-left)   E (bottom-center)   F (bottom-right)

Zone labels are printed in the top-left corner of each cell in the image.
The gripper is the blue horizontal bar at the very top-center of the image.

Answer ONLY the question asked. Output a single short answer:
  - a zone letter (A, B, C, D, E, or F)
  - an integer (e.g. 3)
  - a color word (e.g. red, blue, green, yellow, orange, pink, purple)

Do not add explanations in the answer field -- put your thinking in reasoning."""


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def generate_scene(rng: random.Random, n_objects: Optional[int] = None):
    """Return a list of placed objects.

    Each object has: color_name, color_rgb, shape, zone, x, y.
    Zones are chosen without replacement so no two objects share a zone.
    """
    if n_objects is None:
        n_objects = rng.randint(3, 5)
    selected_objs = rng.sample(OBJECTS, k=min(n_objects, len(OBJECTS)))
    selected_zones = rng.sample(ZONE_LABELS, k=n_objects)

    placed = []
    for obj, zone in zip(selected_objs, selected_zones):
        zx, zy = ZONE_CENTERS[zone]
        jx = rng.randint(-28, 28)
        jy = rng.randint(-22, 22)
        placed.append({
            "color_name": obj["color_name"],
            "color_rgb": obj["color_rgb"],
            "shape": obj["shape"],
            "zone": zone,
            "x": zx + jx,
            "y": zy + jy,
        })
    return placed


def render_scene(objects: list[dict]) -> str:
    """Render scene to base64 PNG data URI."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    # Grid lines
    for c in range(1, 3):
        d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)],
               fill=(200, 200, 210), width=1)
    d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)

    # Zone labels
    for idx, lab in enumerate(ZONE_LABELS):
        r_idx, c_idx = divmod(idx, 3)
        d.text((int(c_idx * cw + 8), int(r_idx * ch + 6)),
               f"Zone {lab}", fill=(170, 170, 180))

    # Objects
    for obj in objects:
        x, y = obj["x"], obj["y"]
        r = 26
        color = obj["color_rgb"]
        shape = obj["shape"]
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

    # Gripper (blue horizontal bar at top-center)
    d.line([(GRIPPER_X - 16, GRIPPER_Y), (GRIPPER_X + 16, GRIPPER_Y)],
           fill=(45, 120, 205), width=4)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def ground_truth_closest_to_gripper(objects: list[dict]) -> str:
    """Zone of the object with minimum Euclidean distance to the gripper."""
    best_obj = min(
        objects,
        key=lambda o: math.hypot(o["x"] - GRIPPER_X, o["y"] - GRIPPER_Y),
    )
    return best_obj["zone"]


def ground_truth_top_row_count(objects: list[dict]) -> str:
    """Count of objects whose zone is in {A, B, C}.  Returned as digit string."""
    count = sum(1 for o in objects if o["zone"] in TOP_ROW_ZONES)
    return str(count)


def ground_truth_color_in_zone(objects: list[dict], zone: str) -> str:
    """Color name of the object in the given zone."""
    for o in objects:
        if o["zone"] == zone:
            return o["color_name"]
    raise ValueError(f"No object in zone {zone}")


def ground_truth_leftmost(objects: list[dict]) -> str:
    """Zone of the object with the minimum rendered x coordinate."""
    return min(objects, key=lambda o: o["x"])["zone"]


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def make_question(q_type: str, objects: list[dict], rng: random.Random):
    """Return (prompt_text, ground_truth_answer, question_kwargs)."""
    if q_type == "closest_to_gripper":
        prompt = (
            "Which single zone contains the object that is closest to the gripper "
            "(the blue bar at the top-center of the image)? "
            "Reply with exactly one zone letter."
        )
        gt = ground_truth_closest_to_gripper(objects)
        return prompt, gt, {}

    elif q_type == "top_row_count":
        prompt = (
            "How many objects are in the top row of the workspace "
            "(zones A, B, or C)? Reply with just an integer."
        )
        gt = ground_truth_top_row_count(objects)
        return prompt, gt, {}

    elif q_type == "color_in_zone":
        occupied = [o["zone"] for o in objects]
        zone = rng.choice(occupied)
        prompt = (
            f"What is the color of the object in Zone {zone}? "
            "Reply with exactly one color word."
        )
        gt = ground_truth_color_in_zone(objects, zone)
        return prompt, gt, {"target_zone": zone}

    elif q_type == "leftmost_zone":
        prompt = (
            "Which zone holds the leftmost object "
            "(the one with the smallest x-coordinate, i.e. closest to the left edge)? "
            "Reply with exactly one zone letter."
        )
        gt = ground_truth_leftmost(objects)
        return prompt, gt, {}

    else:
        raise ValueError(f"Unknown question type: {q_type}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def score_answer(q_type: str, model_answer: str, ground_truth: str) -> bool:
    """Strict exact matching.  No lenient word-overlap."""
    import re
    ma = model_answer.strip().lower()
    gt = ground_truth.strip().lower()

    if q_type == "top_row_count":
        # Integer match -- strip non-digit chars from model output
        digits = re.sub(r"[^0-9]", "", ma)
        return digits == gt

    elif q_type in ("closest_to_gripper", "leftmost_zone"):
        # Zone letter match -- find first standalone A-F letter (case-insensitive)
        letters = re.findall(r"\b[a-fA-F]\b", model_answer.strip())
        if letters:
            return letters[0].upper() == gt.upper()
        return ma == gt

    elif q_type == "color_in_zone":
        # Color word containment in model answer
        return gt in ma

    return ma == gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=160)
    parser.add_argument(
        "--seed", type=int, default=7,
        help="RNG seed for scene generation"
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results: list[dict] = []
    errors = 0
    start = time.time()

    print(f"Running {args.runs} relational runs (seed={args.seed})")
    print(f"  Question types: {', '.join(QUESTION_TYPES)}")
    print(f"  Output: {args.output or '(stdout only)'}")
    print()

    for run_num in range(1, args.runs + 1):
        q_type = QUESTION_TYPES[(run_num - 1) % len(QUESTION_TYPES)]

        objects = generate_scene(rng)
        image_b64 = render_scene(objects)

        question_prompt, ground_truth, q_kwargs = make_question(q_type, objects, rng)

        full_prompt = (
            "Look at the image carefully.\n\n"
            f"Question: {question_prompt}"
        )

        t0 = time.perf_counter()
        entry: dict = {}
        try:
            result = client.image_chat(
                prompt=full_prompt,
                image_b64=image_b64,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_schema", "json_schema": ANSWER_SCHEMA},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = json.loads(result.content)
            model_answer = parsed.get("answer", "").strip()
            correct = score_answer(q_type, model_answer, ground_truth)

            entry = {
                "run": run_num,
                "experiment": "relational_clean",
                "seed": args.seed,
                "q_type": q_type,
                "question": question_prompt,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "model_reasoning": parsed.get("reasoning", ""),
                "correct": correct,
                "n_objects": len(objects),
                "scene": [
                    {"color": o["color_name"], "zone": o["zone"], "x": o["x"], "y": o["y"]}
                    for o in objects
                ],
                "latency_ms": round(latency_ms, 1),
                "success": correct,
                "error": None,
                "timestamp": datetime.utcnow().isoformat(),
                **{f"q_{k}": v for k, v in q_kwargs.items()},
            }
            results.append(entry)

            if run_num % 20 == 0 or run_num == 1:
                print(
                    f"  [{run_num:>4d}/{args.runs}] {q_type:<22s} "
                    f"gt={ground_truth:<8s} model={model_answer:<10s} "
                    f"{'OK' if correct else 'XX'} | {latency_ms:.0f}ms"
                )

        except Exception as e:
            errors += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            entry = {
                "run": run_num,
                "experiment": "relational_clean",
                "seed": args.seed,
                "q_type": q_type,
                "error": str(e)[:300],
                "latency_ms": round(latency_ms, 1),
                "success": False,
                "correct": False,
                "timestamp": datetime.utcnow().isoformat(),
            }
            results.append(entry)
            print(f"  [{run_num:>4d}/{args.runs}] {q_type:<22s} ERROR: {str(e)[:70]}")

            # Exponential back-off on rate-limit errors
            if "rate" in str(e).lower() or "429" in str(e):
                wait = min(60, 2 ** min(errors, 6))
                print(f"    Rate-limited -- waiting {wait}s")
                time.sleep(wait)
            elif errors >= 20:
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

    # Per-type breakdown
    from collections import defaultdict
    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        per_type[r.get("q_type", "unknown")].append(r)

    type_stats: dict = {}
    sample_triples: list[dict] = []

    for qt in QUESTION_TYPES:
        entries = per_type.get(qt, [])
        if not entries:
            type_stats[qt] = {"n": 0, "accuracy": 0.0, "wilson_ci_95": [0.0, 0.0]}
            continue
        k = sum(1 for e in entries if e.get("correct"))
        n = len(entries)
        acc, lo, hi = wilson_ci(k, n)
        type_stats[qt] = {"n": n, "accuracy": acc, "wilson_ci_95": [lo, hi]}

        # Collect 1 example triple per type
        for e in entries[:3]:
            sample_triples.append({
                "q_type": qt,
                "prompt": e.get("question", ""),
                "ground_truth": e.get("ground_truth", ""),
                "model_answer": e.get("model_answer", ""),
                "correct": e.get("correct", False),
            })

    n_all = len(valid)
    k_all = sum(1 for r in valid if r.get("correct"))
    acc_all, lo_all, hi_all = wilson_ci(k_all, n_all)
    latencies = sorted(r["latency_ms"] for r in valid if "latency_ms" in r)

    summary = {
        "runs": n_all,
        "seed": args.seed,
        "experiment": "relational_clean",
        "overall_accuracy": acc_all,
        "overall_wilson_ci_95": [lo_all, hi_all],
        "per_question_type": type_stats,
        "sample_triples": sample_triples,
        "p50_latency_ms": round(latencies[n_all // 2], 1) if latencies else 0,
        "p95_latency_ms": round(latencies[int(n_all * 0.95)], 1) if latencies else 0,
        "error_count": errors,
        "elapsed_s": round(elapsed, 1),
        "completed": errors < 20,
    }

    print()
    print("=" * 70)
    print("RELATIONAL CLEAN -- un-gameable spatial reasoning results")
    print("=" * 70)
    print(f"  Runs (valid): {n_all}  (seed={args.seed})")
    print(f"  Overall accuracy: {acc_all:.1%}  95% CI [{lo_all:.1%}, {hi_all:.1%}]")
    print()
    print(f"  {'Question type':<24} {'N':>5} {'Acc':>8} {'95% CI':>18}")
    print("  " + "-" * 57)
    for qt in QUESTION_TYPES:
        ts = type_stats.get(qt, {})
        n_t = ts.get("n", 0)
        a_t = ts.get("accuracy", 0.0)
        ci = ts.get("wilson_ci_95", [0.0, 0.0])
        print(f"  {qt:<24} {n_t:>5} {a_t:>8.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")
    print()
    print("  Sample triples (prompt | ground_truth | model_answer):")
    for s in sample_triples[:4]:
        tag = "OK" if s["correct"] else "XX"
        print(f"    [{tag}] {s['q_type']}")
        print(f"         Q: {s['prompt'][:90]}")
        print(f"         GT: {s['ground_truth']}  |  Model: {s['model_answer']}")
    print("=" * 70)
    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
