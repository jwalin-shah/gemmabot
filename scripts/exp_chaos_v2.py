"""Chaos Experiment v2: targeted object-centric perturbations.

Usage:
    python scripts/exp_chaos_v2.py --runs 30 --output results.jsonl --variation object_teleport
    python scripts/exp_chaos_v2.py --runs 30 --output results.jsonl --variation new_object_appears
    python scripts/exp_chaos_v2.py --runs 30 --output results.jsonl --variation object_disappears
    python scripts/exp_chaos_v2.py --runs 30 --output results.jsonl --variation color_swap
"""

from __future__ import annotations

import argparse
import json
import math
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
            "changes_detected": {"type": "string"},
        },
        "required": ["observed_objects", "total_objects_visible", "changes_detected"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system. You see a camera image with a Zone A-F grid.
Identify every object by its color, shape, and which zone it occupies.
Also note any changes you detect compared to normal."""

# ─── Chaos Types ─────────────────────────────────────────────────────────
CHAOS_TYPES = [
    "gripper_color_change",
    "gripper_position_change",
    "object_color_swap",
    "object_appears",
    "object_disappears",
    "background_change",
    "grid_removed",
    "gripper_style_change",
]

# Canonical names for user-facing variation args
VARIATION_MAP = {
    "object_teleport": "object_teleport",
    "new_object_appears": "object_appears",
    "object_disappears": "object_disappears",
    "color_swap": "object_color_swap",
}

BASE_SCENE = {"red_cup": "D", "blue_cup": "E", "green_cup": "F"}

OBJECT_TEMPLATES = {
    "red_cup": (210, 60, 60, "round", ""),
    "blue_cup": (60, 90, 210, "round", ""),
    "green_cup": (60, 180, 60, "round", ""),
    "yellow_cube": (220, 200, 40, "square", ""),
}


def render(objects, show_grid=True, gripper_color=(45, 120, 205), gripper_x=None,
           background=None, gripper_width=4):
    from PIL import Image, ImageDraw
    import base64, io

    if background:
        img = Image.new("RGB", (WIDTH, HEIGHT), background)
    else:
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
        r = 26
        color = obj["color"]
        shape = obj.get("shape", "round")
        if shape == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)

    # Gripper
    gx = gripper_x if gripper_x is not None else WIDTH // 2
    gy = 20
    col = gripper_color
    d.line([(gx - 16, gy), (gx + 16, gy)], fill=col, width=gripper_width)
    d.line([(gx, gy), (gx, gy - 24)], fill=col, width=gripper_width)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def build_objects_from_scene(scene_dict, color_override=None):
    objects = []
    for oid, zone in scene_dict.items():
        if oid not in OBJECT_TEMPLATES:
            continue
        cr, cg, cb, shape, attr = OBJECT_TEMPLATES[oid]
        zx, zy = ZONE_CENTERS[zone]
        color = color_override.get(oid, (cr, cg, cb)) if color_override else (cr, cg, cb)
        objects.append({
            "id": oid, "x": zx, "y": zy, "color": color,
            "shape": shape, "attr": attr, "zone": zone,
        })
    return objects


def apply_chaos(chaos_type, base_objects, base_scene, tick_num):
    """Apply a chaos perturbation to the scene and return (new_objects, description)."""
    import copy
    objs = copy.deepcopy(base_objects)

    if chaos_type == "gripper_color_change":
        return objs, "gripper turned RED", {"gripper_color": (205, 45, 45)}

    elif chaos_type == "gripper_position_change":
        return objs, "gripper moved to RIGHT side", {"gripper_x": WIDTH - 50}

    elif chaos_type == "gripper_style_change":
        return objs, "gripper became THICK and ORANGE", {"gripper_color": (255, 165, 0), "gripper_width": 8}

    elif chaos_type == "object_color_swap":
        # Swap colors of red_cup and blue_cup
        for obj in objs:
            if obj["id"] == "red_cup":
                obj["color"] = (60, 90, 210)  # blue
            elif obj["id"] == "blue_cup":
                obj["color"] = (210, 60, 60)  # red
        return objs, "red and blue objects SWAPPED colors", {}

    elif chaos_type == "object_appears":
        # Add a new object
        zx, zy = ZONE_CENTERS["A"]
        objs.append({
            "id": "yellow_cube", "x": zx, "y": zy,
            "color": (220, 200, 40), "shape": "square", "attr": "", "zone": "A",
        })
        return objs, "NEW yellow object appeared in Zone A", {}

    elif chaos_type == "object_disappears":
        # Remove green_cup
        objs = [o for o in objs if o["id"] != "green_cup"]
        return objs, "green object DISAPPEARED from Zone F", {}

    elif chaos_type == "object_teleport":
        # Teleport blue_cup from Zone E to Zone B
        for obj in objs:
            if obj["id"] == "blue_cup":
                zx, zy = ZONE_CENTERS["B"]
                obj["x"], obj["y"] = zx, zy
                obj["zone"] = "B"
        return objs, "blue object TELEPORTED from Zone E to Zone B", {}

    elif chaos_type == "background_change":
        return objs, "background changed to DARK", {"background": (60, 60, 70)}

    elif chaos_type == "grid_removed":
        return objs, "zone grid LINES removed", {"show_grid": False}

    return objs, "no change", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--variation", type=str, default=None,
                        help="Run only one chaos variation: object_teleport, new_object_appears, object_disappears, color_swap")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Resolve variation
    target_chaos_type = None
    if args.variation:
        if args.variation in VARIATION_MAP:
            target_chaos_type = VARIATION_MAP[args.variation]
        elif args.variation in CHAOS_TYPES:
            target_chaos_type = args.variation
        else:
            print(f"Unknown variation '{args.variation}'. Valid: {list(VARIATION_MAP.keys()) + CHAOS_TYPES}")
            sys.exit(1)

    client = CerebrasClient()
    results = []
    run_summaries = []
    errors = 0
    start = time.time()

    for run_num in range(1, args.runs + 1):
        chaos_type = target_chaos_type if target_chaos_type else CHAOS_TYPES[run_num % len(CHAOS_TYPES)]

        run_ticks = []
        for seq_tick in range(8):
            is_post_chaos = seq_tick >= 4
            scene = BASE_SCENE
            base_objects = build_objects_from_scene(scene)

            render_kwargs = {}

            if is_post_chaos and seq_tick == 4:
                # Apply chaos
                chaos_objects, change_desc, render_kwargs = apply_chaos(
                    chaos_type, base_objects, scene, seq_tick
                )
                current_objects = chaos_objects
            elif is_post_chaos:
                # Maintain chaos state from previous tick
                chaos_objects, change_desc, render_kwargs = apply_chaos(
                    chaos_type, base_objects, scene, seq_tick
                )
                current_objects = chaos_objects
            else:
                current_objects = base_objects
                change_desc = "pre-change (normal state)"
                render_kwargs = {}

            image_b64 = render(current_objects, **render_kwargs)

            prompt = f"Tick {seq_tick}. Identify every object and its zone. Note any changes."

            t0 = time.perf_counter()
            try:
                result = client.image_chat(
                    prompt=prompt,
                    image_b64=image_b64,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=args.temperature,
                    max_tokens=500,
                    response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA},
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                parsed = json.loads(result.content)

                # Score this tick
                observed = parsed.get("observed_objects", [])
                reported_count = parsed.get("total_objects_visible", 0)
                changes_detected = parsed.get("changes_detected", "")

                # Match objects
                matched = 0
                for obs in observed:
                    obs_zone = obs.get("zone", "")
                    obs_desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()
                    for gt in current_objects:
                        gt_id = gt["id"].replace("_", " ").lower()
                        if any(w in obs_desc for w in gt_id.split()) and obs_zone == gt["zone"]:
                            matched += 1
                            break

                accuracy = matched / max(len(current_objects), 1)

                entry_tick = {
                    "tick": seq_tick,
                    "run": run_num,
                    "chaos_type": chaos_type if is_post_chaos else "none",
                    "is_post_chaos": is_post_chaos,
                    "n_objects": len(current_objects),
                    "reported_count": reported_count,
                    "matches": matched,
                    "accuracy": round(accuracy, 4),
                    "changes_detected": changes_detected,
                    "latency_ms": round(latency_ms, 1),
                }
                run_ticks.append(entry_tick)
                results.append(entry_tick)

            except Exception as e:
                errors += 1
                if errors >= 20:
                    break

        if errors >= 20:
            break

        # Compute per-run summary
        pre = [t for t in run_ticks if not t.get("is_post_chaos")]
        post = [t for t in run_ticks if t.get("is_post_chaos")]

        # Determine if model detected the change (look at post-chaos ticks for any "change" mention)
        post_changes_detected = [t.get("changes_detected", "") for t in post]
        any_change_detected = any(
            any(kw in (cd or "").lower() for kw in ["change", "swap", "new", "disappear", "teleport", "move", "shift", "different", "absent", "missing", "appear", "color"])
            for cd in post_changes_detected
        )

        # Count ticks to re-acquire: after tick 4 (first post-chaos), find first tick where accuracy >= 0.8 * mean_pre_accuracy
        mean_pre_accuracy = sum(t["accuracy"] for t in pre) / len(pre) if pre else 0
        threshold = mean_pre_accuracy * 0.8
        reacquire_tick = None
        for t in post:
            if t["accuracy"] >= threshold and threshold > 0:
                reacquire_tick = t["tick"]
                break

        summary_entry = {
            "run": run_num,
            "experiment": "chaos",
            "variation": args.variation or chaos_type,
            "chaos_type": chaos_type,
            "mean_pre_chaos_accuracy": round(mean_pre_accuracy, 4),
            "mean_post_chaos_accuracy": round(sum(t["accuracy"] for t in post) / len(post), 4) if post else 0,
            "post_vs_pre_drop": round(mean_pre_accuracy - (sum(t["accuracy"] for t in post) / len(post) if post else 0), 4),
            "any_change_detected": any_change_detected,
            "reacquire_at_tick": reacquire_tick,
            "latency_ms": round(sum(t["latency_ms"] for t in run_ticks) / len(run_ticks), 1) if run_ticks else 0,
            "success": round(sum(t["accuracy"] for t in post) / len(post), 4) >= 0.5 if post else False,
            "timestamp": datetime.utcnow().isoformat(),
        }
        run_summaries.append(summary_entry)

        if args.output:
            # Write per-tick data
            with open(args.output, "a") as f:
                f.write(json.dumps(summary_entry) + "\n")

        if run_num == 1 or run_num % 10 == 0 or run_num == args.runs:
            print(f"  [{run_num:>4d}/{args.runs}] chaos={chaos_type} | "
                  f"pre={summary_entry['mean_pre_chaos_accuracy']:.0%} post={summary_entry['mean_post_chaos_accuracy']:.0%} "
                  f"detected={any_change_detected} reacquire={reacquire_tick}")

    elapsed = time.time() - start

    if not run_summaries:
        summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        pre_accs = [r["mean_pre_chaos_accuracy"] for r in run_summaries]
        post_accs = [r["mean_post_chaos_accuracy"] for r in run_summaries]
        detections = [r["any_change_detected"] for r in run_summaries]
        reacquire_ticks = [r["reacquire_at_tick"] for r in run_summaries if r["reacquire_at_tick"] is not None]
        lats = [r["latency_ms"] for r in run_summaries]

        summary = {
            "runs": len(run_summaries),
            "experiment": "chaos",
            "variation": args.variation or "all",
            "chaos_type": target_chaos_type or "mixed",
            "mean_pre_chaos_accuracy": round(sum(pre_accs) / len(pre_accs), 4) if pre_accs else 0,
            "mean_post_chaos_accuracy": round(sum(post_accs) / len(post_accs), 4) if post_accs else 0,
            "mean_drop": round(sum(pre_accs) / len(pre_accs) - sum(post_accs) / len(post_accs), 4) if pre_accs and post_accs else 0,
            "change_detection_rate": round(sum(1 for d in detections if d) / len(detections), 4) if detections else 0,
            "mean_reacquire_tick": round(sum(reacquire_ticks) / len(reacquire_ticks), 2) if reacquire_ticks else None,
            "mean_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
            "error_count": errors,
            "elapsed_s": round(elapsed, 1),
            "completed": errors < 20,
        }

    # Write final summary as last JSON line
    if args.output:
        with open(args.output, "a") as f:
            f.write("__SUMMARY__:" + json.dumps(summary) + "\n")

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY for variation={args.variation or 'all'}")
    print(f"{'='*60}")
    print(f"  Runs completed:     {summary['runs']}")
    print(f"  Pre-chaos accuracy:  {summary['mean_pre_chaos_accuracy']:.2%}")
    print(f"  Post-chaos accuracy: {summary['mean_post_chaos_accuracy']:.2%}")
    print(f"  Accuracy drop:       {summary['mean_drop']:.2%}")
    print(f"  Change detection:    {summary['change_detection_rate']:.1%}")
    if summary['mean_reacquire_tick'] is not None:
        print(f"  Mean reacquire tick: {summary['mean_reacquire_tick']:.1f}")
    print(f"  Mean latency:        {summary['mean_latency_ms']:.0f} ms")
    print(f"  Elapsed time:        {summary['elapsed_s']:.0f}s")
    print(f"  Errors:              {summary['error_count']}")
    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
