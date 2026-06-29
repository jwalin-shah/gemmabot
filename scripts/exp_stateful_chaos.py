"""Stateful Multi-Frame Chaos Experiment: FIRST EVER test of Gemma 4 with
conversation state passed between frames.

THE PROBLEM ALL prior experiments solved: Each image was an independent API
call. The model had ZERO memory of previous frames. Change detection was 0%.

THE FIX: Pass conversation history between frames so the model can compare
its own descriptions across time.

Tests 4 chaos types with 8-tick sequences:
  - color_swap:    red and blue objects swap colors at tick 4
  - teleport:      gripper moves to unexpected position at tick 4
  - appear:        new yellow object appears in Zone A at tick 4
  - disappear:     green object vanishes from Zone F at tick 4

Usage:
    python scripts/exp_stateful_chaos.py --runs 40 --output overnight_results/stateful/r9_stateful.jsonl --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient

# ─── Constants ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2

ZONE_CENTERS = {}
for i, lab in enumerate(ZONE_LABELS):
    r, c_idx = divmod(i, 3)
    ZONE_CENTERS[lab] = (int(c_idx * cw + cw / 2), int(r * ch + ch / 2))

CHAOS_TYPES = [
    "color_swap",
    "teleport",
    "appear",
    "disappear",
]

BASE_SCENE = {"red_cup": "D", "blue_cup": "E", "green_cup": "F"}

OBJECT_TEMPLATES = {
    "red_cup": (210, 60, 60, "round", ""),
    "blue_cup": (60, 90, 210, "round", ""),
    "green_cup": (60, 180, 60, "round", ""),
    "yellow_cube": (220, 200, 40, "square", ""),
}

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

SYSTEM_PROMPT = """You are a robot vision system monitoring a continuous camera feed. You see a camera image with a Zone A-F grid.
IMPORTANT: You have HISTORY of previous frames in this conversation. Compare each new frame against what you saw before.
Identify every object by its color, shape, and which zone it occupies.
If objects have changed color, moved zones, appeared, or disappeared since the previous frames, you MUST note this in 'changes_detected'."""

# ─── Rendering ─────────────────────────────────────────────────────────────


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


# ─── Chaos Application ─────────────────────────────────────────────────────

def apply_chaos(chaos_type, base_objects, base_scene, tick_num):
    """Apply a chaos perturbation to the scene and return (new_objects, description, render_kwargs, expected_chaos_name)."""
    import copy
    objs = copy.deepcopy(base_objects)

    if chaos_type == "color_swap":
        # Swap colors of red_cup and blue_cup
        for obj in objs:
            if obj["id"] == "red_cup":
                obj["color"] = (60, 90, 210)  # blue
            elif obj["id"] == "blue_cup":
                obj["color"] = (210, 60, 60)  # red
        return objs, "red and blue objects SWAPPED colors", {}, "object_color_swap"

    elif chaos_type == "teleport":
        return objs, "gripper moved to RIGHT side", {"gripper_x": WIDTH - 50}, "gripper_position_change"

    elif chaos_type == "appear":
        zx, zy = ZONE_CENTERS["A"]
        objs.append({
            "id": "yellow_cube", "x": zx, "y": zy,
            "color": (220, 200, 40), "shape": "square", "attr": "", "zone": "A",
        })
        return objs, "NEW yellow object appeared in Zone A", {}, "object_appears"

    elif chaos_type == "disappear":
        objs = [o for o in objs if o["id"] != "green_cup"]
        return objs, "green object DISAPPEARED from Zone F", {}, "object_disappears"

    return objs, "no change", {}, "none"


# ─── Scoring ───────────────────────────────────────────────────────────────

def score_tick(parsed, current_objects, chaos_type, is_post_chaos, tick, tick_messages):
    """Score a single tick response.

    Returns:
        dict with scoring metrics for this tick
    """
    observed = parsed.get("observed_objects", [])
    reported_count = parsed.get("total_objects_visible", 0)
    changes_detected = parsed.get("changes_detected", "")

    # --- Object matching ---
    matched = 0
    zone_correct = 0
    zone_possible = 0
    for obs in observed:
        obs_zone = obs.get("zone", "")
        obs_desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()
        best_match = False
        for gt in current_objects:
            gt_id = gt["id"].replace("_", " ").lower()
            if any(w in obs_desc for w in gt_id.split()) and obs_zone == gt["zone"]:
                best_match = True
                break
        if best_match:
            matched += 1

    accuracy = matched / max(len(current_objects), 1)

    # --- Zone correctness (for matched objects) ---
    zone_correct = 0
    zone_possible = 0
    for gt in current_objects:
        gt_id = gt["id"].replace("_", " ").lower()
        gt_zone = gt["zone"]
        for obs in observed:
            obs_zone = obs.get("zone", "")
            obs_desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()
            if any(w in obs_desc for w in gt_id.split()):
                zone_possible += 1
                if obs_zone == gt_zone:
                    zone_correct += 1
                break

    zone_accuracy = zone_correct / max(zone_possible, 1)

    # --- Change detection ---
    change_keywords = [
        "change", "swap", "different", "appear", "disappear", "vanish",
        "move", "shift", "new", "gone", "missing", "turned", "became",
        "was previously", "used to be", "no longer", "added", "removed",
        "relocat", "reposition",
    ]
    change_lower = changes_detected.lower()
    any_change_flag = any(kw in change_lower for kw in change_keywords)

    # --- Color swap specific detection ---
    color_swap_detected = False
    if chaos_type == "color_swap" and is_post_chaos:
        color_swap_patterns = [
            r"swap", r"turned\s+\w+", r"changed\s+(from\s+\w+\s+)?to",
            r"was\s+\w+\s*[,.]*?\s*now", r"used\s+to\s+be",
            r"became", r"converted", r"red.*blue|blue.*red",
            r"were\s+swapped", r"exchanged",
        ]
        for pattern in color_swap_patterns:
            if re.search(pattern, change_lower):
                color_swap_detected = True
                break
        # Also check object descriptions for explicit color+swap language
        for obs in observed:
            desc = (obs.get("description", "") + " " + obs.get("color", "")).lower()
            if any(kw in desc for kw in ["swap", "turned", "now", "was", "changed"]):
                # Check if it references both red and blue or mentions the swap
                if not color_swap_detected:
                    # Look for the color word that doesn't match the original
                    if ("red" in desc and obs.get("zone") == "E") or \
                       ("blue" in desc and obs.get("zone") == "D"):
                        color_swap_detected = True
                        break

    return {
        "tick": tick,
        "is_post_chaos": is_post_chaos,
        "n_objects": len(current_objects),
        "reported_count": reported_count,
        "matches": matched,
        "accuracy": round(accuracy, 4),
        "zone_accuracy": round(zone_accuracy, 4),
        "any_change_detected": any_change_flag,
        "color_swap_detected": color_swap_detected if chaos_type == "color_swap" else None,
        "changes_detected_text": changes_detected[:200] if changes_detected else "",
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stateful multi-frame chaos experiment")
    parser.add_argument("--runs", type=int, default=40, help="Number of runs")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max_context_images", type=int, default=2,
                        help="Max recent images to include in context (0=all, 2=current+previous)")
    parser.add_argument("--runs_per_type", type=int, default=10,
                        help="Number of runs per chaos type (overrides --runs)")
    args = parser.parse_args()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    client = CerebrasClient()
    results = []
    errors = 0
    start = time.time()
    total_api_calls = 0

    for run_num in range(1, args.runs + 1):
        chaos_type = CHAOS_TYPES[run_num % len(CHAOS_TYPES)]

        # ── Per-tick data store (no images in messages yet) ──
        tick_data: list[dict] = []  # Each: {image, objects, prompt, response, is_post}
        tick_details = []

        for seq_tick in range(8):
            is_post_chaos = seq_tick >= 4
            scene = BASE_SCENE
            base_objects = build_objects_from_scene(scene)

            render_kwargs = {}

            if is_post_chaos:
                if seq_tick == 4:
                    chaos_objects, change_desc, render_kwargs, expected_chaos = apply_chaos(
                        chaos_type, base_objects, scene, seq_tick
                    )
                    current_objects = chaos_objects
                else:
                    chaos_objects, _, render_kwargs, expected_chaos = apply_chaos(
                        chaos_type, base_objects, scene, seq_tick
                    )
                    current_objects = chaos_objects
            else:
                current_objects = base_objects
                expected_chaos = "none"

            image_b64 = render(current_objects, **render_kwargs)

            # Build prompt
            if is_post_chaos:
                prompt = (
                    f"Tick {seq_tick}. IDENTIFY every object and its zone. "
                    f"Then COMPARE with what you described in previous ticks. "
                    f"Has anything changed? Note any changes in 'changes_detected'."
                )
            else:
                prompt = (
                    f"Tick {seq_tick}. IDENTIFY every object and its zone. "
                    f"Describe exactly what you see. This will be used as a reference "
                    f"for comparison in future ticks."
                )

            # Store tick data
            tick_data.append({
                "image": image_b64,
                "objects": current_objects,
                "prompt": prompt,
                "response": None,
                "is_post_chaos": is_post_chaos,
            })

            # ── Build messages with SLIDING WINDOW for images ──
            # Full text history is preserved. Only the last N ticks include images.
            call_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            window_start = max(0, len(tick_data) - args.max_context_images)

            for i, td in enumerate(tick_data):
                # User message with or without image
                if i >= window_start:
                    content = [
                        {"type": "text", "text": td["prompt"]},
                        {"type": "image_url", "image_url": {"url": td["image"]}},
                    ]
                else:
                    content = [{"type": "text", "text": td["prompt"]}]

                call_messages.append({"role": "user", "content": content})

                # Add assistant response if we have one (not the current tick)
                if td["response"] is not None:
                    call_messages.append({"role": "assistant", "content": td["response"]})

            # ── API call ──
            t0 = time.perf_counter()
            try:
                result = client.chat(
                    call_messages,
                    temperature=args.temperature,
                    max_tokens=500,
                    response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA},
                )
                total_api_calls += 1
                latency_ms = (time.perf_counter() - t0) * 1000

                parsed = json.loads(result.content)
                tick_data[-1]["response"] = result.content

                # Score
                score = score_tick(
                    parsed, current_objects, chaos_type, is_post_chaos, seq_tick, call_messages
                )
                score["latency_ms"] = round(latency_ms, 1)
                tick_details.append(score)

            except Exception as e:
                errors += 1
                tick_details.append({
                    "tick": seq_tick,
                    "is_post_chaos": is_post_chaos,
                    "error": str(e),
                    "n_objects": len(current_objects),
                    "accuracy": 0.0,
                    "latency_ms": 0,
                })
                if errors >= 20:
                    break

        if errors >= 20:
            break

        # ── Compute run-level summary ──
        pre_ticks = [t for t in tick_details if not t.get("is_post_chaos") and "error" not in t]
        post_ticks = [t for t in tick_details if t.get("is_post_chaos") and "error" not in t]

        pre_accuracy = sum(t["accuracy"] for t in pre_ticks) / len(pre_ticks) if pre_ticks else 0
        post_accuracy = sum(t["accuracy"] for t in post_ticks) / len(post_ticks) if post_ticks else 0

        post_change_detected = [t.get("any_change_detected", False) for t in post_ticks]

        # Color swap specific: did model explicitly note the swap?
        if chaos_type == "color_swap":
            color_swap_detected = any(
                t.get("color_swap_detected", False) for t in post_ticks
            )
        else:
            color_swap_detected = None

        # Zone accuracy
        pre_zone = sum(t.get("zone_accuracy", 0) for t in pre_ticks) / len(pre_ticks) if pre_ticks else 0
        post_zone = sum(t.get("zone_accuracy", 0) for t in post_ticks) / len(post_ticks) if post_ticks else 0

        # Compute expected chaos name for consistency with prior experiments
        chaos_name_map = {
            "color_swap": "object_color_swap",
            "teleport": "gripper_position_change",
            "appear": "object_appears",
            "disappear": "object_disappears",
        }
        expected_chaos_name = chaos_name_map.get(chaos_type, chaos_type)

        entry = {
            "run": run_num,
            "experiment": "stateful_chaos",
            "variation": chaos_type,
            "chaos_type": expected_chaos_name,
            "mean_pre_chaos_accuracy": round(pre_accuracy, 4),
            "mean_post_chaos_accuracy": round(post_accuracy, 4),
            "post_vs_pre_drop": round(pre_accuracy - post_accuracy, 4),
            "any_change_detected": any(post_change_detected),
            "change_detection_count": sum(post_change_detected),
            "color_swap_explicitly_detected": color_swap_detected,
            "pre_chaos_zone_accuracy": round(pre_zone, 4),
            "post_chaos_zone_accuracy": round(post_zone, 4),
            "mean_latency_ms": round(
                sum(t.get("latency_ms", 0) for t in tick_details if "error" not in t) /
                max(len([t for t in tick_details if "error" not in t]), 1),
                1,
            ),
            "success": post_accuracy >= 0.5,
            "tick_details": tick_details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results.append(entry)

        # ── Save per-run ──
        if args.output:
            # Remove tick_details for compact output, save separately
            output_entry = {k: v for k, v in entry.items() if k != "tick_details"}
            output_entry["n_pre_ticks"] = len(pre_ticks)
            output_entry["n_post_ticks"] = len(post_ticks)
            with open(args.output, "a") as f:
                f.write(json.dumps(output_entry) + "\n")

        # ── Print progress ──
        change_str = f"change={any(post_change_detected)}" if post_change_detected else "change=?"
        if chaos_type == "color_swap":
            change_str += f" swap={color_swap_detected}"
        print(
            f"  [{run_num:>4d}/{args.runs}] {chaos_type:15s} | "
            f"pre={pre_accuracy:.0%} post={post_accuracy:.0%} | "
            f"{change_str}"
        )

    elapsed = time.time() - start

    # ── Final summary ──
    if not results:
        summary = {"runs": 0, "error": "no results", "completed": False}
    else:
        by_type = {}
        for r in results:
            ct = r["variation"]
            if ct not in by_type:
                by_type[ct] = []
            by_type[ct].append(r)

        type_summaries = {}
        for ct, runs_list in by_type.items():
            pre_accs = [r["mean_pre_chaos_accuracy"] for r in runs_list]
            post_accs = [r["mean_post_chaos_accuracy"] for r in runs_list]
            change_rates = [1.0 if r["any_change_detected"] else 0.0 for r in runs_list]
            color_swap_rates = [
                1.0 if r.get("color_swap_explicitly_detected") else 0.0
                for r in runs_list if r.get("color_swap_explicitly_detected") is not None
            ]
            pre_zones = [r.get("pre_chaos_zone_accuracy", 0) for r in runs_list]
            post_zones = [r.get("post_chaos_zone_accuracy", 0) for r in runs_list]
            lats = [r["mean_latency_ms"] for r in runs_list]

            type_summaries[ct] = {
                "n_runs": len(runs_list),
                "mean_pre_accuracy": round(sum(pre_accs) / len(pre_accs), 4),
                "mean_post_accuracy": round(sum(post_accs) / len(post_accs), 4),
                "mean_drop": round(
                    sum(pre_accs) / len(pre_accs) - sum(post_accs) / len(post_accs), 4
                ),
                "change_detection_rate": round(sum(change_rates) / len(change_rates), 4),
                "color_swap_explicit_rate": round(
                    sum(color_swap_rates) / len(color_swap_rates), 4
                ) if color_swap_rates else None,
                "mean_pre_zone_accuracy": round(sum(pre_zones) / len(pre_zones), 4) if pre_zones else 0,
                "mean_post_zone_accuracy": round(sum(post_zones) / len(post_zones), 4) if post_zones else 0,
                "mean_latency_ms": round(sum(lats) / len(lats), 1),
            }

        # Color swap explicit detection
        color_swap_all = [
            1.0 if r.get("color_swap_explicitly_detected") else 0.0
            for r in results if r.get("color_swap_explicitly_detected") is not None
        ]

        summary = {
            "runs": len(results),
            "experiment": "stateful_chaos",
            "types_tested": len(type_summaries),
            "by_type": type_summaries,
            "overall_change_detection_rate": round(
                sum(1.0 if r["any_change_detected"] else 0.0 for r in results) / len(results),
                4,
            ),
            "color_swap_explicit_rate": (
                round(sum(color_swap_all) / len(color_swap_all), 4) if color_swap_all else None
            ),
            "error_count": errors,
            "total_api_calls": total_api_calls,
            "elapsed_s": round(elapsed, 1),
            "completed": errors < 20,
        }

    print("\n\n" + "=" * 70)
    print("STATEFUL CHAOS EXPERIMENT RESULTS")
    print("=" * 70)

    if "type_summaries" in summary:
        for ct, ts in summary["by_type"].items():
            print(f"\n  {ct}:")
            print(f"    Runs:               {ts['n_runs']}")
            print(f"    Pre-chaos accuracy: {ts['mean_pre_accuracy']:.1%}")
            print(f"    Post-chaos accuracy:{ts['mean_post_accuracy']:.1%}")
            print(f"    Drop:               {ts['mean_drop']:.1%}")
            print(f"    Change detection:   {ts['change_detection_rate']:.1%}")
            if ts.get("color_swap_explicit_rate") is not None:
                print(f"    Color swap explicit: {ts['color_swap_explicit_rate']:.1%}")
            if ts.get("mean_pre_zone_accuracy") is not None:
                print(f"    Pre-zone accuracy:  {ts['mean_pre_zone_accuracy']:.1%}")
                print(f"    Post-zone accuracy: {ts['mean_post_zone_accuracy']:.1%}")
            print(f"    Latency:            {ts['mean_latency_ms']:.0f}ms")

    print(f"\n\n  Overall change detection rate: {summary.get('overall_change_detection_rate', 'N/A'):.1%}")
    if summary.get("color_swap_explicit_rate") is not None:
        print(f"  Color swap explicit mention:  {summary['color_swap_explicit_rate']:.1%}")
    print(f"  Total API calls:   {summary.get('total_api_calls', 0)}")
    print(f"  Elapsed:           {summary.get('elapsed_s', 0):.0f}s")
    print(f"  Completed:         {summary.get('completed', False)}")
    print(f"\n  BASELINE COMPARISON (no state):")
    print(f"    color_swap: pre=100% post=33.3% change_detection=0%")
    print(f"    teleport:   pre=100% post=100% change_detection=0%")
    print(f"    appear:     pre=100% post=100%")
    print(f"    disappear:  pre=100% post=100%")
    print("=" * 70)

    if args.output:
        with open(str(args.output) + ".summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
