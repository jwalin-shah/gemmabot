"""Worker A: Pure Vision — Zone Identification Accuracy.

Tests whether Gemma can identify objects by zone from a composite image
(3 camera angles + zone grid overlay) with NO coordinates, NO text labels.

Uses forced structured output (response_format: json_schema) at temperature 0.0.

Output: RESULT: {json} line to stdout at end, JSONL results to WORKER_OUTPUT.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient


# ─── Config ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
RUNS = int(os.environ.get("WORKER_RUNS", "200"))
RESUME_FROM = int(os.environ.get("WORKER_RESUME", "0"))
OUTPUT_PATH = os.environ.get("WORKER_OUTPUT", "")

# ─── Object catalog ──────────────────────────────────────────────────────
OBJECT_TEMPLATES = [
    {"id": "red_cup", "label": "a bright red cup", "color": (210, 60, 60), "radius": 26, "attribute": ""},
    {"id": "blue_cup", "label": "a bright blue cup", "color": (60, 90, 210), "radius": 26, "attribute": ""},
    {"id": "green_cup", "label": "a bright green cup", "color": (60, 180, 60), "radius": 26, "attribute": ""},
    {"id": "cracked_cup", "label": "a cracked tan/brown cup", "color": (200, 175, 120), "radius": 26, "attribute": "cracked"},
    {"id": "yellow_block", "label": "a bright yellow block", "color": (220, 200, 40), "radius": 26, "attribute": ""},
]

# ─── Scene layouts (object → zone mapping) ───────────────────────────────
SCENE_LAYOUTS = [
    {"red_cup": "D", "blue_cup": "E", "cracked_cup": "B"},
    {"red_cup": "A", "blue_cup": "C", "green_cup": "E"},
    {"red_cup": "E", "cracked_cup": "C", "yellow_block": "D"},
    {"blue_cup": "F", "green_cup": "A", "cracked_cup": "D"},
    {"red_cup": "D", "blue_cup": "B", "green_cup": "F", "cracked_cup": "C"},
    {"red_cup": "A", "yellow_block": "B", "cracked_cup": "F"},
    {"blue_cup": "C", "cracked_cup": "E", "yellow_block": "A", "green_cup": "D"},
    {"red_cup": "F", "blue_cup": "A", "green_cup": "C"},
    {"red_cup": "B", "cracked_cup": "D", "yellow_block": "F"},
    {"red_cup": "D", "blue_cup": "F", "cracked_cup": "B", "green_cup": "A", "yellow_block": "C"},
]

# Zone center coordinates for the 384x384 grid
cw, ch = WIDTH / 3, HEIGHT / 2
ZONE_CENTERS = {}
for i, lab in enumerate(ZONE_LABELS):
    r, c = divmod(i, 3)
    ZONE_CENTERS[lab] = (int(c * cw + cw / 2), int(r * ch + ch / 2))


# ─── Structured output schema ────────────────────────────────────────────
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
            "reasoning": {"type": "string"},
        },
        "required": ["observed_objects", "total_objects_visible", "reasoning"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system. You see a composite camera image showing
a tabletop workspace from multiple angles (overhead, front, wrist camera) with a Zone A-F grid.

Identify every object you can see. For each object describe its visual appearance — color,
shape, and which zone (A-F) it is in.

Rules:
- Only describe what you actually see in the image.
- Do not make up objects.
- Each zone letter is shown on the overhead view.

Output the list of observed objects with their zones."""


# ─── Render scene (no text labels!) ──────────────────────────────────────

def render_scene(objects_in_world: list[dict]) -> str:
    """Render scene with colored circles + zone grid. NO text labels on objects."""
    from PIL import Image, ImageDraw
    import base64, io
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
    d = ImageDraw.Draw(img)

    # Zone grid
    for c in range(1, 3):
        d.line([(int(c * cw), 0), (int(c * cw), HEIGHT)], fill=(200, 200, 210), width=1)
    d.line([(0, int(ch)), (WIDTH, int(ch))], fill=(200, 200, 210), width=1)
    for i, lab in enumerate(ZONE_LABELS):
        r_idx, c_idx = divmod(i, 3)
        d.text((int(c_idx * cw + 5), int(r_idx * ch + 4)), f"Zone {lab}", fill=(170, 170, 180))

    # Objects — colored circles only, NO text
    for obj in objects_in_world:
        color = obj["color"]
        r = obj.get("radius", 26)
        x, y = obj["x"], obj["y"]
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)
        # Cracked visual marker (just a line, not text)
        if obj.get("attribute") == "cracked":
            d.line([(x - 11, y - 13), (x + 4, y), (x - 7, y + 13)], fill=(20, 20, 20), width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def place_objects_in_scene(layout: dict) -> list[dict]:
    """Convert zone layout to pixel positions."""
    objects = []
    for oid, zone in layout.items():
        template = next(t for t in OBJECT_TEMPLATES if t["id"] == oid)
        zx, zy = ZONE_CENTERS[zone]
        # Jitter within zone to simulate real positioning
        jitter_x = random.randint(-30, 30)
        jitter_y = random.randint(-20, 20)
        obj = {
            "id": oid,
            "x": zx + jitter_x,
            "y": zy + jitter_y,
            "color": template["color"],
            "radius": template["radius"],
            "attribute": template.get("attribute", ""),
            "expected_zone": zone,
        }
        objects.append(obj)
    return objects


# ─── Experiment ──────────────────────────────────────────────────────────

def run_experiment():
    client = CerebrasClient()
    results = []
    error_count = 0

    scene_count = len(SCENE_LAYOUTS)
    reps_per_scene = max(1, RUNS // scene_count)

    for scene_idx, layout in enumerate(SCENE_LAYOUTS):
        if scene_idx * reps_per_scene >= RUNS:
            break

        for rep in range(reps_per_scene):
            run_num = scene_idx * reps_per_scene + rep + 1
            if run_num <= RESUME_FROM:
                continue
            if run_num > RUNS:
                break

            objects = place_objects_in_scene(layout)
            image_b64 = render_scene(objects)

            prompt = (
                "Look at this camera image of a tabletop workspace. "
                "Identify every object you can see — its color, shape, and which zone (A-F) it is in. "
                "Describe what you actually see visually."
            )

            t0 = time.perf_counter()
            try:
                result = client.image_chat(
                    prompt=prompt,
                    image_b64=image_b64,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=500,
                    response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA},
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                data = json.loads(result.content)
                observed = data.get("observed_objects", [])
                observed_count = data.get("total_objects_visible", 0)

                # Score: zone match per object
                zone_matches = 0
                zone_total = 0
                for obs_obj in observed:
                    obs_zone = obs_obj.get("zone", "")
                    # Match by description keywords
                    obs_desc = obs_obj.get("description", "").lower()
                    best_match = None
                    for obj in objects:
                        oid_lower = obj["id"].lower().replace("_", " ")
                        # Check if description mentions key features
                        if any(word in obs_desc for word in oid_lower.split()):
                            best_match = obj
                            break
                        # Color match
                        color_words = {"red": (210,60,60), "blue": (60,90,210),
                                       "green": (60,180,60), "yellow": (220,200,40),
                                       "tan": (200,175,120), "brown": (200,175,120)}
                        for cname, crgb in color_words.items():
                            if cname in obs_desc and obj["color"] == crgb:
                                best_match = obj
                                break
                        if best_match:
                            break

                    zone_total += 1
                    if best_match and obs_zone == best_match["expected_zone"]:
                        zone_matches += 1
                    elif best_match:
                        pass  # wrong zone — still track

                zone_accuracy = zone_matches / max(zone_total, 1)
                count_error = abs(observed_count - len(objects))

                entry = {
                    "scene": scene_idx,
                    "rep": rep,
                    "run": run_num,
                    "n_objects": len(objects),
                    "observed_count": observed_count,
                    "count_error": count_error,
                    "zone_matches": zone_matches,
                    "zone_total": zone_total,
                    "zone_accuracy": round(zone_accuracy, 4),
                    "latency_ms": round(latency_ms, 1),
                    "success": zone_accuracy >= 0.5,
                    "error": None,
                }
                results.append(entry)

                # Log progress
                if run_num % 10 == 0 or run_num == 1:
                    acc_str = f"{zone_accuracy:.0%}"
                    print(f"  [{run_num:>4d}/{RUNS}] {len(objects)} objects | zone acc: {acc_str} | {latency_ms:.0f}ms")

            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                error_count += 1
                entry = {
                    "scene": scene_idx, "rep": rep, "run": run_num,
                    "n_objects": len(objects), "error": str(e)[:200],
                    "latency_ms": round(latency_ms, 1), "success": False,
                }
                results.append(entry)
                if error_count > 20:
                    print(f"  ❌ Too many errors ({error_count}) — aborting")
                    break

            # Write incrementally
            if OUTPUT_PATH and results:
                with open(OUTPUT_PATH, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        if error_count > 20:
            break

    # Compute summary
    if not results:
        return {"runs": 0, "completed": False, "errors": ["no results"]}

    successes = [r for r in results if r.get("success")]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    zone_accs = [r["zone_accuracy"] for r in results if "zone_accuracy" in r]

    summary = {
        "runs": len(results),
        "completed": error_count <= 20,
        "success_count": len(successes),
        "success_rate": round(len(successes) / max(len(results), 1), 4),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
        "mean_zone_accuracy": round(sum(zone_accs) / max(len(zone_accs), 1), 4),
        "error_count": error_count,
    }
    return summary


if __name__ == "__main__":
    result = run_experiment()
    print(f"RESULT:{json.dumps(result)}")