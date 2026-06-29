"""Experiment: Can Gemma identify objects under partial occlusion?

Places 2 overlapping objects in the same zone.
Tests: does Gemma report 1 object or 2?
Does the description merge the two objects?
Are zone assignments correct for overlapping objects?

Usage:
    python scripts/exp_occlusion.py --runs 100 --output results.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.client import CerebrasClient

WIDTH, HEIGHT = 384, 384
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]
cw, ch = WIDTH / 3, HEIGHT / 2
ZONE_CENTERS = {}
for i, lab in enumerate(ZONE_LABELS):
    r, c_idx = divmod(i, 3)
    ZONE_CENTERS[lab] = (int(c_idx * cw + cw / 2), int(r * ch + ch / 2))

SCENARIOS = [
    ("red_cup", "green_cup", "D", 0),
    ("red_cup", "green_cup", "D", 30),
    ("red_cup", "green_cup", "D", 60),
    ("blue_cup", "yellow_cube", "A", 30),
    ("red_cup", "blue_cup", "E", 50),
    ("red_cup", "green_cup", "B", 40),
    ("blue_cup", "cracked_cup", "F", 30),
]

OBJECT_COLORS = {
    "red_cup": (210, 60, 60), "blue_cup": (60, 90, 210),
    "green_cup": (60, 180, 60), "yellow_cube": (220, 200, 40),
    "cracked_cup": (200, 175, 120),
}
OBJECT_SHAPES = {
    "red_cup": "round", "blue_cup": "round", "green_cup": "round",
    "yellow_cube": "square", "cracked_cup": "round",
}

IDENTIFY_SCHEMA = {
    "name": "identify_objects", "strict": True,
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
                        "zone": {"type": "string", "enum": ["A","B","C","D","E","F"]},
                        "shape": {"type": "string"},
                    },
                    "required": ["description", "color", "zone", "shape"],
                    "additionalProperties": False,
                },
            },
            "total_objects_visible": {"type": "integer"},
            "are_any_objects_overlapping": {"type": "boolean"},
        },
        "required": ["observed_objects", "total_objects_visible", "are_any_objects_overlapping"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are a robot vision system analyzing a tabletop workspace with a Zone A-F grid overlaid. Identify every object: its color, shape, and zone. Check carefully if any objects are touching or overlapping."""


def render(objects):
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
        x, y, r = obj["x"], obj["y"], 26
        color = obj["color"]
        if obj.get("shape") == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(35, 35, 35), width=2)
    d.line([(WIDTH // 2 - 16, 20), (WIDTH // 2 + 16, 20)], fill=(45, 120, 205), width=4)
    d.line([(WIDTH // 2, 20), (WIDTH // 2, 20 - 24)], fill=(45, 120, 205), width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    client = CerebrasClient()
    results = []
    errors = 0

    for run_num in range(1, args.runs + 1):
        s = SCENARIOS[(run_num - 1) % len(SCENARIOS)]
        oid1, oid2, zone, overlap_pct = s
        zx, zy = ZONE_CENTERS[zone]
        radius = 26
        if overlap_pct == 0:
            dx, dy = radius * 2 + 8, 0
        else:
            separation = 2 * radius * (1 - overlap_pct / 100)
            dx, dy = separation, random.randint(-10, 10)

        objects_data = [
            {"id": oid1, "x": zx - dx // 2, "y": zy + dy,
             "color": OBJECT_COLORS[oid1], "shape": OBJECT_SHAPES[oid1], "zone": zone},
            {"id": oid2, "x": zx + dx // 2, "y": zy - dy,
             "color": OBJECT_COLORS[oid2], "shape": OBJECT_SHAPES[oid2], "zone": zone},
        ]

        image_b64 = render(objects_data)
        prompt = "Identify all objects, their colors, shapes, zones, and whether any are overlapping."

        t0 = time.perf_counter()
        try:
            result = client.image_chat(
                prompt=prompt, image_b64=image_b64,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0, max_tokens=500,
                response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA},
            )
            parsed = json.loads(result.content)
            observed = parsed.get("observed_objects", [])
            reported_count = parsed.get("total_objects_visible", 0)
            detected_overlap = parsed.get("are_any_objects_overlapping", False)

            correct_count = 1 if reported_count == 2 else 0
            detected_overlap_correct = (detected_overlap == (overlap_pct > 0))
            merged = (len(observed) == 1 and overlap_pct > 0)

            entry = {
                "run": run_num, "scenario": f"{oid1}_{oid2}_{overlap_pct}pct",
                "overlap_pct": overlap_pct, "n_expected": 2,
                "n_reported": reported_count,
                "count_correct": correct_count,
                "detected_overlap_correct": detected_overlap_correct,
                "merged_single_object": merged,
                "n_observed_listings": len(observed),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            }
            results.append(entry)
            if args.output:
                with open(args.output, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            errors += 1
            if errors >= 20:
                break

    summary = {
        "runs": len(results), "experiment": "occlusion",
        "count_accuracy": round(sum(r["count_correct"] for r in results) / len(results), 4),
        "overlap_detection_accuracy": round(
            sum(1 for r in results if r["detected_overlap_correct"]) / len(results), 4),
        "merged_rate": round(
            sum(1 for r in results if r["merged_single_object"]) / len(results), 4),
    }
    print(f"RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
