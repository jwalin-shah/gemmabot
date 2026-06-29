"""Visualize the no-cheat pipeline: what we SEND Gemma (image + numbered masks,
no names) and what Gemma FIGURES OUT (per-mask identity + chosen target + plan).

Produces an annotated PNG: the camera image with each perceived mask marked
#0..#n (geometry only), then Gemma's own label drawn on each, the chosen target
highlighted, and its plan printed.

Usage:
    uv run python scripts/perception_viz.py --task PickPlace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.client import CerebrasClient                       # noqa: E402
from src.web.lib.imaging import fix_img, img_to_b64, make_composite, overlay_grid  # noqa: E402
from src.web.lib.perception import perceive_instances       # noqa: E402

H = W = 384
CAMERA = "birdview"
TASKS = {"Lift": "lift the cube", "Stack": "pick up a cube",
         "PickPlace": "pick up one of the grocery items"}

VIZ_SCHEMA = {
    "name": "scene_understanding",
    "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "identifications": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": False,
                          "properties": {"index": {"type": "integer"},
                                         "label": {"type": "string"}},
                          "required": ["index", "label"]},
            },
            "target_index": {"type": "integer"},
            "plan": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["identifications", "target_index", "plan", "reasoning"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlace", choices=list(TASKS))
    args = ap.parse_args()
    import robosuite as suite
    env = suite.make(args.task, robots="Panda", has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names=[CAMERA, "frontview", "robot0_eye_in_hand"],
                     camera_heights=H, camera_widths=W,
                     camera_depths=True, camera_segmentations="instance")
    obs = env.reset()

    sm = perceive_instances(env.sim, env.model, obs, CAMERA, H, W)
    _SKIP = ("visual", "bin", "table", "wall", "floor", "mount", "peg")
    dets = [d for d in sm.detections if d.label and not any(s in d.label.lower() for s in _SKIP)]

    # state map WITHOUT names (geometry + color hint only)
    lines = ["PERCEIVED MASKS (no names — identify each from the image):"]
    for i, d in enumerate(dets):
        x, y, z = d.world_xyz
        lines.append(f"  #{i}: color~{d.color_name()}, world=({x:+.3f},{y:+.3f},{z:+.3f})")
    statemap = "\n".join(lines)

    # Multi-view composite for Gemma's identification (3 angles), while perception
    # geometry uses the calibrated birdview only.
    composite = make_composite(
        overlay_grid(fix_img(obs[f"{CAMERA}_image"])),
        fix_img(obs["frontview_image"]),
        fix_img(obs["robot0_eye_in_hand_image"]),
    )
    img_b64 = img_to_b64(composite)
    prompt = (f"Goal: {TASKS[args.task]}.\n\n{statemap}\n\n"
              "The image shows three camera views (top-down with zone grid, front, and "
              "gripper close-up). Identify EACH numbered mask from the views (its real "
              "label, e.g. soda can / milk carton / cereal box / bread), then choose the "
              "target_index to grasp and give a short plan.")
    res = CerebrasClient().image_chat(
        prompt=prompt, image_b64=img_b64, temperature=0.0, seed=42, max_tokens=400,
        response_format={"type": "json_schema", "json_schema": VIZ_SCHEMA})
    out = json.loads(res.content)
    labels = {it["index"]: it["label"] for it in out.get("identifications", [])}
    target = out.get("target_index", 0)
    print("GEMMA UNDERSTANDING:", json.dumps(out, indent=2))

    # annotate the display image
    disp = Image.fromarray(fix_img(obs[f"{CAMERA}_image"])).convert("RGB").resize((W * 2, H * 2))
    d = ImageDraw.Draw(disp)
    for i, det in enumerate(dets):
        cyd = (H - 1 - det.cy) * 2          # array(OpenGL) -> display row, x2 scale
        cxd = det.cx * 2
        chosen = (i == target)
        col = (255, 60, 60) if chosen else (60, 220, 90)
        r = 16
        d.ellipse([cxd - r, cyd - r, cxd + r, cyd + r], outline=col, width=4)
        tag = f"#{i}: {labels.get(i, '?')}" + ("  <= TARGET" if chosen else "")
        d.text((cxd + r + 3, cyd - 8), tag, fill=col)
    d.text((8, 8), f"Gemma plan: {out.get('plan','')[:90]}", fill=(255, 255, 0))
    outp = _ROOT / "overnight_results" / "videos" / f"viz_{args.task}.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    disp.save(outp)
    print(f"VIZ: {outp}")
    env.close()


if __name__ == "__main__":
    main()
