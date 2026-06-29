"""End-to-end NO-CHEAT robot demo.

Proves the full thesis: object positions come from PERCEPTION (camera pixels +
depth + instance seg, ~1cm), Gemma reads the perceived state map + the image and
CHOOSES the target, and the arm grasps + lifts at the perceived coordinates.
The simulator's ground-truth object pose is used ONLY by the final success
check (the judge) — never as an input to perception or to Gemma.

Usage:
    uv run python scripts/nocheat_demo.py --task Lift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.client import CerebrasClient                       # noqa: E402
from src.web.lib.imaging import fix_img, img_to_b64         # noqa: E402
from src.web.lib.perception import perceive_instances       # noqa: E402

H = W = 384
CAMERA = "birdview"

# task -> (robosuite env, natural-language goal, ground-truth key for the judge)
TASKS = {
    "Lift": ("Lift", "lift the cube off the table", "cube_pos"),
    "Stack": ("Stack", "pick up a cube", "cubeA_pos"),
    "PickPlace": ("PickPlace", "pick up one of the grocery items", "Can_pos"),
}

CHOOSE_SCHEMA = {
    "name": "choose_target",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_index": {"type": "integer", "description": "Index (#) of the object to grasp, identified from the IMAGE."},
            "identified_as": {"type": "string", "description": "What you see that object to be (your identification from the image)."},
            "reasoning": {"type": "string"},
        },
        "required": ["target_index", "identified_as", "reasoning"],
    },
}

# Ground-truth keys per task — used ONLY by the success judge, never as input.
GT_KEYS = {
    "Lift": ["cube_pos"],
    "Stack": ["cubeA_pos", "cubeB_pos"],
    "PickPlace": ["Can_pos", "Milk_pos", "Bread_pos", "Cereal_pos"],
}


def _label_free_statemap(dets) -> str:
    """State map with NO simulator names — only geometry + sensor color hint."""
    lines = ["PERCEIVED OBJECTS (geometry from camera depth; color is a raw-pixel hint, "
             "NOT an identity — you must identify each object from the image):"]
    for i, d in enumerate(dets):
        x, y, z = d.world_xyz
        lines.append(f"  #{i}: color~{d.color_name()}, world=({x:+.3f}, {y:+.3f}, {z:+.3f})")
    return "\n".join(lines) + "\n"


_FRAMES: list = []          # captured video frames (frontview) when --record
_RECORD = False


def make_env(env_name):
    import robosuite as suite
    return suite.make(
        env_name, robots="Panda", has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=[CAMERA, "frontview", "agentview"],
        camera_heights=[H, H, 1024],
        camera_widths=[W, W, 1024],
        camera_depths=[True, True, False],
        camera_segmentations=["instance", "instance", None],
    )


def _capture(obs):
    if _RECORD:
        f = obs.get("agentview_image")
        if f is None:
            f = obs.get("frontview_image")
        if f is not None:
            _FRAMES.append(fix_img(f))


def servo(env, obs, adim, tx, ty, tz, grip, steps=70, tol=0.006):
    for _ in range(steps):
        ee = np.asarray(obs["robot0_eef_pos"], float)
        d = np.array([tx, ty, tz]) - ee
        dist = float(np.linalg.norm(d))
        if dist < tol:
            break
        act = np.zeros(adim, dtype=np.float32)
        act[0:3] = np.clip(d / max(dist, 1e-3) * min(dist * 10, 1.0), -1, 1)
        if adim >= 7:
            act[-1] = grip
        obs, _, _, _ = env.step(act)
        _capture(obs)
    return obs, float(np.linalg.norm(np.array([tx, ty, tz]) - np.asarray(obs["robot0_eef_pos"], float)))


def grasp_and_lift(env, obs, adim, tx, ty, tz):
    obs, _ = servo(env, obs, adim, tx, ty, tz + 0.10, -1.0)      # approach, open
    obs, _ = servo(env, obs, adim, tx, ty, tz - 0.02, -1.0)      # descend, open
    for _ in range(20):                                          # close
        act = np.zeros(adim, dtype=np.float32); act[-1] = 1.0
        obs, _, _, _ = env.step(act)
        _capture(obs)
    obs, _ = servo(env, obs, adim, tx, ty, tz + 0.20, 1.0)       # lift
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift", choices=list(TASKS))
    ap.add_argument("--record", action="store_true", help="Save an mp4 of the run.")
    args = ap.parse_args()
    global _RECORD
    _RECORD = args.record
    env_name, goal, gt_key = TASKS[args.task]

    env = make_env(env_name)
    obs = env.reset()
    adim = env.action_spec[0].shape[0]

    # 1) PERCEIVE (no ground truth). Drop visual-only duplicate geoms, bins,
    #    and table — keep only real graspable objects.
    sm = perceive_instances(env.sim, env.model, obs, CAMERA, H, W)
    _SKIP = ("visual", "bin", "table", "wall", "floor", "mount", "peg")
    sm.detections = [
        d for d in sm.detections
        if d.label and not any(s in d.label.lower() for s in _SKIP)
    ]
    statemap = _label_free_statemap(sm.detections)
    print("PERCEIVED (geometry only, NO names — Gemma must identify from the image):")
    print(statemap)
    if not sm.detections:
        print("No objects perceived — abort."); env.close(); return

    # 2) GEMMA identifies objects FROM THE IMAGE and chooses a target by index.
    #    No simulator names are provided — only geometry + a raw color hint.
    img_b64 = img_to_b64(fix_img(obs[f"{CAMERA}_image"]))
    prompt = (
        f"You are a robot arm. Goal: {goal}.\n\n{statemap}\n"
        "Look at the camera IMAGE and identify each numbered object yourself. "
        "Then pick the index (#) of the object to grasp to achieve the goal."
    )
    client = CerebrasClient()
    res = client.image_chat(
        prompt=prompt, image_b64=img_b64, temperature=0.0, seed=42, max_tokens=200,
        response_format={"type": "json_schema", "json_schema": CHOOSE_SCHEMA},
    )
    choice = json.loads(res.content)
    print(f"GEMMA CHOSE: {choice}")

    # 3) Resolve Gemma's index to a PERCEIVED detection (never ground truth).
    idx = int(choice.get("target_index", 0))
    idx = idx if 0 <= idx < len(sm.detections) else 0
    det = sm.detections[idx]
    tx, ty, tz = det.world_xyz
    print(f"Executing grasp at PERCEIVED #{idx} (Gemma id: {choice.get('identified_as')}) "
          f"= ({tx:+.3f},{ty:+.3f},{tz:+.3f})")

    # 4) EXECUTE grasp + lift at perceived coordinates.
    #    For scoring only, find which ground-truth object is nearest the grasp
    #    point (ground truth = judge only, never an input).
    cands = [k for k in GT_KEYS.get(args.task, [gt_key]) if k in obs]
    gt_key = min(cands, key=lambda k: np.linalg.norm(np.asarray(obs[k], float) - np.array(det.world_xyz))) if cands else gt_key
    z0 = float(np.asarray(obs[gt_key], float)[2])   # ground truth: judge only
    obs = grasp_and_lift(env, obs, adim, tx, ty, tz)
    zf = float(np.asarray(obs[gt_key], float)[2])
    lifted = (zf - z0) > 0.05

    print("\n=== RESULT ===")
    print(f"task={args.task}  gemma_target={choice.get('target_object')}  "
          f"perceived_xyz=({tx:+.3f},{ty:+.3f},{tz:+.3f})")
    print(f"object_z {z0:.3f} -> {zf:.3f}   LIFTED={lifted}   (verdict via ground-truth judge)")
    print(f"RESULT:{json.dumps({'task': args.task, 'lifted': lifted, 'gemma_id': choice.get('identified_as')})}")

    if _RECORD and _FRAMES:
        import imageio
        out = _ROOT / "overnight_results" / "videos" / f"nocheat_{args.task}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), _FRAMES[::2], fps=20)
        print(f"VIDEO: {out}  ({len(_FRAMES[::2])} frames)")
    env.close()


if __name__ == "__main__":
    main()
