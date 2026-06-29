"""Measure how well pixel-only perception localizes objects in 3D.

Proves the integrity fix: object positions come from camera pixels + depth
back-projection (perception.py), NEVER from sim ground truth. Ground truth is
used here ONLY to score the error.

Usage:
    uv run python scripts/perception_eval.py
Writes overnight_results/perception/eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.web.lib.perception import ColorContourPerceptor, perceive  # noqa: E402

# (robosuite env name, list of ground-truth obs pos keys)
SCENES = [
    ("Lift", ["cube_pos"]),
    ("PickPlace", ["Can_pos", "Milk_pos", "Bread_pos", "Cereal_pos"]),
    ("Stack", ["cubeA_pos", "cubeB_pos"]),
]
CAMERA = "birdview"
H = W = 384


def make_env(env_name: str):
    import robosuite as suite
    return suite.make(
        env_name, robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=[CAMERA],
        camera_heights=H, camera_widths=W, camera_depths=True,
    )


def nearest(gt_xyz, dets):
    best, bd = None, float("inf")
    for d in dets:
        if d.world_xyz is None:
            continue
        e = float(np.linalg.norm(np.array(d.world_xyz) - np.array(gt_xyz)))
        if e < bd:
            bd, best = e, d
    return best, bd


def main():
    perceptor = ColorContourPerceptor()
    report = {"camera": CAMERA, "detector": "cv2_color_contour", "scenes": {}}

    for env_name, gt_keys in SCENES:
        try:
            env = make_env(env_name)
            obs = env.reset()
        except Exception as e:
            report["scenes"][env_name] = {"error": f"env make/reset failed: {e}"}
            print(f"[{env_name}] FAILED: {e}")
            continue

        sm = perceive(env.sim, obs, CAMERA, perceptor, H, W)
        rows = []
        for k in gt_keys:
            gt = obs.get(k)
            if gt is None:
                rows.append({"object": k, "error": "no ground truth in obs"})
                continue
            gt = np.asarray(gt, dtype=float)
            det, err = nearest(gt, sm.detections)
            rows.append({
                "object": k,
                "gt_xyz": [round(float(v), 3) for v in gt],
                "matched_det_xyz": [round(v, 3) for v in det.world_xyz] if det else None,
                "matched_color": det.color_name() if det else None,
                "error_cm": round(err * 100, 1) if det else None,
            })
        errs = [r["error_cm"] for r in rows if r.get("error_cm") is not None]
        report["scenes"][env_name] = {
            "n_detections": len(sm.detections),
            "n_ground_truth": len(gt_keys),
            "objects": rows,
            "mean_error_cm": round(float(np.mean(errs)), 1) if errs else None,
            "median_error_cm": round(float(np.median(errs)), 1) if errs else None,
        }
        print(f"[{env_name}] dets={len(sm.detections)} "
              f"mean_err={report['scenes'][env_name]['mean_error_cm']}cm")
        for r in rows:
            print(f"    {r['object']}: {r.get('error_cm')}cm "
                  f"(color={r.get('matched_color')})")
        try:
            env.close()
        except Exception:
            pass

    out = _ROOT / "overnight_results" / "perception" / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"RESULT:{json.dumps({k: v.get('mean_error_cm') for k, v in report['scenes'].items()})}")


if __name__ == "__main__":
    main()
