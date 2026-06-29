"""Probe robosuite camera images to tune the perceptor: dtype/range, saturation
coverage, and contour-area distribution across candidate cameras."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CAMS = ["birdview", "agentview", "frontview", "sideview"]
H = W = 384


def main():
    import robosuite as suite
    env = suite.make(
        "PickPlace", robots="Panda", has_renderer=False,
        has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=CAMS, camera_heights=H, camera_widths=W, camera_depths=True,
    )
    obs = env.reset()
    gt = {k: obs[k] for k in ["Can_pos", "Milk_pos", "Bread_pos", "Cereal_pos"] if k in obs}
    print("GT object positions:", {k: [round(float(x), 3) for x in v] for k, v in gt.items()})

    for cam in CAMS:
        img = obs.get(f"{cam}_image")
        if img is None:
            print(f"\n[{cam}] no image"); continue
        print(f"\n[{cam}] dtype={img.dtype} shape={img.shape} "
              f"min={img.min()} max={img.max()} mean={img.mean():.1f}")
        rgb = img if img.dtype == np.uint8 else np.clip(img * (255 if img.max() <= 1.01 else 1), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        table_v = int(np.median(v))
        for st in (30, 45, 60):
            mask = ((s > st) | (np.abs(v.astype(int) - table_v) > 55)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = sorted((int(cv2.contourArea(c)) for c in cnts), reverse=True)[:8]
            cov = 100 * mask.mean() / 255
            print(f"   sat>{st}: coverage={cov:.1f}%  n_contours={len(cnts)}  top_areas={areas}")
    env.close()


if __name__ == "__main__":
    main()
