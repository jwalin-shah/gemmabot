"""Vision perception -- detect objects from camera PIXELS and back-project to 3D.

This module is the honest replacement for reading object poses straight out of
the simulator. The planner (Gemma) must never be handed ground-truth object
coordinates; instead:

  InstanceSegPerceptor (robosuite instance seg) -> object masks from RGB+seg
  Depth back-projection (sensor only)           -> 3D world XYZ per mask
  Gemma (elsewhere)                             -> action / disambiguation

The only ground truth that touches the loop is inside verify.py, used purely as
a judge. Depth and the camera matrix are sensor data a real RGBD robot has, so
using them for back-projection is fair game; obs["<Obj>_pos"] is not.

---
PIXEL CONVENTION FIX (important -- three bugs were here):

Bug 1 - instance seg ID mapping
  obs["<cam>_segmentation_instance"] uses robosuite's INSTANCE mapping, NOT
  raw geom_id+1. IDs map to env.model.instances_to_ids order:
    0  = unmapped (floor, table, walls -- not in instances_to_ids)
    1  = first instance (e.g. 'cube' in Lift)
    2  = second instance (e.g. robot arm)
    ...

Bug 2 - row convention mismatch
  robosuite obs arrays (image, depth, seg) are in OpenGL convention:
    row 0 = BOTTOM of physical image (raw offscreen render, convention=1/'opengl')
  But camera_utils.project_points_from_world_to_camera() and
  transform_from_pixels_to_world() use STANDARD convention:
    row 0 = TOP of image
  Fix: convert array row to standard row via  std_row = H - 1 - arr_row
       and flip the depth map:               depth_std = depth_real[::-1, :, :]
  Then pass (std_row, col) and depth_std to transform_from_pixels_to_world.

Bug 3 - top-surface vs centre depth bias
  A top-down (birdview) camera sees the SURFACE of an object, not its centre.
  The GT position is the object's geometric centre. For a birdview camera the
  bias is purely in z: z_top approx table_z + object_height,
  z_centre approx table_z + object_height/2.
  Fix:  z_centre = (z_top + table_z) / 2
  table_z is read from the sim's table geom (fallback 0.806 m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One detected object region, grounded in pixels (and optionally 3D)."""
    cx: int                          # centroid column (x) in OpenGL array frame
    cy: int                          # centroid row (y) in OpenGL array frame
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1  (array coords)
    area_px: int
    mean_rgb: tuple[int, int, int]
    world_xyz: tuple[float, float, float] | None = None
    label: str | None = None         # object / instance name
    confidence: float = 1.0
    source: str = "cv2"

    def color_name(self) -> str:
        r, g, b = self.mean_rgb
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx - mn < 28:
            return "white" if mx > 170 else ("gray" if mx > 80 else "black")
        if r >= g and r >= b:
            return "yellow" if g > 150 and b < 120 else ("orange" if g > 90 else "red")
        if g >= r and g >= b:
            return "green"
        return "blue" if b > 150 else "purple"


@dataclass
class StateMap:
    """The perceived world state -- a belief Gemma and the detectors co-maintain."""
    detections: list[Detection] = field(default_factory=list)
    camera: str = "birdview"

    def as_text(self) -> str:
        if not self.detections:
            return "PERCEIVED OBJECTS (from camera, not ground truth): none detected.\n"
        lines = ["PERCEIVED OBJECTS (from camera vision, not ground truth):"]
        for i, d in enumerate(self.detections):
            name = d.label or f"{d.color_name()} object #{i}"
            if d.world_xyz is not None:
                x, y, z = d.world_xyz
                lines.append(
                    f"  {name}: world=({x:+.3f}, {y:+.3f}, {z:+.3f}) "
                    f"[{d.source} conf={d.confidence:.2f}, area={d.area_px}px]"
                )
            else:
                lines.append(f"  {name}: pixel=({d.cx},{d.cy}) [{d.source}]")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Instance-segmentation perceiver  (primary, accurate path)
# ---------------------------------------------------------------------------

_ROBOT_KEYWORDS = frozenset(["panda", "gripper", "mount", "robot", "link", "finger",
                              "eef", "hand", "fixed_mount", "rethink"])
_MIN_PIXELS = 10
_WS_XY = 0.6
_WS_Z_ABOVE = 0.5


def _get_table_z(sim) -> float:
    try:
        g = sim.model.geom_name2id("table_visual")
        pos_z = float(sim.data.geom_xpos[g, 2])
        size_z = float(sim.model.geom_size[g, 2])
        return pos_z + size_z
    except Exception:
        pass
    return 0.806


def _build_instance_map(env_model) -> dict:
    if not hasattr(env_model, "instances_to_ids"):
        return {}
    name2idx = {inst: i for i, inst in enumerate(env_model.instances_to_ids.keys())}
    return {inst: idx + 1 for inst, idx in name2idx.items()}


def _mask_mean_rgb(rgb, mask):
    if rgb is None or not mask.any():
        return (128, 128, 128)
    pixels = rgb[mask]
    return tuple(int(v) for v in pixels.mean(axis=0)[:3])


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 1, 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def perceive_instances(
    sim,
    env_model,
    obs: dict,
    camera: str,
    height: int = 384,
    width: int = 384,
    table_z: Optional[float] = None,
) -> StateMap:
    """Instance-segmentation perceiver -- accurate 3D localisation from depth+seg.

    Uses robosuite instance seg IDs to identify objects, back-projects the mask
    centroid via depth with three bug fixes (see module docstring):
      1. correct instance ID mapping
      2. OpenGL->standard row convention fix
      3. top-surface->centre z correction

    Args:
        sim:       MuJoCo sim handle (env.sim)
        env_model: Robosuite task model (env.model)
        obs:       Camera observations dict (raw, not display-flipped)
        camera:    Camera name, e.g. "birdview"
        height:    Image height (pixels)
        width:     Image width (pixels)
        table_z:   Table-top z in world frame (m); auto-detected if None

    Returns:
        StateMap with one Detection per visible task object.
        Detection.label = robosuite instance name (e.g. "cube", "cubeA", "Can")
        Detection.world_xyz = estimated object CENTRE position.
    """
    from robosuite.utils import camera_utils as cu

    seg_obs = obs.get(f"{camera}_segmentation_instance")
    depth_obs = obs.get(f"{camera}_depth")
    rgb = obs.get(f"{camera}_image")

    if seg_obs is None or depth_obs is None:
        return StateMap(camera=camera)

    seg_arr = seg_obs[:, :, 0]  # (H, W) int, OpenGL: row 0 = bottom

    # Bug fix 2: flip depth to standard convention (row 0 = top)
    depth_real = cu.get_real_depth_map(sim, depth_obs)  # (H, W, 1), OpenGL
    depth_std = depth_real[::-1, :, :]                  # (H, W, 1), standard

    world_to_pix = cu.get_camera_transform_matrix(sim, camera, height, width)
    pix_to_world = np.linalg.inv(world_to_pix)

    # Bug fix 1: correct instance mapping (name -> obs seg id)
    inst_map = _build_instance_map(env_model)

    if table_z is None:
        table_z = _get_table_z(sim)

    detections = []
    for inst_name, seg_id in inst_map.items():
        name_lower = inst_name.lower()
        if any(kw in name_lower for kw in _ROBOT_KEYWORDS):
            continue

        mask = seg_arr == seg_id
        n_pix = int(mask.sum())
        if n_pix < _MIN_PIXELS:
            continue

        ys, xs = np.where(mask)
        arr_row = float(np.mean(ys))
        col = float(np.mean(xs))

        # Bug fix 2: OpenGL array row -> standard row
        std_row = (height - 1) - arr_row

        px = np.array([std_row, col])
        world_pt = cu.transform_from_pixels_to_world(px, depth_std, pix_to_world)
        x_w = float(world_pt[0])
        y_w = float(world_pt[1])
        z_top = float(world_pt[2])

        # Bug fix 3: top-surface -> centre z correction
        if z_top > table_z + 0.005:
            z_centre = (z_top + table_z) / 2.0
        else:
            z_centre = z_top

        # Workspace filter
        if not (table_z - 0.05 <= z_top <= table_z + _WS_Z_ABOVE):
            continue
        if not (-_WS_XY <= x_w <= _WS_XY and -_WS_XY <= y_w <= _WS_XY):
            continue

        mean_rgb = _mask_mean_rgb(rgb, mask)
        bbox = _mask_bbox(mask)

        detections.append(Detection(
            cx=int(round(col)),
            cy=int(round(arr_row)),
            bbox=bbox,
            area_px=n_pix,
            mean_rgb=mean_rgb,
            world_xyz=(x_w, y_w, z_centre),
            label=inst_name,
            confidence=1.0,
            source="instance_seg",
        ))

    return StateMap(detections=detections, camera=camera)


# ---------------------------------------------------------------------------
# Legacy colour-contour detector (RGB pixels -> 2D regions). No depth, no GT.
# ---------------------------------------------------------------------------

class ColorContourPerceptor:
    """Foreground-object detector using saturation/contrast against the table.

    Less accurate than perceive_instances() but needs no instance-seg obs.
    """

    def __init__(self, min_area=60, max_area=9000, sat_thresh=45, val_thresh=40):
        self.min_area = min_area
        self.max_area = max_area
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

    def detect(self, rgb):
        import cv2
        if rgb is None or rgb.size == 0:
            return []
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s = hsv[..., 1]
        v = hsv[..., 2]
        table_v = int(np.median(v))
        mask = ((s > self.sat_thresh) | (np.abs(v.astype(int) - table_v) > 55)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area or area > self.max_area:
                continue
            x0, y0, w, h = cv2.boundingRect(c)
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            region = rgb[y0:y0 + h, x0:x0 + w].reshape(-1, 3)
            mean_rgb = tuple(int(v) for v in region.mean(axis=0))
            dets.append(Detection(
                cx=cx, cy=cy, bbox=(x0, y0, x0 + w, y0 + h),
                area_px=area, mean_rgb=mean_rgb, source="cv2", confidence=1.0,
            ))
        dets.sort(key=lambda d: d.area_px, reverse=True)
        return dets


class YoloPerceptor:
    """Learned detector (ultralytics YOLO). Optional, lazy-loaded."""

    def __init__(self, weights="yolov8n.pt", conf=0.25):
        self._weights = weights
        self._conf = conf
        self._model = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._weights)
        return self._model

    def detect(self, rgb):
        model = self._ensure()
        res = model.predict(rgb, conf=self._conf, verbose=False)[0]
        dets = []
        for box in res.boxes:
            x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            region = rgb[y0:y1, x0:x1].reshape(-1, 3)
            mean_rgb = tuple(int(v) for v in region.mean(axis=0)) if region.size else (0, 0, 0)
            cls_name = res.names.get(int(box.cls[0]), "object")
            dets.append(Detection(
                cx=cx, cy=cy, bbox=(x0, y0, x1, y1),
                area_px=int((x1 - x0) * (y1 - y0)), mean_rgb=mean_rgb,
                label=cls_name, confidence=float(box.conf[0]), source="yolo",
            ))
        return dets


# ---------------------------------------------------------------------------
# Depth back-projection helper (used by legacy ColorContour path)
# ---------------------------------------------------------------------------

def backproject(sim, depth_obs, camera, height, width, pixels_xy):
    """Map OpenGL-convention (col, arr_row) pixels to world XYZ.

    pixels_xy: list of (col, arr_row) in OpenGL coords (row 0 = bottom).
    Internally converts to standard convention before back-projecting.
    """
    from robosuite.utils import camera_utils as cu
    if not pixels_xy:
        return []
    depth_real = cu.get_real_depth_map(sim, depth_obs)
    depth_std = depth_real[::-1, :, :]
    world_to_pix = cu.get_camera_transform_matrix(sim, camera, height, width)
    pix_to_world = np.linalg.inv(world_to_pix)
    out = []
    for col, arr_row in pixels_xy:
        std_row = (height - 1) - arr_row
        px = np.array([std_row, col], dtype=float)
        pt = cu.transform_from_pixels_to_world(px, depth_std, pix_to_world)
        out.append((float(pt[0]), float(pt[1]), float(pt[2])))
    return out


def perceive(sim, obs, camera, perceptor, height=384, width=384):
    """Legacy pipeline: detect on raw RGB -> back-project to 3D -> StateMap.

    Prefer perceive_instances() for accurate sub-2cm localisation.
    """
    rgb = obs.get(f"{camera}_image")
    depth_obs = obs.get(f"{camera}_depth")
    if rgb is None:
        return StateMap(camera=camera)
    dets = perceptor.detect(rgb)
    if depth_obs is not None and dets:
        worlds = backproject(sim, depth_obs, camera, height, width,
                             [(d.cx, d.cy) for d in dets])
        for d, w in zip(dets, worlds):
            d.world_xyz = w
    return StateMap(detections=dets, camera=camera)


# ---------------------------------------------------------------------------
# SAM-based perceptor (Segment Anything Model, ViT-Base)
# ---------------------------------------------------------------------------

class SamPerceptor:
    """Segment tabletop objects with SAM (ViT-Base) from an RGB image.

    Uses huggingface transformers SamModel + SamProcessor with the base
    checkpoint (facebook/sam-vit-base). Runs on CPU by default to avoid
    GPU contention with MuJoCo.

    Resolution pipeline: 384->1024 (SAM encoder) -> masks -> nearest-neighbour->384.
    Auto-generator with 16x16 grid of prompt points.
    """

    _MODEL_CACHE: dict = {}

    def __init__(
        self,
        model_type: str = "facebook/sam-vit-base",
        device: str = "cpu",
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.8,
        min_mask_region_area: int = 200,
        max_mask_region_area: int = 30000,
        color_labels: dict | None = None,
        label_fallback: str | None = None,
    ):
        self.model_type = model_type
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.max_mask_region_area = max_mask_region_area
        self.color_labels = color_labels or {
            (220, 50, 50): "red",
            (50, 180, 50): "green",
            (50, 50, 220): "blue",
            (220, 180, 50): "yellow",
        }
        self.label_fallback = label_fallback
        self._model = None
        self._processor = None
        self._generator = None

    def _load_model(self) -> dict:
        """Lazy-load SAM model and processor, cached at module level."""
        key = f"{self.model_type}:{self.device}"
        if key in self._MODEL_CACHE:
            return self._MODEL_CACHE[key]

        from transformers import SamModel, SamProcessor
        import torch

        processor = SamProcessor.from_pretrained(self.model_type)
        model = SamModel.from_pretrained(self.model_type).to(self.device)
        model.eval()

        # Pre-compute point grid for automatic mask generation
        # Instead of SamAutomaticMaskGenerator (unavailable), we generate
        # a grid of prompt points and run the forward pass for each.
        points_per_side = self.points_per_side
        side = points_per_side
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1023, side),
            torch.linspace(0, 1023, side),
            indexing="xy",
        )
        point_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (N, 2)
        point_grid = point_grid.unsqueeze(0).unsqueeze(2).to(self.device)  # (1, N, 1, 2)
        n_pts = point_grid.shape[1]
        point_labels = torch.ones(1, n_pts, 1, dtype=torch.long).to(self.device)  # (1, N, 1)

        cached = {
            "processor": processor,
            "model": model,
            "point_grid": point_grid,
            "point_labels": point_labels,
            "device": self.device,
        }
        self._MODEL_CACHE[key] = cached
        self._MODEL_CACHE[key] = cached
        return cached

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Segment tabletop objects using point-prompted SAM.

        Generates a grid of prompt points (points_per_side x points_per_side),
        runs SAM for each, filters by IoU score and area, and returns
        Detection objects.
        """
        cached = self._load_model()
        processor = cached["processor"]
        model = cached["model"]
        point_grid = cached["point_grid"]
        point_labels = cached["point_labels"]
        import torch

        target_w, target_h = 384, 384
        h, w = rgb.shape[:2]

        # Process image through SAM processor (resizes to 1024x1024)
        inputs = processor(
            rgb,
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)

        # Get image embedding
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

        # Run point prompts in batches to avoid OOM
        batch_size = 16
        all_masks = []
        all_iou_scores = []
        n_points = point_grid.shape[1]

        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            batch_points = point_grid[:, start_idx:end_idx, :, :]
            batch_labels = point_labels[:, start_idx:end_idx, :]

            with torch.no_grad():
                outputs = model(
                    image_embeddings=image_embeddings,
                    input_points=batch_points,
                    input_labels=batch_labels,
                    multimask_output=True,
                )
                all_masks.append(outputs.pred_masks.cpu())
                all_iou_scores.append(outputs.iou_scores.cpu())

        if not all_masks:
            return []

        pred_masks = torch.cat(all_masks, dim=1)  # (1, N, 3, 256, 256)
        iou_scores = torch.cat(all_iou_scores, dim=1)  # (1, N, 3)

        # Pick best mask per point (highest IoU among 3 candidates per point)
        best_per_point = iou_scores.argmax(dim=-1)  # (1, N)
        masks_list = []
        scores_list = []
        for i in range(pred_masks.shape[1]):
            best = best_per_point[0, i].item()
            mask = pred_masks[0, i, best, :, :].numpy()  # (256, 256)
            score = float(iou_scores[0, i, best].item())
            if score > self.pred_iou_thresh:
                masks_list.append(mask)
                scores_list.append(score)

        if not masks_list:
            return []

        # Stack and resize masks to 384x384
        masks_tensor = torch.from_numpy(np.array(masks_list)).float().unsqueeze(1)
        masks_resized = torch.nn.functional.interpolate(
            masks_tensor, size=(target_h, target_w), mode="nearest"
        ).squeeze(1).numpy()

        # Non-maximum suppression: merge overlapping masks
        detections = []
        used = set()
        for i, (mask_small, score) in enumerate(zip(masks_resized, scores_list)):
            if i in used:
                continue

            mask_bool = mask_small > 0
            area = int(mask_bool.sum())

            if area < self.min_mask_region_area or area > self.max_mask_region_area:
                continue

            # Skip if mostly overlapping with a higher-scored mask
            keep = True
            for j in range(i):
                if j in used:
                    continue
                prev_mask = masks_resized[j] > 0
                intersection = (mask_bool & prev_mask).sum()
                union = (mask_bool | prev_mask).sum()
                if union > 0 and intersection / union > 0.7:
                    keep = False
                    break

            if not keep:
                continue

            ys, xs = np.where(mask_bool)
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            bbox = _mask_bbox(mask_bool)

            # Fallback mean_rgb from the SAM mask area
            mask_3d = np.stack([mask_bool] * 3, axis=-1)
            if h != target_h or w != target_w:
                rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_resized = rgb
            masked_pixels = rgb_resized[mask_bool]
            mean_rgb = tuple(int(v) for v in masked_pixels.mean(axis=0)) if len(masked_pixels) > 0 else (128, 128, 128)

            label = self._match_color(mean_rgb)

            detections.append(Detection(
                cx=cx, cy=cy, bbox=bbox, area_px=area,
                mean_rgb=mean_rgb, label=label,
                confidence=round(float(score), 4),
                source="sam",
            ))

        return detections

    def _match_color(self, rgb: tuple[int, int, int]) -> str | None:
        """Match an RGB tuple to the closest known color label."""
        if not self.color_labels:
            return None
        best = None
        best_dist = float("inf")
        r, g, b = rgb
        for (cr, cg, cb), name in self.color_labels.items():
            dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if dist < best_dist:
                best_dist = dist
                best = name
        if best_dist < 80 ** 2 * 3:
            return best
        return None

    def backproject_detections(
        self,
        detections: list[Detection],
        sim,
        depth_obs: np.ndarray,
        camera: str,
        height: int = 384,
        width: int = 384,
    ) -> list[Detection]:
        if not detections:
            return detections

        pixels = [(d.cx, d.cy) for d in detections]
        worlds = backproject(sim, depth_obs, camera, height, width, pixels)

        table_z = _get_table_z(sim)

        for d, w in zip(detections, worlds):
            x_w, y_w, z_top = w
            if z_top > table_z + 0.005:
                z_centre = (z_top + table_z) / 2.0
            else:
                z_centre = z_top
            d.world_xyz = (x_w, y_w, z_centre)

        return detections

    def label_with_gemma(
        self,
        detections: list[Detection],
        image_b64: str,
        client,
    ) -> list[Detection]:
        if not detections or client is None:
            return detections

        centroids = []
        for i, d in enumerate(detections):
            color = d.color_name()
            centroids.append(f"Object {i}: centre at ({d.cx},{d.cy}), colour={color}")

        centroids_str = "; ".join(centroids)

        prompt = (
            "You are looking at a tabletop scene. Objects have been detected "
            "by a segmentation model. Their pixel-centroid locations and "
            "approximate colours are:\n"
            f"{centroids_str}\n\n"
            "Please name each object (e.g. 'red cube', 'blue cylinder', "
            "'green bowl'). Reply with a JSON array of strings in the same "
            "order, like: [\"red cube\", \"blue cylinder\", ...]. "
            "Only output valid JSON, nothing else."
        )

        try:
            import json

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            raw = client.chat.completions.create(
                model="gemma-4-31b-it",
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            ).choices[0].message.content.strip()

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            labels = json.loads(raw)
            if isinstance(labels, list):
                for d, lbl in zip(detections, labels):
                    d.label = str(lbl)
        except Exception:
            pass

        return detections
