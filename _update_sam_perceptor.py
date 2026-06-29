#!/usr/bin/env python3
"""Append SamPerceptor class to perception.py and add transformers dep to pyproject.toml."""

import re

perception_path = "/Users/jwalinshah/projects/cerebras-gemma4-hackathon/src/web/lib/perception.py"
pyproject_path = "/Users/jwalinshah/projects/cerebras-gemma4-hackathon/pyproject.toml"

# --- 1. Update perception.py ---

with open(perception_path) as f:
    perception_src = f.read()

# Verify the file ends with the perceive() function as expected
assert "def perceive(sim, obs, camera, perceptor, height=384, width=384):" in perception_src

sam_perceptor_class = '''

# ---------------------------------------------------------------------------
# SAM-based perceptor (Segment Anything Model, ViT-Tiny)
# ---------------------------------------------------------------------------

class SamPerceptor:
    """Segment tabletop objects with SAM (ViT-Tiny) from an RGB image.

    Uses huggingface transformers SamModel + SamProcessor with the tiny
    checkpoint (facebook/sam-vit-tiny). Runs on CPU by default to avoid
    GPU contention with MuJoCo.

    Resolution pipeline: 384->1024 (SAM encoder) -> masks -> nearest-neighbour->384.
    Auto-generator with 16x16 grid of prompt points.
    """

    _MODEL_CACHE: dict = {}

    def __init__(
        self,
        model_type: str = "facebook/sam-vit-tiny",
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

    def _load_model(self):
        if self.model_type in self._MODEL_CACHE:
            cached = self._MODEL_CACHE[self.model_type]
            self._model = cached["model"]
            self._processor = cached["processor"]
            self._generator = cached["generator"]
            return

        from transformers import SamModel, SamProcessor
        import torch

        processor = SamProcessor.from_pretrained(self.model_type)
        model = SamModel.from_pretrained(self.model_type).to(self.device)
        model.eval()

        from transformers.models.sam.modeling_sam import SamAutomaticMaskGenerator

        generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

        self._MODEL_CACHE[self.model_type] = {
            "model": model,
            "processor": processor,
            "generator": generator,
        }
        self._model = model
        self._processor = processor
        self._generator = generator

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        if self._generator is None:
            self._load_model()

        if rgb is None or rgb.size == 0:
            return []

        import cv2

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        masks = self._generator.generate(rgb)

        detections = []
        target_h, target_w = 384, 384
        orig_h, orig_w = rgb.shape[:2]
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        for mask_data in masks:
            mask = mask_data["segmentation"]
            area = int(mask.sum())

            scaled_area = int(area * scale_x * scale_y)
            if scaled_area < self.min_mask_region_area or scaled_area > self.max_mask_region_area:
                continue

            mask_uint8 = mask.astype(np.uint8) * 255
            mask_small = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mask_small_bool = mask_small > 0

            ys, xs = np.where(mask_small_bool)
            if len(xs) == 0:
                continue
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            bbox = _mask_bbox(mask_small_bool)
            area_px = int(mask_small_bool.sum())

            rgb_small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mean_rgb = _mask_mean_rgb(rgb_small, mask_small_bool)

            label = self._match_color(mean_rgb)
            if label is None and self.label_fallback:
                label = self.label_fallback

            detections.append(Detection(
                cx=cx, cy=cy, bbox=bbox, area_px=area_px,
                mean_rgb=mean_rgb, label=label,
                confidence=float(mask_data.get("predicted_iou", 1.0)),
                source="sam",
            ))

        return detections

    def _match_color(self, rgb):
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
            "order, like: [\\"red cube\\", \\"blue cylinder\\", ...]. "
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
                raw = raw.split("\\n", 1)[1].rsplit("```", 1)[0].strip()
            labels = json.loads(raw)
            if isinstance(labels, list):
                for d, lbl in zip(detections, labels):
                    d.label = str(lbl)
        except Exception:
            pass

        return detections
'''

# Use concatenation to ensure we stay safe with the append
perception_src += sam_perceptor_class

with open(perception_path, "w") as f:
    f.write(perception_src)

print(f"Updated {perception_path}")

# --- 2. Update pyproject.toml ---

with open(pyproject_path) as f:
    pyproject_src = f.read()

# Add transformers dependency after ultralytics or h5py line
# Find the last dependency line and insert after it
if '"transformers>=4.38.0"' not in pyproject_src:
    pyproject_src = pyproject_src.replace(
        '"h5py>=3.16.0",',
        '"h5py>=3.16.0",\n    "transformers>=4.38.0",',
    )

with open(pyproject_path, "w") as f:
    f.write(pyproject_src)

print(f"Updated {pyproject_path}")
print("Done!")
