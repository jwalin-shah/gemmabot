"""Visualization overlays for robot perception debugging.

Provides mask overlays, bounding box rendering, distance vector visualization,
and composite debug views for the robot vision pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISTINCT_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255),
    (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (200, 100, 0), (100, 0, 200), (0, 200, 100),
    (200, 0, 100), (100, 200, 0), (0, 100, 200),
    (255, 160, 160), (160, 255, 160), (160, 160, 255),
]

_FONT = ImageFont.load_default()


# ---------------------------------------------------------------------------
# Mask overlay
# ---------------------------------------------------------------------------

def draw_mask_overlay(
    rgb: np.ndarray,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int]] | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay each mask as a semi-transparent colored region on the RGB image.

    Args:
        rgb: RGB image array (H, W, 3), uint8.
        masks: List of boolean mask arrays, each (H, W).
        colors: Optional list of (R, G, B) tuples per mask.
            If None, distinct colors are used.
        alpha: Transparency factor (0 = fully transparent, 1 = opaque).

    Returns:
        RGB uint8 array (H, W, 3) with mask overlay composited.
    """
    if not masks:
        return rgb.copy()

    h, w = rgb.shape[:2]
    if colors is None:
        colors = [_DISTINCT_COLORS[i % len(_DISTINCT_COLORS)] for i in range(len(masks))]

    # Build composite overlay using PIL alpha compositing
    pil_base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for mask, color in zip(masks, colors):
        # Create mask image
        mask_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mask_data = np.zeros((h, w, 4), dtype=np.uint8)
        if mask.shape[:2] == (h, w):
            mask_data[mask, 0] = color[0]
            mask_data[mask, 1] = color[1]
            mask_data[mask, 2] = color[2]
            mask_data[mask, 3] = int(np.clip(alpha * 255, 0, 255))
        mask_img = Image.fromarray(mask_data, "RGBA")
        overlay = Image.alpha_composite(overlay, mask_img)

    result = Image.alpha_composite(pil_base, overlay)
    return np.array(result.convert("RGB"))


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

def draw_bboxes(
    img: np.ndarray,
    detections: list,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding box rectangles from Detection.bbox.

    Each detection is expected to have ``bbox`` (x0, y0, x1, y1) and
    ``label`` (optional str) attributes.

    Args:
        img: RGB image array (H, W, 3), uint8.
        detections: List of Detection-like objects with ``bbox`` and ``label``.
        color: RGB tuple for bbox outline and label pill fill.

    Returns:
        RGB uint8 array with bboxes and label pills drawn.
    """
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    for det in detections:
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox

        # Draw rectangle outline
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        # Draw label pill above the bbox
        label = (det.label or "?") if hasattr(det, "label") else "?"
        bbox_text = draw.textbbox((0, 0), label, font=_FONT)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        pill_x0 = x0
        pill_y0 = max(0, y0 - th - 6)
        pill_x1 = x0 + tw + 8
        pill_y1 = y0
        draw.rectangle([pill_x0, pill_y0, pill_x1, pill_y1], fill=color)
        draw.text((x0 + 4, pill_y0 + 2), label, fill=(0, 0, 0), font=_FONT)

    return np.array(pil)


# ---------------------------------------------------------------------------
# Distance vectors
# ---------------------------------------------------------------------------

def draw_distance_vectors(
    img: np.ndarray,
    ee_xy: tuple[float, float],
    detections: list,
    ee_pixel: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw arrow from EE position to each detection centroid.

    For each detection a line is drawn from the detection centroid (cx, cy)
    to the EE pixel position.  A distance label in cm is placed at the
    midpoint of the line.  The EE is marked with a cyan circle.

    Color coding:
        - red   if detection's world_xyz z > 0.9
        - blue  otherwise

    Args:
        img: RGB image array (H, W, 3), uint8.
        ee_xy: End-effector (x, y) position in world frame (used for distance
            computation).  Can be a 2-tuple or 3-tuple (z ignored if present).
        detections: List of Detection-like objects with cx, cy, world_xyz.
        ee_pixel: (col, row) pixel position of the EE in this view.
            If None, the arrow and EE marker are skipped.

    Returns:
        RGB uint8 array with vectors and labels drawn.
    """
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    # Draw EE marker if pixel position is known
    if ee_pixel is not None:
        ex, ey = int(round(ee_pixel[0])), int(round(ee_pixel[1]))
        r = 6
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(0, 255, 255))
        draw.ellipse([ex - r - 2, ey - r - 2, ex + r + 2, ey + r + 2], outline=(0, 200, 200), width=2)

    for det in detections:
        cx, cy = getattr(det, "cx", None), getattr(det, "cy", None)
        if cx is None or cy is None:
            continue

        # Determine line color
        wxyz = getattr(det, "world_xyz", None)
        if wxyz is not None and len(wxyz) > 2 and wxyz[2] > 0.9:
            line_color = (255, 0, 0)   # red (high z)
        else:
            line_color = (0, 100, 255)  # blue

        # Draw centroid marker for this detection
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=line_color)

        if ee_pixel is not None:
            ex, ey = int(round(ee_pixel[0])), int(round(ee_pixel[1]))

            # Draw line from detection centroid to EE
            draw.line([(cx, cy), (ex, ey)], fill=line_color, width=2)

            # Arrowhead at EE end
            dx = ex - cx
            dy = ey - cy
            angle = math.atan2(dy, dx)
            arrow_len = 10
            half_angle = math.pi / 7
            ax1 = ex - arrow_len * math.cos(angle - half_angle)
            ay1 = ey - arrow_len * math.sin(angle - half_angle)
            ax2 = ex - arrow_len * math.cos(angle + half_angle)
            ay2 = ey - arrow_len * math.sin(angle + half_angle)
            draw.polygon([(ex, ey), (ax1, ay1), (ax2, ay2)], fill=line_color)

            # Distance label (3D distance in cm)
            dist_cm: float = 0.0
            if wxyz is not None:
                ex_w = ee_xy[0]
                ey_w = ee_xy[1] if len(ee_xy) > 1 else 0.0
                ez_w = ee_xy[2] if len(ee_xy) > 2 else 0.0
                dx3 = wxyz[0] - ex_w
                dy3 = wxyz[1] - ey_w
                dz3 = (wxyz[2] if len(wxyz) > 2 else 0.0) - ez_w
                dist_cm = math.sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3) * 100.0
            else:
                # Fallback: pixel distance
                dist_cm = math.sqrt(dx * dx + dy * dy) * 0.2

            label = f"{dist_cm:.0f}cm"

            # Draw label at midpoint with background pill
            mx = (cx + ex) // 2
            my = (cy + ey) // 2
            bbox_text = draw.textbbox((0, 0), label, font=_FONT)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            pad = 3
            draw.rectangle(
                [mx - tw // 2 - pad, my - th // 2 - pad,
                 mx + tw // 2 + pad, my + th // 2 + pad],
                fill=(0, 0, 0),
            )
            draw.text(
                (mx - tw // 2, my - th // 2),
                label, fill=(255, 255, 255), font=_FONT,
            )

    return np.array(pil)


# ---------------------------------------------------------------------------
# Debug composite view  (single camera)
# ---------------------------------------------------------------------------

def make_debug_view(
    rgb: np.ndarray,
    masks: list[np.ndarray] | None = None,
    mask_colors: list[tuple] | None = None,
    detections: list | None = None,
    ee_xy: tuple[float, float] | None = None,
    ee_pixel: tuple[float, float] | None = None,
    reasoning: str | None = None,
) -> np.ndarray:
    """Compose a debug overlay for one camera view.

    Layers are applied in order:
        1. Mask overlay (semi-transparent)
        2. Bounding boxes with labels
        3. Distance vectors (arrows + EE marker)
        4. Reasoning text bar at bottom

    The final image is always 384x384 RGB uint8.

    Args:
        rgb: Source RGB image (H, W, 3), uint8.
        masks: Optional list of boolean masks for overlay.
        mask_colors: Optional (R, G, B) per mask.
        detections: Optional list of Detection-like objects.
        ee_xy: EE world (x, y) for distance computation.
        ee_pixel: EE pixel (col, row) in this camera view.
        reasoning: Optional text to render in bottom bar.

    Returns:
        RGB uint8 array with shape (384, 384, 3).
    """
    img = rgb.copy()

    # 1. Mask overlay
    if masks:
        img = draw_mask_overlay(img, masks, mask_colors)

    # 2. Bounding boxes
    if detections:
        img = draw_bboxes(img, detections)

    # 3. Distance vectors
    if detections and ee_pixel is not None:
        img = draw_distance_vectors(
            img,
            ee_xy or (0.0, 0.0),
            detections,
            ee_pixel=ee_pixel,
        )

    # 4. Reasoning text bar at bottom
    if reasoning:
        pil = Image.fromarray(img).convert("RGBA")
        w, h = pil.size
        bar_h = 30

        # Semi-transparent bar at bottom
        bar_arr = np.zeros((bar_h, w, 4), dtype=np.uint8)
        bar_arr[:, :, 3] = 200  # alpha
        bar_img = Image.fromarray(bar_arr, "RGBA")
        pil = Image.alpha_composite(pil, bar_img)
        pil_rgb = pil.convert("RGB")

        # Draw text
        draw = ImageDraw.Draw(pil_rgb)
        text = reasoning.strip().replace("\n", " ")[:85]
        if len(text) > 82:
            text = text[:79] + "..."
        draw.text((8, h - bar_h + 6), text, fill=(255, 255, 200), font=_FONT)

        img = np.array(pil_rgb)

    # Ensure 384x384
    if img.shape[:2] != (384, 384):
        pil = Image.fromarray(img).resize((384, 384), Image.LANCZOS)
        img = np.array(pil)

    return img


# ---------------------------------------------------------------------------
# Debug composite  (birdview + frontview)
# ---------------------------------------------------------------------------

def generate_debug_composite(
    birdview: np.ndarray,
    frontview: np.ndarray,
    masks_bird: list[np.ndarray] | None = None,
    masks_front: list[np.ndarray] | None = None,
    detections_bird: list | None = None,
    detections_front: list | None = None,
    ee_xy: tuple[float, float] | None = None,
    reasoning: str | None = None,
) -> np.ndarray:
    """Generate a debug composite from birdview and frontview camera feeds.

    Each view is processed independently through :func:`make_debug_view`,
    then horizontally stacked and resized to 384x384.

    Args:
        birdview: Birdview RGB frame (H, W, 3), uint8.
        frontview: Front-view RGB frame (H, W, 3), uint8.
        masks_bird: Boolean masks for the birdview.
        masks_front: Boolean masks for the frontview.
        detections_bird: Detections from the birdview.
        detections_front: Detections from the frontview.
        ee_xy: EE world (x, y) position.
        reasoning: Optional reasoning text (drawn only on birdview).

    Returns:
        RGB uint8 array with shape (384, 384, 3).
    """
    db = make_debug_view(
        birdview,
        masks=masks_bird,
        detections=detections_bird if detections_bird else None,
        ee_xy=ee_xy,
        ee_pixel=None,
        reasoning=reasoning,
    )

    df = make_debug_view(
        frontview,
        masks=masks_front,
        detections=detections_front if detections_front else None,
        ee_xy=ee_xy,
        ee_pixel=None,
        reasoning=None,
    )

    # Horizontally stack
    combined = np.hstack([db, df])

    # Resize to 384x384
    pil = Image.fromarray(combined).resize((384, 384), Image.LANCZOS)
    return np.array(pil)
