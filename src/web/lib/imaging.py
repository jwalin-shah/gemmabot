"""Image processing utilities for the robot vision pipeline.

Provides blank-frame constants, encoding (base64 data URIs), and composite
image construction, all optimised for the Gemma 4 vision encoder resolution.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Blank frame  (384 x 384 — matches Gemma 4's internal vision encoder)
# ---------------------------------------------------------------------------
_BLANK = np.zeros((384, 384, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def img_to_b64(img: np.ndarray, fmt: str = "JPEG", quality: int = 50) -> str:
    """Encode a numpy RGB image array to a base64 data-URI string.

    Args:
        img: RGB image array with shape (H, W, 3).
        fmt: Image format passed to PIL ``save()`` (default ``"JPEG"``).
        quality: Encoding quality 1-100 (default 50).

    Returns:
        Data-URI string, e.g. ``data:image/webp;base64,...``.
    """
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


# ---------------------------------------------------------------------------
# Composite images
# ---------------------------------------------------------------------------

# Default canvas if a camera is missing.
_BLANK = np.zeros((384, 384, 3), dtype=np.uint8)


def fix_img(img: np.ndarray | None) -> np.ndarray:
    """Flip robosuite camera output to the right orientation."""
    if img is None:
        return _BLANK
    return np.ascontiguousarray(np.flipud(img))


def overlay_grid(img: np.ndarray) -> np.ndarray:
    """Stamp a 3x2 Zone A-F grid onto the birdview."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    w, h = pil.size
    d = ImageDraw.Draw(pil)
    for c in range(1, 3):
        d.line([(c * w // 3, 0), (c * w // 3, h)], fill=(255, 80, 80), width=2)
    d.line([(0, h // 2), (w, h // 2)], fill=(255, 80, 80), width=2)
    for i, lab in enumerate(["A", "B", "C", "D", "E", "F"]):
        r, c_idx = divmod(i, 3)
        d.text((c_idx * w // 3 + 18, r * h // 2 + 12), f"Zone {lab}", fill=(255, 255, 100))
    return np.array(pil)


def make_composite(*images: np.ndarray, layout: str = "horizontal") -> np.ndarray:
    """Combine multiple RGB frames into a single composite image.

    The final composite is always resized to 384x384 to match the
    resolution expected by Gemma 4's vision encoder, which reduces
    latency by avoiding arbitrary-size encoding on the API side.

    Args:
        *images: One or more RGB image arrays (H, W, 3).
        layout: ``"horizontal"`` (default), ``"vertical"``, or ``"grid"``
            (2-column grid).

    Returns:
        RGB image array with shape (384, 384, 3).
    """
    if not images:
        return _BLANK.copy()

    valid = [img for img in images if img is not None and img.size > 0]
    if not valid:
        return _BLANK.copy()

    if layout == "horizontal":
        h = max(img.shape[0] for img in valid)
        padded = []
        for img in valid:
            ih, iw = img.shape[:2]
            if ih < h:
                pad = np.zeros((h - ih, iw, 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        output = np.hstack(padded)
    elif layout == "vertical":
        w = max(img.shape[1] for img in valid)
        padded = []
        for img in valid:
            ih, iw = img.shape[:2]
            if iw < w:
                pad = np.zeros((ih, w - iw, 3), dtype=np.uint8)
                img = np.hstack([img, pad])
            padded.append(img)
        output = np.vstack(padded)
    elif layout == "grid":
        n = len(valid)
        cols = 2
        rows = (n + cols - 1) // cols
        cell_h = max(img.shape[0] for img in valid)
        cell_w = max(img.shape[1] for img in valid)
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for i, img in enumerate(valid):
            r, c = divmod(i, cols)
            ih, iw = img.shape[:2]
            grid[r * cell_h : r * cell_h + ih, c * cell_w : c * cell_w + iw] = img
        output = grid
    else:
        output = valid[0]

    # Resize to match Gemma 4's internal vision encoder resolution
    pil = Image.fromarray(output)
    pil = pil.resize((384, 384), Image.LANCZOS)
    return np.array(pil)
