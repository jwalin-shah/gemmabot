"""Image encoding helpers for Gemma 4 multimodal input."""

from __future__ import annotations

import base64
from pathlib import Path


def encode_image(path: str | Path) -> str:
    """Read an image file and return a base64 data URI.

    Supports PNG and JPEG. Returns ``data:image/<ext>;base64,...``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        msg = f"Unsupported image format: {suffix} (only .png, .jpg, .jpeg)"
        raise ValueError(msg)

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


IMAGE_TOKEN_ESTIMATE_CUTOFF = 645120  # 1280 * 504


def estimate_image_tokens(width: int, height: int) -> int:
    """Approximate image token cost using Gemma 4 31Bs scaling."""
    scale = (IMAGE_TOKEN_ESTIMATE_CUTOFF / (width * height)) ** 0.5
    pw = (int(width * scale) // 48) * 48
    ph = (int(height * scale) // 48) * 48
    return min(pw // 48 * (ph // 48), 280)