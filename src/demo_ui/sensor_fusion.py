#!/usr/bin/env python3
"""
sensor_fusion.py - Multi-sensor image generator for search & rescue drone demo.

Loads a base scene image (workspace.jpg) and generates 4 synthetic sensor views:
  - RGB (original + annotated "survivor" markers)
  - Thermal/IR (colormap overlay with hot spots)
  - Depth map (grayscale gradient - closer = brighter)
  - Motion mask (white blobs on black = changed pixels)

Saves each as JPEG and builds a Gemma 4-compatible payload (5 images max, <4 MB total).
"""

import base64
import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "examples", "images")
SOURCE_IMAGE = os.path.join(IMAGES_DIR, "workspace.jpg")

SENSOR_RGB = os.path.join(IMAGES_DIR, "sensor_rgb.jpg")
SENSOR_THERMAL = os.path.join(IMAGES_DIR, "sensor_thermal.jpg")
SENSOR_DEPTH = os.path.join(IMAGES_DIR, "sensor_depth.jpg")
SENSOR_MOTION = os.path.join(IMAGES_DIR, "sensor_motion.jpg")
SENSOR_LABELED = os.path.join(IMAGES_DIR, "sensor_labeled.jpg")

OUTPUT_IMAGES = [SENSOR_RGB, SENSOR_THERMAL, SENSOR_DEPTH, SENSOR_MOTION, SENSOR_LABELED]

random.seed(42)
np.random.seed(42)


# ===================================================================
# 1. RGB + Survivor Dots
# ===================================================================
def generate_rgb(base: Image.Image) -> Image.Image:
    """Draw colored survivor markers on the source image."""
    img = base.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    survivors = [
        (int(w * 0.25), int(h * 0.35)),
        (int(w * 0.65), int(h * 0.50)),
        (int(w * 0.45), int(h * 0.72)),
        (int(w * 0.80), int(h * 0.28)),
    ]

    for sx, sy in survivors:
        # Glowing outer ring
        for r in range(14, 6, -2):
            alpha = max(0, 255 - int(200 * (r / 14)))
            draw.ellipse(
                [sx - r, sy - r, sx + r, sy + r],
                outline=(255, 50, 50, alpha) if r > 10 else (255, 255, 50, alpha),
                width=2,
            )
        # Center dot
        draw.ellipse(
            [sx - 5, sy - 5, sx + 5, sy + 5], fill=(255, 0, 0), outline=(255, 255, 255)
        )
        # Label
        draw.text((sx + 10, sy - 8), "SURVIVOR", fill=(255, 255, 255))

    return img


# ===================================================================
# 2. Thermal / IR
# ===================================================================
def generate_thermal(base: Image.Image) -> Image.Image:
    """
    Generate a thermal/IR view:
      - Grayscale base with random hot spots (simulating survivors / heat sources)
      - Apply an 'inferno'-like colormap so hot = white/yellow, cold = dark
    """
    gray = base.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Add a few hot spots (gaussian blobs)
    w, h = base.size
    hot_spots = [
        (int(w * 0.25), int(h * 0.35), 60),
        (int(w * 0.65), int(h * 0.50), 50),
        (int(w * 0.45), int(h * 0.72), 55),
        (int(w * 0.80), int(h * 0.28), 45),
        (int(w * 0.50), int(h * 0.10), 35),
        (int(w * 0.15), int(h * 0.80), 40),
    ]

    # Add a subtle heat gradient (warmer at ground level)
    for y in range(h):
        arr[:, y] += 15 * (y / h)  # ground is warmer

    # Add hot spots
    mask = np.zeros((h, w), dtype=np.float32)
    for cx, cy, radius in hot_spots:
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        blob = np.exp(-(dist**2) / (2 * (radius / 2) ** 2))
        mask += blob * 80

    arr = np.clip(arr + mask, 0, 255).astype(np.uint8)

    # Apply inferno-like colormap
    # Create a simple lookup table mapping 0->dark purple/black, 128->red/orange, 255->white/yellow
    colors = []
    for i in range(256):
        t = i / 255.0  # Normalize
        if t < 0.25:
            r, g, b = 0, 0, int(t * 4 * 100)
        elif t < 0.5:
            r = int((t - 0.25) * 4 * 200)
            g = 0
            b = 100 - int((t - 0.25) * 4 * 50)
        elif t < 0.75:
            r = 200 + int((t - 0.5) * 4 * 55)
            g = int((t - 0.5) * 4 * 150)
            b = 50 - int((t - 0.5) * 4 * 50)
        else:
            r = 255
            g = 150 + int((t - 0.75) * 4 * 105)
            b = int((t - 0.75) * 4 * 200)
        colors.append((min(255, r), min(255, g), min(255, b)))

    lut = np.array(colors, dtype=np.uint8)
    colored = lut[arr]

    # Smooth it
    result = Image.fromarray(colored, "RGB")
    result = result.filter(ImageFilter.GaussianBlur(radius=1))

    # Draw survivor markers as bright white/yellow circles on thermal too
    draw = ImageDraw.Draw(result)
    survivors = [
        (int(w * 0.25), int(h * 0.35)),
        (int(w * 0.65), int(h * 0.50)),
        (int(w * 0.45), int(h * 0.72)),
        (int(w * 0.80), int(h * 0.28)),
    ]
    for sx, sy in survivors:
        draw.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], outline=(255, 255, 200), width=2)
        draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(255, 50, 50))

    return result


# ===================================================================
# 3. Depth Map
# ===================================================================
def generate_depth(base: Image.Image) -> Image.Image:
    """
    Generate a depth map: closer = brighter (white), farther = darker (black).

    Strategy:
      - Assume the bottom of the image is closer (drone looking down/forward)
      - Use image gradients + random object-like blobs for structure
    """
    gray = base.convert("L")
    arr = np.array(gray, dtype=np.float32)
    w, h = base.size

    # Base depth: closer at bottom, farther at top
    y_grad = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
    depth = y_grad * 200  # 0-200 range

    # Add edge-based depth cues: edges (high contrast) = objects = closer
    from PIL import ImageFilter as IF

    edges = gray.filter(IF.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32)
    edge_arr = np.clip(edge_arr, 0, 50)

    depth = np.clip(depth + edge_arr, 0, 255)

    # Add some random blob "objects" that are closer
    num_objects = 12
    for _ in range(num_objects):
        ox = random.randint(0, w - 1)
        oy = random.randint(0, h - 1)
        radius = random.randint(20, 60)
        brightness = random.randint(150, 230)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - ox) ** 2 + (y_grid - oy) ** 2)
        blob = np.exp(-(dist**2) / (2 * (radius / 2) ** 2))
        depth += blob * brightness * 0.3

    depth = np.clip(depth, 0, 255).astype(np.uint8)
    result = Image.fromarray(depth, "L").convert("RGB")
    result = result.filter(ImageFilter.GaussianBlur(radius=2))

    return result


# ===================================================================
# 4. Motion Mask
# ===================================================================
def generate_motion(base: Image.Image) -> Image.Image:
    """
    Generate a motion mask: white blobs on black background.
    Simulates "changed" pixels between frames (moving survivors, drone motion).
    """
    w, h = base.size
    arr = np.zeros((h, w), dtype=np.uint8)

    # Random motion blobs (simulating movement)
    num_blobs = random.randint(8, 15)
    for _ in range(num_blobs):
        bx = random.randint(0, w - 1)
        by = random.randint(0, h - 1)
        radius = random.randint(8, 35)
        intensity = random.randint(120, 255)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - bx) ** 2 + (y_grid - by) ** 2)
        blob = np.exp(-(dist**2) / (2 * (radius / 2.5) ** 2))
        arr = np.maximum(arr, (blob * intensity).astype(np.uint8))

    # Add survivor-specific motion (larger, more coherent)
    survivors = [
        (int(w * 0.25), int(h * 0.35)),
        (int(w * 0.65), int(h * 0.50)),
        (int(w * 0.45), int(h * 0.72)),
        (int(w * 0.80), int(h * 0.28)),
    ]
    for sx, sy in survivors:
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - sx) ** 2 + (y_grid - sy) ** 2)
        blob = np.exp(-(dist**2) / (2 * 20**2))
        arr = np.maximum(arr, (blob * 255).astype(np.uint8))

    # Add some "noise" motion (drone vibration)
    noise = np.random.randint(0, 30, (h, w), dtype=np.uint8)
    arr = np.clip(arr.astype(np.uint16) + noise.astype(np.uint16), 0, 255).astype(np.uint8)

    # Blur for soft motion edges
    result = Image.fromarray(arr, mode="L").convert("RGB")
    result = result.filter(ImageFilter.GaussianBlur(radius=3))

    # Threshold to make it binary-looking but with soft edges
    gray_arr = np.array(result.convert("L"))
    gray_arr = np.where(gray_arr > 40, 255, 0).astype(np.uint8)
    result = Image.fromarray(gray_arr, mode="L").convert("RGB")
    result = result.filter(ImageFilter.GaussianBlur(radius=1))

    return result


# ===================================================================
# 5. Labeled Overlay (all sensors combined with labels)
# ===================================================================
def generate_labeled_overlay(
    rgb: Image.Image,
    thermal: Image.Image,
    depth: Image.Image,
    motion: Image.Image,
) -> Image.Image:
    """
    Create a 2x2 grid showing all four sensor views with text labels.
    """
    # Resize all to same dimensions
    tw, th = rgb.size
    thumb_w, thumb_h = tw // 2, th // 2

    images = {
        "RGB": rgb.resize((thumb_w, thumb_h)),
        "THERMAL": thermal.resize((thumb_w, thumb_h)),
        "DEPTH": depth.resize((thumb_w, thumb_h)),
        "MOTION": motion.resize((thumb_w, thumb_h)),
    }

    canvas = Image.new("RGB", (tw, th), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    positions = {
        "RGB": (0, 0),
        "THERMAL": (thumb_w, 0),
        "DEPTH": (0, thumb_h),
        "MOTION": (thumb_w, thumb_h),
    }

    # Try to use a nice font, fall back to default
    font = None
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            from PIL import ImageFont

            try:
                font = ImageFont.truetype(fp, 24)
            except Exception:
                pass
            break

    for label, (px, py) in positions.items():
        img = images[label]
        canvas.paste(img, (px, py))
        # Add border
        draw.rectangle([px, py, px + thumb_w, py + thumb_h], outline=(100, 100, 100), width=2)
        # Add label
        label_bg = (0, 0, 0, 160)
        draw.rectangle(
            [px + 4, py + 4, px + 8 + len(label) * 12, py + 30], fill=(0, 0, 0)
        )
        draw.text((px + 8, py + 6), label, fill=(255, 255, 255), font=font)

    return canvas


# ===================================================================
# Save with size management
# ===================================================================
def save_jpeg(img: Image.Image, path: str, max_size_kb: int = 900) -> None:
    """
    Save as JPEG, reducing quality until under max_size_kb.
    """
    quality = 92
    while quality >= 15:
        img.save(path, "JPEG", quality=quality, optimize=True)
        size_kb = os.path.getsize(path) / 1024
        if size_kb <= max_size_kb:
            break
        quality -= 5

    final_kb = os.path.getsize(path) / 1024
    print(f"  Saved {os.path.basename(path)}: {final_kb:.1f} KB (q={quality})")


# ===================================================================
# Payload builder (Gemma 4 format)
# ===================================================================
def build_payload(image_paths: list[str]) -> dict:
    """
    Build a Gemma 4 chat completions payload with up to 5 images.
    """
    content = [
        {
            "type": "text",
            "text": (
                "You are a search and rescue drone AI. Analyze these 5 sensor views "
                "of the same scene: RGB, Thermal/IR, Depth, Motion, and a Labeled overlay. "
                "Identify any survivors, hazards, and suggest a search path."
            ),
        }
    ]

    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            }
        )

    payload = {
        "model": "gemma-4-31b",
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_completion_tokens": 500,
    }

    return payload


def payload_size_kb(payload: dict) -> float:
    """Calculate the total base64 image payload size in KB."""
    total_b64_len = 0
    for msg in payload["messages"]:
        for item in msg["content"]:
            if item.get("type") == "image_url":
                url = item["image_url"]["url"]
                # Extract base64 data after comma
                b64_data = url.split(",", 1)[1]
                total_b64_len += len(b64_data)
    return total_b64_len / 1024


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("Multi-Sensor Image Generator for Search & Rescue Demo")
    print("=" * 60)

    # Load source
    if not os.path.exists(SOURCE_IMAGE):
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE}")

    base = Image.open(SOURCE_IMAGE).convert("RGB")
    print(f"\nLoaded source: {os.path.basename(SOURCE_IMAGE)} ({base.size[0]}x{base.size[1]})")

    # Generate sensor images
    print("\n--- Generating sensor images ---")

    print("  [1/4] RGB + survivor markers...")
    rgb = generate_rgb(base)

    print("  [2/4] Thermal / IR...")
    thermal = generate_thermal(base)

    print("  [3/4] Depth map...")
    depth = generate_depth(base)

    print("  [4/4] Motion mask...")
    motion = generate_motion(base)

    # Labeled overlay (5th image)
    print("  [5/5] Labeled overlay (2x2 grid)...")
    labeled = generate_labeled_overlay(rgb, thermal, depth, motion)

    # Save all
    print("\n--- Saving images ---")
    save_jpeg(rgb, SENSOR_RGB)
    save_jpeg(thermal, SENSOR_THERMAL)
    save_jpeg(depth, SENSOR_DEPTH)
    save_jpeg(motion, SENSOR_MOTION)
    save_jpeg(labeled, SENSOR_LABELED)

    # Build payload
    print("\n--- Building Gemma 4 payload ---")
    payload = build_payload(OUTPUT_IMAGES)
    total_img_kb = payload_size_kb(payload)
    total_json_kb = len(json.dumps(payload)) / 1024

    print(f"  Total base64 image payload: {total_img_kb:.1f} KB")
    print(f"  Total JSON payload: {total_json_kb:.1f} KB")

    # Check constraints
    print("\n--- Constraint checks ---")
    for path in OUTPUT_IMAGES:
        kb = os.path.getsize(path) / 1024
        status = "OK" if kb < 1024 else "OVER LIMIT"
        print(f"  {os.path.basename(path)}: {kb:.1f} KB [{status}]")

    print(f"\n  Combined image payload: {total_img_kb:.1f} KB")
    if total_img_kb < 4 * 1024:
        print("  Free tier limit (4 MB): PASS")
    else:
        print("  Free tier limit (4 MB): FAIL - above limit!")

    print(f"  Max images per request (5): {'PASS' if len(OUTPUT_IMAGES) <= 5 else 'FAIL'}")

    # Print payload example
    print("\n--- Payload snippet (first 500 chars) ---")
    payload_str = json.dumps(payload, indent=2)
    # Truncate base64 data for display
    import re

    display_payload = re.sub(
        r'("data:image/jpeg;base64,)[^"]+',
        lambda m: m.group(1) + "...TRUNCATED...",
        payload_str,
    )
    print(display_payload[:500] + "...")

    print("\nDone! All sensor images saved to:", IMAGES_DIR)


if __name__ == "__main__":
    main()
