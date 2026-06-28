"""Multi-image reasoning test for Gemma 4 31B via Cerebras.

Tests whether Gemma 4 can reason across 1, 2, and 5 images simultaneously.
Generates synthetic test images (colored shapes on backgrounds) to make
verification unambiguous.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
if not API_KEY:
    msg = "CEREBRAS_API_KEY not found in .env"
    raise RuntimeError(msg)

from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=API_KEY, base_url="https://api.cerebras.ai")
MODEL = "gemma-4-31b"

OUT_DIR = Path("examples/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_rgb_circle(
    color: str = "red",
    label: str = "RGB",
    size: tuple[int, int] = (640, 480),
    bg: str = "white",
) -> Image.Image:
    """Simple image: a colored circle on a background with a text label."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    draw.ellipse([200, 150, 440, 350], fill=color)
    draw.text((10, 10), label, fill="black")
    return img


def make_thermal_variant(
    base_label: str = "Thermal",
    size: tuple[int, int] = (640, 480),
) -> Image.Image:
    """Simulated thermal image: dark bg with a hot (yellow/white) circle."""
    img = Image.new("RGB", size, (10, 10, 40))  # dark blue-black
    draw = ImageDraw.Draw(img)
    # Hot core
    draw.ellipse([200, 150, 440, 350], fill=(255, 220, 50))
    # Heat glow
    draw.ellipse([180, 130, 460, 370], fill=(200, 100, 30))
    draw.text((10, 10), base_label, fill="white")
    return img


def make_image_variant(
    variant_id: str,
    color: str,
    label: str,
    shape_offset: tuple[int, int] = (0, 0),
    bg: str = "white",
    size: tuple[int, int] = (640, 480),
) -> Image.Image:
    """Make a distinct image variant with a colored shape + label."""
    ox, oy = shape_offset
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    # Draw a rectangle
    draw.rectangle(
        [200 + ox, 150 + oy, 440 + ox, 350 + oy], fill=color, outline="black"
    )
    # Text label in top-left
    draw.text((10, 10), label, fill="black")
    # Also put the variant ID in the bottom-right corner
    draw.text((size[0] - 100, size[1] - 30), variant_id, fill="black")
    return img


# ---------------------------------------------------------------------------
# Cerebras multi-image helper
# ---------------------------------------------------------------------------


def multimodal_chat(
    prompt: str,
    images_b64: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    stream: bool = False,
) -> dict:
    """Send one prompt with N images to Gemma 4 and return full result."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": b64}})

    messages = [{"role": "user", "content": content}]

    body = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        content_parts: list[str] = []
        final_usage: dict = {}
        final_time: dict = {}
        stream_resp = client.chat.completions.create(**body)
        for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                final_usage = chunk.usage.model_dump()
            if getattr(chunk, "time_info", None):
                final_time = chunk.time_info.model_dump()
        elapsed = time.perf_counter() - start
        return {
            "content": "".join(content_parts),
            "usage": final_usage,
            "time_info": final_time,
            "latency_s": elapsed,
        }

    resp = client.chat.completions.create(**body)
    elapsed = time.perf_counter() - start

    choice = resp.choices[0]
    usage = resp.usage.model_dump() if resp.usage else {}
    time_info = resp.time_info.model_dump() if getattr(resp, "time_info", None) else {}

    return {
        "content": choice.message.content or "",
        "model": resp.model,
        "usage": usage,
        "time_info": time_info,
        "latency_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

results: list[dict] = []


def run_test(name: str, prompt: str, images_b64: list[str], note: str = "") -> dict:
    print(f"\n{'='*72}")
    print(f"TEST: {name}")
    print(f"{'='*72}")
    print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"Number of images: {len(images_b64)}")

    result = multimodal_chat(prompt, images_b64)

    print(f"\nLatency: {result['latency_s']:.2f}s")
    usage = result.get("usage", {})
    if usage:
        print(f"Token usage: {json.dumps(usage, indent=2)}")
    time_info = result.get("time_info", {})
    if time_info:
        print(f"Time info: {json.dumps(time_info, indent=2)}")

    content = result["content"]
    print(f"\nResponse ({len(content)} chars):")
    print(content)
    print()

    entry = {
        "test_name": name,
        "prompt": prompt,
        "num_images": len(images_b64),
        "response": content,
        "latency_s": result["latency_s"],
        "usage": usage,
        "time_info": time_info,
        "note": note,
    }
    results.append(entry)
    return entry


def analyze_results() -> str:
    """Analyze whether Gemma 4 actually reasons across multiple images."""
    lines: list[str] = []
    lines.append("# Multi-Image Reasoning Test Results")
    lines.append("")
    lines.append(
        "**Date:** 2026-06-28  **Model:** gemma-4-31b (Cerebras)  "
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in results:
        lines.append(f"## {r['test_name']}")
        lines.append("")
        lines.append(f"- **Prompt:** {r['prompt']}")
        lines.append(f"- **Number of images:** {r['num_images']}")
        lines.append(f"- **Latency (wall clock):** {r['latency_s']:.2f}s")
        lines.append("")

        usage = r.get("usage", {})
        if usage:
            lines.append("### Token Usage")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(usage, indent=2))
            lines.append("```")
            lines.append("")

        time_info = r.get("time_info", {})
        if time_info:
            lines.append("### Time Info")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(time_info, indent=2))
            lines.append("```")
            lines.append("")

        lines.append("### Response")
        lines.append("")
        lines.append("```")
        lines.append(r["response"])
        lines.append("```")
        lines.append("")

        if r.get("note"):
            lines.append(f"**Note:** {r['note']}")
            lines.append("")

    # --- Overall analysis ---
    lines.append("---")
    lines.append("")
    lines.append("## Overall Analysis: Does Gemma 4 Reason Across Multiple Images?")
    lines.append("")

    # Analyze each test for multi-image reasoning
    for r in results:
        resp_lower = r["response"].lower()
        name = r["test_name"]
        ni = r["num_images"]

        if ni == 1:
            lines.append(f"### {name} (1 image)")
            lines.append("- Baseline single-image test. Model should describe the image correctly.")
            if "circle" in resp_lower or "red" in resp_lower or "thermal" in resp_lower:
                lines.append("- **PASS:** Model correctly identifies the single image content.")
            else:
                lines.append("- **CHECK:** Response seems generic or unrelated to the image.")
            lines.append("")

        elif ni == 2:
            lines.append(f"### {name} (2 images)")
            lines.append("- Test of two-image comparison. Model should reference both images explicitly.")
            ref_count = sum(1 for kw in ["image 1", "image 2", "first image", "second image", "rgb", "thermal", "left", "right"] if kw in resp_lower)
            if ref_count >= 2:
                lines.append(f"- **PARTIAL PASS:** Model references specific images ({ref_count} reference keywords found).")
            else:
                lines.append(f"- **WEAK:** Only {ref_count} reference keywords found -- may be looking at only one image.")
            if "thermal" in resp_lower and ("hot" in resp_lower or "heat" in resp_lower or "warm" in resp_lower or "temperature" in resp_lower):
                lines.append("- **PASS:** Model appears to distinguish thermal content from RGB.")
            else:
                lines.append("- **CHECK:** Model may not be distinguishing the thermal content.")
            lines.append("")

        elif ni >= 5:
            lines.append(f"### {name} ({ni} images)")
            if "identical" in resp_lower or "duplicate" in resp_lower or "same" in resp_lower:
                lines.append("- **RELEVANT:** Model discusses image relationships (identical/duplicate/same).")
            else:
                lines.append("- **CHECK:** Model does not explicitly address image relationships.")
            ref_count = sum(1 for kw in ["image 1", "image 2", "image 3", "image 4", "image 5", "first", "second", "third", "fourth", "fifth", "variant", "version"] if kw in resp_lower)
            lines.append(f"- Image reference keywords found: {ref_count}")
            if ref_count >= 3:
                lines.append("- **STRONG:** Model references multiple distinct images in its response.")
            elif ref_count >= 1:
                lines.append("- **PARTIAL:** Some image references, but may not be reasoning across all.")
            else:
                lines.append("- **WEAK:** No clear references to multiple distinct images.")
            lines.append("")

    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    lines.append("1. **Token counts** -- Look at `image_tokens` in the usage to see how many tokens each image consumed.")
    lines.append("2. **Response specificity** -- Does the model mention details unique to each image, or does it give a generic description?")
    lines.append("3. **Comparison ability** -- For the side-by-side test, does the model correctly identify differences?")
    lines.append("4. **Duplicate detection** -- For the 5-identical-images test, does the model realize they're all the same?")

    return "\n".join(lines)


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("Generating test images...")

    # --- Generate all test images ---
    img_rgb = make_rgb_circle(color="red", label="RGB Camera View")
    img_rgb.save(str(OUT_DIR / "test_rgb.jpg"), quality=92)

    img_thermal = make_thermal_variant()
    img_thermal.save(str(OUT_DIR / "test_thermal.jpg"), quality=92)

    # 5 variants for Test C
    variants: list[Image.Image] = [
        make_image_variant("V1", "red", "Variant 1 - Red Square", shape_offset=(0, 0), bg="white"),
        make_image_variant("V2", "blue", "Variant 2 - Blue Square", shape_offset=(30, 20), bg="lightgray"),
        make_image_variant("V3", "green", "Variant 3 - Green Square", shape_offset=(-20, 40), bg="ivory"),
        make_image_variant("V4", "orange", "Variant 4 - Orange Square", shape_offset=(10, -30), bg="lavender"),
        make_image_variant("V5", "purple", "Variant 5 - Purple Square", shape_offset=(-10, 10), bg="lightyellow"),
    ]
    for i, v in enumerate(variants, 1):
        v.save(str(OUT_DIR / f"test_variant_{i}.jpg"), quality=92)

    # Encode all to base64
    b64_rgb = _pil_to_b64(img_rgb)
    b64_thermal = _pil_to_b64(img_thermal)
    b64_variants = [_pil_to_b64(v) for v in variants]
    b64_all_5 = [b64_variants[0]] * 5  # same image 5x for Test D

    print(f"Images saved to {OUT_DIR.resolve()}")
    print(f"RGB: {len(b64_rgb)} chars b64")
    print(f"Thermal: {len(b64_thermal)} chars b64")
    for i in range(5):
        print(f"Variant {i+1}: {len(b64_variants[i])} chars b64")

    # ==================================================================
    # Test A: Single image baseline
    # ==================================================================
    run_test(
        "Test A: Single Image",
        "What do you see in this image? Describe it in detail.",
        [b64_rgb],
        note="Baseline test -- single image of a red circle on white background labeled 'RGB Camera View'.",
    )

    # ==================================================================
    # Test B: Two images side-by-side (RGB + Thermal)
    # ==================================================================
    run_test(
        "Test B: Two Images (RGB + Thermal)",
        "I am showing you two images of the same scene. The first is an RGB camera view and the second is a thermal camera view. "
        "Compare these two images carefully. What can you see in the thermal image that you cannot see in the RGB image? "
        "Be specific about both images.",
        [b64_rgb, b64_thermal],
        note="Critical test: does the model actually compare two images, or just describe one?",
    )

    # ==================================================================
    # Test C: Five different image variants
    # ==================================================================
    run_test(
        "Test C: Five Different Images",
        "I am showing you 5 different images. Each has a colored square on a different background. "
        "Analyze ALL 5 images together. Describe what each image shows, noting the color, background, and position of each square. "
        "What is the complete picture across all 5 images?",
        b64_variants,
        note="Stress test: 5 distinct images. Does the model reference each one or just summarize generically?",
    )

    # ==================================================================
    # Test D: Same image 5 times
    # ==================================================================
    run_test(
        "Test D: Same Image 5 Times",
        "I am showing you 5 images. Are all these images identical to each other, or are there differences? "
        "If they are identical, say so explicitly. If there are differences, describe exactly what differs between them. "
        "Examine each image carefully before answering.",
        b64_all_5,
        note="Duplicate detection test: all 5 images are the same variant. Can the model detect they are identical?",
    )

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    report = analyze_results()
    doc_path = Path("docs/research/14-multi-image-test.md")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    print(f"\nReport written to {doc_path.resolve()}")

    # Also print summary
    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"\n{r['test_name']}:")
        print(f"  Latency: {r['latency_s']:.2f}s")
        print(f"  Response length: {len(r['response'])} chars")
        usage = r.get("usage", {})
        if usage:
            img_tok = usage.get("image_tokens", usage.get("images", "N/A"))
            print(f"  Image tokens: {img_tok}")
        print(f"  Response preview: {r['response'][:100]}...")


if __name__ == "__main__":
    main()
