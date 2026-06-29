"""Run Gemma 4 on real photo-quality images and show what it identifies."""

from __future__ import annotations

import base64, json, sys, time, io
from pathlib import Path
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.client import CerebrasClient

IMAGE_DIR = Path("/Users/jwalinshah/projects/cerebras-gemma4-hackathon/examples/images/real")
OUT_DIR = Path("/Users/jwalinshah/projects/cerebras-gemma4-hackathon/runs")

SYSTEM_PROMPT = """You are a robot vision system looking at a real camera feed from a workspace.
Identify every object you can see in the image. For each object describe:
- What it is (name the object)
- Its color
- Its approximate position in the frame (left/center/right, top/bottom)
- Its size relative to other objects

Also describe the overall scene layout and note any hazards or obstacles."""


def run_gemma_on_image(image_path: Path, task: str) -> dict:
    """Send a real image to Gemma 4 and get back the analysis."""
    img = Image.open(image_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = f"""Task: {task}

Look at this camera image from a robot workspace. Identify every object you can see — describe its appearance, color, approximate position in the frame (left/center/right, top/bottom), shape, and any hazards present."""

    client = CerebrasClient()
    t0 = time.perf_counter()
    result = client.image_chat(
        prompt=prompt,
        image_b64=f"data:image/png;base64,{b64}",
        system_prompt=SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1024,
    )
    latency = (time.perf_counter() - t0) * 1000

    return {
        "file": image_path.name,
        "size": img.size,
        "prompt": prompt,
        "response": result.content,
        "latency_ms": latency,
        "usage": result.usage,
        "time_info": result.time_info,
    }


def main():
    tasks = {
        "scene_01_objects.png": "Describe the objects on this workspace table. What colors and positions do you see?",
        "scene_02_cluttered.png": "List all objects visible on this desk. Describe colors, positions, and identify any hazards.",
        "scene_03_tools.png": "What objects are on this work surface? Describe their colors and arrangement.",
        "scene_04_sparse.png": "What do you see in this workspace? Describe the object and its position.",
        "scene_05_dense.png": "This is a cluttered workspace. Identify every object, its color, and position.",
    }

    print("=" * 80)
    print("  GEMMA 4 ON PHOTO-REALISTIC WORKSPACE IMAGES")
    print("=" * 80)
    print()
    print(f"{'Image':25s} | {'Objects Identified':60s}")
    print("-" * 85)

    for fname, task in tasks.items():
        path = IMAGE_DIR / fname
        if not path.exists():
            print(f"{fname:25s} | ⚠️  FILE NOT FOUND")
            continue

        result = run_gemma_on_image(path, task)
        resp = result["response"]
        latency = result["latency_ms"]

        # Extract first line or summary
        first_line = resp.split("\n")[0][:58] if resp else "no response"
        print(f"{fname:25s} | {first_line:60s}")
        print(f"{'':25s} | ⚡ {latency:.0f}ms")

        # Save full response
        out_path = OUT_DIR / f"gemma_analysis_{fname.replace('.png', '.txt')}"
        with open(out_path, "w") as f:
            f.write(f"Image: {fname}\n")
            f.write(f"Size: {result['size'][0]}x{result['size'][1]}\n")
            f.write(f"Latency: {latency:.0f}ms\n")
            f.write(f"Task: {task}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"GEMMA 4 RESPONSE:\n")
            f.write(f"{'='*60}\n\n")
            f.write(resp)
            if result["usage"]:
                f.write(f"\n\nTokens: {result['usage']}\n")

        print(f"{'':25s} | 📝 Full response saved to {out_path.name}")
        print()

    print("-" * 85)


if __name__ == "__main__":
    main()
