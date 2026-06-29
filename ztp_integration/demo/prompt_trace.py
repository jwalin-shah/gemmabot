"""
PROMPT TRACE: Exact text + image sent to Gemma 4 on Cerebras.

This uses a REAL PHOTO (workspace.jpg) and shows exactly what
we send, what Gemma sees, and what it responds with.
"""

from __future__ import annotations

import base64, json, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ===================================================================
# Load real image
# ===================================================================
from PIL import Image
import io

img = Image.open("/Users/jwalinshah/projects/cerebras-gemma4-hackathon/examples/images/workspace.jpg")
print(f"📷 REAL IMAGE: workspace.jpg — {img.size[0]}x{img.size[1]}px")
print()

# Encode to base64
buf = io.BytesIO()
img.save(buf, format="JPEG")
image_bytes = buf.getvalue()
image_b64 = base64.b64encode(image_bytes).decode()
print(f"   Raw bytes: {len(image_bytes):,} bytes ({len(image_bytes)/1024:.1f}KB)")
print(f"   Base64:    {len(image_b64):,} characters")
print()

# ===================================================================
# Build the EXACT prompt text
# ===================================================================
instruction = "Identify the objects visible in this workspace image. List everything you see, their colors, positions (left/center/right, top/bottom), and describe the layout of the scene. Also identify any hazards."

prompt = f"""Task: {instruction}

Look at this camera image from a robot workspace. Identify every object you can see — describe its appearance, color, approximate position in the frame, and any hazards or obstacles present."""

print("=" * 70)
print("  EXACT TEXT PROMPT SENT TO GEMMA 4")
print("=" * 70)
print(prompt)
print()
print(f"  Total prompt length: {len(prompt)} chars")
print()

# ===================================================================
# Send to Cerebras
# ===================================================================
from src.client import CerebrasClient

client = CerebrasClient()
system = "You are a robot vision system. Analyze workspace images and identify objects, their positions, colors, and hazards. Be thorough and specific."

print("⚡ Sending to Cerebras API...")
print(f"   Model: gemma-4-31b")
print(f"   Image: workspace.jpg ({len(image_bytes)} bytes)")
print()

t0 = time.perf_counter()
result = client.image_chat(
    prompt=prompt,
    image_b64=f"data:image/jpeg;base64,{image_b64}",
    system_prompt=system,
    temperature=0.1,
    max_tokens=1024,
)
latency = (time.perf_counter() - t0) * 1000

print("=" * 70)
print(f"  GEMMA 4 RESPONSE [{latency:.0f}ms]")
print("=" * 70)
print(result.content)
print()

# ===================================================================
# Show usage info
# ===================================================================
if result.usage:
    print(f"  Token usage: {result.usage}")
if result.time_info:
    print(f"  Time info:   {result.time_info}")
print()
