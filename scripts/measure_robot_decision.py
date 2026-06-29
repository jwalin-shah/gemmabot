#!/usr/bin/env python3
"""Measure REAL robot-decision latency on both providers, paired.

Not the bare 18-token API ping — the actual decision the robot makes:
one scene image + a tool choice + 2-3 sentences of reasoning (~300 tokens out).
This is the honest number for the split-screen timers.
"""
from __future__ import annotations
import os, sys, json, time, statistics, urllib.request, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING", "1")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from src.client import CerebrasClient
from src.web.lib.imaging import img_to_b64
import numpy as np, imageio

frame = imageio.get_reader(str(_ROOT / "overnight_results/videos/full_pick.mp4")).get_data(0)
img_b64 = img_to_b64(np.array(frame), fmt="JPEG", quality=60)

PROMPT = ("You are a Franka Panda robot arm. Look at this scene with grocery items "
          "(soda can, milk, bread, cereal) and choose ONE tool to pick up the soda can. "
          "Explain your reasoning in 2-3 sentences. Respond as JSON: "
          '{"tool":"grasp","object":"Can","reasoning":"..."}')

OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = os.environ["OPENROUTER_BASE_URL"]
OR_MODEL = os.environ["OPENROUTER_MODEL"]


def call_or():
    body = json.dumps({"model": OR_MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": img_b64}}]}],
        "max_tokens": 300, "temperature": 0}).encode()
    req = urllib.request.Request(OR_URL + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=90).read())
    return (time.perf_counter() - t0) * 1000, r.get("usage", {}).get("completion_tokens")


c = CerebrasClient()
def call_cer():
    t0 = time.perf_counter()
    r = c.image_chat(prompt=PROMPT, image_b64=img_b64, max_tokens=300, temperature=0)
    return (time.perf_counter() - t0) * 1000, (r.usage or {}).get("completion_tokens")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cer, orr = [], []
    print(f"Measuring {n} paired robot-decision calls (image + ~300tok reasoning)...")
    for i in range(n):
        lc, tc = call_cer(); cer.append(lc)
        lo, to = call_or(); orr.append(lo)
        print(f"  [{i+1}] Cerebras {lc:6.0f}ms ({tc}tok)  |  OpenRouter {lo:7.0f}ms ({to}tok)")
    cer.sort(); orr.sort()
    out = {
        "cerebras_ms": [round(x) for x in cer],
        "openrouter_ms": [round(x) for x in orr],
        "cer_p50": round(statistics.median(cer)),
        "or_p50": round(statistics.median(orr)),
        "cer_max": round(cer[-1]), "or_max": round(orr[-1]),
        "speedup_p50": round(statistics.median(orr) / statistics.median(cer), 1),
        "prompt": "scene image + tool choice + 2-3 sentence reasoning (~300 tok out)",
        "n": n,
    }
    (_ROOT / "overnight_results/compare_or/robot_decision.json").write_text(json.dumps(out, indent=2))
    print(f"\nCEREBRAS  robot-decision p50 = {out['cer_p50']}ms  (max {out['cer_max']}ms)")
    print(f"OPENROUTER robot-decision p50 = {out['or_p50']}ms  (max {out['or_max']}ms)")
    print(f"SPEEDUP p50 = {out['speedup_p50']}x")
    print("saved overnight_results/compare_or/robot_decision.json")


if __name__ == "__main__":
    main()
