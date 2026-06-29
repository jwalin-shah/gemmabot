#!/usr/bin/env python3
"""Clean video-first hero chart: Cerebras vs OpenRouter latency.

PIL-rendered (full layout control, no matplotlib cramping). Designed to be
shown FULL-SCREEN in the demo with no overlay badges on top — the speedups
are built in. 1920x1080, dark theme.

Reads the real summaries from overnight_results/compare_or/<kind>/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "overnight_results" / "compare_or"
OUT = BASE / "hero_chart.png"

W, H = 1920, 1080
BG = (10, 14, 23)
CARD = (19, 26, 43)
BORDER = (34, 46, 74)
CERE = (255, 107, 53)     # Cerebras orange
OPEN = (74, 158, 255)     # OpenRouter blue
WHITE = (240, 245, 255)
SUB = (150, 162, 186)
GREEN = (0, 212, 170)


def font(size, weight="regular"):
    paths = ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/SFNS.ttf"]
    if weight == "mono":
        paths = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"] + paths
    for p in paths:
        try: return ImageFont.truetype(p, size, index=1 if weight == "bold" else 0)
        except Exception:
            try: return ImageFont.truetype(p, size)
            except Exception: continue
    return ImageFont.load_default()


def load():
    rows = []
    for kind, label in [("text", "TEXT"), ("image", "VISION"), ("json", "VISION + JSON")]:
        s = json.loads((BASE / kind / "summary.json").read_text())
        rows.append({
            "label": label,
            "c_p50": s["cerebras"]["p50_ms"],
            "o_p50": s["openrouter"]["p50_ms"],
            "c_p95": s["cerebras"]["p95_ms"],
            "o_p95": s["openrouter"]["p95_ms"],
            "su50": s["speedup_p50_x"],
            "su95": s["speedup_p95_x"],
        })
    return rows


def fmt(ms):
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.1f} s"


def main():
    rows = load()
    img = Image.new("RGB", (W, H), BG)

    # subtle glow
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    gd.ellipse([-300, -300, 700, 500], fill=(255, 107, 53, 28))
    gd.ellipse([W-700, H-500, W+300, H+300], fill=(74, 158, 255, 24))
    g = g.filter(ImageFilter.GaussianBlur(130)); img.paste(g, (0, 0), g)

    d = ImageDraw.Draw(img)

    # ── Header ──
    f_eye = font(26, "mono")
    f_title = font(60, "bold")
    f_sub = font(34)
    d.text((90, 64), "SAME MODEL · SAME PROMPTS · ONLY THE SILICON CHANGES",
           fill=GREEN, font=f_eye)
    d.text((90, 108), "Gemma 4 31B inference latency", fill=WHITE, font=f_title)
    d.text((90, 184), "Cerebras WSE-3   vs   OpenRouter GPU   ·   p50, 20 calls each",
           fill=SUB, font=f_sub)

    # ── Legend (top-right) ──
    f_leg = font(28, "bold")
    lx = W - 520
    d.rounded_rectangle([lx, 70, lx+26, 96], radius=6, fill=CERE)
    d.text((lx+38, 66), "Cerebras", fill=WHITE, font=f_leg)
    d.rounded_rectangle([lx+230, 70, lx+256, 96], radius=6, fill=OPEN)
    d.text((lx+268, 66), "OpenRouter", fill=WHITE, font=f_leg)

    # ── Rows ──
    # Layout: label column | bar zone | speedup column
    label_x = 90
    bar_x = 430
    bar_max_w = 620          # width representing the largest latency on the chart
    # value label sits after the bar (~150px); speedup column must clear that
    speed_x = bar_x + bar_max_w + 230
    row_top = 290
    row_h = 250              # per workload
    bar_h = 56
    gap = 26                 # between the two bars in a row

    # Global scale: longest bar = max openrouter p50 across rows
    max_ms = max(r["o_p50"] for r in rows)

    f_row = font(40, "bold")
    f_val = font(30, "bold")
    f_su = font(76, "bold")
    f_su_cap = font(24, "mono")

    for i, r in enumerate(rows):
        y0 = row_top + i * row_h
        # workload label
        d.text((label_x, y0 + 34), r["label"], fill=WHITE, font=f_row)

        # Cerebras bar
        cw = max(int(r["c_p50"] / max_ms * bar_max_w), 8)
        d.rounded_rectangle([bar_x, y0, bar_x+cw, y0+bar_h], radius=10, fill=CERE)
        d.text((bar_x + cw + 18, y0 + 12), fmt(r["c_p50"]), fill=CERE, font=f_val)

        # OpenRouter bar
        ow = max(int(r["o_p50"] / max_ms * bar_max_w), 8)
        yb = y0 + bar_h + gap
        d.rounded_rectangle([bar_x, yb, bar_x+ow, yb+bar_h], radius=10, fill=OPEN)
        d.text((bar_x + ow + 18, yb + 12), fmt(r["o_p50"]), fill=OPEN, font=f_val)

        # Speedup callout (right)
        su = f"{r['su50']:.1f}×"
        d.text((speed_x, y0 + 18), su, fill=GREEN, font=f_su)
        d.text((speed_x, y0 + 108), "faster · p50", fill=SUB, font=f_su_cap)
        # p95 sub-note
        d.text((speed_x, y0 + 140), f"{r['su95']:.0f}× at p95 tail", fill=SUB, font=f_su_cap)

        # row divider
        if i < len(rows) - 1:
            d.line([(label_x, y0 + row_h - 36), (W - 90, y0 + row_h - 36)],
                   fill=BORDER, width=2)

    # ── Footer ──
    f_foot = font(24, "mono")
    foot = ("Cerebras: api.cerebras.ai · gemma-4-31b      "
            "OpenRouter: openrouter.ai · google/gemma-3-27b-it      "
            "live measurement · scripts/compare_cerebras_openrouter.py")
    d.text((90, H - 70), foot, fill=SUB, font=f_foot)

    img.save(OUT)
    print(f"WROTE: {OUT}")
    # 720 variant
    img.resize((1280, 720), Image.LANCZOS).save(BASE / "hero_chart_720.png")
    print(f"WROTE: {BASE / 'hero_chart_720.png'}")


if __name__ == "__main__":
    main()
