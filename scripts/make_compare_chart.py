#!/usr/bin/env python3
"""Render the Cerebras-vs-OpenRouter comparison chart from the JSONL outputs
of compare_cerebras_openrouter.py. Writes a 1920x1080 PNG suitable for the
submission video.

Usage:
    uv run python scripts/make_compare_chart.py
    # → overnight_results/compare_or/comparison_chart.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "overnight_results" / "compare_or"
KINDS = ["text", "image", "json"]
KIND_LABEL = {
    "text":  "Text\n(no image)",
    "image": "Vision\n(1 image)",
    "json":  "Vision + JSON\n(structured out)",
}

CEREBRAS_COLOR = "#FF6B35"     # warm orange — Cerebras brand
OPENROUTER_COLOR = "#4A9EFF"   # blue — generic GPU
BG = "#0a0e17"
GRID = "#1e2a45"
TEXT = "#e8edf5"
SUB = "#8892b0"


def load_summary(kind: str) -> dict:
    p = OUT_DIR / kind / "summary.json"
    return json.loads(p.read_text())


def fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def main() -> None:
    summaries = {k: load_summary(k) for k in KINDS}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": SUB,
        "ytick.color": SUB,
        "text.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
    })

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.suptitle("Gemma 4 31B inference: Cerebras WSE-3 vs OpenRouter GPU",
                 fontsize=28, fontweight="bold", color=TEXT, y=0.96)
    fig.text(0.5, 0.91,
             "Same model. Same prompts. Same images. Only the silicon changes.",
             ha="center", fontsize=15, color=SUB, style="italic")

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                          left=0.06, right=0.96, top=0.85, bottom=0.07)

    # ── Top row: p50 / p95 bar charts per kind ──────────────────────────
    x = np.arange(len(KINDS))
    width = 0.36
    p50_c = [summaries[k]["cerebras"]["p50_ms"] for k in KINDS]
    p50_o = [summaries[k]["openrouter"]["p50_ms"] for k in KINDS]
    p95_c = [summaries[k]["cerebras"]["p95_ms"] for k in KINDS]
    p95_o = [summaries[k]["openrouter"]["p95_ms"] for k in KINDS]
    speedup_p50 = [round(o / c, 1) for c, o in zip(p50_c, p50_o)]
    speedup_p95 = [round(o / c, 1) for c, o in zip(p95_c, p95_o)]

    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x - width/2, p50_c, width, label="Cerebras",  color=CEREBRAS_COLOR, edgecolor="none")
    bars2 = ax1.bar(x + width/2, p50_o, width, label="OpenRouter", color=OPENROUTER_COLOR, edgecolor="none")
    ax1.set_title("Median latency (p50)", fontsize=16, color=TEXT, pad=14, fontweight="bold")
    ax1.set_ylabel("ms", fontsize=12, color=SUB)
    ax1.set_xticks(x)
    ax1.set_xticklabels([KIND_LABEL[k] for k in KINDS], fontsize=11)
    ax1.legend(loc="upper left", facecolor=BG, edgecolor=GRID, framealpha=0.9, fontsize=11)
    for b, v in zip(bars1, p50_c):
        ax1.text(b.get_x() + b.get_width()/2, v, fmt_ms(v), ha="center", va="bottom",
                 color=CEREBRAS_COLOR, fontsize=11, fontweight="bold")
    for b, v in zip(bars2, p50_o):
        ax1.text(b.get_x() + b.get_width()/2, v, fmt_ms(v), ha="center", va="bottom",
                 color=OPENROUTER_COLOR, fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    bars3 = ax2.bar(x - width/2, p95_c, width, label="Cerebras",  color=CEREBRAS_COLOR, edgecolor="none")
    bars4 = ax2.bar(x + width/2, p95_o, width, label="OpenRouter", color=OPENROUTER_COLOR, edgecolor="none")
    ax2.set_title("Tail latency (p95)", fontsize=16, color=TEXT, pad=14, fontweight="bold")
    ax2.set_ylabel("ms", fontsize=12, color=SUB)
    ax2.set_xticks(x)
    ax2.set_xticklabels([KIND_LABEL[k] for k in KINDS], fontsize=11)
    for b, v in zip(bars3, p95_c):
        ax2.text(b.get_x() + b.get_width()/2, v, fmt_ms(v), ha="center", va="bottom",
                 color=CEREBRAS_COLOR, fontsize=11, fontweight="bold")
    for b, v in zip(bars4, p95_o):
        ax2.text(b.get_x() + b.get_width()/2, v, fmt_ms(v), ha="center", va="bottom",
                 color=OPENROUTER_COLOR, fontsize=11, fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2])
    bars5 = ax3.bar(x, speedup_p50, width=0.55, color=CEREBRAS_COLOR, edgecolor="none")
    ax3.axhline(1, color=GRID, linestyle="--", linewidth=1)
    ax3.set_title("Cerebras speedup over OpenRouter (p50)",
                  fontsize=16, color=TEXT, pad=14, fontweight="bold")
    ax3.set_ylabel("× faster", fontsize=12, color=SUB)
    ax3.set_xticks(x)
    ax3.set_xticklabels([KIND_LABEL[k] for k in KINDS], fontsize=11)
    for b, v in zip(bars5, speedup_p50):
        ax3.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}×",
                 ha="center", va="bottom", color=CEREBRAS_COLOR, fontsize=13, fontweight="bold")
    ax3.set_ylim(0, max(speedup_p50) * 1.2)

    # ── Bottom row: per-call latency distribution scatter (text kind) ─
    # Stack all three kinds along x so the eye sees the WHOLE distribution
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title("Every individual call, sorted (lower = faster, log scale)",
                  fontsize=16, color=TEXT, pad=14, fontweight="bold")

    offset = 0
    tick_positions = []
    tick_labels = []
    for kind in KINDS:
        s = summaries[kind]
        c_lats = sorted(s["cerebras"]["raw_latencies_ms"])
        o_lats = sorted(s["openrouter"]["raw_latencies_ms"])
        xc = np.arange(len(c_lats)) + offset
        xo = np.arange(len(o_lats)) + offset
        ax4.plot(xc, c_lats, "o-", color=CEREBRAS_COLOR, markersize=6,
                 linewidth=1.5, alpha=0.95, label="Cerebras" if offset == 0 else None)
        ax4.plot(xo, o_lats, "o-", color=OPENROUTER_COLOR, markersize=6,
                 linewidth=1.5, alpha=0.95, label="OpenRouter" if offset == 0 else None)
        tick_positions.append(offset + len(c_lats) / 2 - 0.5)
        tick_labels.append(KIND_LABEL[kind].replace("\n", " "))
        if kind != KINDS[-1]:
            ax4.axvline(offset + len(c_lats) - 0.5, color=GRID, linestyle="--", alpha=0.6)
        offset += len(c_lats) + 2

    ax4.set_yscale("log")
    ax4.set_xlabel("Calls (within each prompt kind, sorted by latency)", fontsize=12, color=SUB)
    ax4.set_ylabel("Latency (ms, log scale)", fontsize=12, color=SUB)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, fontsize=12, color=TEXT)
    ax4.legend(loc="upper left", facecolor=BG, edgecolor=GRID, framealpha=0.9, fontsize=12)
    ax4.grid(True, which="both", alpha=0.4)

    # Footer
    fig.text(0.5, 0.01,
             "Cerebras: api.cerebras.ai · gemma-4-31b  |  OpenRouter: openrouter.ai · google/gemma-3-27b-it"
             "  |  Run from scripts/compare_cerebras_openrouter.py · live measurement, no synthetic data",
             ha="center", fontsize=10, color=SUB, style="italic")

    out_path = OUT_DIR / "comparison_chart.png"
    fig.savefig(str(out_path), dpi=100, facecolor=BG)
    print(f"WROTE: {out_path}")
    # Also a smaller social-format
    fig.set_size_inches(12.8, 7.2)
    out_path2 = OUT_DIR / "comparison_chart_720.png"
    fig.savefig(str(out_path2), dpi=100, facecolor=BG)
    print(f"WROTE: {out_path2}")


if __name__ == "__main__":
    main()
