#!/usr/bin/env python3
"""Stitch the 60-second submission demo from clips + chart.

Output: overnight_results/videos/gemmabot_demo_60s.mp4

Captions are pre-rendered as PNG overlays (PIL) and composited by ffmpeg --
the homebrew ffmpeg build lacks libfreetype, so drawtext isn't available.

Beats (sum to 60s):
  title 4s | lift 8s | stack 12s | chart 20s | fail 8s | outro 8s
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "overnight_results" / "videos"
CHART = ROOT / "overnight_results" / "compare_or" / "comparison_chart.png"
OUT = VID / "gemmabot_demo_60s.mp4"
WORK = VID / "_build"

W, H = 1920, 1080
FPS = 30
BG = (10, 14, 23)        # #0a0e17
ACCENT = (255, 107, 53)  # #FF6B35
WHITE = (240, 245, 255)
SUB = (170, 180, 200)

DUR_TITLE, DUR_LIFT, DUR_STACK, DUR_CHART, DUR_FAIL, DUR_OUTRO = 4, 8, 12, 20, 8, 8

CAPTIONS = {
    "lift":   "No ground truth. Gemma identified the cube. Arm picked it up.",
    "stack":  "Perception (~1cm) + Gemma (vision) + executor + judge",
    "chart":  "Same model. Same prompts. Only the silicon changes.",
    "fail":   "Honest: at low resolution it confuses items. We name the limit.",
}

# ── Font loading (Helvetica on macOS) ─────────────────────────────────
def font(size: int) -> ImageFont.FreeTypeFont:
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/SFNS.ttf",
              "/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def run(cmd: list[str]) -> None:
    print(">>", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:]); sys.exit(1)


# ── Image helpers ─────────────────────────────────────────────────────
def make_title_png(out: Path, title: str, subtitle: str) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title = font(140)
    f_sub = font(48)
    t_w = d.textlength(title, font=f_title)
    d.text(((W - t_w) / 2, H/2 - 130), title, fill=WHITE, font=f_title)
    s_w = d.textlength(subtitle, font=f_sub)
    d.text(((W - s_w) / 2, H/2 + 50), subtitle, fill=ACCENT, font=f_sub)
    img.save(out)


def make_caption_overlay(out: Path, caption: str) -> None:
    """RGBA caption strip on a transparent background — composited later."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(46)
    pad_x, pad_y = 48, 28
    text_w = d.textlength(caption, font=f)
    box_w = int(text_w) + pad_x * 2
    box_h = 46 + pad_y * 2
    box_x = (W - box_w) // 2
    box_y = H - 140
    d.rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                fill=(0, 0, 0, 180))
    d.text((box_x + pad_x, box_y + pad_y - 5), caption, fill=WHITE, font=f)
    img.save(out)


# ── Clip builders ─────────────────────────────────────────────────────
def make_image_clip(img: Path, dur: int, out: Path) -> None:
    """Hold a still image full-screen for `dur` seconds."""
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(dur), "-i", str(img),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def make_chart_clip_with_caption(img: Path, overlay: Path, dur: int, out: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(dur), "-i", str(img),
        "-loop", "1", "-t", str(dur), "-i", str(overlay),
        "-filter_complex",
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def make_video_clip_with_caption(src: Path, overlay: Path, dur: int, out: Path) -> None:
    """Loop+trim a source mp4, scaled to 1920x1080, with a PNG caption overlay."""
    run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(src),
        "-loop", "1", "-t", str(dur), "-i", str(overlay),
        "-t", str(dur),
        "-filter_complex",
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]",
        "-an",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


# ── Main ──────────────────────────────────────────────────────────────
def main():
    for p in [VID / "nocheat_Lift.mp4", VID / "nocheat_Stack.mp4",
              VID / "nocheat_PickPlace.mp4", CHART]:
        if not p.exists():
            print(f"MISSING: {p}"); sys.exit(1)

    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print(f"Building 60s demo -> {OUT.relative_to(ROOT)}")

    # 1. Title card (PNG -> mp4)
    make_title_png(WORK / "title.png", "GemmaBot", "Gemma 4 31B  ·  Cerebras WSE-3  ·  honest pipeline")
    make_image_clip(WORK / "title.png", DUR_TITLE, WORK / "00_title.mp4")

    # Caption overlays
    for key, text in CAPTIONS.items():
        make_caption_overlay(WORK / f"cap_{key}.png", text)

    # 2-5: clip+caption
    make_video_clip_with_caption(VID / "nocheat_Lift.mp4", WORK / "cap_lift.png",
                                  DUR_LIFT, WORK / "10_lift.mp4")
    make_video_clip_with_caption(VID / "nocheat_Stack.mp4", WORK / "cap_stack.png",
                                  DUR_STACK, WORK / "20_stack.mp4")
    make_chart_clip_with_caption(CHART, WORK / "cap_chart.png",
                                  DUR_CHART, WORK / "30_chart.mp4")
    make_video_clip_with_caption(VID / "nocheat_PickPlace.mp4", WORK / "cap_fail.png",
                                  DUR_FAIL, WORK / "40_fail.mp4")

    # 6. Outro card
    make_title_png(WORK / "outro.png", "Honest pipeline.",
                   "Honest numbers.  Code in the repo.")
    make_image_clip(WORK / "outro.png", DUR_OUTRO, WORK / "50_outro.mp4")

    # Concat
    concat_list = WORK / "concat.txt"
    concat_list.write_text("\n".join([
        f"file '{(WORK / n).resolve()}'"
        for n in ["00_title.mp4", "10_lift.mp4", "20_stack.mp4",
                  "30_chart.mp4", "40_fail.mp4", "50_outro.mp4"]
    ]))
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(OUT),
    ])
    size_mb = OUT.stat().st_size / 1_048_576
    print(f"\nDONE: {OUT}  ({size_mb:.1f} MB, ~60s @ 1920x1080)")


if __name__ == "__main__":
    main()
