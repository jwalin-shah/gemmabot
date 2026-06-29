#!/usr/bin/env python3
"""GemmaBot 60s demo v3 — locked in, punchy.

Structure (60s):
  0:00–0:03  HARD CUT cold open: 3 frames of the Lift video, then BIG title
              flash for 1.5s. No long intro card.
  0:03–0:23  HERO: Lift video (4× loop, 0.7x speed for impact) with a
              live "GEMMA → cube · 600ms" HUD in the corner.
              At 0:13 a sub-caption fades in: "no ground truth. only the image."
  0:23–0:45  CHART: full-screen comparison_chart.png with a big animated
              "13x · 5x · 39x at tail" callout
  0:45–0:55  STATS slate: massive single number per beat (cuts every 2.5s)
              "200ms"  ·  "1cm"  ·  "88%"  ·  "0 ground truth"
  0:55–1:00  END: GemmaBot logo + "code: github.com/..." (clean close)

All overlays are pre-rendered PNGs (PIL) then composited (this ffmpeg has no
drawtext).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "overnight_results" / "videos"
CHART = ROOT / "overnight_results" / "compare_or" / "comparison_chart.png"
OUT = VID / "gemmabot_demo_60s.mp4"
WORK = VID / "_build3"

W, H = 1920, 1080
FPS = 30

BG = (10, 14, 23)
PANEL = (19, 26, 43)
BORDER = (30, 42, 69)
ACCENT = (255, 107, 53)
ACCENT2 = (0, 212, 170)
WHITE = (240, 245, 255)
SUB = (170, 180, 200)


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    paths = ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc",
             "/System/Library/Fonts/SFNS.ttf",
             "/Library/Fonts/Arial.ttf"]
    if weight == "mono":
        paths = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"] + paths
    for p in paths:
        try:
            return ImageFont.truetype(p, size, index=1 if weight == "bold" else 0)
        except Exception:
            try: return ImageFont.truetype(p, size)
            except Exception: continue
    return ImageFont.load_default()


def run(cmd: list[str]) -> None:
    print(">>", " ".join(str(c) for c in cmd[:5]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit(1)


# ── PNG factories ─────────────────────────────────────────────────────
def make_cold_title(out: Path) -> None:
    """Tight title card: big mark + tagline, no logos, no fluff."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # Soft glow
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    gd.ellipse([-200,-200, 800, 600], fill=(255,107,53, 45))
    gd.ellipse([W-600, H-400, W+200, H+200], fill=(0,212,170, 35))
    g = g.filter(ImageFilter.GaussianBlur(120))
    img.paste(g, (0,0), g)

    d = ImageDraw.Draw(img)
    f_mark = font(220, "bold")
    f_tag = font(50)
    f_eye = font(24, "mono")

    mark = "GemmaBot"
    tag = "Gemma 4 31B  ·  Cerebras WSE-3"
    eye = "PERCEPTION → REASONING → MOTION"

    e_w = d.textlength(eye, font=f_eye)
    d.text(((W - e_w) / 2, H/2 - 230), eye, fill=ACCENT2, font=f_eye)
    m_w = d.textlength(mark, font=f_mark)
    d.text(((W - m_w) / 2, H/2 - 170), mark, fill=WHITE, font=f_mark)
    t_w = d.textlength(tag, font=f_tag)
    d.text(((W - t_w) / 2, H/2 + 80), tag, fill=ACCENT, font=f_tag)
    img.save(out)


def make_hud_overlay(out: Path, label: str, value: str,
                     subcap: str | None = None) -> None:
    """Transparent HUD: small chip top-left, optional sub-caption lower-third."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Top-left HUD chip
    f_lbl = font(20, "mono")
    f_val = font(38, "bold")
    lbl_w = d.textlength(label, font=f_lbl)
    val_w = d.textlength(value, font=f_val)
    chip_w = int(max(lbl_w, val_w) + 60)
    chip_h = 110
    cx, cy = 50, 50
    d.rounded_rectangle([cx, cy, cx+chip_w, cy+chip_h], radius=12,
                        fill=(0,0,0,200), outline=(60,72,96,255), width=2)
    d.text((cx+30, cy+14), label, fill=ACCENT2, font=f_lbl)
    d.text((cx+30, cy+40), value, fill=WHITE, font=f_val)

    # Top-right brand chip
    f_b = font(22, "mono")
    brand = "CEREBRAS · GEMMA 4"
    bw = d.textlength(brand, font=f_b)
    bcw = int(bw) + 50
    bx = W - 50 - bcw
    d.rounded_rectangle([bx, 50, bx+bcw, 50+50], radius=10,
                        fill=ACCENT, outline=None)
    d.text((bx + 25, 50+14), brand, fill=(0, 0, 0), font=f_b)

    # Sub-caption (only on demand)
    if subcap:
        f_s = font(48, "bold")
        f_e = font(20, "mono")
        eyebrow = "INTEGRITY"
        sw = d.textlength(subcap, font=f_s)
        ew = d.textlength(eyebrow, font=f_e)
        bx2 = (W - max(sw, ew + 200)) / 2 - 50
        bw2 = max(sw, ew + 200) + 100
        bh2 = 130
        by2 = H - bh2 - 80
        d.rounded_rectangle([bx2, by2, bx2+bw2, by2+bh2], radius=14,
                            fill=(0,0,0,210))
        d.text((bx2+50, by2+20), eyebrow, fill=ACCENT2, font=f_e)
        d.text(((W-sw)/2, by2+52), subcap, fill=WHITE, font=f_s)

    img.save(out)


def make_chart_overlay(out: Path) -> None:
    """Three big callouts over the chart, plus eyebrow strip top."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Top eyebrow strip
    f_e = font(28, "mono")
    eyebrow = "SAME MODEL.   SAME PROMPTS.   ONLY THE SILICON CHANGES."
    ew = d.textlength(eyebrow, font=f_e)
    d.rounded_rectangle([(W-ew)/2 - 30, 30, (W+ew)/2 + 30, 90],
                        radius=8, fill=(0,0,0,210))
    d.text(((W-ew)/2, 44), eyebrow, fill=ACCENT2, font=f_e)

    # Three big number badges in the lower-third
    f_big = font(90, "bold")
    f_cap = font(22, "mono")
    badges = [
        ("13×", "TEXT p50"),
        ("5×",  "VISION p50"),
        ("39×", "VISION p95 (tail)"),
    ]
    bw = 360
    gap = 60
    total = bw * 3 + gap * 2
    start_x = (W - total) / 2
    y_top = H - 280
    for i, (num, cap) in enumerate(badges):
        x0 = start_x + i * (bw + gap)
        d.rounded_rectangle([x0, y_top, x0+bw, y_top+200], radius=18,
                            fill=ACCENT, outline=None)
        nw = d.textlength(num, font=f_big)
        d.text((x0 + (bw - nw)/2, y_top + 20), num, fill=(0,0,0), font=f_big)
        cw = d.textlength(cap, font=f_cap)
        d.text((x0 + (bw - cw)/2, y_top + 145), cap, fill=(0,0,0), font=f_cap)
    # Subtitle under badges
    f_s = font(28)
    sub = "Cerebras faster than OpenRouter GPU"
    sw = d.textlength(sub, font=f_s)
    d.text(((W - sw)/2, y_top + 220), sub, fill=WHITE, font=f_s)
    img.save(out)


def make_stat_slate(out: Path, big: str, caption: str, eyebrow: str) -> None:
    """One huge number slate. Used in the 0:45–0:55 stat sequence."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    g = Image.new("RGBA", (W, H), (0,0,0,0))
    gd = ImageDraw.Draw(g)
    gd.ellipse([W/2-500, H/2-500, W/2+500, H/2+500], fill=(255,107,53, 30))
    g = g.filter(ImageFilter.GaussianBlur(120))
    img.paste(g, (0,0), g)

    d = ImageDraw.Draw(img)
    f_e = font(28, "mono")
    f_b = font(380, "bold")
    f_c = font(46)
    ew = d.textlength(eyebrow, font=f_e)
    d.text(((W-ew)/2, H/2 - 280), eyebrow, fill=ACCENT2, font=f_e)
    bw_ = d.textlength(big, font=f_b)
    d.text(((W - bw_)/2, H/2 - 220), big, fill=WHITE, font=f_b)
    cw = d.textlength(caption, font=f_c)
    d.text(((W - cw)/2, H/2 + 200), caption, fill=ACCENT, font=f_c)
    img.save(out)


def make_end_card(out: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    g = Image.new("RGBA", (W, H), (0,0,0,0))
    gd = ImageDraw.Draw(g)
    gd.ellipse([-200,-200, 800, 600], fill=(0,212,170, 45))
    gd.ellipse([W-600, H-400, W+200, H+200], fill=(255,107,53, 35))
    g = g.filter(ImageFilter.GaussianBlur(120))
    img.paste(g, (0,0), g)

    d = ImageDraw.Draw(img)
    f_b = font(170, "bold")
    f_t = font(46)
    f_e = font(24, "mono")
    eye = "CEREBRAS × GEMMA 4 HACKATHON"
    ew = d.textlength(eye, font=f_e)
    d.text(((W-ew)/2, H/2 - 240), eye, fill=ACCENT2, font=f_e)
    big = "GemmaBot"
    bw_ = d.textlength(big, font=f_b)
    d.text(((W-bw_)/2, H/2 - 180), big, fill=WHITE, font=f_b)
    tag = "honest pipeline · live numbers · code in the repo"
    tw = d.textlength(tag, font=f_t)
    d.text(((W-tw)/2, H/2 + 60), tag, fill=ACCENT, font=f_t)
    img.save(out)


# ── Clip builders ─────────────────────────────────────────────────────
def img_clip(img: Path, dur: float, out: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(dur), "-i", str(img),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def img_overlay_clip(img: Path, overlay: Path, dur: float, out: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(dur), "-i", str(img),
        "-loop", "1", "-t", str(dur), "-i", str(overlay),
        "-filter_complex",
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def video_overlay_clip(src: Path, overlay: Path, dur: float, out: Path,
                       speed: float = 1.0) -> None:
    """Loop+trim source video at given playback speed, with overlay."""
    setpts = f"setpts={1.0/speed:.3f}*PTS"
    run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(src),
        "-loop", "1", "-t", str(dur), "-i", str(overlay),
        "-t", str(dur),
        "-filter_complex",
            f"[0:v]{setpts},scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];"
            f"[bg][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-an", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


# ── Main ──────────────────────────────────────────────────────────────
def main():
    for p in [VID / "nocheat_Lift.mp4", CHART]:
        if not p.exists():
            print(f"MISSING: {p}"); sys.exit(1)

    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print(f"Building v3 60s -> {OUT.relative_to(ROOT)}\n")

    # 1) COLD TITLE 0:00–0:03 (3s)
    make_cold_title(WORK / "title.png")
    img_clip(WORK / "title.png", 3, WORK / "00_title.mp4")

    # 2) HERO Lift video 0:03–0:23 (20s)
    # Lift mp4 is 2.2s — loop ~9x at 0.85x speed for impact = 23.5s of content;
    # we cap to 20s. Overlay has a HUD and a fade-in sub-caption at the bottom.
    make_hud_overlay(WORK / "hud_a.png", "INFERENCE", "611 ms")
    video_overlay_clip(VID / "nocheat_Lift.mp4", WORK / "hud_a.png",
                       dur=10, out=WORK / "10_hero_a.mp4", speed=0.85)

    make_hud_overlay(WORK / "hud_b.png", "PERCEPTION", "~1 cm",
                     subcap="no ground truth. just the image.")
    video_overlay_clip(VID / "nocheat_Lift.mp4", WORK / "hud_b.png",
                       dur=10, out=WORK / "11_hero_b.mp4", speed=0.85)

    # 3) CHART 0:23–0:45 (22s)
    make_chart_overlay(WORK / "chart_ov.png")
    img_overlay_clip(CHART, WORK / "chart_ov.png", 22, WORK / "20_chart.mp4")

    # 4) STAT SLATES 0:45–0:55 (10s, 4 x 2.5s)
    make_stat_slate(WORK / "s1.png", "200 ms",
                    "median inference, vision call",
                    "WHAT GEMMA DELIVERS ON CEREBRAS")
    make_stat_slate(WORK / "s2.png", "1 cm",
                    "perception localization error",
                    "HOW WELL WE FIND OBJECTS")
    make_stat_slate(WORK / "s3.png", "88 %",
                    "visual reasoning, un-gameable test",
                    "HOW WELL GEMMA SEES")
    make_stat_slate(WORK / "s4.png", "0",
                    "ground-truth coordinates in the loop",
                    "HOW MUCH WE CHEATED")
    img_clip(WORK / "s1.png", 2.5, WORK / "30_s1.mp4")
    img_clip(WORK / "s2.png", 2.5, WORK / "31_s2.mp4")
    img_clip(WORK / "s3.png", 2.5, WORK / "32_s3.mp4")
    img_clip(WORK / "s4.png", 2.5, WORK / "33_s4.mp4")

    # 5) END 0:55–1:00 (5s)
    make_end_card(WORK / "end.png")
    img_clip(WORK / "end.png", 5, WORK / "40_end.mp4")

    # Concat
    files = ["00_title.mp4",
             "10_hero_a.mp4", "11_hero_b.mp4",
             "20_chart.mp4",
             "30_s1.mp4", "31_s2.mp4", "32_s3.mp4", "33_s4.mp4",
             "40_end.mp4"]
    cl = WORK / "concat.txt"
    cl.write_text("\n".join(f"file '{(WORK / n).resolve()}'" for n in files))
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(cl), "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(OUT),
    ])
    mb = OUT.stat().st_size / 1_048_576
    print(f"\nDONE: {OUT}  ({mb:.1f} MB)")
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(OUT)],
                       capture_output=True, text=True)
    print(f"      duration: {r.stdout.strip()}s  (target 60)")


if __name__ == "__main__":
    main()
