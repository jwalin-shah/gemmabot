#!/usr/bin/env python3
"""Split-screen "speed race": Cerebras WSE-3 vs OpenRouter GPU.

Same robot Lift task, side by side. Both panes play the *same* physical
motion (overnight_results/videos/nocheat_Lift.mp4) — what differs is the
THINKING TIME between moves. We pace each pane's playback to its REAL
measured per-inference latency so the Cerebras arm finishes the task while
the OpenRouter arm is still only partway.

Honest model (per the brief):
  total wall time = motion_time + (num_decisions * per_call_latency)
    Cerebras   : 2.2s motion + 5 * 0.200s = 3.2s
    OpenRouter : 2.2s motion + 5 * 1.088s = 7.6s   (~5.4x slower per call)

Measured numbers (overnight_results/compare_or/image/summary.json):
    Cerebras   vision p50 ~= 199 ms   (we use 200 ms)
    OpenRouter vision p50 ~= 1088 ms
    speedup p50 = 5.47x , p95 = 38.7x  -> headline "5x faster ... 39x at tail"

This ffmpeg has no drawtext (no libfreetype), so ALL text is rendered as
PNGs with PIL and composited via `overlay`. We pre-render a per-frame PNG
overlay sequence (10 fps) carrying the pane labels, the live elapsed
stopwatch / decision counter on each side, the headline, and the disclaimer,
then composite it onto the stacked video.

Run:  uv run python scripts/split_compare.py
Out:  overnight_results/videos/split_race.mp4   (1920x1080, h264, yuv420p, ~14s)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "overnight_results" / "videos"
LIFT = VID / "nocheat_Lift.mp4"
OUT = VID / "split_race.mp4"
WORK = VID / "_build_split"

W, H = 1920, 1080
PANE_W = 960
FPS = 30
OVR_FPS = 10  # overlay PNG sequence rate

# ── timing (seconds) ──────────────────────────────────────────────────
MOTION = 2.2          # raw Lift clip length
N_DEC = 5             # decisions in the task
CER_CALL = 0.200      # Cerebras per-call latency (p50, measured ~199ms)
OR_CALL = 1.088       # OpenRouter per-call latency (p50, measured)
CER_TOTAL = round(MOTION + N_DEC * CER_CALL, 1)   # 3.2s
OR_TOTAL = round(MOTION + N_DEC * OR_CALL, 1)     # 7.6s

TITLE_DUR = 3.0
RACE_DUR = 8.0        # both panes play; OR finishes ~7.6s
HOLD_DUR = 3.0        # both hold final frame so the gap lands
RACEHOLD = RACE_DUR + HOLD_DUR   # 11s of video portion
TOTAL = TITLE_DUR + RACEHOLD     # 14s

# ── palette ───────────────────────────────────────────────────────────
BG = (10, 14, 23)             # #0a0e17
CER = (255, 107, 53)          # #FF6B35 orange
ORN = (74, 158, 255)          # #4A9EFF blue
WHITE = (240, 245, 255)       # #f0f5ff
CYAN = (0, 212, 170)          # #00d4aa
SUB = (150, 162, 185)
GREEN = (60, 220, 130)
AMBER = (255, 190, 70)
DIVIDER = (40, 54, 86)


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    paths = ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc",
             "/Library/Fonts/Arial.ttf"]
    if weight == "mono":
        paths = ["/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Monaco.ttf"] + paths
    for p in paths:
        try:
            return ImageFont.truetype(p, size, index=1 if weight == "bold" else 0)
        except Exception:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def run(cmd: list[str]) -> None:
    print(">>", " ".join(str(c) for c in cmd[:6]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2500:])
        sys.exit(1)


def ctext(d, cx, y, s, fnt, fill):
    """Draw text horizontally centered on cx."""
    w = d.textlength(s, font=fnt)
    d.text((cx - w / 2, y), s, fill=fill, font=fnt)


# ── title card (split) ────────────────────────────────────────────────
def make_title(out: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    gd.ellipse([-150, -150, 760, 620], fill=(255, 107, 53, 50))
    gd.ellipse([W - 760, H - 620, W + 150, H + 150], fill=(74, 158, 255, 50))
    g = g.filter(ImageFilter.GaussianBlur(130))
    img.paste(g, (0, 0), g)
    d = ImageDraw.Draw(img)

    f_eye = font(28, "mono")
    f_big = font(108, "bold")
    f_sub = font(40)
    f_hl = font(46, "bold")

    ctext(d, W / 2, H / 2 - 250, "SAME ROBOT TASK · SAME MODEL · DIFFERENT SILICON",
          f_eye, CYAN)
    ctext(d, W / 2, H / 2 - 190, "THE INFERENCE RACE", f_big, WHITE)

    # vs row
    ctext(d, W / 2 - 360, H / 2 + 10, "CEREBRAS WSE-3", f_sub, CER)
    ctext(d, W / 2, H / 2 + 6, "vs", font(44, "bold"), SUB)
    ctext(d, W / 2 + 360, H / 2 + 10, "OPENROUTER GPU", f_sub, ORN)

    ctext(d, W / 2, H / 2 + 150, "5x faster per decision  ·  39x at the tail (p95)",
          f_hl, CYAN)
    img.save(out)


# ── per-frame race overlay (transparent, 10 fps) ──────────────────────
def make_overlay_frame(idx: int, out: Path) -> None:
    """idx is the overlay-sequence frame index (0-based) over RACEHOLD secs."""
    t = idx / OVR_FPS  # seconds into the race+hold portion

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    f_pane = font(44, "bold")
    f_chip = font(20, "mono")
    f_clock = font(82, "bold")
    f_clk_lbl = font(20, "mono")
    f_dec = font(26, "mono")
    f_badge = font(34, "bold")
    f_disc = font(22)
    f_hl = font(30, "bold")

    # ── vertical divider ──
    d.rectangle([PANE_W - 2, 0, PANE_W + 2, H], fill=DIVIDER)

    # ── top headline strip ──
    hl = "5x FASTER PER DECISION   ·   39x AT THE TAIL (p95)"
    hlw = d.textlength(hl, font=f_hl)
    d.rounded_rectangle([(W - hlw) / 2 - 26, 14, (W + hlw) / 2 + 26, 64],
                        radius=10, fill=(0, 0, 0, 200))
    ctext(d, W / 2, 22, hl, f_hl, CYAN)

    # ── per-pane state ──
    # Cerebras (left, cx=480), OpenRouter (right, cx=1440)
    cer_elapsed = min(t, CER_TOTAL)
    or_elapsed = min(t, OR_TOTAL)
    cer_done = t >= CER_TOTAL
    or_done = t >= OR_TOTAL

    # decision counter: linear over the call-thinking budget; reach N_DEC at done
    cer_dec = N_DEC if cer_done else min(N_DEC, int(cer_elapsed / CER_TOTAL * N_DEC) + 1)
    or_dec = N_DEC if or_done else min(N_DEC, int(or_elapsed / OR_TOTAL * N_DEC) + 1)

    def draw_pane(cx, accent, name, elapsed, total, done, dec, call_ms):
        # pane label chip near top
        ctext(d, cx, 86, name, f_pane, accent)
        chip = f"GEMMA · {call_ms} ms / call"
        cw = d.textlength(chip, font=f_chip)
        d.rounded_rectangle([cx - cw / 2 - 18, 148, cx + cw / 2 + 18, 184],
                            radius=8, fill=accent)
        ctext(d, cx, 154, chip, f_chip, (0, 0, 0))

        # stopwatch
        ctext(d, cx, H - 330, "ELAPSED", f_clk_lbl, SUB)
        clk_col = GREEN if done else (WHITE if accent == CER else AMBER)
        ctext(d, cx, H - 306, f"{elapsed:0.1f}s", f_clock, clk_col)

        # decision counter
        ctext(d, cx, H - 198, f"decision {dec} / {N_DEC}", f_dec, SUB)

        # status badge
        if done:
            txt = f"DONE  {total:0.1f}s"
            bcol = GREEN
            tcol = (0, 0, 0)
        else:
            txt = "THINKING..."
            bcol = AMBER if accent == ORN else (60, 72, 96)
            tcol = (0, 0, 0) if accent == ORN else WHITE
        bw = d.textlength(txt, font=f_badge)
        d.rounded_rectangle([cx - bw / 2 - 26, H - 150, cx + bw / 2 + 26, H - 96],
                            radius=12, fill=bcol)
        ctext(d, cx, H - 144, txt, f_badge, tcol)

    draw_pane(480, CER, "CEREBRAS WSE-3", cer_elapsed, CER_TOTAL,
              cer_done, cer_dec, "200")
    draw_pane(1440, ORN, "OPENROUTER GPU", or_elapsed, OR_TOTAL,
              or_done, or_dec, "1088")

    # ── winner flag once Cerebras done but OR still going ──
    if cer_done and not or_done:
        flag = "WINNER"
        d.rounded_rectangle([480 - 90, 198, 480 + 90, 246], radius=10, fill=GREEN)
        ctext(d, 480, 206, flag, font(26, "bold"), (0, 0, 0))

    # ── bottom disclaimer ──
    disc = ("playback paced to measured per-call latency  "
            "(Cerebras 200ms vs OpenRouter 1088ms, vision)")
    dw = d.textlength(disc, font=f_disc)
    d.rounded_rectangle([(W - dw) / 2 - 22, H - 56, (W + dw) / 2 + 22, H - 18],
                        radius=8, fill=(0, 0, 0, 190))
    ctext(d, W / 2, H - 52, disc, f_disc, SUB)

    img.save(out)


# ── clip builders ─────────────────────────────────────────────────────
def img_clip(img: Path, dur: float, out: Path) -> None:
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{dur}", "-i", str(img),
        "-vf", f"scale={W}:{H},fps={FPS}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def make_pane(src: Path, total_play: float, hold_to: float, out: Path) -> None:
    """Build one 960x1080 pane.

    Plays the Lift motion stretched to `total_play` seconds (setpts), then
    freezes on the last frame out to `hold_to` seconds. Letterboxed onto a
    960x1080 #0a0e17 canvas.
    """
    speed = MOTION / total_play          # <1 means slow down
    setpts = f"setpts={1.0/speed:.4f}*PTS"
    # tpad clones the final frame to extend the (now longer) clip to hold_to.
    run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-filter_complex",
            f"[0:v]{setpts},fps={FPS},"
            f"scale={PANE_W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={PANE_W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,"
            f"tpad=stop_mode=clone:stop_duration={hold_to:.2f}[v]",
        "-map", "[v]", "-t", f"{hold_to}", "-an",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def build_race(left: Path, right: Path, ovr_glob: str, out: Path) -> None:
    """hstack the two panes, then overlay the 10fps PNG sequence."""
    run([
        "ffmpeg", "-y",
        "-i", str(left),
        "-i", str(right),
        "-framerate", str(OVR_FPS), "-i", ovr_glob,
        "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[stack];"
            f"[2:v]fps={FPS},scale={W}:{H}[ov];"
            "[stack][ov]overlay=0:0:format=auto:shortest=1[v]",
        "-map", "[v]", "-an",
        "-t", f"{RACEHOLD}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def main():
    if not LIFT.exists():
        print(f"MISSING: {LIFT}")
        sys.exit(1)

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    ovr_dir = WORK / "ovr"
    ovr_dir.mkdir()

    print(f"Building split race -> {OUT.relative_to(ROOT)}")
    print(f"  Cerebras total {CER_TOTAL}s · OpenRouter total {OR_TOTAL}s · "
          f"clip {TOTAL}s\n")

    # 1) title
    make_title(WORK / "title.png")
    img_clip(WORK / "title.png", TITLE_DUR, WORK / "00_title.mp4")

    # 2) panes (paced to measured latency)
    make_pane(LIFT, CER_TOTAL, RACEHOLD, WORK / "left.mp4")    # Cerebras
    make_pane(LIFT, OR_TOTAL, RACEHOLD, WORK / "right.mp4")    # OpenRouter

    # 3) overlay PNG sequence (10 fps over the 11s race+hold)
    n_ovr = int(round(RACEHOLD * OVR_FPS)) + 1
    print(f"Rendering {n_ovr} overlay frames...")
    for i in range(n_ovr):
        make_overlay_frame(i, ovr_dir / f"f{i:04d}.png")

    # 4) race = stacked panes + overlay
    build_race(WORK / "left.mp4", WORK / "right.mp4",
               str(ovr_dir / "f%04d.png"), WORK / "10_race.mp4")

    # 5) concat title + race
    files = ["00_title.mp4", "10_race.mp4"]
    cl = WORK / "concat.txt"
    cl.write_text("\n".join(f"file '{(WORK / n).resolve()}'" for n in files))
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(cl),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(OUT),
    ])

    mb = OUT.stat().st_size / 1_048_576
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(OUT)],
        capture_output=True, text=True)
    print(f"\nDONE: {OUT}  ({mb:.1f} MB)")
    print(f"      ffprobe duration: {r.stdout.strip()}s  (target ~{TOTAL})")


if __name__ == "__main__":
    main()
