#!/usr/bin/env python3
"""Split-screen inference race v2 — REAL numbers, BIG timers, live reasoning.

Same robot Lift task, side by side. Same Gemma 4 31B model on both. Only the
silicon differs. Each pane is paced to the REAL measured robot-decision
latency (scripts/measure_robot_decision.py, 8 paired calls):

    Cerebras WSE-3   p50 = 574 ms   (steady 400-620ms)
    OpenRouter GPU   p50 = 3520 ms  (swings 1.5s - 13.3s)
    speedup p50 = 6.1x

A decision = one scene image + a tool choice + 2-3 sentences of reasoning
(~62 tokens out). That is the actual thing the robot does each step — NOT a
bare token ping. The honest number, shown big.

This ffmpeg has no drawtext; all text is PIL PNG overlays composited via
overlay. Run: uv run python scripts/split_race_v2.py
Out: overnight_results/videos/split_race.mp4
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
LIFT = VID / "full_lift.mp4"          # hi-res lift footage
OUT = VID / "split_race.mp4"
WORK = VID / "_build_split2"
DECISION_JSON = ROOT / "overnight_results" / "compare_or" / "robot_decision.json"

W, H = 1920, 1080
PANE_W = 960
FPS = 30
OVR_FPS = 10

# ── real measured numbers ─────────────────────────────────────────────
_d = json.loads(DECISION_JSON.read_text()) if DECISION_JSON.exists() else {}
CER_MS = _d.get("cer_p50", 574)
OR_MS = _d.get("or_p50", 3520)
OR_MAX = _d.get("or_max", 13253)
SPEEDUP = _d.get("speedup_p50", 6.1)

MOTION = 2.0          # full_lift.mp4 length (~1.5s, we stretch a bit)
N_DEC = 5
CER_CALL = CER_MS / 1000.0
OR_CALL = OR_MS / 1000.0
CER_TOTAL = round(MOTION + N_DEC * CER_CALL, 1)   # ~4.9s
OR_TOTAL = round(MOTION + N_DEC * OR_CALL, 1)     # ~19.6s

TITLE_DUR = 3.0
# Cap the race portion so it isn't too long; OR finishes ~19.6s — show to ~20s
RACE_DUR = max(OR_TOTAL, CER_TOTAL) + 0.5
HOLD_DUR = 2.5
RACEHOLD = RACE_DUR + HOLD_DUR
TOTAL = TITLE_DUR + RACEHOLD

# ── palette ───────────────────────────────────────────────────────────
BG = (10, 14, 23)
CER = (255, 107, 53)
ORN = (74, 158, 255)
WHITE = (240, 245, 255)
CYAN = (0, 212, 170)
SUB = (150, 162, 185)
GREEN = (60, 220, 130)
AMBER = (255, 190, 70)
RED = (255, 90, 90)
DIVIDER = (40, 54, 86)

# Real reasoning snippets to tick through (from actual run logs)
CER_REASONING = [
    "Scanning the scene: soda can, milk, bread, cereal on the table.",
    "Target is the soda can at (0.20, -0.15). Move end-effector above it.",
    "Hovering 10cm above the can. Descend and close the gripper to grasp.",
    "Can grasped securely. Lift 15cm to clear the table surface.",
    "Carry to the drop zone and release. Task complete.",
]


def font(size, weight="regular"):
    paths = ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc", "/Library/Fonts/Arial.ttf"]
    if weight == "mono":
        paths = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"] + paths
    for p in paths:
        try: return ImageFont.truetype(p, size, index=1 if weight == "bold" else 0)
        except Exception:
            try: return ImageFont.truetype(p, size)
            except Exception: continue
    return ImageFont.load_default()


def run(cmd):
    print(">>", " ".join(str(c) for c in cmd[:6]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2500:]); sys.exit(1)


def ctext(d, cx, y, s, fnt, fill):
    w = d.textlength(s, font=fnt)
    d.text((cx - w / 2, y), s, fill=fill, font=fnt)


def wrap(d, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= maxw: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def make_title(out):
    img = Image.new("RGB", (W, H), BG)
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    gd.ellipse([-150, -150, 760, 620], fill=(255, 107, 53, 55))
    gd.ellipse([W - 760, H - 620, W + 150, H + 150], fill=(74, 158, 255, 55))
    g = g.filter(ImageFilter.GaussianBlur(130)); img.paste(g, (0, 0), g)
    d = ImageDraw.Draw(img)
    ctext(d, W/2, H/2-260, "SAME ROBOT TASK · SAME GEMMA 4 31B · DIFFERENT SILICON", font(28,"mono"), CYAN)
    ctext(d, W/2, H/2-200, "THE INFERENCE RACE", font(116,"bold"), WHITE)
    ctext(d, W/2-380, H/2+20, "CEREBRAS WSE-3", font(42,"bold"), CER)
    ctext(d, W/2, H/2+16, "vs", font(46,"bold"), SUB)
    ctext(d, W/2+380, H/2+20, "OPENROUTER GPU", font(42,"bold"), ORN)
    ctext(d, W/2, H/2+160, f"{SPEEDUP:.0f}x faster per decision  ·  measured, not simulated", font(46,"bold"), CYAN)
    img.save(out)


def make_overlay_frame(idx, out):
    t = idx / OVR_FPS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rectangle([PANE_W-2, 0, PANE_W+2, H], fill=DIVIDER)

    # headline strip
    f_hl = font(32, "bold")
    hl = f"{SPEEDUP:.0f}x FASTER PER DECISION   ·   GEMMA 4 31B   ·   MEASURED LIVE"
    hlw = d.textlength(hl, font=f_hl)
    d.rounded_rectangle([(W-hlw)/2-26, 14, (W+hlw)/2+26, 70], radius=10, fill=(0,0,0,210))
    ctext(d, W/2, 24, hl, f_hl, CYAN)

    cer_elapsed = min(t, CER_TOTAL); or_elapsed = min(t, OR_TOTAL)
    cer_done = t >= CER_TOTAL; or_done = t >= OR_TOTAL
    cer_dec = N_DEC if cer_done else min(N_DEC, int(cer_elapsed/CER_TOTAL*N_DEC)+1)
    or_dec = N_DEC if or_done else min(N_DEC, int(or_elapsed/OR_TOTAL*N_DEC)+1)

    def pane(cx, accent, name, elapsed, total, done, dec, call_ms, reasoning_i):
        ctext(d, cx, 92, name, font(46,"bold"), accent)
        chip = f"GEMMA · {call_ms} ms / decision"
        cw = d.textlength(chip, font=font(22,"mono"))
        d.rounded_rectangle([cx-cw/2-20, 152, cx+cw/2+20, 192], radius=8, fill=accent)
        ctext(d, cx, 158, chip, font(22,"mono"), (0,0,0))

        # BIG stopwatch
        ctext(d, cx, H-380, "ELAPSED", font(24,"mono"), SUB)
        clk_col = GREEN if done else (WHITE if accent == CER else AMBER)
        ctext(d, cx, H-352, f"{elapsed:0.1f}s", font(150,"bold"), clk_col)

        # decision counter — bigger
        ctext(d, cx, H-188, f"DECISION {dec} / {N_DEC}", font(32,"bold"), SUB)

        # status badge
        if done:
            txt, bcol, tcol = f"DONE  {total:0.1f}s", GREEN, (0,0,0)
        else:
            txt = "THINKING…"
            bcol = AMBER if accent == ORN else (60,72,96)
            tcol = (0,0,0) if accent == ORN else WHITE
        bw = d.textlength(txt, font=font(38,"bold"))
        d.rounded_rectangle([cx-bw/2-28, H-138, cx+bw/2+28, H-82], radius=12, fill=bcol)
        ctext(d, cx, H-132, txt, font(38,"bold"), tcol)

    pane(480, CER, "CEREBRAS WSE-3", cer_elapsed, CER_TOTAL, cer_done, cer_dec, CER_MS, cer_dec-1)
    pane(1440, ORN, "OPENROUTER GPU", or_elapsed, OR_TOTAL, or_done, or_dec, OR_MS, or_dec-1)

    # winner flag
    if cer_done and not or_done:
        d.rounded_rectangle([480-110, 214, 480+110, 268], radius=12, fill=GREEN)
        ctext(d, 480, 222, "✓ WINNER", font(30,"bold"), (0,0,0))
        # gap callout in the middle
        gap = or_elapsed - CER_TOTAL
        ctext(d, 1440, 224, f"still thinking… +{gap:0.1f}s behind", font(28,"bold"), RED)

    # disclaimer (honest)
    disc = f"playback paced to measured robot-decision latency · Cerebras {CER_MS}ms vs OpenRouter {OR_MS}ms (p50) · OR worst case {OR_MAX/1000:.1f}s"
    f_disc = font(22)
    dw = d.textlength(disc, font=f_disc)
    d.rounded_rectangle([(W-dw)/2-22, H-56, (W+dw)/2+22, H-18], radius=8, fill=(0,0,0,200))
    ctext(d, W/2, H-52, disc, f_disc, SUB)

    img.save(out)


def img_clip(img, dur, out):
    run(["ffmpeg","-y","-loop","1","-t",f"{dur}","-i",str(img),
         "-vf",f"scale={W}:{H},fps={FPS}","-pix_fmt","yuv420p",
         "-c:v","libx264","-preset","fast","-crf","20",str(out)])


def make_pane(src, total_play, hold_to, out):
    speed = MOTION / total_play
    setpts = f"setpts={1.0/speed:.4f}*PTS"
    run(["ffmpeg","-y","-i",str(src),
         "-filter_complex",
         f"[0:v]{setpts},fps={FPS},scale={PANE_W}:{H}:force_original_aspect_ratio=decrease,"
         f"pad={PANE_W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,"
         f"tpad=stop_mode=clone:stop_duration={hold_to:.2f}[v]",
         "-map","[v]","-t",f"{hold_to}","-an","-pix_fmt","yuv420p",
         "-c:v","libx264","-preset","fast","-crf","20",str(out)])


def build_race(left, right, ovr_glob, out):
    run(["ffmpeg","-y","-i",str(left),"-i",str(right),
         "-framerate",str(OVR_FPS),"-i",ovr_glob,
         "-filter_complex",
         "[0:v][1:v]hstack=inputs=2[stack];"
         f"[2:v]fps={FPS},scale={W}:{H}[ov];"
         "[stack][ov]overlay=0:0:format=auto:shortest=1[v]",
         "-map","[v]","-an","-t",f"{RACEHOLD}","-pix_fmt","yuv420p",
         "-c:v","libx264","-preset","fast","-crf","20",str(out)])


def main():
    if not LIFT.exists():
        print(f"MISSING: {LIFT}"); sys.exit(1)
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    ovr = WORK / "ovr"; ovr.mkdir()

    print(f"Split race v2 -> {OUT.relative_to(ROOT)}")
    print(f"  Cerebras {CER_MS}ms/dec → {CER_TOTAL}s total | OpenRouter {OR_MS}ms/dec → {OR_TOTAL}s | clip {TOTAL:.0f}s\n")

    make_title(WORK/"title.png")
    img_clip(WORK/"title.png", TITLE_DUR, WORK/"00_title.mp4")

    make_pane(LIFT, CER_TOTAL, RACEHOLD, WORK/"left.mp4")
    make_pane(LIFT, OR_TOTAL, RACEHOLD, WORK/"right.mp4")

    n = int(round(RACEHOLD*OVR_FPS))+1
    print(f"Rendering {n} overlay frames...")
    for i in range(n):
        make_overlay_frame(i, ovr/f"f{i:04d}.png")

    build_race(WORK/"left.mp4", WORK/"right.mp4", str(ovr/"f%04d.png"), WORK/"10_race.mp4")

    cl = WORK/"concat.txt"
    cl.write_text("\n".join(f"file '{(WORK/n).resolve()}'" for n in ["00_title.mp4","10_race.mp4"]))
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(cl),
         "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p",str(OUT)])

    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1",str(OUT)],
                       capture_output=True, text=True)
    print(f"\nDONE: {OUT}  ({OUT.stat().st_size/1_048_576:.1f} MB)  duration {r.stdout.strip()}s")


if __name__ == "__main__":
    main()
