#!/usr/bin/env python3
"""Reasoning-overlay generator — "what Gemma is thinking" split-screen clip.

LEFT half : the robot Lift video (looped to fill the duration), centered in a
            rounded dark card.
RIGHT half: a "GEMMA 4 · thinking" panel that types out the model's *actual*
            reasoning text word-by-word (typewriter effect), one reasoning step
            at a time, synced so each step finishes before the next appears.

Data comes from runs/lift_cube_20260629T041700Z/step_*.json — each step's
intent.reasoning is the text we reveal, along with its stage / latency / target.

This ffmpeg build has NO drawtext (no libfreetype), so ALL text is rendered to
PNG with PIL and composited via the overlay filter.

Typewriter implementation: we render the right-pane overlay as a numbered PNG
sequence at OVL_FPS (10 fps). For each overlay frame we compute how many words
of the current step should be visible and draw exactly that many. ffmpeg reads
the sequence as an image input and overlays it on the looped robot video.

Run:  uv run python scripts/reasoning_overlay.py
Out:  overnight_results/videos/reasoning_panel.mp4  (1920x1080, ~18s, h264)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "lift_cube_20260629T041700Z"
VID = ROOT / "overnight_results" / "videos"
ROBOT = VID / "nocheat_Lift.mp4"
OUT = VID / "reasoning_panel.mp4"
WORK = VID / "_reasoning_build"

W, H = 1920, 1080
FPS = 30
OVL_FPS = 10            # cadence of the typewriter overlay sequence
DURATION = 18.0        # total clip length (seconds)

# Dark theme palette
BG = (10, 14, 23)        # #0a0e17
PANEL = (19, 26, 43)     # #131a2b
BORDER = (30, 42, 69)    # #1e2a45
ACCENT = (255, 107, 53)  # #FF6B35 orange
CYAN = (0, 212, 170)     # #00d4aa
WHITE = (240, 245, 255)  # #f0f5ff
SUB = (170, 180, 200)    # #aab4c8

# Right panel geometry (left half = robot, right half = panel)
PANEL_X = W // 2 + 40
PANEL_Y = 80
PANEL_W = W - PANEL_X - 60
PANEL_H = H - 2 * PANEL_Y


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    paths = ["/System/Library/Fonts/Helvetica.ttc",
             "/System/Library/Fonts/HelveticaNeue.ttc",
             "/System/Library/Fonts/SFNS.ttf",
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
        print(r.stderr[-2000:])
        sys.exit(1)


def load_steps() -> list[dict]:
    steps = []
    for p in sorted(RUN.glob("step_*.json")):
        data = json.loads(p.read_text())
        intent = data.get("intent", {})
        steps.append({
            "step": data.get("step", 0),
            "stage": str(intent.get("stage", "")).upper(),
            "latency_ms": int(data.get("latency_ms", 0)),
            "target": intent.get("target", []),
            "gripper": intent.get("gripper", ""),
            "reasoning": str(intent.get("reasoning", "")).strip(),
        })
    return steps


def wrap_words(draw: ImageDraw.ImageDraw, words: list[str],
               fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """Wrap a list of words into lines that fit within max_w pixels."""
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and draw.textlength(trial, font=fnt) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_panel_base(d: ImageDraw.ImageDraw) -> None:
    """Right-side panel card + static header (title + meta placeholder)."""
    d.rounded_rectangle(
        [PANEL_X, PANEL_Y, PANEL_X + PANEL_W, PANEL_Y + PANEL_H],
        radius=24, fill=PANEL + (255,), outline=BORDER + (255,), width=2)


def render_overlay_frame(out: Path, step: dict, words_visible: int,
                         cursor_on: bool) -> None:
    """Render one full 1920x1080 RGBA overlay: left transparent (robot shows
    through), right panel drawn with `words_visible` words of the reasoning."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    draw_panel_base(d)

    pad = 36
    cx = PANEL_X + pad
    cy = PANEL_Y + pad
    inner_w = PANEL_W - 2 * pad

    # Title: "GEMMA 4 · thinking"
    f_title = font(40, "bold")
    f_dot = font(40, "bold")
    title_a = "GEMMA 4 "
    title_b = "· thinking"
    d.text((cx, cy), title_a, fill=WHITE, font=f_title)
    aw = d.textlength(title_a, font=f_title)
    d.text((cx + aw, cy), title_b, fill=ACCENT, font=f_dot)
    cy += 64

    # Accent rule under title
    d.rectangle([cx, cy, cx + inner_w, cy + 3], fill=BORDER + (255,))
    cy += 24

    # Meta mono line: "stage: APPROACH   ·   817ms   ·   Cerebras WSE-3"
    f_meta = font(20, "mono")
    stage = step["stage"] or "—"
    lat = f"{step['latency_ms']}ms"
    meta = f"stage: {stage}   ·   {lat}   ·   Cerebras WSE-3"
    d.text((cx, cy), meta, fill=CYAN, font=f_meta)
    cy += 34

    # Target coords line (mono, sub color)
    tgt = step.get("target") or []
    if isinstance(tgt, (list, tuple)) and len(tgt) == 3:
        tgt_txt = f"target: ({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f})"
    else:
        tgt_txt = "target: —"
    grip = step.get("gripper", "")
    if grip:
        tgt_txt += f"   ·   gripper: {grip}"
    f_tgt = font(18, "mono")
    d.text((cx, cy), tgt_txt, fill=SUB, font=f_tgt)
    cy += 40

    # Step badge ("STEP 1 / 5") top-right of panel
    f_badge = font(18, "mono")
    badge = f"STEP {step['step'] + 1} / {step['total']}"
    bw = d.textlength(badge, font=f_badge)
    d.text((PANEL_X + PANEL_W - pad - bw, PANEL_Y + pad + 8),
           badge, fill=ACCENT, font=f_badge)

    # Reasoning body — typewriter
    f_body = font(28)
    words = step["reasoning"].split()
    shown = words[:max(0, words_visible)]
    text = " ".join(shown)
    if cursor_on and words_visible < len(words):
        text = (text + " ▌").strip() if text else "▌"
    lines = wrap_words(d, text.split(" ") if text else [], f_body, inner_w)
    line_h = 40
    for ln in lines:
        d.text((cx, cy), ln, fill=WHITE, font=f_body)
        cy += line_h

    img.save(out)


def build_robot_layer(out: Path) -> None:
    """Looped robot video scaled to ~900px and placed in a rounded card on the
    LEFT half, on a full dark background. This is the base layer."""
    # Left-half center
    card_size = 940
    card_x = (W // 2 - card_size) // 2
    card_y = (H - card_size) // 2
    vid_size = 880
    vid_x = card_x + (card_size - vid_size) // 2
    vid_y = card_y + (card_size - vid_size) // 2

    run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(ROBOT),
        "-f", "lavfi", "-t", str(DURATION),
        "-i", f"color=c=0x0a0e17:s={W}x{H}:r={FPS}",
        "-f", "lavfi", "-t", str(DURATION),
        "-i", f"color=c=0x131a2b:s={card_size}x{card_size}:r={FPS}",
        "-t", str(DURATION),
        "-filter_complex",
            f"[0:v]scale={vid_size}:{vid_size}:force_original_aspect_ratio=increase,"
            f"crop={vid_size}:{vid_size},fps={FPS}[vid];"
            f"[1:v][2:v]overlay={card_x}:{card_y}[withcard];"
            f"[withcard][vid]overlay={vid_x}:{vid_y}:format=auto[v]",
        "-map", "[v]", "-an", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out),
    ])


def main():
    if not ROBOT.exists():
        print(f"MISSING robot video: {ROBOT}")
        sys.exit(1)
    if not RUN.exists():
        print(f"MISSING run dir: {RUN}")
        sys.exit(1)

    steps = load_steps()
    if not steps:
        print("No step_*.json found")
        sys.exit(1)
    for s in steps:
        s["total"] = len(steps)
    print(f"Loaded {len(steps)} reasoning steps")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    # ── 1) Build the typewriter overlay PNG sequence ──────────────────
    total_frames = int(round(DURATION * OVL_FPS))
    sec_per_step = DURATION / len(steps)
    frames_per_step = total_frames / len(steps)

    # Reveal all words within the first ~75% of each step's window, then hold
    # the full text for the remainder so the viewer can read it.
    REVEAL_FRAC = 0.78

    print(f"Rendering {total_frames} overlay frames at {OVL_FPS}fps "
          f"({sec_per_step:.2f}s/step)...")

    for fi in range(total_frames):
        t = fi / OVL_FPS
        step_idx = min(int(t / sec_per_step), len(steps) - 1)
        step = steps[step_idx]
        # progress within this step (0..1)
        local = (t - step_idx * sec_per_step) / sec_per_step
        reveal_p = min(1.0, local / REVEAL_FRAC) if REVEAL_FRAC > 0 else 1.0
        n_words = len(step["reasoning"].split())
        words_visible = int(round(n_words * reveal_p))
        words_visible = max(0, min(words_visible, n_words))
        # blinking cursor while still typing
        cursor_on = (fi % 2 == 0)
        render_overlay_frame(WORK / f"ovl_{fi:05d}.png", step,
                             words_visible, cursor_on)

    # ── 2) Build the robot base layer ─────────────────────────────────
    print("Building robot base layer...")
    build_robot_layer(WORK / "robot.mp4")

    # ── 3) Composite overlay sequence onto robot layer ────────────────
    print("Compositing typewriter overlay onto robot layer...")
    run([
        "ffmpeg", "-y",
        "-i", str(WORK / "robot.mp4"),
        "-framerate", str(OVL_FPS),
        "-i", str(WORK / "ovl_%05d.png"),
        "-t", str(DURATION),
        "-filter_complex",
            f"[1:v]fps={FPS}[ovl];"
            f"[0:v][ovl]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-an", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(OUT),
    ])

    # ── 4) Report ─────────────────────────────────────────────────────
    mb = OUT.stat().st_size / 1_048_576
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(OUT)],
        capture_output=True, text=True)
    dur = r.stdout.strip()
    print("\nDONE")
    print(f"  output:   {OUT}")
    print(f"  size:     {mb:.1f} MB")
    print(f"  duration: {dur}s  (target {DURATION})")


if __name__ == "__main__":
    main()
