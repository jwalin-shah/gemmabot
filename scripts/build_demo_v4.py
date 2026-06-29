#!/usr/bin/env python3
"""GemmaBot 60s demo v4 — the real reel.

Assembles the full-task footage + reasoning panel + speed race + chart into
one punchy 60s cut. All segment clips are normalized to 1920x1080@30fps with
PNG overlays composited (this ffmpeg has no drawtext).

Segment plan (auto-rebalances if an optional asset is missing):
  title        2.5s
  full_lift    5.0s   (slowed for impact) + HUD
  full_stack   5.0s   + HUD
  full_pick    6.0s   + HUD
  reasoning    11.0s  (trim of reasoning_panel.mp4)
  split_race   10.0s  (trim of split_race.mp4, optional)
  chart        12.0s  + big callouts
  end          2.5s
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "overnight_results" / "videos"
CHART = ROOT / "overnight_results" / "compare_or" / "hero_chart.png"
OUT = VID / "gemmabot_demo_60s.mp4"
WORK = VID / "_build4"

W, H, FPS = 1920, 1080, 30
BG = (10, 14, 23); PANEL = (19, 26, 43); BORDER = (30, 42, 69)
ACCENT = (255, 107, 53); ACCENT2 = (0, 212, 170)
WHITE = (240, 245, 255); SUB = (170, 180, 200)


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


def run(cmd):
    print(">>", " ".join(str(c) for c in cmd[:5]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit(1)


def dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except Exception: return 0.0


# ── PNG factories ─────────────────────────────────────────────────────
def glow(img, spots):
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    for (x0, y0, x1, y1, col) in spots: gd.ellipse([x0, y0, x1, y1], fill=col)
    g = g.filter(ImageFilter.GaussianBlur(120)); img.paste(g, (0, 0), g)


def title_png(out):
    img = Image.new("RGB", (W, H), BG)
    glow(img, [(-200,-200,800,600,(255,107,53,55)), (W-600,H-400,W+200,H+200,(0,212,170,40))])
    d = ImageDraw.Draw(img)
    eye, mark, tag = "PERCEPTION → REASONING → MOTION", "GemmaBot", "Gemma 4 31B  ·  Cerebras WSE-3"
    fe, fm, ft = font(26,"mono"), font(210,"bold"), font(48)
    d.text(((W-d.textlength(eye,font=fe))/2, H/2-230), eye, fill=ACCENT2, font=fe)
    d.text(((W-d.textlength(mark,font=fm))/2, H/2-170), mark, fill=WHITE, font=fm)
    d.text(((W-d.textlength(tag,font=ft))/2, H/2+90), tag, fill=ACCENT, font=ft)
    img.save(out)


def hud_png(out, step_label, task_name, latency, integrity=None):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    # top-left task chip
    fl, fv = font(20,"mono"), font(40,"bold")
    chip_w = int(max(d.textlength(step_label,font=fl), d.textlength(task_name,font=fv))+60)
    d.rounded_rectangle([50,50,50+chip_w,170], radius=12, fill=(0,0,0,205), outline=(60,72,96,255), width=2)
    d.text((80,64), step_label, fill=ACCENT2, font=fl)
    d.text((80,92), task_name, fill=WHITE, font=fv)
    # top-right latency chip
    fb = font(34,"bold"); lat = f"{latency}"
    lw = int(d.textlength(lat,font=fb))+50; bx = W-50-lw
    d.rounded_rectangle([bx,50,bx+lw,110], radius=10, fill=ACCENT)
    d.text((bx+25,60), lat, fill=(0,0,0), font=fb)
    fcap = font(18,"mono"); cap="GEMMA DECISION · MEASURED"
    d.text((W-50-int(d.textlength(cap,font=fcap)),120), cap, fill=SUB, font=fcap)
    if integrity:
        fs, fe = font(40,"bold"), font(18,"mono")
        sw = d.textlength(integrity, font=fs)
        bw = sw+100; bx2=(W-bw)/2; by2=H-200
        d.rounded_rectangle([bx2,by2,bx2+bw,by2+110], radius=14, fill=(0,0,0,210))
        d.text((bx2+50,by2+18), "INTEGRITY", fill=ACCENT2, font=fe)
        d.text(((W-sw)/2,by2+48), integrity, fill=WHITE, font=fs)
    img.save(out)


def chart_overlay_png(out):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    fe = font(28,"mono"); eyebrow="SAME MODEL.   SAME PROMPTS.   ONLY THE SILICON CHANGES."
    ew=d.textlength(eyebrow,font=fe)
    d.rounded_rectangle([(W-ew)/2-30,30,(W+ew)/2+30,90], radius=8, fill=(0,0,0,210))
    d.text(((W-ew)/2,44), eyebrow, fill=ACCENT2, font=fe)
    fbig, fcap = font(88,"bold"), font(22,"mono")
    badges=[("13×","TEXT p50"),("5×","VISION p50"),("39×","VISION p95 tail")]
    bw=360; gap=60; tot=bw*3+gap*2; sx=(W-tot)/2; yt=H-260
    for i,(num,cap) in enumerate(badges):
        x0=sx+i*(bw+gap)
        d.rounded_rectangle([x0,yt,x0+bw,yt+190], radius=18, fill=ACCENT)
        d.text((x0+(bw-d.textlength(num,font=fbig))/2, yt+18), num, fill=(0,0,0), font=fbig)
        d.text((x0+(bw-d.textlength(cap,font=fcap))/2, yt+140), cap, fill=(0,0,0), font=fcap)
    img.save(out)


def stat_png(out, big, caption, eyebrow):
    img = Image.new("RGB", (W, H), BG)
    glow(img, [(W/2-500,H/2-500,W/2+500,H/2+500,(255,107,53,35))])
    d = ImageDraw.Draw(img)
    fe,fb,fc = font(28,"mono"), font(360,"bold"), font(46)
    d.text(((W-d.textlength(eyebrow,font=fe))/2, H/2-270), eyebrow, fill=ACCENT2, font=fe)
    d.text(((W-d.textlength(big,font=fb))/2, H/2-210), big, fill=WHITE, font=fb)
    d.text(((W-d.textlength(caption,font=fc))/2, H/2+190), caption, fill=ACCENT, font=fc)
    img.save(out)


def end_png(out):
    img = Image.new("RGB", (W, H), BG)
    glow(img, [(-200,-200,800,600,(0,212,170,55)), (W-600,H-400,W+200,H+200,(255,107,53,40))])
    d = ImageDraw.Draw(img)
    fe,fb,ft = font(24,"mono"), font(160,"bold"), font(44)
    eye="CEREBRAS × GEMMA 4 HACKATHON"
    d.text(((W-d.textlength(eye,font=fe))/2, H/2-230), eye, fill=ACCENT2, font=fe)
    d.text(((W-d.textlength("GemmaBot",font=fb))/2, H/2-170), "GemmaBot", fill=WHITE, font=fb)
    tag="honest pipeline · live numbers · code in the repo"
    d.text(((W-d.textlength(tag,font=ft))/2, H/2+70), tag, fill=ACCENT, font=ft)
    img.save(out)


def stat_png(out, big, caption, eyebrow):
    img = Image.new("RGB", (W, H), BG)
    glow(img, [(W/2-500,H/2-500,W/2+500,H/2+500,(255,107,53,32))])
    d = ImageDraw.Draw(img)
    fe,fb,fc = font(28,"mono"), font(340,"bold"), font(46)
    d.text(((W-d.textlength(eyebrow,font=fe))/2, H/2-260), eyebrow, fill=ACCENT2, font=fe)
    d.text(((W-d.textlength(big,font=fb))/2, H/2-205), big, fill=WHITE, font=fb)
    d.text(((W-d.textlength(caption,font=fc))/2, H/2+185), caption, fill=ACCENT, font=fc)
    img.save(out)


# ── clip builders ─────────────────────────────────────────────────────
def img_clip(img, d, out, overlay=None):
    if overlay:
        run(["ffmpeg","-y","-loop","1","-t",str(d),"-i",str(img),
             "-loop","1","-t",str(d),"-i",str(overlay),
             "-filter_complex",
             f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];[bg][1:v]overlay=0:0:format=auto[v]",
             "-map","[v]","-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])
    else:
        run(["ffmpeg","-y","-loop","1","-t",str(d),"-i",str(img),
             "-vf",f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}",
             "-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])


def video_clip(src, d, out, overlay=None, speed=1.0):
    """Loop+trim src to d seconds at playback `speed`, optional PNG overlay."""
    setpts=f"setpts={1.0/speed:.3f}*PTS,"
    if overlay:
        run(["ffmpeg","-y","-stream_loop","-1","-i",str(src),
             "-loop","1","-t",str(d),"-i",str(overlay),"-t",str(d),
             "-filter_complex",
             f"[0:v]{setpts}scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];[bg][1:v]overlay=0:0:format=auto[v]",
             "-map","[v]","-an","-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])
    else:
        run(["ffmpeg","-y","-stream_loop","-1","-i",str(src),"-t",str(d),
             "-vf",f"{setpts}scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}",
             "-an","-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])


def main():
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    print(f"Building v4 -> {OUT.relative_to(ROOT)}\n")

    has_split = (VID / "split_race.mp4").exists()
    has_reason = (VID / "reasoning_panel.mp4").exists()
    print(f"  reasoning_panel: {'YES' if has_reason else 'no'} | split_race: {'YES' if has_split else 'no'}")

    seg = []  # (name, mp4path)

    # title
    title_png(WORK/"title.png"); img_clip(WORK/"title.png", 2.5, WORK/"00.mp4"); seg.append(WORK/"00.mp4")

    # full tasks — slow short clips so motion reads clearly
    lift_d = dur(VID/"full_lift.mp4"); stack_d = dur(VID/"full_stack.mp4")
    hud_png(WORK/"h_lift.png","TASK · LIVE LOOP","Lift the red cube","~560 ms")
    video_clip(VID/"full_lift.mp4", 5.0, WORK/"10.mp4", overlay=WORK/"h_lift.png",
               speed=max(lift_d/5.0, 0.25))
    seg.append(WORK/"10.mp4")
    hud_png(WORK/"h_stack.png","TASK · LIVE LOOP","Stack red on green","~560 ms")
    video_clip(VID/"full_stack.mp4", 5.0, WORK/"11.mp4", overlay=WORK/"h_stack.png",
               speed=max(stack_d/5.0, 0.25))
    seg.append(WORK/"11.mp4")
    hud_png(WORK/"h_pick.png","TASK · LIVE LOOP","Pick & place the can","~560 ms",
            integrity="perception + Gemma + grasp + place")
    video_clip(VID/"full_pick.mp4", 6.0, WORK/"12.mp4", overlay=WORK/"h_pick.png",
               speed=max(dur(VID/'full_pick.mp4')/6.0, 0.5))
    seg.append(WORK/"12.mp4")

    # reasoning panel
    if has_reason:
        video_clip(VID/"reasoning_panel.mp4", 12.0, WORK/"20.mp4", speed=1.0)
        seg.append(WORK/"20.mp4")

    # split race (optional — slots in if the agent produced it)
    if has_split:
        video_clip(VID/"split_race.mp4", 10.0, WORK/"30.mp4", speed=1.0)
        seg.append(WORK/"30.mp4")

    # chart — shown PLAIN (speedups are built into the hero chart, no overlay)
    chart_d = 9.0 if has_split else 11.0
    img_clip(CHART, chart_d, WORK/"40.mp4")
    seg.append(WORK/"40.mp4")

    # stat slates — punchy summary hammer (skip 1 if split race ate the time)
    slates = [
        ("~560 ms", "per decision · see → reason → act", "MEASURED ROBOT-LOOP LATENCY · 146 STEPS"),
        ("1 cm",    "perception localization error", "HOW WELL WE FIND OBJECTS"),
        ("88 %",    "visual reasoning · un-gameable test", "HOW WELL GEMMA SEES"),
        ("0",       "ground-truth coords fed to perception", "THE NO-CHEAT PROOF"),
    ]
    if has_split:
        slates = slates[:3]
    for i, (big, cap, eye) in enumerate(slates):
        stat_png(WORK/f"s{i}.png", big, cap, eye)
        img_clip(WORK/f"s{i}.png", 2.6, WORK/f"5{i}.mp4")
        seg.append(WORK/f"5{i}.mp4")

    # end
    end_png(WORK/"end.png"); img_clip(WORK/"end.png", 3.0, WORK/"90.mp4"); seg.append(WORK/"90.mp4")

    cl = WORK/"concat.txt"
    cl.write_text("\n".join(f"file '{p.resolve()}'" for p in seg))
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(cl),
         "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p",str(OUT)])
    print(f"\nDONE: {OUT}  ({OUT.stat().st_size/1_048_576:.1f} MB)  duration {dur(OUT):.1f}s")


if __name__ == "__main__":
    main()
