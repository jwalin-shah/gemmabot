#!/usr/bin/env python3
"""FINAL 60s cut. Hero finale = the measured inference race.

  title         2.5s
  lift          4.0s   + HUD
  stack         4.0s   + HUD
  pick          6.0s   + HUD
  reasoning    13.0s   (trim of reasoning_panel.mp4 — shows Gemma thinking)
  split_race   25.6s   (the measured Cerebras-vs-OpenRouter race, full)
  end           3.0s
  ----------------------------------------
  ~58s
"""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "overnight_results" / "videos"
OUT = VID / "gemmabot_final_60s.mp4"
WORK = VID / "_final"
W, H, FPS = 1920, 1080, 30
BG=(10,14,23); ACCENT=(255,107,53); ACCENT2=(0,212,170); WHITE=(240,245,255); SUB=(150,162,186)


def font(size, weight="regular"):
    paths=["/System/Library/Fonts/Helvetica.ttc","/System/Library/Fonts/HelveticaNeue.ttc"]
    if weight=="mono": paths=["/System/Library/Fonts/Menlo.ttc"]+paths
    for p in paths:
        try: return ImageFont.truetype(p,size,index=1 if weight=="bold" else 0)
        except Exception:
            try: return ImageFont.truetype(p,size)
            except Exception: continue
    return ImageFont.load_default()

def run(cmd):
    print(">>"," ".join(str(c) for c in cmd[:5]),"...")
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode!=0: print(r.stderr[-2000:]); sys.exit(1)

def dur(p):
    r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                      "-of","default=noprint_wrappers=1:nokey=1",str(p)],capture_output=True,text=True)
    try: return float(r.stdout.strip())
    except Exception: return 0.0

def glow(img,spots):
    g=Image.new("RGBA",(W,H),(0,0,0,0)); gd=ImageDraw.Draw(g)
    for s in spots: gd.ellipse(s[:4],fill=s[4])
    img.paste(g.filter(ImageFilter.GaussianBlur(120)),(0,0),g.filter(ImageFilter.GaussianBlur(120)))

def ctext(d,cx,y,s,f,fill):
    d.text((cx-d.textlength(s,font=f)/2,y),s,fill=fill,font=f)

def title_png(out):
    img=Image.new("RGB",(W,H),BG)
    glow(img,[(-200,-200,800,600,(255,107,53,55)),(W-600,H-400,W+200,H+200,(0,212,170,40))])
    d=ImageDraw.Draw(img)
    ctext(d,W/2,H/2-230,"PERCEPTION → REASONING → MOTION",font(26,"mono"),ACCENT2)
    ctext(d,W/2,H/2-170,"GemmaBot",font(210,"bold"),WHITE)
    ctext(d,W/2,H/2+90,"Gemma 4 31B  ·  Cerebras WSE-3",font(48),ACCENT)
    img.save(out)

def hud_png(out,task,latency,integrity=None):
    img=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(img)
    fl,fv=font(20,"mono"),font(40,"bold")
    cw=int(max(d.textlength("TASK · LIVE LOOP",font=fl),d.textlength(task,font=fv))+60)
    d.rounded_rectangle([50,50,50+cw,170],radius=12,fill=(0,0,0,205),outline=(60,72,96,255),width=2)
    d.text((80,64),"TASK · LIVE LOOP",fill=ACCENT2,font=fl); d.text((80,92),task,fill=WHITE,font=fv)
    fb=font(34,"bold"); lw=int(d.textlength(latency,font=fb))+50; bx=W-50-lw
    d.rounded_rectangle([bx,50,bx+lw,110],radius=10,fill=ACCENT)
    d.text((bx+25,60),latency,fill=(0,0,0),font=fb)
    fc=font(18,"mono"); cap="GEMMA DECISION · MEASURED"
    d.text((W-50-int(d.textlength(cap,font=fc)),120),cap,fill=SUB,font=fc)
    if integrity:
        fs,fe=font(40,"bold"),font(18,"mono"); sw=d.textlength(integrity,font=fs)
        bw=sw+100; bx2=(W-bw)/2; by2=H-200
        d.rounded_rectangle([bx2,by2,bx2+bw,by2+110],radius=14,fill=(0,0,0,210))
        d.text((bx2+50,by2+18),"FULL PIPELINE",fill=ACCENT2,font=fe)
        ctext(d,W/2,by2+48,integrity,fs,WHITE)
    img.save(out)

def end_png(out):
    img=Image.new("RGB",(W,H),BG)
    glow(img,[(-200,-200,800,600,(0,212,170,55)),(W-600,H-400,W+200,H+200,(255,107,53,40))])
    d=ImageDraw.Draw(img)
    ctext(d,W/2,H/2-230,"CEREBRAS × GEMMA 4 HACKATHON",font(24,"mono"),ACCENT2)
    ctext(d,W/2,H/2-170,"GemmaBot",font(160,"bold"),WHITE)
    ctext(d,W/2,H/2+70,"6x faster decisions · honest numbers · code in the repo",font(42),ACCENT)
    img.save(out)

def img_clip(img,d,out,overlay=None):
    if overlay:
        run(["ffmpeg","-y","-loop","1","-t",str(d),"-i",str(img),"-loop","1","-t",str(d),"-i",str(overlay),
             "-filter_complex",f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];[bg][1:v]overlay=0:0:format=auto[v]",
             "-map","[v]","-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])
    else:
        run(["ffmpeg","-y","-loop","1","-t",str(d),"-i",str(img),
             "-vf",f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}",
             "-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])

def video_clip(src,d,out,overlay=None,speed=1.0):
    sp=f"setpts={1.0/speed:.3f}*PTS,"
    if overlay:
        run(["ffmpeg","-y","-stream_loop","-1","-i",str(src),"-loop","1","-t",str(d),"-i",str(overlay),"-t",str(d),
             "-filter_complex",f"[0:v]{sp}scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}[bg];[bg][1:v]overlay=0:0:format=auto[v]",
             "-map","[v]","-an","-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])
    else:
        run(["ffmpeg","-y","-stream_loop","-1","-i",str(src),"-t",str(d),
             "-vf",f"{sp}scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=#0a0e17,fps={FPS}",
             "-an","-pix_fmt","yuv420p","-c:v","libx264","-preset","fast","-crf","20",str(out)])

def main():
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    seg=[]

    title_png(WORK/"t.png"); img_clip(WORK/"t.png",2.5,WORK/"00.mp4"); seg.append(WORK/"00.mp4")

    ld,sd,pd=dur(VID/"full_lift.mp4"),dur(VID/"full_stack.mp4"),dur(VID/"full_pick.mp4")
    hud_png(WORK/"hl.png","Lift the red cube","~560 ms")
    video_clip(VID/"full_lift.mp4",4.0,WORK/"10.mp4",overlay=WORK/"hl.png",speed=max(ld/4.0,0.25)); seg.append(WORK/"10.mp4")
    hud_png(WORK/"hs.png","Stack red on green","~560 ms")
    video_clip(VID/"full_stack.mp4",4.0,WORK/"11.mp4",overlay=WORK/"hs.png",speed=max(sd/4.0,0.25)); seg.append(WORK/"11.mp4")
    hud_png(WORK/"hp.png","Pick & place the can","~560 ms",integrity="perception · Gemma reasoning · grasp · place")
    video_clip(VID/"full_pick.mp4",6.0,WORK/"12.mp4",overlay=WORK/"hp.png",speed=max(pd/6.0,0.5)); seg.append(WORK/"12.mp4")

    if (VID/"reasoning_panel.mp4").exists():
        video_clip(VID/"reasoning_panel.mp4",13.0,WORK/"20.mp4"); seg.append(WORK/"20.mp4")

    # HERO FINALE: the measured race (full, untrimmed — it has its own title)
    if (VID/"split_race.mp4").exists():
        video_clip(VID/"split_race.mp4",dur(VID/"split_race.mp4"),WORK/"30.mp4"); seg.append(WORK/"30.mp4")

    end_png(WORK/"e.png"); img_clip(WORK/"e.png",3.0,WORK/"90.mp4"); seg.append(WORK/"90.mp4")

    cl=WORK/"c.txt"; cl.write_text("\n".join(f"file '{p.resolve()}'" for p in seg))
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(cl),
         "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p",str(OUT)])
    print(f"\nDONE: {OUT}  ({OUT.stat().st_size/1_048_576:.1f} MB)  duration {dur(OUT):.1f}s")

if __name__=="__main__":
    main()
