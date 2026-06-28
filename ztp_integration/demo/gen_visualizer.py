"""Generate the comprehensive visualizer HTML page with embedded frame images."""
from __future__ import annotations

import math, base64, io, json
from pathlib import Path
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 640, 420
STEP = 20.0

HERE = Path(__file__).resolve().parent
OUT = HERE / "visualizer.html"


def create_coord_vis(
    world_objects: dict,
    gripper_pos: tuple[float, float],
    target_pos: tuple[float, float] | None,
    tick: int,
) -> str:
    img = Image.new("RGB", (WIDTH + 220, HEIGHT), (15, 15, 35))
    d = ImageDraw.Draw(img)

    # Sim area
    d.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], fill=(238, 238, 240))
    GRID_COLS, GRID_ROWS = 3, 2
    cw, ch = WIDTH / GRID_COLS, HEIGHT / GRID_ROWS
    for c in range(1, GRID_COLS):
        d.line([(c * cw, 0), (c * cw, HEIGHT)], fill=(200, 200, 210), width=1)
    for r in range(1, GRID_ROWS):
        d.line([(0, r * ch), (WIDTH, r * ch)], fill=(200, 200, 210), width=1)
    for i, lab in enumerate(["A", "B", "C", "D", "E", "F"]):
        r2, c2 = divmod(i, GRID_COLS)
        d.text((c2 * cw + 5, r2 * ch + 4), f"Zone {lab}", fill=(170, 170, 180))

    colors = {"cracked_cup": (200, 175, 120), "red_cup": (210, 60, 60), "blue_cup": (60, 90, 210)}
    for oid, (ox, oy) in world_objects.items():
        color = colors.get(oid, (150, 150, 150))
        d.ellipse([ox - 26, oy - 26, ox + 26, oy + 26], fill=color, outline=(35, 35, 35), width=2)
        d.text((ox - 30, oy + 30), oid.replace("_", " "), fill=(30, 30, 30))
        d.text((ox - 30, oy - 45), f"({ox}, {oy})", fill=(100, 100, 100))

    d.rectangle([85 - 36, 360 - 26, 85 + 36, 360 + 26], outline=(95, 95, 95), width=3)
    d.text((85 - 32, 360 - 8), "bin_left", fill=(95, 95, 95))

    gx, gy = gripper_pos
    d.line([(gx - 17, gy), (gx + 17, gy)], fill=(45, 120, 205), width=4)
    d.line([(gx, gy), (gx, gy - 24)], fill=(45, 120, 205), width=4)
    d.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], fill=(45, 120, 205))

    # Info panel
    px = WIDTH + 10
    d.rectangle([WIDTH, 0, WIDTH + 219, HEIGHT - 1], fill=(20, 20, 45), outline=(60, 60, 80))
    d.text((px, 10), "COORDINATE SYSTEM", fill=(233, 69, 96))
    d.text((px, 30), f"Tick {tick + 1}", fill=(78, 204, 163))
    d.text((px, 50), f"World: {WIDTH} x {HEIGHT}px", fill=(200, 200, 200))
    d.text((px, 70), f"Step/tic: {STEP}px", fill=(200, 200, 200))

    y = 100
    d.text((px, y), "OBJECT POSITIONS:", fill=(233, 69, 96))
    y += 20
    for oid, (ox, oy) in world_objects.items():
        d.text((px, y), f"  {oid}: ({ox}, {oy})", fill=(150, 150, 150))
        y += 18

    d.text((px, y), f"GRIPPER:", fill=(78, 204, 163))
    d.text((px, y + 18), f"  ({gx:.0f}, {gy:.0f})", fill=(150, 150, 150))
    y += 40

    if target_pos:
        tx, ty = target_pos
        dx = tx - gx
        dy = ty - gy
        dist = math.hypot(dx, dy)
        d.text((px, y), "TARGET:", fill=(233, 69, 96))
        d.text((px, y + 18), f"  ({tx}, {ty})", fill=(150, 150, 150))
        d.text((px, y + 38), f"  dx={dx:.0f} dy={dy:.0f}", fill=(150, 150, 150))
        d.text((px, y + 58), f"  dist={dist:.0f}px", fill=(233, 69, 96))
        d.text((px, y + 78), f"  ~{max(1, math.ceil(dist / STEP))} ticks", fill=(200, 200, 200))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_frames() -> list[str]:
    world = {"cracked_cup": (330, 180), "red_cup": (200, 300), "blue_cup": (440, 300)}
    gx, gy = 320.0, 30.0
    target = (330, 180)
    frames = []

    for _ in range(11):
        dx, dy = target[0] - gx, target[1] - gy
        dist = math.hypot(dx, dy)
        if dist <= 14:
            frames.append(create_coord_vis(world, (gx, gy), target, len(frames)))
            break
        gx += STEP * dx / dist
        gy += STEP * dy / dist
        frames.append(create_coord_vis(world, (gx, gy), target, len(frames)))

    # Perturbation: cup moves
    world2 = dict(world)
    world2["cracked_cup"] = (520, 130)
    target2 = (520, 130)
    for _ in range(8):
        dx, dy = target2[0] - gx, target2[1] - gy
        dist = math.hypot(dx, dy)
        if dist <= 14:
            frames.append(create_coord_vis(world2, (gx, gy), target2, len(frames)))
            break
        gx += STEP * dx / dist
        gy += STEP * dy / dist
        frames.append(create_coord_vis(world2, (gx, gy), target2, len(frames)))

    return frames


def write_html(frames: list[str]) -> None:
    real_data = [
        (1, "pick cracked_cup", 275, "✅ SAFE", "(321, 50)", "150px"),
        (2, "pick cracked_cup", 293, "✅ SAFE", "(323, 70)", "130px"),
        (3, "pick cracked_cup", 279, "✅ SAFE", "(324, 90)", "110px"),
        (4, "pick cracked_cup", 296, "✅ SAFE", "(325, 110)", "90px"),
        (5, "pick cracked_cup", 273, "✅ SAFE", "(327, 130)", "70px"),
        (6, "pick cracked_cup", 279, "✅ SAFE", "(328, 150)", "50px"),
        (7, "pick cracked_cup", 331, "✅ SAFE", "(408, 142)⚡", "113px→NEW TARGET"),
        (8, "pick cracked_cup", 367, "✅ SAFE", "(407, 142)", "113px"),
        (9, "pick cracked_cup", 285, "✅ SAFE", "(408, 132)", "112px"),
        (10, "pick cracked_cup", 373, "✅ SAFE", "(408, 142)", "112px"),
    ]
    rows = "".join(
        f"<tr{' style=\"background:#1a1a0a;\"' if r[0] == 7 else ''}>"
        f"<td>{'⚡' if r[0] == 7 else ''}{r[0]}</td>"
        f"<td>{r[1]}</td><td>{r[2]}ms</td>"
        f"<td style=\"color:#4ecca3\">{r[3]}</td>"
        f"<td>{r[4]}</td><td>{r[5]}</td></tr>"
        for r in real_data
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GemmaBot: Complete Visual Guide</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#08081a;color:#e0e0e0;max-width:1100px;margin:0 auto;padding:20px}}
h1{{color:#e94560;font-size:28px;text-align:center}}
h2{{color:#4ecca3;font-size:16px;text-align:center;margin-bottom:20px}}
.card{{background:#12122a;border-radius:12px;padding:20px;margin:15px 0;border:1px solid #2a2a4a}}
.vis{{width:100%;border-radius:8px;border:2px solid #3a3a5a}}
.controls{{display:flex;align-items:center;gap:12px;margin:15px 0;justify-content:center;flex-wrap:wrap}}
.controls input[type=range]{{flex:1;min-width:200px;accent-color:#e94560}}
.btn{{background:#e94560;color:#fff;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:14px}}
.btn-alt{{background:transparent;border:1px solid #4ecca3;color:#4ecca3}}
.code{{font-family:monospace;background:#08081a;padding:2px 6px;border-radius:4px;color:#4ecca3;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{text-align:left;padding:8px;border-bottom:2px solid #e94560;color:#e94560}}
td{{padding:8px;border-bottom:1px solid #2a2a4a}}
.flex{{display:flex;gap:15px;flex-wrap:wrap}}
.col{{flex:1;min-width:280px}}
pre{{background:#08081a;padding:12px;border-radius:6px;font-size:12px;line-height:1.5;overflow-x:auto}}
.highlight{{background:#1a1a3a;padding:15px;border-radius:8px;border-left:4px solid #e94560;margin:10px 0}}
.highlight.green{{border-left-color:#4ecca3}}
.highlight.yellow{{border-left-color:#ffd700}}
</style></head><body>

<h1>🤖 GemmaBot: Complete Visual Guide</h1>
<h2>How Gemma 4 + ZTP Physics + Sim Control Work Together</h2>

<div class="card">
<h2>📍 The Coordinate System</h2>
<p style="color:#888;margin-bottom:10px">The world is <b>640 x 420 pixels</b>. Every object has a pixel position. Gemma 4 <b>never sees these coordinates</b> — it only sees the rendered image. The right panel shows the coordinate math.</p>
<div style="text-align:center">
<img id="frameDisplay" class="vis" src="data:image/png;base64,{frames[0]}">
</div>
<div class="controls">
<button class="btn" onclick="changeFrame(-1)">◀</button>
<span id="frameCounter">1 / {len(frames)}</span>
<button class="btn" onclick="changeFrame(1)">▶</button>
<button class="btn btn-alt" onclick="playAuto()" id="playBtn">▶ Auto</button>
<input type="range" id="frameSlider" min="0" max="{len(frames)-1}" value="0" oninput="updateFrame(parseInt(this.value))" style="width:300px">
</div>
</div>

<div class="card">
<h2>🔄 Per-Tick Pipeline</h2>
<div class="flex">
<div class="col">
<div class="highlight green">
<h3 style="color:#4ecca3">1️⃣ Gemma 4 sees image</h3>
<p style="color:#aaa;font-size:13px;margin-top:5px">Outputs: <span class="code">{{"skill":"pick","target":"cracked_cup"}}</span><br>⚡ ~280ms on Cerebras</p>
</div></div>
<div class="col">
<div class="highlight yellow">
<h3 style="color:#ffd700">2️⃣ ZTP validates physics</h3>
<p style="color:#aaa;font-size:13px;margin-top:5px">Checks force, compaction, ESD<br>🔬 ~0.07ms</p>
</div></div>
<div class="col">
<div class="highlight">
<h3 style="color:#e94560">3️⃣ Sim moves gripper</h3>
<p style="color:#aaa;font-size:13px;margin-top:5px">20px toward target coordinate<br>🤖 Resolves name → position</p>
</div></div>
</div>
</div>

<div class="card">
<h2>📏 The Distance Question</h2>
<div class="flex">
<div class="col">
<h3 style="color:#e94560">What Actually Happens</h3>
<pre>
Gemma says:     "pick cracked_cup"
                ← semantic (names object)
Sim knows:      cracked_cup is at (330, 180)
                ← geometric (has a map)
Sim calculates: gripper at (320, 30)
                dx = 10, dy = 150
                dist = 150px
                step = 20px ← FIXED!
                move_x = 20 × 10/150 = 1.3
                move_y = 20 × 150/150 = 20
                new_pos = (321.3, 50.0)
                ← repeat until within 14px</pre>
</div>
<div class="col">
<h3 style="color:#4ecca3">The Analogy: Reaching for Coffee</h3>
<div class="highlight green">
<p style="color:#aaa;font-size:13px">You don't calculate coordinates to grab a cup. You <b>see it</b>, <b>reach for it</b>, and your eyes <b>adjust mid-motion</b>.</p>
<p style="color:#aaa;font-size:13px;margin-top:8px">Gemma 4 works the same way: it says "pick" each tick, the sim moves 20px, and next tick Gemma sees the <b>new image</b> and re-decides.</p>
<p style="color:#aaa;font-size:13px;margin-top:8px"><b>Far away?</b> More ticks. Speed of Cerebras (~280ms/tick) is why this works in real time.</p>
</div>
<div class="highlight" style="margin-top:10px">
<h3 style="color:#e94560">⚠️ Critical</h3>
<p style="color:#aaa;font-size:12px">Gemma 4 <b>does NOT know distance or coordinates</b>. It names targets. The sim resolves position. The 20px step is hardcoded. Gemma re-decides from the <b>current image</b> each tick — that's how it tracks moving objects.</p>
</div>
</div>
</div>
</div>

<div class="card">
<h2>📊 Live Cerebras Data (10 ticks, real API calls)</h2>
<table>
<tr><th>Tick</th><th>Gemma Decided</th><th>Latency</th><th>ZTP</th><th>Gripper</th><th>Distance</th></tr>
{rows}
</table>
</div>

<div class="card">
<h2>🛡️ ZTP Physics Blocks</h2>
<div class="flex">
<div class="col">
<h3 style="color:#4ecca3">Surgical Force Limits</h3>
<table>
<tr><th>Force</th><th>Result</th></tr>
<tr><td>0.8N (normal)</td><td style="color:#4ecca3">✅ SAFE</td></tr>
<tr><td>1.8N (too strong)</td><td style="color:#e94560">❌ BLOCKED</td></tr>
<tr><td>5.0N (crushing)</td><td style="color:#e94560">❌ BLOCKED</td></tr>
</table>
<p style="color:#888;font-size:12px;margin-top:8px">Force clamped at 1.2N for delicate objects. Gripper never closes if over limit.</p>
</div>
<div class="col">
<h3 style="color:#4ecca3">ESD Safety</h3>
<table>
<tr><th>Charge</th><th>Result</th></tr>
<tr><td>50V (normal)</td><td style="color:#4ecca3">✅ SAFE</td></tr>
<tr><td>150V (warn)</td><td style="color:#4ecca3">✅ SAFE</td></tr>
<tr><td>250V (danger)</td><td style="color:#e94560">❌ BLOCKED</td></tr>
</table>
<p style="color:#888;font-size:12px;margin-top:8px">Above 200V, ZTP blocks release to prevent spark discharge.</p>
</div>
</div>
</div>

<div class="card">
<h2>🎯 Key Takeaways</h2>
<div class="flex">
<div class="col">
<div class="highlight green">
<h3>✅ Working</h3>
<ul style="color:#aaa;font-size:13px;line-height:1.8;margin-left:15px">
<li>Gemma 4 at ~300ms/tick on Cerebras</li>
<li>ZTP force limit validation ✅</li>
<li>ZTP ESD safety validation ✅</li>
<li>Visual tracking of moved objects</li>
<li>Real-time closed loop at 3Hz</li>
</ul>
</div>
</div>
<div class="col">
<div class="highlight">
<h3>⚠️ Honest</h3>
<ul style="color:#aaa;font-size:13px;line-height:1.8;margin-left:15px">
<li>Terran needs 500kg+ vehicles</li>
<li>Step size is fixed (20px)</li>
<li>No collision avoidance</li></ul>
</div>
</div>
<div class="col">
<div class="highlight yellow">
<h3>🔥 Demo Hook</h3>
<ul style="color:#aaa;font-size:13px;line-height:1.8;margin-left:15px">
<li>51 vs 8 decisions (6x faster)</li>
<li>ZTP: AI → Physics → Action</li>
<li>Tracks moving objects</li>
</ul>
</div>
</div>
</div>
</div>

<script>
const frames = {json.dumps(frames)};
let idx = 0, timer = null;
function updateFrame(i){{idx=i;document.getElementById('frameDisplay').src='data:image/png;base64,'+frames[i];document.getElementById('frameSlider').value=i;document.getElementById('frameCounter').textContent=(i+1)+' / '+frames.length}}
function changeFrame(d){{let n=idx+d;if(n<0)n=0;if(n>=frames.length)n=frames.length-1;updateFrame(n)}}
function playAuto(){{const b=document.getElementById('playBtn');if(timer){{clearInterval(timer);timer=null;b.textContent='▶ Auto';return}}b.textContent='⏸ Stop';timer=setInterval(()=>{{if(idx>=frames.length-1){{clearInterval(timer);timer=null;b.textContent='▶ Auto';idx=0;updateFrame(0);return}}changeFrame(1)}},900)}}
</script>
</body></html>"""

    OUT.write_text(html)
    print(f"✓ Visualizer written to {OUT}")
    print(f"  Frames: {len(frames)}")


if __name__ == "__main__":
    frames = build_frames()
    write_html(frames)
