"""Standalone PushT Live Server — with hybrid search+LLM controller.

Uses the HybridPushtController from robot_video.pusht_controller:
  Phase 1: Systematic grid search (no LLM) — finds the T-block
  Phase 2: Gemma-guided pushing — once contact is established
  Phase 3: Adaptive reasoning — reward delta feedback loop

The /gemma-step endpoint drives this pipeline. The HTML reports detailed status:
  "Searching grid 12/64 at (224, 160)"
  "Contact made! Reward=0.120 at (128, 96)"
  "Good push! Reward +0.042 (now 0.235)"
  "Goal reached! Reward=0.853"
"""

from __future__ import annotations
import base64, io, os, sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
from PIL import Image
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

import gym_pusht.envs
import gymnasium as gym
from src.provider import ProviderRegistry
import json

from robot_video.pusht_controller import HybridPushtController, Phase

env = None
frame_count = 0
auto_mode = False

# ---- Hybrid controller singleton -------------------------------------------
CONTROLLER: HybridPushtController | None = None

def get_controller() -> HybridPushtController:
    global CONTROLLER
    if CONTROLLER is None:
        CONTROLLER = HybridPushtController()
        CONTROLLER.reset()
    return CONTROLLER


# ---- Legacy helpers kept for backward compat --------------------------------
def get_env():
    global env, frame_count
    if env is None:
        import warnings
        warnings.filterwarnings("ignore")
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels")
        env.reset()
        frame_count = 0
    return env

GEMMA_SCHEMA = {
    "name": "push_target",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_x": {"type": "number", "description": "X to move pusher (0-512)"},
            "target_y": {"type": "number", "description": "Y to move pusher (0-512)"},
            "reasoning": {"type": "string", "description": "Why this move?"},
        },
        "required": ["target_x", "target_y", "reasoning"],
    },
}

GEMMA_CLIENT = None
def get_gemma():
    global GEMMA_CLIENT
    if GEMMA_CLIENT is None:
        GEMMA_CLIENT = ProviderRegistry.default()
    return GEMMA_CLIENT

PUSHER_POS = [256.0, 256.0]
SWEEP_PHASE = 0
SWEEP_PATH = [
    [256, 100], [400, 100], [400, 400], [256, 400], [100, 400], [100, 100],
    [256, 200], [350, 200], [350, 350], [256, 350], [150, 350], [150, 200],
    [256, 256],
]


# ---- Endpoints --------------------------------------------------------------

async def step(request):
    global env, frame_count, PUSHER_POS, SWEEP_PHASE, auto_mode
    e = get_env()
    direction = request.query_params.get("dir", "")
    manual = request.query_params.get("manual", "0")

    if manual == "1" and direction:
        offsets = {"up": [0, -30], "down": [0, 30], "left": [-30, 0], "right": [30, 0]}
        off = offsets.get(direction, [0, 0])
        PUSHER_POS[0] = max(0, min(512, PUSHER_POS[0] + off[0]))
        PUSHER_POS[1] = max(0, min(512, PUSHER_POS[1] + off[1]))
        action = np.array(PUSHER_POS, dtype=np.float32)
    else:
        PUSHER_POS = list(SWEEP_PATH[SWEEP_PHASE % len(SWEEP_PATH)])
        action = np.array(PUSHER_POS, dtype=np.float32)
        SWEEP_PHASE += 1

    total_reward = 0.0
    for _ in range(4):
        obs, reward, term, trunc, _ = e.step(action)
        total_reward += float(reward)
        if term or trunc:
            e.reset()
            PUSHER_POS = [256.0, 256.0]
            break

    frame_img = e.render()
    frame_count += 1

    pil = Image.fromarray(frame_img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "image": f"data:image/jpeg;base64,{b64}",
        "reward": total_reward,
        "frame": frame_count,
        "pos": PUSHER_POS,
        "mode": "auto" if not manual == "1" else "manual"
    })


async def reset(request):
    global env, frame_count, PUSHER_POS, SWEEP_PHASE, CONTROLLER
    if env:
        env.close()
    env = None
    frame_count = 0
    PUSHER_POS = [256.0, 256.0]
    SWEEP_PHASE = 0
    if CONTROLLER is not None:
        CONTROLLER.reset()
        CONTROLLER = None
    return JSONResponse({"status": "reset"})


async def gemma_step(request):
    """Hybrid controller step: search -> contact -> push with feedback.

    Returns phase, status message, reward, reasoning, and frame.
    This IS the hybrid pipeline -- no separate endpoint needed.
    """
    ctrl = get_controller()

    # Execute one step of whatever phase we're in
    result = ctrl.step()

    return JSONResponse({
        "image": result.image_b64,
        "reward": round(result.reward, 4),
        "reward_delta": round(result.reward_delta, 4),
        "frame": result.step,
        "pos": result.pos,
        "phase": result.phase,
        "message": result.message,
        "reasoning": result.gemma_reasoning,
        "latency_ms": round(result.gemma_latency_ms, 0),
        "best_reward": round(ctrl.best_reward, 4),
    })


async def hybrid_run(request):
    """Run the full hybrid controller for N steps and return results summary."""
    ctrl = get_controller()
    steps_param = request.query_params.get("steps", "40")
    try:
        max_steps = min(int(steps_param), 80)
    except ValueError:
        max_steps = 40

    results = ctrl.run_episode(max_steps=max_steps)

    return JSONResponse({
        "steps": len(results),
        "final_phase": ctrl.phase,
        "best_reward": round(ctrl.best_reward, 4),
        "final_reward": round(results[-1].reward, 4) if results else 0,
        "total_steps": ctrl._step_count,
        "results": [
            {
                "step": r.step,
                "phase": r.phase,
                "reward": round(r.reward, 4),
                "reward_delta": round(r.reward_delta, 4),
                "message": r.message,
            }
            for r in results
        ],
    })


async def homepage(request):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PushT -- Hybrid Controller (Search + Gemma)</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#e8edf5;font-family:Inter,sans-serif;display:flex;flex-direction:column;align-items:center;padding:30px 20px}
h1{font-size:22px;font-weight:700;background:linear-gradient(135deg,#00d4aa,#4a9eff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.sub{color:#8892b0;font-size:13px;margin-bottom:16px}
.main-box{background:#131a2b;border:1px solid #1e2a45;border-radius:10px;padding:16px;width:100%;max-width:700px}
#frame{width:100%;border-radius:6px;display:block;image-rendering:pixelated}
.ctrl{display:flex;gap:6px;justify-content:center;margin-top:12px;flex-wrap:wrap}
.btn{background:linear-gradient(135deg,#00d4aa,#00b894);color:#0a0e17;border:none;padding:8px 18px;border-radius:6px;font-weight:600;cursor:pointer;font-size:13px;min-width:50px}
.btn:hover{opacity:0.9}
.btn-red{background:#ff6b6b}
.btn-blue{background:linear-gradient(135deg,#4a9eff,#7c5cfc)}
.btn-purple{background:linear-gradient(135deg,#a855f7,#7c5cfc)}
.btn-amber{background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff}
.info{color:#8892b0;font-size:12px;margin-top:10px;text-align:center}
.active{outline:2px solid #00d4aa}
.phase-badge{display:inline-block;padding:3px 12px;border-radius:12px;font-weight:600;font-size:12px}
.phase-search{background:#f59e0b33;color:#f59e0b;border:1px solid #f59e0b}
.phase-contact{background:#00d4aa33;color:#00d4aa;border:1px solid #00d4aa}
.phase-push{background:#4a9eff33;color:#4a9eff;border:1px solid #4a9eff}
.phase-done{background:#22c55e33;color:#22c55e;border:1px solid #22c55e}
.status-line{padding:10px;border-radius:6px;margin-top:10px;font-size:13px;line-height:1.5}
.status-search{background:#f59e0b11;border-left:3px solid #f59e0b}
.status-contact{background:#00d4aa11;border-left:3px solid #00d4aa}
.status-push{background:#4a9eff11;border-left:3px solid #4a9eff}
.status-done{background:#22c55e11;border-left:3px solid #22c55e}
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;font-size:12px}
.stat-label{color:#6b7280}.stat-value{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<h1>PushT -- Hybrid Controller</h1>
<p class="sub">Systematic grid search + Gemma-guided pushing + adaptive feedback</p>
<div class="main-box">
<img id="frame" src="" alt="PushT simulation">
<div class="ctrl" id="manualCtrls">
<button class="btn" onmousedown="move('up')" ontouchstart="move('up')">↑</button>
<button class="btn" onmousedown="move('left')" ontouchstart="move('left')">←</button>
<button class="btn" onmousedown="move('down')" ontouchstart="move('down')">↓</button>
<button class="btn" onmousedown="move('right')" ontouchstart="move('right')">→</button>
<button class="btn btn-red" onclick="resetSim()">↺ Reset</button>
</div>
<div class="ctrl">
<button class="btn btn-blue" id="hybridBtn" onclick="stepHybrid()">Step Hybrid</button>
<button class="btn btn-purple" id="loopBtn" onclick="toggleLoop()">Auto-Run</button>
<button class="btn btn-amber" onclick="runFull()">Full Episode</button>
</div>

<div id="statusBox" class="status-line status-search">
  <strong><span id="phaseBadge" class="phase-badge phase-search">SEARCH</span></strong>
  <span id="statusMsg">Ready. Click "Step Hybrid" to begin searching for the T-block.</span>
</div>

<div class="stats-grid">
  <span class="stat-label">Pusher Position</span>
  <span class="stat-value" id="posDisplay">(256, 256)</span>
  <span class="stat-label">Reward</span>
  <span class="stat-value" id="rewardDisplay">0.0000</span>
  <span class="stat-label">Best Reward</span>
  <span class="stat-value" id="bestRewardDisplay">0.0000</span>
  <span class="stat-label">Reward Delta</span>
  <span class="stat-value" id="deltaDisplay">--</span>
  <span class="stat-label">Step</span>
  <span class="stat-value" id="stepDisplay">0</span>
  <span class="stat-label">Gemma Latency</span>
  <span class="stat-value" id="latencyDisplay">--</span>
</div>

<div id="reasoningBox" style="margin-top:10px;padding:8px 10px;background:#0a0e17;border-radius:6px;font-size:11px;color:#6b7280;min-height:20px;display:none">
  <strong>Gemma:</strong> <span id="reasoningText"></span>
</div>
</div>

<script>
let loopTimer = null;
let loopRunning = false;

async function move(dir) {
    const r = await fetch('/step?dir=' + dir + '&manual=1');
    const d = await r.json();
    document.getElementById('frame').src = d.image;
}

async function resetSim() {
    if (loopRunning) toggleLoop();
    const r = await fetch('/reset');
    await r.json();
    const s = await fetch('/step');
    const d = await s.json();
    document.getElementById('frame').src = d.image;
    document.getElementById('statusMsg').textContent = 'Reset. Click "Step Hybrid" to begin.';
    document.getElementById('rewardDisplay').textContent = '0.0000';
    document.getElementById('bestRewardDisplay').textContent = '0.0000';
    document.getElementById('deltaDisplay').textContent = '--';
    document.getElementById('stepDisplay').textContent = '0';
    document.getElementById('posDisplay').textContent = '(256, 256)';
    document.getElementById('latencyDisplay').textContent = '--';
    document.getElementById('reasoningBox').style.display = 'none';
    setPhase('search', 'SEARCH', 'Reset complete. Click "Step Hybrid" to begin.');
}

function setPhase(phaseClass, phaseLabel, msg) {
    const box = document.getElementById('statusBox');
    box.className = 'status-line status-' + phaseClass;
    document.getElementById('phaseBadge').className = 'phase-badge phase-' + phaseClass;
    document.getElementById('phaseBadge').textContent = phaseLabel;
    document.getElementById('statusMsg').textContent = msg;
}

async function stepHybrid() {
    document.getElementById('hybridBtn').disabled = true;
    document.getElementById('hybridBtn').textContent = 'Working...';
    try {
        const r = await fetch('/gemma-step');
        const d = await r.json();

        document.getElementById('frame').src = d.image;
        document.getElementById('stepDisplay').textContent = d.frame;
        document.getElementById('rewardDisplay').textContent = d.reward.toFixed(4);
        document.getElementById('bestRewardDisplay').textContent = d.best_reward.toFixed(4);
        document.getElementById('posDisplay').textContent = '(' + d.pos[0].toFixed(0) + ', ' + d.pos[1].toFixed(0) + ')';

        if (d.reward_delta !== 0) {
            const sign = d.reward_delta > 0 ? '+' : '';
            document.getElementById('deltaDisplay').textContent = sign + d.reward_delta.toFixed(4);
        } else {
            document.getElementById('deltaDisplay').textContent = '0.0000';
        }

        if (d.latency_ms > 0) {
            document.getElementById('latencyDisplay').textContent = d.latency_ms + 'ms';
        } else {
            document.getElementById('latencyDisplay').textContent = '--';
        }

        let phaseClass = 'search';
        let phaseLabel = 'SEARCH';
        if (d.phase === 'contact' || d.phase === 'push') {
            phaseClass = 'push';
            phaseLabel = 'PUSHING';
        } else if (d.phase === 'done') {
            phaseClass = 'done';
            phaseLabel = 'DONE';
        }
        setPhase(phaseClass, phaseLabel, d.message);

        if (d.reasoning) {
            document.getElementById('reasoningBox').style.display = 'block';
            document.getElementById('reasoningText').textContent = d.reasoning;
        }

        if (d.phase === 'done' && loopRunning) {
            toggleLoop();
        }
    } finally {
        document.getElementById('hybridBtn').disabled = false;
        document.getElementById('hybridBtn').textContent = 'Step Hybrid';
    }
}

function toggleLoop() {
    loopRunning = !loopRunning;
    if (loopRunning) {
        loopTimer = setInterval(stepHybrid, 800);
        document.getElementById('loopBtn').textContent = 'Stop';
        document.getElementById('loopBtn').classList.add('active');
    } else {
        clearInterval(loopTimer);
        document.getElementById('loopBtn').textContent = 'Auto-Run';
        document.getElementById('loopBtn').classList.remove('active');
    }
}

async function runFull() {
    document.getElementById('loopBtn').disabled = true;
    document.getElementById('hybridBtn').disabled = true;
    document.getElementById('loopBtn').textContent = 'Running...';
    if (loopRunning) toggleLoop();

    await fetch('/reset');
    await new Promise(r => setTimeout(r, 100));

    const r = await fetch('/hybrid-run?steps=60');
    const d = await r.json();

    const last = await fetch('/gemma-step');
    const lastD = await last.json();
    document.getElementById('frame').src = lastD.image;

    document.getElementById('stepDisplay').textContent = d.total_steps;
    document.getElementById('bestRewardDisplay').textContent = d.best_reward.toFixed(4);

    let phaseClass = 'done';
    let phaseLabel = 'DONE';
    const msg = 'Episode complete. ' + d.steps + ' steps, best reward ' + d.best_reward.toFixed(4);
    setPhase(phaseClass, phaseLabel, msg);

    document.getElementById('loopBtn').disabled = false;
    document.getElementById('hybridBtn').disabled = false;
    document.getElementById('loopBtn').textContent = 'Auto-Run';
}

window.onload = async function() {
    const s = await fetch('/step');
    const d = await s.json();
    document.getElementById('frame').src = d.image;
};
</script>
</body>
</html>"""
    return HTMLResponse(html)

app = Starlette(debug=False, routes=[
    Route("/", homepage),
    Route("/step", step),
    Route("/reset", reset),
    Route("/gemma-step", gemma_step),
    Route("/hybrid-run", hybrid_run),
])

if __name__ == "__main__":
    print("PushT hybrid controller: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
