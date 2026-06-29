"""
Live Sim Dashboard — Full Perception → Plan → Execute Loop

Shows the MuJoCo robot simulation, Gemma 4's camera view,
what it perceives, the plans it makes, and real-time execution.

Usage:
    python scripts/live_dashboard.py
    # Then open http://localhost:8900
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ["OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING"] = "1"

from src.web.lib.sim import PandaSim, Snapshot
from src.web.lib.brain import GemmaBrain, HistoryItem, Intent
from src.web.lib.executor import MotionExecutor, Disturbance
from src.web.lib.verify import verify, env_success
from src.web.lib.tasks import TASKS, get as get_task
from src.web.lib.imaging import img_to_b64, fix_img, overlay_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("dashboard")

HOST = "0.0.0.0"
PORT = 8900

# --- Live state (thread-safe) ---
_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "status": "idle",
    "task": "",
    "step": 0,
    "max_steps": 0,
    "camera_frame": None,  # base64 JPEG
    "gemma_view": None,    # base64 of what Gemma sees (with grid overlay)
    "perception": {},      # what Gemma identified
    "intent": {},          # current intent
    "history": [],         # full history
    "metrics": {},         # latency, tokens, etc.
    "sim_state": {},       # object positions, gripper state
    "error": None,
    "log": [],
    "disturbance": None,
}

def _log_msg(msg: str):
    with _state_lock:
        _state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        if len(_state["log"]) > 200:
            _state["log"] = _state["log"][-200:]

def _update(**kw):
    with _state_lock:
        _state.update(kw)


# --- The loop runner ---
def run_loop(task_key: str = "lift_cube", max_steps: int = 15,
             disturb_every: int = 4, max_disturbances: int = 2):
    try:
        spec = get_task(task_key)
        _update(
            running=True,
            status="initializing",
            task=spec.label,
            step=0,
            max_steps=max_steps,
            error=None,
            history=[],
        )
        _log_msg(f"Starting task: {spec.label} (mode={spec.mode})")

        sim = PandaSim(spec)
        brain = GemmaBrain()
        executor = MotionExecutor(sim)
        disturber = Disturbance(sim)
        history: list[HistoryItem] = []
        disturbances_applied = 0

        snap = sim.reset()
        executor.seed_from(snap)

        # Send initial camera frame
        cam_img = fix_img(getattr(snap, 'img', None))
        _update(camera_frame=img_to_b64(cam_img, quality=75))

        prev_snap: Snapshot | None = None

        for step_idx in range(1, max_steps + 1):
            _update(step=step_idx, status=f"step_{step_idx}")

            snap_before = sim.snapshot()
            cam_img = fix_img(getattr(snap_before, 'img', None))
            _update(camera_frame=img_to_b64(cam_img, quality=75))

            # Build Gemma's view (with grid)
            gemma_view = cam_img.copy()
            gemma_view = overlay_grid(gemma_view)
            _update(gemma_view=img_to_b64(gemma_view, quality=75))

            # Sim state
            obj_pos = {}
            if hasattr(snap_before, 'objects') and snap_before.objects:
                for oid, odata in snap_before.objects.items():
                    if hasattr(odata, 'pos'):
                        obj_pos[oid] = [round(float(v), 4) for v in odata.pos]
            _update(sim_state={
                "ee_pos": [round(float(v), 4) for v in snap_before.ee_pos],
                "gripper_open": snap_before.gripper_open,
                "objects": obj_pos,
            })

            # Apply disturbance?
            disturb_info = None
            if disturbances_applied < max_disturbances and step_idx > 1 and step_idx % disturb_every == 0:
                if hasattr(snap_before, 'objects') and snap_before.objects:
                    obj = spec.target_object or list(snap_before.objects.keys())[0]
                    if obj and obj in snap_before.objects:
                        dx, dy = np.random.uniform(-0.08, 0.08, 2).tolist()
                        disturb_info = disturber.move_object(obj, dx=dx, dy=dy)
                        disturbances_applied += 1
                        snap_before = sim.snapshot()
                        _update(disturbance={
                            "type": "object_move",
                            "object": obj,
                            "dx": dx,
                            "dy": dy,
                            "count": disturbances_applied,
                        })
                        _log_msg(f"Disturbance {disturbances_applied}: {obj} moved ({dx:.3f},{dy:.3f})")
                        cam_img = fix_img(getattr(snap_before, 'img', None))
                        _update(camera_frame=img_to_b64(cam_img, quality=75))

            # Gemma thinks
            _update(status=f"thinking_{step_idx}")
            t0 = time.perf_counter()
            intent = brain.think(spec.description, snap_before, history, spec, prev_snap)
            latency = (time.perf_counter() - t0) * 1000

            _update(intent={
                "tool": intent.tool,
                "params": intent.params,
                "reasoning": intent.reasoning[:300] if intent.reasoning else None,
                "latency_ms": round(latency, 1),
            })
            _log_msg(f"Step {step_idx}: tool={intent.tool} reason={intent.reasoning[:80] if intent.reasoning else None}")

            # Execute
            if intent.tool == "done":
                _update(status="done_early")
                _log_msg("DONE (early)")
                break

            _update(status=f"executing_{step_idx}_{intent.tool}")
            try:
                result = executor.execute_tool(snap_before, intent.tool, intent.params)
            except Exception as e:
                _log_msg(f"Tool error: {e}, falling back to move_to")
                result = executor.execute(snap_before, {
                    "target_x": intent.params.get("x", snap_before.ee_pos[0]),
                    "target_y": intent.params.get("y", snap_before.ee_pos[1]),
                    "target_z": intent.params.get("z", snap_before.ee_pos[2]),
                }, "hold")

            final = result.final_snapshot
            v = verify(snap_before, final, spec)
            if env_success(final, spec):
                v.success = True

            history.append(HistoryItem(
                step=step_idx,
                tool=intent.tool,
                tool_params=str(intent.params),
                reasoning=intent.reasoning or "",
                ee_x=float(final.ee_pos[0]),
                ee_y=float(final.ee_pos[1]),
                ee_z=float(final.ee_pos[2]),
                gripper_open=final.gripper_open,
                success=v.success,
                reached=v.reached,
                grasped=v.grasped,
                lifted=v.lifted,
                placed=v.placed,
            ))

            _update(history=[
                {"step": h.step, "tool": h.tool, "success": h.success,
                 "reached": h.reached, "grasped": h.grasped, "lifted": h.lifted, "placed": h.placed}
                for h in history
            ])

            # Perception: extract from intent reasoning
            _update(perception={
                "objects_detected": len(snap_before.objects) if hasattr(snap_before, 'objects') else 0,
                "ee_position": [round(float(v), 4) for v in final.ee_pos],
                "gripper_open": final.gripper_open,
            })

            cam_img = fix_img(getattr(final, 'img', None))
            _update(camera_frame=img_to_b64(cam_img, quality=75))

            prev_snap = final

            # Small delay so UI can update
            time.sleep(0.1)

        final_snap = sim.snapshot()
        success = env_success(final_snap, spec)
        _update(
            status="complete_success" if success else "complete_fail",
            running=False,
        )
        _log_msg(f"Task complete: {'SUCCESS' if success else 'FAIL'} after {len(history)} steps")

    except Exception as e:
        _log_msg(f"ERROR: {e}")
        _update(status="error", error=str(e), running=False)
        import traceback
        _log_msg(traceback.format_exc())


# --- HTTP Server ---
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gemma Robotics Live Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0d1117; color: #e6edf3; padding: 20px; }
h1 { font-size: 22px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; }
h1 span.status { font-size: 13px; padding: 3px 10px; border-radius: 10px; font-weight: 500; }
.idle { background: #30363d; color: #8b949e; }
.running { background: #1f6feb22; color: #58a6ff; }
.thinking { background: #d2992222; color: #d29922; }
.executing { background: #23863622; color: #3fb950; }
.error { background: #da363322; color: #f85149; }
.success { background: #23863622; color: #3fb950; }

.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.panel h2 { font-size: 14px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 12px; }
.panel.full { grid-column: 1 / -1; }
.row { display: flex; gap: 12px; align-items: start; flex-wrap: wrap; }
.col { flex: 1; min-width: 200px; }

img { max-width: 100%; border-radius: 4px; border: 1px solid #30363d; }
.camera-row { display: flex; gap: 12px; }
.camera-row figure { flex: 1; text-align: center; }
.camera-row figcaption { font-size: 11px; color: #8b949e; margin-top: 4px; }

.metrics { display: flex; gap: 16px; flex-wrap: wrap; }
.metric { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px 14px; min-width: 100px; }
.metric .val { font-size: 20px; font-weight: 600; color: #f0f6fc; }
.metric .lbl { font-size: 11px; color: #8b949e; margin-top: 2px; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #8b949e; padding: 6px 8px; border-bottom: 1px solid #21262d; font-weight: 500; }
td { padding: 6px 8px; border-bottom: 1px solid #21262d; color: #c9d1d9; }
tr.success td { color: #3fb950; }
tr.fail td { color: #f85149; }
tr.current td { background: #1f6feb11; }

pre { font-size: 12px; background: #0d1117; padding: 10px; border-radius: 4px; overflow-x: auto; color: #c9d1d9; max-height: 200px; overflow-y: auto; }

.log-box { max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
.log-box div { padding: 2px 0; color: #8b949e; border-bottom: 1px solid #21262d11; }
.log-box div:nth-child(odd) { background: #0d1117; }

.controls { margin-bottom: 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.controls button { padding: 6px 16px; border-radius: 6px; border: 1px solid #30363d;
                   background: #21262d; color: #c9d1d9; cursor: pointer; font-size: 13px; }
.controls button:hover { background: #30363d; }
.controls button:disabled { opacity: .5; cursor: not-allowed; }
.controls button.primary { background: #238636; border-color: #238636; color: #fff; }
.controls button.danger { background: #da3633; border-color: #da3633; color: #fff; }
.controls select, .controls input { padding: 5px 8px; border-radius: 6px; border: 1px solid #30363d;
                                     background: #0d1117; color: #c9d1d9; font-size: 13px; }

.json-toggle { cursor: pointer; color: #58a6ff; font-size: 12px; }
.json-toggle:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>🤖 Gemma Robotics Live Dashboard
  <span class="status" id="statusBadge">idle</span>
</h1>

<div class="controls">
  <select id="taskSelect">
    <option value="pick_can">Pick Can</option>
    <option value="pick_milk">Pick Milk</option>
    <option value="pick_bread">Pick Bread</option>
    <option value="pick_cereal">Pick Cereal</option>
    <option value="lift_cube" selected>Lift Cube</option>
    <option value="stack_cubes">Stack Cubes</option>
  </select>
  <input type="number" id="maxSteps" value="15" min="5" max="30" style="width:60px">
  <button class="primary" id="startBtn">▶ Start</button>
  <button id="refreshBtn">↻ Refresh</button>
  <span id="autoBadge" style="font-size:12px;color:#8b949e;margin-left:8px;">auto-refresh on</span>
</div>

<div class="grid">
  <div class="panel">
    <h2>📷 Robot Camera</h2>
    <div class="camera-row">
      <figure>
        <img id="cameraFrame" src="" alt="Waiting for camera..." style="max-width:384px">
        <figcaption>Raw Camera Feed</figcaption>
      </figure>
      <figure>
        <img id="gemmaView" src="" alt="Waiting..." style="max-width:384px">
        <figcaption>Gemma's View (with grid)</figcaption>
      </figure>
    </div>
  </div>

  <div class="panel">
    <h2>📊 Metrics</h2>
    <div class="metrics" id="metrics">
      <div class="metric"><div class="val" id="mStep">0</div><div class="lbl">Step</div></div>
      <div class="metric"><div class="val" id="mLatency">-</div><div class="lbl">Latency</div></div>
      <div class="metric"><div class="val" id="mObjects">-</div><div class="lbl">Objects</div></div>
      <div class="metric"><div class="val" id="mGripper">-</div><div class="lbl">Gripper</div></div>
      <div class="metric"><div class="val" id="mTool">-</div><div class="lbl">Current Tool</div></div>
    </div>

    <h2 style="margin-top:16px;">🧠 Gemma's Intent</h2>
    <pre id="intentDisplay">Waiting...</pre>

    <h2 style="margin-top:12px;">🔍 Perception</h2>
    <pre id="perceptionDisplay">Waiting...</pre>
  </div>
</div>

<div class="panel full">
  <h2>📜 Step History</h2>
  <table>
    <thead><tr><th>Step</th><th>Tool</th><th>Reached</th><th>Grasped</th><th>Lifted</th><th>Placed</th><th>Success</th></tr></thead>
    <tbody id="historyTable"><tr><td colspan="7" style="color:#8b949e;text-align:center;">No steps yet</td></tr></tbody>
  </table>
</div>

<div class="panel full">
  <h2>💬 Log</h2>
  <div class="log-box" id="logBox"><div>Waiting...</div></div>
</div>

<div class="panel full">
  <h2>⚙️ Sim State</h2>
  <pre id="simStateDisplay">{}</pre>
</div>

<script>
let autoRefresh = true;
let refreshTimer = null;

function startAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(fetchState, 800);
}

function fetchState() {
  fetch('/api/state').then(r => r.json()).then(s => {
    updateUI(s);
  }).catch(() => {});
}

function updateUI(s) {
  // Status badge
  const badge = document.getElementById('statusBadge');
  const statusMap = {
    'idle': ['idle', 'idle'],
    'initializing': ['running', 'initializing'],
    'thinking': ['thinking', 'thinking...'],
    'executing': ['executing', 'executing'],
    'done_early': ['success', 'done (early)'],
    'complete_success': ['success', '✓ complete'],
    'complete_fail': ['error', '✗ failed'],
    'error': ['error', 'error'],
    'step': ['running', 'step ' + s.step],
  };
  const [cls, label] = statusMap[s.status] || ['running', s.status];
  badge.className = 'status ' + cls;
  badge.textContent = label;

  // Images
  if (s.camera_frame) document.getElementById('cameraFrame').src = 'data:image/jpeg;base64,' + s.camera_frame;
  if (s.gemma_view) document.getElementById('gemmaView').src = 'data:image/jpeg;base64,' + s.gemma_view;

  // Metrics
  document.getElementById('mStep').textContent = s.step + '/' + s.max_steps;
  if (s.intent && s.intent.latency_ms) document.getElementById('mLatency').textContent = Math.round(s.intent.latency_ms) + 'ms';
  if (s.perception && s.perception.objects_detected !== undefined) document.getElementById('mObjects').textContent = s.perception.objects_detected;
  if (s.sim_state) document.getElementById('mGripper').textContent = s.sim_state.gripper_open ? 'open' : 'closed';
  if (s.intent && s.intent.tool) document.getElementById('mTool').textContent = s.intent.tool;

  // Intent display
  if (s.intent && s.intent.tool) {
    document.getElementById('intentDisplay').textContent = JSON.stringify(s.intent, null, 2);
  }

  // Perception
  if (s.perception) {
    document.getElementById('perceptionDisplay').textContent = JSON.stringify(s.perception, null, 2);
  }

  // History table
  if (s.history && s.history.length > 0) {
    const tbody = document.getElementById('historyTable');
    let html = '';
    const currentStep = s.step;
    s.history.forEach((h, i) => {
      const isCurrent = (i + 1) === currentStep;
      html += `<tr class="${isCurrent ? 'current' : ''} ${h.success ? 'success' : 'fail'}">
        <td>${h.step}</td>
        <td>${h.tool}</td>
        <td>${h.reached ? '✓' : '✗'}</td>
        <td>${h.grasped ? '✓' : '✗'}</td>
        <td>${h.lifted ? '✓' : '✗'}</td>
        <td>${h.placed ? '✓' : '✗'}</td>
        <td>${h.success ? '✓' : '✗'}</td>
      </tr>`;
    });
    tbody.innerHTML = html;
  }

  // Log
  if (s.log && s.log.length > 0) {
    const logBox = document.getElementById('logBox');
    logBox.innerHTML = s.log.map(l => '<div>' + l.replace(/</g, '&lt;') + '</div>').join('');
    logBox.scrollTop = logBox.scrollHeight;
  }

  // Sim state
  if (s.sim_state) {
    document.getElementById('simStateDisplay').textContent = JSON.stringify(s.sim_state, null, 2);
  }

  // Start button
  document.getElementById('startBtn').disabled = s.running;
  document.getElementById('startBtn').textContent = s.running ? '⏳ Running...' : '▶ Start';
}

// Start button
document.getElementById('startBtn').addEventListener('click', () => {
  const task = document.getElementById('taskSelect').value;
  const steps = document.getElementById('maxSteps').value;
  fetch('/api/start?task=' + task + '&steps=' + steps).then(r => r.json()).then(d => {
    if (d.error) alert(d.error);
  });
});

document.getElementById('refreshBtn').addEventListener('click', fetchState);

// Auto-refresh
startAutoRefresh();
fetchState();
</script>
</body>
</html>"""

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif path == "/api/state":
            with _state_lock:
                data = {
                    "running": _state["running"],
                    "status": _state["status"],
                    "task": _state["task"],
                    "step": _state["step"],
                    "max_steps": _state["max_steps"],
                    "camera_frame": _state["camera_frame"],
                    "gemma_view": _state["gemma_view"],
                    "perception": _state["perception"],
                    "intent": _state["intent"],
                    "history": _state["history"],
                    "metrics": _state["metrics"],
                    "sim_state": _state["sim_state"],
                    "error": _state["error"],
                    "log": _state["log"][-50:],
                    "disturbance": _state["disturbance"],
                }
            self._send_json(data)
        elif path == "/api/start":
            task = params.get("task", ["lift_cube"])[0]
            steps = int(params.get("steps", [15])[0])
            with _state_lock:
                if _state["running"]:
                    self._send_json({"error": "Already running"})
                    return
            thread = threading.Thread(target=run_loop, args=(task, steps), daemon=True)
            thread.start()
            self._send_json({"started": True, "task": task, "steps": steps})
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress HTTP log noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    server = HTTPServer((HOST, args.port), DashboardHandler)
    log.info(f"Live dashboard at http://localhost:{args.port}")
    log.info("Open in browser and click 'Start' to run a task")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
