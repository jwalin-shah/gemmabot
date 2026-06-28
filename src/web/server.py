"""FastAPI server for the Cerebras x Gemma 4 reactive simulation web app.

Serves the reactive loop and multi-agent pipeline via SSE streaming and REST.

Usage (from project root):
    python -m src.web.server          (no reload, single worker)
    uvicorn src.web.server:app --reload
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, AsyncGenerator

# --- Path setup: allow ``python -m src.web.server`` from project root ---------
_HERE = Path(__file__).resolve().parent  # src/web
_SRC = _HERE.parent                       # src
_PROJ = _SRC.parent                       # project root
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# --- FastAPI & friends --------------------------------------------------------
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.client import CerebrasClient
from src.config import PROJECT_ROOT
from src.orchestrator import AgentOrchestrator, PipelineMode
from src.sim.brain import Decision, RobotBrain
from src.sim.compare import ThrottledClient
from src.sim.loop import ReactiveLoop
from src.sim.run_sim import INSTRUCTION, build_world
from src.sim.skills import REACH
from src.sim.world import World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERTURB_POS = (520.0, 130.0)  # Zone C — where the cup gets dragged
GPU_EXTRA_LATENCY_S = 1.7  # extra latency for the GPU throttled side
STATIC_DIR = _HERE / "static"
WORKSPACE_IMAGE = str(PROJECT_ROOT / "examples" / "images" / "workspace.jpg")
POLL_INTERVAL_S = 0.01  # how often the SSE generator polls the event queue

# ---------------------------------------------------------------------------
# Global state (thread-safe via locks / atomic operations)
# ---------------------------------------------------------------------------
_lock = threading.Lock()


@dataclass
class SimSideState:
    """Mutable stats for one simulation side, shared between threads."""
    label: str = ""  # "cerebras" | "gpu"
    tick: int = 0
    decision_count: int = 0
    latency_sum_ms: float = 0.0
    avg_latency_ms: float = 0.0
    hz: float = 0.0
    status: str = "idle"  # idle | running | done | error
    reacquired_ms: float | None = None
    perturbed: bool = False
    last_decision: dict[str, Any] = field(default_factory=dict)
    frame_png: str = ""  # latest base64 PNG data URI


@dataclass
class SimRunner:
    """Holds one simulation side's world, loop, brain, and thread handle."""
    label: str
    world: World
    loop: ReactiveLoop
    brain: RobotBrain
    thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    state: SimSideState = field(default_factory=SimSideState)


# Shared state
_cerebras_runner: SimRunner | None = None
_gpu_runner: SimRunner | None = None
_event_queue: Queue[dict[str, Any]] = Queue(maxsize=500)
_running = False
_start_time: float = 0.0

# Pipeline cache
_pipeline_cache: dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cerebras x Gemma 4 — Reactive Sim Race",
    version="0.1.0",
    description="Web interface for the Cerebras vs GPU reactive robot sim race.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at /
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse({"status": "error", "message": "index.html not found"}, status_code=404)
    return index_path.read_text()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_to_data_uri(world: World) -> str:
    """Render the world to a PNG, return a base64 data URI."""
    img = world.render()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _stats_dict(runner: SimRunner | None) -> dict[str, Any]:
    """Return a JSON-safe stats dict for one runner."""
    if runner is None:
        return {"ticks": 0, "avg_latency_ms": 0, "decision_count": 0, "hz": 0,
                "reacquired_ms": None}
    s = runner.state
    return {
        "ticks": s.tick,
        "avg_latency_ms": round(s.avg_latency_ms, 1),
        "decision_count": s.decision_count,
        "hz": round(s.hz, 2),
        "reacquired_ms": round(s.reacquired_ms, 1) if s.reacquired_ms is not None else None,
    }


def _check_reacquired(runner: SimRunner, cup_x: float, cup_y: float) -> None:
    """Check if the gripper has re-acquired the cracked cup after perturbation."""
    if not runner.state.perturbed or runner.state.reacquired_ms is not None:
        return
    g = runner.world.gripper
    dist = math.hypot(g.x - cup_x, g.y - cup_y)
    if g.holding == "cracked_cup" or dist <= REACH + 2:
        runner.state.reacquired_ms = (time.perf_counter() - _start_time) * 1000


def _determine_winner() -> str | None:
    """Return 'cerebras' or 'gpu' if one has clearly won, else None.

    Winner is the side that has completed more ticks (reached "done" status).
    If both have same tick count and at least one is done, it's a tie (None).
    """
    c = _cerebras_runner
    g = _gpu_runner
    if c is None or g is None:
        return None
    cs = c.state
    gs = g.state
    if cs.status == "done" and gs.status != "done":
        return "cerebras"
    if gs.status == "done" and cs.status != "done":
        return "gpu"
    return None


def _build_sim_runner(label: str, throttled: bool = False) -> SimRunner:
    """Create a world, brain, and loop for one simulation side."""
    world = build_world()
    client = CerebrasClient()
    if throttled:
        client = ThrottledClient(client, extra_latency_s=GPU_EXTRA_LATENCY_S)
    brain = RobotBrain(client)
    loop = ReactiveLoop(world, brain)
    loop.set_instruction(INSTRUCTION)

    runner = SimRunner(label=label, world=world, loop=loop, brain=brain)
    runner.state.label = label
    return runner


# ---------------------------------------------------------------------------
# Background simulation thread
# ---------------------------------------------------------------------------

def _sim_thread(runner: SimRunner) -> None:
    """Background thread: run the reactive loop until stopped or task done."""
    global _running

    loop: ReactiveLoop = runner.loop
    world: World = runner.world
    state: SimSideState = runner.state
    stop = runner.stop_event
    cup_x, cup_y = PERTURB_POS

    while not stop.is_set():
        # --- Perturbation check (wall-clock time) ---
        if not state.perturbed and _running:
            # Always perturb on first tick after start for consistency
            # In this thread, we perturb after a brief delay if running
            pass  # Perturbation is externally triggered via API

        # --- Tick ---
        try:
            result = loop.tick()
        except Exception as exc:
            state.status = "error"
            _event_queue.put({
                "type": "tick",
                "side": runner.label,
                "tick": state.tick,
                "frame_png": "",
                "decision": {"skill": "error", "target": "", "reasoning": str(exc), "latency_ms": 0},
                "status": "error",
                "avg_latency_ms": state.avg_latency_ms,
                "decision_count": state.decision_count,
                "hz": state.hz,
            }, timeout=1.0)
            break

        # --- Update state ---
        state.tick = result.tick
        d = result.decision
        state.last_decision = {
            "skill": d.skill,
            "target": d.target,
            "target_zone": d.target_zone,
            "reasoning": d.reasoning,
            "latency_ms": round(d.latency_ms, 1),
        }
        if d.skill in ("pick", "place", "move_to", "stop"):
            state.decision_count += 1
            state.latency_sum_ms += d.latency_ms

        state.avg_latency_ms = (
            state.latency_sum_ms / state.decision_count if state.decision_count else 0
        )
        elapsed = (time.perf_counter() - _start_time)
        state.hz = state.decision_count / elapsed if elapsed > 0 else 0

        # --- Check re-acquisition ---
        _check_reacquired(runner, cup_x, cup_y)

        # --- Determine status ---
        if d.skill == "done":
            state.status = "done"
        elif state.status != "error":
            state.status = "running"

        # --- Render frame ---
        try:
            frame_uri = _render_to_data_uri(world)
            state.frame_png = frame_uri
        except Exception:
            frame_uri = ""

        # --- Emit tick event ---
        _event_queue.put({
            "type": "tick",
            "side": runner.label,
            "tick": state.tick,
            "frame_png": frame_uri,
            "decision": state.last_decision,
            "status": state.status,
            "avg_latency_ms": round(state.avg_latency_ms, 1),
            "decision_count": state.decision_count,
            "hz": round(state.hz, 2),
        }, timeout=1.0)

        # --- Emit status event ---
        c_stats = _stats_dict(_cerebras_runner)
        g_stats = _stats_dict(_gpu_runner)
        winner = _determine_winner()
        _event_queue.put({
            "type": "status",
            "cerebras": c_stats,
            "gpu": g_stats,
            "perturbed": state.perturbed,
            "winner": winner,
        }, timeout=1.0)

        # --- Stop if task done ---
        if d.skill == "done" or result.status == "done":
            # Keep running one more tick after done to let the UI catch up,
            # but don't do further work.
            state.status = "done"
            break

    _running = False


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

@app.get("/stream")
async def stream_events(request: Request) -> StreamingResponse:
    """SSE endpoint: streams tick and status events from background sims."""

    async def event_generator() -> AsyncGenerator[str, None]:
        global _running
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                # Non-blocking poll of the thread-safe queue
                event = _event_queue.get_nowait()
                event_type = event.pop("type", "message")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            except Empty:
                # If both sims are stopped and no events, end the stream
                if not _running and _event_queue.empty():
                    # Send a final event so the client knows
                    yield f"event: done\ndata: {json.dumps({'message': 'stream ended'})}\n\n"
                    break
                await asyncio.sleep(POLL_INTERVAL_S)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/api/start")
async def api_start() -> JSONResponse:
    """Start both sim instances in background threads."""
    global _cerebras_runner, _gpu_runner, _running, _start_time

    with _lock:
        if _running:
            return JSONResponse({"message": "Already running", "status": "ok"})

        # Clear old events
        while not _event_queue.empty():
            try:
                _event_queue.get_nowait()
            except Empty:
                break

        # Create runners
        _cerebras_runner = _build_sim_runner("cerebras", throttled=False)
        _gpu_runner = _build_sim_runner("gpu", throttled=True)

        # Set stop events
        _cerebras_runner.stop_event.clear()
        _gpu_runner.stop_event.clear()

        # Start threads
        _running = True
        _start_time = time.perf_counter()

        _cerebras_runner.thread = threading.Thread(
            target=_sim_thread, args=(_cerebras_runner,), daemon=True,
            name="sim-cerebras",
        )
        _gpu_runner.thread = threading.Thread(
            target=_sim_thread, args=(_gpu_runner,), daemon=True,
            name="sim-gpu",
        )
        _cerebras_runner.thread.start()
        _gpu_runner.thread.start()

        return JSONResponse({
            "message": "Simulation started — Cerebras (fast) vs GPU (throttled)",
            "status": "running",
            "instruction": INSTRUCTION,
        })


@app.post("/api/stop")
async def api_stop() -> JSONResponse:
    """Stop both sim instances."""
    global _running

    with _lock:
        _running = False
        for runner in (_cerebras_runner, _gpu_runner):
            if runner is not None:
                runner.stop_event.set()
        _cerebras_runner = None
        _gpu_runner = None

    return JSONResponse({"message": "Simulation stopped", "status": "stopped"})


@app.post("/api/perturb")
async def api_perturb() -> JSONResponse:
    """Drag the cracked cup to a new position in both worlds."""
    with _lock:
        for runner in (_cerebras_runner, _gpu_runner):
            if runner is None:
                continue
            cup = runner.world.get("cracked_cup")
            if cup is not None:
                cup.x, cup.y = PERTURB_POS
                runner.state.perturbed = True

    return JSONResponse({
        "message": f"Cracked cup moved to {PERTURB_POS}",
        "position": list(PERTURB_POS),
        "status": "perturbed",
    })


@app.get("/api/stats")
async def api_stats() -> JSONResponse:
    """Return current stats for both sides."""
    return JSONResponse({
        "cerebras": _stats_dict(_cerebras_runner),
        "gpu": _stats_dict(_gpu_runner),
        "running": _running,
        "perturbed": any(
            r.state.perturbed
            for r in (_cerebras_runner, _gpu_runner)
            if r is not None
        ),
        "winner": _determine_winner(),
    })


# ---------------------------------------------------------------------------
# Multi-agent pipeline endpoint
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    task: str = (
        "Identify objects in the scene, their positions, "
        "and potential grasp targets."
    )
    mode: str = "parallel"  # "sequential", "parallel", "single_shot"


@app.post("/api/pipeline")
async def api_pipeline(req: PipelineRequest) -> JSONResponse:
    """Run the multi-agent pipeline (Vision -> Action -> Safety -> Execute).

    Modes:
      - sequential: original one-by-one (Vision then Action then Safety)
      - parallel (default): Vision + Action run in parallel, both see the image
      - single_shot: one multimodal call produces all three outputs at once
    """
    global _pipeline_cache

    # Check for workspace image
    if not Path(WORKSPACE_IMAGE).is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Workspace image not found at {WORKSPACE_IMAGE}",
        )

    try:
        client = CerebrasClient()
        orchestrator = AgentOrchestrator(client)
        result = orchestrator.run(WORKSPACE_IMAGE, task=req.task, mode=req.mode)

        response = {
            "task": req.task,
            "image": WORKSPACE_IMAGE,
            "mode": result.mode,
            "vision_analysis": result.scene_analysis,
            "action_plan": result.action_plan,
            "safety_review": result.safety_review,
            "executed_actions": [
                asdict(a) for a in result.executed_actions
            ],
            "timing": {
                "total_time_s": round(result.total_time_s, 3),
                "pipeline_steps": {
                    k: round(v, 3) for k, v in result.pipeline.items()
                },
            },
        }

        _pipeline_cache = response
        return JSONResponse(response)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")


@app.get("/api/pipeline-results")
async def api_pipeline_results(mode: str = "parallel") -> JSONResponse:
    """Return cached pipeline results from a previous run."""
    global _pipeline_cache

    if _pipeline_cache is None:
        # Auto-run pipeline if no cache exists
        try:
            client = CerebrasClient()
            orchestrator = AgentOrchestrator(client)
            result = orchestrator.run(
                WORKSPACE_IMAGE,
                task="Identify objects in the scene, their positions, "
                     "and potential grasp targets.",
                mode=mode,
            )

            _pipeline_cache = {
                "task": "Identify objects in the scene, their positions, "
                        "and potential grasp targets.",
                "mode": result.mode,
                "image": WORKSPACE_IMAGE,
                "vision_analysis": result.scene_analysis,
                "action_plan": result.action_plan,
                "safety_review": result.safety_review,
                "executed_actions": [
                    asdict(a) for a in result.executed_actions
                ],
                "timing": {
                    "total_time_s": round(result.total_time_s, 3),
                    "pipeline_steps": {
                        k: round(v, 3) for k, v in result.pipeline.items()
                    },
                },
            }
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline auto-run error: {exc}",
            )

    return JSONResponse(_pipeline_cache)


# ---------------------------------------------------------------------------
# Health / debug
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "running": _running,
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run with uvicorn directly when invoked as ``python -m src.web.server``."""
    import uvicorn
    uvicorn.run(
        "src.web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
