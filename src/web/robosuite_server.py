"""Gemma 4 → Robosuite Panda live-control server.

Pipeline per /api/step:
  Snapshot (with object positions) → Gemma 4 → structured intent
  Executor drives the Panda OSC controller toward the target → final snapshot
  verify.py reads ground-truth physics → Verdict
  recorder.py persists everything so the run can be replayed cold later

The heavy lifting lives in src/web/lib/:
  - imaging.py   — orientation, JPEG-b64, grid, composite
  - sim.py       — robosuite env lifecycle + observation snapshots
  - brain.py     — Gemma prompt + structured-output call
  - executor.py  — motion + gripper position state-machine
  - tasks.py     — task registry + success thresholds
  - verify.py    — per-step ground-truth verdict (NOT Gemma\'s own opinion)
  - recorder.py  — write each step to runs/<task>_<ts>/ for replay
"""

from __future__ import annotations

from dataclasses import asdict
import os
import sys
import warnings
from pathlib import Path

# Repo root on sys.path so `src.*` imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING", "1")
warnings.filterwarnings("ignore")

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from src.web.lib.brain import GemmaBrain, HistoryItem  # noqa: E402
from src.web.lib.executor import MotionExecutor  # noqa: E402
from src.web.lib.grounding import VisionGroundingModule  # noqa: E402
from src.web.lib.imaging import img_to_b64, overlay_grid  # noqa: E402
from src.web.lib.recorder import RunRecorder, list_runs, load_run  # noqa: E402
from src.web.lib import tasks as tasks_mod  # noqa: E402
from src.web.lib.sim import PandaSim, Snapshot  # noqa: E402
from src.web.lib.verify import env_success, verify
from src.web.lib.viz import generate_debug_composite  # noqa: E402
from src.web.lib.perception import SamPerceptor  # noqa: E402  # noqa: E402


app = FastAPI(title="Gemma 4 → Panda Robot")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Session singletons (one process owns one MuJoCo env) ──
# Single-threaded by construction — uvicorn runs async handlers on one event
# loop and our handlers never yield mid-MuJoCo, so no lock is needed.
_sim = PandaSim()
_brain = GemmaBrain()              # CerebrasClient init is itself lazy
_executor = MotionExecutor(_sim)
_history: list[HistoryItem] = []
_step_count = 0
# A recorder is created on every /api/reset so each task attempt lands in its
# own runs/<task>_<ts>/ folder. The server holds one at a time.
_recorder: RunRecorder | None = None
# Vision grounding module -- lazily initialised when first vision-mode step is
# requested, so it does not slow down the initial page load.
_vision_grounding: VisionGroundingModule | None = None
_sam_perceptor: SamPerceptor | None = None


def _snapshot_payload(snap: Snapshot) -> dict:
    """Camera frames + EE state the UI uses for init/reset."""
    return {
        "birdview": img_to_b64(overlay_grid(snap.birdview)),
        "frontview": img_to_b64(snap.frontview),
        "ee": [round(float(snap.ee_pos[i]), 3) for i in range(3)],
        "gripper_open": snap.gripper_open,
        "objects": {k: [round(float(v[i]), 3) for i in range(3)] for k, v in snap.objects.items()},
        "task": _task_payload(_sim.task),
    }


def _task_payload(spec) -> dict:
    return {
        "key": spec.key,
        "label": spec.label,
        "description": spec.description,
        "mode": spec.mode,
        "target_object": spec.target_object,
        "objects": [{"key": k, "label": label} for k, label in spec.visible_objects],
    }


def _new_recorder() -> RunRecorder:
    global _recorder
    if _recorder is not None:
        _recorder.finalize()
    _recorder = RunRecorder(task=_sim.task.key)
    return _recorder


# ── API endpoints ──

@app.get("/api/tasks")
async def api_tasks() -> dict:
    """Task registry for the picker dropdown."""
    return {"tasks": tasks_mod.list_specs(), "active": _sim.task.key}


@app.get("/api/init")
async def api_init() -> dict:
    """Return the current observation snapshot. Does NOT reset the env."""
    global _step_count, _history
    snap = _sim.snapshot()
    _step_count = 0
    _history = []
    _executor.seed_from(snap)
    _new_recorder()
    return _snapshot_payload(snap)


@app.get("/api/reset")
async def api_reset(task: str | None = None) -> dict:
    """Full env teardown + rebuild. History, executor state, step count all clear.

    If ``task`` is provided and known, the env is switched first.
    """
    global _step_count, _history
    if task is not None and task in tasks_mod.TASKS:
        _sim.set_task(task)
    snap = _sim.reset()
    _step_count = 0
    _history = []
    _executor.seed_from(snap)
    _new_recorder()
    return _snapshot_payload(snap)


@app.get("/api/step")
async def api_step(task: str | None = None, vision: bool = False) -> dict:
    """One Gemma call + one tool execution. Returns frames + intent + verdict.

    When ``vision=true`` the entire pipeline runs through the vision-grounding
    (blinding) layer: Gemma receives camera-derived object positions instead of
    simulator ground truth, and tool execution resolves targets against the
    vision belief instead of the physics state. The verifier still reads ground
    truth behind the scenes so the judge is never fooled.

    Without vision (default): feeds ground-truth coordinates to Gemma
    (coordinate-fed baseline for comparison).
    """
    global _step_count, _vision_grounding
    import json as _json

    # 1) Snapshot — what Gemma sees (ground truth here is used only to judge).
    snap_before = _sim.snapshot()
    prompt_task = task or _sim.task.description
    
    # 1b) SAM perception + debug visualization (non-vision mode)
    _sam_debug_bird = None
    _sam_debug_front = None
    _sam_detections = []
    try:
        global _sam_perceptor
        if _sam_perceptor is None:
            _sam_perceptor = SamPerceptor()
        # Run detection in parallel (quick, ~200ms on CPU)
        dets_bird = _sam_perceptor.detect(snap_before.birdview)
        dets_front = _sam_perceptor.detect(snap_before.frontview)
        _sam_detections = [
            {"label": d.label or f"obj_{i}", "cx": d.cx, "cy": d.cy,
             "bbox": list(d.bbox), "world_xyz": list(d.world_xyz) if d.world_xyz else None,
             "confidence": d.confidence, "color": d.color_name(), "source": d.source,
             "area_px": d.area_px}
            for i, d in enumerate(dets_bird)
        ]
        # Build debug composite
        _sam_debug_bird = img_to_b64(generate_debug_composite(
            snap_before.birdview, snap_before.frontview,
            masks_bird=None, masks_front=None,
            detections_bird=dets_bird, detections_front=dets_front,
            ee_xy=(float(snap_before.ee_pos[0]), float(snap_before.ee_pos[1])),
            reasoning=intent.reasoning if 'intent' in dir() else "",
        ))
    except Exception as exc:
        _sam_debug_bird = None

    # Vision-mode setup: run the pixel-only pipeline and blind Gemma.
    vision_belief = None
    vision_errors = None
    vision_detections = None
    vision_text_override = None

    if vision:
        # Lazy init the vision grounding module.
        if _vision_grounding is None:
            _vision_grounding = VisionGroundingModule(
                _sim.env().sim,
                _sim.env().model,
                _sim.task,
            )
        obs = _sim.env()._get_observations(force_update=True)
        vision_belief = _vision_grounding.perceive(obs, gt_snapshot=snap_before)
        vision_text_override = vision_belief.as_prompt_block()
        vision_errors = vision_belief.errors
        vision_detections = [
            {
                "label": d.label,
                "world_xyz": list(d.world_xyz) if d.world_xyz else None,
                "area_px": d.area_px,
                "confidence": d.confidence,
                "source": d.source,
            }
            for d in vision_belief.detections
        ]

    intent = _brain.think(
        prompt_task, snap_before, _history, _sim.task,
        vision_text_override=vision_text_override,
    )
    _step_count += 1

    # 2) Execute the tool Gemma chose. In vision mode, targets are resolved
    #    from the vision belief rather than simulator ground truth.
    if intent.tool == "done":
        frames: list = []
        final = snap_before
    else:
        try:
            if vision and vision_belief is not None:
                result = _executor.execute_tool_vision(
                    snap_before, intent.tool, intent.params, vision_belief,
                )
            else:
                result = _executor.execute_tool(snap_before, intent.tool, intent.params)
            frames = result.frames
            final = result.final_snapshot
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the demo
            frames = []
            final = snap_before
            intent.reasoning = (intent.reasoning or "") + f" [executor error: {str(exc)[:120]}]"

    # 3) Verify against ground-truth physics (judge only, not Gemma's opinion).
    verdict = verify(snap_before, final, _sim.task)
    if env_success(final, _sim.task):
        verdict.success = True
        if _sim.task.mode in ("stack", "pick_place", "clear_table"):
            verdict.placed = True
        if _sim.task.mode in ("door", "wipe", "nut_assembly", "lift", "stack"):
            verdict.lifted = True

    # 4) Record to history (Gemma reads this as text on the next call).
    _history.append(HistoryItem(
        step=_step_count,
        tool=intent.tool,
        tool_params=_json.dumps(intent.params),
        reasoning=intent.reasoning,
        ee_x=float(final.ee_pos[0]),
        ee_y=float(final.ee_pos[1]),
        ee_z=float(final.ee_pos[2]),
        gripper_open=final.gripper_open,
        verdict_note=verdict.notes,
    ))

    # 5) Persist to disk for replay.
    bird_b64 = img_to_b64(overlay_grid(final.birdview))
    front_b64 = img_to_b64(final.frontview)
    if _recorder is not None:
        _recorder.step(
            intent={"tool": intent.tool, "params": intent.params, "reasoning": intent.reasoning},
            ee=[round(float(final.ee_pos[i]), 3) for i in range(3)],
            gripper_open=final.gripper_open,
            objects={k: [round(float(v[i]), 3) for i in range(3)] for k, v in final.objects.items()},
            verdict=verdict,
            latency_ms=intent.latency_ms,
            birdview_b64=bird_b64,
            frontview_b64=front_b64,
        )

    return {
        "frames": frames,
        "debug_birdview": _sam_debug_bird if '_sam_debug_bird' in dir() else None,
        "debug_frontview": _sam_debug_front if '_sam_debug_front' in dir() else None,
        "detections": _sam_detections if '_sam_detections' in dir() else [],
        "reasoning_text": intent.reasoning,
        "gemma_output": {
            "tool": intent.tool,
            "params": intent.params,
            "reasoning": intent.reasoning,
            "stage": intent.tool,  # legacy alias for the older frontend
        },
        "latency_ms": intent.latency_ms,
        "step": _step_count,
        "ee": [round(float(final.ee_pos[i]), 3) for i in range(3)],
        "gripper_open": final.gripper_open,
        "objects": {k: [round(float(v[i]), 3) for i in range(3)] for k, v in final.objects.items()},
        "verdict": verdict.to_json(),
        "history": [asdict(h) for h in _history[-5:]],
        "task": _task_payload(_sim.task),
        "run_id": _recorder.run_id if _recorder else None,
        "done": verdict.success or intent.tool == "done",
        # Vision (blinding) mode fields
        "vision_mode": vision,
        "vision_errors": vision_errors,
        "vision_detections": vision_detections,
    }


# ── Vision-step endpoint (convenience wrapper around api_step) ──

@app.get("/api/vision-step")
async def api_vision_step(task: str | None = None) -> dict:
    """Full blind pipeline: Gemma sees only camera-derived positions.

    Equivalent to /api/step?vision=true -- provided as a separate endpoint
    so the frontend does not need to manage the query parameter.
    """
    return await api_step(task=task, vision=True)


# ── Replay endpoints ──

@app.get("/api/runs")
async def api_runs() -> dict:
    """Index of every recorded run for the replay tab."""
    return {"runs": list_runs()}


@app.get("/api/replay/{run_id}")
async def api_replay(run_id: str) -> dict:
    try:
        return load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/health")
async def health() -> dict:
    return {"status": "healthy", "step": _step_count, "task": _sim.task.key}


# ── HTML routes ──

@app.get("/robot_live")
async def robot_live_page() -> HTMLResponse:
    p = STATIC_DIR / "robot_live.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return HTMLResponse("<h1>robot_live.html not found</h1>", status_code=404)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        "<h1>Gemma 4 → Panda Robot</h1>"
        "<p><a href=\'/robot_live\'>/robot_live</a> — Live control dashboard</p>"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  Gemma 4 → Panda Robot  |  http://localhost:8002/robot_live")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
