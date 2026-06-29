"""ZTP / Real Dataset Replay Server — offline benchmarking via LeRobot datasets.

Serves a dark-themed web UI on port 8003 that lets you:
  - Browse available LeRobot datasets
  - Step through dataset episode frames
  - Submit frames to Gemma 4 and compare predictions against ground truth
  - Run full episode benchmarks with aggregate reports

Usage:
    python -m src.web.replay_server
    uvicorn src.web.replay_server:app --reload  (for development)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

# -- Path setup -------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_PROJ = _SRC.parent
import sys
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# -- FastAPI -----------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from robot_video.replay_engine import (
    DatasetReplayEngine,
    available_datasets,
    KNOWN_DATASET_IDS,
    FrameComparison,
)
from robot_video.action_comparator import ActionComparator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATIC_DIR = _HERE / "static"
PORT = 8003

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="GemmaBot ZTP Replay", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_engine = DatasetReplayEngine()
_engine_lock = threading.Lock()

# Background benchmark jobs
_benchmark_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class SelectEpisodeRequest(BaseModel):
    dataset: str = "lerobot/aloha_mobile_cabinet"
    episode: int = 0

class GemmaIntentRequest(BaseModel):
    dataset: str = "lerobot/aloha_mobile_cabinet"
    episode: int = 0
    frame: int = 0
    intent: dict[str, Any] | None = None
    latency_ms: float = 0.0
    object_positions: dict[str, list[float]] | None = None

class BenchmarkRequest(BaseModel):
    dataset: str = "lerobot/aloha_mobile_cabinet"
    episode: int = 0
    max_frames: int = -1
    # Provider URL is not used directly — the client sends intents per frame
    intent_provider: str = "manual"

# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@app.get("/api/replay/datasets")
async def api_list_datasets() -> list[dict[str, Any]]:
    """List all available datasets with metadata."""
    result = []
    for ds_id, meta in available_datasets().items():
        result.append({
            "id": ds_id,
            "task_type": meta["task_type"],
            "description": meta["description"],
        })
    return result


@app.get("/api/replay/dataset/{name:path}/info")
async def api_dataset_info(name: str) -> dict[str, Any]:
    """Get info for a specific dataset (loads if not already loaded)."""
    with _engine_lock:
        if not _engine.is_loaded or _engine.dataset_info.get("repo_id") != name:
            try:
                info = _engine.load_dataset(name)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            info = _engine.dataset_info
    return info


@app.get("/api/replay/dataset/{name:path}/episodes")
async def api_dataset_episodes(name: str) -> list[dict[str, Any]]:
    """List all episodes in a dataset with frame counts."""
    with _engine_lock:
        if not _engine.is_loaded or _engine.dataset_info.get("repo_id") != name:
            try:
                _engine.load_dataset(name)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        episodes = []
        for ep_idx in range(_engine.dataset_info["num_episodes"]):
            try:
                frames = _engine._source.episode_frames(ep_idx) if _engine._source else 0
            except Exception:
                frames = 0
            episodes.append({"episode_index": ep_idx, "num_frames": frames})
    return episodes


# ---------------------------------------------------------------------------
# Episode / frame endpoints
# ---------------------------------------------------------------------------

@app.post("/api/replay/episode/select")
async def api_select_episode(req: SelectEpisodeRequest) -> dict[str, Any]:
    """Select a dataset and episode for replay."""
    with _engine_lock:
        if not _engine.is_loaded or _engine.dataset_info.get("repo_id") != req.dataset:
            try:
                _engine.load_dataset(req.dataset)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        try:
            result = _engine.select_episode(req.episode)
        except (IndexError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/api/replay/episode/frame")
async def api_get_frame(
    frame: int = Query(..., description="Frame index within the episode"),
) -> dict[str, Any]:
    """Get a frame at the given index within the selected episode."""
    with _engine_lock:
        try:
            vf = _engine.get_frame(frame)
        except (RuntimeError, IndexError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "episode_index": vf.episode_index,
        "frame_index": vf.frame_index,
        "image_uri": vf.image_uri,
        "image_size": vf.image_size,
        "action": vf.action,
        "state": vf.state,
        "timestamp": vf.timestamp,
        "camera_key": vf.camera_key,
    }


@app.post("/api/replay/episode/gemma-intent")
async def api_record_intent(req: GemmaIntentRequest) -> dict[str, Any]:
    """Record a Gemma intent for the current frame and return comparison."""
    with _engine_lock:
        # Ensure correct dataset/episode
        if not _engine.is_loaded or _engine.dataset_info.get("repo_id") != req.dataset:
            try:
                _engine.load_dataset(req.dataset)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        try:
            _engine.select_episode(req.episode)
        except (IndexError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get the frame
        try:
            _engine.get_frame(req.frame)
        except (RuntimeError, IndexError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Record intent
        try:
            fc = _engine.record_gemma_intent(
                req.intent,
                latency_ms=req.latency_ms,
                object_positions=req.object_positions,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return fc.to_dict()


@app.get("/api/replay/episode/log")
async def api_frame_log() -> list[dict[str, Any]]:
    """Get the full frame log for the current episode."""
    with _engine_lock:
        return _engine.frame_log()


@app.post("/api/replay/episode/clear")
async def api_clear_log() -> dict[str, Any]:
    """Clear the frame log for the current episode."""
    with _engine_lock:
        _engine._prev_action = None
        _engine._log.clear()
        if _engine._comparator is not None:
            _engine._comparator.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------

@app.post("/api/replay/benchmark")
async def api_start_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    """Start a background benchmark job and return a job ID.

    Since the intent provider is external, the benchmark collects frames
    and the client supplies intents via /api/replay/episode/gemma-intent.
    The benchmark aggregates results from the frame log.
    """
    job_id = uuid.uuid4().hex[:12]

    with _engine_lock:
        if not _engine.is_loaded or _engine.dataset_info.get("repo_id") != req.dataset:
            try:
                _engine.load_dataset(req.dataset)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        try:
            _engine.select_episode(req.episode)
        except (IndexError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        _engine._prev_action = None
        _engine._log.clear()
        if _engine._comparator is not None:
            _engine._comparator.clear()

        total_frames = (
            _engine._episode_frames if req.max_frames < 0
            else min(req.max_frames, _engine._episode_frames)
        )

    with _jobs_lock:
        _benchmark_jobs[job_id] = {
            "job_id": job_id,
            "dataset": req.dataset,
            "episode": req.episode,
            "total_frames": total_frames,
            "processed_frames": 0,
            "status": "running",
            "started_at": time.time(),
            "completed_at": None,
            "report": None,
        }

    return {
        "job_id": job_id,
        "total_frames": total_frames,
        "status": "running",
        "message": "Submit intents via POST /api/replay/episode/gemma-intent, "
                   "then GET /api/replay/benchmark/{job_id}/status to check progress "
                   "and /api/replay/benchmark/{job_id}/report to get the final report.",
    }


@app.get("/api/replay/benchmark/{job_id}/status")
async def api_benchmark_status(job_id: str) -> dict[str, Any]:
    """Get the status of a benchmark job."""
    with _jobs_lock:
        job = _benchmark_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Update processed frames from the engine log
        with _engine_lock:
            log_len = len(_engine._log)
            job["processed_frames"] = log_len
            if log_len >= job["total_frames"] and job["status"] == "running":
                job["status"] = "ready_for_report"

        return dict(job)


@app.get("/api/replay/benchmark/{job_id}/report")
async def api_benchmark_report(job_id: str) -> dict[str, Any]:
    """Get the final benchmark report for a job."""
    with _jobs_lock:
        job = _benchmark_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    with _engine_lock:
        if _engine._comparator is None:
            report = ActionComparator.compute_benchmark([], task_type="unknown")
        else:
            report = _engine._comparator.compute_benchmark(
                _engine._comparator.results,
                task_type=_engine._comparator.task_type,
            )

    with _jobs_lock:
        job["status"] = "completed"
        job["completed_at"] = time.time()
        job["report"] = report

    return {
        "job_id": job_id,
        "status": "completed",
        "report": report,
    }


@app.post("/api/replay/benchmark/{job_id}/finalize")
async def api_finalize_benchmark(job_id: str) -> dict[str, Any]:
    """Finalize a benchmark job and compute the report."""
    return await api_benchmark_report(job_id)


# ---------------------------------------------------------------------------
# State / health
# ---------------------------------------------------------------------------

@app.get("/api/replay/state")
async def api_engine_state() -> dict[str, Any]:
    """Get the current engine state (dataset, episode, log length)."""
    with _engine_lock:
        return _engine.state_dict()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "replay-server"}


# ---------------------------------------------------------------------------
# Serve the viewer HTML
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_viewer() -> str:
    """Serve the replay viewer HTML."""
    viewer_path = STATIC_DIR / "replay_viewer.html"
    if not viewer_path.exists():
        return HTMLResponse(
            content="<h1>replay_viewer.html not found</h1><p>Create it under src/web/static/</p>",
            status_code=404,
        )
    return viewer_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    print(f"  ZTP Replay Server: http://localhost:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
