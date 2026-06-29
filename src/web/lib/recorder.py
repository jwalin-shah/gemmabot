"""Record live runs to disk so the demo can be replayed verbatim later.

Why this exists: at demo time we want to (a) prove a task actually succeeded
on hardware-equivalent physics, and (b) have a fallback that plays a known-
good run if the Cerebras API is hot or rate-limited. The recorder writes:

    runs/<task>_<utc>/
        manifest.json              ← {task, started_at, steps, success, ...}
        step_000.json              ← {intent, ee, gripper, objects, verdict, latency_ms}
        step_000_birdview.jpg
        step_000_frontview.jpg
        step_001.json
        ...

A replay reads the manifest + every step JSON in order and yields step
payloads shaped exactly like ``/api/step`` returns them — so the UI does not
need a separate code path for "replay" vs "live"; it just feeds the same
``renderStep(d)`` function from a different source.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.web.lib.verify import Verdict


# Stored under <repo>/runs/. The web UI also exposes this as /api/runs.
RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs"


@dataclass
class RunRecorder:
    """Writes one run\'s worth of step payloads + frames.

    The recorder is purely write-only. It never reads back its own outputs;
    that\'s ``Replay``\'s job. Keep step payloads JSON-clean — no numpy arrays.
    """

    task: str
    run_dir: Path = field(init=False)
    started_at: float = field(init=False)
    _steps: list[dict] = field(default_factory=list)
    _success: bool = False
    _last_verdict: dict | None = None

    def __post_init__(self) -> None:
        self.started_at = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.started_at))
        self.run_dir = RUNS_ROOT / f"{self.task}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def step(
        self,
        *,
        intent: dict,
        ee: list[float],
        gripper_open: bool,
        objects: dict[str, list[float]],
        verdict: Verdict,
        latency_ms: int,
        birdview_b64: str | None = None,
        frontview_b64: str | None = None,
    ) -> dict:
        """Persist one step and return the JSON payload that was written."""
        idx = len(self._steps)
        payload: dict[str, Any] = {
            "step": idx,
            "ts": time.time(),
            "intent": intent,
            "ee": ee,
            "gripper_open": gripper_open,
            "objects": objects,
            "verdict": verdict.to_json(),
            "latency_ms": latency_ms,
        }
        if birdview_b64:
            payload["birdview"] = _save_image(self.run_dir, idx, "birdview", birdview_b64)
        if frontview_b64:
            payload["frontview"] = _save_image(self.run_dir, idx, "frontview", frontview_b64)

        (self.run_dir / f"step_{idx:03d}.json").write_text(json.dumps(payload, indent=2))
        self._steps.append(payload)
        self._last_verdict = payload["verdict"]
        if verdict.success:
            self._success = True
        self._write_manifest()
        return payload

    def finalize(self, *, success: bool | None = None) -> dict:
        """Write the final manifest. Returns it. Safe to call multiple times."""
        if success is not None:
            self._success = success
        return self._write_manifest()

    # ------------------------------------------------------------------
    def _write_manifest(self) -> dict:
        manifest = {
            "run_id": self.run_id,
            "task": self.task,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "num_steps": len(self._steps),
            "success": self._success,
            "last_verdict": self._last_verdict,
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def _save_image(run_dir: Path, idx: int, name: str, b64_uri: str) -> str:
    """Strip the data: URI prefix and write the raw bytes to disk."""
    if "," in b64_uri:
        b64_uri = b64_uri.split(",", 1)[1]
    ext = "jpg"
    out = run_dir / f"step_{idx:03d}_{name}.{ext}"
    out.write_bytes(base64.b64decode(b64_uri))
    return out.name  # relative path inside the run dir


# ── Replay ────────────────────────────────────────────────────────────

def list_runs() -> list[dict]:
    """Index of every recorded run, newest first. Used by the UI replay tab."""
    if not RUNS_ROOT.exists():
        return []
    out = []
    for d in sorted(RUNS_ROOT.iterdir(), reverse=True):
        manifest = d / "manifest.json"
        if not d.is_dir() or not manifest.exists():
            continue
        try:
            out.append(json.loads(manifest.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def load_run(run_id: str) -> dict:
    """Return {manifest, steps[]} for the given run id.

    Steps are returned with their stored frame paths rebuilt as data URIs so
    the UI can render them without a second HTTP round-trip per frame.
    """
    run_dir = RUNS_ROOT / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    steps: list[dict] = []
    for step_file in sorted(run_dir.glob("step_*.json")):
        payload = json.loads(step_file.read_text())
        for cam in ("birdview", "frontview"):
            fname = payload.get(cam)
            if fname:
                img_path = run_dir / fname
                if img_path.exists():
                    payload[cam] = _to_data_uri(img_path)
        steps.append(payload)

    return {"manifest": manifest, "steps": steps}


def _to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{data}"
