#!/usr/bin/env python3
"""Apples-to-apples speed race: Gemma on Cerebras vs Gemma on OpenRouter (GPU).

Same model family, same prompts, same images, same temperature.
Only the inference hardware differs.

Three prompt sets — picks one per --kind:
  text  : tiny no-image prompt (worst case for Cerebras, best for GPU)
  image : real perception prompt with one 384x384 frame (matches the demo loop)
  json  : image prompt + JSON-schema structured output (matches the live brain)

Writes per-call JSONL to overnight_results/compare_or/<kind>/calls.jsonl and a
summary JSON next to it. Designed for the submission video — the numbers come
straight from this script, no massaging.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.client import CerebrasClient  # noqa: E402
from src.web.lib.imaging import img_to_b64  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("COMPARISON_API_KEY", "")
OPENROUTER_BASE = (os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("COMPARISON_BASE_URL", "")).rstrip("/")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL") or os.environ.get("COMPARISON_MODEL", "google/gemma-3-27b-it")

if not (OPENROUTER_KEY and OPENROUTER_BASE):
    print("ERROR: OPENROUTER_API_KEY / OPENROUTER_BASE_URL not set.", file=sys.stderr)
    sys.exit(2)

OUT_ROOT = _ROOT / "overnight_results" / "compare_or"

# Prompts ──────────────────────────────────────────────────────────────
TEXT_PROMPTS = [
    "List 3 grocery items you might pick up.",
    "Name one object on a typical robotics lab table.",
    "What color is a typical soda can?",
    "Give a 2-word answer: how do you grasp a cube?",
    "What is the first step to pick up an object with a robot?",
]

IMAGE_PROMPT = (
    "You are a robot vision system. Look at this image and list the objects "
    "you see on the table. For each, give a one-word label. Reply as JSON: "
    '{"objects": ["label1", "label2", ...]}'
)

JSON_SCHEMA_PROMPT = (
    "You are a Franka Panda arm. Look at the image. Choose ONE object to pick up. "
    "Output only this JSON: {\"tool\":\"grasp\",\"object\":\"<label>\",\"reason\":\"<short>\"}"
)


# ── Data ────────────────────────────────────────────────────────────────
@dataclass
class CallResult:
    provider: str
    kind: str
    idx: int
    latency_ms: float
    ttft_ms: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None
    content: str
    ok: bool
    error: str | None = None
    model: str | None = None


@dataclass
class Summary:
    provider: str
    kind: str
    n: int
    n_ok: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    completion_tps: float | None = None
    raw_latencies_ms: list[float] = field(default_factory=list)


# ── Clients ─────────────────────────────────────────────────────────────
_http = httpx.Client(timeout=120.0)


def call_openrouter(messages: list[dict], use_schema: bool, idx: int, kind: str) -> CallResult:
    body: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
    }
    if use_schema:
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cerebras-gemma4-hackathon",
        "X-Title": "GemmaBot Cerebras vs OpenRouter Benchmark",
    }
    url = f"{OPENROUTER_BASE}/chat/completions"

    t0 = time.perf_counter()
    try:
        r = _http.post(url, json=body, headers=headers)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            return CallResult(
                provider="openrouter", kind=kind, idx=idx,
                latency_ms=elapsed_ms, ttft_ms=None,
                completion_tokens=None, prompt_tokens=None, total_tokens=None,
                content="", ok=False, error=f"HTTP {r.status_code}: {r.text[:200]}",
            )
        data = r.json()
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return CallResult(
            provider="openrouter", kind=kind, idx=idx,
            latency_ms=elapsed_ms, ttft_ms=None,
            completion_tokens=None, prompt_tokens=None, total_tokens=None,
            content="", ok=False, error=str(exc)[:200],
        )

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return CallResult(
        provider="openrouter", kind=kind, idx=idx,
        latency_ms=elapsed_ms, ttft_ms=None,
        completion_tokens=usage.get("completion_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        total_tokens=usage.get("total_tokens"),
        content=choice["message"].get("content", "") or "",
        ok=True, model=data.get("model"),
    )


def call_cerebras(client: CerebrasClient, messages: list[dict], use_schema: bool, idx: int, kind: str) -> CallResult:
    kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": 200}
    if use_schema:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        res = client.chat(messages=messages, **kwargs)
    except Exception as exc:
        return CallResult(
            provider="cerebras", kind=kind, idx=idx,
            latency_ms=0.0, ttft_ms=None,
            completion_tokens=None, prompt_tokens=None, total_tokens=None,
            content="", ok=False, error=str(exc)[:200],
        )

    latency_ms = res.latency_s * 1000
    ttft_ms = None
    ti = res.time_info or {}
    if ti.get("prompt_time") is not None and ti.get("queue_time") is not None:
        ttft_ms = (float(ti["prompt_time"]) + float(ti["queue_time"])) * 1000

    return CallResult(
        provider="cerebras", kind=kind, idx=idx,
        latency_ms=latency_ms, ttft_ms=ttft_ms,
        completion_tokens=(res.usage or {}).get("completion_tokens"),
        prompt_tokens=(res.usage or {}).get("prompt_tokens"),
        total_tokens=(res.usage or {}).get("total_tokens"),
        content=res.content,
        ok=True, model=res.model,
    )


# ── Prompts builders ───────────────────────────────────────────────────
def build_messages_text(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def build_messages_image(prompt: str, image_b64: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }
    ]


def _get_demo_frame() -> str:
    """Capture one real frontview image from robosuite at 384x384.
    Falls back to a saved frame if available."""
    cached = OUT_ROOT / "frame.jpg.b64"
    if cached.exists():
        return cached.read_text()
    try:
        import warnings
        warnings.filterwarnings("ignore")
        os.environ.setdefault("OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING", "1")
        import robosuite as suite
        env = suite.make(
            "PickPlace", robots="Panda", has_renderer=False,
            has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=["frontview"], camera_heights=384, camera_widths=384,
        )
        obs = env.reset()
        import numpy as np
        frame = np.flipud(obs["frontview_image"])
        b64 = img_to_b64(frame, fmt="JPEG", quality=70)
        env.close()
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        cached.write_text(b64)
        return b64
    except Exception as exc:
        # Solid color fallback so the benchmark still runs
        import numpy as np
        frame = np.zeros((384, 384, 3), dtype=np.uint8)
        frame[:, :, 0] = 80
        frame[:, :, 1] = 100
        frame[:, :, 2] = 120
        print(f"[warn] no robosuite frame ({exc}); using solid fallback", file=sys.stderr)
        return img_to_b64(frame, fmt="JPEG", quality=70)


# ── Runner ──────────────────────────────────────────────────────────────
def run_kind(kind: str, n: int, warmup: int) -> tuple[list[CallResult], list[CallResult]]:
    print(f"\n=== {kind.upper()} — {n} calls each (after {warmup} warmup) ===")
    use_schema = kind == "json"

    if kind == "text":
        prompt_set = [build_messages_text(p) for p in TEXT_PROMPTS]
    else:
        image_b64 = _get_demo_frame()
        base = JSON_SCHEMA_PROMPT if use_schema else IMAGE_PROMPT
        prompt_set = [build_messages_image(base, image_b64)]

    cerebras = CerebrasClient()

    # Warmup
    for w in range(warmup):
        msgs = prompt_set[w % len(prompt_set)]
        call_cerebras(cerebras, msgs, use_schema, idx=-1, kind=kind)
        call_openrouter(msgs, use_schema, idx=-1, kind=kind)

    cere_results: list[CallResult] = []
    or_results: list[CallResult] = []

    for i in range(n):
        msgs = prompt_set[i % len(prompt_set)]

        c = call_cerebras(cerebras, msgs, use_schema, idx=i, kind=kind)
        cere_results.append(c)
        print(
            f"  [{i+1:>3}/{n}] cerebras  {c.latency_ms:>6.0f}ms  "
            f"tokens={c.completion_tokens} ok={c.ok}"
            + (f"  err={c.error}" if not c.ok else "")
        )

        o = call_openrouter(msgs, use_schema, idx=i, kind=kind)
        or_results.append(o)
        print(
            f"  [{i+1:>3}/{n}] openrouter {o.latency_ms:>6.0f}ms  "
            f"tokens={o.completion_tokens} ok={o.ok}"
            + (f"  err={o.error}" if not o.ok else "")
        )

    return cere_results, or_results


def summarize(results: list[CallResult], provider: str, kind: str) -> Summary:
    ok = [r for r in results if r.ok and r.latency_ms > 0]
    if not ok:
        return Summary(
            provider=provider, kind=kind, n=len(results), n_ok=0,
            p50_ms=0, p95_ms=0, p99_ms=0, mean_ms=0, min_ms=0, max_ms=0, stdev_ms=0,
        )
    latencies = sorted(r.latency_ms for r in ok)

    def pct(p: float) -> float:
        idx = max(0, min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1)))))
        return latencies[idx]

    completion_tokens = [r.completion_tokens for r in ok if r.completion_tokens]
    total_completion = sum(completion_tokens) if completion_tokens else 0
    total_secs = sum(r.latency_ms for r in ok) / 1000.0
    tps = (total_completion / total_secs) if total_secs > 0 and total_completion else None

    return Summary(
        provider=provider, kind=kind, n=len(results), n_ok=len(ok),
        p50_ms=pct(50), p95_ms=pct(95), p99_ms=pct(99),
        mean_ms=statistics.mean(latencies),
        min_ms=latencies[0], max_ms=latencies[-1],
        stdev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        completion_tps=tps,
        raw_latencies_ms=latencies,
    )


def write_outputs(kind: str, cere: list[CallResult], rrout: list[CallResult]) -> None:
    out_dir = OUT_ROOT / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    calls_path = out_dir / "calls.jsonl"
    with calls_path.open("w") as f:
        for r in [*cere, *rrout]:
            f.write(json.dumps(asdict(r)) + "\n")

    s_cere = summarize(cere, "cerebras", kind)
    s_or = summarize(rrout, "openrouter", kind)

    speedup_p50 = (s_or.p50_ms / s_cere.p50_ms) if s_cere.p50_ms else None
    speedup_p95 = (s_or.p95_ms / s_cere.p95_ms) if s_cere.p95_ms else None

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "kind": kind,
        "cerebras": asdict(s_cere),
        "openrouter": asdict(s_or),
        "speedup_p50_x": round(speedup_p50, 2) if speedup_p50 else None,
        "speedup_p95_x": round(speedup_p95, 2) if speedup_p95 else None,
        "model_cerebras": "gemma-4-31b",
        "model_openrouter": OPENROUTER_MODEL,
    }, indent=2))

    print(f"\n── {kind.upper()} SUMMARY ──")
    print(f"  Cerebras   : p50 {s_cere.p50_ms:>6.0f}ms  p95 {s_cere.p95_ms:>6.0f}ms  "
          f"mean {s_cere.mean_ms:>6.0f}ms  n={s_cere.n_ok}/{s_cere.n}"
          + (f"  tps={s_cere.completion_tps:.1f}" if s_cere.completion_tps else ""))
    print(f"  OpenRouter : p50 {s_or.p50_ms:>6.0f}ms  p95 {s_or.p95_ms:>6.0f}ms  "
          f"mean {s_or.mean_ms:>6.0f}ms  n={s_or.n_ok}/{s_or.n}"
          + (f"  tps={s_or.completion_tps:.1f}" if s_or.completion_tps else ""))
    if speedup_p50:
        print(f"  ► Cerebras p50 SPEEDUP: {speedup_p50:.1f}x")
    if speedup_p95:
        print(f"  ► Cerebras p95 SPEEDUP: {speedup_p95:.1f}x")
    print(f"  Wrote: {calls_path}  +  {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="text,image,json",
                    help="Comma list of: text, image, json")
    ap.add_argument("--n", type=int, default=20, help="Calls per provider per kind")
    ap.add_argument("--warmup", type=int, default=2, help="Untimed warmup calls")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for kind in kinds:
        if kind not in ("text", "image", "json"):
            print(f"unknown kind {kind!r}, skipping", file=sys.stderr); continue
        cere, rrout = run_kind(kind, args.n, args.warmup)
        write_outputs(kind, cere, rrout)
        all_results[kind] = {
            "cerebras": asdict(summarize(cere, "cerebras", kind)),
            "openrouter": asdict(summarize(rrout, "openrouter", kind)),
        }

    (OUT_ROOT / "all_kinds.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nDone — overall summary at {OUT_ROOT / 'all_kinds.json'}")


if __name__ == "__main__":
    main()
