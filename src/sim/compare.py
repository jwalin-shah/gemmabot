"""Stage 2: Cerebras vs GPU-provider speed race on the same reactive task.

Same Gemma 4 31B on both sides — only the inference hardware differs. Each
backend drives its own copy of the world at its *real* measured latency; we
perturb both at the same virtual wall-clock moment (drag the cup) and measure
how fast each re-acquires it and how many decisions each gets to make.

Usage:
    uv run python -m src.sim.compare --throttle           # stand-in GPU (no key needed)
    uv run python -m src.sim.compare                       # real GPU via COMPARISON_* in .env
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import httpx

from src.client import CerebrasClient, InferenceResult
from src.config import (
    COMPARISON_API_KEY,
    COMPARISON_BASE_URL,
    COMPARISON_MODEL,
    PROJECT_ROOT,
)
from src.sim.brain import RobotBrain
from src.sim.loop import ReactiveLoop
from src.sim.run_sim import INSTRUCTION, build_world
from src.sim.skills import REACH

PERTURB_POS = (520.0, 130.0)  # where the cup gets dragged to (Zone C)


class ComparisonClient:
    """OpenAI-compatible client for the GPU baseline (TokenRouter / OpenRouter /
    any /v1 chat-completions endpoint). Mirrors CerebrasClient.image_chat."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "") -> None:
        self.api_key = api_key or COMPARISON_API_KEY
        self.base_url = (base_url or COMPARISON_BASE_URL).rstrip("/")
        self.model = model or COMPARISON_MODEL
        if not (self.api_key and self.base_url and self.model):
            raise RuntimeError(
                "GPU baseline not configured. Set COMPARISON_API_KEY, "
                "COMPARISON_BASE_URL and COMPARISON_MODEL in .env."
            )
        self._http = httpx.Client(timeout=90.0)

    def image_chat(self, prompt: str, image_b64: str, system_prompt: str | None = None, **kwargs):
        img = {"type": "image_url", "image_url": {"url": image_b64}}
        with_sys = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": [{"type": "text", "text": prompt}, img]}
        ]
        folded_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        folded = [{"role": "user", "content": [{"type": "text", "text": folded_text}, img]}]

        body = {
            "model": self.model,
            "messages": with_sys,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 350),
        }
        if "response_format" in kwargs:
            body["response_format"] = kwargs["response_format"]

        start = time.perf_counter()
        data = self._post(body, folded)
        elapsed = time.perf_counter() - start
        content = (data["choices"][0]["message"].get("content") or "") if data.get("choices") else ""
        return InferenceResult(content=content, model=self.model, latency_s=elapsed)

    def _post(self, body: dict, folded_msgs: list) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        r = self._http.post(url, json=body, headers=headers)
        # Some Gemma deployments reject system messages -> fold into the user turn.
        if r.status_code >= 400:
            b2 = {**body, "messages": folded_msgs}
            r = self._http.post(url, json=b2, headers=headers)
            # ...and some don't support strict json_schema -> drop it.
            if r.status_code >= 400 and "response_format" in body:
                b3 = {k: v for k, v in b2.items() if k != "response_format"}
                r = self._http.post(url, json=b3, headers=headers)
        r.raise_for_status()
        return r.json()


class ThrottledClient:
    """Wraps a real client and adds latency to stand in for a slow GPU while you
    wire up the real provider. Lets you see the race before the key lands."""

    def __init__(self, inner, extra_latency_s: float = 1.7) -> None:
        self._inner = inner
        self._extra = extra_latency_s

    def image_chat(self, *args, **kwargs):
        res = self._inner.image_chat(*args, **kwargs)
        time.sleep(self._extra)
        return res


def run_backend(label: str, brain, duration_s: float, perturb_at_s: float, out_dir: Path) -> dict:
    world = build_world()
    loop = ReactiveLoop(world, brain)
    loop.set_instruction(INSTRUCTION)

    elapsed = 0.0
    latency_sum = 0.0
    perturbed = False
    reacquire: float | None = None
    i = 0

    while elapsed < duration_s:
        if not perturbed and elapsed >= perturb_at_s:
            cup = world.get("cracked_cup")
            cup.x, cup.y = PERTURB_POS
            perturbed = True

        result = loop.tick()
        i += 1
        lat = result.decision.latency_ms / 1000.0
        elapsed += lat
        latency_sum += lat
        world.render().save(out_dir / f"{label}_{i:02d}.png")

        g = world.gripper
        if perturbed and reacquire is None:
            on_cup = math.hypot(g.x - PERTURB_POS[0], g.y - PERTURB_POS[1]) <= REACH + 2
            if on_cup or world.gripper.holding == "cracked_cup":
                reacquire = elapsed - perturb_at_s

    return {
        "label": label,
        "decisions": i,
        "hz": i / duration_s,
        "avg_ms": (latency_sum / i * 1000) if i else 0.0,
        "reacquire_s": reacquire,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--throttle", action="store_true", help="use throttled Cerebras as a GPU stand-in")
    parser.add_argument("--duration", type=float, default=12.0, help="virtual wall-clock seconds per side")
    parser.add_argument("--perturb", type=float, default=4.0, help="when to drag the cup (s)")
    args = parser.parse_args()

    out_dir = Path(PROJECT_ROOT) / "runs" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = RobotBrain(CerebrasClient())
    if args.throttle:
        slow = RobotBrain(ThrottledClient(CerebrasClient()))
        slow_name = "GPU (throttled stand-in)"
    else:
        slow = RobotBrain(ComparisonClient())
        slow_name = f"GPU ({COMPARISON_MODEL})"

    print(f"Task: {INSTRUCTION}")
    print(f"Window: {args.duration:.0f}s  |  cup dragged at t={args.perturb:.0f}s\n{'=' * 72}")

    fast_m = run_backend("cerebras", fast, args.duration, args.perturb, out_dir)
    slow_m = run_backend("gpu", slow, args.duration, args.perturb, out_dir)

    def fmt(m: dict, name: str) -> str:
        ra = f"{m['reacquire_s']:.2f}s" if m["reacquire_s"] is not None else "never"
        return (f"{name:28s} | {m['decisions']:3d} decisions | {m['hz']:4.1f} Hz | "
                f"avg {m['avg_ms']:5.0f}ms | re-acquired cup in {ra}")

    print(fmt(fast_m, "Cerebras (Gemma 4 31B)"))
    print(fmt(slow_m, slow_name))
    behind = fast_m["decisions"] - slow_m["decisions"]
    print(f"{'=' * 72}\nHeadline: Cerebras made {behind} more decisions in the same window "
          f"({fast_m['decisions']} vs {slow_m['decisions']}).")
    print(f"Frames in {out_dir}")


if __name__ == "__main__":
    main()
