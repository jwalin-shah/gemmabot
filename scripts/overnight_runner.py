"""Overnight experiment runner — crash-proof, subagent-based, self-checkpointing.

Each worker runs as an isolated subprocess with its own rate limiter.
The coordinator tracks progress and saves checkpoints every 10 calls.

Usage:
    python scripts/overnight_runner.py                  # full run
    python scripts/overnight_runner.py --budget 200      # max 200 API calls total
    python scripts/overnight_runner.py --resume          # resume from checkpoint

Bash wrapper (auto-restart on crash):
    bash scripts/overnight.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.client import CerebrasClient

# ─── Paths ───────────────────────────────────────────────────────────────
RESULTS_DIR = _PROJECT_ROOT / "overnight_results"
CHECKPOINT_PATH = RESULTS_DIR / "CHECKPOINT.json"
EXPERIMENTS_DIR = _PROJECT_ROOT / "scripts" / "experiments"

# ─── Experiment registry ─────────────────────────────────────────────────
EXPERIMENTS = {
    "vision": {
        "module": "worker_vision.py",
        "description": "Pure vision: zone identification accuracy with 4-camera composite",
        "default_n": 200,
    },
    "perturb": {
        "module": "worker_perturb.py",
        "description": "Perturbation: move objects mid-task, measure re-acquisition",
        "default_n": 50,
    },
    "multistep": {
        "module": "worker_multistep.py",
        "description": "Multi-step: plan two-object sequences from one image",
        "default_n": 100,
    },
}


def get_worker_script(name: str) -> Path:
    return EXPERIMENTS_DIR / EXPERIMENTS[name]["module"]


# ═══════════════════════════════════════════════════════════════════════════
# Coordinator
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        print(f"  ✓ Resumed checkpoint: {data.get('_saved_at', 'unknown')}")
        return data
    return {"_experiments": {}, "_created_at": datetime.utcnow().isoformat()}


def save_checkpoint(state: dict, completed: int, total: int, budget: int) -> None:
    state["_completed"] = completed
    state["_total"] = total
    state["_budget"] = budget
    state["_saved_at"] = datetime.utcnow().isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2, default=str))
    print(f"  💾 Checkpoint saved ({completed}/{total} runs)")


def run_worker(
    name: str,
    count: int,
    budget_remaining: int,
    resume_state: dict,
) -> dict:
    """Run ONE experiment as a subprocess worker."""
    script = get_worker_script(name)
    if not script.exists():
        print(f"  ⚠️  Worker script not found: {script} — skipping {name}")
        return {"runs": 0, "completed": False, "errors": ["script not found"]}

    worker_state = resume_state.get(name, {})
    completed_runs = worker_state.get("completed_runs", 0)
    remaining = max(0, count - completed_runs)

    if remaining <= 0:
        print(f"  ✓ {name}: already completed ({completed_runs}/{count})")
        return {"runs": completed_runs, "completed": True, "errors": []}

    calls_for_this_worker = min(remaining, budget_remaining)
    if calls_for_this_worker <= 0:
        print(f"  ⏸ {name}: budget exhausted")
        return {"runs": completed_runs, "completed": False, "errors": ["budget"]}

    print(f"\n  ▶ Starting {name}: {calls_for_this_worker} calls (completed {completed_runs}/{count})")
    sys.stdout.flush()

    env = os.environ.copy()
    env["WORKER_RUNS"] = str(calls_for_this_worker)
    env["WORKER_RESUME"] = str(completed_runs)
    env["WORKER_OUTPUT"] = str(RESULTS_DIR / f"worker_{name}.jsonl")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hours max per worker
    )
    elapsed = time.time() - t0

    # Parse worker output for summary
    errors = []
    try:
        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                summary = json.loads(line[7:])
                return {**summary, "_elapsed_s": elapsed}
    except Exception:
        pass

    if result.returncode != 0:
        errors.append(f"exit={result.returncode}")
        if result.stderr:
            errors.append(result.stderr[-500:])

    return {
        "runs": calls_for_this_worker,
        "completed": result.returncode == 0,
        "errors": errors,
        "_elapsed_s": elapsed,
        "_stdout_tail": result.stdout[-300:] if result.stdout else "",
    }


def write_summary(results: dict, budget: int) -> None:
    """Write summary.md + index.html to results dir."""
    summary = []
    summary.append("# Overnight Results\n")
    summary.append(f"- **Run**: {datetime.utcnow().isoformat()}")
    summary.append(f"- **Budget**: {budget} API calls")
    summary.append(f"- **Total completed**: {results.get('_total_completed', 0)}")
    summary.append("")

    for name, data in results.get("_experiments", {}).items():
        if not data:
            continue
        summary.append(f"## {name}")
        summary.append(f"- Runs: {data.get('runs', 0)}")
        summary.append(f"- Completed: {data.get('completed', False)}")
        summary.append(f"- Errors: {data.get('errors', [])}")
        summary.append(f"- Time: {data.get('_elapsed_s', 0):.0f}s")
        summary.append("")

    (RESULTS_DIR / "summary.md").write_text("\n".join(summary))
    print(f"  📝 Summary written to overnight_results/summary.md")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Overnight experiment runner")
    parser.add_argument("--budget", type=int, default=1000, help="Max total API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS),
                        help="Which experiments to run")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    title = "=" * 65
    print(f"\n{title}")
    print("  🧪  OVERNIGHT EXPERIMENT RUNNER")
    print(f"{title}")
    print(f"  Budget:   {args.budget} API calls")
    print(f"  Workers:  {', '.join(args.workers)}")
    print(f"  Resume:   {args.resume}")
    print(f"  Output:   {RESULTS_DIR}")
    print(f"{title}\n")

    # Verify API key
    if not os.environ.get("CEREBRAS_API_KEY"):
        print("  ❌ CEREBRAS_API_KEY not set")
        sys.exit(1)

    # Quick API connectivity test
    try:
        t0 = time.time()
        client = CerebrasClient()
        test = client.chat(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0.0,
        )
        api_ms = (time.time() - t0) * 1000
        print(f"  ✅ API connectivity OK ({api_ms:.0f}ms)\n")
    except Exception as e:
        print(f"  ❌ API connectivity FAILED: {e}")
        sys.exit(1)

    # Load or init checkpoint
    state = load_checkpoint() if args.resume else {"_experiments": {}, "_created_at": datetime.utcnow().isoformat()}
    budget_remaining = args.budget
    total_completed = 0

    for name in args.workers:
        if budget_remaining <= 0:
            print(f"  ⏸ Budget exhausted after {name}")
            break

        n = EXPERIMENTS[name]["default_n"]
        worker_result = run_worker(name, n, budget_remaining, state.get("_experiments", {}))
        state["_experiments"][name] = worker_result
        runs_done = worker_result.get("runs", 0)
        total_completed += runs_done
        budget_remaining -= runs_done
        completed = worker_result.get("completed", False)
        icon = "✓" if completed else "⚠️"
        errors = worker_result.get("errors", [])
        elapsed = worker_result.get("_elapsed_s", 0)
        print(f"\n  {icon} {name}: {runs_done} runs in {elapsed:.0f}s {'| ERRORS: ' + str(errors) if errors else ''}")

        save_checkpoint(state, total_completed, sum(EXPERIMENTS[n2]["default_n"] for n2 in args.workers), args.budget)

    # Final summary
    state["_total_completed"] = total_completed
    state["_finished_at"] = datetime.utcnow().isoformat()
    save_checkpoint(state, total_completed, sum(EXPERIMENTS[n]["default_n"] for n in args.workers), args.budget)
    write_summary(state, args.budget)

    print(f"\n{'=' * 65}")
    print(f"  🏁 OVERNIGHT COMPLETE: {total_completed} total runs")
    print(f"  📊 Results in overnight_results/")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
