# Loop-based runner for Claude Code overnight experiments

# How to start:
#   /loop "Run the overnight experiment plan from scripts/loop_runner.py using 5 parallel agents, saving all results to overnight_results/"

import subprocess, sys, json, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Result directory ─────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "overnight_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = RESULTS_DIR / "CHECKPOINT.json"
LOCK = RESULTS_DIR / ".running"
ERROR_LOG = RESULTS_DIR / "errors.log"

# ─── Available experiment scripts ────────────────────────────────────────
# Each one is self-contained: python scripts/exp_vision.py --runs N
# They use structured output (response_format) so temperature 0.0 works fine.
# They render scenes internally (no MuJoCo dependency) so they run anywhere.
# They write JSONL to overnight_results/ and print RESULT:{json} at the end.

EXPERIMENTS = {
    "vision": {
        "script": "scripts/exp_vision.py",
        "runs": 200,
        "description": "Pure zone identification from 4-camera composite image",
    },
    "perturb": {
        "script": "scripts/exp_perturb.py",
        "runs": 50,
        "description": "Object moves mid-task — measure re-acquisition ticks",
    },
    "multistep": {
        "script": "scripts/exp_multistep.py",
        "runs": 100,
        "description": "Multi-object sequential reasoning from one image",
    },
    "distractors": {
        "script": "scripts/exp_distractors.py",
        "runs": 50,
        "description": "Similar-colored distractor objects — is color enough?",
    },
    "size_variation": {
        "script": "scripts/exp_size_variation.py",
        "runs": 100,
        "description": "Different object sizes — does Gemma estimate Z from apparent size?",
    },
}

# ─── Load checkpoint ─────────────────────────────────────────────────────
def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"completed": {}, "started_at": None, "round": 0}

def save_checkpoint(state):
    state["_saved_at"] = time.time()
    CHECKPOINT.write_text(json.dumps(state, indent=2))

# ─── Run one experiment ──────────────────────────────────────────────────
def run_experiment(name, runs, temp=0.0):
    """Run an experiment and return structured results."""
    script = ROOT / EXPERIMENTS[name]["script"]
    if not script.exists():
        return {"error": f"Script not found: {script}", "runs": 0, "completed": False}

    result_dir = RESULTS_DIR / name
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / "results.jsonl"

    cmd = [
        sys.executable, str(script),
        "--runs", str(runs),
        "--temperature", str(temp),
        "--output", str(output_file),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    # Parse final RESULT line
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("RESULT:"):
            try:
                summary = json.loads(line[7:])
                summary["elapsed_s"] = round(elapsed, 1)
                return summary
            except json.JSONDecodeError:
                pass

    # If we didn't find RESULT, something went wrong
    return {
        "error": "No RESULT line found",
        "stdout": proc.stdout[-500:],
        "stderr": proc.stderr[-500:],
        "returncode": proc.returncode,
        "runs": 0,
        "completed": False,
        "elapsed_s": round(elapsed, 1),
    }

# ─── Aggregate all results ──────────────────────────────────────────────
def write_dashboard(state):
    """Write a dashboard HTML and summary."""
    results = state.get("completed", {})

    total = sum(r.get("runs", 0) for r in results.values())
    total_ok = sum(r.get("success_count", 0) for r in results.values())

    lines = ["# Overnight Results\n"]
    lines.append(f"- **Total runs**: {total}")
    lines.append(f"- **Successful**: {total_ok}")
    lines.append(f"- **Overall rate**: {total_ok/max(total,1):.0%}")
    lines.append(f"- **Round**: {state.get('round', 0)}")
    lines.append("")

    for name, r in results.items():
        if "error" in r:
            lines.append(f"## {name} — ❌ {r['error']}")
            continue
        lines.append(f"## {name}")
        lines.append(f"- Runs: {r.get('runs', 0)}")
        lines.append(f"- Success rate: {r.get('success_rate', 0):.0%}")
        lines.append(f"- Zone accuracy: {r.get('mean_zone_accuracy', 0):.0%}")
        lines.append(f"- P50 latency: {r.get('p50_latency_ms', 0)}ms")
        lines.append(f"- P95 latency: {r.get('p95_latency_ms', 0)}ms")
        lines.append(f"- Completed: {r.get('completed', False)}")
        lines.append("")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))

    # Also write a simple JSON blob for the dashboard
    dashboard = {
        "total_runs": total,
        "successful": total_ok,
        "overall_rate": round(total_ok/max(total,1), 4),
        "per_experiment": results,
    }
    (RESULTS_DIR / "dashboard.json").write_text(json.dumps(dashboard, indent=2))

# ─── Decide what to run next ────────────────────────────────────────────
import random

PENDING_EXPERIMENTS = [
    ("vision", "test with 200 runs, temp=0.0 — this is the baseline"),
    ("perturb", "test with 50 runs — how fast does it re-acquire?"),
    ("multistep", "test with 100 instructions — can it plan sequences?"),
    ("distractors", "test with 50 runs — does it confuse similar colors?"),
    ("size_variation", "test with 100 runs — can it estimate Z from size?"),
    ("vision", "retest with temp=0.3 and higher JPEG quality"),
    ("vision", "retest with only 2 objects instead of 3-5"),
    ("perturb", "retest with larger perturbations (full-zone jumps)"),
    ("multistep", "retest with 3-object sequences"),
    ("vision", "retest with monochrome objects — no color, only shape+zone"),
]

def pick_next_experiment(state):
    """Pick the most informative next experiment based on current results."""
    completed = state.get("completed", {})
    current_round = state.get("round", 0)

    # If we haven't started, begin with vision baseline
    if not completed or current_round == 0:
        return "vision", 200

    # Check if baseline is done
    vision = completed.get("vision", {})
    if not vision.get("completed"):
        return "vision", 200

    # If baseline accuracy is low, don't waste budget on harder tests
    if vision.get("mean_zone_accuracy", 0) < 0.4:
        return None, 0  # signal: baseline failed

    # Cycle through remaining experiments in priority order
    priority = ["perturb", "multistep", "distractors", "size_variation"]
    for name in priority:
        exp = completed.get(name, {})
        if not exp.get("completed"):
            runs = EXPERIMENTS.get(name, {}).get("runs", 50)
            return name, runs

    # All experiments done — start variation sweeps
    sweep_round = current_round - len(priority) - 1
    sweeps = [
        ("vision", 50, "retest at temp=0.3"),
        ("vision", 50, "retest at temp=0.8"),
        ("vision", 50, "retest with JPEG quality=95"),
        ("vision", 50, "retest with monochrome objects"),
    ]
    if sweep_round < len(sweeps):
        return sweeps[sweep_round][0], sweeps[sweep_round][1]

    return None, 0  # everything done


def main():
    state = load_checkpoint()
    if state.get("started_at") is None:
        state["started_at"] = time.time()

    # Pick next experiment
    name, runs = pick_next_experiment(state)
    if name is None:
        print("All experiments complete! Writing final dashboard.")
        write_dashboard(state)
        state["finished_at"] = time.time()
        save_checkpoint(state)
        return

    # Check if already completed
    completed = state.get("completed", {})
    if name in completed and completed[name].get("completed"):
        print(f"{name} already completed, skipping.")
        state["round"] = state.get("round", 0) + 1
        save_checkpoint(state)
        return

    print(f"\n{'='*60}")
    print(f"  ROUND {state.get('round', 0) + 1}: {name} — {runs} runs")
    print(f"{'='*60}")

    # Actually run it via Bash in the loop
    import subprocess
    script = ROOT / EXPERIMENTS[name]["script"]
    output_file = RESULTS_DIR / name / "results.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--runs", str(runs),
        "--temperature", "0.0",
        "--output", str(output_file),
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0

        # Parse result
        summary = None
        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                summary = json.loads(line[7:])
                summary["elapsed_s"] = round(elapsed, 1)
                summary["script"] = str(script)
                break

        if summary:
            completed[name] = summary
            state["round"] = state.get("round", 0) + 1
            save_checkpoint(state)
            write_dashboard(state)

            print(f"\n  ✓ {name}: {summary.get('runs', 0)} runs, "
                  f"accuracy {summary.get('mean_zone_accuracy', 0):.0%}, "
                  f"P95 {summary.get('p95_latency_ms', 0)}ms")
        else:
            print(f"  ⚠️ {name}: no RESULT line. stdout: {result.stdout[-300:]}")
            completed[name] = {
                "error": "no result",
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:],
            }
            state["round"] = state.get("round", 0) + 1
            save_checkpoint(state)

    except subprocess.TimeoutExpired:
        print(f"  ❌ {name}: timed out after 3600s")
        completed[name] = {"error": "timeout", "runs": 0}
        save_checkpoint(state)
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        completed[name] = {"error": str(e)[:200]}
        save_checkpoint(state)


# ─── When run directly: print what to do next ───────────────────────────

if __name__ == "__main__":
    state = load_checkpoint()
    name, runs = pick_next_experiment(state)
    if name:
        print(f"Next: {name} ({runs} runs)")
        print(f"Run: python scripts/loop_runner.py")
    else:
        print("All experiments complete!")
        print(f"See: overnight_results/summary.md")
