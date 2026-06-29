#!/usr/bin/env python3
"""Overnight experiment runner: tests each config independently."""
import json, os, sys, time, random, statistics, shutil
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING"] = "1"

from src.web.lib.sim import PandaSim
from src.web.lib.brain import GemmaBrain, HistoryItem
from src.web.lib.executor import MotionExecutor, Disturbance
from src.web.lib.verify import verify, env_success
from src.web.lib.tasks import TASKS, get as get_task
from src.client import CerebrasClient

MAX_STEPS = 20
N_RUNS = 3  # 3 per cell to keep overnight feasible
TASK_KEYS = ["pick_can", "pick_milk", "pick_bread", "pick_cereal", "lift_cube", "stack_cubes"]
DISTURB_LEVELS = [0, 1, 2]
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")

# ── Data carriers ──
@dataclass
class RunRecord:
    task: str; config: str; max_steps: int; max_disturbances: int
    run_idx: int; success: bool; steps_used: int; disturbances_applied: int
    failure_stage: str; avg_latency_ms: float; max_latency_ms: float
    p50_latency_ms: float; p95_latency_ms: float; started_at: float; ended_at: float

@dataclass
class CellAggregate:
    config: str; task: str; max_disturbances: int; n_runs: int
    success_rate: float; n_success: int; mean_steps: float
    mean_avg_latency: float; mean_max_latency: float; median_avg_latency: float; p95_avg_latency: float
    failure_distribution: dict[str, int]

FailureStage = Literal["success", "exhausted_steps", "early_done", "tool_error", "hallucinated_coords"]

def classify_failure(success, steps_used, max_steps, hallucinated, early_done, tool_error):
    if tool_error: return "tool_error"
    if hallucinated: return "hallucinated_coords"
    if success: return "success"
    if early_done: return "early_done"
    return "exhausted_steps"

def run_single(task_key, max_steps, max_disturbances, disturb_every, run_idx, client, config):
    spec = get_task(task_key)
    sim = PandaSim(spec)
    brain = GemmaBrain(client=client)
    executor = MotionExecutor(sim)
    disturber = Disturbance(sim)
    history = []
    snap = sim.reset(); executor.seed_from(snap)
    prev_snap = None; disturbances_applied = 0
    step_records = []; latencies = []
    early_done = False; tool_error = False; hallucinated = False
    started_at = time.time()
    
    # Random interval setup
    next_disturb = random.randint(2, 5)
    
    for step_idx in range(1, max_steps + 1):
        snap_before = sim.snapshot()
        if disturbances_applied < max_disturbances and step_idx > 1 and step_idx >= next_disturb:
            obj = spec.target_object or (list(snap_before.objects.keys())[0] if snap_before.objects else None)
            if obj and obj in snap_before.objects:
                dx = random.uniform(-0.08, 0.08); dy = random.uniform(-0.08, 0.08)
                disturber.move_object(obj, dx=dx, dy=dy)
                disturbances_applied += 1
                next_disturb = step_idx + random.randint(2, 5)
                snap_before = sim.snapshot()
        
        intent = brain.think(spec.description, snap_before, history, spec, prev_snap)
        latencies.append(intent.latency_ms)
        
        z = intent.params.get("z", 0.85); x = intent.params.get("x", 0.0); y = intent.params.get("y", 0.0)
        if (z is not None and z < 0.5) or (x is not None and abs(x) > 1.0) or (y is not None and abs(y) > 1.0):
            hallucinated = True
        
        if intent.tool == "done":
            early_done = True; break
        
        try:
            result = executor.execute_tool(snap_before, intent.tool, intent.params)
        except Exception:
            tool_error = True
            result = executor.execute(snap_before, {
                "target_x": intent.params.get("x", snap_before.ee_pos[0]),
                "target_y": intent.params.get("y", snap_before.ee_pos[1]),
                "target_z": intent.params.get("z", snap_before.ee_pos[2]),
            }, "hold")
        
        final = result.final_snapshot
        v = verify(snap_before, final, spec)
        if env_success(final, spec): v.success = True
        
        history.append(HistoryItem(step=step_idx, tool=intent.tool,
            tool_params=str(intent.params), reasoning=intent.reasoning,
            ee_x=float(final.ee_pos[0]), ee_y=float(final.ee_pos[1]),
            ee_z=float(final.ee_pos[2]), gripper_open=final.gripper_open,
            verdict_note=v.notes))
        prev_snap = snap_before
        step_records.append({"step": step_idx, "tool": intent.tool, "latency_ms": intent.latency_ms,
            "reached": v.reached, "grasped": v.grasped, "lifted": v.lifted, "placed": v.placed,
            "success": v.success, "distance_xy": round(v.distance_xy, 4), "distance_z": round(v.distance_z, 4)})
        if v.success: break
    
    ended_at = time.time()
    steps_used = len(step_records)
    success = step_records[-1]["success"] if step_records else False
    fs = classify_failure(success, steps_used, max_steps, hallucinated, early_done, tool_error)
    
    stats = dict(avg=statistics.mean(latencies) if latencies else 0.0,
                 max=max(latencies) if latencies else 0.0,
                 p50=statistics.median(latencies) if latencies else 0.0,
                 p95=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0)
    
    return RunRecord(task=task_key, config=config, max_steps=max_steps,
        max_disturbances=max_disturbances, run_idx=run_idx,
        success=success, steps_used=steps_used, disturbances_applied=disturbances_applied,
        failure_stage=fs, avg_latency_ms=stats["avg"], max_latency_ms=stats["max"],
        p50_latency_ms=stats["p50"], p95_latency_ms=stats["p95"],
        started_at=started_at, ended_at=ended_at)

def aggregate_cell(runs):
    task = runs[0].task; n_dist = runs[0].max_disturbances; config = runs[0].config
    avg_latencies = [r.avg_latency_ms for r in runs]
    fail_dist = defaultdict(int)
    for r in runs: fail_dist[r.failure_stage] += 1
    return CellAggregate(config=config, task=task, max_disturbances=n_dist, n_runs=len(runs),
        success_rate=sum(1 for r in runs if r.success)/len(runs),
        n_success=sum(1 for r in runs if r.success),
        mean_steps=statistics.mean([r.steps_used for r in runs]),
        mean_avg_latency=statistics.mean(avg_latencies),
        mean_max_latency=statistics.mean([r.max_latency_ms for r in runs]),
        median_avg_latency=statistics.median(avg_latencies),
        p95_avg_latency=sorted(avg_latencies)[int(len(avg_latencies)*0.95)] if len(avg_latencies)>=3 else avg_latencies[-1],
        failure_distribution=dict(fail_dist))

def run_config(config_name, brain_patches=None, executor_patches=None):
    """Run full benchmark for a config. Optionally apply patches first."""
    print(f"\n{'='*60}")
    print(f"  CONFIG: {config_name}")
    print(f"{'='*60}")
    
    # Apply patches if any
    if brain_patches:
        with open('src/web/lib/brain.py') as f: orig_brain = f.read()
        for old, new in brain_patches:
            with open('src/web/lib/brain.py') as f: c = f.read()
            c = c.replace(old, new)
            with open('src/web/lib/brain.py', 'w') as f: f.write(c)
    if executor_patches:
        with open('src/web/lib/executor.py') as f: orig_exec = f.read()
        for old, new in executor_patches:
            with open('src/web/lib/executor.py') as f: c = f.read()
            c = c.replace(old, new)
            with open('src/web/lib/executor.py', 'w') as f: f.write(c)
    
    client = CerebrasClient()
    random.seed(42)
    cells = []
    
    for task_key in TASK_KEYS:
        for n_dist in DISTURB_LEVELS:
            runs = []
            for run_idx in range(N_RUNS):
                prefix = f"[{config_name:>8} {task_key:>12} d={n_dist} r={run_idx+1}/{N_RUNS}]"
                sys.stdout.write(f"{prefix} running..."); sys.stdout.flush()
                t0 = time.time()
                record = run_single(task_key, MAX_STEPS, n_dist, 4, run_idx, client, config_name)
                elapsed = time.time() - t0
                status = "OK" if record.success else "XX"
                sys.stdout.write(f"\r{prefix} {status} {record.steps_used:>2} steps {elapsed:.0f}s {record.avg_latency_ms:.0f}ms\n"); sys.stdout.flush()
                runs.append(record)
            cells.append(aggregate_cell(runs))
    
    # Restore files
    if brain_patches:
        with open('src/web/lib/brain.py', 'w') as f: f.write(orig_brain)
    if executor_patches:
        with open('src/web/lib/executor.py', 'w') as f: f.write(orig_exec)
    
    return cells

def print_table(cells):
    header = f"{'Config':<10} {'Task':<14} {'Dist':>4} {'Succ':>6} {'Steps':>6} {'AvgLat':>7} {'Failures':<40}"
    print(f"\n{'='*len(header)}")
    print(header)
    print('-'*len(header))
    for c in cells:
        fail_str = ", ".join(f"{k}:{v}" for k,v in sorted(c.failure_distribution.items()))
        print(f"{c.config:<10} {c.task:<14} {c.max_disturbances:>4} {c.success_rate:>6.2f} {c.mean_steps:>6.1f} {c.mean_avg_latency:>7.0f} {fail_str:<40}")
    print('-'*len(header))

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    all_results = {}
    
    # Save original files for restoration
    with open('src/web/lib/brain.py') as f: brain_orig = f.read()
    with open('src/web/lib/executor.py') as f: exec_orig = f.read()
    
    # ── Config A: Baseline (camera 384 + JPEG q50 only) ──
    cells_a = run_config("baseline")
    all_results["baseline"] = [asdict(c) for c in cells_a]
    print_table(cells_a)
    
    # ── Config B: + Shorter prompt ──
    # Replace the long workflow with short version
    # Already done in round 2 changes, so this is redundant
    # Skip for now
    
    # ── Config C: + Grasp 2cm lower + Holding status ──
    cells_c = run_config("grasp_fix",
        brain_patches=[
            # Add holding status after object positions
            ('    return "\\n".join(lines) + "\\n"',
             '    if not snap.gripper_open:\n'
             '        for key, pos in snap.objects.items():\n'
             '            if float(pos[2]) > 0.87:\n'
             '                label = dict(spec.visible_objects).get(key, key)\n'
             '                lines.append(f"HOLDING: {label}")\n'
             '                break\n'
             '        else:\n'
             '            lines.append("HOLDING: Nothing (gripper closed but empty)")\n'
             '    else:\n'
             '        lines.append("HOLDING: Nothing (gripper open)")\n'
             '    return "\\n".join(lines) + "\\n"'),
        ],
        executor_patches=[
            # Grasp 2cm lower
            ('"target_z": tz}, "open")\\n        snap = r2.final_snapshot\\n\\n        # Step 3: close gripper around object',
             '"target_z": tz - 0.02}, "open")\\n        snap = r2.final_snapshot\\n\\n        # Step 3: close gripper around object'),
        ])
    all_results["grasp_fix"] = [asdict(c) for c in cells_c]
    print_table(cells_c)
    
    # ── Config D: Random intervals ──  
    # Already baked into run_single, so this is a no-op config test
    # Skip
    
    # ── Config E: Combined ──
    cells_e = run_config("combined",
        brain_patches=[
            ('    return "\\n".join(lines) + "\\n"',
             '    if not snap.gripper_open:\n'
             '        for key, pos in snap.objects.items():\n'
             '            if float(pos[2]) > 0.87:\n'
             '                label = dict(spec.visible_objects).get(key, key)\n'
             '                lines.append(f"HOLDING: {label}")\n'
             '                break\n'
             '        else:\n'
             '            lines.append("HOLDING: Nothing (gripper closed but empty)")\n'
             '    else:\n'
             '        lines.append("HOLDING: Nothing (gripper open)")\n'
             '    return "\\n".join(lines) + "\\n"'),
        ],
        executor_patches=[
            ('"target_z": tz}, "open")\\n        snap = r2.final_snapshot\\n\\n        # Step 3: close gripper around object',
             '"target_z": tz - 0.02}, "open")\\n        snap = r2.final_snapshot\\n\\n        # Step 3: close gripper around object'),
        ])
    all_results["combined"] = [asdict(c) for c in cells_e]
    print_table(cells_e)
    
    # ── Write full results ──
    output = {
        "config": {"max_steps": MAX_STEPS, "n_runs": N_RUNS, "tasks": TASK_KEYS, "disturb_levels": DISTURB_LEVELS},
        "results": all_results,
        "timestamp": time.time(),
    }
    with open(os.path.join(RUNS_DIR, "overnight_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to runs/overnight_results.json")
    
    # Restore originals
    with open('src/web/lib/brain.py', 'w') as f: f.write(brain_orig)
    with open('src/web/lib/executor.py', 'w') as f: f.write(exec_orig)

if __name__ == "__main__":
    main()
