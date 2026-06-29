#!/usr/bin/env python3
"""Benchmark suite: runs each (task x disturbance) cell 5 times with full metrics."""
import json, os, sys, time, random, statistics
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

# -- Config ------------------------------------------------------------------
MAX_STEPS = 20
DISTURB_EVERY = 4
N_RUNS = 5
TASK_KEYS = ["pick_can", "pick_milk", "pick_bread", "pick_cereal", "lift_cube", "stack_cubes"]
DISTURB_LEVELS = [0, 1, 2]
BENCHMARK_OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "benchmark_results.json")

# -- Failure classification --------------------------------------------------
FailureStage = Literal["success", "exhausted_steps", "early_done", "tool_error", "hallucinated_coords"]

def classify_failure(success: bool, steps_used: int, max_steps: int,
                     hallucinated: bool, early_done: bool, tool_error: bool) -> FailureStage:
    if tool_error:
        return "tool_error"
    if hallucinated:
        return "hallucinated_coords"
    if success:
        return "success"
    if early_done:
        return "early_done"
    return "exhausted_steps"

def check_hallucinated(step_data: list[dict]) -> bool:
    for s in step_data:
        params = s.get("intent", {}).get("params", {})
        z = params.get("z", 0.85)
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)
        if z is not None and z < 0.5:
            return True
        if x is not None and abs(x) > 1.0:
            return True
        if y is not None and abs(y) > 1.0:
            return True
    return False

# -- Data carriers -----------------------------------------------------------
@dataclass
class StepRecord:
    step: int
    tool: str
    latency_ms: float
    reached: bool
    grasped: bool
    lifted: bool
    placed: bool
    success: bool
    distance_xy: float
    distance_z: float
    params: dict = field(default_factory=dict)

@dataclass
class RunRecord:
    task: str
    max_steps: int
    max_disturbances: int
    disturb_every: int
    run_idx: int
    success: bool
    steps_used: int
    disturbances_applied: int
    failure_stage: str
    avg_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    started_at: float
    ended_at: float
    steps: list[dict] = field(default_factory=list)

@dataclass
class CellAggregate:
    task: str
    max_disturbances: int
    n_runs: int
    success_rate: float
    n_success: int
    mean_steps: float
    mean_avg_latency: float
    mean_max_latency: float
    median_avg_latency: float
    p95_avg_latency: float
    failure_distribution: dict[str, int]
    runs: list[dict] = field(default_factory=list)

# -- Single run --------------------------------------------------------------
def run_single(task_key: str, max_steps: int, max_disturbances: int,
               disturb_every: int, run_idx: int, client: CerebrasClient | None = None) -> RunRecord:
    spec = get_task(task_key)
    sim = PandaSim(spec)
    brain = GemmaBrain(client=client)
    executor = MotionExecutor(sim)
    disturber = Disturbance(sim)
    history: list[HistoryItem] = []

    snap = sim.reset()
    executor.seed_from(snap)
    prev_snap = None
    disturbances_applied = 0
    _next_disturb = random.randint(2, 5)
    _need_image = True  # send image on step 1  # first disturbance between step 2-5
    step_records: list[dict] = []
    latencies: list[float] = []
    early_done = False
    tool_error = False
    hallucinated = False

    started_at = time.time()

    for step_idx in range(1, max_steps + 1):
        snap_before = sim.snapshot()

        # Disturbance at random intervals
        if disturbances_applied < max_disturbances and step_idx > 1 and step_idx >= _next_disturb:
            obj = spec.target_object or (list(snap_before.objects.keys())[0] if snap_before.objects else None)
            if obj and obj in snap_before.objects:
                dx = random.uniform(-0.08, 0.08)
                dy = random.uniform(-0.08, 0.08)
                disturber.move_object(obj, dx=dx, dy=dy)
                disturbances_applied += 1
                _next_disturb = step_idx + random.randint(2, 5)
                snap_before = sim.snapshot()

        # Think
        intent = brain.think(spec.description, snap_before, history, spec, prev_snap, send_image=_need_image)
        _need_image = False  # text-only until next disturbance
        latencies.append(intent.latency_ms)

        # Check hallucination
        z = intent.params.get("z", 0.85)
        x = intent.params.get("x", 0.0)
        y = intent.params.get("y", 0.0)
        if (z is not None and z < 0.5) or (x is not None and abs(x) > 1.0) or (y is not None and abs(y) > 1.0):
            hallucinated = True

        # Execute
        if intent.tool == "done":
            early_done = True
            break

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
        if env_success(final, spec):
            v.success = True

        history.append(HistoryItem(
            step=step_idx, tool=intent.tool,
            tool_params=str(intent.params), reasoning=intent.reasoning,
            ee_x=float(final.ee_pos[0]), ee_y=float(final.ee_pos[1]),
            ee_z=float(final.ee_pos[2]), gripper_open=final.gripper_open,
            verdict_note=v.notes,
        ))
        prev_snap = snap_before

        step_records.append({
            "step": step_idx, "tool": intent.tool,
            "latency_ms": intent.latency_ms,
            "reached": v.reached, "grasped": v.grasped,
            "lifted": v.lifted, "placed": v.placed,
            "success": v.success,
            "distance_xy": round(v.distance_xy, 4),
            "distance_z": round(v.distance_z, 4),
            "params": intent.params,
        })

        if v.success:
            break

    ended_at = time.time()
    steps_used = len(step_records)
    success = step_records[-1]["success"] if step_records else False
    failure_stage = classify_failure(success, steps_used, max_steps, hallucinated, early_done, tool_error)

    stats = dict(
        avg=statistics.mean(latencies) if latencies else 0.0,
        max=max(latencies) if latencies else 0.0,
        p50=statistics.median(latencies) if latencies else 0.0,
        p95=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
    )

    return RunRecord(
        task=task_key, max_steps=max_steps,
        max_disturbances=max_disturbances, disturb_every=disturb_every,
        run_idx=run_idx,
        success=success, steps_used=steps_used,
        disturbances_applied=disturbances_applied,
        failure_stage=failure_stage,
        avg_latency_ms=stats["avg"],
        max_latency_ms=stats["max"],
        p50_latency_ms=stats["p50"],
        p95_latency_ms=stats["p95"],
        started_at=started_at, ended_at=ended_at,
        steps=step_records,
    )

# -- Aggregate ---------------------------------------------------------------
def aggregate_cell(runs: list[RunRecord]) -> CellAggregate:
    task = runs[0].task
    n_dist = runs[0].max_disturbances
    avg_latencies = [r.avg_latency_ms for r in runs]
    steps_used = [r.steps_used for r in runs]
    fail_dist = defaultdict(int)
    for r in runs:
        fail_dist[r.failure_stage] += 1

    return CellAggregate(
        task=task, max_disturbances=n_dist, n_runs=len(runs),
        success_rate=sum(1 for r in runs if r.success) / len(runs),
        n_success=sum(1 for r in runs if r.success),
        mean_steps=statistics.mean(steps_used),
        mean_avg_latency=statistics.mean(avg_latencies),
        mean_max_latency=statistics.mean([r.max_latency_ms for r in runs]),
        median_avg_latency=statistics.median(avg_latencies),
        p95_avg_latency=sorted(avg_latencies)[int(len(avg_latencies) * 0.95)] if len(avg_latencies) >= 5 else avg_latencies[-1],
        failure_distribution=dict(fail_dist),
        runs=[asdict(r) for r in runs],
    )

# -- Table -------------------------------------------------------------------
def print_table(cells: list[CellAggregate]) -> None:
    header = f"{'Task':<14} {'Dist':>4} {'Succ':>6} {'Steps':>6} {'AvgLat':>7} {'MaxLat':>7} {'Failures':<40}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))
    for c in cells:
        fail_str = ", ".join(f"{k}:{v}" for k, v in sorted(c.failure_distribution.items()))
        print(f"{c.task:<14} {c.max_disturbances:>4} {c.success_rate:>6.2f} {c.mean_steps:>6.1f} {c.mean_avg_latency:>7.0f} {c.mean_max_latency:>7.0f} {fail_str:<40}")
    print("-" * len(header))

# -- Main --------------------------------------------------------------------
def main() -> None:
    random.seed(42)
    all_cells: list[CellAggregate] = []
    total_runs = len(TASK_KEYS) * len(DISTURB_LEVELS) * N_RUNS
    print(f"Benchmark: {len(TASK_KEYS)} tasks x {len(DISTURB_LEVELS)} disturb levels x {N_RUNS} runs = {total_runs} total")
    print(f"Max steps: {MAX_STEPS}, disturb every {DISTURB_EVERY} steps")
    print()

    shared_client = CerebrasClient()  # reuse across all 90 runs
    for task_key in TASK_KEYS:
        for n_dist in DISTURB_LEVELS:
            runs: list[RunRecord] = []
            for run_idx in range(N_RUNS):
                prefix = f"[{task_key:>12} d={n_dist} r={run_idx+1}/{N_RUNS}]"
                sys.stdout.write(f"{prefix} running...")
                sys.stdout.flush()
                t0 = time.time()
                record = run_single(task_key, MAX_STEPS, n_dist, DISTURB_EVERY, run_idx, shared_client)
                elapsed = time.time() - t0
                status = "OK" if record.success else "XX"
                sys.stdout.write(f"\r{prefix} {status}  {record.steps_used:>2} steps  {elapsed:.0f}s  {record.avg_latency_ms:.0f}ms avg\n")
                sys.stdout.flush()
                runs.append(record)
            cell = aggregate_cell(runs)
            all_cells.append(cell)

    print_table(all_cells)

    output = {
        "config": dict(max_steps=MAX_STEPS, disturb_every=DISTURB_EVERY,
                       n_runs=N_RUNS, tasks=TASK_KEYS, disturb_levels=DISTURB_LEVELS),
        "results": [asdict(c) for c in all_cells],
        "timestamp": time.time(),
    }
    os.makedirs(os.path.dirname(BENCHMARK_OUT), exist_ok=True)
    with open(BENCHMARK_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {BENCHMARK_OUT}")

if __name__ == "__main__":
    main()
