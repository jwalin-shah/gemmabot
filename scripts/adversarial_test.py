#!/usr/bin/env python3
"""Adversarial test: runs a task with disturbances and verifies adaptation."""
import json, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING"] = "1"

from src.web.lib.sim import PandaSim
from src.web.lib.brain import GemmaBrain, HistoryItem
from src.web.lib.executor import MotionExecutor, Disturbance
from src.web.lib.verify import verify, env_success
from src.web.lib.recorder import RunRecorder
from src.web.lib.tasks import get as get_task

task_key = sys.argv[1] if len(sys.argv) > 1 else "lift_cube"
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
disturb_every = int(sys.argv[3]) if len(sys.argv) > 3 else 3
max_disturbances = int(sys.argv[4]) if len(sys.argv) > 4 else 2

spec = get_task(task_key)
print(f"Task: {spec.label}  mode={spec.mode}")
print(f"Steps: {max_steps}, disturb every {disturb_every} steps, max disturbances: {max_disturbances}")

sim = PandaSim(spec)
brain = GemmaBrain()
executor = MotionExecutor(sim)
disturber = Disturbance(sim)
history: list[HistoryItem] = []

snap = sim.reset()
executor.seed_from(snap)
prev_snap = None  # for position delta tracking
recorder = RunRecorder(task=spec.key)
disturbances_applied = 0

for step_idx in range(1, max_steps + 1):
    snap_before = sim.snapshot()
    
    # Apply disturbance at interval
    disturb_info = None
    if disturbances_applied < max_disturbances and step_idx > 1 and step_idx % disturb_every == 0 and snap_before.objects:
        obj = spec.target_object or list(snap_before.objects.keys())[0]
        if obj and obj in snap_before.objects:
            dj = random.uniform(-0.08, 0.08)
            dk = random.uniform(-0.08, 0.08)
            disturb_info = disturber.move_object(obj, dx=dj, dy=dk)
            disturbances_applied += 1
            snap_before = sim.snapshot()  # re-snapshot after disturbance
            print(f"  ** DISTURBANCE {disturbances_applied}: {obj} moved ({dj:.3f},{dk:.3f})")
    
    intent = brain.think(spec.description, snap_before, history, spec, prev_snap)
    print(f"  Step {step_idx}: tool={intent.tool} params={intent.params}")

    # Execute via tool dispatch
    if intent.tool == "done":
        recorder.finalize(success=False)
        print(f"DONE (early): task={spec.key} at step {step_idx}")
        break
    
    try:
        result = executor.execute_tool(snap_before, intent.tool, intent.params)
    except Exception as e:
        print(f"  Tool error: {e}, falling back to move_to")
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
        step=step_idx,
        tool=intent.tool,
        tool_params=str(intent.params),
        reasoning=intent.reasoning,
        ee_x=float(final.ee_pos[0]), ee_y=float(final.ee_pos[1]), ee_z=float(final.ee_pos[2]),
        gripper_open=final.gripper_open,
        verdict_note=v.notes,
    ))

    prev_snap = snap_before  # store for next iteration's delta
    recorder.step(
        intent={"tool": intent.tool, "params": intent.params, "reasoning": intent.reasoning},
        ee=[round(float(final.ee_pos[i]), 3) for i in range(3)],
        gripper_open=final.gripper_open,
        objects={k: [round(float(v[i]), 3) for i in range(3)] for k, v in final.objects.items()},
        verdict=v, latency_ms=intent.latency_ms,
    )

    print(f"    -> result: reached={v.reached} grasped={v.grasped} lifted={v.lifted} placed={v.placed} success={v.success} | {v.notes}")
    
    if v.success:
        recorder.finalize(success=True)
        print(f"DONE: task={spec.key} success=True in {step_idx} steps, {disturbances_applied} disturbances")
        print(json.dumps({"task": spec.key, "success": True, "steps": step_idx, "disturbances": disturbances_applied}))
        sys.exit(0)

recorder.finalize(success=False)
print(f"EXHAUSTED: task={spec.key} after {max_steps} steps, {disturbances_applied} disturbances")
print(json.dumps({"task": spec.key, "success": False, "steps": max_steps, "disturbances": disturbances_applied}))
sys.exit(1)
