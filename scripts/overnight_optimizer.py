#!/usr/bin/env python3
"""Autonomous overnight optimizer: mutates configs, runs benchmark, keeps best."""
import json, os, sys, time, random, copy, subprocess, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["OBJC_DISABLE_MULTIPLE_CLASS_IMPLEMENTATION_WARNING"] = "1"

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "overnight_auto")
os.makedirs(RUNS_DIR, exist_ok=True)

# ── Config space to search ──────────────────────────────────────────
# Each key is a (file, old_string, description)
# Each value is a list of possible (new_string, description) options

PARAM_SPACE = {
    "temperature": {
        "file": "src/web/lib/brain.py",
        "old": "temperature=0.0,",
        "options": [
            ("temperature=0.0,", "temp_0.0 (deterministic)"),
            ("temperature=0.1,", "temp_0.1 (slight noise)"),
            ("temperature=0.2,", "temp_0.2 (creative)"),
        ]
    },
    "max_tokens": {
        "file": "src/web/lib/brain.py",
        "old": "max_tokens=300,",
        "options": [
            ("max_tokens=300,", "tok_300 (default)"),
            ("max_tokens=200,", "tok_200 (tight)"),
            ("max_tokens=400,", "tok_400 (roomy)"),
        ]
    },
    "jpeg_quality": {
        "file": "src/web/lib/imaging.py",
        "old": "quality: int = 50",
        "options": [
            ("quality: int = 50", "jpeg_q50 (small)"),
            ("quality: int = 70", "jpeg_q70 (standard)"),
            ("quality: int = 30", "jpeg_q30 (tiny)"),
        ]
    },
    "grasp_height": {
        "file": "src/web/lib/executor.py",
        "old": '"target_z": tz - 0.02}',
        "options": [
            ('"target_z": tz}', "grasp_center"),
            ('"target_z": tz - 0.015}', "grasp_-1.5cm"),
            ('"target_z": tz - 0.025}', "grasp_-2.5cm"),
            ('"target_z": tz - 0.035}', "grasp_-3.5cm"),
        ]
    },
    "workflow_prompt": {
        "file": "src/web/lib/brain.py",
        "old": 'WORKFLOW',
        # This is handled by checking if the short or long workflow is present
        "special": "prompt",
        "options": [
            ("short", "short_workflow"),
            ("long", "long_workflow"),
            ("none", "no_workflow"),
        ]
    },
}

# ── Score function ──────────────────────────────────────────────────
def evaluate_config(config_name: str) -> dict:
    """Run a single task cell (pick_can, lift_cube, 0 disturbances, 2 runs) 
    and return score."""
    results = {}
    total_score = 0.0
    
    for task in ["lift_cube", "pick_can", "stack_cubes"]:
        successes = 0
        latencies = []
        steps_list = []
        
        for run_idx in range(2):  # 2 runs per cell
            t0 = time.time()
            res = subprocess.run(
                ["uv", "run", "python", "scripts/adversarial_test.py", task, "15", "0", "0"],
                capture_output=True, text=True, timeout=120
            )
            elapsed = time.time() - t0
            
            # Parse last JSON line
            for line in res.stdout.split('\n'):
                if line.strip().startswith('{'):
                    try:
                        d = json.loads(line)
                        if d.get("task"):
                            successes += 1 if d.get("success") else 0
                            steps_list.append(d.get("steps", 20))
                            break
                    except:
                        pass
            # Extract latency from output
            for line in res.stdout.split('\n'):
                if 'latency=' in line:
                    try:
                        ls = line.split('latency=')[1].split('ms')[0]
                        latencies.append(float(ls))
                    except:
                        pass
        
        success_rate = successes / 2
        avg_lat = sum(latencies) / len(latencies) if latencies else 999
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 20
        
        results[task] = {
            "success_rate": success_rate,
            "avg_latency_ms": avg_lat,
            "avg_steps": avg_steps,
        }
        # Score: 100 points per success, bonus for fast + few steps
        task_score = (success_rate * 100) + max(0, 20 - avg_steps) + max(0, 600 - avg_lat) / 10
        total_score += task_score
    
    results["total_score"] = total_score
    return results

# ── Config mutation ─────────────────────────────────────────────────
def apply_config(config: dict) -> str:
    """Apply a config dict to the source files. Returns config name."""
    name_parts = []
    
    for param_key, choice in config.items():
        param = PARAM_SPACE[param_key]
        choice_idx = choice
        chosen = param["options"][choice_idx]
        
        if param_key == "workflow_prompt":
            # Special handling for prompt text
            path = param["file"]
            with open(path) as f:
                c = f.read()
            if chosen[0] == "short":
                # Already has short workflow, do nothing
                pass
            elif chosen[0] == "long":
                # Replace short with long
                c = c.replace(
                    '"WORKFLOW (descend below top before closing, then lift slowly):\\n"',
                    '"WORKFLOW - DESCEND BELOW THE OBJECT TOP BEFORE CLOSING, then LIFT SLOWLY.\\n"'
                )
            elif chosen[0] == "none":
                # Remove workflow entirely
                old_workflow = '"WORKFLOW (descend below top before closing, then lift slowly):\\n"'
                c = c.replace(old_workflow, '')
            with open(path, 'w') as f:
                f.write(c)
        else:
            path = param["file"]
            with open(path) as f:
                c = f.read()
            c = c.replace(param["old"], chosen[0])
            with open(path, 'w') as f:
                f.write(c)
        
        name_parts.append(chosen[1])
    
    return "+".join(name_parts)

def restore_config(config: dict):
    """Reverse the config changes."""
    for param_key, choice in config.items():
        param = PARAM_SPACE[param_key]
        # No need to restore — we'll re-apply next config
        pass

def mutate_config(current: dict | None) -> dict:
    """Create a new config, possibly mutating from current."""
    if current is None:
        # Random initial config
        return {k: random.randrange(len(v["options"])) for k, v in PARAM_SPACE.items()}
    
    # Mutate 1-2 params
    new = dict(current)
    n_mutations = random.randint(1, 2)
    for _ in range(n_mutations):
        key = random.choice(list(PARAM_SPACE.keys()))
        options = len(PARAM_SPACE[key]["options"])
        new[key] = random.randrange(options)
    
    return new

# ── Safety checks ──────────────────────────────────────────────────
def verify_files() -> bool:
    """Check all source files still parse."""
    try:
        import ast
        for f in ['src/web/lib/brain.py', 'src/web/lib/executor.py', 'src/web/lib/imaging.py', 'src/web/lib/sim.py']:
            with open(f) as fh: ast.parse(fh.read())
        return True
    except:
        return False

# ── Main loop ──────────────────────────────────────────────────────
def main():
    print(f"AUTONOMOUS OVERNIGHT OPTIMIZER")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Config space: {len(PARAM_SPACE)} params with {sum(len(v['options']) for v in PARAM_SPACE.values())} total options")
    print(f"Search: random-start + hill-climbing")
    print(f"{'='*70}")
    
    # Save originals
    originals = {}
    for key, param in PARAM_SPACE.items():
        if key != "workflow_prompt":
            with open(param["file"]) as f:
                originals[key] = f.read()
    
    history = []
    best_score = -1
    best_config = None
    current_config = None
    
    iteration = 0
    max_iterations = 50
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'─'*70}")
        print(f"Iteration {iteration}/{max_iterations}  ({datetime.now().isoformat()})")
        
        # 1. Mutate
        current_config = mutate_config(current_config)
        config_name = apply_config(current_config)
        
        # 2. Verify files aren't broken
        if not verify_files():
            print(f"  ❌ Config {config_name} breaks file syntax — skipping")
            continue
        
        # 3. Save snapshot
        snapshot_dir = os.path.join(RUNS_DIR, f"iter_{iteration:03d}_{config_name[:40]}")
        os.makedirs(snapshot_dir, exist_ok=True)
        for f in ['src/web/lib/brain.py', 'src/web/lib/executor.py', 'src/web/lib/imaging.py', 'src/web/lib/sim.py']:
            shutil.copy2(f, os.path.join(snapshot_dir, os.path.basename(f)))
        
        # 4. Evaluate
        print(f"  Config: {config_name}")
        t0 = time.time()
        
        try:
            results = evaluate_config(config_name)
            elapsed = time.time() - t0
            score = results["total_score"]
            
            print(f"  Score: {score:.1f}  (time: {elapsed:.0f}s)")
            for task, r in results.items():
                if task != "total_score":
                    print(f"    {task}: succ={r['success_rate']:.0%} lat={r['avg_latency_ms']:.0f}ms steps={r['avg_steps']:.0f}")
            
            # Save results
            record = {
                "iteration": iteration,
                "config": current_config,
                "config_name": config_name,
                "score": score,
                "results": {k:v for k,v in results.items() if k != "total_score"},
                "time_seconds": elapsed,
                "timestamp": time.time(),
            }
            history.append(record)
            
            # 5. Keep best
            if score > best_score:
                best_score = score
                best_config = dict(current_config)
                print(f"  🏆 NEW BEST! Score: {score:.1f}")
                # Save best config
                with open(os.path.join(RUNS_DIR, "best_config.json"), "w") as f:
                    json.dump({"config": best_config, "name": config_name, "score": best_score, "results": results}, f, indent=2)
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            record = {"iteration": iteration, "config": current_config, "config_name": config_name, "score": -1, "error": str(e)}
            history.append(record)
        
        # Save full history every iteration
        with open(os.path.join(RUNS_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        # 6. Restore originals for next iteration
        for key, param in PARAM_SPACE.items():
            if key != "workflow_prompt" and key in originals:
                with open(param["file"], "w") as f:
                    f.write(originals[key])
        
        # Brief pause to let API cool
        time.sleep(5)
    
    # Final report
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS after {iteration} iterations")
    print(f"{'='*70}")
    print(f"Best config: {best_config}")
    print(f"Best score: {best_score:.1f}")
    print(f"Results saved to: {RUNS_DIR}")

if __name__ == "__main__":
    random.seed(42)
    main()
