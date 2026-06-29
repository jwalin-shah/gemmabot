# Self-Improving Overnight Research Loop — Chaos Edition

Every iteration: run experiments, adversarially review, generate new hypotheses, test them, repeat.
Data is NEVER lost. The loop adapts to whatever happens.

## The Breakthrough

Gemma 4 looked at a raw camera image with only a zone grid overlay and said:
"I see a red block in Zone D and a black block in Zone E" — and planned a full sequence.
The model DOES visual spatial reasoning. We've been cheating by giving it coordinates.

## Principles (Baked Into Everything)

1. **Zero answer leakage** — NO coordinates, NO object names/IDs, NO text labels on images
2. **Temperature 0.0 + structured output** — no regex, no phrasing brittleness
3. **No ZTP** — removed entirely
4. **384×384** — matches MuJoCo + Gemma vision encoder
5. **Data saved to disk always** — every experiment write JSONL, every iteration updates CHECKPOINT.json + results_all.csv. If the process dies, nothing is lost.

## What The Loop Tests

The loop tests Gemma 4's ability to identify objects spatially from vision alone.
It systematically tries harder and harder conditions to find where it breaks.

## Types of Variation to Test

### Visual Variations
- **Colors**: red, blue, green, yellow, tan, pink, orange, purple, all-gray (thermal sim)
- **Shapes**: round, square, triangular, star, irregular blob
- **Counts**: 1 object, 2, 3, 4, 5, 7, 10
- **Zones**: every zone (A-F), edge zones, all clustered in one zone
- **Sizes**: small (r=15), medium (r=26), large (r=40), mixed sizes
- **Zone boundaries**: objects exactly on grid lines — which zone does it report?
- **JPEG quality**: 95, 75, 50, 25, 10, 5 — at what point does it break?
- **Monochrome**: all objects same color, different shapes only — can it identify by shape?
- **No zone grid at all**: can it still say "top-left" instead of "Zone A"?

### Task Variations
- **Simple**: "Identify the objects"
- **Constraint**: "Identify but do NOT describe the blue object"
- **Counting**: "Count how many objects are in Zone B"
- **Spatial**: "Which object is closest to the gripper?"
- **Change detection**: "Did anything move since the last image?"
- **Multi-step**: "If I want to move the red cup to Zone C and the blue cup to Zone A, what's the first action?"

### Chaos Variations (Mid-Sequence Changes)
- **Gripper swap**: The gripper drawing changes color/style mid-sequence — does Gemma notice?
- **Hand change**: The gripper is drawn differently (thicker, different color, different shape)
- **Object teleport**: An object vanishes from one frame and appears in another zone
- **Object swap**: Two objects swap colors (red turns blue, blue turns red) — does Gemma detect the swap?
- **New object appears**: A 4th object appears mid-sequence that wasn't there before
- **Object disappears**: An object is removed — does Gemma notice it's gone?
- **Grid shift**: The zone grid labels move or change format — does Gemma adapt?
- **Background change**: Table color changes suddenly (wood → white → black)
- **Multiple simultaneous changes**: Object moves + gripper changes + background changes

### Proprioception Variations
- **No gripper at all**: Just objects on a table — can Gemma still reason about them?
- **Gripper in weird position**: Gripper is at the bottom of the frame instead of top
- **Gripper over an object**: Gripper visually overlaps an object — does Gemma still identify the object?

## Data That Must Always Be Saved

Every single run produces a JSONL entry. The format:

```json
{
  "run": 1,
  "experiment": "vision",
  "variation": "standard",
  "n_objects": 3,
  "n_objects_reported": 3,
  "zone_accuracy": 0.83,
  "zone_matches": 5,
  "zone_total": 6,
  "hallucinations": 0,
  "misses": 0,
  "latency_ms": 287.3,
  "temperature": 0.0,
  "jpeg_quality": 50,
  "monochrome": false,
  "success": true,
  "error": null,
  "prompt_sent": "Look at this camera image...",
  "raw_response": "{\"observed_objects\": [...]}",
  "timestamp": "2026-06-29T23:45:12Z"
}
```

Critical fields that must always be saved:
- **prompt_sent** — the exact prompt text
- **raw_response** — Gemma's raw output (for post-hoc analysis)
- **all variation parameters** — so we know what conditions produced each result

## The Meta-Loop: Per Iteration

### 0. Load State

```python
cp = json.loads(Path("overnight_results/CHECKPOINT.json").read_text())
```

Checkpoint structure:
```json
{
  "completed": {
    "vision": {"runs": 200, "mean_zone_accuracy": 0.85, ...}
  },
  "new_experiments": {
    "monochrome": {"script": "scripts/exp_monochrome.py", "runs": 100, "description": "..."}
  },
  "total_calls_used": 850,
  "round": 12,
  "last_review_notes": "Models confuses zone B and E at temperature 0.8",
  "tried_variations": ["standard", "temp_0.3", "jpeg_25", "monochrome"],
  "untried_variations": ["no_grid", "objects_10", "thermal_sim", "gripper_swap"],
  "_saved_at": "..."
}
```

### 1. Pick What To Run

Priority:
1. **Core experiments first** — vision baseline, then perturb
2. **Untried variations** — pick from `untried_variations`, or have the review subagent propose new ones
3. **New experiments** — check `new_experiments` for review-generated scripts

Always try to run something different from last time. Avoid repeating the exact same parameters.

### 2. Spawn Subagents (Parallel)

Spawn 1-2 **experiment subagents** + 1 **adversarial review subagent** simultaneously.

**Experiment subagent prompt:**
```
Run: python scripts/exp_vision.py --runs 200 --output overnight_results/vision/results.jsonl
The last line of stdout will be RESULT:{json}. Report the structured data back to me.
Also save the raw prompt_sent and raw_response fields.
```

**Adversarial review subagent prompt:**
```
You are an adversarial reviewer for a research project testing Gemma 4's visual spatial reasoning.

Read these files:
- overnight_results/CHECKPOINT.json
- overnight_results/results_all.csv (if it exists)
- OVERNIGHT_PLAN.md
- Any JSONL files in overnight_results/

Then answer:

1. WHAT IS WEAK? Find patterns in the failures. (e.g., "always confuses Zone B and E", "hallucinates a 4th object in 30% of runs", "fails when objects are close together")

2. WHAT NEW VARIATION SHOULD WE TEST? Propose something WE HAVEN'T TRIED YET. Look at `untried_variations` in CHECKPOINT. Pick one we haven't done, or invent a completely new one. Be creative.

3. WHAT IF WE CHANGE THE HAND/G RIPPER MID-RUN? The user wants to know: if the gripper changes color, shape, or position mid-experiment, does Gemma adapt? Design a quick test for this.

4. WHAT ASSUMPTION MIGHT BE WRONG? Challenge something. (e.g., "Maybe the zone grid lines themselves are confusing the model" or "Maybe temperature 0.0 is hiding that the model can't actually see colors")

5. WHAT'S THE CEILING? If everything works at 95% accuracy, what harder test breaks it?

Be specific. Be critical. Propose at least one concrete new experiment with implementation details.
```

The review subagent runs in the background while experiments run.

### 3. Collect ALL Data

When experiment subagents return:
- Parse the `RESULT:{json}` line
- Also save the raw `prompt_sent` and `raw_response` if available
- If the JSONL file was written, note its path

### 4. Synthesize + Generate

When ALL subagents are back:

1. **Save experiment results** to CHECKPOINT
2. **Save review findings** to CHECKPOINT (`last_review_notes`)
3. **If review proposed a good new experiment** and it's not already done:
   - Write the new script based on `scripts/exp_vision.py`
   - Add it to `new_experiments`
   - Mark the variation as tried: add to `tried_variations`
4. **If review proposed a new untried variation** of an existing experiment:
   - Add it to `untried_variations` if not already there — but DO NOT create a new script
   - Run it next iteration with a different argument to the same script (or just vary the scene generation)

### 5. Handle Chaos Proposals

If the review subagent proposes something like "test gripper change mid-sequence", implement it:

```python
# The review said: "test if Gemma notices when the gripper changes color"
# Write a quick variant of exp_perturb.py that changes the gripper on tick 5
new_script = """...modify render() to accept a gripper_color parameter..."""
```

Or if it's simpler: just run the existing perturb experiment with a different scene configuration.

### 6. Save & Schedule

```python
cp["round"] += 1
cp["_saved_at"] = now
```

ScheduleWakeup:
- **If experiments are still in-flight or just finished**: continue immediately (delaySeconds=60)
- **If everything completed and nothing new to run**: wait 30 min for the review agent to generate ideas (delaySeconds=1800)
- **If API is down**: wait 5 min (delaySeconds=300)

## Experiment Scripts

### Existing:

| Script | Runs | Time | What |
|--------|------|------|------|
| `python scripts/exp_vision.py --runs N --output path` | N | ~5s/run | Standard zone identification |
| `python scripts/exp_perturb.py --runs N --output path` | N×10 ticks | ~50s/seq | 10-tick sequences with perturbation on tick 5 |

### Adapting Existing Scripts For New Experiments:

When the review subagent proposes something, you can often run it as a variation of exp_vision.py:

| New experiment | How to run it (no new script needed) |
|---------------|--------------------------------------|
| Monochrome | Modify scene rendering in exp_vision.py: set all colors to gray |
| More objects | Add more entries to OBJECT_TEMPLATES and SCENE_LAYOUTS |
| No grid | Comment out the zone grid lines in render_scene() |
| JPEG quality | Change the save() quality parameter |
| Different sizes | Vary the radius parameter per object |

If the variation is SIMPLE (just parameter changes), modify exp_vision.py in-place and run it again.

If the variation is COMPLEX (new behavior like mid-run gripper change), write a new script.

## Failure Modes to Watch For

| Symptom | What's probably happening | Action |
|---------|--------------------------|--------|
| Zone accuracy < 40% | Gemma can't see the grid or objects | Run diagnosis: send image + "what do you see?" |
| Hallucinations > 1 per scene | Model is making up objects | Lower temperature, add stricter schema |
| Latency > 1000ms consistently | API overload or network issue | Reduce concurrency to 1 subagent |
| Error rate > 20% | Something is broken | Stop and inspect the error messages |
| Same experiment always fails | Systematic bias | Ask review agent to analyze failure patterns |

## Success Criteria

1. **Zone accuracy >= 80% on standard test** — Core claim proven
2. **Handles at least 3 chaos variations without accuracy dropping below 50%** — Robustness
3. **Re-acquires objects within 2 ticks after perturbation** — Adaptability
4. **Can identify objects without color (shape-only or thermal mode)** — Multi-modal
5. **The adversarial review finds at least one meaningful failure mode** — We know the ceiling

## How To Start

```bash
/loop Run the self-improving chaos research loop from OVERNIGHT_PLAN.md. No budget cap. Each iteration: spawn 1-2 experiment subagents + 1 adversarial review subagent IN PARALLEL. After they return, synthesize results, generate new experiments if review finds gaps, try new variations, and continue. Save ALL data. Keep going forever — try every variation you can think of. Save to overnight_results/.
```
