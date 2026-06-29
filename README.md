# GemmaBot

Multi-provider (OpenRouter/Cerebras) robot vision pipeline — Gemma 4 31B drives simulated
robots and evaluates against real human teleoperation data from LeRobot datasets.

```
Camera image ─▶ SamPerceptor ─▶ Detection masks + bboxes + vectors ─▶ debug_view (b64)
                      │
                      ▼
         VisionGroundingModule ─▶ vision-only text block ─▶ Gemma 4 (free tier)
                      │                                          │
                      ▼                                          ▼
               Debug overlays                              structured intent
               canvas + toggles                                │
                                                                ▼
                                                          MotionExecutor ─▶ robot
                                                               │
                                                               ▼
                                                          verify.py (judge)
```

## What ships

| Demo | Port | What it shows |
|------|------|---------------|
| **Panda pick-and-place** | `:8002/robot_live` | Gemma drives a robosuite Franka Panda. SAM detection overlays, debug view, layer toggles (masks/bboxes/vectors), reasoning panel. `?vision=true` for anti-cheat blind mode. |
| **PushT hybrid** | `:8001/` | Gymnasium PushT task. Phase 1 = grid search. Phase 2 = Gemma-guided pushes. |
| **ZTP Replay / Benchmark** | `:8003/` | Compare Gemma's intended actions against human teleoperation across 10+ LeRobot datasets (pusht, aloha, droid, bridge, libero, kuka). Side-by-side viewer, frame timeline, benchmark reports. |

## Provider support

| Provider | Default? | Cost | Models |
|----------|----------|------|--------|
| **OpenRouter** | ✅ Default | 🆓 Free tier available | `google/gemma-4-31b-it:free` (vision), `qwen/qwen3-coder:free` (JSON), `nvidia/nemotron-nano-12b-v2-vl:free` (labeling) |
| **Cerebras** | ❌ Fallback | Pay-as-you-go | Gemma 4 31B (faster, ~500ms) |

Auto-fallback: if primary fails, tries the next available provider.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/jwalin-shah/gemmabot
cd cerebras-gemma4-hackathon
uv venv && source .venv/bin/activate
uv pip install -e .

# 2. Set up API keys
cp .env.example .env
# Edit .env — add your OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)

# 3. Launch everything
./run.sh
# Panda:  http://localhost:8002/robot_live
# PushT:  http://localhost:8001/
# Replay: http://localhost:8003/
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
```

## Running

```bash
# All demos (recommended):
./run.sh

# Individual servers:
uv run python -m src.web.robosuite_server  # :8002/robot_live
uv run python -m src.web.pusht_server      # :8001/
uv run python -m src.web.replay_server     # :8003/
```

## Key features

### 🔍 SAM Visual Overlays
Every camera frame runs through SamPerceptor (MobileSAM via transformers, CPU).
The frontend shows: instance mask overlays, bounding boxes with labels, distance
vectors from end-effector to objects, Gemma's reasoning text rendered on frame.

### 🕵️ Anti-Cheat Blinding
`?vision=true` mode strips ground-truth positions from Gemma's prompt.
Only camera-derived coordinates are shown. The verifier (verify.py) still judges
against ground truth — but Gemma never sees it.

### 📊 ZTP Dataset Benchmarking
Replay real robot trajectories from 10+ LeRobot datasets. For each frame, Gemma
outputs what she would do. Compare against the human teleoperator's actual action:

- Tool choice accuracy (did Gemma pick the right action?)
- Action vector distance (how close was her target?)
- Per-dataset and per-tool breakdowns
- Frame-by-frame timeline with color-coded scores

### 🧠 Smart Model Router
Automatically picks the right OpenRouter model for each task:
- **Vision + action decisions**: `google/gemma-4-31b-it:free` (best vision model, free)
- **JSON parsing**: `qwen/qwen3-coder:free` (1M ctx, excellent structured output)
- **Object labeling**: `nvidia/nemotron-nano-12b-v2-vl:free` (vision, free)
- **Benchmark reports**: `meta-llama/llama-3.3-70b-instruct:free` (strong text reasoning)

## Architecture

```
src/
  provider.py              LLMProvider ABC, Cerebras/OpenRouter providers, ProviderRegistry
  provider_router.py       Smart model selection per task type
  client.py                CerebrasClient (thin SDK wrapper)
  config.py                All tunable knobs with env-var overrides
  web/
    server.py              (deprecated — old landing page)
    robosuite_server.py    :8002 Panda demo (main FastAPI app)
    pusht_server.py        :8001 PushT hybrid controller
    replay_server.py       :8003 ZTP replay / benchmark
    lib/
      brain.py             Gemma prompt construction + structured-output call
      executor.py          Motion executor with tool-calling + vision-coord execution
      imaging.py           Image encoding, composite, grid overlay
      sim.py               PandaSim — robosuite env lifecycle + Snapshot
      perception.py        Detection, SamPerceptor (MobileSAM), instance seg, contour
      grounding.py         VisionGroundingModule — anti-cheat blinding layer
      viz.py               Visualization: mask overlays, bboxes, distance vectors, debug composite
      schema.py            Single source of truth for tool schemas
      tasks.py             Task registry + success thresholds
      verify.py            Per-step ground-truth judge (never shown to Gemma)
      recorder.py          Step recording + replay
      exceptions.py        Structured error hierarchy
    static/
      robot_live.html      Main dashboard with canvas overlays, layer toggles, detection panel
      replay_viewer.html   Dataset comparison viewer
      index.html           Old landing page
      real_vision.html     Old real-image viewer
robot_video/
  frame_loader.py          LeRobot dataset frame loader
  pusht_controller.py      PushT hybrid controller
  action_mapper.py         Dataset registry + human↔Gemma intent mapping (10+ datasets)
  action_comparator.py     Comparison metrics + benchmark aggregation
  replay_engine.py         Dataset replay engine
   _archive/               Old experimental code — kept for reference
```

## Performance

| Stage | Latency | Notes |
|-------|---------|-------|
| OpenRouter Gemma 4 31B (vision + JSON schema) | ~1-3s | Free tier, varies by load |
| Cerebras Gemma 4 31B (vision + JSON schema) | ~450-600ms | Faster but may not have Gemma access |
| SAM MobileSeg (CPU, 384x384) | ~200-300ms | Per frame, M3 Mac |
| OSC motion segment | ~1.25s | 25 physics steps + gripper confirm |

## License

MIT
