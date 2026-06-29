# GemmaBot

Three live demos of **Gemma 4 31B** on **Cerebras Inference** driving simulated
and real-camera robots through a closed-loop control stack.

```
Camera image  ─▶  Gemma 4 (one call, ~500ms)  ─▶  structured intent  ─▶  controller  ─▶  robot
                       ▲
                       └─── EE proprioception + recent step history (as text)
```

## What ships

| Demo | Port | What it shows |
|------|------|---------------|
| **Panda pick-and-place** (`robosuite_server`) | `:8002/robot_live` | Gemma drives a robosuite Franka Panda. Composite image (birdview + frontview) → target XYZ + gripper command → OSC controller drives the arm for 25 steps, then re-evaluates. |
| **PushT hybrid** (`pusht_server`) | `:8001/` | Gymnasium PushT task. Phase 1 = grid search to find the T-block (no LLM). Phase 2 = Gemma-guided pushes with reward-delta feedback. |
| **LeRobot real-camera viewer** (`server`) | `:8000/real_vision` | Real ALOHA cabinet-opening frames from `lerobot/aloha_mobile_cabinet`, with the Zone A-F overlay grid that Gemma sees. |

The landing page at `:8000/` is a hub linking to all three.

## Architecture (Panda demo)

```
src/web/robosuite_server.py        FastAPI routes (slim — ~170 lines)
src/web/lib/
  imaging.py                       orientation, JPEG b64, grid overlay, composite stitch
  sim.py                           PandaSim — robosuite env lifecycle + Snapshot dataclass
  brain.py                         GemmaBrain — prompt construction + structured-output call
  executor.py                      MotionExecutor — motion + gripper position state-machine
```

The executor owns a **`desired_closed: bool | None`** that only flips on
explicit `open` / `close` commands from Gemma. Every physics step asserts the
position command for the current desired state. `hold` literally means
"don't change desired_closed" — never a zero velocity, which the OSC controller
otherwise interprets as "let the gripper drift". After a state change the
executor runs extra ticks until `robot0_gripper_qpos` settles, so a grasp is
confirmed before the next Gemma call.

## Setup

```bash
git clone <repo-url>
cd cerebras-gemma4-hackathon
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # paste your CEREBRAS_API_KEY
```

## Running

Each demo is a standalone uvicorn app. Run them in separate terminals:

```bash
# Panda pick-and-place — http://localhost:8002/robot_live
uv run python -m src.web.robosuite_server

# PushT hybrid controller — http://localhost:8001/
uv run python -m src.web.pusht_server

# Landing page + real-camera viewer — http://localhost:8000/
./run_web.sh
```

Or use the bundled launcher to run all three:

```bash
./run.sh           # tails all three logs into runs/*.log
```

## Performance

| Stage | Latency | Notes |
|-------|---------|-------|
| Gemma call (composite image, JSON schema) | ~450-600 ms | Single image_url, p50 measured from US west |
| OSC motion segment | ~1.25 s | 25 physics steps + gripper confirm |
| Frames per Gemma call | 4-6 | Returned to UI as base64 JPEGs |

## Repo layout

```
src/
  client.py          CerebrasClient (thin Cerebras SDK wrapper with timing)
  config.py          env loader (CEREBRAS_API_KEY etc.)
  __init__.py        encode_image helper
  web/
    server.py        :8000 landing + real-image viewer
    pusht_server.py  :8001 PushT hybrid controller
    robosuite_server.py  :8002 Panda demo (slim FastAPI routes)
    lib/             building blocks for the Panda demo
    static/          HTML pages
robot_video/
  frame_loader.py    LeRobot dataset → PIL frames
  pusht_controller.py HybridPushtController (search + Gemma + feedback)
scripts/             experiment harnesses, visualizer, benchmark runners
```

## License

MIT
