# 🤖 GemmaBot — Multi-Agent Robotics Demo

**Powered by Gemma 4 31B on Cerebras Inference**

[![Cerebras](https://img.shields.io/badge/Cerebras-00d4aa?style=flat-square)](https://cerebras.ai)
[![Gemma 4](https://img.shields.io/badge/Gemma_4_31B-4285F4?style=flat-square)](https://blog.google/technology/developers/gemma-4/)
[![Hackathon](https://img.shields.io/badge/Gemma_4_Hackathon-2026-ff6b6b?style=flat-square)](HACKATHON_RULES.md)

---

## 🏆 Tracks Entered

| Track | Prize | Focus |
|---|---|---|
| **Track 1: Multiverse Agents** | $2,000 | Multi-agent pipeline + multimodal scene understanding |
| **Track 2: People's Choice** | $2,000 | Shareable demo video showing Cerebras speed |
| **Track 3: Enterprise Impact** | $1,000 | Physical AI / robotics use case |

---

## ⚡ The Speed Story

**Cerebras Inference makes reactive robotics possible.**

On GPU: one reasoning step takes 1–5 seconds. Too slow for a robot to react to a changing world.

On Cerebras: one reasoning step takes **50–150ms**. The robot can perceive, decide, and act at **3–5 Hz**.

That's the difference between a robot that sees a cup get knocked over and **instantly reacquires it** — vs. a robot still planning its first move while the cup is already somewhere else.

---

## 🎮 Live Demo

### Web UI Dashboard

```
./run_web.sh
# Open http://localhost:8000
```

The web UI shows **two robot simulations running side-by-side in real time**:

- **⚡ Cerebras** — Gemma 4 31B running at full Cerebras speed
- **🐢 GPU Baseline** — same model, same task, with simulated GPU latency (~1.7s/tick)

Watch the speed race live. Press **"⚠️ Perturb Cup"** to drag the cracked cup to a new position — see Cerebras reacquire it in milliseconds while the GPU is still planning.

### Real-time Metrics (from actual benchmark)

| Metric | Cerebras ⚡ | GPU 🐢 | Advantage |
|---|---|---|---|
| Reasoning latency | **~300ms** | ~2,000ms | **6.6× faster** |
| Decision frequency | **3.2 Hz** | 0.5 Hz | **6.4× more responsive** |
| Multi-agent pipeline | **868ms total** | 5–10s | **6–12× faster** |

---

## 🧠 Architecture

```
                          ┌─────────────────────────────┐
                          │     Web UI (FastAPI)         │
                          │  Side-by-side sim viewer     │
                          │  SSE streaming · Race timer  │
                          └──────────┬──────────────────┘
                                     │ SSE events (frames, decisions)
                   ┌─────────────────┼─────────────────┐
                   ▼ Cerebras        │                 ▼ GPU (throttled)
          ┌──────────────────┐       │        ┌──────────────────┐
          │  Reactive Loop   │       │        │  Reactive Loop   │
          │  3–5 Hz          │       │        │  0.5 Hz          │
          └──────┬───────────┘       │        └──────┬───────────┘
                 ▼                   │               ▼
        ┌────────────────┐          │       ┌────────────────┐
        │  RobotBrain    │          │       │  RobotBrain    │
        │  (Gemma 4 31B) │          │       │  (Gemma 4 31B) │
        └───────┬────────┘          │       └───────┬────────┘
                ▼                   │               ▼
       ┌────────────────┐          │       ┌────────────────┐
       │  Skill Layer   │          │       │  Skill Layer   │
       │  pick · place   │          │       │  pick · place   │
       │  move_to · stop │          │       │  move_to · stop │
       └───────┬────────┘          │       └───────┬────────┘
               ▼                   │               ▼
       ┌────────────────┐          │       ┌────────────────┐
       │  2D Tabletop   │          │       │  2D Tabletop   │
       │  World (Zone   │          │       │  World (Zone   │
       │  Grid A–F)     │          │       │  Grid A–F)     │
       └────────────────┘          │       └────────────────┘
```

### Multi-Agent Pipeline

```
[Camera Image] ──▶ [Vision Agent] ──▶ [Action Agent] ──▶ [Safety Agent] ──▶ [Robot Execute]
                    Gemma 4             Gemma 4             Gemma 4            Simulated
                   Multimodal            Text                Text              Hardware

    291ms            372ms               204ms                0ms
    ─────────────────────────────────────────────────────────────────
                           TOTAL: 868ms
```

### Reactive Simulation Loop

```
Perceive ──▶ Decide ──▶ Act ──▶ (repeat every ~300ms)
    │           │          │
    ▼           ▼          ▼
 Render    Gemma 4     Skill Layer
 World     31B on     (pick, place,
 (640×420  Cerebras   move_to, stop)
 zone grid) Structured
            Outputs
```

### Sensory Separation (Semantic → Geometric Bridge)

**The model NEVER sees coordinates.** It perceives the world purely through the camera image (rendered with labeled zone grid). When it says "pick up the red cup in Zone B," the skill layer resolves that object ID to ground-truth coordinates. This mirrors how a real robot works — vision is approximate, actuation is precise.

---

## 🔧 Setup

```bash
# Clone and install
git clone <repo-url>
cd cerebras-gemma4-hackathon
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env with your CEREBRAS_API_KEY

# Run the web UI
./run_web.sh

# Or run the CLI demo
./run_demo.sh --image examples/images/workspace.jpg

# Run reactive simulation (CLI)
uv run python -m src.sim.run_sim --mock     # offline test
uv run python -m src.sim.run_sim            # live Cerebras

# Run speed comparison benchmark
uv run python -m src.sim.compare --throttle   # simulated GPU
uv run python -m src.sim.compare              # real GPU provider
```

---

## 🚀 Deploy to Render

```bash
# 1. Push to GitHub
# 2. Create a Render Blueprint:
#    - Connect repo → Render detects render.yaml
#    - Set CEREBRAS_API_KEY as an environment secret
# 3. Deploy
```

Or manually:
```bash
# 1. Build Docker image
docker build -t gemmabot .
docker run -p 8000:8000 -e CEREBRAS_API_KEY=your_key gemmabot
# 2. Deploy to your preferred cloud
```

---

## 📁 Project Structure

```
src/
├── client.py              # Cerebras API wrapper with timing
├── orchestrator.py        # Vision → Action → Safety → Execute pipeline
├── demo.py                # CLI demo entry point
├── config.py              # Environment configuration
├── agents/
│   ├── vision_agent.py    # Gemma 4 multimodal scene analysis
│   ├── action_agent.py    # Robot command sequence generation
│   └── safety_agent.py    # Hazard and constraint checking
├── sim/
│   ├── world.py           # 2D tabletop simulation (zone grid, physics)
│   ├── brain.py           # Gemma 4 decision maker (structured outputs)
│   ├── skills.py          # Low-level controllers (pick, place, move_to)
│   ├── loop.py            # Perceive → Decide → Act reactive loop
│   ├── run_sim.py         # Runnable demo, build_world(), INSTRUCTION
│   └── compare.py         # Cerebras vs GPU speed benchmark
├── web/
│   ├── server.py          # FastAPI SSE server
│   └── static/
│       └── index.html     # Live web dashboard
└── command_center/        # Dynamic multi-agent router (experimental)
    ├── root.py            # Always-on Gemma 4 watcher/router
    ├── branches.py        # Specialist branch registry
    └── types.py           # Type definitions
```

---

## 📊 Benchmark Results

Run `uv run python -m src.sim.compare` to reproduce:

```
Task: Put the cracked cup into bin_left. Do not touch the blue cup.
Window: 12s | cup dragged at t=4s

Cerebras (Gemma 4 31B)        |  38 decisions | 3.2 Hz | avg  312ms | re-acquired cup in 0.42s
GPU (throttled ~1.7s)         |   7 decisions | 0.6 Hz | avg 2025ms | re-acquired cup in 3.85s

Headline: Cerebras made 31 more decisions in the same window (38 vs 7).
```

---

## 🎥 Demo Video

See [demo-video-script.md](./demo-video-script.md) for the 60-second script.

Key shots for the video:
1. **Split screen** — Web UI with both sims running, Cerebras ticks visibly faster
2. **Perturbation test** — Click "Perturb Cup", watch Cerebras reacquire instantly
3. **Pipeline timing** — Show Vision 291ms / Action 372ms / Safety 204ms
4. **End card** — 6.6× faster, enter both tracks

---

## 📜 License

MIT
