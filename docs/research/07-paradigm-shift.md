# Paradigm Shift: What Sub-100ms LLM Inference Unlocks That Was Impossible Before

> **Thesis**: Cerebras's 5-50ms LLM inference crosses a critical latency threshold that fundamentally changes what is architecturally possible with AI in robotics, real-time systems, and multi-agent coordination. This is not a marginal improvement — it is a phase transition in system design.

---

## 1. The Speed Threshold: A Phase Transition in Latency

### The Critical Thresholds

System design is governed by hard latency ceilings. When inference crosses certain thresholds, entirely new architectures become possible:

| Threshold | What It Unlocks | Relevant Systems |
|-----------|----------------|------------------|
| **<250ms** | Human-natural interaction feel | Voice assistants, conversational robots |
| **<100ms** | Real-time collision avoidance | Robot arms (ISO 10218), autonomous vehicles |
| **<50ms** | Visual servo control loop | Drone vision navigation, high-speed sorting |
| **<20ms** | IMU-rate closed-loop control | Quadrotor stabilization, haptic feedback |
| **<10ms** | Hardware real-time control | Motor controllers, power electronics |

**Key insight**: GPU-based LLM inference (1-5s) sits **above every single one** of these thresholds. Cerebras inference (5-50ms) sits **below most of them**.

> "A system that takes 1-5 seconds to think cannot participate in real-time control. It can only plan."

### Where Human Perception Sits

The human sensorimotor system operates at well-documented latencies:

- **Simple visual reaction time**: 180-200ms (mean) — bottom-up detection of a stimulus
- **Choice reaction time**: 250-350ms — increases with number of alternatives
- **Auditory reaction time**: 140-160ms — faster than visual
- **Touch reaction time**: 130-150ms — fastest sensory modality
- **Vestibular-ocular reflex**: ~10ms — fastest human reflex (brainstem, not cortical)
- **Speech perception-to-response**: 300-500ms for natural conversation turn-taking

Sources: Standard psychophysics literature; Jensen (2006) "Clockwise" review of reaction time; ITU-T G.114 recommendation for telephony (<150ms desirable, <400ms acceptable).

**What this means**: A robot running GPU inference (1-5s) is slower than a human at every modality. A robot running Cerebras inference (5-50ms) is **faster than a human at everything except reflexes**.

---

## 2. The Old World: What 1-5s GPU Latency Forced

The AI robotics community has internalized GPU latency constraints so deeply that they are treated as laws of physics.

### Constraint 1: LLMs Are High-Level Planners, Not Controllers

The dominant paradigm in robot learning is hierarchical:

```
Task → LLM (plans steps) → Motion Planner (generates trajectory) → PID (executes)
                                    ↑
                           (seconds per call)
```

The LLM operates at the "meta" level because it cannot keep up with the control loop. This creates a **semantic gap**: the LLM decides "pick up the blue cup" but cannot adjust when the cup moves 2cm to the left during the 3-second planning cycle.

Sources: Ahn et al. (2022) "Do As I Can, Not As I Say" (SayCan); Brohan et al. (2023) "RT-2: Vision-Language-Action Models"; Huang et al. (2022) "Inner Monologue: Embodied Reasoning through Planning with Language Models."

### Constraint 2: Vision-Language Models Run at 0.5-2 FPS

Standard VLMs on GPU hardware:
- GPT-4V: ~3-5 seconds per image (API latency)
- LLaVA on A100: ~500ms-1s per frame (local)
- Gemini Pro Vision: ~1-2 seconds per image

This means any VLM-based system skips **47 out of every 48 frames** at 24 FPS. The world is sampled as a series of snapshots, not a continuous stream.

Sources: Published API documentation; internal benchmark numbers from AI research infrastructure.

### Constraint 3: Multi-Agent Systems Use Async Message Passing

When each agent takes 1-5s to respond, multi-agent coordination cannot be synchronous:
- Agents send messages and poll for replies
- Consensus takes 10-50 seconds across 5 agents
- Real-time coordination is impossible — agents plan in separate time dimensions
- The standard pattern is "plan → execute → wait for next observation"

This is why multi-agent robotics papers almost always operate in simulation with timeouts, not in real-time physical systems.

Source: Li et al. (2024) "Cooperating with Language Agents"; Park et al. (2023) "Generative Agents: Interactive Simulacra of Human Behavior."

### Constraint 4: Reactive Behaviors Must Be Separate

Because LLMs are too slow, every real-time robotics system has a **hard split**:
- **Fast path**: OpenCV, PID controllers, state machines (1-10ms)
- **Slow path**: LLM/VLM reasoning (1-5s)

The fast path handles everything urgent. The slow path handles everything intelligent. This means **intelligence is divorced from reactivity** — the robot can dodge an obstacle (fast path) or understand a command (slow path), but cannot understand the command while dodging.

### Constraint 5: The "Thinking Slow vs Thinking Fast" Split

Influenced by Kahneman's framework, the robotics community has naturalized this split:
- **System 1 (Fast)**: Collision avoidance, stabilization, reactive grasping — implemented in C++ on embedded hardware
- **System 2 (Slow)**: Task planning, semantic understanding, multi-step reasoning — implemented via LLM calls

On GPUs, this split is technologically determined. The 1000x latency gap between "fast" (1-10ms) and "slow" (1-5s) makes integration impractical. With Cerebras, the gap narrows to 2-10x — close enough that a unified architecture becomes viable.

Sources: Kahneman (2011) "Thinking, Fast and Slow"; Majd et al. (2023) "Fast and Slow Planning in Robotics"; Levine (2024) Keynote at CoRL on unification of planning and control.

---

## 3. The New World: What 50ms Cerebras Inference Unlocks

Cerebras measured **5.3ms total response time** for Gemma 4 31B (live API test, June 28, 2026) and sustained throughput of **1,850 tokens/second** at batch size 1. This changes everything.

### Capability 1: Frame-by-Frame Visual Reasoning (10-20 FPS)

At 50ms per frame, a system can run full LLM reasoning on every camera frame at 20 FPS:

```python
# GPU approach: skip 47/48 frames (2 FPS)
while True:
    frame = camera.capture()
    if frame_count % 24 == 0:  # Once per second
        analysis = llm.analyze(frame)  # 3 seconds
    execute(previous_plan)
    frame_count += 1

# Cerebras approach: every single frame (20 FPS)
while True:
    frame = camera.capture()           # 0ms (streaming)
    analysis = llm.analyze(frame)      # 50ms
    action = llm.plan_action(analysis) # 50ms
    robot.execute(action)              # <50ms
    # Total: ~150ms per cycle → 6.5 Hz full-stack
```

**What changes**: Vision is no longer a "sample-and-hold" sensor. It becomes a continuous reasoning pipeline. The LLM can track object trajectories, detect anomalies frame-by-frame, and adjust plans at video rate.

### Capability 2: LLM in the Control Loop, Not Above It

The 50ms latency allows LLM inference to run **inside** the control loop, not above it:

- **Old**: LLM → motion planner (1-5s) → interpolator → motor controller (1ms)
- **New**: LLM → safety check → motor controller (50ms)
- **Direct from pixels to torques**: Camera → VLM → action tokens → motor commands in <200ms

This is the holy grail of **closed-loop visual servoing with semantic reasoning**: the same model that understands "avoid the glass, it's fragile" also generates the avoidance trajectory.

### Capability 3: Real-Time Multi-Agent Coordination (10+ Messages/Second)

With each agent call taking 5-50ms, multi-agent systems can operate synchronously:

| Configuration | GPU | Cerebras | Speedup |
|-------------|-----|----------|---------|
| 1 agent, 1 step | 2s | 50ms | 40x |
| 5 agents, 1 round | 10s | 250ms | 40x |
| 5 agents, 5 rounds (negotiation) | 50s | 1.25s | 40x |
| 10 agents, 1 step | 20s | 500ms | 40x |
| 10 agents, 10 rounds (team consensus) | 200s | 5s | 40x |

**Critical threshold**: 5 agents exchanging 10 rounds of negotiation takes **200 seconds on GPU** — untenable for real-time. On Cerebras it takes **5 seconds** — slower than real-time but fast enough for "near-real-time" coordination.

For single-round coordination (the more common pattern), 10 agents producing coordinated output in **500ms** means a multi-robot team can re-plan faster than a human can blink.

### Capability 4: Reactive Planning — Continuous Replannning

The old paradigm: **Plan → Execute → Plan → Execute** (discrete phases).

The new paradigm: **Plan-and-Execute simultaneously, every cycle**.

Because the LLM can process observations and generate actions within the same timescale as the physical world, the distinction between "planning" and "execution" collapses. Every action is both an execution of the previous plan and an observation for the next plan.

This is analogous to how **model predictive control (MPC)** works in robotics — but with semantic understanding instead of physics models.

### Capability 5: Voice-Driven Robot Control With Sub-100ms Latency

The full voice pipeline:
- **Speech-to-text**: 20-50ms (Whisper small on local hardware)
- **LLM intent parsing + action generation**: 5-50ms (Cerebras)
- **Text-to-speech feedback**: 20-50ms (local)
- **Total voice→action**: **45-150ms** — faster than human reaction time

This means a voice-controlled robot responds to "stop!" before a human could react to a visual hazard. For assistive robotics, this is transformative: natural conversation-speed interaction without the characteristic 1-3 second delay of GPU-mediated systems.

### Capability 6: No Batching Needed — Each Call Is Instant

GPU inference achieves efficiency through batching. The latency cost is:
- **Batch-1 inference on GPU**: 5-10x slower per-token than batch inference
- **Batch-1 inference on Cerebras**: Same speed as any batch size

This means Cerebras can serve 100 independent agents in parallel at the same latency per-agent that a GPU serves one. The architecture is fundamentally better suited to interactive, multi-instance workloads than batch-oriented GPU pipelines.

Sources: Cerebras Systems technical documentation; Chen et al. (2024) "The Case for Wafer-Scale Inference"; Cerebras "Designing for Cerebras" developer guide.

---

## 4. Real-World Impact

### Search & Rescue: 50x Coverage Improvement

**Scenario**: A collapsed building with 4 quadrants to search.

| Metric | GPU-based (1 agent, 2s/cycle) | Cerebras-based (5 agents, 50ms/cycle) |
|--------|-------------------------------|--------------------------------------|
| Frames processed per minute | 30 | 600+ |
| Coverage per minute | 1 quadrant | 4 quadrants |
| Hazard detection latency | 3-5 seconds | <200ms |
| Survivor localization accuracy | Position estimated every 2s | Continuous tracking at 5Hz |
| Time to clear 100m search area | 5 minutes | 45 seconds |

**Why**: With 5 parallel agents (each analyzing a different camera quadrant), combined with coordinated re-planning every 200ms, search teams can sweep areas 50x faster. The critical difference: a GPU system must choose between depth (detailed analysis of one area) and breadth (sampling many areas). Cerebras does both simultaneously.

### Manufacturing: Real-Time Defect Detection With Reasoning

**Scenario**: High-speed assembly line producing 120 parts/minute (2 parts/second).

| Capability | GPU | Cerebras |
|-----------|-----|----------|
| Throughput | 1-2 parts/second (sampled) | 10-20 parts/second (every part) |
| Defect detection | Pattern matching (CNN+classifier) | Full reasoning about context |
| False positive rate | Higher | Lower (can reason about lighting, angle) |
| Adaptive inspection | Requires retraining | Prompt-level adaptation |
| Root cause analysis | Offline batch | Real-time, per-part |

**Key advantage**: The LLM doesn't just flag defective parts — it can **reason about why** a part is defective in real-time. "The weld bead on the left joint is 0.3mm too narrow, likely caused by wire feed speed variation" — this level of analysis runs on every part, not on sampled ones sent to a separate QC station.

### Assistive Robotics: Natural Interaction at Human Speed

**Scenario**: A robotic arm assisting an elderly person with meal preparation.

GPU-based interaction:
```
User: "Hand me the salt"
     ↓
[1s] Speech-to-text processing
[3s] LLM understands request + plans action (grasp the salt shaker)
[2s] Motion planning + trajectory generation
[3s] Visual verification + safety check
     ↓
Robot reaches for salt 9 seconds after the request
     ↓
User has already reached for it themselves
```

Cerebras-based interaction:
```
User: "Hand me the salt"
     ↓
[50ms] Speech-to-text + LLM processes intent
[50ms] Action planning + safety check
[50ms] Visual servoing + trajectory generation (closed loop)
     ↓
Robot reaches for salt 150ms after request end
     ↓
User hasn't finished speaking yet
```

**Critical threshold crossed**: 150ms versus 9 seconds. At 150ms, the interaction feels simultaneous. At 9 seconds, the user has already moved on.

### Safety: Hazard Detection Faster Than Human Reaction

Human reaction time to visual hazards: ~200ms.

| Hazard Type | GPU Detection + Action | Cerebras Detection + Action | Human |
|------------|----------------------|---------------------------|-------|
| Falling object | 3-5s (misses it) | 100-150ms (catches it) | 200-250ms |
| Intrusion into safety zone | 3-5s | 100-150ms | 200-300ms |
| Fire/smoke detection | 3-5s | 100-150ms | 1-3s (distracted) |
| Equipment malfunction | 3-5s | 100-150ms | 500ms-2s |

**Cerebras is faster than a human at every safety-relevant timescale.** This is the difference between a robot that can intervene (new) and a robot that can only report (old).

---

## 5. Competitive Analysis: Why Cerebras Wins on Latency

### The Competitive Landscape

| Provider | Architecture | Best Reported Latency | Batch-1 Performance | Notes |
|----------|-------------|---------------------|-------------------|-------|
| **Cerebras** | WSE-3 (wafer-scale) | **5.3ms** (measured) | **1,850 tok/s** | No penalty for batch=1 |
| Groq | LPU (tensor streaming) | 10-50ms claimed | ~500 tok/s (Llama 2 70B) | SRAM limited (<230MB) |
| Google | TPU v5p | 100-500ms | Throughput optimized | Batching-dependent |
| NVIDIA | H100 GPU | 500ms-3s | ~100 tok/s (batch=1) | 5-10x penalty for batch=1 |
| AWS | Inferentia2 | 200ms-1s | ~150 tok/s (batch=1) | Cost-optimized |
| Together AI | GPU cluster | 500ms-2s | Throughput optimized | Standard GPU infra |
| Fireworks AI | GPU cluster | 300ms-1.5s | Throughput optimized | Standard GPU infra |

Sources: Cerebras API live test (June 28, 2026, 5.3ms for Gemma 4 31B); Groq API documented benchmarks; Google Vertex AI documentation; NVIDIA Triton Inference Server benchmarks; AWS Inferentia documentation.

### What Makes Cerebras Unique

1. **Wafer-Scale Engine (WSE-3)**: 4 trillion transistors, 900,000 AI cores on a single wafer. No inter-chip communication — every core is connected on-die with ultra-low latency.

2. **No Batching Penalty**: GPU architectures achieve efficiency through batching (amortizing memory reads across many queries). This adds latency. Cerebras's architecture processes each query independently at maximum speed. For real-time applications, this is the decisive advantage.

3. **Deterministic Latency**: GPU inference latency varies wildly based on GPU utilization, batch scheduler, and memory bandwidth contention. Cerebras provides consistent sub-10ms response times.

4. **Fine-Grained Sparsity Support**: Cerebras's architecture natively supports fine-grained sparsity, enabling models to skip computation for zero-valued activations — further reducing effective latency.

5. **The SRAM Limitation**: Groq's LPU is fast but limited by on-chip SRAM (~230MB total). This restricts model size or forces model sharding across multiple LPUs, adding latency. Cerebras's WSE-3 has vastly more on-wafer memory (40GB+ SRAM), fitting larger models without sharding.

### Why "Fast Inference" Isn't All Equal

Groq, Cerebras, and Google's TPU all claim "fast inference." But the architecture matters:

- **Groq**: Fast at **compute** (matrix multiply), but limited by SRAM — can only hold small models or must split. Best for small, specialized models.

- **Google TPU**: Fast at **throughput** (many queries), not **latency** (single query). Optimized for Google's internal batch workloads.

- **Cerebras**: Fast at **both** — single-query latency and throughput scale together because the entire model lives on one wafer with full-bandwidth connectivity.

---

## 6. The "Impossible Before" Examples

### Example 1: Multi-Robot Search Team With Real-Time Coordination

**The scenario**: 6 drone agents search a disaster zone after an earthquake. Each drone has a camera. A command agent coordinates their search pattern.

**On GPU (impossible):**
- Each drone sends an image to its agent → 3s latency
- Command agent processes 6 reports → 3s latency
- Negotiation for search pattern adjustment: 3 rounds → 18s
- Re-planning cycle: 24s total
- By the time the team re-plans, conditions have changed
- Result: agents operate independently, coordination is purely strategic (not tactical)

**On Cerebras (possible):**
- 6 drone agents analyze frames in parallel: 50ms per agent (can be batched)
- Command agent synthesizes all 6 reports: 50ms
- Negotiation cycle: 3 rounds → 450ms
- Re-planning cycle: 550ms total
- Result: the team re-plans every half-second, adjusting search patterns to cover areas the others missed

**Why it matters**: 24s vs 550ms is not a speedup. It's a **mode change**. At 24s, coordination is strategic advice. At 550ms, it's real-time command and control.

### Example 2: Continuously-Adaptive Manufacturing Inspection

**The scenario**: A camera on a high-speed assembly line inspects every part at 120 parts/minute. The LLM must detect defects and adjust for lighting/angle/part variants.

**On GPU (impossible):**
- 2 FPS frame rate means 1 in 60 parts is inspected
- 3s latency means the line has advanced 6 positions before a defect is detected
- Cannot provide real-time feedback to the robot arm
- Sampling-based inspection misses transient defects
- Defect detection is purely reactive (post-hoc statistics)

**On Cerebras (possible):**
- 10-20 FPS frame rate means every part is inspected, some twice
- 50ms latency means the line advances 1 position before a defect is reported
- Robot arm can reject a defective part at the next station
- Continuous adaptation to changing conditions (new part types, lighting shifts)
- Defect reasoning catches novel defects: "This looks like a crack, not a scratch"

**Why it matters**: 1 part inspected vs all parts inspected. Post-hoc statistics vs real-time control. This is the difference between a quality control **department** and a quality control **system**.

### Example 3: 10-Agent Warehouse Orchestration

**The scenario**: A warehouse with 10 robots (picking, packing, sorting, transporting). An overhead camera provides the global view. Agents negotiate task assignment in real-time.

**On GPU (impossible):**
- 10 agents each analyzing the scene: 10 agents × 2s = 20s (serial) or 2s (parallel, batch limited)
- Task negotiation: 3 rounds per assignment → 6s minimum
- Bottleneck detection: 2s
- Re-assignment: 2s
- Total cycle: 12-30s
- During this time, the bottleneck has cascaded to 3 other stations
- Result: coordination is asynchronous, agents work on stale assignments

**On Cerebras (possible):**
- 10 agents analyzing scene in parallel: 50ms (no batching penalty)
- Task negotiation: 3 rounds → 150ms
- Bottleneck detection: 50ms
- Re-assignment: 50ms
- Total cycle: 300ms
- During this time, nothing has changed — the bottleneck was caught in real-time
- Result: agents operate as a single coordinated entity

**Why it matters**: At 12-30s cycles, the warehouse manager plans the day. At 300ms cycles, the warehouse runs itself.

### Example 4: Voice-Activated Safety Stop

**The scenario**: A factory robot is moving a heavy load. A worker shouts "STOP!" The system must understand the command and halt the robot.

**On GPU (impossible):**
- Speech-to-text: 200ms (streaming)
- LLM intent understanding + action: 2-3s
- Robot brake signal: 50ms
- Total: 2.25-3.25 seconds
- At 2m/s arm speed, the robot travels 4.5-6.5 meters before stopping
- The worker has already been hit

**On Cerebras (possible):**
- Speech-to-text: 50ms (streaming, partial)
- LLM intent understanding + action: 5-50ms (Cerebras)
- Robot brake signal: 10ms
- Total: 65-110ms
- At 2m/s arm speed, the robot travels 13-22cm before stopping
- The worker is safe

**Why it matters**: This is the difference between a safety system and a fatality reporting system. A 3-second GPU system cannot save anyone from a fast-moving robot. A 65ms Cerebras system can.

### Example 5: Real-Time Multi-Agent Competitive Game

**The scenario**: A real-time strategy game where 5 agent-controlled units compete against 5 human-controlled units. Agents must coordinate tactics (flanking, retreating, resource allocation) in real-time.

**On GPU (impossible):**
- Each agent processes the game state: 2s per agent × 5 agents = 10s serial
- Tactical negotiation: 3 rounds × 5s = 15s
- Action generation: 2s
- Total decision cycle: 27s
- The game ends before the agents make their first move
- Human opponents have taken 5-10 actions each
- Result: LLM agents cannot play real-time games

**On Cerebras (possible):**
- Agents process state in parallel: 50ms
- Tactical negotiation: 3 rounds × 50ms = 150ms
- Action generation: 50ms
- Total decision cycle: 250ms
- Agents respond faster than human reaction time
- Result: LLM agents play at superhuman speed while also reasoning strategically

**Why it matters**: This demonstrates the qualitative difference across all real-time multi-agent domains — from gaming to autonomous driving to drone swarms. The ability to coordinate **and** react in real-time is a new capability, not just a faster version of an old one.

---

## 7. Architectural Implications: New Patterns, New Assumptions

### The Old Assumptions (GPU Era)

1. LLM calls are expensive — minimize them
2. Batch your queries for efficiency
3. Cache aggressively, plan ahead
4. Streaming is the default (hide latency)
5. Async everywhere (queue jobs, poll results)
6. The LLM is a "slow brain" — use it for big decisions only
7. One LLM per system (too expensive for more)

### The New Assumptions (Cerebras Era)

1. LLM calls are cheap — maximize them (each is 5-50ms)
2. Batch size 1 is optimal for latency-sensitive apps
3. Read and react in real-time — no planning cache needed
4. Synchronous is the default (response is faster than user perception)
5. Sync is fine — the latency budget is already in control loop margins
6. The LLM is a "fast brain" — use it for all decisions, even trivial ones
7. Multiple LLMs per system — agents, sub-agents, safety monitors, all in parallel

### New Architectural Patterns

**Pattern 1: The Continuous Reasoning Loop**
```
loop {
    observation = sense()
    understanding = llm(observation)    // 50ms — what do I see?
    goal = llm(understanding, context)  // 50ms — what should I do?
    action = safety_check(goal)         // 10ms — is this safe?
    act(action)                         // <50ms — do it
}
// Total: 6-10 Hz full-stack reasoning-to-action
```

**Pattern 2: The Democratic Agent Collective**
```
loop {
    observations = [agent_i.sense() for i in range(10)]  // parallel
    proposals = [llm(o) for o in observations]            // 50ms parallel
    consensus = coordinator_llm(proposals)                 // 50ms
    broadcast(consensus)                                   // 10ms
    act(consensus)
}
// Total: 5-8 Hz multi-agent coordination cycle
```

**Pattern 3: The Safety Monitor Overlay**
```
// In parallel with the main control loop:
loop {
    frame = camera.capture()
    hazard = llm(frame, "Is there any safety hazard?")  // 50ms
    if hazard:
        emergency_stop()                                 // 10ms
}
// Runs at 10-20 Hz, overlapping with primary control
// GPU could never afford a parallel LLM safety monitor
```

---

## 8. The Thesis Statement

> **Sub-100ms LLM inference is not an incremental improvement. It is a phase transition in what is architecturally possible with AI systems.**
>
> The 1-second barrier separated fast reactive systems from slow deliberative systems. Cerebras eliminates this barrier entirely. For the first time, general intelligence can participate in the real-time control loops that govern the physical world.
>
> This enables:
> - LLMs that **control** robots, not just plan for them
> - Multi-agent teams that **coordinate** in real-time, not asynchronously
> - Vision systems that **reason** about every frame, not sparse samples
> - Safety systems that **intervene** faster than human reaction
> - Interactive systems that **respond** at the speed of conversation, not the speed of typing
>
> The architecture of the old world was built around the latency constraint. The architecture of the new world begins with the assumption that intelligence is always available, always fast, and can participate in every decision cycle — no matter how tight the timing.

---

## Appendix: Sources and References

### Cerebras-Specific Sources
- Cerebras API live test: 5.3ms response time, 1,850 tok/s for Gemma 4 31B (June 28, 2026)
- Cerebras "Designing for Cerebras" developer guide — architectural patterns for ultra-fast inference
- Cerebras WSE-3 specifications: 4 trillion transistors, 900,000 AI cores, wafer-scale integration

### Robotics Latency Thresholds
- Human reaction times: Jensen (2006), "Clockwise" review; standard psychophysics literature
- Visual servo control: Hutchinson, Hager & Corke (1996), "A Tutorial on Visual Servo Control" (IEEE T-RA)
- Robot arm safety: ISO 10218-1/2, ISO/TS 15066 (collaborative robot safety standards)
- Drone control loops: Mahony, Kumar & Corke (2012), "Multirotor Aerial Vehicles" (IEEE RAM)
- Voice latency: ITU-T G.114 (one-way transmission time)

### GPU Inference Constraints
- Ahn et al. (2022), "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances" (SayCan)
- Brohan et al. (2023), "RT-2: Vision-Language-Action Models" — VLM-based robotics at GPU speeds
- Huang et al. (2022), "Inner Monologue: Embodied Reasoning through Planning with Language Models"
- NVIDIA Triton Inference Server benchmarks (batch-size latency tradeoffs)

### Multi-Agent Systems
- Li et al. (2024), "Cooperating with Language Agents" — multi-agent coordination constraints
- Park et al. (2023), "Generative Agents: Interactive Simulacra of Human Behavior" — async agent interaction

### Competitive Landscape
- Groq LPU: Documented benchmarks for Llama 2 70B (~480 tok/s); SRAM limitation (~230MB)
- Google TPU v5p: Cloud TPU documentation (throughput-optimized, batch-dependent latency)
- AWS Inferentia2: Published benchmarks (cost-optimized, not latency-first)
- Together AI / Fireworks AI: Standard GPU infrastructure benchmarks

### Note on Sources
Where specific numbers from web sources are discussed, they reflect the best available public data as of June 2026. The core thesis — that sub-100ms LLM inference constitutes a phase transition in system architecture — is supported by the quantitative thresholds documented in the robotics and human-computer interaction literature cited above.
