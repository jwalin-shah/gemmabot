# 03 Idea Brainstorm

TODO: write documentation.

## Idea Candidates (Ranked)

### 🥇 Ant-Scale Warehouse (Score: 33/40)
- **Novelty**: 7 | **Impact**: 9 | **Feasibility**: 7 | **Speed showcase**: 10
- 6 agent-robots (simulated) share overhead camera feed
- Bottleneck appears → agents re-route packages in real-time via Gemma 4 tool calls
- Side-by-side timer: CPU orchestration (2s) vs Cerebras (<100ms)
- **Demo hook**: "Laggy vs liquid" split-screen
- **Why it wins**: Speed is the entire point

### 🥈 Disaster Rescue Countdown (Score: 34/40)
- **Novelty**: 7 | **Impact**: 10 | **Feasibility**: 8 | **Speed showcase**: 9
- 5 search agents + 1 command agent. Collapsed building (simulated)
- Each agent searches one quadrant, sends camera images to command
- 60s demo = 60s in-sim timer. Speed = survivors saved
- **Demo hook**: Countdown + scoreboard. Emotional stakes

### 🥉 Blind Spot Handoff (Score: 30/40)
- **Novelty**: 8 | **Impact**: 8 | **Feasibility**: 6 | **Speed showcase**: 8
- 3 camera agents each see 1/3 of an assembly line
- Part falls in blind spot → agents share multimodal context to reconstruct
- Speed = arm doesnt crash vs collision
- **Demo hook**: Tension, visible "save" vs "fail"

### 4️⃣ Robot Teacher / One-Shot Learning (Score: 30/40)
- **Novelty**: 9 | **Impact**: 8 | **Feasibility**: 7 | **Speed showcase**: 6
- Human performs task (5 image frames). 3 robot agents watch via Gemma 4 multimodal
- Each performs differently (different morphology). Teacher evaluates all 3
- Teaching loop runs in sub-second cycles
- **Demo hook**: "Robots learning from watching"

### 5️⃣ Adversarial Red vs Blue (Score: 33/40)
- **Novelty**: 9 | **Impact**: 9 | **Feasibility**: 6 | **Speed showcase**: 9
- 2 red-team camera agents (trying to sneak past) vs 3 blue-team security agents
- Red uses Gemma 4 to analyze blues blind spots. Blue coordinates to cover them
- Speed determines who wins
- **Demo hook**: Game-like, competitive, clear winner/loser

### 6️⃣ Kitchen Chaos (Score: 30/40)
- **Novelty**: 8 | **Impact**: 9 | **Feasibility**: 8 | **Speed showcase**: 6
- 4 chef agents (grill, prep, plating, expo) + 1 expediter
- Fire starts (simulated smoke on camera) → expedition re-routes
- **Demo hook**: Narrative arc — normal → fire → chaos → recovery

### 7️⃣ Construction Site Inspection (Score: 25/40)
- **Novelty**: 6 | **Impact**: 6 | **Feasibility**: 9 | **Speed showcase**: 5
- 4 drone camera agents + 1 safety inspector. Detect cracks, missing bolts
- Inspector aggregates 5 images → decide: continue work or evacuate
- **Demo hook**: Safety is serious

## Design Space

| Dimension | Options |
|-----------|---------|
| Robot type | Arm, wheeled drone, multi-robot fleet |
| Environment | Warehouse, factory, kitchen, lab, disaster zone |
| Sensors | Single camera, multi-camera, overhead, robot-mounted |
| Agent roles | Vision, planner, safety, executor, supervisor, adversary |
| Failure mode | Collision, missed item, safety violation, time pressure |
| Demo format | Terminal, browser sim, video overlay, split-screen comparison |