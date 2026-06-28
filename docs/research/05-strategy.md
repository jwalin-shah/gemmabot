# Hackathon Strategy — What Wins in 24 Hours

## Judging Criteria Breakdown (Track 1)

| Criterion | Weight | What Judges Want |
|-----------|--------|------------------|
| Agent Collaboration | High | Effective coordination between multiple agents, not just sequential calls |
| Multimodal Intelligence | High | Meaningful use of Gemma 4 31B with text + images |
| Speed in Action | High | Demonstrates impact of Cerebras ultra-fast inference |
| Innovation | Medium | Creative, outside-the-box applications, physical AI, embodied agents |

## Winning Strategy

### 1. Speed Must Be The Star

The judging explicitly says: "clearly demonstrate how fast inference improves the user experience" and "Recommended: Include a side-by-side comparison with a GPU-based provider."

**Must-have**: A visible latency comparison. Cerebras vs GPU provider side-by-side.

### 2. 60-Second Demo Video Structure

| Time | Content |
|------|---------|
| 0-5s | Hook: split screen Cerebras vs GPU |
| 5-20s | Show the problem / setup |
| 20-40s | The pipeline in action (fast cuts) |
| 40-50s | Speed comparison + side-by-side |
| 50-60s | Result + end card with tracks |

### 3. What to Avoid

- Spending time on real hardware (50% of 24h goes to debugging)
- Over-engineering the UI (terminal output is fine)
- Building more than 1-2 agent interactions (depth > breadth)
- Complex state machines (prompts + tool calling is simpler)

### 4. What to Prioritize

1. **Working pipeline** (vision → action → safety → execute)
2. **Speed comparison** (side-by-side timer)
3. **Demo video script** (plan before filming)
4. **README + submission** (polish matters)

### 5. Deadline Check

| Event | Time (PDT) |
|-------|------------|
| Kickoff + Q&A | Sun June 28, 10:00 AM |
| Live support ends | Sun June 28, 12:30 PM |
| Submission deadline | Mon June 29, 10:00 AM |
| Intermittent support overnight | Limited |