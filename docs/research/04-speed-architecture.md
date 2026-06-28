# Designing for Cerebras Speed — Architecture Patterns

> Key insights from Cerebras "Designing for Cerebras" guide + latency analysis

## The Core Insight

At GPU speeds (1-5s per response), you design AROUND latency:
- Async job queues
- Elaborate progress/loading UIs
- Batch processing
- Streaming as default

At Cerebras speeds (5-100ms per response), the simplest architecture wins:
- Synchronous request paths
- No progress UI needed
- Real-time agent loops
- Stream only for long outputs

## Pattern 1: Synchronous AI in the Request Path

```python
# GPU approach: job queue + polling (overengineered)
job = queue.submit(llm_call)
result = poll_for_result(job.id)  # 3-5s later

# Cerebras approach: just call it
result = client.chat.completions.create(...)  # 50ms later
return result
```

## Pattern 2: Multi-Step Agent Loops in Real-Time

A coding task needing 5-10 LLM steps:
- **GPU**: 30-60 seconds → must run as background job
- **Cerebras**: 2-3 seconds → runs synchronously in request path

For robotics: vision → action → safety → execute loop:
- **GPU**: 2-5 Hz (skips frames, reactive only at high level)
- **Cerebras**: 10-20 Hz (every frame processed, LLM in low-level loop)

## Pattern 3: Stream Selectively

- **Short responses (<200 tok)**: synchronous — completes faster than streaming
- **Long responses (>200 tok)**: stream — user sees content immediately

## Pattern 4: Dont Let UI Become Bottleneck

At Cerebras speeds, the UI rendering can take longer than inference.
- Buffer incoming tokens, update on 50ms timer instead of per-event
- Skip progress animations for fast operations
- Measure full request cycle, not just model call

## Pattern 5: More Agent Steps Are Free

Cerebras can afford more LLM calls in the same time budget:
- Read → plan → edit → test → fix → verify
- All within a single synchronous request

## For This Hackathon

The speed enables:
1. **Reactive control**: LLM in the low-level loop, not just high-level planning
2. **Continuous perception**: Every camera frame processed, no skipping
3. **Real-time multi-agent negotiation**: Agents debate plans in sub-second cycles
4. **No batching needed**: Each agent call is synchronous, no queue management
5. **Closed-loop visual servoing**: See → think → act in <100ms