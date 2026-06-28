# Multi-Agent Robotics Architectures

> Patterns for combining multiple LLM agents in a robotics context

## The 3 Dominant Patterns

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| Monolithic VLA | One model: pixels → text → actions | Maximum generalization, hard to debug |
| Orchestrator + Specialists | Central LLM routes to sub-agents | Modular, easy to swap components |
| Hierarchical Plan→Execute | High-level LLM plans, low-level executes | Long-horizon tasks needing re-planning |

## Recommended: Orchestrator + Specialists

Best for a 24h hackathon — each agent independently testable:

```
Robot Camera Frame
        ↓
[Orchestrator Agent] ── assigns tasks ──┐
        │                                │
        ├── [Vision Agent] analyze scene │
        ├── [Action Agent] plan actions  │
        ├── [Safety Agent] validate      │
        └── [Execute Agent] run commands │
        │                                │
        └──── collect results ←──────────┘
```

## Context Sharing Mechanisms

| Method | Latency | Complexity | Best For |
|--------|---------|------------|----------|
| Shared JSON state | Low | Medium | Scene graphs, structured data |
| Tool calling | Medium | Low | Agent asks another agent for info |
| Full prompt concatenation | High | Lowest | Simple chains, small context |
| Message passing | Deterministic | High | Real-time ROS2 control loops |

## Key Design Principles (from Designing for Cerebras)

1. **Synchronous agent loops**: Cerebras is fast enough that agent pipelines run in the request path — no job queues needed
2. **Stream only when needed**: Short responses (<200 tokens) complete faster than streaming overhead
3. **More steps are free**: Multi-turn agent loops that take 30-60s on GPU complete in 2-3s on Cerebras
4. **Dont let UI become bottleneck**: At Cerebras speeds, rendering can take longer than inference