# Real GPU Comparison: Cerebras vs OpenRouter (Gemma 4 31B)

> Live benchmark: June 28, 2026. Same model (Gemma 4 31B), same prompts, same output length.
> Cerebras via api.cerebras.ai, OpenRouter via openrouter.ai (routes to GPU backend).

## The Headline

| Metric | OpenRouter (GPU) | Cerebras (WSE-3) | **Speedup** |
|--------|:----------------:|:----------------:|:-----------:|
| **Wall time** (avg) | **9,773 ms** | **465 ms** | **21.0x** |
| **Tokens/second** (avg) | **25 TPS** | **505 TPS** | **20.2x** |
| **Consistency** | 6,027 – 15,049 ms (3x jitter) | 451 – 481 ms (±15ms) | **Stable** |
| Cost per req | ~$0.000005 (fractional) | N/A (dev tier) | — |

## Raw Data

### Prompt: "Write a story about a robot in exactly 3 paragraphs." (~200 output tok)

| Run | OpenRouter | Cerebras |
|-----|:----------:|:--------:|
| 1 | 6,027 ms (36 TPS) | 481 ms (493 TPS) |
| 2 | 8,243 ms (26 TPS) | 451 ms (510 TPS) |
| 3 | 15,049 ms (13 TPS) | 464 ms (513 TPS) |

### Simple Prompt: "Say hello in 2 words" (~5 output tok)

| Run | OpenRouter | Cerebras |
|-----|:----------:|:--------:|
| 1 | 384 ms | 235 ms (5.4ms model) |
| 2 | — | 284 ms (4.3ms model) |
| 3 | — | 277 ms (5.5ms model) |
| 4 | — | 191 ms (5.5ms model) |
| 5 | — | 297 ms (5.2ms model) |
| **Avg** | **384 ms** | **257 ms** |

### Also tested: Pioneer (google/gemma-4-31B-it)
| Metric | Pioneer | Cerebras | Speedup |
|--------|:-------:|:--------:|:-------:|
| Avg wall | 1,129 ms | 257 ms | 4.4x |
| Range | 722 – 2,157 ms | 191 – 297 ms | — |

Pioneer is faster than OpenRouter but still 4x slower than Cerebras and less consistent.

## Key Takeaways for the Demo

1. **21x speedup is real** — same model, same hardware class (Gemma 4 31B), just Cerebras's WSE-3 architecture vs GPU routing
2. **GPU is unpredictable** — OpenRouter swings 6-15s per request. Cerebras is rock-solid at ~460ms
3. **The gap grows with output length** — more tokens = bigger TPS difference
4. **Network latency is shared** — both providers take ~150ms RTT from this machine. The 21x gap is server-side

## Configuration

```env
# Cerebras (direct)
CEREBRAS_API_KEY=csk-...
CEREBRAS_BASE_URL=https://api.cerebras.ai
CEREBRAS_MODEL=gemma-4-31b

# Comparison (OpenRouter)
COMPARISON_API_KEY=sk-or-v1-...
COMPARISON_BASE_URL=https://openrouter.ai/api/v1
COMPARISON_MODEL=google/gemma-4-31b-it
```

## What the Judges Care About

The 21x gap translates to:
- Cerebras: **~200-400ms per pipeline loop** → 2.5-5 Hz, real-time reactive
- GPU: **~6-15s per decision** → 0.07-0.17 Hz, always planning for the past
- In 60 seconds: Cerebras makes **~150 decisions** vs GPU makes **~6 decisions**

That's the demo. 150 decisions vs 6. Frame-by-frame visual reasoning vs blind between frames.
