# Hackathon Rules & FAQ — Gemma 4 24-Hour Hackathon

> Source: Official Cerebras Discord announcement + live API test (June 28 2026)
> **Deadline: Mon June 29, 10:00 AM PT** (~23 hours from kickoff)

---

## Prizes

| Track | Prize | Where to Submit |
|-------|-------|-----------------|
| Track 1: Multiverse Agents | **$2,000** | `#g4hackathon-multiverse-agents` |
| Track 2: People's Choice | **$2,000** | X/Twitter — tag `@Cerebras` + `@googlegemma` |
| Track 3: Enterprise Impact | **$1,000** | `#g4hackathon-enterprise-impact` |

You may submit to multiple tracks. **Each track = a separate Discord post.**

---

## Timeline

| Event | Time (PDT) |
|-------|------------|
| Kickoff + live Q&A | Sun June 28, 10:00 AM |
| Live support ends | Sun June 28, 12:30 PM |
| Overnight support | Limited / intermittent |
| **Submission deadline** | **Mon June 29, 10:00 AM** |

---

## Rate Limits (Hackathon Elevated Tier — Discord-confirmed, not in public docs)

| Limit | Value | What It Means for Us |
|-------|-------|----------------------|
| Requests per minute | **100 RPM** | 1.67 req/s sustained |
| Input tokens per minute | **100K TPM** | Fine for our payloads |
| Max sequence length | **65K tokens** | Fine |
| Max completion length | **32K tokens** | Fine |
| Images per request | **5 max** | VisionAgent sends 1–2 |
| Max payload | **10 MB** | ~3–4 base64 JPEGs |

### What the limits actually mean for the demo

- Pipeline = Vision + Action + Safety = **3 req/cycle**
- Sustained rate = 100 RPM ÷ 3 = **33 cycles/min = 0.55 Hz** (not 5 Hz as claimed)
- Burst works: 5 req in ~185ms is achievable for a **~15–20 second window**
- 60-second video **must be pre-recorded / edited** — live continuous demo at full speed is impossible

---

## Model Details

| Field | Value |
|-------|--------|
| Model ID | `gemma-4-31b` (only Cerebras variant available) |
| Endpoint | `https://api.cerebras.ai/v1` — standard, no preview endpoint |
| Speed (live-measured June 28) | **~1,850 tok/s** |
| Latency (live-measured June 28) | **5.3ms total** (0.28ms queue · 1.8ms prompt · 1.8ms completion) |
| Image format | **Base64 data URIs only** — `data:image/jpeg;base64,...` |
| Hosted image URLs | ❌ — 400 error |
| Reasoning | Off by default; `reasoning_effort` opt-in. All levels behave identically. |
| Structured outputs | ✅ `response_format: {type: "json_schema"}` with strict mode |
| Tool calling | ✅ Parallel by default |
| Streaming | ✅ ~200 events/sec |

### Critical API constraint: tools ⊕ response_format are mutually exclusive

> `tools` and `response_format` **cannot be used in the same request.**

`CommandCenterRoot._route()` uses `response_format` for the routing JSON schema.
This means it **cannot simultaneously use tool calling** — it is prompt-in → structured-JSON-out only.
Any branch that needs tool calling must be a separate request.

---

## Submission Rules

- Multiple track entries allowed — one Discord post per track
- Can resubmit / update any time before the deadline
- Pre-existing scaffolding is fine — **core logic must use Gemma 4 on Cerebras**
- Any team size (2 recommended)

---

## Demo Video Requirements

| Requirement | Detail |
|-------------|--------|
| Max length | **60 seconds** |
| Required | Clearly show Cerebras speed impact |
| Recommended | Side-by-side comparison with a GPU-based provider |
| Prohibited | Personal / sensitive data visible on screen |

The side-by-side GPU comparison is listed as "recommended" but effectively required to score on "Speed in Action."

---

## Judging Criteria

### Track 1: Multiverse Agents

| Criterion | Weight | What Judges Want |
|-----------|--------|-----------------|
| Agent Collaboration | **High** | Real multi-agent coordination — not just sequential LLM calls |
| Multimodal Intelligence | **High** | Meaningful image + text integration (not decorative) |
| Speed in Action | **High** | Visible latency delta — side-by-side preferred |
| Innovation | Medium | Physical AI, robotics, embodied agents called out explicitly |

### Track 3: Enterprise Impact

Stated examples: enterprise search, multimodal RAG, incident response, cybersecurity, customer support, knowledge management.
**Search & rescue is not on this list.** Reframe as warehouse safety or manufacturing inspection to score here.

---

## Our Status vs. the Rules

| Rule | Status |
|------|--------|
| Core uses Gemma 4 on Cerebras | ✅ All agents call `gemma-4-31b` |
| Multimodal | ✅ VisionAgent sends base64 images |
| Multi-agent coordination | ✅ Vision → Action → Safety + `CommandCenterRoot` router |
| Speed visibly demonstrated | ⚠️ Timing logged in code; no side-by-side GPU comparison built |
| GPU provider API key | ❌ Not set up |
| Router tested against live API | ❌ `_route()` with `response_format` never called end-to-end |
| 60s demo video | ❌ Not recorded |
| Submission Discord post | ❌ Not submitted |