# Gemma 4 31B on Cerebras — Capabilities Reference

> Model ID: `gemma-4-31b` on `https://api.cerebras.ai/v1`
> Source: Cerebras Inference Docs + live API test

## Model Stats

| Metric | Value |
|--------|-------|
| Speed | ~1,850 tokens/sec |
| Context (free) | 65K tokens |
| Context (paid) | 131K tokens |
| Max output (free) | 32K tokens |
| Max output (paid) | 40K tokens |
| Pricing | Developer tier — essentially free for experimentation |
| Modality | Text + Image input, Text output |

## Image/Multimodal

- **Format**: Base64 data URIs only (`data:image/png;base64,...` or `data:image/jpeg;base64,...`)
- **No HTTPS URLs** — 400 error if you pass a URL
- **Max 5 images per request**, 10 MB total payload
- **Token cost**: Up to 280 tokens per image (capped, auto-scaled)
- **Endpoint**: Chat Completions only (not Completions endpoint)
- **Quality**: Good for objects, diagrams, screenshots, documents, real-world scenes

## Capabilities

| Feature | Support | Notes |
|---------|---------|-------|
| Tool Calling | ✅ Yes | OpenAI-compatible `tools` array |
| Parallel Tool Calls | ✅ Default on | `parallel_tool_calls=True` |
| Strict Mode | ✅ Yes | `strict: true` + `additionalProperties: false` |
| Structured Outputs | ✅ Yes | `response_format: { type: "json_schema", ... }` |
| Reasoning | ✅ Opt-in | `reasoning_effort: "none"|"low"|"medium"|"high"` (all active = same) |
| Streaming | ✅ Yes | ~200 events/sec, multiple tokens per event |
| Prompt Caching | ✅ Auto | Automatic; explicit key via account enablement |
| Sampling | ✅ | temperature, top_p, frequency_penalty, seed, stop, logprobs |

## API Test (Live, June 28 2026)

```
Total time: 5.3ms
Queue time: 0.28ms
Prompt time: 1.8ms
Completion time: 1.8ms
```

## Key Limitations

1. `tools` + `response_format` cannot be used in the same request
2. Reasoning levels (low/medium/high) are all equivalent — no graduated control
3. No raw/hidden reasoning formats (only `parsed`)
4. Free tier: 5 RPM, 30K input TPM, 1M tokens/day
5. Pay-as-you-go: 300 RPM, 500K input TPM
6. Image via base64 only — no URL support