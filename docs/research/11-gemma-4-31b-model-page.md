# Gemma 4 31B
Source: https://inference-docs.cerebras.ai/models/gemma-4-31b

This model excels at multimodal reasoning across screenshots, documents, diagrams, and design assets. Ideal for visual agentic workflows, image-aware copilots, and teams migrating from closed multimodal APIs to an open model.

> ⏳ **Preview**
  This model is coming soon.



  Model ID: `gemma-4-31b`. Speed: \~1500 tokens/sec. Context window: 65k tokens (free tier), 131k tokens (paid). Max output: 32k tokens (free tier), 40k tokens (paid). Pricing: coming soon. Modality: multimodal — accepts text and image inputs (base64 PNG or JPEG data URI only; external URLs not supported), produces text output. Max 5 images per request, 10 MB total image payload. Image inputs are only available in Chat Completions; the Completions endpoint does not support images. Capabilities: Image Inputs, Reasoning, Streaming, Sampling Controls, Structured Outputs, Tool Calling, Parallel Tool Calling, Prompt Caching. Reasoning is disabled by default; enable it with the `reasoning_effort` parameter. The `raw` and `hidden` reasoning formats are not supported. Structured outputs and tool calling with `strict: true` (constrained decoding) are supported. Rate limits (free tier): 5 requests/min, 30k input tokens/min, 1M tokens/day. Rate limits (Pay as You Go): 300 requests/min, 500k input tokens/min. Notes: Recommended starting parameters: temperature=1.0, top\_p=0.95.




