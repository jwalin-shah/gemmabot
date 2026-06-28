# Cerebras Inference API Reference

> **Source:** Official Cerebras Inference documentation at https://inference-docs.cerebras.ai
> **Base URL:** `https://api.cerebras.ai/v1`
> **Auth:** `Authorization: Bearer YOUR_CEREBRAS_API_KEY`
> **SDK:** `pip install cerebras-cloud-sdk` or OpenAI SDK with `base_url="https://api.cerebras.ai/v1"`

---

## Table of Contents

1. [Chat Completions API](#1-chat-completions-api)
2. [Multimodal / Image Inputs](#2-multimodal--image-inputs)
3. [Structured Outputs](#3-structured-outputs)
4. [Tool Calling / Function Calling](#4-tool-calling--function-calling)
5. [Reasoning / Thinking](#5-reasoning--thinking)
6. [Streaming](#6-streaming)
7. [Models](#7-models)
8. [Error Handling](#8-error-handling)
9. [Prompt Caching](#9-prompt-caching)
10. [Payload Optimization](#10-payload-optimization)
11. [Rate Limits](#11-rate-limits)
12. [Best Practices](#12-best-practices)

---

## 1. Chat Completions API

### Endpoint

```
POST https://api.cerebras.ai/v1/chat/completions
```

### Headers

```
Authorization: Bearer YOUR_CEREBRAS_API_KEY
Content-Type: application/json
```

### Request Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model` | string | **Yes** | Model ID (e.g., `gpt-oss-120b`). Call `GET /v1/models` for available IDs. |
| `messages` | array | **Yes** | Conversation messages with roles: `system`, `developer`, `user`, `assistant`, `tool`. |
| `max_completion_tokens` | integer | No | **Preferred.** Max tokens to generate, including reasoning tokens. |
| `max_tokens` | integer | No | **Legacy** alias for `max_completion_tokens`. Prefer `max_completion_tokens`. |
| `temperature` | number | No | Sampling temperature, 0-2. Higher = more random. Default 1. |
| `top_p` | number | No | Nucleus sampling, 0-1. Use instead of `temperature`, not both. |
| `stop` | string or array | No | Up to 4 stop sequences. Generation halts before emitting them. |
| `seed` | integer | No | Best-effort deterministic sampling. Same seed + params -> near-identical output. |
| `n` | integer | No | Number of completions to generate. Default 1. |
| `frequency_penalty` | number | No | -2.0 to 2.0. Penalizes tokens by existing frequency. |
| `presence_penalty` | number | No | -2.0 to 2.0. Penalizes tokens that have appeared at all. |
| `stream` | boolean | No | If `true`, stream Server-Sent Events of `chat.completion.chunk` objects. |
| `tools` | array | No | Tool/function definitions the model may call. |
| `tool_choice` | string or object | No | `auto`, `none`, `required`, or `{"type":"function","function":{"name":"..."}}`. |
| `parallel_tool_calls` | boolean | No | Allow multiple tool calls in one turn. Default `true`. |
| `response_format` | object | No | `{"type":"json_object"}` or `{"type":"json_schema","json_schema":{...}}`. |
| `reasoning_effort` | string | No | Reasoning models only: `none`, `low`, `medium`, `high`. |
| `reasoning_format` | string | No | `parsed`, `raw`, `hidden`, `none`. Controls how reasoning appears. |
| `logprobs` | boolean | No | Return log probabilities of output tokens. |
| `top_logprobs` | integer | No | 0-20. Most likely tokens per position (requires `logprobs: true`). |
| `prompt_cache_key` | string | No | Opaque key to improve cache routing for repeated prompts. Max 1024 chars. |
| `service_tier` | string | No | Service tier selector (e.g., `auto`, `default`). |
| `disable_reasoning` | boolean | No | **Deprecated.** Z.ai GLM only. Use `reasoning_effort="none"` instead. Removal: July 21, 2026. |

### Message Roles

Messages are an ordered array. Each item has a `role` and `content`:

- **`system`** -- Global instructions / persona.
- **`developer`** -- Developer instructions (treated similarly to system; takes precedence over user).
- **`user`** -- End-user input.
- **`assistant`** -- Prior model turns; may include `tool_calls` and `reasoning`.
- **`tool`** -- A tool/function result, referencing `tool_call_id`.

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"}
]
```

### Example Request (Python with OpenAI SDK)

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["CEREBRAS_API_KEY"],
    base_url="https://api.cerebras.ai/v1",
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Name three primary colors."},
    ],
    max_completion_tokens=100,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Example Request (First-party SDK)

```python
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Name three primary colors."}],
)

print(response.choices[0].message.content)
# Server-side timing available via:
print(response.time_info)
```

### Response Object

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1780595565,
  "model": "gpt-oss-120b",
  "system_fingerprint": "fp_xxxxxxxx",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "PONG",
        "reasoning": "The user asked me to reply with PONG."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 73,
    "completion_tokens": 20,
    "total_tokens": 93,
    "completion_tokens_details": {
      "reasoning_tokens": 17
    },
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "image_tokens": 0
    }
  },
  "time_info": {
    "queue_time": 0.001,
    "prompt_time": 0.002,
    "completion_time": 0.011,
    "total_time": 0.015
  }
}
```

### Response Fields

| Field | Description |
|---|---|
| `id` | Completion ID. |
| `object` | `chat.completion` (or `chat.completion.chunk` when streaming). |
| `created` | Unix timestamp (seconds). |
| `model` | Model that served the request. |
| `system_fingerprint` | Backend configuration fingerprint. |
| `choices[].index` | Position in the `choices` array. |
| `choices[].finish_reason` | `stop`, `length`, `tool_calls`, `content_filter`. |
| `choices[].message.content` | The assistant's text (may be `null` on a tool call). |
| `choices[].message.reasoning` | **Reasoning models only** -- the model's chain of thought. |
| `choices[].message.tool_calls` | Tool calls requested by the model. |
| `usage.prompt_tokens` | Input tokens. |
| `usage.completion_tokens` | Generated tokens (includes reasoning tokens). |
| `usage.total_tokens` | Sum. |
| `usage.completion_tokens_details.reasoning_tokens` | Tokens spent on internal reasoning. |
| `usage.prompt_tokens_details.cached_tokens` | Tokens served from prompt cache. |
| `usage.prompt_tokens_details.image_tokens` | Tokens consumed by image inputs. |
| `time_info` | Cerebras server-side timing in seconds (see below). |

### The `time_info` Performance Block

Every response includes a Cerebras-specific timing block:

```json
"time_info": {
  "queue_time": 0.001,
  "prompt_time": 0.002,
  "completion_time": 0.011,
  "total_time": 0.015
}
```

| Field | Meaning |
|---|---|
| `queue_time` | Time the request waited in queue before processing. |
| `prompt_time` | Time to process (prefill) the prompt tokens. |
| `completion_time` | Time spent generating completion tokens. |
| `total_time` | End-to-end server-side time. |

Compute output throughput: `completion_tokens / completion_time`.

```python
resp = client.chat.completions.create(...)
ti = resp.model_extra["time_info"]  # OpenAI SDK
# or: ti = resp.time_info  # Cerebras SDK
tokens_per_second = resp.usage.completion_tokens / ti["completion_time"]
print(f"{tokens_per_second:,.0f} tok/s")
```

---

## 2. Multimodal / Image Inputs

> **Status:** Private Preview. Currently only available with `gemma-4-31b`.
> Contact Cerebras for access.

Images are sent through the Chat Completions API as base64-encoded data URIs in the `messages` array. External image URLs are not supported during Public Preview.

### Message Format

Add an `image_url` object to the `content` array in a user message:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image."},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgo..."
      }
    }
  ]
}
```

### Input Requirements

| Requirement | Details |
|---|---|
| Supported formats | PNG (`.png`), JPEG (`.jpeg`, `.jpg`) |
| Encoding | Base64 data URI (e.g., `data:image/png;base64,...`) |
| External image URLs | Not supported during Public Preview |
| Max payload size | 10 MB total image payload per request (shared tier). Free trial: 4 MB. |
| Max images per request | 5 (shared tier). Free trial: 2. |
| `image_url` placement | Only on `user` role messages |

### Python Example (Single Image)

```python
from cerebras.cloud.sdk import Cerebras
import os
import base64

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("screenshot.png")

response = client.chat.completions.create(
    model="gemma-4-31b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one concise sentence."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

### Python Example (Multiple Images -- up to 5)

```python
base64_image_1 = encode_image("image1.jpeg")
base64_image_2 = encode_image("image2.png")

response = client.chat.completions.create(
    model="gemma-4-31b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_2}"}},
            ],
        }
    ],
)
```

### Token Cost Per Image

`gemma-4-31b` uses up to **280 image tokens per image**. Token usage depends on processed image dimensions, not file size.

The model preserves aspect ratio during preprocessing. The scale factor is:

```
scale = sqrt(645120 / (width x height))
```

Processed dimensions are rounded down to the nearest multiple of 48. The final token count:

```
image_tokens = min((processed_width / 48) x (processed_height / 48), 280)
```

**Example token costs:**

| Input resolution | Processed resolution | Image tokens |
|---|---|---|
| 336 x 226 | 960 x 624 | 260 |
| 512 x 512 | 768 x 768 | 256 |
| 1024 x 1024 | 768 x 768 | 256 |
| 1280 x 720 | 1056 x 576 | 264 |
| 1920 x 1080 | 1056 x 576 | 264 |
| 3840 x 2160 | 1056 x 576 | 264 |

Image tokens appear in `usage.prompt_tokens_details.image_tokens`.

### Limitations

- Medical images (CT scans, MRIs) are not suitable
- May have difficulty reading small/low-resolution text
- May misinterpret rotated content
- May struggle with precise spatial reasoning
- Object counting gives approximate results
- CAPTCHAs are not supported
- The model cannot access original filenames or metadata
- Images are stateless -- must be resent on each turn

---

## 3. Structured Outputs

### JSON Schema Mode

Force the model to return JSON conforming to a supplied schema:

```python
import json
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_CEREBRAS_API_KEY",
    base_url="https://api.cerebras.ai/v1",
)

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "birth_year": {"type": "integer"},
        "occupation": {"type": "string"},
    },
    "required": ["name", "birth_year", "occupation"],
    "additionalProperties": False,
}

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "Extract structured data."},
        {"role": "user", "content": "Ada Lovelace, born 1815, is a mathematician."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": schema,
        },
    },
)

data = json.loads(response.choices[0].message.content)
print(data["name"], data["birth_year"])
```

### Strict Mode (`strict: true`)

When `strict: true` is set, Cerebras uses **constrained decoding** to guarantee schema conformance at the token level. Invalid outputs become impossible.

**Requirements when strict is on:**
- `additionalProperties: false` is **required** for every object in the schema
- Root must have `type: "object"`
- Max schema depth: 10 levels
- Max schema length: 5,000 characters
- Max object properties: 500
- Max total enum values: 500

> **Starting July 21, 2026**: These requirements will be strictly enforced for all models. Non-conforming schemas will return a validation error.

### JSON Object Mode (Looser)

```python
response_format={"type": "json_object"}
```

The model will return valid JSON, but no specific schema is enforced. Instruct the model on desired shape in the prompt.

### Unsupported Features in Strict Mode

- Recursive schemas (self-referencing)
- External `$ref` (references to external URLs)
- `$anchor` keyword
- String `pattern` (regex)
- String `format` (email, date-time, uuid)
- `minItems` / `maxItems` constraints

### Limitations

- **`tools` and `response_format` cannot be used in the same request.**
- When using strict mode with reasoning models, `message.content` is constrained by the schema; `message.reasoning` remains free-form text.
- Always `json.loads()` the `content` -- it is returned as a JSON string, not a parsed object.

### Schema References & Definitions

Use `$ref` with `$defs` for reusable components:

```python
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "director": {"$ref": "#/$defs/person"},
    },
    "required": ["title", "director"],
    "additionalProperties": False,
    "$defs": {
        "person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
    },
}
```

---

## 4. Tool Calling / Function Calling

### Tool Definition Format

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "strict": True,  # Enables constrained decoding for arguments
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]
```

### `tool_choice` Options

| Value | Behavior |
|---|---|
| `"auto"` | Model decides whether to call a tool (default). |
| `"none"` | Model will not call any tool. |
| `"required"` | Model must call a tool. |
| `{"type":"function","function":{"name":"get_weather"}}` | Force a specific tool. |

### `parallel_tool_calls`

- Default: `true`. The model may emit multiple `tool_calls` in a single turn.
- Set to `false` to force sequential execution (one tool call per turn).

### Full Tool Calling Loop (Python)

```python
import json
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_CEREBRAS_API_KEY",
    base_url="https://api.cerebras.ai/v1",
)

# Mock weather function
def get_weather(city):
    return {"city": city, "temp_c": 18, "sky": "clear"}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

available_functions = {"get_weather": get_weather}
messages = [{"role": "user", "content": "What is the weather in Paris?"}]

# First call -- model decides to use tool
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message

if msg.tool_calls:
    # Append assistant message with tool_calls
    messages.append(msg.model_dump())

    # Execute each tool call
    for tc in msg.tool_calls:
        fn = available_functions[tc.function.name]
        args = json.loads(tc.function.arguments)
        result = fn(**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result),
        })

    # Second call -- model has tool results
    final = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=messages,
        tools=tools,
    )
    print(final.choices[0].message.content)
else:
    print(msg.content)
```

### Strict Mode for Tool Calling

When `strict: true` is set inside the `function` object:

- Tool call arguments are guaranteed to match the schema exactly via constrained decoding
- `additionalProperties: false` is required for every object
- Works with parallel tool calling

### Multi-Turn Tool Calling

The model can call tools multiple times within a single conversation. Keep calling `client.chat.completions.create()` until you get a message without `tool_calls`. The model itself decides when enough information has been gathered.

---

## 5. Reasoning / Thinking

### Supported Models

- `gpt-oss-120b` -- Reasoning enabled by default
- `zai-glm-4.7` -- Reasoning enabled by default
- `gemma-4-31b` -- Reasoning disabled by default

### `reasoning_effort` Parameter

| Value | Behavior | When to use |
|---|---|---|
| `none` | Minimal/no internal reasoning (fastest). | Simple lookups, formatting. |
| `low` | Light reasoning. | Easy reasoning, latency-sensitive. |
| `medium` | Balanced. | General problem solving (default for GPT-OSS/GLM). |
| `high` | Deep, thorough reasoning (most tokens, slower). | Hard math, multi-step logic, code. |

Model-specific behavior:
- **`gpt-oss-120b`**: Supports `low`, `medium`, `high`. No `none`.
- **`zai-glm-4.7`**: Supports `none` (disables reasoning). `disable_reasoning` is **deprecated** (removal: July 21, 2026).
- **`gemma-4-31b`**: Reasoning disabled by default. `none` = disabled. `low`/`medium`/`high` all currently equivalent (enable reasoning without graduated effort).

### `reasoning_format` Parameter

Controls how reasoning text appears in responses:

| Format | Description |
|---|---|
| `parsed` | Reasoning returned in separate `reasoning` field; logprobs separated into `reasoning_logprobs`. Default for GPT-OSS and GLM. |
| `raw` | Reasoning prepended to content. GLM uses `<think>...</think>` tokens. |
| `hidden` | Reasoning text and logprobs dropped (tokens still counted). |
| `none` | Uses model's default behavior. |

Gemma 4 does not support `raw` or `hidden` formats.

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_CEREBRAS_API_KEY",
    base_url="https://api.cerebras.ai/v1",
)

resp = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "A train travels 60 km in 45 minutes. Speed in km/h?"}],
    reasoning_effort="high",
    max_completion_tokens=1024,
)

msg = resp.choices[0].message
print("ANSWER:", msg.content)
print("THOUGHT:", getattr(msg, "reasoning", None) or msg.model_extra.get("reasoning"))

usage = resp.usage
reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
answer_tokens = usage.completion_tokens - reasoning_tokens
print(f"reasoning_tokens={reasoning_tokens} answer_tokens={answer_tokens}")
```

### Budgeting Reasoning Tokens

- Reasoning tokens are **counted inside `completion_tokens`** and consume `max_completion_tokens` budget.
- If `max_completion_tokens` is too small, the model may exhaust the budget on reasoning and return a truncated answer (`finish_reason: "length"`).
- Set the budget generously for `reasoning_effort: "high"`.
- To make a reasoning model behave like a fast, direct model, use `reasoning_effort: "none"` (GLM) or default (no effort set, which uses the model's default).
- Reasoning tokens count toward TPM/TPD rate limits.

### Reasoning Context Retention

Reasoning tokens are not automatically retained across requests. To maintain awareness of prior reasoning in multi-turn conversations, include the reasoning text in the `content` field of `assistant` messages:

**GPT-OSS**: Prepending reasoning directly before the answer.
**GLM**: Wrapping reasoning in `<think>...</think>` tags.

---

## 6. Streaming

### How to Enable

Set `stream: true` in the request. The API returns an iterable of `chat.completion.chunk` objects as Server-Sent Events (SSE).

### Python Example

```python
from cerebras.cloud.sdk import Cerebras
import os

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

stream = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Stream Event Format

Each chunk has `object: "chat.completion.chunk"`:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1780595565,
  "model": "gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "One"
      },
      "finish_reason": null
    }
  ]
}
```

The final event is `data: [DONE]`.

When reasoning is enabled, streaming delivers reasoning tokens in the `reasoning` field of the delta:

```json
{
  "choices": [
    {
      "delta": {
        "reasoning": " should"
      },
      "index": 0
    }
  ]
}
```

### Including Usage in Stream

To receive usage information in the final stream chunk, use `stream_options`:

```python
stream = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
    stream_options={"include_usage": True},
)
```

When `include_usage` is `true`, the final chunk before `[DONE]` will include a `usage` field.

### Time-to-First-Token

Cerebras's Wafer-Scale Engine provides very low time-to-first-token (TTFT). For the best TTFT, use streaming with short prompts. For large payloads, consider [payload optimization](#10-payload-optimization) to reduce network transfer time.

---

## 7. Models

### Endpoint

```
GET https://api.cerebras.ai/v1/models
```

Lists the models available to **your** account. **Availability varies by account and tier.** Always call this endpoint before hardcoding a model name.

### Response

```json
{
  "object": "list",
  "data": [
    {"id": "gpt-oss-120b", "object": "model", "created": 0, "owned_by": "Cerebras"},
    {"id": "zai-glm-4.7", "object": "model", "created": 0, "owned_by": "Cerebras"}
  ]
}
```

### Production Models

| Model Name | Model ID | Parameters | Speed | Type |
|---|---|---|---|---|
| OpenAI GPT OSS | `gpt-oss-120b` | 120B | ~3000 tok/s | Reasoning |

### Preview Models

| Model Name | Model ID | Parameters | Speed | Type |
|---|---|---|---|---|
| Gemma 4 31B (coming soon) | `gemma-4-31b` | 31B | ~1850 tok/s | Vision + Reasoning |
| Z.ai GLM 4.7 | `zai-glm-4.7` | 355B | ~1000 tok/s | Reasoning |

Preview models are intended for evaluation only and may be discontinued on short notice.

### Standard Self-Serve Models

Typical standard accounts see:
- `gpt-oss-120b`
- `zai-glm-4.7`

### Dedicated / Partner Models

Available through dedicated deployments (subject to change):
- Llama 3.1 / 3.3 family (e.g., `llama3.1-8b`, `llama-3.3-70b`)
- Llama 4 family (Scout, Maverick)
- Qwen-3 variants
- `qwen-3-coder-480b`

### Model Compression

All public models are **unpruned** (original versions). Cerebras uses selective weight-only quantization only during storage. Activations, attention, and KV cache remain in full precision (unquantized).

### Error on Invalid Model

```json
{
  "message": "Model llama3.1-8b does not exist or you do not have access to it.",
  "type": "not_found_error",
  "param": "model",
  "code": "model_not_found"
}
```

---

## 8. Error Handling

### Error Format

Cerebras error bodies are **flat** -- fields are at the top level, **not** wrapped in an `error` object:

```json
{
  "message": "Model llama3.1-8b does not exist or you do not have access to it.",
  "type": "not_found_error",
  "param": "model",
  "code": "model_not_found"
}
```

### Error Table

| HTTP | `type` | `code` (example) | Cause |
|---|---|---|---|
| 400 | `invalid_request_error` | (varies) | Malformed request, bad/out-of-range parameter. |
| 401 | `authentication_error` | `wrong_api_key` | Missing or invalid API key. |
| 402 | `payment_required` | -- | Payment required. |
| 403 | `permission_denied_error` | -- | Insufficient permissions. |
| 404 | `not_found_error` | `model_not_found` | Model name unknown or not accessible. |
| 413 | `content_too_large` | -- | Image payload exceeds limit. |
| 422 | `unprocessable_entity_error` | -- | Validation error. |
| 429 | `rate_limit_error` | -- | RPM/TPM/TPD limit hit. |
| 500 | `internal_server_error` | -- | Transient backend error. |
| 503 | `service_unavailable` | -- | Service temporarily unavailable. |

### Image Input Errors

| Status | Condition | Error Detail |
|---|---|---|
| 413 | Total image payload exceeds limit | `"Total request size exceeds maximum"` |
| 413 | Too many image inputs | `"Number of image inputs exceeds maximum"` |
| 413 | Decompressed RGB bytes exceed 350 MB | `"Image decompression exceeds maximum memory limit"` |
| 400 | Invalid data URI | `"Invalid image_url: expected base64 data URI"` |
| 400 | HTTPS URL used | `"HTTPS image URLs are not supported. Use a base64-encoded data URI instead."` |
| 400 | Corrupt base64 | `"Image data could not be decoded"` |
| 400 | `image_url` on non-`user` role | `"image_url content parts are only supported on user messages"` |
| 400 | Non-vision model | `"Model {model} does not support image inputs"` |

### Python Error Handling (First-party SDK)

```python
import cerebras.cloud.sdk
from cerebras.cloud.sdk import Cerebras

client = Cerebras()

try:
    response = client.chat.completions.create(...)
except cerebras.cloud.sdk.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)
except cerebras.cloud.sdk.RateLimitError as e:
    print("429 received; back off")
except cerebras.cloud.sdk.APIStatusError as e:
    print(f"Non-200 status: {e.status_code}")
    print(e.response)
```

### Default Retry Behavior

The SDK automatically retries **2 times** with exponential backoff for:
- Connection errors
- 408 Request Timeout
- 429 Rate Limit
- 500+ Internal errors

Configure retries:

```python
# Disable retries
client = Cerebras(max_retries=0)

# Per-request override
client.with_options(max_retries=5).chat.completions.create(...)
```

### Timeout Configuration

Default timeout: **1 minute**.

```python
# Custom timeout
client = Cerebras(timeout=20.0)  # 20 seconds

# Granular control
import httpx
client = Cerebras(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)

# Per-request override
client.with_options(timeout=5.0).chat.completions.create(...)
```

### Retry Guidance (Manual)

- **429 (rate limit)** and **5xx (server)**: Retry with exponential backoff + jitter. On 429, prefer waiting for the `x-ratelimit-reset-*` window.
- **400 / 401 / 404**: Do **not** retry -- fix the request, key, or model name first.
- Cap total retries at 3-5.

```python
import time, random
from openai import OpenAI, RateLimitError, APIStatusError

client = OpenAI(
    api_key="YOUR_CEREBRAS_API_KEY",
    base_url="https://api.cerebras.ai/v1",
)

def chat_with_retry(**kwargs):
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = min(2 ** attempt + random.random(), 30)
            time.sleep(wait)
        except APIStatusError as e:
            if 500 <= e.status_code < 600:
                time.sleep(min(2 ** attempt + random.random(), 30))
            else:
                raise
    raise RuntimeError("exhausted retries")
```

---

## 9. Prompt Caching

### How It Works

Prompt caching is **automatic** on all supported API requests. No code changes are required.

1. **Prefix Matching**: The system analyzes the beginning of your prompt (system prompts, tool definitions, few-shot examples).
2. **Block-Based Caching**: Prompts are processed in 128-token blocks. Matching blocks reuse cached computation.
3. **Automatic Expiration**: Guaranteed TTL of **5 minutes**, may persist up to **1 hour** depending on system load.
4. **Organization-Isolated**: Caches are never shared between organizations.

### Cache Hit Requirements

To get a cache hit, the **entire beginning** of your prompt must match **exactly** with a previously cached prefix. Even a single character difference in the first token causes a cache miss.

### Tracking Cache Usage

Check the `usage.prompt_tokens_details.cached_tokens` field in the response:

```json
"usage": {
  "prompt_tokens": 3000,
  "completion_tokens": 150,
  "total_tokens": 3150,
  "prompt_tokens_details": {
    "cached_tokens": 2800
  }
}
```

In this example, 2,800 of 3,000 prompt tokens were served from cache.

### `prompt_cache_key`

> Requires account-level enablement. Contact Cerebras for access.

An optional opaque string that tells the system which requests share a common prompt prefix, improving cache routing.

```python
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain recursion in Python."},
    ],
    prompt_cache_key="conversation-abc-123",
)
```

**When to use:** Multi-turn chat sessions (use conversation ID) or agentic/RAG workflows (use workflow ID). Reuse the same key across all requests in a conversation.

**When NOT to use:** For prefixes shared across many users (e.g., common system prompt). This creates a bottleneck.

**Limits:** Max 1024 characters. Longer values rejected with 400 error.

### Structuring Prompts for Caching

**Static content first** (gets cached):
- System instructions
- Tool definitions and schemas
- Few-shot examples
- Large context documents

**Dynamic content last** (processed fresh):
- User-specific questions
- Session variables
- Timestamps

### Supported Models

Prompt caching is enabled by default for:
- `zai-glm-4.7`
- `gpt-oss-120b`

### FAQs

- **Cached tokens DO count toward TPM rate limits.** `cached_tokens + fresh_tokens = total TPM usage`.
- **No additional fee** for caching. Cached tokens billed at standard input rate.
- **No manual cache clearing** -- the system automatically manages eviction based on TTL (5 min to 1 hour).
- **Cache misses** can happen due to: block size (<128 tokens), data center routing changes, TTL expiration.
- Prompt caching is secure and **ZDR-compliant** -- ephemeral in-memory only, never persisted.

---

## 10. Payload Optimization

> Reduce latency by compressing request payloads with msgpack encoding and gzip.

The Cerebras API supports `application/vnd.msgpack` encoding and `Content-Encoding: gzip` for reducing request body size.

### Encoding Options

| Content-Type | Description | Chat Completions savings | Completions savings |
|---|---|---|---|
| `application/json` | Default | Baseline | Baseline |
| `application/vnd.msgpack` | msgpack binary | up to ~5% | up to ~56% |
| `application/json` + `Content-Encoding: gzip` | JSON + gzip | up to ~98% | up to ~68% |
| `application/vnd.msgpack` + `Content-Encoding: gzip` | Both | up to ~98% | up to ~69% |

### Python Example (gzip)

```python
import gzip, json, requests

payload = {
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain quantum computing."}],
}

json_bytes = json.dumps(payload).encode("utf-8")
compressed = gzip.compress(json_bytes, compresslevel=5)

response = requests.post(
    "https://api.cerebras.ai/v1/chat/completions",
    data=compressed,
    headers={
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",
        "Authorization": f"Bearer {os.environ['CEREBRAS_API_KEY']}",
    },
)
```

### Python Example (msgpack + gzip)

```python
import gzip, msgpack, requests

payload = {
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain quantum computing."}],
}

data = msgpack.packb(payload)
compressed = gzip.compress(data, compresslevel=5)

response = requests.post(
    "https://api.cerebras.ai/v1/chat/completions",
    data=compressed,
    headers={
        "Content-Type": "application/vnd.msgpack",
        "Content-Encoding": "gzip",
        "Authorization": f"Bearer {os.environ['CEREBRAS_API_KEY']}",
    },
)
```

---

## 11. Rate Limits

### How Limits Are Measured

Limits are enforced per model along three axes:
- **RPM**: Requests per minute
- **TPM**: Tokens per minute
- **TPD**: Tokens per day

Whichever limit is hit first triggers a `429`. Rate limits apply at the **organization level**.

### Token Rate Limiting

The system estimates total tokens as: `input_tokens + max_completion_tokens`. If this exceeds available quota, the request is rate-limited before processing begins.

**Best practice:** Set `max_completion_tokens` appropriately to avoid overestimating and triggering unnecessary rate limits.

**Quota replenishment:** Uses the token bucket algorithm -- capacity replenishes continuously, not at fixed intervals.

### Free Trial Limits

| Model | RPM | TPM | TPH | TPD |
|---|---|---|---|---|
| `gpt-oss-120b` | 5 | 30K | 1M | 1M |
| `zai-glm-4.7` | 5 | 30K | 1M | 1M |
| `gemma-4-31b` | 5 | 30K | 1M | 1M |

Gemma 4 image limits (free): 2 per request, 4 MB payload.

### Developer (Pay as You Go) Limits

| Model | TPM | RPM |
|---|---|---|
| `gpt-oss-120b` | 1M | 1K |
| `zai-glm-4.7` | 500K | 500 |
| `gemma-4-31b` | 500K | 300 |

Gemma 4 image limits (developer): 5 per request, 10 MB payload.
Hourly/daily restrictions do not apply to developer tier.

### Rate Limit Headers

Every API response includes these headers:

| Header | Description |
|---|---|
| `x-ratelimit-limit-requests-day` | Max requests per day |
| `x-ratelimit-limit-tokens-minute` | Max tokens per minute |
| `x-ratelimit-remaining-requests-day` | Requests remaining today |
| `x-ratelimit-remaining-tokens-minute` | Tokens remaining this minute |
| `x-ratelimit-reset-requests-day` | Seconds until daily limit resets |
| `x-ratelimit-reset-tokens-minute` | Seconds until token limit resets |

### Rate Limit Response

```json
{
  "message": "Rate limit exceeded",
  "type": "rate_limit_error",
  "param": null,
  "code": "rate_limit_error"
}
```

HTTP Status: **429 Too Many Requests**

---

## 12. Best Practices

### Connection Reuse

- Always reuse the same SDK client instance to benefit from connection pooling.
- The Cerebras SDK and OpenAI SDK both use `httpx` under the hood, which maintains a connection pool by default.

```python
# GOOD: Reuse the client
client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
for _ in range(100):
    client.chat.completions.create(...)

# BAD: Create a new client for every request
```

### Timeout Settings

- Default timeout is 60 seconds (1 minute) for the Cerebras SDK.
- For quick chat completions, consider a lower timeout (e.g., 10-20s).
- For reasoning-heavy requests with `reasoning_effort: "high"`, allow more time (30-60s+).
- Configure per-request timeouts for different types of calls.

```python
# Fast lookup: short timeout
client.with_options(timeout=10.0).chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    reasoning_effort="none",
)

# Deep reasoning: generous timeout
client.with_options(timeout=60.0).chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Solve a complex math problem..."}],
    reasoning_effort="high",
    max_completion_tokens=4096,
)
```

### Retry Strategy

- **429 (rate limit)**: Exponential backoff with jitter. Honor `Retry-After` if present.
- **5xx (server error)**: Retry with backoff, up to 3-5 attempts.
- **400/401/404**: Do not retry -- fix the input first.
- The SDK retries 429 and 5xx automatically (2 retries by default).

```python
def chat_with_backoff(**kwargs):
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIStatusError) as e:
            if e.status_code in (429,) or 500 <= e.status_code < 600:
                wait = min(2 ** attempt + random.random(), 30)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("exhausted retries")
```

### Model Discovery

- **Always call `GET /v1/models`** before hardcoding a model name. Availability varies by account and tier.
- Never assume a model exists -- it may return `404 model_not_found` on another account.

### Bound Output Tokens

- Always set `max_completion_tokens` to cap latency and cost.
- On reasoning models, this includes reasoning tokens -- set it generously for `reasoning_effort: "high"`.
- Prefer `max_completion_tokens` over the legacy `max_tokens`.

### Speed Measurement

- Use `time_info.completion_time` to compute real output throughput: `completion_tokens / completion_time`.
- Typical Cerebras throughput: ~2,000-3,000 tokens/second.
- Wall-clock latency includes network round-trip; `time_info.total_time` is server-side only.

### Deterministic Output

- For reproducible/cacheable results, set `seed` + `temperature: 0`.
- Note: exact reproduction is not guaranteed (best-effort).

### Reasoning Best Practices

- Separate `message.reasoning` from `message.content`. Never present reasoning as the final answer.
- Use `reasoning_effort: "low"` (or `"none"`) when full reasoning isn't required -- directly cuts latency and tokens.
- Reserve `"high"` for genuinely hard problems.
- Reasoning tokens count toward `max_completion_tokens` and TPM/TPD limits.

### Key Security

- Never hardcode API keys. Use environment variables (`CEREBRAS_API_KEY`).
- Store keys in `.env` files excluded from version control.
- Use `"YOUR_CEREBRAS_API_KEY"` as a placeholder in documentation and examples.

### OpenAI SDK Compatibility

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["CEREBRAS_API_KEY"],
    base_url="https://api.cerebras.ai/v1",
)
```

Non-standard parameters (e.g., `clear_thinking` for GLM) must be passed through `extra_body` when using the OpenAI SDK:

```python
response = client.chat.completions.create(
    model="zai-glm-4.7",
    messages=[...],
    reasoning_effort="none",
    extra_body={"clear_thinking": False},
)
```

With the Cerebras SDK, non-standard parameters can be passed directly as named arguments.

### Payload Optimization

- For long prompts, use gzip compression to reduce network transfer time and improve TTFT.
- For `/v1/completions` with token arrays, msgpack encoding gives the largest benefit.
- For small requests (< a few KB), compression overhead may outweigh savings.

---

## Sources

- Official Cerebras Inference Docs: https://inference-docs.cerebras.ai
- Cerebras Cloud Console: https://cloud.cerebras.ai
- GitHub (community docs): https://github.com/simonpierreboucher02/agentilab-cerebras
