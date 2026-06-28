"""Cerebras Gemma 4 client wrapper with timing instrumentation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cerebras.cloud.sdk import Cerebras

from src.config import (
    CEREBRAS_API_KEY,
    CEREBRAS_BASE_URL,
    GEMMA_MODEL,
    MAX_TOKENS,
    REASONING_EFFORT,
    TEMPERATURE,
)


__all__: list[str] = []


@dataclass
class InferenceResult:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    time_info: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    tool_calls: list[dict[str, Any]] | None = None


class CerebrasClient:
    """Thin wrapper around the Cerebras Python SDK for Gemma 4 31B."""

    def __init__(self) -> None:
        self._client = Cerebras(
            api_key=CEREBRAS_API_KEY,
            base_url=CEREBRAS_BASE_URL,
            warm_tcp_connection=True,
        )
        self._model = GEMMA_MODEL

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceResult:
        """Send a chat completion request and return the result with timing.

        Accepts ``system_prompt`` as a keyword argument (converted to a
        ``{"role": "system"}`` message internally), along with any other
        parameters supported by the Cerebras Chat Completions API.
        """
        # Inject system_prompt as a system-role message if provided
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": kwargs.pop("max_tokens", MAX_TOKENS),
            "temperature": kwargs.pop("temperature", TEMPERATURE),
        }
        if REASONING_EFFORT:
            body["reasoning_effort"] = REASONING_EFFORT
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True
        body.update(kwargs)

        start = time.perf_counter()

        if stream:
            return self._stream(body)

        resp = self._client.chat.completions.create(**body)
        elapsed = time.perf_counter() - start

        choice = resp.choices[0]
        usage = resp.usage.model_dump() if resp.usage else {}
        time_info = resp.time_info.model_dump() if getattr(resp, "time_info", None) else {}

        # Capture tool calls from the response if present
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return InferenceResult(
            content=choice.message.content or "",
            model=resp.model,
            usage=usage,
            time_info=time_info,
            latency_s=elapsed,
            tool_calls=tool_calls,
        )

    def _stream(self, body: dict[str, Any]) -> InferenceResult:
        """Handle streaming response — collects chunks and final metadata."""
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        stream = self._client.chat.completions.create(**body)
        content_parts: list[str] = []
        final_usage: dict[str, Any] = {}
        final_time: dict[str, Any] = {}

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                final_usage = chunk.usage.model_dump()
            if getattr(chunk, "time_info", None):
                final_time = chunk.time_info.model_dump()

        return InferenceResult(
            content="".join(content_parts),
            model=self._model,
            usage=final_usage,
            time_info=final_time,
            latency_s=0.0,  # streaming time is tracked by time_info
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def image_chat(
        self,
        prompt: str,
        image_b64: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Send a text + image prompt to the multimodal model.

        *image_b64* should be a base64 data URI (e.g. ``data:image/jpeg;base64,...``).
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        })
        return self.chat(messages, **kwargs)