"""Multi-provider inference abstraction for Gemma 4 31B.

Supports swapping Cerebras <-> OpenRouter <-> other providers.

Usage:
    from src.provider import ProviderRegistry, LLMProvider

    # Get default provider from env var LLM_PROVIDER
    provider = ProviderRegistry.default()

    # Or get by name
    provider = ProviderRegistry.get("openrouter")

    # Auto-fallback: Cerebras -> OpenRouter on failure
    result = ProviderRegistry.chat_with_fallback(messages)
"""

from __future__ import annotations

import abc
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


__all__: list[str] = [
    "LLMProvider",
    "InferenceResult",
    "CerebrasProvider",
    "OpenRouterProvider",
    "ProviderRegistry",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Unified result type returned by every provider's chat/image_chat."""
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    time_info: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    tool_calls: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class LLMProvider(abc.ABC):
    """Abstract interface for LLM inference providers."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceResult:
        """Text-only chat completion."""

    @abc.abstractmethod
    def image_chat(
        self,
        prompt: str,
        image_b64: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Multimodal chat with a base64-encoded image."""

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'cerebras', 'openrouter')."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether the provider is configured and ready to serve."""


# ---------------------------------------------------------------------------
# Cerebras provider
# ---------------------------------------------------------------------------

class CerebrasProvider(LLMProvider):
    """Provider wrapping the existing CerebrasClient for Gemma 4 31B."""

    def __init__(self, client: Any | None = None) -> None:
        """Wrap an existing CerebrasClient, or create one lazily."""
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            from src.client import CerebrasClient
            self._client = CerebrasClient()
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceResult:
        return self._ensure_client().chat(messages, tools=tools, stream=stream, **kwargs)

    def image_chat(
        self,
        prompt: str,
        image_b64: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        return self._ensure_client().image_chat(prompt, image_b64, system_prompt=system_prompt, **kwargs)

    def name(self) -> str:
        return "cerebras"

    def is_available(self) -> bool:
        if self._client is not None:
            return True
        key = os.environ.get("CEREBRAS_API_KEY", "")
        return bool(key)


# ---------------------------------------------------------------------------
# OpenRouter provider (OpenAI-compatible)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using httpx (OpenAI-compatible API).

    Model: ``google/gemma-4-31b`` (configurable via env var OPENROUTER_MODEL).
    Supports image content blocks and JSON schema via ``response_format``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 120,
    ) -> None:
        self._api_key = api_key or OPENROUTER_API_KEY
        self._model = model or OPENROUTER_MODEL
        self._base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self._timeout_s = timeout_s
        self._http_client: httpx.Client | None = None

    # ------------------------------------------------------------------
    # HTTP client (lazy init)
    # ------------------------------------------------------------------
    @property
    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceResult:
        """OpenAI-compatible chat completion via OpenRouter."""
        if stream:
            return self._stream(messages, tools=tools, **kwargs)

        body = self._build_body(messages, tools=tools, **kwargs)
        start = time.perf_counter()
        response = self._client.post("/chat/completions", json=body)
        elapsed = time.perf_counter() - start

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            detail = response.text[:500]
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {detail}"
            ) from None

        data = response.json()
        return self._parse_response(data, elapsed)

    def image_chat(
        self,
        prompt: str,
        image_b64: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """OpenAI-format image content: content array with image_url."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_b64, "detail": "high"},
                },
            ],
        })
        return self.chat(messages, **kwargs)

    def name(self) -> str:
        return "openrouter"

    def is_available(self) -> bool:
        key = self._api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return bool(key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Construct the request body for /chat/completions."""
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        max_tokens = kwargs.pop(
            "max_tokens",
            kwargs.pop("max_completion_tokens", 1024),
        )
        temperature = kwargs.pop("temperature", 0.2)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # JSON schema response format (OpenAI-compatible)
        response_format = kwargs.pop("response_format", None)
        if response_format:
            body["response_format"] = response_format

        # Tool calling
        if tools:
            body["tools"] = tools
            body["tool_choice"] = kwargs.pop("tool_choice", "auto")
            body["parallel_tool_calls"] = kwargs.pop("parallel_tool_calls", True)

        # Extra kwargs passthrough
        body.update(kwargs)

        return body

    def _stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """SSE streaming support for OpenRouter."""
        body = self._build_body(messages, tools=tools, **kwargs)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        content_parts: list[str] = []
        final_usage: dict[str, Any] = {}

        with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    content_parts.append(choices[0]["delta"]["content"])

                if chunk.get("usage"):
                    final_usage = chunk["usage"]

        return InferenceResult(
            content="".join(content_parts),
            model=self._model,
            usage=final_usage,
            latency_s=0.0,
        )

    def _parse_response(self, data: dict[str, Any], elapsed: float) -> InferenceResult:
        """Parse a non-streaming /chat/completions response."""
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content", "") or ""
        model = data.get("model", self._model)
        usage = data.get("usage", {})

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in message["tool_calls"]
            ]

        return InferenceResult(
            content=content,
            model=model,
            usage=usage,
            latency_s=elapsed,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# Provider Registry (singleton)
# ---------------------------------------------------------------------------

class ProviderRegistry:
    """Singleton registry of available LLM providers.

    Usage::

        # Get default (reads LLM_PROVIDER env var, falls back to 'cerebras')
        provider = ProviderRegistry.default()

        # Auto-fallback: call with retry logic across providers
        result = ProviderRegistry.chat_with_fallback(messages)

        # List available providers
        for name in ProviderRegistry.available():
            print(f"  {name}")
    """

    _instance: ProviderRegistry | None = None
    _providers: dict[str, LLMProvider] = {}

    def __new__(cls) -> ProviderRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._providers = cls._build_defaults()
        return cls._instance

    # ------------------------------------------------------------------
    # Class-level convenience interface
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> LLMProvider:
        """Return the default provider based on ``LLM_PROVIDER`` env var.

        Falls back to ``cerebras`` if the env var is unset or unknown.
        """
        preferred = os.environ.get("LLM_PROVIDER", "cerebras").strip().lower()
        self = cls()
        if preferred in self._providers:
            return self._providers[preferred]
        available = self.available()
        if available:
            return self._providers[available[0]]
        return self._providers.get("cerebras", CerebrasProvider())

    @classmethod
    def get(cls, name: str) -> LLMProvider:
        """Get a provider by name. Raises KeyError if not registered."""
        self = cls()
        if name not in self._providers:
            raise KeyError(
                f"Unknown provider '{name}'. "
                f"Available: {list(self._providers)}"
            )
        return self._providers[name]

    @classmethod
    def available(cls) -> list[str]:
        """Return sorted list of names of configured/available providers."""
        self = cls()
        return sorted(
            name for name, prov in self._providers.items() if prov.is_available()
        )

    @classmethod
    def register(cls, name: str, provider: LLMProvider) -> None:
        """Register a custom provider. Overwrites if name exists."""
        self = cls()
        self._providers[name] = provider

    @classmethod
    def chat_with_fallback(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        preferred: str | None = None,
        fallbacks: list[str] | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Call the preferred provider with auto-fallback on failure.

        Args:
            messages: Chat messages.
            tools: Optional tool definitions.
            stream: Whether to stream.
            preferred: Provider name to try first (default: LLM_PROVIDER env).
            fallbacks: Ordered fallback providers (default: [openrouter]).
            **kwargs: Passed through to chat().

        Returns:
            InferenceResult from the first successful provider.

        Raises:
            RuntimeError: If all providers fail.
        """
        self = cls()
        preferred = preferred or os.environ.get("LLM_PROVIDER", "cerebras")
        fallbacks = fallbacks or ["openrouter"]

        order = [preferred] + [f for f in fallbacks if f != preferred]
        errors: list[str] = []

        for name in order:
            if name not in self._providers:
                continue
            provider = self._providers[name]
            if not provider.is_available():
                errors.append(f"{name}: not configured (missing API key)")
                continue
            try:
                return provider.chat(messages, tools=tools, stream=stream, **kwargs)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                continue

        raise RuntimeError(
            "All providers failed:\n  " + "\n  ".join(errors)
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @classmethod
    def _build_defaults(cls) -> dict[str, LLMProvider]:
        """Build the default set of providers (lazy CerebrasClient creation)."""
        return {
            "cerebras": CerebrasProvider(),
            "openrouter": OpenRouterProvider(),
        }

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (useful for tests)."""
        cls._instance = None
        cls._providers = {}
