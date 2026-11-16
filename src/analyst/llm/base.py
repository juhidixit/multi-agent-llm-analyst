"""Provider-agnostic LLM interface.

Agents build a full natural-language prompt (``system`` + ``prompt``) *and* a structured
``payload`` with the same information. Real providers only read the prompt. The offline
``MockLLM`` reads the payload, which lets the entire pipeline (prompt construction, JSON
parsing, validation, retries, tracing, cost accounting) run in CI without an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMRequest:
    role: str
    system: str
    prompt: str
    payload: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 4000


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    simulated_latency_s: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def make_client(provider: str | None = None, **kwargs: Any) -> LLMClient:
    """Return a client. ``provider`` defaults to $ANALYST_PROVIDER, then ``mock``."""
    provider = (provider or os.getenv("ANALYST_PROVIDER") or "mock").lower()
    if provider == "mock":
        from analyst.llm.mock import MockLLM

        return MockLLM(**kwargs)
    if provider == "anthropic":
        from analyst.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(**kwargs)
    raise ValueError(f"unknown provider: {provider!r} (expected 'mock' or 'anthropic')")
