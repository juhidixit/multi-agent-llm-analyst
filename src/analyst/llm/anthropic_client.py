"""Claude via the official Anthropic SDK. Credentials come from the environment only."""

from __future__ import annotations

import os

from analyst.llm.base import LLMRequest, LLMResponse


class RefusalError(RuntimeError):
    pass


class AnthropicLLM:
    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("pip install 'multi-agent-llm-analyst[live]'") from exc
        # Anthropic() resolves ANTHROPIC_API_KEY (or another configured credential) itself.
        # Keys are never read from files in this repo or passed on the command line.
        self._client = anthropic.Anthropic(max_retries=3)
        self.model: str = model or os.getenv("ANALYST_MODEL") or "claude-opus-5-5"
        self.effort = effort or os.getenv("ANALYST_EFFORT", "low")

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.prompt}],
            output_config={"effort": self.effort},
            # Server-side fallback: if a request is declined, the API retries it on a
            # fallback model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RefusalError(f"model declined the {request.role} request")
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )
