from __future__ import annotations

import time

from .base import BaseProvider, LLMResponse


class LiteLLMProvider(BaseProvider):
    """Wraps litellm.acompletion — the single provider abstraction for every LLM backend
    (Anthropic, OpenAI, Mistral, Gemini, Ollama, and anything else litellm supports).
    n00dles never implements a provider-specific SDK; this adapter is the entire surface."""

    def __init__(self, model: str, **defaults):
        self.model = model
        self.defaults = defaults

    async def complete(self, prompt: str, system: str, **kwargs) -> LLMResponse:
        import litellm

        t0 = time.monotonic()
        resp = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **{**self.defaults, **kwargs},
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        provider = "unknown"
        hidden = getattr(resp, "_hidden_params", None)
        if hidden:
            provider = hidden.get("custom_llm_provider", "unknown")
        return LLMResponse(
            content=choice,
            model=resp.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.monotonic() - t0) * 1000,
            provider=provider,
        )
