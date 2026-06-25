from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str


class BaseProvider(ABC):
    """A provider takes a system/user prompt and returns a completion. Implementations
    never see n00dles-specific concepts (agents, pipelines, retries) — those live one
    layer up in AgentNode."""

    @abstractmethod
    async def complete(self, prompt: str, system: str, **kwargs) -> LLMResponse: ...
