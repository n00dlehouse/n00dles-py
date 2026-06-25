from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import TypeAdapter

from .context import AgentOutputError
from .retry import RetryPolicy

if TYPE_CHECKING:
    from ..providers.base import BaseProvider
    from .pipeline import ParallelGroup, Pipeline, PipelineNode


@dataclass
class AgentTraceAttempt:
    attempt: int
    error: str


class AgentNode:
    """The runtime object behind every `@agent`-decorated function. Holds the parsed
    docstring/type-hint contract plus model/retry/timeout config, and knows how to
    build a prompt, call its provider, and validate the response — but never sees
    the surrounding pipeline or executor."""

    #: Library-wide fallback defaults, used only when neither the agent nor its
    #: enclosing pipeline() set a value explicitly.
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_RETRY = RetryPolicy()

    def __init__(
        self,
        fn: Callable,
        model: str,
        *,
        prompt: str | None = None,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        fallback: AgentNode | None = None,
        provider: BaseProvider | None = None,
    ):
        self.fn = fn
        self.name = fn.__name__
        self.docstring = inspect.cleandoc(fn.__doc__ or "")
        self.model = model
        self.prompt_override = prompt
        # None means "not explicitly set" — pipeline() fills these in for nodes that
        # didn't configure their own, and the executor falls back to DEFAULT_* if
        # the node is run standalone (never wrapped in a pipeline).
        self.timeout = timeout
        self.retry = retry
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tags = tags or []
        self.fallback = fallback
        self.provider = provider
        self._cached_provider: BaseProvider | None = provider

        hints = get_type_hints(fn)
        self.output_type: Any = hints.get("return", str)
        self.input_types: dict[str, Any] = {k: v for k, v in hints.items() if k != "return"}
        self.param_names: list[str] = list(self.input_types.keys()) or list(
            inspect.signature(fn).parameters.keys()
        )

    @property
    def effective_timeout(self) -> float:
        return self.timeout if self.timeout is not None else self.DEFAULT_TIMEOUT

    @property
    def effective_retry(self) -> RetryPolicy:
        return self.retry if self.retry is not None else self.DEFAULT_RETRY

    def __rshift__(self, other: PipelineNode | Pipeline) -> Pipeline:
        from .pipeline import Pipeline

        if isinstance(other, Pipeline):
            return Pipeline([self, *other.nodes])
        return Pipeline([self, other])

    def __or__(self, other: AgentNode | ParallelGroup) -> ParallelGroup:
        from .pipeline import ParallelGroup

        other_nodes = other.nodes if isinstance(other, ParallelGroup) else [other]
        return ParallelGroup([self, *other_nodes])

    def __repr__(self) -> str:
        return f"AgentNode({self.name!r}, model={self.model!r})"

    def _get_provider(self) -> BaseProvider:
        if self._cached_provider is None:
            from ..providers.litellm_provider import LiteLLMProvider

            self._cached_provider = LiteLLMProvider(self.model)
        return self._cached_provider

    def _build_system_prompt(self) -> str:
        base = self.prompt_override or self.docstring or f"You are {self.name}, an AI agent."
        if self.output_type is not str:
            schema = TypeAdapter(self.output_type).json_schema()
            base += (
                "\n\nRespond with ONLY valid JSON matching this schema. "
                "No prose, no markdown code fences, just the JSON object.\n"
                + json.dumps(schema)
            )
        return base

    def _build_user_prompt(self, inputs: dict[str, Any]) -> str:
        if len(inputs) == 1:
            return str(next(iter(inputs.values())))
        return json.dumps(inputs, default=str)

    def _parse_output(self, raw: str) -> Any:
        if self.output_type is str:
            return raw
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            parsed = json.loads(text)
            return TypeAdapter(self.output_type).validate_python(parsed)
        except Exception as exc:
            raise AgentOutputError(self.name, raw, exc) from exc

    async def _call(self, inputs: dict[str, Any]):
        """Invoke the provider once (no retry — that's the executor's job) and return
        (parsed_output, LLMResponse)."""
        system = self._build_system_prompt()
        user = self._build_user_prompt(inputs)
        provider = self._get_provider()
        kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = await provider.complete(user, system, **kwargs)
        return self._parse_output(resp.content), resp

    def __call__(self, **inputs: Any) -> Any:
        """Convenience: run() this single agent synchronously, e.g. `researcher(topic="x")`."""
        from .. import run

        return run(self, **inputs).output


def agent(
    model: str,
    *,
    prompt: str | None = None,
    timeout: float | None = None,
    retry: int | RetryPolicy | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    fallback: AgentNode | None = None,
    provider: BaseProvider | None = None,
) -> Callable[[Callable], AgentNode]:
    """Decorate a function into an LLM-backed agent. The docstring becomes the system
    prompt (unless `prompt` is given explicitly); type hints become the I/O contract.

    `timeout`/`retry` left unset here default to whatever the enclosing `pipeline()`
    specifies; if the agent is never wrapped in a pipeline, AgentNode.DEFAULT_TIMEOUT
    / DEFAULT_RETRY apply. Setting either explicitly here always wins over the
    pipeline's setting — the most specific configuration applies."""

    def decorator(fn: Callable) -> AgentNode:
        if retry is None or isinstance(retry, RetryPolicy):
            retry_policy = retry
        else:
            retry_policy = RetryPolicy(max_attempts=retry)
        return AgentNode(
            fn,
            model,
            prompt=prompt,
            timeout=timeout,
            retry=retry_policy,
            temperature=temperature,
            max_tokens=max_tokens,
            tags=tags,
            fallback=fallback,
            provider=provider,
        )

    return decorator
