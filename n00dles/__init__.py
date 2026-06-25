"""n00dles — open-source multi-agent AI orchestration.

    from n00dles import agent, pipeline, run

    @agent(model="claude-sonnet-4-6")
    def researcher(topic: str) -> str:
        '''Research the topic. Return 3 key facts.'''

    @agent(model="claude-sonnet-4-6")
    def writer(research: str) -> str:
        '''Write a short article from the research.'''

    result = run(pipeline(researcher >> writer), topic="multi-agent orchestration")
    print(result.output)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from .core.agent import AgentNode, agent
from .core.context import (
    AgentOutputError,
    AgentTrace,
    BranchError,
    PipelineContext,
    PipelineFailure,
    RunResult,
)
from .core.executor import PipelineExecutor
from .core.pipeline import BranchAgent, ParallelGroup, Pipeline, branch, parallel, pipeline
from .core.retry import RetryPolicy
from .core.state import InMemoryStateStore, SQLiteStateStore, StateStore
from .telemetry.tracer import Tracer

__version__ = "0.2.0"

__all__ = [
    "agent",
    "pipeline",
    "parallel",
    "branch",
    "run",
    "arun",
    "configure",
    "AgentNode",
    "Pipeline",
    "ParallelGroup",
    "BranchAgent",
    "RunResult",
    "AgentTrace",
    "PipelineContext",
    "RetryPolicy",
    "StateStore",
    "SQLiteStateStore",
    "InMemoryStateStore",
    "PipelineFailure",
    "AgentOutputError",
    "BranchError",
    "Tracer",
]

_default_state_store: StateStore | None = None
_default_tracer: Tracer | None = None


def _get_default_state_store() -> StateStore:
    global _default_state_store
    if _default_state_store is None:
        _default_state_store = SQLiteStateStore()
    return _default_state_store


def _get_default_tracer() -> Tracer:
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer


def _resolve_state_store(value: StateStore | str) -> StateStore:
    if isinstance(value, StateStore):
        return value
    if isinstance(value, str):
        if value.startswith("sqlite://"):
            prefix = "sqlite:///" if value.startswith("sqlite:///") else "sqlite://"
            path = value[len(prefix) :]
            return SQLiteStateStore(path=path or "./n00dles_state.db")
        if value.startswith("redis://"):
            raise NotImplementedError(
                "The Redis state backend isn't implemented in this release yet. "
                "Use a sqlite:// URL (the default), InMemoryStateStore(), or pass your "
                "own StateStore instance."
            )
    raise TypeError(f"unsupported state_store: {value!r}")


def _resolve_exporter(value: Any, **kwargs: Any) -> Any:
    if isinstance(value, str):
        if value == "otel":
            from .telemetry.exporters.otel import OTelExporter

            return OTelExporter(**kwargs)
        if value in ("langfuse", "helicone"):
            raise NotImplementedError(
                f"The {value} exporter isn't implemented in this release yet. "
                "OpenTelemetry is available now via trace_exporter='otel' "
                "(requires: pip install n00dles[otel])."
            )
        raise ValueError(f"unknown trace_exporter: {value!r}")
    return value  # assume a duck-typed object with an .export(event) method


def configure(
    *,
    state_store: StateStore | str | None = None,
    trace_exporter: Any | None = None,
    **kwargs: Any,
) -> None:
    """Set process-wide defaults used by run()/arun(). Safe to call more than once —
    later calls only override the arguments they're given.

        configure(state_store="sqlite:///my_app.db")
        configure(trace_exporter="otel", service_name="my-service")
    """
    global _default_state_store
    if state_store is not None:
        _default_state_store = _resolve_state_store(state_store)
    if trace_exporter is not None:
        _get_default_tracer().add_exporter(_resolve_exporter(trace_exporter, **kwargs))


async def arun(
    target: AgentNode | ParallelGroup | BranchAgent | Pipeline,
    *,
    timeout: float | None = None,
    tags: list[str] | None = None,
    **inputs: Any,
) -> RunResult:
    """Async entrypoint. Use this directly if you're already inside an event loop
    (FastAPI handlers, Jupyter, etc.) — `run()` cannot be called there."""
    if isinstance(target, (AgentNode, ParallelGroup, BranchAgent)):
        target = pipeline(target, timeout=timeout if timeout is not None else 60.0)
    elif not isinstance(target, Pipeline):
        raise TypeError(
            "run()/arun() expects an @agent-decorated function, parallel()/branch(), "
            f"or a pipeline(...), got {type(target)!r}"
        )

    run_id = str(uuid.uuid4())
    executor = PipelineExecutor(state=_get_default_state_store(), tracer=_get_default_tracer())
    result = await executor.execute(target, run_id, inputs)
    result.tags = tags or []
    return result


def run(
    target: AgentNode | ParallelGroup | BranchAgent | Pipeline,
    *,
    timeout: float | None = None,
    tags: list[str] | None = None,
    **inputs: Any,
) -> RunResult:
    """Sync entrypoint — what you'll use in a plain script. Wraps `arun()` in
    `asyncio.run()`. Raises a clear error if called from inside a running event loop;
    use `await arun(...)` there instead."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — the expected case, proceed below
    else:
        raise RuntimeError(
            "run() can't be called from inside a running event loop "
            "(e.g. FastAPI, Jupyter). Use `await arun(...)` instead."
        )
    return asyncio.run(arun(target, timeout=timeout, tags=tags, **inputs))
