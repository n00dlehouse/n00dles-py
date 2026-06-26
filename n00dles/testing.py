"""Test doubles for n00dles agents and pipelines — no network calls, no API keys.

    from n00dles.testing import mock_agent, mock_pipeline

    def test_writer_uses_research():
        with mock_agent(researcher, returns="fact 1, fact 2, fact 3"):
            result = run(content_pipeline, topic="test topic")
        assert "fact 1" in result.output

Both mock_agent() and mock_pipeline() are also usable as decorators, since
contextlib.contextmanager gives that for free:

    @mock_agent(researcher, returns="fact 1, fact 2, fact 3")
    def test_writer_uses_research():
        ...
"""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import TypeAdapter

from .core.agent import AgentNode
from .core.context import AgentOutputError
from .core.pipeline import BranchAgent, ParallelGroup, Pipeline
from .providers.base import LLMResponse


def _fake_response() -> LLMResponse:
    return LLMResponse(
        content="", model="mock", input_tokens=0, output_tokens=0, latency_ms=0.0, provider="mock"
    )


def _coerce(node: AgentNode, value: Any) -> Any:
    """Validate a mocked return value against the agent's declared output type, the
    same way a real (parsed) response would be — so a test that mocks a Pydantic-typed
    agent with a plain dict gets the same typed object downstream code expects."""
    if node.output_type is str or value is None:
        return value
    try:
        return TypeAdapter(node.output_type).validate_python(value)
    except Exception as exc:
        raise AgentOutputError(node.name, repr(value), exc) from exc


@contextlib.contextmanager
def mock_agent(node: AgentNode, *, returns: Any = None, raises: Exception | None = None):
    """Replace one agent's LLM call with a fixed return value (or exception) for the
    duration of the `with` block — no network call happens, no API key needed.

    `returns` is validated against the agent's declared return type exactly like a
    real response would be. `raises` is raised on every call attempt, so it goes
    through the agent's normal retry/fallback handling — useful for testing those
    paths without queuing fake provider failures."""
    if not isinstance(node, AgentNode):
        raise TypeError(f"mock_agent() expects an @agent-decorated function, got {type(node)!r}")

    value = _coerce(node, returns) if raises is None else None
    original_call = node._call

    async def fake_call(inputs: dict[str, Any]):
        if raises is not None:
            raise raises
        return value, _fake_response()

    node._call = fake_call
    try:
        yield
    finally:
        node._call = original_call


def _stub_node() -> AgentNode:
    def _mocked_pipeline() -> str:
        """Stand-in node installed by mock_pipeline() — never actually called."""

    return AgentNode(_mocked_pipeline, model="mock")


@contextlib.contextmanager
def mock_pipeline(target: Pipeline, *, returns: Any = None, raises: Exception | None = None):
    """Replace an entire pipeline's execution with a fixed return value (or
    exception) for the duration of the `with` block — none of its agents are called.

    Use this when you only care about what calls the pipeline (an API endpoint, a
    cron job) rather than what the pipeline itself does; use mock_agent() to stub
    out one step of a pipeline you still want to actually run."""
    if isinstance(target, (AgentNode, ParallelGroup, BranchAgent)):
        raise TypeError(
            "mock_pipeline() expects a pipeline(...) object, not a single agent — "
            "use mock_agent() to mock one agent."
        )
    if not isinstance(target, Pipeline):
        raise TypeError(f"mock_pipeline() expects a pipeline(...) object, got {type(target)!r}")

    stub = _stub_node()

    with mock_agent(stub, returns=returns, raises=raises):
        original_topo = target.topological_order
        target.topological_order = lambda: [stub]
        try:
            yield
        finally:
            target.topological_order = original_topo
