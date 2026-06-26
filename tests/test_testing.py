from __future__ import annotations

import pytest
from pydantic import BaseModel

from n00dles import AgentOutputError, PipelineFailure, RetryPolicy, agent, parallel, pipeline, run
from n00dles.testing import mock_agent, mock_pipeline

from .conftest import FakeProvider


class Sentiment(BaseModel):
    label: str
    confidence: float


def _explode(*_a, **_k):
    raise AssertionError("the real provider should never be called while mocked")


class ExplodingProvider:
    async def complete(self, prompt, system, **kwargs):
        _explode()


def test_mock_agent_replaces_return_value_without_calling_provider():
    @agent(model="fake-model", provider=ExplodingProvider())
    def researcher(topic: str) -> str:
        """Research."""

    with mock_agent(researcher, returns="fact 1, fact 2, fact 3"):
        result = run(researcher, topic="anything")

    assert result.output == "fact 1, fact 2, fact 3"


def test_mock_agent_coerces_dict_into_declared_pydantic_model():
    @agent(model="fake-model", provider=ExplodingProvider())
    def classify(review: str) -> Sentiment:
        """Classify."""

    with mock_agent(classify, returns={"label": "positive", "confidence": 0.9}):
        result = run(classify, review="great product")

    assert isinstance(result.output, Sentiment)
    assert result.output.label == "positive"
    assert result.output.confidence == 0.9


def test_mock_agent_raises_agent_output_error_on_bad_shape():
    """Validation happens eagerly, as soon as mock_agent() is entered — not lazily on
    the next run() — so a wrong-shaped `returns` fails at the mock site, not buried
    inside a PipelineFailure raised somewhere later in the test."""

    @agent(model="fake-model", provider=ExplodingProvider())
    def classify(review: str) -> Sentiment:
        """Classify."""

    with pytest.raises(AgentOutputError):
        with mock_agent(classify, returns={"label": "positive"}):  # missing confidence
            pass


def test_mock_agent_raises_goes_through_retry_and_fallback():
    fallback_provider = FakeProvider(["fallback result"])

    @agent(model="fake-model", provider=fallback_provider)
    def backup(x: str) -> str:
        """Backup."""

    @agent(
        model="fake-model",
        provider=ExplodingProvider(),
        retry=RetryPolicy(max_attempts=1, base_delay=0),
        fallback=backup,
    )
    def flaky(x: str) -> str:
        """Flaky."""

    with mock_agent(flaky, raises=TimeoutError("simulated provider timeout")):
        result = run(flaky, x="go")

    assert result.output == "fallback result"
    assert any("simulated provider timeout" in (t.error or "") for t in result.agent_traces)


def test_mock_agent_restores_original_call_after_exiting():
    provider = FakeProvider(["real response"])

    @agent(model="fake-model", provider=provider)
    def researcher(topic: str) -> str:
        """Research."""

    with mock_agent(researcher, returns="mocked"):
        mocked_result = run(researcher, topic="x")

    real_result = run(researcher, topic="x")

    assert mocked_result.output == "mocked"
    assert real_result.output == "real response"
    assert provider.call_count == 1  # only the second, unmocked run hit the provider


def test_mock_agent_works_as_a_decorator():
    @agent(model="fake-model", provider=ExplodingProvider())
    def researcher(topic: str) -> str:
        """Research."""

    @mock_agent(researcher, returns="decorated mock")
    def run_inside_decorator():
        return run(researcher, topic="x")

    result = run_inside_decorator()
    assert result.output == "decorated mock"


def test_mock_agent_rejects_non_agent_node():
    with pytest.raises(TypeError, match="mock_agent"):
        with mock_agent("not an agent", returns="x"):  # type: ignore[arg-type]
            pass


def test_mock_pipeline_replaces_whole_run_without_calling_any_agent():
    @agent(model="fake-model", provider=ExplodingProvider())
    def researcher(topic: str) -> str:
        """Research."""

    @agent(model="fake-model", provider=ExplodingProvider())
    def writer(research: str) -> str:
        """Write."""

    content_pipeline = pipeline(researcher >> writer)

    with mock_pipeline(content_pipeline, returns="mocked article"):
        result = run(content_pipeline, topic="x")

    assert result.output == "mocked article"
    assert len(result.agent_traces) == 1  # only the stub ran, not researcher/writer


def test_mock_pipeline_with_parallel_group_still_short_circuits():
    @agent(model="fake-model", provider=ExplodingProvider())
    def a(x: str) -> str:
        """A."""

    @agent(model="fake-model", provider=ExplodingProvider())
    def b(x: str) -> str:
        """B."""

    fan_out = pipeline(parallel(a, b))

    with mock_pipeline(fan_out, returns={"a": "1", "b": "2"}):
        result = run(fan_out, x="go")

    assert result.output == {"a": "1", "b": "2"}


def test_mock_pipeline_raises_propagates_as_pipeline_failure():
    @agent(model="fake-model", provider=ExplodingProvider())
    def researcher(topic: str) -> str:
        """Research."""

    content_pipeline = pipeline(researcher, retry=RetryPolicy(max_attempts=1, base_delay=0))

    with mock_pipeline(content_pipeline, raises=RuntimeError("downstream is down")):
        with pytest.raises(PipelineFailure, match="downstream is down"):
            run(content_pipeline, topic="x")


def test_mock_pipeline_restores_topological_order_after_exiting():
    provider = FakeProvider(["real output"])

    @agent(model="fake-model", provider=provider)
    def researcher(topic: str) -> str:
        """Research."""

    content_pipeline = pipeline(researcher)

    with mock_pipeline(content_pipeline, returns="mocked"):
        mocked_result = run(content_pipeline, topic="x")

    real_result = run(content_pipeline, topic="x")

    assert mocked_result.output == "mocked"
    assert real_result.output == "real output"


def test_mock_pipeline_works_as_a_decorator():
    @agent(model="fake-model", provider=ExplodingProvider())
    def researcher(topic: str) -> str:
        """Research."""

    content_pipeline = pipeline(researcher)

    @mock_pipeline(content_pipeline, returns="decorated mock")
    def run_inside_decorator():
        return run(content_pipeline, topic="x")

    result = run_inside_decorator()
    assert result.output == "decorated mock"


def test_mock_pipeline_rejects_a_bare_agent_node():
    @agent(model="fake-model", provider=ExplodingProvider())
    def researcher(topic: str) -> str:
        """Research."""

    with pytest.raises(TypeError, match="mock_agent"):
        with mock_pipeline(researcher, returns="x"):  # type: ignore[arg-type]
            pass
