from __future__ import annotations

import pytest
from pydantic import BaseModel

from n00dles import AgentOutputError, agent, run
from n00dles.core.agent import AgentNode

from .conftest import FakeProvider


# Module-level, not nested in a test function: get_type_hints() (used by AgentNode to
# parse the return-type contract) resolves forward references against the function's
# __globals__, so a return-type annotation referencing a class only defined inside the
# test function body would raise NameError.
class Sentiment(BaseModel):
    label: str
    confidence: float


class SimpleSentiment(BaseModel):
    label: str


def test_agent_decorator_returns_agent_node():
    @agent(model="fake-model")
    def researcher(topic: str) -> str:
        """Research the topic."""

    assert isinstance(researcher, AgentNode)
    assert researcher.name == "researcher"
    assert researcher.docstring == "Research the topic."
    assert researcher.model == "fake-model"
    assert researcher.output_type is str
    assert researcher.param_names == ["topic"]


def test_docstring_becomes_system_prompt(fake_provider: FakeProvider):
    @agent(model="fake-model", provider=fake_provider)
    def researcher(topic: str) -> str:
        """Research the topic thoroughly."""

    run(researcher, topic="multi-agent systems")
    system_prompt, user_prompt = fake_provider.calls[0]
    assert system_prompt == "Research the topic thoroughly."
    assert user_prompt == "multi-agent systems"


def test_explicit_prompt_overrides_docstring(fake_provider: FakeProvider):
    @agent(model="fake-model", provider=fake_provider, prompt="Custom system prompt.")
    def researcher(topic: str) -> str:
        """This docstring should be ignored."""

    run(researcher, topic="x")
    system_prompt, _ = fake_provider.calls[0]
    assert system_prompt == "Custom system prompt."


def test_str_output_passthrough():
    provider = FakeProvider(["plain text response"])

    @agent(model="fake-model", provider=provider)
    def echo(text: str) -> str:
        """Echo it back."""

    result = run(echo, text="hello")
    assert result.output == "plain text response"


def test_structured_output_validates_json():
    provider = FakeProvider(['{"label": "positive", "confidence": 0.97}'])

    @agent(model="fake-model", provider=provider)
    def analyze(review: str) -> Sentiment:
        """Analyze sentiment."""

    result = run(analyze, review="great product")
    assert isinstance(result.output, Sentiment)
    assert result.output.label == "positive"
    assert result.output.confidence == 0.97


def test_structured_output_strips_markdown_fences():
    provider = FakeProvider(['```json\n{"label": "positive"}\n```'])

    @agent(model="fake-model", provider=provider)
    def analyze(review: str) -> SimpleSentiment:
        """Analyze sentiment."""

    result = run(analyze, review="x")
    assert result.output.label == "positive"


def test_bad_json_raises_agent_output_error():
    provider = FakeProvider(["not json at all"])

    @agent(model="fake-model", provider=provider, retry=1)
    def analyze(review: str) -> SimpleSentiment:
        """Analyze sentiment."""

    with pytest.raises(Exception) as exc_info:
        run(analyze, review="x")
    # AgentOutputError is raised by the agent, then wrapped in PipelineFailure by the
    # executor, so check either the exception itself or its __cause__.
    err = exc_info.value
    assert isinstance(err, AgentOutputError) or isinstance(err.__cause__, AgentOutputError)


def test_multi_param_agent_serializes_inputs_as_json(fake_provider: FakeProvider):
    @agent(model="fake-model", provider=fake_provider)
    def merge(a: str, b: str) -> str:
        """Merge two things."""

    run(merge, a="x", b="y")
    _, user_prompt = fake_provider.calls[0]
    assert '"a": "x"' in user_prompt
    assert '"b": "y"' in user_prompt
