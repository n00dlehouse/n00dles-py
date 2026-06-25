from __future__ import annotations

import pytest

from n00dles import PipelineFailure, RetryPolicy, agent, arun, pipeline, run
from n00dles.core.context import PipelineContext
from n00dles.core.state import InMemoryStateStore

from .conftest import FakeProvider


def test_run_single_agent_directly():
    provider = FakeProvider(["the answer"])

    @agent(model="fake-model", provider=provider)
    def researcher(topic: str) -> str:
        """Research."""

    result = run(researcher, topic="multi-agent AI")
    assert result.output == "the answer"
    assert result.run_id
    assert len(result.agent_traces) == 1
    assert result.agent_traces[0].name == "researcher"
    assert result.agent_traces[0].status == "ok"


def test_run_two_agent_pipeline_passes_output_forward():
    provider = FakeProvider(["research facts", "final article"])

    @agent(model="fake-model", provider=provider)
    def researcher(topic: str) -> str:
        """Research the topic."""

    @agent(model="fake-model", provider=provider)
    def writer(research: str) -> str:
        """Write from the research."""

    result = run(pipeline(researcher >> writer), topic="orchestration")

    assert result.output == "final article"
    # writer's user-prompt should be researcher's output, proving data flowed through
    _, writer_user_prompt = provider.calls[1]
    assert writer_user_prompt == "research facts"
    assert [t.name for t in result.agent_traces] == ["researcher", "writer"]


def test_run_retries_transient_failure_then_succeeds():
    provider = FakeProvider([RuntimeError("transient"), RuntimeError("transient"), "success"])

    @agent(model="fake-model", provider=provider, retry=RetryPolicy(max_attempts=5, base_delay=0))
    def flaky(x: str) -> str:
        """Sometimes fails."""

    result = run(flaky, x="go")
    assert result.output == "success"
    assert result.agent_traces[0].attempts == 3
    assert provider.call_count == 3


def test_run_raises_pipeline_failure_after_exhausting_retries():
    provider = FakeProvider([RuntimeError("down")] * 10)

    @agent(model="fake-model", provider=provider, retry=RetryPolicy(max_attempts=2, base_delay=0))
    def always_fails(x: str) -> str:
        """Always fails."""

    with pytest.raises(PipelineFailure) as exc_info:
        run(always_fails, x="go")
    assert exc_info.value.node_name == "always_fails"
    assert provider.call_count == 2


def test_run_uses_fallback_agent_on_primary_failure():
    primary_provider = FakeProvider([RuntimeError("primary down")] * 5)
    fallback_provider = FakeProvider(["fallback result"])

    @agent(model="fake-model", provider=fallback_provider)
    def backup(x: str) -> str:
        """Backup agent."""

    @agent(model="fake-model", provider=primary_provider, retry=1, fallback=backup)
    def primary(x: str) -> str:
        """Primary agent, will fail."""

    result = run(primary, x="go")
    assert result.output == "fallback result"
    assert result.agent_traces[0].status == "ok"
    assert "fell back" in result.agent_traces[0].error


@pytest.mark.asyncio
async def test_resume_skips_already_completed_nodes():
    # researcher is pre-seeded as already complete (simulating a crash after it ran),
    # so only writer actually calls the provider — queue just its response.
    provider = FakeProvider(["final article"])
    state = InMemoryStateStore()

    @agent(model="fake-model", provider=provider)
    def researcher(topic: str) -> str:
        """Research."""

    @agent(model="fake-model", provider=provider)
    def writer(research: str) -> str:
        """Write."""

    p = pipeline(researcher >> writer)

    # Simulate a crash after the first node: pre-seed state with researcher already done.
    ctx = PipelineContext(run_id="resumed_run", inputs={"topic": "x"})
    ctx.checkpoint("researcher", "pre-seeded research")
    await state.save("resumed_run", ctx)

    from n00dles.core.executor import PipelineExecutor

    executor = PipelineExecutor(state=state)
    result = await executor.execute(p, "resumed_run", {"topic": "x"})

    # researcher must NOT have been called again — only writer should have run
    assert provider.call_count == 1
    assert provider.calls[0][1] == "pre-seeded research"  # writer received the seeded value
    assert result.output == "final article"


def test_arun_works_in_an_event_loop_context():
    import asyncio

    provider = FakeProvider(["ok"])

    @agent(model="fake-model", provider=provider)
    def a(x: str) -> str:
        """A."""

    async def main():
        return await arun(a, x="go")

    result = asyncio.run(main())
    assert result.output == "ok"


def test_run_inside_running_loop_raises_helpful_error():
    import asyncio

    provider = FakeProvider(["ok"])

    @agent(model="fake-model", provider=provider)
    def a(x: str) -> str:
        """A."""

    async def main():
        run(a, x="go")  # calling sync run() from inside a running loop

    with pytest.raises(RuntimeError, match="arun"):
        asyncio.run(main())
