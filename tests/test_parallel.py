from __future__ import annotations

import asyncio

import pytest

from n00dles import RetryPolicy, agent, parallel, pipeline, run
from n00dles.core.pipeline import ParallelGroup
from n00dles.core.state import InMemoryStateStore

from .conftest import FakeProvider


def test_or_operator_builds_parallel_group():
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider)
    def a(x: str) -> str:
        """A."""

    @agent(model="fake-model", provider=provider)
    def b(x: str) -> str:
        """B."""

    group = a | b
    assert isinstance(group, ParallelGroup)
    assert [n.name for n in group.nodes] == ["a", "b"]


def test_or_operator_chains_three_way_flat():
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider)
    def a(x: str) -> str:
        """A."""

    @agent(model="fake-model", provider=provider)
    def b(x: str) -> str:
        """B."""

    @agent(model="fake-model", provider=provider)
    def c(x: str) -> str:
        """C."""

    group = a | b | c
    assert [n.name for n in group.nodes] == ["a", "b", "c"]


def test_parallel_function_requires_at_least_two_agents():
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider)
    def a(x: str) -> str:
        """A."""

    with pytest.raises(ValueError, match="at least 2"):
        parallel(a)


def test_run_parallel_group_directly_merges_by_name():
    provider = FakeProvider(["news result", "tweets result"])

    @agent(model="fake-model", provider=provider)
    def scrape_news(query: str) -> str:
        """Scrape news."""

    @agent(model="fake-model", provider=provider)
    def scrape_twitter(query: str) -> str:
        """Scrape twitter."""

    result = run(parallel(scrape_news, scrape_twitter), query="AI")
    assert result.output == {"scrape_news": "news result", "scrape_twitter": "tweets result"}
    assert {t.name for t in result.agent_traces} == {"scrape_news", "scrape_twitter"}


@pytest.mark.asyncio
async def test_parallel_members_run_concurrently_not_sequentially():
    """Two agents that each take 50ms should finish in ~50ms together, not ~100ms
    sequentially, proving asyncio.gather actually overlaps them."""

    class SlowProvider:
        async def complete(self, prompt, system, **kwargs):
            from n00dles.providers.base import LLMResponse

            await asyncio.sleep(0.05)
            return LLMResponse(
                content="ok", model="fake", input_tokens=1, output_tokens=1,
                latency_ms=50, provider="fake",
            )

    slow = SlowProvider()

    @agent(model="fake-model", provider=slow)
    def a(x: str) -> str:
        """A."""

    @agent(model="fake-model", provider=slow)
    def b(x: str) -> str:
        """B."""

    t0 = asyncio.get_event_loop().time()
    from n00dles import arun

    await arun(parallel(a, b), x="go")
    elapsed = asyncio.get_event_loop().time() - t0
    assert elapsed < 0.09, f"expected concurrent execution (~0.05s), took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_max_concurrency_limits_in_flight_calls():
    in_flight = 0
    max_seen = 0
    lock = asyncio.Lock()

    class TrackingProvider:
        async def complete(self, prompt, system, **kwargs):
            nonlocal in_flight, max_seen
            from n00dles.providers.base import LLMResponse

            async with lock:
                in_flight += 1
                max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.02)
            async with lock:
                in_flight -= 1
            return LLMResponse(
                content="ok", model="fake", input_tokens=1, output_tokens=1,
                latency_ms=1, provider="fake",
            )

    tracking = TrackingProvider()

    def make_agent(name):
        async def fn(x: str) -> str:
            """An agent."""

        fn.__name__ = name
        return agent(model="fake-model", provider=tracking)(fn)

    agents = [make_agent(f"agent_{i}") for i in range(4)]

    from n00dles import arun

    await arun(parallel(*agents, max_concurrency=2), x="go")
    assert max_seen <= 2


def test_downstream_agent_receives_merged_dict_by_matching_param_names():
    provider = FakeProvider(["news facts", "tweet facts", "merged summary"])

    @agent(model="fake-model", provider=provider)
    def scrape_news(query: str) -> str:
        """Scrape news."""

    @agent(model="fake-model", provider=provider)
    def scrape_twitter(query: str) -> str:
        """Scrape twitter."""

    @agent(model="fake-model", provider=provider)
    def merge_signals(scrape_news: str, scrape_twitter: str) -> str:
        """Merge signals."""

    intel = pipeline(parallel(scrape_news, scrape_twitter) >> merge_signals)
    result = run(intel, query="AI regulation")

    assert result.output == "merged summary"
    merge_call_system, merge_call_user = provider.calls[2]
    assert "news facts" in merge_call_user
    assert "tweet facts" in merge_call_user


def test_parallel_member_fallback_works():
    primary = FakeProvider([RuntimeError("down")] * 5)
    fallback_provider = FakeProvider(["fallback result"])
    healthy = FakeProvider(["healthy result"])

    @agent(model="fake-model", provider=fallback_provider)
    def backup(x: str) -> str:
        """Backup."""

    @agent(
        model="fake-model",
        provider=primary,
        retry=RetryPolicy(max_attempts=1, base_delay=0),
        fallback=backup,
    )
    def flaky(x: str) -> str:
        """Flaky."""

    @agent(model="fake-model", provider=healthy)
    def stable(x: str) -> str:
        """Stable."""

    result = run(parallel(flaky, stable), x="go")
    assert result.output == {"flaky": "fallback result", "stable": "healthy result"}


@pytest.mark.asyncio
async def test_parallel_resume_skips_already_completed_members():
    from n00dles.core.context import PipelineContext
    from n00dles.core.executor import PipelineExecutor

    provider = FakeProvider(["b result"])
    state = InMemoryStateStore()

    @agent(model="fake-model", provider=provider)
    def a(x: str) -> str:
        """A."""

    @agent(model="fake-model", provider=provider)
    def b(x: str) -> str:
        """B."""

    p = pipeline(parallel(a, b))

    # "a" already succeeded before a simulated crash; "b" never ran.
    ctx = PipelineContext(run_id="r1", inputs={"x": "go"})
    ctx.checkpoint("a", "pre-seeded a result")
    await state.save("r1", ctx)

    executor = PipelineExecutor(state=state)
    result = await executor.execute(p, "r1", {"x": "go"})

    assert provider.call_count == 1  # only "b" actually called the provider
    assert result.output == {"a": "pre-seeded a result", "b": "b result"}
