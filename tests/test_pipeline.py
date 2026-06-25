from __future__ import annotations

from n00dles import RetryPolicy, agent, pipeline
from n00dles.core.pipeline import Pipeline

from .conftest import FakeProvider


def make_agents(provider: FakeProvider):
    @agent(model="fake-model", provider=provider)
    def researcher(topic: str) -> str:
        """Research."""

    @agent(model="fake-model", provider=provider)
    def writer(research: str) -> str:
        """Write."""

    return researcher, writer


def test_rshift_builds_pipeline(fake_provider: FakeProvider):
    researcher, writer = make_agents(fake_provider)
    chain = researcher >> writer
    assert isinstance(chain, Pipeline)
    assert chain.nodes == [researcher, writer]


def test_rshift_three_way_chain_is_flat(fake_provider: FakeProvider):
    researcher, writer = make_agents(fake_provider)

    @agent(model="fake-model", provider=fake_provider)
    def editor(draft: str) -> str:
        """Edit."""

    chain = researcher >> writer >> editor
    assert [n.name for n in chain.nodes] == ["researcher", "writer", "editor"]


def test_topological_order_matches_declared_order(fake_provider: FakeProvider):
    researcher, writer = make_agents(fake_provider)
    p = pipeline(researcher >> writer)
    assert p.topological_order() == [researcher, writer]


def test_pipeline_fills_in_unset_node_retry_and_timeout(fake_provider: FakeProvider):
    researcher, writer = make_agents(fake_provider)
    p = pipeline(researcher >> writer, retry=5, timeout=42)
    assert p.resolve_retry(researcher).max_attempts == 5
    assert p.resolve_timeout(researcher) == 42


def test_agent_explicit_retry_wins_over_pipeline_default():
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider, retry=7, timeout=99)
    def picky(x: str) -> str:
        """Picky agent with its own settings."""

    p = pipeline(picky, retry=2, timeout=10)
    assert p.resolve_retry(picky).max_attempts == 7
    assert p.resolve_timeout(picky) == 99


def test_same_node_in_two_pipelines_does_not_cross_contaminate():
    """Regression test: pipeline() must not mutate the AgentNode in place — otherwise
    reusing an agent across two pipelines with different settings leaks state."""
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider)
    def shared(x: str) -> str:
        """Shared agent, no explicit retry/timeout."""

    p1 = pipeline(shared, retry=2, timeout=10)
    p2 = pipeline(shared, retry=9, timeout=99)

    assert p1.resolve_retry(shared).max_attempts == 2
    assert p1.resolve_timeout(shared) == 10
    assert p2.resolve_retry(shared).max_attempts == 9
    assert p2.resolve_timeout(shared) == 99
    # the node itself was never mutated
    assert shared.retry is None
    assert shared.timeout is None


def test_pipeline_without_explicit_settings_falls_back_to_node_defaults():
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider)
    def bare(x: str) -> str:
        """No explicit settings anywhere."""

    from n00dles.core.pipeline import Pipeline

    p = Pipeline([bare])  # constructed directly, bypassing pipeline()'s concrete defaults
    assert p.resolve_retry(bare) is bare.DEFAULT_RETRY
    assert p.resolve_timeout(bare) == bare.DEFAULT_TIMEOUT
    assert isinstance(p.resolve_retry(bare), RetryPolicy)
