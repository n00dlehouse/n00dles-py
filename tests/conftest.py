from __future__ import annotations

import itertools

import pytest

import n00dles
from n00dles.core.state import InMemoryStateStore
from n00dles.providers.base import BaseProvider, LLMResponse


@pytest.fixture(autouse=True)
def isolate_default_state_store():
    """n00dles caches a process-wide default StateStore (a real SQLiteStateStore
    pointing at ./n00dles_state.db) the first time run()/arun() is called without an
    explicit configure(). Without this fixture every test that calls bare run() would
    share that one on-disk file and leave it behind in the repo. Force an in-memory
    store for the duration of each test instead, and restore whatever was there after."""
    previous = n00dles._default_state_store
    n00dles._default_state_store = InMemoryStateStore()
    yield
    n00dles._default_state_store = previous


class FakeProvider(BaseProvider):
    """A BaseProvider that never touches the network. Returns a queued sequence of
    responses (or raises queued exceptions) so tests can script exact agent behavior,
    including transient-failure-then-success sequences for retry tests."""

    def __init__(self, responses: list[str | Exception] | None = None):
        self._queue = list(responses or ["ok"])
        last = self._queue[-1] if self._queue else "ok"
        self._iter = itertools.chain(self._queue, itertools.repeat(last))
        self.call_count = 0
        self.calls: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.calls.append((system, prompt))
        item = next(self._iter)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            content=item,
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            provider="fake",
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
