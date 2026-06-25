from __future__ import annotations

import pytest

from n00dles.core.retry import RetryPolicy, with_retry


def test_delay_grows_exponentially_without_jitter():
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=False)
    assert policy.delay(0) == 1.0
    assert policy.delay(1) == 2.0
    assert policy.delay(2) == 4.0
    assert policy.delay(3) == 8.0


def test_delay_is_capped_at_max_delay():
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=False)
    assert policy.delay(10) == 5.0


def test_delay_with_jitter_is_never_less_than_base():
    policy = RetryPolicy(base_delay=2.0, max_delay=30.0, jitter=True)
    for attempt in range(5):
        d = policy.delay(attempt)
        base = min(2.0 * (2**attempt), 30.0)
        assert base <= d <= base * 1.1


@pytest.mark.asyncio
async def test_with_retry_succeeds_without_retrying_on_first_success():
    calls = []

    async def call():
        calls.append(1)
        return "ok"

    result = await with_retry(RetryPolicy(max_attempts=3, base_delay=0), call)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_retries_then_succeeds():
    attempts = {"n": 0}

    async def call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "ok"

    result = await with_retry(RetryPolicy(max_attempts=5, base_delay=0), call)
    assert result == "ok"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_with_retry_exhausts_and_raises_last_exception():
    async def call():
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        await with_retry(RetryPolicy(max_attempts=3, base_delay=0), call)


@pytest.mark.asyncio
async def test_with_retry_calls_on_attempt_for_each_failure():
    seen = []

    async def call():
        raise ValueError("nope")

    policy = RetryPolicy(max_attempts=3, base_delay=0)
    with pytest.raises(ValueError):
        await with_retry(policy, call, lambda a, e: seen.append(a))
    assert seen == [0, 1, 2]


@pytest.mark.asyncio
async def test_non_retryable_exception_raises_immediately():
    calls = {"n": 0}

    async def call():
        calls["n"] += 1
        raise KeyError("not retryable")

    policy = RetryPolicy(max_attempts=5, base_delay=0, retryable_exceptions=(ValueError,))
    with pytest.raises(KeyError):
        await with_retry(policy, call)
    assert calls["n"] == 1
