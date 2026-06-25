from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryPolicy:
    """Exponential backoff with jitter, applied per agent call."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(default=(Exception,))

    def delay(self, attempt: int) -> float:
        d = min(self.base_delay * (2**attempt), self.max_delay)
        return d + (random.uniform(0, d * 0.1) if self.jitter else 0)

    def is_retryable(self, exc: Exception) -> bool:
        return isinstance(exc, self.retryable_exceptions)


async def with_retry(
    policy: RetryPolicy,
    call: Callable[[], Any],
    on_attempt: Callable[[int, Exception], None] | None = None,
) -> Any:
    """Call `call()` (a zero-arg async callable), retrying per `policy` until it succeeds."""
    last_exc: Exception | None = None
    for attempt in range(policy.max_attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - intentionally broad, filtered below
            last_exc = exc
            if on_attempt:
                on_attempt(attempt, exc)
            if not policy.is_retryable(exc):
                raise
            if attempt < policy.max_attempts - 1:
                await asyncio.sleep(policy.delay(attempt))
    assert last_exc is not None
    raise last_exc
