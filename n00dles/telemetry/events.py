from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TraceEvent:
    run_id: str
    node_name: str
    event_type: str  # "start" | "complete" | "retry" | "fail"
    timestamp: datetime
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    attempt: int = 0
    error: str | None = None

    @classmethod
    def now(cls, **kwargs) -> TraceEvent:
        return cls(timestamp=datetime.now(timezone.utc), **kwargs)
