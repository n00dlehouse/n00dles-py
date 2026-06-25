from __future__ import annotations

import time
from typing import Any, Protocol

from .events import TraceEvent


class Exporter(Protocol):
    def export(self, event: TraceEvent) -> None: ...


class Tracer:
    """Internal event bus. The executor opens a `span()` per node call; exporters are
    pluggable sinks (OTel, Langfuse, Helicone, or a user's own Exporter) that never
    block or fail a pipeline run — export errors are swallowed, not raised."""

    def __init__(self, exporters: list[Exporter] | None = None):
        self.exporters = list(exporters or [])

    def add_exporter(self, exporter: Exporter) -> None:
        self.exporters.append(exporter)

    def emit(self, event: TraceEvent) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(event)
            except Exception:  # noqa: BLE001 - an exporter must never break a run
                pass

    def span(self, run_id: str, node_name: str, model: str | None = None) -> Span:
        return Span(self, run_id, node_name, model)


class Span:
    def __init__(self, tracer: Tracer, run_id: str, node_name: str, model: str | None):
        self.tracer = tracer
        self.run_id = run_id
        self.node_name = node_name
        self.model = model
        self._t0 = 0.0

    def __enter__(self) -> Span:
        self._t0 = time.monotonic()
        self.tracer.emit(
            TraceEvent.now(
                run_id=self.run_id,
                node_name=self.node_name,
                event_type="start",
                model=self.model,
            )
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # never suppress — ok()/fail() are called explicitly by the executor

    def record_attempt(self, attempt: int, exc: Exception) -> None:
        self.tracer.emit(
            TraceEvent.now(
                run_id=self.run_id,
                node_name=self.node_name,
                event_type="retry",
                model=self.model,
                attempt=attempt,
                error=str(exc),
            )
        )

    def ok(self, llm_response: Any = None) -> None:
        latency_ms = (time.monotonic() - self._t0) * 1000
        self.tracer.emit(
            TraceEvent.now(
                run_id=self.run_id,
                node_name=self.node_name,
                event_type="complete",
                model=self.model,
                latency_ms=latency_ms,
                input_tokens=getattr(llm_response, "input_tokens", None),
                output_tokens=getattr(llm_response, "output_tokens", None),
            )
        )

    def fail(self, exc: Exception) -> None:
        latency_ms = (time.monotonic() - self._t0) * 1000
        self.tracer.emit(
            TraceEvent.now(
                run_id=self.run_id,
                node_name=self.node_name,
                event_type="fail",
                model=self.model,
                latency_ms=latency_ms,
                error=str(exc),
            )
        )
