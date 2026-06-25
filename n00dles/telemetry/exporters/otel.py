from __future__ import annotations

from ..events import TraceEvent


class OTelExporter:
    """Exports every TraceEvent as an OpenTelemetry span. Requires the `otel` extra
    (`pip install n00dles[otel]`) — the import is deferred to construction time so the
    base package never requires opentelemetry-sdk."""

    def __init__(self, service_name: str = "n00dles"):
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:
            raise ImportError(
                "OTelExporter requires the 'otel' extra: pip install n00dles[otel]"
            ) from exc
        self._tracer = otel_trace.get_tracer(service_name)

    def export(self, event: TraceEvent) -> None:
        span = self._tracer.start_span(f"{event.run_id}/{event.node_name}")
        span.set_attribute("n00dles.node", event.node_name)
        span.set_attribute("n00dles.event_type", event.event_type)
        span.set_attribute("n00dles.model", event.model or "")
        span.set_attribute("n00dles.input_tokens", event.input_tokens or 0)
        span.set_attribute("n00dles.output_tokens", event.output_tokens or 0)
        if event.error:
            span.set_attribute("n00dles.error", event.error)
        span.end()
