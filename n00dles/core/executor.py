from __future__ import annotations

import asyncio
import time

from ..telemetry.tracer import Tracer
from .context import AgentTrace, PipelineFailure, RunResult
from .pipeline import Pipeline
from .retry import with_retry
from .state import StateStore


class PipelineExecutor:
    """Walks a Pipeline's nodes in order, dispatching each agent call via the provider,
    checkpointing state after every node, and propagating structured failure through
    the call chain. Resumes from the last checkpoint if `run_id` already has state."""

    def __init__(self, state: StateStore, tracer: Tracer | None = None):
        self.state = state
        self.tracer = tracer or Tracer()

    async def execute(self, pipeline: Pipeline, run_id: str, inputs: dict) -> RunResult:
        t0 = time.monotonic()
        ctx = await self.state.load_or_create(run_id, inputs)
        traces: list[AgentTrace] = []

        # Tracks the immediately preceding node's output for sequential `>>` chaining,
        # where a single-parameter node receives the prior node's output positionally
        # regardless of parameter-name matching (e.g. researcher() -> writer(research)).
        # Multi-parameter nodes (relevant once parallel()/branch() land) still resolve
        # by parameter name via ctx.resolve_inputs().
        prev_result: object = None
        have_prev = False

        for node in pipeline.topological_order():
            if ctx.is_complete(node.name):
                prev_result = ctx.completed[node.name]
                have_prev = True
                continue

            if have_prev and len(node.param_names) == 1:
                node_inputs = {node.param_names[0]: prev_result}
            else:
                node_inputs = ctx.resolve_inputs(node.param_names)
            timeout = pipeline.resolve_timeout(node)
            retry_policy = pipeline.resolve_retry(node)
            attempt_count = 0

            with self.tracer.span(run_id, node.name, model=node.model) as span:
                try:

                    async def call(_node=node, _node_inputs=node_inputs, _timeout=timeout):
                        nonlocal attempt_count
                        attempt_count += 1
                        return await asyncio.wait_for(_node._call(_node_inputs), timeout=_timeout)

                    result, llm_resp = await with_retry(retry_policy, call, span.record_attempt)
                    ctx.checkpoint(node.name, result)
                    await self.state.save(run_id, ctx)
                    prev_result, have_prev = result, True
                    span.ok(llm_resp)
                    traces.append(
                        AgentTrace(
                            name=node.name,
                            status="ok",
                            duration_ms=llm_resp.latency_ms,
                            input_tokens=llm_resp.input_tokens,
                            output_tokens=llm_resp.output_tokens,
                            attempts=attempt_count,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - converted to PipelineFailure below
                    span.fail(exc)
                    if node.fallback is not None:
                        result, llm_resp = await node.fallback._call(node_inputs)
                        ctx.checkpoint(node.name, result)
                        await self.state.save(run_id, ctx)
                        prev_result, have_prev = result, True
                        traces.append(
                            AgentTrace(
                                name=node.name,
                                status="ok",
                                duration_ms=llm_resp.latency_ms,
                                attempts=attempt_count,
                                error=f"primary failed, fell back: {exc}",
                            )
                        )
                    else:
                        traces.append(
                            AgentTrace(
                                name=node.name,
                                status="failed",
                                duration_ms=0,
                                attempts=attempt_count,
                                error=str(exc),
                            )
                        )
                        raise PipelineFailure(run_id, node.name, exc) from exc

        total_tokens = sum((t.input_tokens or 0) + (t.output_tokens or 0) for t in traces)
        return RunResult(
            run_id=run_id,
            output=ctx.final_output(),
            duration_ms=(time.monotonic() - t0) * 1000,
            total_tokens=total_tokens,
            agent_traces=traces,
        )
