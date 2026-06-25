from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel

from ..telemetry.tracer import Tracer
from .agent import AgentNode
from .context import AgentTrace, BranchError, PipelineContext, PipelineFailure, RunResult
from .pipeline import BranchAgent, ParallelGroup, Pipeline
from .retry import with_retry
from .state import StateStore


class PipelineExecutor:
    """Walks a Pipeline's nodes in order, dispatching each step (a single agent, a
    ParallelGroup fan-out, or a BranchAgent route) via the provider, checkpointing
    state after every step, and propagating structured failure through the call
    chain. Resumes from the last checkpoint if `run_id` already has state."""

    def __init__(self, state: StateStore, tracer: Tracer | None = None):
        self.state = state
        self.tracer = tracer or Tracer()

    async def execute(self, pipeline: Pipeline, run_id: str, inputs: dict) -> RunResult:
        t0 = time.monotonic()
        ctx = await self.state.load_or_create(run_id, inputs)
        traces: list[AgentTrace] = []

        # Tracks the immediately preceding step's output for sequential `>>` chaining,
        # where a single-parameter agent receives the prior step's output positionally
        # regardless of parameter-name matching (e.g. researcher() -> writer(research)).
        # Multi-parameter agents — and anything downstream of a ParallelGroup, where
        # there's no single coherent "previous result" to position-pass — resolve by
        # parameter name via ctx.resolve_inputs() instead.
        prev_result: Any = None
        have_prev = False

        for node in pipeline.topological_order():
            if ctx.is_complete(node.name):
                prev_result = ctx.completed[node.name]
                have_prev = not isinstance(node, ParallelGroup)
                continue

            if isinstance(node, ParallelGroup):
                result = await self._execute_parallel(node, pipeline, ctx, run_id, traces)
                ctx.checkpoint(node.name, result)
                await self.state.save(run_id, ctx)
                prev_result, have_prev = result, False
                continue

            if isinstance(node, BranchAgent):
                selected, node_inputs = self._resolve_branch(node, ctx, prev_result, have_prev)
                result = await self._execute_node(
                    selected, node_inputs, pipeline, ctx, run_id, traces
                )
                ctx.checkpoint(node.name, result)
                await self.state.save(run_id, ctx)
                prev_result, have_prev = result, True
                continue

            if have_prev and len(node.param_names) == 1:
                node_inputs = {node.param_names[0]: prev_result}
            else:
                node_inputs = ctx.resolve_inputs(node.param_names)
            result = await self._execute_node(node, node_inputs, pipeline, ctx, run_id, traces)
            prev_result, have_prev = result, True

        total_tokens = sum((t.input_tokens or 0) + (t.output_tokens or 0) for t in traces)
        return RunResult(
            run_id=run_id,
            output=ctx.final_output(),
            duration_ms=(time.monotonic() - t0) * 1000,
            total_tokens=total_tokens,
            agent_traces=traces,
        )

    async def _execute_node(
        self,
        node: AgentNode,
        node_inputs: dict,
        pipeline: Pipeline,
        ctx: PipelineContext,
        run_id: str,
        traces: list[AgentTrace],
    ) -> Any:
        """Runs a single AgentNode (retry + tracing + fallback), checkpoints its
        result, and returns the parsed output. Shared by the main sequential loop,
        parallel fan-out, and branch routing — they only differ in how `node_inputs`
        and `node` itself are chosen."""
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
                return result
            except Exception as exc:  # noqa: BLE001 - converted to PipelineFailure below
                span.fail(exc)
                if node.fallback is not None:
                    fb_result, fb_resp = await node.fallback._call(node_inputs)
                    ctx.checkpoint(node.name, fb_result)
                    await self.state.save(run_id, ctx)
                    traces.append(
                        AgentTrace(
                            name=node.name,
                            status="ok",
                            duration_ms=fb_resp.latency_ms,
                            attempts=attempt_count,
                            error=f"primary failed, fell back: {exc}",
                        )
                    )
                    return fb_result
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

    async def _execute_parallel(
        self,
        group: ParallelGroup,
        pipeline: Pipeline,
        ctx: PipelineContext,
        run_id: str,
        traces: list[AgentTrace],
    ) -> dict[str, Any]:
        """Fans out every member of `group` concurrently, each resolving its own
        inputs from the shared context (typically the same upstream value). Returns —
        and checkpoints each member under — a dict keyed by member name, so a
        downstream agent with matching parameter names receives them with no manual
        merging. Members already checkpointed from a prior partial run (crash mid-
        fan-out) are skipped, not re-run."""
        semaphore = asyncio.Semaphore(group.max_concurrency) if group.max_concurrency else None

        async def run_one(node: AgentNode) -> tuple[str, Any]:
            if ctx.is_complete(node.name):
                return node.name, ctx.completed[node.name]
            node_inputs = ctx.resolve_inputs(node.param_names)
            if semaphore is not None:
                async with semaphore:
                    result = await self._execute_node(
                        node, node_inputs, pipeline, ctx, run_id, traces
                    )
            else:
                result = await self._execute_node(node, node_inputs, pipeline, ctx, run_id, traces)
            return node.name, result

        pairs = await asyncio.gather(*(run_one(n) for n in group.nodes))
        return dict(pairs)

    def _resolve_branch(
        self,
        branch_node: BranchAgent,
        ctx: PipelineContext,
        prev_result: Any,
        have_prev: bool,
    ) -> tuple[AgentNode, dict]:
        """Picks the route agent and resolves its inputs, but doesn't execute it —
        execution goes through the shared _execute_node so retry/tracing/fallback stay
        identical to every other agent call."""
        key = self._route_key(prev_result)
        selected = branch_node.routes.get(key) if key is not None else None
        if selected is None:
            selected = branch_node.default
        if selected is None:
            raise BranchError(branch_node.name, key)

        if have_prev and len(selected.param_names) == 1:
            node_inputs = {selected.param_names[0]: prev_result}
        else:
            node_inputs = ctx.resolve_inputs(selected.param_names)
        return selected, node_inputs

    @staticmethod
    def _route_key(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("category")
        if isinstance(value, BaseModel):
            return getattr(value, "category", None)
        return None
