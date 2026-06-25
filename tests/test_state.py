from __future__ import annotations

import pytest

from n00dles.core.context import PipelineContext
from n00dles.core.state import InMemoryStateStore, SQLiteStateStore


@pytest.mark.asyncio
async def test_sqlite_save_and_load_roundtrip(tmp_path):
    store = SQLiteStateStore(path=str(tmp_path / "state.db"))
    ctx = PipelineContext(run_id="run_1", inputs={"topic": "x"})
    ctx.checkpoint("researcher", "facts about x")

    await store.save("run_1", ctx)
    loaded = await store.load("run_1")

    assert loaded is not None
    assert loaded.run_id == "run_1"
    assert loaded.inputs == {"topic": "x"}
    assert loaded.completed == {"researcher": "facts about x"}
    assert loaded.is_complete("researcher")


@pytest.mark.asyncio
async def test_sqlite_load_missing_run_returns_none(tmp_path):
    store = SQLiteStateStore(path=str(tmp_path / "state.db"))
    assert await store.load("does-not-exist") is None


@pytest.mark.asyncio
async def test_sqlite_load_or_create_creates_fresh_context(tmp_path):
    store = SQLiteStateStore(path=str(tmp_path / "state.db"))
    ctx = await store.load_or_create("new_run", {"topic": "x"})
    assert ctx.run_id == "new_run"
    assert ctx.inputs == {"topic": "x"}
    assert ctx.completed == {}


@pytest.mark.asyncio
async def test_sqlite_delete_removes_run(tmp_path):
    store = SQLiteStateStore(path=str(tmp_path / "state.db"))
    ctx = PipelineContext(run_id="run_1", inputs={})
    await store.save("run_1", ctx)
    await store.delete("run_1")
    assert await store.load("run_1") is None


@pytest.mark.asyncio
async def test_in_memory_store_roundtrip():
    store = InMemoryStateStore()
    ctx = PipelineContext(run_id="run_1", inputs={"a": 1})
    ctx.checkpoint("step1", "result1")
    await store.save("run_1", ctx)

    loaded = await store.load("run_1")
    assert loaded.completed == {"step1": "result1"}

    await store.delete("run_1")
    assert await store.load("run_1") is None


def test_pipeline_context_resolve_inputs_prefers_completed_over_run_inputs():
    # "x" exists both as an original run input and as a prior node's checkpointed
    # output (same param name reused, as `>>` chaining naturally produces) — the
    # prior node's output must win, matching the DAG-chaining semantics of `>>`.
    ctx = PipelineContext(run_id="r1", inputs={"x": "original-run-input"})
    ctx.checkpoint("step1", "step1-output")
    ctx.completed["x"] = "step-output-named-x"  # simulate a node literally named/aliased "x"
    resolved = ctx.resolve_inputs(["x"])
    assert resolved == {"x": "step-output-named-x"}

    # falls back to run inputs when the name was never checkpointed
    resolved2 = ctx.resolve_inputs(["topic"])
    assert resolved2 == {}

    ctx2 = PipelineContext(run_id="r2", inputs={"topic": "original"})
    assert ctx2.resolve_inputs(["topic"]) == {"topic": "original"}


def test_pipeline_context_final_output_is_last_checkpointed():
    ctx = PipelineContext(run_id="r1", inputs={})
    ctx.checkpoint("a", "first")
    ctx.checkpoint("b", "second")
    assert ctx.final_output() == "second"


def test_pipeline_context_json_roundtrip():
    ctx = PipelineContext(run_id="r1", inputs={"x": 1})
    ctx.checkpoint("a", "result")
    restored = PipelineContext.from_json(ctx.to_json())
    assert restored.to_dict() == ctx.to_dict()
