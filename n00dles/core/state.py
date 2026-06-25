from __future__ import annotations

import asyncio
import sqlite3
import time
from abc import ABC, abstractmethod

from .context import PipelineContext


class StateStore(ABC):
    """Pluggable checkpoint backend. The executor calls save() after every node and
    load() at the start of a run to support resume-on-restart."""

    @abstractmethod
    async def save(self, run_id: str, ctx: PipelineContext) -> None: ...

    @abstractmethod
    async def load(self, run_id: str) -> PipelineContext | None: ...

    @abstractmethod
    async def delete(self, run_id: str) -> None: ...

    async def load_or_create(self, run_id: str, inputs: dict) -> PipelineContext:
        existing = await self.load(run_id)
        if existing is not None:
            return existing
        return PipelineContext(run_id=run_id, inputs=inputs)


class SQLiteStateStore(StateStore):
    """Default backend. Zero external dependencies, zero config."""

    def __init__(self, path: str = "./n00dles_state.db"):
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )

    async def save(self, run_id: str, ctx: PipelineContext) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_save, run_id, ctx)

    def _sync_save(self, run_id: str, ctx: PipelineContext) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
                (run_id, ctx.to_json(), time.time()),
            )

    async def load(self, run_id: str) -> PipelineContext | None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_load, run_id)

    def _sync_load(self, run_id: str) -> PipelineContext | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return PipelineContext.from_json(row[0]) if row else None

    async def delete(self, run_id: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_delete, run_id)

    def _sync_delete(self, run_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


class InMemoryStateStore(StateStore):
    """Process-local, non-persistent. Useful for tests and short-lived scripts that
    don't want a database file on disk."""

    def __init__(self):
        self._runs: dict[str, PipelineContext] = {}

    async def save(self, run_id: str, ctx: PipelineContext) -> None:
        self._runs[run_id] = ctx

    async def load(self, run_id: str) -> PipelineContext | None:
        return self._runs.get(run_id)

    async def delete(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
