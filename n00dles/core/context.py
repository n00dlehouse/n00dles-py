from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    """Lets checkpointing handle agent outputs that are Pydantic models, not just
    plain str/dict/list. Note: a restored checkpoint comes back as a plain dict, not
    the original model class — there's no per-node type registry to reconstruct it."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


@dataclass
class AgentTrace:
    name: str
    status: str  # "ok" | "failed"
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    error: str | None = None


@dataclass
class RunResult:
    run_id: str
    output: Any
    duration_ms: float = 0.0
    total_tokens: int = 0
    agent_traces: list[AgentTrace] = field(default_factory=list)
    #: Stored as-given for forward-compatibility; tag-based trace filtering ships
    #: with the Cloud dashboard, not in this release.
    tags: list[str] = field(default_factory=list)


class PipelineFailure(Exception):
    """Raised when a node fails after exhausting retries and has no fallback."""

    def __init__(self, run_id: str, node_name: str, cause: Exception):
        self.run_id = run_id
        self.node_name = node_name
        self.cause = cause
        super().__init__(f"pipeline {run_id} failed at node {node_name!r}: {cause}")


class AgentOutputError(Exception):
    """Raised when an LLM response can't be validated against the agent's declared return type."""

    def __init__(self, agent_name: str, raw_output: str, cause: Exception):
        self.agent_name = agent_name
        self.raw_output = raw_output
        self.cause = cause
        super().__init__(f"{agent_name}: output failed validation against return type: {cause}")


class BranchError(Exception):
    """Raised by branch() when the routing key matches none of its routes and no
    default agent was given."""

    def __init__(self, branch_name: str, key: Any):
        self.branch_name = branch_name
        self.key = key
        super().__init__(f"{branch_name}: no route for key {key!r} and no default agent set")


@dataclass
class PipelineContext:
    """Per-run state: inputs plus the output of every completed node, checkpointed
    after each step."""

    run_id: str
    inputs: dict[str, Any]
    completed: dict[str, Any] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def is_complete(self, node_name: str) -> bool:
        return node_name in self.completed

    def checkpoint(self, node_name: str, result: Any) -> None:
        self.completed[node_name] = result
        if node_name not in self.order:
            self.order.append(node_name)

    def resolve_inputs(self, param_names: list[str]) -> dict[str, Any]:
        """Resolve a node's declared parameter names from prior outputs, falling back
        to the original run inputs."""
        resolved: dict[str, Any] = {}
        for name in param_names:
            if name in self.completed:
                resolved[name] = self.completed[name]
            elif name in self.inputs:
                resolved[name] = self.inputs[name]
        return resolved

    def final_output(self) -> Any:
        if not self.order:
            return None
        return self.completed[self.order[-1]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "inputs": self.inputs,
            "completed": self.completed,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineContext:
        return cls(
            run_id=data["run_id"],
            inputs=data["inputs"],
            completed=data["completed"],
            order=data["order"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default)

    @classmethod
    def from_json(cls, raw: str) -> PipelineContext:
        return cls.from_dict(json.loads(raw))
