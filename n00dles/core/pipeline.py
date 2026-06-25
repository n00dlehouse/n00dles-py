from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Union

from .retry import RetryPolicy

if TYPE_CHECKING:
    from .agent import AgentNode

# A node in a Pipeline's chain — either a single agent call, a fan-out group, or a
# conditional router. Pipeline.nodes can hold a mix of all three.
PipelineNode = Union["AgentNode", "ParallelGroup", "BranchAgent"]


class Pipeline:
    """A chain of agents/parallel groups/branches, built with `>>` and wrapped by
    `pipeline()`. Users never construct this directly."""

    def __init__(
        self,
        nodes: list[PipelineNode],
        *,
        name: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        on_error: Callable | None = None,
    ):
        self.nodes = nodes
        self.name = name or "_".join(n.name for n in nodes)
        self.retry = retry
        self.timeout = timeout
        self.on_error = on_error

    def __rshift__(self, other: PipelineNode | Pipeline) -> Pipeline:
        other_nodes = other.nodes if isinstance(other, Pipeline) else [other]
        return Pipeline(
            [*self.nodes, *other_nodes],
            name=self.name,
            retry=self.retry,
            timeout=self.timeout,
            on_error=self.on_error,
        )

    def topological_order(self) -> list[PipelineNode]:
        """The execution order — a flat chain where each step is a single agent, a
        ParallelGroup (fan-out), or a BranchAgent (conditional routing)."""
        return list(self.nodes)

    def resolve_timeout(self, node: AgentNode) -> float:
        """Most-specific-wins: the node's own timeout, else this pipeline's, else the
        library default. Resolved per-call rather than mutating the node, since the
        same AgentNode can be reused across multiple pipelines with different settings.
        Only meaningful for individual AgentNodes — ParallelGroup/BranchAgent delegate
        to their member agents' own resolution."""
        if node.timeout is not None:
            return node.timeout
        if self.timeout is not None:
            return self.timeout
        return node.DEFAULT_TIMEOUT

    def resolve_retry(self, node: AgentNode) -> RetryPolicy:
        if node.retry is not None:
            return node.retry
        if self.retry is not None:
            return self.retry
        return node.DEFAULT_RETRY

    def __repr__(self) -> str:
        return f"Pipeline({self.name!r}, nodes={[n.name for n in self.nodes]})"


class ParallelGroup:
    """Built by `a | b` or `parallel(a, b, ...)`. Runs every member agent concurrently
    against the same resolved inputs; the next step in the chain receives a dict keyed
    by each member's function name — no manual merging required."""

    def __init__(self, nodes: list[AgentNode], max_concurrency: int | None = None):
        if len(nodes) < 2:
            raise ValueError("parallel() / `|` requires at least 2 agents")
        self.nodes = nodes
        self.max_concurrency = max_concurrency
        # Synthetic name for resume-tracking — there's no single function behind a
        # group, so the group itself is checkpointed under a name derived from its
        # members (in addition to each member being checkpointed under its own name).
        self.name = "parallel[" + ",".join(n.name for n in nodes) + "]"

    def __rshift__(self, other: PipelineNode | Pipeline) -> Pipeline:
        other_nodes = other.nodes if isinstance(other, Pipeline) else [other]
        return Pipeline([self, *other_nodes])

    def __or__(self, other: AgentNode | ParallelGroup) -> ParallelGroup:
        other_nodes = other.nodes if isinstance(other, ParallelGroup) else [other]
        return ParallelGroup([*self.nodes, *other_nodes], max_concurrency=self.max_concurrency)

    def __repr__(self) -> str:
        return f"ParallelGroup({[n.name for n in self.nodes]})"


class BranchAgent:
    """Built by `branch(default=..., **routes)`. Routes to exactly one member agent
    based on a key read off the previous step's output: the string itself if it's a
    plain str, the `category` field if it's a dict or a Pydantic model."""

    def __init__(self, routes: dict[str, AgentNode], default: AgentNode | None = None):
        if not routes:
            raise ValueError("branch() requires at least one route")
        self.routes = routes
        self.default = default
        self.name = "branch[" + ",".join(routes.keys()) + "]"

    def __rshift__(self, other: PipelineNode | Pipeline) -> Pipeline:
        other_nodes = other.nodes if isinstance(other, Pipeline) else [other]
        return Pipeline([self, *other_nodes])

    def __repr__(self) -> str:
        default_name = self.default and self.default.name
        return f"BranchAgent({list(self.routes.keys())}, default={default_name})"


def pipeline(
    chain: PipelineNode | Pipeline,
    *,
    name: str | None = None,
    retry: int | RetryPolicy = 3,
    timeout: float = 60.0,
    on_error: Callable | None = None,
) -> Pipeline:
    """Wrap a composed chain (built with `>>`, optionally including `|`/`parallel()`
    fan-out or `branch()` routing) into a single named, runnable pipeline. `retry`/
    `timeout` here are the fallback for every agent in the chain that didn't set its
    own — an agent's own `@agent(retry=..., timeout=...)` always takes precedence,
    resolved per-run via Pipeline.resolve_retry()/resolve_timeout()."""
    from .agent import AgentNode

    nodes = chain.nodes if isinstance(chain, Pipeline) else [chain]
    if not nodes or not all(isinstance(n, (AgentNode, ParallelGroup, BranchAgent)) for n in nodes):
        raise TypeError(
            "pipeline() expects an AgentNode, or a chain built with `>>`/`|`/branch()"
        )

    retry_policy = retry if isinstance(retry, RetryPolicy) else RetryPolicy(max_attempts=retry)
    return Pipeline(nodes, name=name, retry=retry_policy, timeout=timeout, on_error=on_error)


def parallel(*agents: AgentNode, max_concurrency: int | None = None) -> ParallelGroup:
    """Run every agent concurrently against the same resolved inputs. The next step
    receives a dict keyed by each agent's function name."""
    return ParallelGroup(list(agents), max_concurrency=max_concurrency)


def branch(default: AgentNode | None = None, **routes: AgentNode) -> BranchAgent:
    """Route to exactly one agent based on a key read off the previous step's output.
    `default` runs when the key matches none of `routes`; with no default, an
    unmatched key raises BranchError."""
    return BranchAgent(routes=routes, default=default)
