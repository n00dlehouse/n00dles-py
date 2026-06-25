from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .retry import RetryPolicy

if TYPE_CHECKING:
    from .agent import AgentNode


class Pipeline:
    """A sequential chain of AgentNodes, built with `>>` and wrapped by `pipeline()`.
    Users never construct this directly."""

    def __init__(
        self,
        nodes: list[AgentNode],
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

    def __rshift__(self, other: AgentNode | Pipeline) -> Pipeline:
        other_nodes = other.nodes if isinstance(other, Pipeline) else [other]
        return Pipeline(
            [*self.nodes, *other_nodes],
            name=self.name,
            retry=self.retry,
            timeout=self.timeout,
            on_error=self.on_error,
        )

    def topological_order(self) -> list[AgentNode]:
        """The execution order. Always a flat chain in this release — fan-out/fan-in
        (parallel()/branch()) is a planned follow-up, not yet implemented."""
        return list(self.nodes)

    def resolve_timeout(self, node: AgentNode) -> float:
        """Most-specific-wins: the node's own timeout, else this pipeline's, else the
        library default. Resolved per-call rather than mutating the node, since the
        same AgentNode can be reused across multiple pipelines with different settings."""
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


def pipeline(
    chain: AgentNode | Pipeline,
    *,
    name: str | None = None,
    retry: int | RetryPolicy = 3,
    timeout: float = 60.0,
    on_error: Callable | None = None,
) -> Pipeline:
    """Wrap a composed chain of agents (built with `>>`) into a single named, runnable
    pipeline. `retry`/`timeout` here are the fallback for every node in the chain that
    didn't set its own — an agent's own `@agent(retry=..., timeout=...)` always takes
    precedence, resolved per-run via Pipeline.resolve_retry()/resolve_timeout()."""
    from .agent import AgentNode

    nodes = chain.nodes if isinstance(chain, Pipeline) else [chain]
    if not nodes or not all(isinstance(n, AgentNode) for n in nodes):
        raise TypeError("pipeline() expects an AgentNode or a chain built with `>>`")

    retry_policy = retry if isinstance(retry, RetryPolicy) else RetryPolicy(max_attempts=retry)
    return Pipeline(nodes, name=name, retry=retry_policy, timeout=timeout, on_error=on_error)
