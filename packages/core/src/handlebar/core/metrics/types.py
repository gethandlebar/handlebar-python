"""Metric hook type definitions — mirrors packages/core/src/metrics/types.ts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..run import Run

AgentMetricHookPhase = Literal["tool.before", "tool.after"]


class MetricInfo:
    def __init__(self, value: float, unit: str | None = None) -> None:
        self.value = value
        self.unit = unit


class AgentMetricInputToolBefore:
    def __init__(self, tool_name: str, args: Any, run: Run) -> None:
        self.tool_name = tool_name
        self.args = args
        self.run = run


class AgentMetricInputToolAfter:
    def __init__(
        self,
        tool_name: str,
        args: Any,
        run: Run,
        result: Any = None,
        error: Any = None,
    ) -> None:
        self.tool_name = tool_name
        self.args = args
        self.run = run
        self.result = result
        self.error = error


AgentMetricHookContext = AgentMetricInputToolBefore | AgentMetricInputToolAfter

# Hook callable signature.
AgentMetricHookFn = Callable[
    [AgentMetricHookContext],
    "MetricInfo | None | Awaitable[MetricInfo | None]",
]


class AgentMetricHook:
    """A named hook that produces metric values at a specific lifecycle phase."""

    def __init__(
        self,
        key: str,
        phase: AgentMetricHookPhase,
        run: AgentMetricHookFn,
        when: Callable[[AgentMetricHookContext], bool] | None = None,
        timeout_ms: int | None = None,
        blocking: bool = True,
    ) -> None:
        self.key = key
        self.phase = phase
        self.run = run
        self.when = when
        self.timeout_ms = timeout_ms
        self.blocking = blocking
