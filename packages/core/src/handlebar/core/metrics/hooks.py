"""Metric hook registry — mirrors packages/core/src/metrics/hooks.ts."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable

from .types import (
    AgentMetricHook,
    AgentMetricHookContext,
    AgentMetricHookPhase,
    MetricInfo,
)
from .utils import validate_metric_key


class AgentMetricHookRegistry:
    """Registry that collects and runs per-phase metric hooks."""

    def __init__(self) -> None:
        self._store: dict[AgentMetricHookPhase, dict[str, AgentMetricHook]] = {
            "tool.before": {},
            "tool.after": {},
        }

    def register_hook(self, hook: AgentMetricHook) -> None:
        if not validate_metric_key(hook.key):
            raise ValueError(f"Invalid metric key: {hook.key!r}")
        self._store[hook.phase][hook.key] = hook

    def unregister_hook(self, key: str, phase: AgentMetricHookPhase) -> None:
        self._store[phase].pop(key, None)

    async def run_phase(
        self,
        phase: AgentMetricHookPhase,
        ctx: AgentMetricHookContext,
        on_metric: Callable[[str, float, str | None], None],
    ) -> None:
        """Run all hooks for ``phase``, calling ``on_metric`` for each result."""
        hooks = self._store[phase]

        for hook_key, hook in hooks.items():
            if hook.when is not None and not hook.when(ctx):
                continue

            async def _run_hook(h: AgentMetricHook, c: AgentMetricHookContext) -> MetricInfo | None:
                raw = h.run(c)
                if inspect.isawaitable(raw):
                    result = await raw
                else:
                    result = raw
                return result

            coro = _run_hook(hook, ctx)

            if hook.timeout_ms is not None:
                try:
                    result = await asyncio.wait_for(coro, timeout=hook.timeout_ms / 1000)
                except TimeoutError:
                    result = None
            else:
                result = await coro

            if result is not None:
                if hook.blocking is False:
                    # Fire-and-forget: schedule emission without blocking.
                    asyncio.get_event_loop().call_soon(
                        lambda r=result, k=hook_key: on_metric(k, r.value, r.unit) if r else None
                    )
                else:
                    on_metric(hook_key, result.value, result.unit)
