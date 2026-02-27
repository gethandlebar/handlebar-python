"""SinkBus — mirrors packages/core/src/sinks/bus.ts."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .types import Sink

if TYPE_CHECKING:
    from ..schema.events import AuditEvent

logger = logging.getLogger("handlebar")


class SinkBus:
    """Fans out audit events to multiple sinks.

    ``emit`` is synchronous and fire-and-forget; errors in individual sinks
    are caught and logged but do not propagate.
    """

    def __init__(self) -> None:
        self._sinks: list[Sink] = []
        self._closed = False

    def add(self, *sinks: Sink) -> None:
        self._sinks.extend(sinks)

    async def init(self) -> None:
        await asyncio.gather(*(s.init() for s in self._sinks), return_exceptions=True)

    def emit(self, agent_id: str, event: AuditEvent) -> None:
        """Fire-and-forget emit — safe to call from sync or async code."""
        if self._closed:
            return
        for sink in self._sinks:
            try:
                sink.write_batch(agent_id, [event])
            except Exception as exc:
                logger.error("[Handlebar] Sink error: %s", exc)

    async def drain(self) -> None:
        results = await asyncio.gather(*(s.drain() for s in self._sinks), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error("[Handlebar] Sink drain error: %s", r)

    async def close(self) -> None:
        self._closed = True
        results = await asyncio.gather(*(s.close() for s in self._sinks), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error("[Handlebar] Sink close error: %s", r)
