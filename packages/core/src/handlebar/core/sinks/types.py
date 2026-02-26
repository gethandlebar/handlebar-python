"""Sink protocol — mirrors packages/core/src/sinks/types.ts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema.events import AuditEvent


class Sink(ABC):
    """A sink receives audit events and writes them somewhere.

    Only ``write_batch`` is mandatory; ``init``, ``drain``, and ``close``
    are optional no-ops by default (matching the JS interface where they are
    optional).
    """

    async def init(self) -> None:
        """Called once before the sink is first used."""

    @abstractmethod
    def write_batch(self, agent_id: str, events: list[AuditEvent]) -> None:
        """Write a batch of events. Must be non-blocking (e.g. queues internally)."""

    async def drain(self) -> None:
        """Flush pending writes without closing."""

    async def close(self) -> None:
        """Flush pending writes and release resources."""
