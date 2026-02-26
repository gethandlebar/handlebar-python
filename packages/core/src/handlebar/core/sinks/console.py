"""Console sink — mirrors packages/core/src/sinks/console.ts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .types import Sink

if TYPE_CHECKING:
    from ..schema.events import AuditEvent


class ConsoleSink(Sink):
    """Writes audit events to stdout."""

    def __init__(self, format: Literal["json", "pretty"] = "json") -> None:
        self._format = format

    def write_batch(self, agent_id: str, events: list[AuditEvent]) -> None:
        for event in events:
            if self._format == "json":
                # Pydantic serialise with camelCase aliases to match the wire format.
                print(event.model_dump_json(by_alias=True))
            else:
                step = getattr(event, "step_index", None)
                step_str = str(step) if step is not None else "-"
                print(f"[handlebar] {event.kind} run={event.run_id} step={step_str}")


def create_console_sink(format: Literal["json", "pretty"] = "json") -> Sink:
    """Factory — equivalent to JS ``createConsoleSink``."""
    return ConsoleSink(format=format)
