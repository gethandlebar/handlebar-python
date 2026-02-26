"""Subject registry — mirrors packages/core/src/subjects.ts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .run import Run


class SubjectRef:
    """A typed reference to an entity acted upon by a tool call."""

    def __init__(
        self,
        subject_type: str,
        value: str,
        id_system: str | None = None,
        role: str | None = None,
    ) -> None:
        self.subject_type = subject_type
        self.value = value
        self.id_system = id_system
        self.role = role

    def to_dict(self) -> dict:
        d: dict = {"subjectType": self.subject_type, "value": self.value}
        if self.id_system:
            d["idSystem"] = self.id_system
        if self.role:
            d["role"] = self.role
        return d


# A callable that extracts subject refs from a tool invocation context.
SubjectExtractor = Callable[
    [Any],  # ExtractArgs dict-like
    "list[SubjectRef] | Awaitable[list[SubjectRef]]",
]


class SubjectRegistry:
    """Registry mapping tool names to subject extractor functions."""

    def __init__(self) -> None:
        self._by_tool_name: dict[str, SubjectExtractor] = {}

    def register(self, tool_name: str, extractor: SubjectExtractor) -> None:
        self._by_tool_name[tool_name] = extractor

    def unregister(self, tool_name: str) -> None:
        self._by_tool_name.pop(tool_name, None)

    async def extract(
        self,
        tool: Any,
        tool_name: str,
        tool_args: Any,
        run: "Run",
    ) -> list[SubjectRef]:
        extractor = self._by_tool_name.get(tool_name)
        if extractor is None:
            return []

        try:
            import asyncio
            import inspect

            ctx = {"tool": tool, "toolName": tool_name, "toolArgs": tool_args, "run": run}
            result = extractor(ctx)
            if inspect.isawaitable(result):
                return await result
            return result  # type: ignore[return-value]
        except Exception:
            return []


def sanitise_subjects(subjects: list[SubjectRef]) -> list[SubjectRef]:
    """Clamp subjects list and truncate fields — mirrors JS sanitiseSubjects."""
    return [
        SubjectRef(
            subject_type=s.subject_type[:256],
            value=s.value[:256],
            id_system=s.id_system[:256] if s.id_system else None,
            role=s.role[:256] if s.role else None,
        )
        for s in subjects[:100]
    ]
