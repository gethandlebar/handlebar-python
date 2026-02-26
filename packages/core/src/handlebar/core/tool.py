"""Tool wrapping and definition helpers — mirrors packages/core/src/tool.ts."""

from __future__ import annotations

from typing import Any


class ToolMeta:
    """Metadata overlay applied when wrapping a tool."""

    def __init__(
        self,
        tags: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        self.tags = tags
        self.description = description


def wrap_tool(tool: Any, meta: ToolMeta) -> Any:
    """Attach Handlebar metadata (tags, description) to any tool object.

    Returns the same object with ``tags`` and ``description`` set; does not
    alter the tool's callable interface.

    Example::

        search = wrap_tool(search_fn, ToolMeta(tags=["read-only"]))
        decision = await run.before_tool(search.name, args, search.tags)
    """
    tool.tags = meta.tags if meta.tags is not None else getattr(tool, "tags", []) or []
    tool.description = (
        meta.description
        if meta.description is not None
        else getattr(tool, "description", "") or ""
    )
    return tool


def define_tool(name: str, meta: ToolMeta | None = None) -> Any:
    """Build a tool descriptor inline without a framework wrapper.

    Example::

        read_file = define_tool("read_file", ToolMeta(
            description="Read a file from disk",
            tags=["filesystem", "read-only"],
        ))
    """
    from .types import Tool

    return Tool(
        name=name,
        description=meta.description if meta else "",
        tags=meta.tags if meta else [],
    )
