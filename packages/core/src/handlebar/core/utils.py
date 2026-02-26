"""Utility helpers — mirrors packages/core/src/utils.ts."""

from __future__ import annotations

import hashlib
import json
import os
import time as _time
from typing import Any

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def milliseconds_since(initial_time: float) -> float:
    """Return milliseconds elapsed since ``initial_time`` (from time.perf_counter())."""
    return round((_time.perf_counter() - initial_time) * 1000, 3)


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stable JSON (sorted keys, cycle-safe)
# ---------------------------------------------------------------------------


def stable_json(v: Any) -> str:
    """JSON-stringify with sorted keys; handles circular references."""
    seen: set[int] = set()

    def _norm(x: Any) -> Any:
        if isinstance(x, dict):
            obj_id = id(x)
            if obj_id in seen:
                return "[Circular]"
            seen.add(obj_id)
            result = {k: _norm(v) for k, v in sorted(x.items())}
            seen.discard(obj_id)
            return result
        if isinstance(x, (list, tuple)):
            obj_id = id(x)
            if obj_id in seen:
                return "[Circular]"
            seen.add(obj_id)
            result_list = [_norm(i) for i in x]
            seen.discard(obj_id)
            return result_list
        return x

    return json.dumps(_norm(v), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Dot-path accessor
# ---------------------------------------------------------------------------


def get_by_dot_path(obj: Any, path: str) -> Any:
    """Navigate a nested dict/object using a dot-separated path."""
    parts = [p for p in path.split(".") if p]
    cur = obj
    for p in parts:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return cur


# ---------------------------------------------------------------------------
# Slug generation (deterministic, based on CWD — mirrors JS mulberry32 impl)
# ---------------------------------------------------------------------------

_SLUG_PARTS = [
    "chainring",
    "spoke",
    "handlebar",
    "bell",
    "seatpost",
    "frame",
    "drivetrain",
    "cassette",
    "derailleur",
    "crankset",
    "saddle",
    "brake",
]


def _hash_to_seed(input_str: str) -> int:
    """SHA-256 hash → uint32 seed (little-endian first 4 bytes)."""
    digest = hashlib.sha256(input_str.encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little")


def _mulberry32(seed: int):
    """Seeded PRNG — direct port of the JS mulberry32 implementation."""
    t = seed & 0xFFFFFFFF

    def _u32(x: int) -> int:
        return x & 0xFFFFFFFF

    def _imul(a: int, b: int) -> int:
        """Simulate JavaScript Math.imul (C-style 32-bit multiply)."""
        return _u32(a * b)

    def rand() -> float:
        nonlocal t
        t = _u32(t + 0x6D2B79F5)
        t = _imul(_u32(t ^ (t >> 15)), _u32(t | 1))
        t = _u32(t ^ _u32(t + _imul(_u32(t ^ (t >> 7)), _u32(t | 61))))
        return _u32(t ^ (t >> 14)) / 4_294_967_296

    return rand


def generate_slug() -> str:
    """Generate a deterministic 4-word slug seeded by the current working directory."""
    wd = os.getcwd()
    seed = _hash_to_seed(wd)
    rand = _mulberry32(seed)

    words = []
    for _ in range(4):
        idx = int(rand() * len(_SLUG_PARTS))
        words.append(_SLUG_PARTS[idx])
    return "-".join(words)


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def _slugify(input_str: str) -> str:
    import re

    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", input_str.lower()))


def tool_to_insertable_tool(tool: Any) -> dict:
    """Convert a Tool descriptor to the shape the server expects."""
    d: dict = {
        "key": f"function:{_slugify(tool.name)}",
        "name": tool.name,
        "version": 1,
        "kind": "function",
    }
    if tool.description:
        d["description"] = tool.description
    if tool.tags:
        d["metadata"] = {"metadata": tool.tags}
    return d
