from __future__ import annotations

import datetime as _dt
import time as _time
import re

from .types import ISO8601

def slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = v.strip("-")
    return v

def milliseconds_since(initial_time: float) -> float:
    """
    Return elapsed time in milliseconds since `initial_time` (a time.perf_counter() snapshot).
    Rounded to the nearest microsecond (0.001 ms) to mirror the JS behavior.
    """
    return round((_time.perf_counter() - initial_time) * 1000.0, 3)


def now() -> ISO8601:
    """
    Current UTC time as an ISO 8601 string with a trailing 'Z'.
    """
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
