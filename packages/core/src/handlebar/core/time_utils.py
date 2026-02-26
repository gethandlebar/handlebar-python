"""Time and timezone helpers — mirrors packages/core/src/time.ts."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def now_to_time_parts(timestamp: datetime, timezone: str) -> tuple[str, str]:
    """Convert a datetime to (day_of_week, HH:MM) in the given IANA timezone.

    Returns a tuple of:
    - ``day``: three-letter lowercase weekday (e.g. ``"mon"``)
    - ``hhmm``: zero-padded 24-hour time string (e.g. ``"09:05"``)
    """
    tz = ZoneInfo(timezone)
    dt = timestamp.astimezone(tz)
    day = _DAYS[dt.weekday()]
    hhmm = dt.strftime("%H:%M")
    return day, hhmm


def hhmm_to_minutes(hhmm: str) -> int:
    """Convert a ``HH:MM`` string to the number of minutes since midnight."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)
