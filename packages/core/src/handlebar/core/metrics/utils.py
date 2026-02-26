"""Metric utility helpers — mirrors packages/core/src/metrics/utils.ts."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from .types import MetricInfo


def now_ms() -> float:
    """Return a high-resolution monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000


def approx_bytes(value: Any) -> int | None:
    """Estimate the byte size of a value when serialised to UTF-8 JSON."""
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (int, float, bool)):
        return len(str(value).encode("utf-8"))
    try:
        return len(json.dumps(value).encode("utf-8"))
    except Exception:
        return None


def approx_records(value: Any) -> int | None:
    """Estimate the number of records in a value."""
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("records", "items"):
            if isinstance(value.get(key), list):
                return len(value[key])
        if isinstance(value.get("count"), (int, float)):
            return int(value["count"])
    return None


def validate_metric_key(key: str) -> bool:
    """Return True if ``key`` matches the allowed pattern: [a-zA-Z_0-9]{1,64}."""
    import re

    return bool(re.fullmatch(r"[a-zA-Z_0-9]{1,64}", key))


def validate_metric(metric: MetricInfo) -> bool:
    """Return True if the metric value is a finite number."""
    return not (math.isnan(metric.value) or math.isinf(metric.value))
