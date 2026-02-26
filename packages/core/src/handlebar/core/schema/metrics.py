"""Metric kind definitions."""

from __future__ import annotations

from enum import Enum


class InbuiltAgentMetricKind(str, Enum):
    BYTES_IN = "bytes_in"
    BYTES_OUT = "bytes_out"
    RECORDS_OUT = "records_out"
    DURATION_MS = "duration_ms"
    LLM_TOKENS_IN = "llm_tokens_in"
    LLM_TOKENS_OUT = "llm_tokens_out"
