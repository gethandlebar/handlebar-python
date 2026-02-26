"""Rule and RuleCondition types (used when defining governance rules)."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---------------------------------------------------------------------------
# Rule conditions — mirrors the discriminated union in governance-schema
# ---------------------------------------------------------------------------

# A recursive type alias: RuleCondition is defined at the bottom of this module.
# We use forward references throughout.


class ToolNameCondition(_Base):
    kind: Literal["toolName"]
    op: Literal["eq", "neq", "contains", "startsWith", "endsWith", "glob", "in"]
    value: str | list[str]


class ToolTagCondition(_Base):
    kind: Literal["toolTag"]
    op: Literal["eq", "neq", "in"]
    value: str | list[str]


class ToolArgCondition(_Base):
    kind: Literal["toolArg"]
    path: str
    op: Literal["eq", "neq", "contains", "startsWith", "endsWith", "glob", "in", "lt", "lte", "gt", "gte", "exists"]
    value: Any = None


class EnduserTagCondition(_Base):
    kind: Literal["enduserTag"]
    key: str
    op: Literal["eq", "neq", "in", "exists"]
    value: str | list[str] | None = None


class MaxCallsCondition(_Base):
    kind: Literal["maxCalls"]
    limit: int
    window_seconds: int | None = None


class SequenceCondition(_Base):
    kind: Literal["sequence"]
    tools: list[str]
    within_steps: int | None = None


class ExecutionTimeCondition(_Base):
    kind: Literal["executionTime"]
    op: Literal["lt", "lte", "gt", "gte"]
    value_ms: int


class TimeGateCondition(_Base):
    kind: Literal["timeGate"]
    timezone: str
    days: list[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]] | None = None
    start: str | None = None  # HH:MM
    end: str | None = None    # HH:MM


class RequireSubjectCondition(_Base):
    kind: Literal["requireSubject"]
    subject_type: str


class SignalCondition(_Base):
    kind: Literal["signal"]
    name: str
    args: dict[str, Any] | None = None
    op: Literal["eq", "neq", "lt", "lte", "gt", "gte", "in", "contains", "exists"]
    value: Any = None


class MetricWindowCondition(_Base):
    kind: Literal["metricWindow"]
    metric: str
    aggregate: Literal["sum", "avg", "count", "max", "min"]
    op: Literal["eq", "neq", "lt", "lte", "gt", "gte"]
    value: float
    window_seconds: int
    scope: Literal["agent", "agent_user"] = "agent"


class AndCondition(_Base):
    kind: Literal["and"]
    conditions: list["RuleCondition"]


class OrCondition(_Base):
    kind: Literal["or"]
    conditions: list["RuleCondition"]


class NotCondition(_Base):
    kind: Literal["not"]
    condition: "RuleCondition"


RuleCondition = Union[
    ToolNameCondition,
    ToolTagCondition,
    ToolArgCondition,
    EnduserTagCondition,
    MaxCallsCondition,
    SequenceCondition,
    ExecutionTimeCondition,
    TimeGateCondition,
    RequireSubjectCondition,
    SignalCondition,
    MetricWindowCondition,
    AndCondition,
    OrCondition,
    NotCondition,
]

# Rebuild models that reference the forward ref.
AndCondition.model_rebuild()
OrCondition.model_rebuild()
NotCondition.model_rebuild()


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


class _RuleSelector(_Base):
    phase: Literal["tool.before", "tool.after"]
    tool_name: str | None = None
    tool_tags: list[str] | None = None


class Rule(_Base):
    id: str
    name: str | None = None
    enabled: bool = True
    selector: _RuleSelector
    condition: RuleCondition
    effect: Literal["allow", "block", "hitl"]
    message: str | None = None
