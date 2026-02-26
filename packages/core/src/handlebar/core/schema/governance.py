"""Decision and verdict types returned from the Handlebar API."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class RunControl(StrEnum):
    CONTINUE = "CONTINUE"
    TERMINATE = "TERMINATE"


# ---------------------------------------------------------------------------
# Decision cause (discriminated union on `kind`)
# ---------------------------------------------------------------------------


class AllowCause(_Base):
    kind: Literal["ALLOW"]


class RuleViolationCause(_Base):
    kind: Literal["RULE_VIOLATION"]
    rule_id: str


class HitlPendingCause(_Base):
    kind: Literal["HITL_PENDING"]
    approval_id: str
    rule_id: str | None = None


class LockdownCause(_Base):
    kind: Literal["LOCKDOWN"]
    lockdown_id: str | None = None


DecisionCause = Annotated[
    AllowCause | RuleViolationCause | HitlPendingCause | LockdownCause,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Rule evaluation summary
# ---------------------------------------------------------------------------


class RuleEval(_Base):
    rule_id: str
    enabled: bool
    matched: bool
    violated: bool


# ---------------------------------------------------------------------------
# Decision — returned from POST /v1/runs/{id}/evaluate
# ---------------------------------------------------------------------------


class Decision(_Base):
    verdict: Verdict
    control: RunControl
    cause: DecisionCause
    message: str
    evaluated_rules: list[RuleEval] = Field(default_factory=list)
    final_rule_id: str | None = None
