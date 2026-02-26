"""Audit event types emitted by the Handlebar SDK."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---------------------------------------------------------------------------
# Shared envelope fields
# ---------------------------------------------------------------------------

# TODO: reallow None options when the zod data schemas have been made nullish

class _EventEnvelope(_Base):
    schema_version: str = Field("handlebar.audit.v1", alias="schema")
    ts: datetime
    run_id: str
    session_id: str = ""
    actor_external_id: str = ""
    step_index: int = 0


# ---------------------------------------------------------------------------
# run.started
# ---------------------------------------------------------------------------


class _RunStartedAgent(_Base):
    id: str = ""


class _RunStartedActor(_Base):
    external_id: str
    metadata: dict[str, str] = {}


class _RunStartedAdapter(_Base):
    name: str


class _RunStartedData(_Base):
    agent: _RunStartedAgent | None = None
    actor: _RunStartedActor | None = None
    adapter: _RunStartedAdapter | None = None


class RunStartedEvent(_EventEnvelope):
    kind: Literal["run.started"]
    data: _RunStartedData


# ---------------------------------------------------------------------------
# run.ended
# ---------------------------------------------------------------------------


class _RunEndedData(_Base):
    status: str
    total_steps: int


class RunEndedEvent(_EventEnvelope):
    kind: Literal["run.ended"]
    data: _RunEndedData


# ---------------------------------------------------------------------------
# tool.decision
# ---------------------------------------------------------------------------


class _ToolInfo(_Base):
    name: str
    categories: list[str] | None = None


class _ToolDecisionData(_Base):
    verdict: str
    control: str
    cause: dict[str, Any]
    message: str
    evaluated_rules: list[dict[str, Any]] = Field(default_factory=list)
    final_rule_id: str | None = None
    tool: _ToolInfo


class ToolDecisionEvent(_EventEnvelope):
    kind: Literal["tool.decision"]
    data: _ToolDecisionData


# ---------------------------------------------------------------------------
# tool.result
# ---------------------------------------------------------------------------


class _ToolResultError(_Base):
    name: str
    message: str


class _ToolResultData(_Base):
    tool: _ToolInfo
    outcome: Literal["success", "error"]
    duration_ms: float = 0.0
    # error: _ToolResultError | None = None


class ToolResultEvent(_EventEnvelope):
    kind: Literal["tool.result"]
    data: _ToolResultData


# ---------------------------------------------------------------------------
# llm.result
# ---------------------------------------------------------------------------


class _LlmModelInfo(_Base):
    name: str
    provider: str | None = None


class _LlmTokens(_Base):
    # stored as `in` in JSON — use alias
    tokens_in: int = Field(0, alias="in")
    tokens_out: int = Field(0, alias="out")

    model_config = ConfigDict(populate_by_name=True)


class _LlmResultData(_Base):
    model: _LlmModelInfo
    tokens: _LlmTokens
    message_count: int
    duration_ms: float | None = None


class LlmResultEvent(_EventEnvelope):
    kind: Literal["llm.result"]
    data: _LlmResultData


# ---------------------------------------------------------------------------
# message.raw.created
# ---------------------------------------------------------------------------


class _MessageRawCreatedData(_Base):
    message_id: str
    role: str
    kind: Literal["input", "output", "tool_call", "tool_result", "observation"]
    content: str
    content_truncated: bool = False


class MessageRawCreatedEvent(_EventEnvelope):
    kind: Literal["message.raw.created"]
    data: _MessageRawCreatedData


# ---------------------------------------------------------------------------
# Discriminated union of all event types
# ---------------------------------------------------------------------------

AuditEvent = Annotated[
    Union[
        RunStartedEvent,
        RunEndedEvent,
        ToolDecisionEvent,
        ToolResultEvent,
        LlmResultEvent,
        MessageRawCreatedEvent,
    ],
    Field(discriminator="kind"),
]
