# ==== Audit bus and sinks ====================================================
from handlebar_core.types import GovernanceDecision, JSONValue
from typing import Literal, Union, TypedDict, Dict, List, Optional

class AuditUserInfo(TypedDict, total=False):
    userId: str
    userCategory: str
    sessionId: str


class AuditOtelInfo(TypedDict, total=False):
    traceId: str
    spanId: str


class AuditSampleInfo(TypedDict, total=False):
    rate: float
    reason: str


class AuditRedactionInfo(TypedDict, total=False):
    level: Literal["none", "partial", "strict"]


class Counters(TypedDict, total=False):
    # string counter name -> integer value
    __any__: Union[int, str]


class ToolMetaEntry(TypedDict, total=False):
    # Free-form metadata about tool args/result.
    # Keep this loose; it just needs to be JSON-serialisable.
    name: Optional[str]
    summary: Optional[str]
    # arbitrary extra fields allowed


ToolMetaMap = Dict[str, JSONValue]


# ---- Envelope ---------------------------------------------------------------

class AuditEnvelope(TypedDict, total=False):
    schema: Literal["handlebar.audit.v1"]
    ts: str  # ISO8601 on the wire
    runId: str
    stepIndex: int
    decisionId: str
    user: AuditUserInfo
    otel: AuditOtelInfo
    sample: AuditSampleInfo
    redaction: AuditRedactionInfo
    agentId: str

# ---- run.started ------------------------------------------------------------

class RunStartedAgentInfo(TypedDict, total=False):
    framework: str
    version: str
    id: str
    name: str


class RunStartedModelInfo(TypedDict, total=False):
    provider: str
    name: str


class RunStartedPolicyInfo(TypedDict, total=False):
    version: str
    ruleCount: int
    sequenceId: str


class RunStartedRequestInfo(TypedDict, total=False):
    id: str
    traceparent: str


class AdapterInfo(TypedDict):
    name: str
    version: Optional[str]


class RunStartedData(TypedDict, total=False):
    agent: RunStartedAgentInfo
    model: RunStartedModelInfo
    adapter: AdapterInfo
    policy: RunStartedPolicyInfo
    request: RunStartedRequestInfo


class RunStartedEvent(AuditEnvelope):
    kind: Literal["run.started"]
    data: RunStartedData


# ---- tool.decision ----------------------------------------------------------

class ToolInfo(TypedDict, total=False):
    name: str
    categories: List[str]


class ToolDecisionData(GovernanceDecision, total=False):
    tool: ToolInfo
    counters: Counters
    latencyMs: int
    argsMeta: ToolMetaMap


class ToolDecisionEvent(AuditEnvelope):
    kind: Literal["tool.decision"]
    data: ToolDecisionData


# ---- tool.result ------------------------------------------------------------

class ToolErrorInfo(TypedDict, total=False):
    name: str
    message: str
    stack: str


class ToolResultData(TypedDict, total=False):
    tool: ToolInfo
    outcome: Literal["success", "error"]
    durationMs: int
    counters: Counters
    error: ToolErrorInfo
    resultMeta: ToolMetaMap


class ToolResultEvent(AuditEnvelope):
    kind: Literal["tool.result"]
    data: ToolResultData


# ---- run.ended --------------------------------------------------------------

class RunEndedData(TypedDict, total=False):
    status: Literal["ok", "error", "blocked"]
    totalSteps: int
    firstErrorDecisionId: str
    summary: str


class RunEndedEvent(AuditEnvelope):
    kind: Literal["run.ended"]
    data: RunEndedData


# ---- error ------------------------------------------------------------------

class ErrorData(TypedDict, total=False):
    scope: Literal["governance", "adapter", "transport", "agent"]
    message: str
    details: Dict[str, str]
    fatal: bool


class ErrorEvent(AuditEnvelope):
    kind: Literal["error"]
    data: ErrorData


# ---- Discriminated union ----------------------------------------------------

AuditEvent = Union[
    RunStartedEvent,
    ToolDecisionEvent,
    ToolResultEvent,
    RunEndedEvent,
    ErrorEvent,
]

AuditKind = Literal["run.started", "tool.decision", "tool.result", "run.ended", "error"]
