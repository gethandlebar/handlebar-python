from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, TypedDict, TypeVar, Union
from typing import NotRequired, Required

# Basic aliases
Id = str
ISO8601 = str
Glob = str

# JSON-safe value
JSONValue = Union[str, int, float, bool, None, Dict[str, "JSONValue"], List["JSONValue"]]

# Tools and calls/results

class Tool(TypedDict):
    name: str
    categories: NotRequired[List[str]]

ToolMeta = Tool

class ToolCall(TypedDict):
    tool: ToolMeta
    args: Any

class ToolResult(TypedDict):
    tool: ToolMeta
    args: Any
    result: Any
    error: NotRequired[Any]


# Governance schema (rules/actions/conditions)

RuleWhen = Literal["pre", "post", "both"]

class BlockAction(TypedDict):
    type: Literal["block"]

class AllowAction(TypedDict):
    type: Literal["allow"]

RuleAction = Union[BlockAction, AllowAction]

# Conditions

class ToolNameConditionSingle(TypedDict):
    kind: Literal["toolName"]
    op: Literal["eq", "neq", "contains", "startsWith", "endsWith", "glob"]
    value: Union[str, Glob]

class ToolNameConditionIn(TypedDict):
    kind: Literal["toolName"]
    op: Literal["in"]
    value: List[Union[str, Glob]]

ToolNameCondition = Union[ToolNameConditionSingle, ToolNameConditionIn]

class ToolTagHas(TypedDict):
    kind: Literal["toolTag"]
    op: Literal["has"]
    tag: str

class ToolTagAnyOf(TypedDict):
    kind: Literal["toolTag"]
    op: Literal["anyOf"]
    tags: List[str]

class ToolTagAllOf(TypedDict):
    kind: Literal["toolTag"]
    op: Literal["allOf"]
    tags: List[str]

ToolTagCondition = Union[ToolTagHas, ToolTagAnyOf, ToolTagAllOf]

ExecutionTimeScope = Literal["tool", "total"]

class ExecutionTimeCondition(TypedDict):
    kind: Literal["executionTime"]
    scope: ExecutionTimeScope
    op: Literal["gt", "gte", "lt", "lte", "eq", "neq"]
    ms: int

class SequenceCondition(TypedDict, total=False):
    kind: Literal["sequence"]
    mustHaveCalled: List[Glob]
    mustNotHaveCalled: List[Glob]

class MaxCallsSelector(TypedDict, total=False):
    by: Literal["toolName", "toolTag"]
    patterns: List[Glob]
    tags: List[str]

class MaxCallsCondition(TypedDict):
    kind: Literal["maxCalls"]
    selector: MaxCallsSelector
    max: int

class CustomFunctionCondition(TypedDict, total=False):
    kind: Literal["custom"]
    name: str
    args: JSONValue

class AndCondition(TypedDict):
    kind: Literal["and"]
    all: List["RuleCondition"]

class OrCondition(TypedDict):
    kind: Literal["or"]
    any: List["RuleCondition"]

NotCondition = TypedDict(
    "NotCondition",
    {
        "kind": Literal["not"],
        "not": "RuleCondition",
    },
)

RuleCondition = Union[
    ToolNameCondition,
    ToolTagCondition,
    ExecutionTimeCondition,
    SequenceCondition,
    MaxCallsCondition,
    CustomFunctionCondition,
    AndCondition,
    OrCondition,
    NotCondition,
    Dict[str, Any],
]

class RuleConfig(TypedDict):
    priority: int
    when: RuleWhen
    condition: RuleCondition
    actions: List[RuleAction]

class Rule(RuleConfig):
    id: Id
    policy_id: Id


# Governance decisions / applied actions

GovernanceEffect = Literal["allow", "block"]

# Note: broader action set for applied actions (matches audit/governance-actions in TS).
AppliedActionType = Literal["allow", "block", "notify", "log", "hitl"]

class AppliedAction(TypedDict):
    type: AppliedActionType
    ruleId: Id

GovernanceCode = Literal[
    "BLOCKED_UNCATEGORISED",
    "BLOCKED_RULE",
    "BLOCKED_CUSTOM",
    "ALLOWED",
    "NO_OP",
]

class GovernanceDecision(TypedDict, total=False):
    effect: GovernanceEffect
    code: GovernanceCode
    matchedRuleIds: List[Id]
    appliedActions: List[AppliedAction]
    reason: str


# Engine/SDK config and runtime context

T = TypeVar("T")
SyncOrAsync = Union[T, Awaitable[T]]

BeforeCheck = Callable[["RunContext", ToolCall], SyncOrAsync[Optional[GovernanceDecision]]]
AfterCheck = Callable[["RunContext", ToolResult], SyncOrAsync[None]]

class CustomCheck(TypedDict, total=False):
    before: BeforeCheck
    after: AfterCheck

class GovernanceConfig(TypedDict):
    tools: Required[List[ToolMeta]]
    rules: NotRequired[List[Rule]]
    defaultUncategorised: NotRequired[GovernanceEffect]
    checks: NotRequired[List[CustomCheck]]
    mode: NotRequired[Literal["monitor", "enforce"]]
    verbose: NotRequired[bool]
    agentId: NotRequired[str]

class RunContext(TypedDict):
    agentId: Optional[str]
    runId: str
    userCategory: str
    stepIndex: int
    otel: Dict[str, str]
    history: List[ToolResult]
    counters: Dict[str, int]
    state: Dict[str, Any]
    now: Callable[[], int]


__all__ = [
    "Id",
    "ISO8601",
    "Glob",
    "JSONValue",
    "Tool",
    "ToolMeta",
    "ToolCall",
    "ToolResult",
    "RuleWhen",
    "BlockAction",
    "AllowAction",
    "RuleAction",
    "ToolNameCondition",
    "ToolTagCondition",
    "ExecutionTimeScope",
    "ExecutionTimeCondition",
    "SequenceCondition",
    "MaxCallsSelector",
    "MaxCallsCondition",
    "CustomFunctionCondition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "RuleCondition",
    "RuleConfig",
    "Rule",
    "GovernanceEffect",
    "AppliedActionType",
    "AppliedAction",
    "GovernanceCode",
    "GovernanceDecision",
    "SyncOrAsync",
    "BeforeCheck",
    "AfterCheck",
    "CustomCheck",
    "GovernanceConfig",
    "RunContext",
]
