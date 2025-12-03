from __future__ import annotations

import uuid
from typing import Any, Dict, List, TypedDict, cast

from .types import (
    Glob,
    JSONValue,
    Rule,
    RuleAction,
    RuleCondition,
    RuleConfig,
    RuleWhen,
    BlockAction,
    AllowAction,
)


def and_(*all_: RuleCondition) -> RuleCondition:
    return {"kind": "and", "all": list(all_)}


def or_(*any_: RuleCondition) -> RuleCondition:
    return {"kind": "or", "any": list(any_)}


def not_(cond: RuleCondition) -> RuleCondition:
    return {"kind": "not", "not": cond}


class _ToolName:
    def eq(self, value: str | Glob) -> RuleCondition:
        return {"kind": "toolName", "op": "eq", "value": value}

    def neq(self, value: str | Glob) -> RuleCondition:
        return {"kind": "toolName", "op": "neq", "value": value}

    def glob(self, pattern: Glob) -> RuleCondition:
        return {"kind": "toolName", "op": "glob", "value": pattern}

    def in_(self, values: List[str | Glob]) -> RuleCondition:
        return {"kind": "toolName", "op": "in", "value": values}

    def starts_with(self, value: str) -> RuleCondition:
        return {"kind": "toolName", "op": "startsWith", "value": value}

    def ends_with(self, value: str) -> RuleCondition:
        return {"kind": "toolName", "op": "endsWith", "value": value}

    def contains(self, value: str) -> RuleCondition:
        return {"kind": "toolName", "op": "contains", "value": value}


tool_name = _ToolName()


class _ToolTag:
    def has(self, tag: str) -> RuleCondition:
        return {"kind": "toolTag", "op": "has", "tag": tag}

    def any_of(self, tags: List[str]) -> RuleCondition:
        return {"kind": "toolTag", "op": "anyOf", "tags": tags}

    def all_of(self, tags: List[str]) -> RuleCondition:
        return {"kind": "toolTag", "op": "allOf", "tags": tags}


tool_tag = _ToolTag()


class _ExecTime:
    def gt(self, scope: str, ms: int) -> RuleCondition:
        return {"kind": "executionTime", "scope": scope, "op": "gt", "ms": ms}

    def gte(self, scope: str, ms: int) -> RuleCondition:
        return {"kind": "executionTime", "scope": scope, "op": "gte", "ms": ms}

    def lt(self, scope: str, ms: int) -> RuleCondition:
        return {"kind": "executionTime", "scope": scope, "op": "lt", "ms": ms}

    def lte(self, scope: str, ms: int) -> RuleCondition:
        return {"kind": "executionTime", "scope": scope, "op": "lte", "ms": ms}

    def eq(self, scope: str, ms: int) -> RuleCondition:
        return {"kind": "executionTime", "scope": scope, "op": "eq", "ms": ms}

    def neq(self, scope: str, ms: int) -> RuleCondition:
        return {"kind": "executionTime", "scope": scope, "op": "neq", "ms": ms}


exec_time = _ExecTime()


def sequence(
    *,
    must_have_called: List[Glob] | None = None,
    must_not_have_called: List[Glob] | None = None,
) -> RuleCondition:
    cond: Dict[str, Any] = {"kind": "sequence"}
    if must_have_called:
        cond["mustHaveCalled"] = list(must_have_called)
    if must_not_have_called:
        cond["mustNotHaveCalled"] = list(must_not_have_called)
    return cond  # type: ignore[return-value]


def max_calls(
    *,
    selector: Dict[str, Any],
    max_: int,
) -> RuleCondition:
    return {"kind": "maxCalls", "selector": selector, "max": max_}


def custom(name: str, args: JSONValue | None = None) -> RuleCondition:
    cond: Dict[str, Any] = {"kind": "custom", "name": name}
    if args is not None:
        cond["args"] = args
    return cond  # type: ignore[return-value]


def block() -> RuleAction:
    return cast(BlockAction, {"type": "block"})


def allow() -> RuleAction:
    return cast(AllowAction, {"type": "allow"})


class _RuleBuilder:
    def __call__(self, when: RuleWhen, *, priority: int, if_: RuleCondition, do: List[RuleAction]) -> RuleConfig:
        return {"priority": priority, "when": when, "condition": if_, "actions": list(do)}

    def pre(self, *, priority: int, if_: RuleCondition, do: List[RuleAction]) -> RuleConfig:
        return self("pre", priority=priority, if_=if_, do=do)

    def post(self, *, priority: int, if_: RuleCondition, do: List[RuleAction]) -> RuleConfig:
        return self("post", priority=priority, if_=if_, do=do)

    def both(self, *, priority: int, if_: RuleCondition, do: List[RuleAction]) -> RuleConfig:
        return self("both", priority=priority, if_=if_, do=do)


rule = _RuleBuilder()


def _uuid7_str() -> str:
    """
    uuid7 not introduced until 3.14.
    If present, use it, otherwise fallback to uuid4.
    """
    gen = getattr(uuid, "uuid7", None)
    if callable(gen):
        return str(gen())
    return str(uuid.uuid4())


def config_to_rule(config: RuleConfig) -> Rule:
    rid = _uuid7_str()
    pid = _uuid7_str()
    return {"id": rid, "policy_id": pid, **config}  # type: ignore[return-value]


__all__ = [
    "and_",
    "or_",
    "not_",
    "tool_name",
    "tool_tag",
    "exec_time",
    "sequence",
    "max_calls",
    "custom",
    "block",
    "allow",
    "rule",
    "config_to_rule",
]
