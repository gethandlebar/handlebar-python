from __future__ import annotations
from handlebar_core.types import RunContext

import inspect
import logging
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, cast, Callable

from .telemetry import emit, get_run_context, inc_step
from .types import (
    AppliedAction,
    CustomCheck,
    ExecutionTimeCondition,
    GovernanceConfig,
    GovernanceDecision,
    GovernanceEffect,
    MaxCallsCondition,
    Rule,
    RuleCondition,
    Tool,
    ToolCall,
    ToolMeta,
    ToolResult,
    ToolNameCondition,
    ToolTagCondition,
    SequenceCondition,
    CustomFunctionCondition,
    AndCondition,
    OrCondition,
    NotCondition,
    ToolTagHas,
    ToolTagAnyOf,
    ToolTagAllOf,
)
from .utils import milliseconds_since

TOTAL_DURATION_COUNTER = "__hb_totalDurationMs"

logger = logging.getLogger("handlebar. " + __name__)

@DeprecationWarning
class GovernanceLog(TypedDict):
    tool: ToolCall
    decision: GovernanceDecision
    when: Literal["before", "after"]


def _match_glob(value: str, pattern: str) -> bool:
    esc = re.escape(pattern).replace(r"\*", ".*")
    return re.fullmatch(esc, value, flags=re.IGNORECASE) is not None


class GovernanceEngine:
    def __init__(self, cfg: GovernanceConfig, bus: Any | None = None) -> None:
        tools_list = cfg.get("tools", [])
        self.tools: Dict[str, ToolMeta] = {t["name"]: t for t in tools_list}
        self.rules: List[Rule] = list(cfg.get("rules", []))  # type: ignore[arg-type]
        self.default_uncategorised: GovernanceEffect = cast(
            GovernanceEffect,
            cfg.get("defaultUncategorised", cfg.get("default_uncategorised", "allow")),
        )
        self.checks: List[CustomCheck] = list(cfg.get("checks", []))  # type: ignore[arg-type]
        self.mode: Literal["monitor", "enforce"] = cfg.get("mode", "enforce")
        self.verbose: bool = bool(cfg.get("verbose", False))

        # Should be uuidv7 with "agnt-" prefix
        self.agent_id: Optional[str] = cfg.get("agentId")

        # Deprecated: audit sink takes precedence.
        self.governance_log: List[GovernanceLog] = []

        # 'bus' is accepted for signature parity; emission is handled via audit Telemetry singleton.
        self._bus = bus

    def create_run_context(
        self,
        run_id: str,
        user_category: str,
        now: Optional[Callable] = None,
        initial_counters: Optional[Dict[str, int]] = None,
    ) -> RunContext:
        return {
            "agentId": self.agent_id,
            "runId": run_id,
            "userCategory": user_category,
            "stepIndex": 0,
            "history": [],
            "otel": {},
            "counters": {**(initial_counters or {}), TOTAL_DURATION_COUNTER: 0},
            "state": {},
            "now": now or (lambda: int(time.time() * 1000)),
        }

    def get_tool(self, name: str) -> ToolMeta:
        t = self.tools.get(name)
        if not t:
            raise KeyError(f'Unknown tool "{name}"')
        return t

    def _decide_by_rules(
        self,
        phase: Literal["pre", "post"],
        ctx: RunContext,
        call: ToolCall,
        execution_time_ms: Optional[int],
    ) -> GovernanceDecision:
        if (not call["tool"].get("categories") or len(call["tool"].get("categories", [])) == 0) and self.default_uncategorised == "block":
            return {
                "effect": "block",
                "code": "BLOCKED_UNCATEGORISED",
                "matchedRuleIds": [],
                "appliedActions": [],
                "reason": f'Tool "{call["tool"]["name"]}" has no categories',
            }

        applicable = [r for r in self.rules if r["when"] in (phase, "both")]
        ordered = sorted(applicable, key=lambda r: r["priority"])

        decision_effect_code: Optional[Tuple[Literal["allow", "block"], str]] = None
        applied_rules: List[AppliedAction] = []
        matching_rules: List[str] = []

        for rule in ordered:
            matches = self._eval_condition(
                rule["condition"],
                {
                    "phase": phase,
                    "ctx": ctx,
                    "call": call,
                    "executionTimeMS": execution_time_ms,
                },
            )

            if not matches:
                continue

            matching_rules.append(rule["id"])

            if not rule["actions"]:
                continue

            for action in rule["actions"]:
                applied_rules.append({"ruleId": rule["id"], "type": action["type"]})  # type: ignore[typeddict-item]

                if action["type"] == "block":
                    return {
                        "effect": "block",
                        "code": "BLOCKED_RULE",
                        "appliedActions": applied_rules,
                        "matchedRuleIds": [ar["ruleId"] for ar in applied_rules],
                    }
                elif action["type"] == "allow" and (decision_effect_code is None or decision_effect_code[0] != "block"):
                    decision_effect_code = ("allow", "ALLOWED")

        final_decision: GovernanceDecision = {
            "matchedRuleIds": matching_rules,
            "appliedActions": applied_rules,
            **({"effect": decision_effect_code[0], "code": decision_effect_code[1]} if decision_effect_code else {"effect": "allow", "code": "ALLOWED"}),
        }
        return final_decision

    def _eval_condition(
        self,
        cond: RuleCondition,
        args: Dict[str, Any],
    ) -> bool:
        kind = cond["kind"]

        if kind == "toolName":
            return self._eval_tool_name(cast(ToolNameCondition, cond), args["call"]["tool"]["name"])

        if kind == "toolTag":
            return self._eval_tool_tag(cast(ToolTagCondition, cond), args["call"]["tool"].get("categories", []) or [])

        if kind == "executionTime":
            if args["phase"] != "post":
                return False
            return self._eval_execution_time(cast(ExecutionTimeCondition, cond), args["executionTimeMS"], args["ctx"])

        if kind == "sequence":
            return self._eval_sequence(cast(SequenceCondition, cond), args["ctx"]["history"], args["call"]["tool"]["name"])

        if kind == "maxCalls":
            return self._eval_max_calls(cast(MaxCallsCondition, cond), args["ctx"]["history"])

        if kind == "custom":
            return self._eval_custom(cast(CustomFunctionCondition, cond), args["ctx"], args["call"])

        if kind == "and":
            all_conds = cast(AndCondition, cond).get("all", [])
            if not all_conds:
                return True
            for child in all_conds:
                if not self._eval_condition(child, args):
                    return False
            return True

        if kind == "or":
            any_conds = cast(OrCondition, cond).get("any", [])
            if not any_conds:
                return False
            for child in any_conds:
                if self._eval_condition(child, args):
                    return True
            return False

        if kind == "not":
            return not self._eval_condition(cast(NotCondition, cond)["not"], args)

        return False

    def _eval_tool_name(self, cond: ToolNameCondition, tool_name: str) -> bool:
        name = tool_name.lower()
        op = cond["op"]
        value = cond["value"]

        if op == "eq":
            return name == str(value).lower()
        if op == "neq":
            return name != str(value).lower()
        if op == "contains":
            return str(value).lower() in name
        if op == "startsWith":
            return name.startswith(str(value).lower())
        if op == "endsWith":
            return name.endswith(str(value).lower())
        if op == "glob":
            return _match_glob(name, cast(str, value))
        if op == "in":
            return any(_match_glob(name, str(v)) for v in cast(List[str], value))
        return False

    def _eval_tool_tag(self, cond: ToolTagCondition, tags: List[str]) -> bool:
        lower = [t.lower() for t in tags]
        op = cond["op"]

        if op == "has":
            return cast(ToolTagHas, cond)["tag"].lower() in lower
        if op == "anyOf":
            return any(t.lower() in lower for t in cast(ToolTagAnyOf, cond)["tags"])
        if op == "allOf":
            return all(t.lower() in lower for t in cast(ToolTagAllOf, cond)["tags"])
        return False

    def _eval_execution_time(
        self,
        cond: ExecutionTimeCondition,
        execution_time_ms: Optional[int],
        ctx: RunContext,
    ) -> bool:
        if execution_time_ms is None:
            return False

        total_ms = ctx["counters"].get(TOTAL_DURATION_COUNTER, 0)
        value_ms = execution_time_ms if cond["scope"] == "tool" else total_ms

        op = cond["op"]
        ms = cond["ms"]

        if op == "gt":
            return value_ms > ms
        if op == "gte":
            return value_ms >= ms
        if op == "lt":
            return value_ms < ms
        if op == "lte":
            return value_ms <= ms
        if op == "eq":
            return value_ms == ms
        if op == "neq":
            return value_ms != ms
        return False

    def _eval_sequence(
        self,
        cond: SequenceCondition,
        history: List[ToolResult],
        current_tool_name: str,
    ) -> bool:
        names = [h["tool"]["name"] for h in history]

        must_have = cond.get("mustHaveCalled") or []
        if must_have:
            for pattern in must_have:
                found = any(_match_glob(n, pattern) for n in names)
                if not found:
                    return False

        must_not = cond.get("mustNotHaveCalled") or []
        if must_not:
            for pattern in must_not:
                found = any(_match_glob(n, pattern) for n in names)
                if found:
                    return False

        return True

    def _eval_max_calls(self, cond: MaxCallsCondition, history: List[ToolResult]) -> bool:
        count = 0
        selector = cond["selector"]

        if selector["by"] == "toolName":
            patterns = selector["patterns"]
            for h in history:
                if any(_match_glob(h["tool"]["name"], p) for p in patterns):
                    count += 1
        else:
            tags = [t.lower() for t in selector["tags"]]
            for h in history:
                htags = [t.lower() for t in (h["tool"].get("categories") or [])]
                if any(tag in htags for tag in tags):
                    count += 1

        return count >= cond["max"]

    def _eval_custom(self, cond: CustomFunctionCondition, ctx: RunContext, call: ToolCall) -> bool:
        # For now, no central registry; user can still use CustomCheck.before
        return False

    async def _run_before_check(
        self, check: CustomCheck, ctx: RunContext, call: ToolCall
    ) -> Optional[GovernanceDecision]:
        before_fn = check.get("before")
        if not before_fn:
            return None
        res = before_fn(ctx, call)
        if inspect.isawaitable(res):
            res = await res
        return res

    async def _run_after_check(
        self, check: CustomCheck, ctx: RunContext, result: ToolResult
    ) -> None:
        after_fn = check.get("after")
        if not after_fn:
            return
        res = after_fn(ctx, result)
        if inspect.isawaitable(res):
            await res

    async def before_tool(self, ctx: RunContext, tool_name: str, args: Any) -> GovernanceDecision:
        run_ctx = get_run_context()
        local_step = (run_ctx or {}).get("stepIndex", 0)
        t0 = time.perf_counter()

        tool = self.get_tool(tool_name)
        call: ToolCall = {"tool": tool, "args": args}

        for check in self.checks:
            d = await self._run_before_check(check, ctx, call)
            if d and d.get("effect") == "block":
                emit(
                    "tool.decision",
                    {
                        "tool": {"name": tool_name, "categories": tool.get("categories")},
                        "effect": d["effect"],
                        "code": d.get("code"),
                        "reason": d.get("reason"),
                        "matchedRuleIds": d.get("matchedRuleIds"),
                        "appliedActions": d.get("appliedActions"),
                        "counters": {**ctx.get("counters", {})},
                        "latencyMs": milliseconds_since(t0),
                    },
                    {"stepIndex": local_step, "agentId": self.agent_id},
                )
                return self._finalise_decision(
                    ctx,
                    call,
                    {
                        **d,
                        "code": d.get("code", "BLOCKED_CUSTOM"),  # type: ignore[arg-type]
                    },
                )

        decision = self._decide_by_rules("pre", ctx, call, None)
        final_decision = self._finalise_decision(ctx, call, decision)

        logger.debug("[Handlebar] %s %s", tool_name, decision.get("code"))
        emit(
            "tool.decision",
            {
                "tool": {"name": tool_name, "categories": tool.get("categories", [])},
                "effect": decision["effect"],
                "code": decision.get("code"),
                "reason": decision.get("reason", ""),
                "matchedRuleIds": decision.get("matchedRuleIds", []),
                "appliedActions": decision.get("appliedActions", []),
                "counters": {**ctx.get("counters", {})},
                "latencyMs": milliseconds_since(t0),
            },
            {"stepIndex": local_step, "decisionId": "", "agentId": self.agent_id},
        )

        return final_decision

    def _finalise_decision(self, ctx: RunContext, call: ToolCall, decision: GovernanceDecision) -> GovernanceDecision:
        if self.verbose:
            tag = "✅" if decision["effect"] == "allow" else "⛔"
            last_rule_id = None
            actions = decision.get("appliedActions") or []
            if actions:
                last_rule_id = actions[-1].get("ruleId")
            reason = decision.get("reason")

        self.governance_log.append({"tool": call, "decision": decision, "when": "before"})
        return decision

    async def after_tool(
        self,
        ctx: RunContext,
        tool_name: str,
        execution_time_ms: Optional[int],
        args: Any,
        result: Any,
        error: Any = None,
    ) -> None:
        run_ctx = get_run_context()
        local_step = (run_ctx or {}).get("stepIndex", 0)
        decision_id = (run_ctx or {}).get("decisionId")

        tool = self.get_tool(tool_name)

        tr: ToolResult = {
            "tool": tool,
            "args": args,
            "result": result,
            **({"error": error} if error is not None else {}),
        }

        ctx["history"].append(tr)
        ctx["stepIndex"] = int(ctx.get("stepIndex", 0)) + 1

        if execution_time_ms is not None:
            ctx["counters"][TOTAL_DURATION_COUNTER] = int(ctx["counters"].get(TOTAL_DURATION_COUNTER, 0)) + int(execution_time_ms)

        for check in self.checks:
            await self._run_after_check(check, ctx, tr)

        post_decision = self._decide_by_rules("post", ctx, {"tool": tool, "args": args}, execution_time_ms)
        if post_decision["effect"] == "block" and self.verbose:
            logger.warning('[Handlebar] ⛔ post-tool rule would block "%s" (not enforced yet).', tool_name)


        error_exc = error if isinstance(error, BaseException) else None
        emit_data = {
            "tool": {"name": tool_name, "categories": tool.get("categories", [])},
            "outcome": "error" if error is not None else "success",
            **({"durationMs": execution_time_ms} if execution_time_ms is not None else {}),
            "counters": {**ctx.get("counters", {})},
        }

        if error_exc:
            emit_data["error"] = {"name": type(error_exc).__name__, "message": str(error_exc)}

        emit(
            "tool.result",
            emit_data,
            {"stepIndex": local_step, "decisionId": "", "agentId": self.agent_id},
        )

        inc_step()

    def should_block(self, decision: GovernanceDecision) -> bool:
        return self.mode == "enforce" and decision["effect"] == "block"


__all__ = ["GovernanceEngine", "GovernanceLog", "TOTAL_DURATION_COUNTER"]
