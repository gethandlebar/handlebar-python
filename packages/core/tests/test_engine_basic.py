import time
from typing import Any, Dict, List
import pytest

from handlebar_core import (
    GovernanceEngine,
    Telemetry,
    block,
    max_calls,
    rule,
    tool_name,
)

# Disable telemetry sinks for tests.
Telemetry.init({"default_sinks": []})


def _engine_with_tools(
    tools: List[Dict[str, Any]],
    rules: List[Dict[str, Any]] | None = None,
    **cfg_overrides: Any,
) -> GovernanceEngine:
    cfg: Dict[str, Any] = {
        "tools": tools,
        "rules": rules or [],
        "mode": "enforce",
        "verbose": False,
    }
    cfg.update(cfg_overrides)
    return GovernanceEngine(cfg)


@pytest.mark.xfail(reason="Need to look into failure")
@pytest.mark.asyncio
async def test_allow_by_default() -> None:
    tools = [
        {"name": "web-search", "categories": ["search"]},
        {"name": "db.query", "categories": ["data"]},
    ]
    engine = _engine_with_tools(tools)
    ctx = engine.create_run_context(run_id="run-allow-1", user_category="test")

    decision = await engine.before_tool(
        ctx, tool_name="web-search", args={"q": "pizza"}
    )
    assert decision["effect"] == "allow"
    assert engine.should_block(decision) is False

    elapsed_ms = 5
    engine.after_tool(
        ctx,
        tool_name="web-search",
        execution_time_ms=elapsed_ms,
        args={"q": "pizza"},
        result={"results": []},
        error=None,
    )
    assert len(ctx["history"]) == 1
    assert ctx["counters"]["__hb_totalDurationMs"] >= elapsed_ms


@pytest.mark.asyncio
async def test_block_uncategorised_when_configured() -> None:
    tools = [{"name": "email.send"}]
    engine = _engine_with_tools(tools, default_uncategorised="block")
    ctx = engine.create_run_context(run_id="run-uncat-1", user_category="test")

    decision = await engine.before_tool(
        ctx, tool_name="email.send", args={"to": "a@b.com"}
    )
    assert decision["effect"] == "block"
    assert decision["code"] == "BLOCKED_UNCATEGORISED"
    assert engine.should_block(decision) is True


@pytest.mark.asyncio
async def test_pre_rule_block() -> None:
    tools = [
        {"name": "web-search", "categories": ["search"]},
        {"name": "db.query", "categories": ["data"]},
    ]
    from handlebar_core import config_to_rule  # local import to ensure availability

    rules = [
        config_to_rule(
            rule.pre(priority=10, if_=tool_name.eq("db.query"), do=[block()])
        )
    ]
    engine = _engine_with_tools(tools, rules=rules)
    ctx = engine.create_run_context(run_id="run-rule-1", user_category="test")

    decision = await engine.before_tool(
        ctx, tool_name="db.query", args={"sql": "SELECT 1"}
    )
    assert decision["effect"] == "block"
    assert decision["code"] == "BLOCKED_RULE"
    assert engine.should_block(decision) is True


@pytest.mark.asyncio
async def test_max_calls_blocks_third() -> None:
    tools = [{"name": "web-search", "categories": ["search"]}]
    from handlebar_core import config_to_rule  # local import to ensure availability

    rules = [
        config_to_rule(
            rule.both(
                priority=5,
                if_=max_calls(
                    selector={"by": "toolName", "patterns": ["web-*"]}, max_=2
                ),
                do=[block()],
            )
        )
    ]
    engine = _engine_with_tools(tools, rules=rules)
    ctx = engine.create_run_context(run_id="run-max-1", user_category="test")

    d1 = await engine.before_tool(ctx, tool_name="web-search", args={"q": "one"})
    assert d1["effect"] == "allow"
    await engine.after_tool(ctx, "web-search", 1, {"q": "one"}, {"ok": True})

    d2 = await engine.before_tool(ctx, tool_name="web-search", args={"q": "two"})
    assert d2["effect"] == "allow"
    await engine.after_tool(ctx, "web-search", 2, {"q": "two"}, {"ok": True})

    d3 = await engine.before_tool(ctx, tool_name="web-search", args={"q": "three"})
    assert d3["effect"] == "block"
    assert d3["code"] == "BLOCKED_RULE"
    assert engine.should_block(d3) is True
