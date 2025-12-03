import time
from typing import Any, Dict, List

import pytest

from handlebar_core import (
    GovernanceEngine,
    Telemetry,
    config_to_rule,
    rule,
    tool_name,
    tool_tag,
    sequence,
    block,
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

@pytest.mark.asyncio
async def test_tool_tag_has_blocks_danger_tag() -> None:
    tools = [
        {"name": "web.search", "categories": ["search", "internet"]},
        {"name": "file.write", "categories": ["fs", "danger"]},
    ]

    rules = [
        config_to_rule(
            rule.pre(priority=1, if_=tool_tag.has("danger"), do=[block()])
        )
    ]

    engine = _engine_with_tools(tools, rules=rules)
    ctx = engine.create_run_context(run_id="run-tag-1", user_category="test")

    d1 = await engine.before_tool(ctx, tool_name="web.search", args={})
    assert d1["effect"] == "allow"

    d2 = await engine.before_tool(ctx, tool_name="file.write", args={})
    assert d2["effect"] == "block"
    assert d2["code"] == "BLOCKED_RULE"

@pytest.mark.asyncio
async def test_tool_tag_any_of_and_all_of() -> None:
    tools = [
        {"name": "web.search", "categories": ["search", "internet"]},
        {"name": "db.query", "categories": ["data"]},
    ]

    rules_any = [
        config_to_rule(
            rule.pre(priority=1, if_=tool_tag.any_of(["internet", "compute"]), do=[block()])
        )
    ]
    engine_any = _engine_with_tools(tools, rules=rules_any)
    ctx_any = engine_any.create_run_context(run_id="run-tag-2a", user_category="test")

    d_any_web = await engine_any.before_tool(ctx_any, tool_name="web.search", args={})
    assert d_any_web["effect"] == "block"

    d_any_db = await engine_any.before_tool(ctx_any, tool_name="db.query", args={})
    assert d_any_db["effect"] == "allow"

    rules_all = [
        config_to_rule(
            rule.pre(priority=1, if_=tool_tag.all_of(["search", "internet"]), do=[block()])
        )
    ]
    engine_all = _engine_with_tools(tools, rules=rules_all)
    ctx_all = engine_all.create_run_context(run_id="run-tag-2b", user_category="test")

    d_all_web = await engine_all.before_tool(ctx_all, tool_name="web.search", args={})
    assert d_all_web["effect"] == "block"

    d_all_db = await engine_all.before_tool(ctx_all, tool_name="db.query", args={})
    assert d_all_db["effect"] == "allow"

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cond,tool,should_block",
    [
        (tool_name.eq("db.query"), "db.query", True),
        (tool_name.neq("db.query"), "db.query", False),
        (tool_name.contains("QUERY"), "db.query", True),
        (tool_name.starts_with("db."), "db.query", True),
        (tool_name.ends_with(".query"), "db.query", True),
        (tool_name.glob("db.*"), "db.query", True),
        (tool_name.in_(["web.*", "db.*"]), "db.query", True),
        (tool_name.in_(["web.*"]), "db.query", False),
    ],
)
async def test_tool_name_conditions(cond, tool: str, should_block: bool) -> None:
    tools = [
        {"name": "db.query", "categories": ["data"]},
        {"name": "web.search", "categories": ["search"]},
    ]
    rules = [config_to_rule(rule.pre(priority=1, if_=cond, do=[block()]))]
    engine = _engine_with_tools(tools, rules=rules)
    ctx = engine.create_run_context(run_id="run-name-1", user_category="test")

    d = await engine.before_tool(ctx, tool_name=tool, args={})
    if should_block:
        assert d["effect"] == "block"
        assert d["code"] == "BLOCKED_RULE"
    else:
        assert d["effect"] == "allow"

@pytest.mark.asyncio
async def test_sequence_must_have_called() -> None:
    tools = [
        {"name": "web.search", "categories": ["search"]},
        {"name": "math.eval", "categories": ["compute"]},
        {"name": "db.query", "categories": ["data"]},
    ]
    rules = [
        config_to_rule(
            rule.pre(
                priority=1,
                if_=sequence(must_have_called=["web.*", "math.*"]),
                do=[block()],
            )
        )
    ]
    engine = _engine_with_tools(tools, rules=rules)
    ctx = engine.create_run_context(run_id="run-seq-1", user_category="test")

    await engine.after_tool(ctx, "web.search", 1, {}, {"ok": True})
    await engine.after_tool(ctx, "math.eval", 1, {}, {"ok": True})

    d = await engine.before_tool(ctx, tool_name="db.query", args={})
    assert d["effect"] == "block"
    assert d["code"] == "BLOCKED_RULE"

@pytest.mark.asyncio
async def test_sequence_must_not_have_called() -> None:
    tools = [
        {"name": "danger.warn", "categories": ["danger"]},
        {"name": "db.query", "categories": ["data"]},
    ]
    rules = [
        config_to_rule(
            rule.pre(
                priority=1,
                if_=sequence(must_not_have_called=["danger.*"]),
                do=[block()],
            )
        )
    ]
    engine = _engine_with_tools(tools, rules=rules)

    ctx1 = engine.create_run_context(run_id="run-seq-2a", user_category="test")
    d1 = await engine.before_tool(ctx1, tool_name="db.query", args={})
    assert d1["effect"] == "block"
    assert d1["code"] == "BLOCKED_RULE"

    ctx2 = engine.create_run_context(run_id="run-seq-2b", user_category="test")
    await engine.after_tool(ctx2, "danger.warn", 1, {}, {"ok": True})
    d2 = await engine.before_tool(ctx2, tool_name="db.query", args={})
    assert d2["effect"] == "allow"
