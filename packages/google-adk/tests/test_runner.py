import pytest
from typing import Any, Dict, List, Optional, cast

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext

from handlebar_core.types import Rule, ToolMeta, GovernanceConfig
from handlebar_google_adk.plugin import HandlebarPlugin
from handlebar_google_adk.runner import HandlebarRunner


class DummyCallbackContext:
    """
    Minimal stand-in for google.adk.agents.callback_context.CallbackContext.

    Our plugin only reads:
      - invocation_id
      - agent_name
      - session_id (maybe later)
    """

    def __init__(self, invocation_id: str, agent_name: str, session_id: str):
        self.invocation_id = invocation_id
        self.agent_name = agent_name
        self.session_id = session_id
        self.state: Dict[str, Any] = {}  # optional, for future use


class DummyToolContext:
    """
    Minimal stand-in for google.adk.tools.tool_context.ToolContext.

    Our plugin only reads:
      - invocation_id
    """

    def __init__(self, invocation_id: str):
        self.invocation_id = invocation_id
        self.state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


class DummyTool(BaseTool):
    """Simple tool for testing; only name matters for our plugin."""

    def __init__(self, name: str):
        super().__init__(name=name, description="")

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        # In these tests we never actually run via Runner, but this keeps it valid.
        return {"ok": True, "kwargs": kwargs}


def make_rule_blocking_tool(tool_name: str) -> Rule:
    return cast(
        Rule,
        {
            "id": "rule-block",
            "policy_id": "policy-1",
            "priority": 0,
            "when": "pre",
            "condition": {
                "kind": "toolName",
                "op": "eq",
                "value": tool_name,
            },
            "actions": [{"type": "block"}],
        },
    )


def make_allow_config(tools: List[ToolMeta]) -> GovernanceConfig:
    return {
        "tools": tools,
        "rules": [],
        "defaultUncategorised": "allow",
        "mode": "enforce",
        "verbose": False,
    }


def make_block_config(tools: List[ToolMeta], tool_name: str) -> GovernanceConfig:
    return {
        "tools": tools,
        "rules": [make_rule_blocking_tool(tool_name)],
        "defaultUncategorised": "allow",
        "mode": "enforce",
        "verbose": False,
    }


# ---------------------------------------------------------------------------
# Plugin tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_allows_tool_when_no_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If backend returns no rules, tool should be allowed and after_tool should record history."""

    # Arrange: dummy tool + agent
    tool = DummyTool(name="safe_tool")
    agent = Agent(
        name="test_agent",
        model="gemini-2.0-flash",  # just a string; we don't call the model
        description="Test agent",
        instruction="Test instruction",
        tools=[tool],
    )

    # Patch fetch_governance_config to return an "allow-all" config
    from handlebar_google_adk import api as api_client_module

    captured_tools: List[ToolMeta] = []

    async def fake_fetch_governance_config(
        *,
        api_key: str,
        base_url: str,
        app_name: str,
        org_id: Optional[str],
        tools: List[ToolMeta],
        default_uncategorised: str,
    ) -> GovernanceConfig:
        captured_tools[:] = tools
        return make_allow_config(tools)

    monkeypatch.setattr(
        api_client_module,
        "fetch_governance_config",
        fake_fetch_governance_config,
    )

    plugin = HandlebarPlugin(
        app_name="test_app",
        handlebar_api_key="test-key",
        handlebar_base_url="https://example.com",
    )

    # Simulate ADK invocation: before_agent -> before_tool -> after_tool
    callback_ctx = cast(
        CallbackContext,
        DummyCallbackContext(
            invocation_id="inv-1",
            agent_name=agent.name,
            session_id="sess-1",
        ),
    )

    # Initialise engine + RunContext
    await plugin.before_agent_callback(agent=agent, callback_context=callback_ctx)

    tool_context = cast(ToolContext, DummyToolContext(invocation_id="inv-1"))
    tool_args = {"x": 1}

    # Act: before_tool_callback (no rules → allow → returns None)
    result_pre = await plugin.before_tool_callback(
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
    )

    # Assert: no override, tool should execute
    assert result_pre is None

    # Act: pretend the tool ran and returned a result; after_tool should record it
    tool_result = {"ok": True}
    result_post = await plugin.after_tool_callback(
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
        result=tool_result,
    )

    # Assert: plugin doesn't override result
    assert result_post is None

    # Verify history and counters updated
    engine = plugin._engine
    assert engine is not None
    run_ctx = plugin._get_ctx_for_invocation("inv-1")
    assert run_ctx is not None
    assert len(run_ctx["history"]) == 1
    last = run_ctx["history"][-1]
    assert last["tool"]["name"] == "safe_tool"
    assert last["result"] == tool_result
    assert "__hb_totalDurationMs" in run_ctx["counters"]


@pytest.mark.asyncio
async def test_plugin_blocks_tool_when_rule_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If backend returns a blocking rule for the tool, plugin should short-circuit and not call the tool."""

    blocked_name = "dangerous_tool"
    tool = DummyTool(name=blocked_name)
    agent = Agent(
        name="test_agent_block",
        model="gemini-2.0-flash",
        description="Test agent",
        instruction="Test instruction",
        tools=[tool],
    )

    from handlebar_google_adk import api as api_client_module

    async def fake_fetch_governance_config(
        *,
        api_key: str,
        base_url: str,
        app_name: str,
        org_id: Optional[str],
        tools: List[ToolMeta],
        default_uncategorised: str,
    ) -> GovernanceConfig:
        return make_block_config(tools, blocked_name)

    monkeypatch.setattr(
        api_client_module,
        "fetch_governance_config",
        fake_fetch_governance_config,
    )

    plugin = HandlebarPlugin(
        app_name="test_app",
        handlebar_api_key="test-key",
        handlebar_base_url="https://example.com",
    )

    callback_ctx = cast(
        CallbackContext,
        DummyCallbackContext(
            invocation_id="inv-2",
            agent_name=agent.name,
            session_id="sess-2",
        ),
    )

    await plugin.before_agent_callback(agent=agent, callback_context=callback_ctx)

    tool_context = cast(ToolContext, DummyToolContext(invocation_id="inv-2"))
    tool_args = {"x": 42}

    # Act: before_tool_callback with blocking rule
    result_pre = await plugin.before_tool_callback(
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
    )

    # Assert: plugin returns synthetic result, so tool should be skipped
    assert isinstance(result_pre, dict)
    assert result_pre.get("status") == "blocked"
    hb_info = result_pre.get("handlebar") or {}
    assert hb_info.get("effect") == "block"
    # Tool didn't actually run; history must still be empty at this point.
    run_ctx = plugin._get_ctx_for_invocation("inv-2")
    assert run_ctx is not None
    assert len(run_ctx["history"]) == 0

    # Simulate ADK calling after_tool_callback with the synthetic result:
    result_post = await plugin.after_tool_callback(
        tool=tool,
        tool_args=tool_args,
        tool_context=tool_context,
        result=result_pre,
    )
    # We don't change that result either
    assert result_post is None
    # Now history should have exactly one entry (the synthetic "result")
    assert len(run_ctx["history"]) == 1
    last = run_ctx["history"][-1]
    assert last["result"] == result_pre


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


def test_handlebar_runner_injects_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    HandlebarRunner should:
      - be a Runner
      - inject HandlebarPlugin into the plugins list
      - preserve the provided session_service
    """

    agent = Agent(
        name="runner_test_agent",
        model="gemini-2.0-flash",
        description="Runner test agent",
        instruction="Just a test.",
        tools=[],
    )
    session_service = InMemorySessionService()

    runner = HandlebarRunner(
        agent=agent,
        app_name="runner_test_app",
        session_service=session_service,
        handlebar_api_key="test-key",
    )

    # If you've subclassed Runner, this will be True; if you wrap, adjust accordingly.
    assert isinstance(runner, Runner)

    # Session service should be what we passed in
    assert runner.session_service is session_service

    # Plugins should include a HandlebarPlugin
    plugins = getattr(runner, "plugins", None)
    assert plugins is not None
    from handlebar_google_adk.plugin import HandlebarPlugin as HbPluginClass

    assert any(isinstance(p, HbPluginClass) for p in plugins)
