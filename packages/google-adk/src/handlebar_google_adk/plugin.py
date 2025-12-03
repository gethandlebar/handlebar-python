from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Literal

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk import __version__ as adk_version
from google.genai import types as genai_types


from handlebar_core.engine import GovernanceEngine
from handlebar_core.utils import slugify
from handlebar_core.types import GovernanceDecision, RunContext, ToolMeta
from handlebar_core.telemetry import Telemetry, HttpSink, emit, push_run_context, pop_run_context

from .api import fetch_governance_config, upsert_agent


class HandlebarPlugin(BasePlugin):
    """
    ADK Plugin that wires Handlebar governance into model + tool execution.

    - Uses ADK's invocation_id as Handlebar runId.
    - Creates/stashes RunContext per invocation.
    - Runs GovernanceEngine.before_tool / after_tool on every tool call.
    """

    def __init__(
        self,
        *,
        app_name: str,
        handlebar_api_key: str,
        handlebar_base_url: str = "https://api.gethandlebar.com",
        handlebar_org_id: Optional[str] = None, # TODO: remove
        handlebar_user_category: str = "default", # TODO: remove
        tool_categories: Optional[Dict[str, List[str]]] = None,
        default_uncategorised: Literal["allow", "block"] = "allow",
    ) -> None:
        super().__init__(name="handlebar")
        self.app_name = app_name
        self.api_key = handlebar_api_key
        self.base_url = handlebar_base_url
        self.org_id = handlebar_org_id
        self.user_category_default = handlebar_user_category
        self.tool_categories = tool_categories or {}
        self.default_uncategorised = default_uncategorised

        self._agent_id: Optional[str] = None
        self._engine: Optional[GovernanceEngine] = None
        self._contexts: Dict[str, RunContext] = {}
        self._ctx_tokens: Dict[str, Any] = {}
        self._tool_metas: List[ToolMeta] = []
        self._last_cfg_etag: Optional[str] = None  # Later, for caching

        if self.api_key:
            audit_endpoint = f"{self.base_url}/v1/audit/ingest"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            Telemetry.add_sink(HttpSink(audit_endpoint, headers=headers))

    async def _ensure_engine(self, agent: BaseAgent) -> GovernanceEngine:
        if self._engine is not None:
            return self._engine

        # Build ToolMeta list up front from the root agent.tools
        tool_metas: List[ToolMeta] = []
        for tool in getattr(agent, "tools", []) or []:
            # Best-effort name; ADK BaseTool always has .name
            name = getattr(tool, "name", getattr(tool, "__name__", "unnamed_tool"))
            cats = self.tool_categories.get(name, [])
            tm: ToolMeta = {"name": name}
            if cats:
                tm["categories"] = cats
            tool_metas.append(tm)

        self._tool_metas = tool_metas
        slug = slugify(f"{self.app_name}-{agent.name}")
        description = getattr(agent, "description", None)
        tags = []

        self._agent_id = await upsert_agent(
            api_key=self.api_key,
            base_url=self.base_url,
            slug=slug,
            name=agent.name,
            description=description,
            tags=tags,
        )

        cfg = await fetch_governance_config(
            api_key=self.api_key,
            base_url=self.base_url,
            app_name=self.app_name,
            org_id=self.org_id,
            tools=tool_metas,
            default_uncategorised=self.default_uncategorised,
            agent_id=self._agent_id,
        )
        self._engine = GovernanceEngine(cfg)
        return self._engine

    def _get_ctx_for_invocation(self, invocation_id: str) -> Optional[RunContext]:
        return self._contexts.get(invocation_id)

    def _set_ctx_for_invocation(self, invocation_id: str, ctx: RunContext) -> None:
        self._contexts[invocation_id] = ctx

    def _resolve_user_category(self, callback_context: CallbackContext) -> str:
        # TODO: read from callback_context.state or session state.
        return self.user_category_default

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        """
        Create a RunContext for this invocation and ensure engine is ready.
        """
        engine = await self._ensure_engine(agent)

        invocation_id = getattr(callback_context, "invocation_id", None)
        if not invocation_id:
            # Shouldn't happen, but don't crash if ADK changes something.
            return

        user_category = self._resolve_user_category(callback_context)

        ctx: RunContext = engine.create_run_context(
            run_id=invocation_id,
            user_category=user_category,
        )
        self._set_ctx_for_invocation(invocation_id, ctx)
        token = push_run_context(ctx)
        self._ctx_tokens[invocation_id] = token

        emit("run.started",
            {
                "agent": {
                    "framework": "google-adk",
                    "version": adk_version,
                    "id": self._agent_id,
                    "name": agent.name,
                },
                "adapter": {
                    "name": "handlebar-google-adk",
                    # "version": None,  # TODO: get package version
                },
                # "request": {
                #     "id": invocation_id,
                #     # TODO: check traceparent header in callback_context, drop it here
                #     "traceparent": None,
                # }
            })

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """
        Final run cleanup: audit logging the end

        IMPORTANT: There is a known bug that stops this callback from being invoked if you break using "is_final_response"
        (the quickstart example code)
        https://github.com/google/adk-python/discussions/1927
        """
        invocation_id = invocation_context.invocation_id
        ctx = self._get_ctx_for_invocation(invocation_id)
        emit("run.ended", {
            "status": "ok",
            "totalSteps": len(ctx.get("history", [])) if ctx is not None else 0,
            "firstErrorDecisionId": "", # deprecated: this field needs to be removed from source types.
            "summary": "",
        })

        token = self._ctx_tokens.pop(invocation_id, None)
        if token is not None:
            pop_run_context(token)

        if invocation_id in self._contexts:
            self._contexts.pop(invocation_id, None)
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        # TODO: emit model request event.
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        # TODO: emit model response event.
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Run Handlebar governance *before* a tool executes.

        If we return a dict, ADK skips the actual tool call and uses this as
        the tool result instead (block / HITL / synthetic response).
        """
        invocation_id = getattr(tool_context, "invocation_id", None)
        if not invocation_id:
            return None

        engine = self._engine
        if engine is None:
            # No engine: nothing to do.
            return None

        ctx = self._get_ctx_for_invocation(invocation_id)
        if ctx is None:
            # Fallback safety: create a context lazily.
            ctx = engine.create_run_context(
                run_id=invocation_id,
                user_category=self.user_category_default,
            )
            self._set_ctx_for_invocation(invocation_id, ctx)

        tool_name = getattr(tool, "name", "unnamed_tool")
        call_args = tool_args or args or {}

        # Run governance
        decision: GovernanceDecision = await engine.before_tool(
            ctx,
            tool_name=tool_name,
            args=call_args,
        )

        if engine.should_block(decision):
            return {
                "status": "blocked",
                "reason": decision.get("reason")
                or decision.get("code")
                or "Blocked by Handlebar policy",
                "handlebar": {
                    "effect": decision.get("effect"),
                    "code": decision.get("code"),
                    "matchedRuleIds": decision.get("matchedRuleIds", []),
                    "appliedActions": decision.get("appliedActions", []),
                },
            }

        # Allowed: proceed with normal tool execution.
        # Track start time in context state so we can compute duration later.
        start_ms = int(time.time() * 1000)
        state = ctx.get("state", {})
        state[f"__hb_tool_start__:{tool_name}"] = start_ms
        ctx["state"] = state
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: Dict,
    ) -> Optional[dict]:
        """
        Called after the tool has executed successfully.

        We call GovernanceEngine.after_tool to update history and emit telemetry.
        """
        invocation_id = getattr(tool_context, "invocation_id", None)
        if not invocation_id:
            return None

        engine = self._engine
        if engine is None:
            return None

        ctx = self._get_ctx_for_invocation(invocation_id)
        if ctx is None:
            return None

        tool_name = getattr(tool, "name", "unnamed_tool")

        # Compute duration from stored start time, if available
        state = ctx.get("state", {})
        key = f"__hb_tool_start__:{tool_name}"
        start_ms = state.pop(key, None)
        ctx["state"] = state
        now_ms = int(time.time() * 1000)
        duration_ms: Optional[int] = None
        if isinstance(start_ms, int):
            duration_ms = now_ms - start_ms

        await engine.after_tool(
            ctx,
            tool_name=tool_name,
            execution_time_ms=duration_ms,
            args=tool_args,
            result=result,
            error=None,
        )

        # No response modifications now (blocking on _after_ not yet supported)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: Dict[str, Any],
        error: Exception,
    ) -> Optional[Dict[str, Any]]:
        """
        Ensure tool errors are also visible to Handlebar via after_tool.
        """
        invocation_id = getattr(tool_context, "invocation_id", None)
        if not invocation_id or self._engine is None:
            return None

        ctx = self._get_ctx_for_invocation(invocation_id)
        if ctx is None:
            return None

        tool_name = getattr(tool, "name", "unnamed_tool")

        # Duration is unknown here?
        self._engine.after_tool(
            ctx,
            tool_name=tool_name,
            execution_time_ms=None,
            args=tool_args,
            result=None,
            error=error,
        )

        # Return None to let ADK raise the original error.
        return None
