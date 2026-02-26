"""HandlebarPlugin for Google ADK — wraps the Handlebar governance core."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from handlebar.core import (
    AgentDescriptor,
    EnforceMode,
    HandlebarClient,
    HandlebarClientConfig,
    LLMMessage,
    LLMResponse as HBLLMResponse,
    ModelInfo,
    Run,
    RunConfig,
    SinkConfig,
    TokenUsage,
    Tool,
)
from handlebar.core.schema.governance import RunControl, Verdict

logger = logging.getLogger("handlebar.google_adk")


# ---------------------------------------------------------------------------
# Per-invocation state
# ---------------------------------------------------------------------------


@dataclass
class _InvocationState:
    run: Run
    current_model: str = ""
    model_start_ms: float = 0.0
    # Set to a RunEndStatus value when a TERMINATE decision is received.
    terminate_status: str | None = None


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class HandlebarPlugin(BasePlugin):
    """Handlebar governance plugin for Google ADK agents.

    Attaches to a ``Runner`` via the ``plugins=`` parameter and hooks into
    every LLM call and tool invocation within each agent run.

    Minimal usage (``HANDLEBAR_API_KEY`` read from environment)::

        from handlebar.google_adk import HandlebarPlugin
        from google.adk.runners import InMemoryRunner

        plugin = HandlebarPlugin(agent_slug="my-agent")
        runner = InMemoryRunner(agent=my_agent, plugins=[plugin])

    With explicit config::

        plugin = HandlebarPlugin(
            agent=AgentDescriptor(slug="my-agent", name="My Agent"),
            api_key="hb_...",
            enforce_mode="enforce",
        )

    With a pre-initialized client::

        client = await HandlebarClient.init(config)
        plugin = HandlebarPlugin(client=client)
    """

    def __init__(
        self,
        *,
        # Option A: pass an already-initialised client.
        client: HandlebarClient | None = None,
        # Option B: minimal config for lazy auto-init.
        agent_slug: str | None = None,
        agent: AgentDescriptor | None = None,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        enforce_mode: EnforceMode = "enforce",
        fail_closed: bool = False,
        sinks: list[SinkConfig] | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        if client is None and agent_slug is None and agent is None:
            raise ValueError(
                "HandlebarPlugin requires at least one of: "
                "`client`, `agent_slug`, or `agent`."
            )

        super().__init__(name="handlebar")

        self._client: HandlebarClient | None = client
        self._init_lock = asyncio.Lock()

        # Config stored for lazy client creation.
        self._agent_descriptor: AgentDescriptor | None = (
            agent or (AgentDescriptor(slug=agent_slug) if agent_slug else None)
        )
        self._api_key = api_key
        self._api_endpoint = api_endpoint
        self._enforce_mode = enforce_mode
        self._fail_closed = fail_closed
        self._sinks = sinks
        self._tools = tools

        # Per-invocation state keyed by invocation_id.
        self._states: dict[str, _InvocationState] = {}

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> HandlebarClient:
        """Return the client, initialising it lazily on first call."""
        if self._client is not None:
            return self._client
        async with self._init_lock:
            if self._client is not None:  # double-checked locking
                return self._client
            cfg = HandlebarClientConfig(
                agent=self._agent_descriptor,  # type: ignore[arg-type]
                api_key=self._api_key,
                api_endpoint=self._api_endpoint,
                enforce_mode=self._enforce_mode,
                fail_closed=self._fail_closed,
                sinks=self._sinks,
                tools=self._tools,
            )
            self._client = await HandlebarClient.init(cfg)
        return self._client

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    async def before_agent_callback(self, *, agent, callback_context) -> None:
        """Start a Handlebar run for this ADK invocation."""
        inv_id = callback_context.invocation_id
        if inv_id in self._states:
            # Nested/sub-agent within the same invocation — reuse the run.
            return None

        client = await self._get_client()
        run = await client.start_run(RunConfig(run_id=inv_id))
        self._states[inv_id] = _InvocationState(run=run)
        return None

    async def after_agent_callback(self, *, agent, callback_context, result=None) -> None:
        """End the Handlebar run, flushing all pending audit events."""
        inv_id = callback_context.invocation_id
        state = self._states.pop(inv_id, None)
        if state is None:
            return None

        status = state.terminate_status or "success"
        await state.run.end(status)  # type: ignore[arg-type]
        return None

    # ------------------------------------------------------------------
    # LLM lifecycle
    # ------------------------------------------------------------------

    async def before_model_callback(
        self, *, callback_context, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """Gate the LLM call.

        If a prior tool was blocked with ``TERMINATE`` control, returns a
        synthetic ``LlmResponse`` (no function calls) so the ADK agent loop
        treats this invocation as complete without hitting the real model.
        """
        inv_id = callback_context.invocation_id
        state = self._states.get(inv_id)
        if state is None:
            return None

        # Short-circuit: a governance TERMINATE was already issued this turn.
        if state.terminate_status is not None:
            return _make_terminate_response(
                "This agent run has been paused by a governance policy."
            )

        # Track model name and timing so after_model_callback can build
        # a complete LLMResponse for the audit log.
        state.current_model = getattr(llm_request, "model", "") or ""
        state.model_start_ms = time.monotonic() * 1000

        messages = _adk_request_to_messages(llm_request)
        await state.run.before_llm(messages)
        return None

    async def after_model_callback(
        self, *, callback_context, llm_response: LlmResponse
    ) -> None:
        """Record LLM token usage and response in the audit log."""
        inv_id = callback_context.invocation_id
        state = self._states.get(inv_id)
        if state is None:
            return None

        duration_ms = time.monotonic() * 1000 - state.model_start_ms
        hb_response = _adk_response_to_llm_response(
            llm_response, state.current_model, duration_ms
        )
        await state.run.after_llm(hb_response)
        return None

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool,
        tool_args: dict[str, Any],
        tool_context,
    ) -> Optional[dict]:
        """Evaluate the tool call against governance rules.

        Returns a dict to use as the tool result (bypassing real execution)
        when the decision is BLOCK, or when control is TERMINATE.
        """
        inv_id = tool_context.invocation_id
        state = self._states.get(inv_id)
        if state is None:
            return None

        # Extract tags stored by the developer in tool.custom_metadata.
        tool_tags: list[str] | None = (
            (getattr(tool, "custom_metadata", None) or {}).get("handlebar_tags")
        )

        decision = await state.run.before_tool(
            tool_name=tool.name,
            args=tool_args,
            tool_tags=tool_tags,
        )

        if decision.verdict == Verdict.BLOCK or decision.control == RunControl.TERMINATE:
            if decision.control == RunControl.TERMINATE:
                # Signal before_model_callback to stop the agent loop.
                state.terminate_status = "interrupted"
            return {
                "error": (
                    f"Blocked by Handlebar governance: {decision.message}"
                )
            }

        return None

    async def after_tool_callback(
        self,
        *,
        tool,
        tool_args: dict[str, Any],
        tool_context,
        result: dict,
    ) -> None:
        """Record the tool result in the audit log."""
        inv_id = tool_context.invocation_id
        state = self._states.get(inv_id)
        if state is None:
            return None

        tool_tags: list[str] | None = (
            (getattr(tool, "custom_metadata", None) or {}).get("handlebar_tags")
        )

        await state.run.after_tool(
            tool_name=tool.name,
            args=tool_args,
            result=result,
            tool_tags=tool_tags,
        )
        return None


# ---------------------------------------------------------------------------
# ADK ↔ Handlebar type converters
# ---------------------------------------------------------------------------


def _adk_request_to_messages(llm_request: LlmRequest) -> list[LLMMessage]:
    """Convert ADK ``LlmRequest.contents`` to a list of ``LLMMessage``."""
    messages: list[LLMMessage] = []
    for content in llm_request.contents or []:
        role: str = getattr(content, "role", "user") or "user"
        # ADK uses "model" for the assistant role.
        hb_role = "assistant" if role == "model" else role
        if hb_role not in ("system", "user", "assistant", "tool"):
            hb_role = "user"

        chunks: list[str] = []
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
            fc = getattr(part, "function_call", None)
            if fc:
                chunks.append(f"[tool_call:{fc.name}]")
            fr = getattr(part, "function_response", None)
            if fr:
                chunks.append(f"[tool_result:{fr.name}]")

        messages.append(LLMMessage(role=hb_role, content="\n".join(chunks)))  # type: ignore[arg-type]
    return messages


def _adk_response_to_llm_response(
    adk_response: LlmResponse,
    model_name: str,
    duration_ms: float,
) -> HBLLMResponse:
    """Convert an ADK ``LlmResponse`` to a Handlebar ``LLMResponse``."""
    content = getattr(adk_response, "content", None)
    parts = getattr(content, "parts", None) or []

    output_text: str | None = None
    response_parts: list[dict] = []

    for part in parts:
        text = getattr(part, "text", None)
        if text:
            output_text = (output_text or "") + text
            response_parts.append({"type": "text", "text": text})
        fc = getattr(part, "function_call", None)
        if fc:
            response_parts.append(
                {"type": "tool_call", "toolName": fc.name, "input": fc.args}
            )

    usage: TokenUsage | None = None
    usage_meta = getattr(adk_response, "usage_metadata", None)
    if usage_meta:
        usage = TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", None),
            output_tokens=getattr(usage_meta, "candidates_token_count", None),
        )

    return HBLLMResponse(
        model=ModelInfo(name=model_name or "unknown", provider="google"),
        content=response_parts,
        output_text=output_text,
        usage=usage,
        duration_ms=duration_ms,
    )


def _make_terminate_response(message: str) -> LlmResponse:
    """Build a synthetic LlmResponse that stops the ADK agent loop.

    Returning a non-None value from ``before_model_callback`` bypasses the
    real model call.  A response with no function calls causes ADK to treat
    the invocation as complete.
    """
    from google.genai import types as genai_types

    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=message)],
        ),
        turn_complete=True,
    )
