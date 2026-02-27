"""HandlebarMiddleware for LangChain / LangGraph agents."""

from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
from typing import Any, Callable, Literal

from handlebar.core import (
    Actor,
    AgentDescriptor,
    EnforceMode,
    HandlebarClient,
    HandlebarClientConfig,
    LLMMessage,
    ModelInfo,
    Run,
    RunConfig,
    SinkConfig,
    TokenUsage,
    Tool,
)
from handlebar.core import LLMResponse as HBLLMResponse
from handlebar.core.schema.governance import RunControl, Verdict
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger("handlebar.langchain")

# Per-async-task context vars — naturally scoped to each agent invocation so
# a shared middleware instance handles concurrent runs without cross-talk.
_active_run: ContextVar[Run | None] = ContextVar("_hb_active_run", default=None)
_terminate_pending: ContextVar[bool] = ContextVar("_hb_terminate_pending", default=False)


# ---------------------------------------------------------------------------
# LangChain ↔ Handlebar type converters
# ---------------------------------------------------------------------------


def _lc_role(msg_type: str) -> Literal["system", "user", "assistant", "tool"]:
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
    }.get(msg_type, "user")  # ty:ignore[invalid-return-type]


def _base_messages_to_hb(messages: list[BaseMessage]) -> list[LLMMessage]:
    return [
        LLMMessage(
            role=_lc_role(m.type),
            content=m.content if isinstance(m.content, str) else str(m.content),
        )
        for m in messages
    ]


def _ai_message_to_hb_response(
    message: AIMessage,
    model_info: ModelInfo,
    duration_ms: float,
) -> HBLLMResponse:
    content_parts: list[dict] = []
    output_texts: list[str] = []

    raw = message.content
    if isinstance(raw, str) and raw:
        content_parts.append({"type": "text", "text": raw})
        output_texts.append(raw)
    elif isinstance(raw, list):
        for part in raw:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    content_parts.append({"type": "text", "text": text})
                    output_texts.append(text)

    for tc in getattr(message, "tool_calls", None) or []:
        content_parts.append(
            {"type": "tool_call", "toolName": tc.get("name", ""), "input": tc.get("args", {})}
        )

    # langchain_core >= 0.2 exposes usage_metadata on AIMessage.
    usage: TokenUsage | None = None
    usage_meta = getattr(message, "usage_metadata", None)
    if usage_meta:
        usage = TokenUsage(
            input_tokens=usage_meta.get("input_tokens"),
            output_tokens=usage_meta.get("output_tokens"),
        )

    return HBLLMResponse(
        model=model_info,
        content=content_parts,
        output_text="".join(output_texts) or None,
        usage=usage,
        duration_ms=duration_ms,
    )


def _model_info_from_response(response: Any) -> ModelInfo:
    meta = getattr(response, "response_metadata", None) or {}
    name = meta.get("model_name") or meta.get("model") or "unknown"
    return ModelInfo(name=str(name))


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class HandlebarMiddleware(AgentMiddleware):
    """Handlebar governance middleware for LangChain / LangGraph agents.

    Attach via the ``middleware=`` parameter when creating your agent::

        from handlebar.langchain import HandlebarMiddleware

        middleware = HandlebarMiddleware(agent_slug="my-agent")
        agent = create_agent(model="gpt-4o", tools=[...], middleware=[middleware])

    With explicit config::

        middleware = HandlebarMiddleware(
            agent=AgentDescriptor(slug="my-agent", name="My Agent"),
            api_key="hb_...",
            enforce_mode="enforce",
        )

    Tool blocking
    -------------
    ``wrap_tool_call`` calls ``run.before_tool`` before the tool body executes.
    If the decision is BLOCK or TERMINATE, a synthetic ``ToolMessage`` is
    returned and the real tool is never called.

    LLM short-circuit
    -----------------
    If a prior tool decision issued TERMINATE, ``wrap_model_call`` returns a
    synthetic ``AIMessage`` on the next model call without hitting the real
    model, ending the agent loop cleanly.
    """

    def __init__(
        self,
        *,
        client: HandlebarClient | None = None,
        agent_slug: str | None = None,
        agent: AgentDescriptor | None = None,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        enforce_mode: EnforceMode = "enforce",
        fail_closed: bool = False,
        sinks: list[SinkConfig] | None = None,
        tools: list[Tool] | None = None,
        session_id: str | None = None,
        actor: Actor | None = None,
    ) -> None:
        if client is None and agent_slug is None and agent is None:
            raise ValueError(
                "HandlebarMiddleware requires at least one of: "
                "`client`, `agent_slug`, or `agent`."
            )

        super().__init__()

        self._client: HandlebarClient | None = client
        self._init_lock = asyncio.Lock()

        self._agent_descriptor: AgentDescriptor | None = agent or (
            AgentDescriptor(slug=agent_slug) if agent_slug else None
        )
        self._api_key = api_key
        self._api_endpoint = api_endpoint
        self._enforce_mode = enforce_mode
        self._fail_closed = fail_closed
        self._sinks = sinks
        self._tools = tools
        self._session_id = session_id
        self._actor = actor

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def _get_client(self) -> HandlebarClient:
        if self._client is not None:
            return self._client

        async with self._init_lock:
            if self._client is not None:
                return self._client

            cfg = HandlebarClientConfig(
                agent=self._agent_descriptor,
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

    async def before_agent(self, state: Any, runtime: Any) -> None:
        """Start a Handlebar run for this agent invocation."""
        from uuid6 import uuid7

        logger.debug("[Handlebar] before_agent")
        client = await self._get_client()
        run = await client.start_run(
            RunConfig(
                run_id=str(uuid7()),
                session_id=self._session_id,
                actor=self._actor,
            )
        )
        _active_run.set(run)
        _terminate_pending.set(False)

    async def after_agent(self, state: Any, runtime: Any) -> None:
        """End the Handlebar run, flushing all pending audit events."""
        logger.debug("[Handlebar] after_agent")
        run = _active_run.get()
        if run and not run.is_ended:
            await run.end("success")
        _active_run.set(None)
        _terminate_pending.set(False)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Wrap the LLM call: emit before_llm/after_llm and short-circuit on TERMINATE."""
        run = _active_run.get()

        # A prior tool decision issued TERMINATE — return a synthetic response
        # without hitting the model so the agent loop ends cleanly.
        if _terminate_pending.get():
            logger.info("[Handlebar] Suppressing model call due to TERMINATE")
            if run and not run.is_ended:
                await run.end("interrupted")
            _active_run.set(None)
            return AIMessage("This agent run has been paused by a governance policy.")

        if run:
            messages: list[BaseMessage] = getattr(request, "messages", None) or []
            await run.before_llm(_base_messages_to_hb(messages))

        start_ms = time.monotonic() * 1000
        result = handler(request)
        if asyncio.iscoroutine(result):
            result = await result
        duration_ms = time.monotonic() * 1000 - start_ms

        if run:
            model_info = _model_info_from_response(result)
            await run.after_llm(_ai_message_to_hb_response(result, model_info, duration_ms))

        return result

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    async def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Gate the tool call: before_tool → optional short-circuit → after_tool."""
        run = _active_run.get()

        tool_call: dict = getattr(request, "tool_call", None) or {}
        tool_name: str = tool_call.get("name", "unknown_tool")
        tool_args: dict = tool_call.get("args", {})
        tool_call_id: str = tool_call.get("id", "")

        tool_tags: list[str] | None = None
        tool_meta = getattr(request, "metadata", None)
        if isinstance(tool_meta, dict):
            tool_tags = tool_meta.get("handlebar_tags")

        logger.debug("[Handlebar] wrap_tool_call tool=%s", tool_name)

        if run:
            decision = await run.before_tool(
                tool_name=tool_name,
                args=tool_args,
                tool_tags=tool_tags,
            )

            if decision.verdict == Verdict.BLOCK or decision.control == RunControl.TERMINATE:
                if decision.control == RunControl.TERMINATE:
                    _terminate_pending.set(True)
                msg = f"Blocked by Handlebar governance: {decision.message}"
                logger.info("[Handlebar] Blocking tool %s: %s", tool_name, decision.message)
                return ToolMessage(content=msg, tool_call_id=tool_call_id)

        start_ms = time.monotonic() * 1000
        result = handler(request)
        if asyncio.iscoroutine(result):
            result = await result
        duration_ms = time.monotonic() * 1000 - start_ms

        if run:
            result_content = getattr(result, "content", str(result))
            await run.after_tool(
                tool_name=tool_name,
                args=tool_args,
                result=result_content,
                duration_ms=duration_ms,
            )

        return result
