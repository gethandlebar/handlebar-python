"""HandlebarCallbackHandler for LangChain — wraps the Handlebar governance core.

.. deprecated::
    Use :class:`handlebar.langchain.HandlebarMiddleware` instead.
    ``HandlebarCallbackHandler`` is retained only for LangChain versions that
    do not support the middleware API (``langchain < 1.0``).  The middleware
    provides cleaner tool blocking, native LLM short-circuiting, and better
    concurrency semantics.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from typing import Any
from uuid import UUID

from handlebar.core import (
    Actor,
    AgentDescriptor,
    EnforceMode,
    HandlebarClient,
    HandlebarClientConfig,
    LLMMessage,
    ModelInfo,
    RunConfig,
    SinkConfig,
    TokenUsage,
    Tool,
)
from handlebar.core import LLMResponse as HBLLMResponse
from handlebar.core.schema.governance import RunControl, Verdict
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

logger = logging.getLogger("handlebar.langchain")


# ---------------------------------------------------------------------------
# LangChain ↔ Handlebar type converters
# ---------------------------------------------------------------------------


def _model_info_from_serialized(serialized: dict[str, Any]) -> ModelInfo:
    """Extract model name and provider from LangChain's serialized model dict."""
    kwargs = serialized.get("kwargs", {})
    model_name = (
        kwargs.get("model_name") or kwargs.get("model") or serialized.get("name") or "unknown"
    )
    # Infer provider from the class id list, e.g.
    # ['langchain_openai', 'chat_models', 'base', 'ChatOpenAI'] → 'openai'
    ids: list[str] = serialized.get("id", [])
    provider: str | None = None
    if ids:
        # The first segment often encodes the provider, e.g. 'langchain_openai'
        first = ids[0]
        if "_" in first:
            provider = first.split("_", 1)[-1]  # e.g. 'openai', 'anthropic'
    return ModelInfo(name=str(model_name), provider=provider)


def _lc_role(msg_type: str) -> str:
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
    }.get(msg_type, "user")


def _lc_messages_to_hb(messages: list[list[BaseMessage]]) -> list[LLMMessage]:
    """Flatten a list-of-lists of LangChain BaseMessages into LLMMessages."""
    hb_messages: list[LLMMessage] = []
    for batch in messages:
        for msg in batch:
            role = _lc_role(msg.type)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            hb_messages.append(LLMMessage(role=role, content=content))  # ty:ignore[invalid-argument-type]
    return hb_messages


def _lc_result_to_hb(
    response: LLMResult,
    model_info: ModelInfo,
    duration_ms: float,
) -> HBLLMResponse:
    """Convert a LangChain LLMResult to a Handlebar LLMResponse."""
    llm_output = response.llm_output or {}

    # Token usage — OpenAI style and Anthropic style field names.
    usage_info: dict = llm_output.get("token_usage") or llm_output.get("usage") or {}
    usage: TokenUsage | None = None
    if usage_info:
        usage = TokenUsage(
            input_tokens=(usage_info.get("prompt_tokens") or usage_info.get("input_tokens")),
            output_tokens=(usage_info.get("completion_tokens") or usage_info.get("output_tokens")),
        )

    # Override model name from llm_output if serialized info was incomplete.
    if not model_info.name or model_info.name == "unknown":
        raw = llm_output.get("model_name") or llm_output.get("model")
        if raw:
            model_info = ModelInfo(name=str(raw), provider=model_info.provider)

    content_parts: list[dict] = []
    output_texts: list[str] = []

    for gen_list in response.generations:
        for gen in gen_list:
            if isinstance(gen, ChatGeneration):
                msg = gen.message
                # Capture tool calls emitted by the model (OpenAI / Anthropic format).
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    content_parts.append(
                        {
                            "type": "tool_call",
                            "toolName": tc.get("name", ""),
                            "input": tc.get("args", {}),
                        }
                    )
            text = gen.text or ""
            if text:
                content_parts.append({"type": "text", "text": text})
                output_texts.append(text)

    return HBLLMResponse(
        model=model_info,
        content=content_parts,
        output_text="".join(output_texts) or None,
        usage=usage,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


class HandlebarCallbackHandler(AsyncCallbackHandler):
    """Handlebar governance callback handler for LangChain agents.

    Attach to a LangChain runnable or agent executor via ``callbacks=``.

    Minimal usage (``HANDLEBAR_API_KEY`` read from environment)::

        from handlebar.langchain import HandlebarCallbackHandler

        handler = HandlebarCallbackHandler(agent_slug="my-agent")
        result = await agent_executor.ainvoke(input, config={"callbacks": [handler]})

    With explicit config::

        handler = HandlebarCallbackHandler(
            agent=AgentDescriptor(slug="my-agent", name="My Agent"),
            api_key="hb_...",
            enforce_mode="enforce",
        )

    With a pre-initialized client::

        client = await HandlebarClient.init(config)
        handler = HandlebarCallbackHandler(client=client)

    Tool blocking
    -------------
    When ``enforce_mode="enforce"`` (the default) and a governance policy blocks a
    tool call, ``on_tool_start`` raises an exception which LangChain propagates
    before the tool body executes.  Set ``raise_error = False`` on the handler
    instance to disable propagation and fall back to observability-only mode.
    """

    # Propagate exceptions raised inside callbacks so tool blocking works.
    raise_error: bool = True

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
        # Optional per-run metadata.
        run_id: str | None = None,
        session_id: str | None = None,
        actor: Actor | None = None,
    ) -> None:
        warnings.warn(
            "HandlebarCallbackHandler is deprecated. Use HandlebarMiddleware instead. "
            "HandlebarCallbackHandler should only be used with langchain < 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        if client is None and agent_slug is None and agent is None:
            raise ValueError(
                "HandlebarCallbackHandler requires at least one of: "
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

        self._preset_run_id = run_id
        self._session_id = session_id
        self._actor = actor

        # Active Handlebar run — one per root chain invocation.
        self._run = None
        # LangChain UUID of the outermost chain (parent_run_id is None).
        self._root_lc_run_id: UUID | None = None

        # Per-LangChain-run_id state (keyed by str(UUID)).
        self._llm_start_ms: dict[str, float] = {}
        self._llm_models: dict[str, ModelInfo] = {}
        self._tool_start_ms: dict[str, float] = {}
        self._tool_names: dict[str, str] = {}
        self._tool_inputs: dict[str, Any] = {}
        # Track tool lc_run_ids that we blocked (to skip their on_tool_error).
        self._blocked_tool_ids: set[str] = set()

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
    # Chain lifecycle → Handlebar run lifecycle
    # ------------------------------------------------------------------

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a Handlebar run when the outermost chain begins."""
        if parent_run_id is not None:
            return  # Not the root chain — ignore.

        if self._run is not None and not self._run.is_ended:
            return  # Guard against re-entrant calls.

        self._root_lc_run_id = run_id
        logger.debug("[Handlebar] on_chain_start root run_id=%s", run_id)

        from uuid6 import uuid7

        hb_run_id = self._preset_run_id or str(uuid7())
        client = await self._get_client()
        self._run = await client.start_run(
            RunConfig(
                run_id=hb_run_id,
                session_id=self._session_id,
                actor=self._actor,
            )
        )

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id != self._root_lc_run_id:
            return
        logger.debug("[Handlebar] on_chain_end run_id=%s", run_id)
        if self._run and not self._run.is_ended:
            await self._run.end("success")
        self._run = None
        self._root_lc_run_id = None

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id != self._root_lc_run_id:
            return
        logger.debug("[Handlebar] on_chain_error run_id=%s error=%s", run_id, error)
        if self._run and not self._run.is_ended:
            await self._run.end("error")
        self._run = None
        self._root_lc_run_id = None

    # ------------------------------------------------------------------
    # LLM lifecycle
    # ------------------------------------------------------------------

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called before a chat model call (preferred path for modern LangChain)."""
        run = self._run
        if run is None:
            return

        key = str(run_id)
        self._llm_models[key] = _model_info_from_serialized(serialized)
        self._llm_start_ms[key] = time.monotonic() * 1000

        hb_messages = _lc_messages_to_hb(messages)
        await run.before_llm(hb_messages)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called before a completion model call."""
        run = self._run
        if run is None:
            return

        key = str(run_id)
        self._llm_models[key] = _model_info_from_serialized(serialized)
        self._llm_start_ms[key] = time.monotonic() * 1000

        hb_messages = [LLMMessage(role="user", content=p) for p in prompts]
        await run.before_llm(hb_messages)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._run
        if run is None:
            return

        key = str(run_id)
        model_info = self._llm_models.pop(key, ModelInfo(name="unknown"))
        start_ms = self._llm_start_ms.pop(key, time.monotonic() * 1000)
        duration_ms = time.monotonic() * 1000 - start_ms

        hb_response = _lc_result_to_hb(response, model_info, duration_ms)
        await run.after_llm(hb_response)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._llm_models.pop(key, None)
        self._llm_start_ms.pop(key, None)

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Gate the tool call against governance rules.

        Raises if the decision is BLOCK or TERMINATE (when enforce_mode="enforce").
        Because ``raise_error = True`` on this handler, LangChain propagates the
        exception before the tool body runs.

        Attach ``handlebar_tags`` to a tool's metadata to pass tags through::

            @tool(metadata={"handlebar_tags": ["dangerous", "write"]})
            def my_tool(input: str) -> str: ...
        """
        run = self._run
        if run is None:
            return

        tool_name: str = serialized.get("name") or "unknown_tool"
        key = str(run_id)

        self._tool_names[key] = tool_name
        self._tool_start_ms[key] = time.monotonic() * 1000
        tool_args: dict = inputs or {"input": input_str}
        self._tool_inputs[key] = tool_args

        logger.debug("[Handlebar] on_tool_start tool=%s run_id=%s", tool_name, run_id)

        tool_tags: list[str] | None = (metadata or {}).get("handlebar_tags")
        decision = await run.before_tool(
            tool_name=tool_name,
            args=tool_args,
            tool_tags=tool_tags,
        )

        if decision.verdict == Verdict.BLOCK or decision.control == RunControl.TERMINATE:
            self._blocked_tool_ids.add(key)
            msg = f"Blocked by Handlebar governance: {decision.message}"
            logger.info("[Handlebar] Blocking tool %s: %s", tool_name, decision.message)
            raise RuntimeError(msg)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._run
        key = str(run_id)

        if key in self._blocked_tool_ids:
            # Tool was blocked by us — clean up and skip recording.
            self._blocked_tool_ids.discard(key)
            self._tool_names.pop(key, None)
            self._tool_start_ms.pop(key, None)
            self._tool_inputs.pop(key, None)
            return

        if run is None:
            return

        tool_name = self._tool_names.pop(key, "unknown_tool")
        start_ms = self._tool_start_ms.pop(key, time.monotonic() * 1000)
        duration_ms = time.monotonic() * 1000 - start_ms
        tool_args = self._tool_inputs.pop(key, {})

        logger.debug("[Handlebar] on_tool_end tool=%s run_id=%s", tool_name, run_id)
        await run.after_tool(
            tool_name=tool_name,
            args=tool_args,
            result=output,
            duration_ms=duration_ms,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        run = self._run
        key = str(run_id)

        if key in self._blocked_tool_ids:
            # This error originates from our own RuntimeError raised in on_tool_start.
            self._blocked_tool_ids.discard(key)
            self._tool_names.pop(key, None)
            self._tool_start_ms.pop(key, None)
            self._tool_inputs.pop(key, None)
            return

        if run is None:
            return

        tool_name = self._tool_names.pop(key, "unknown_tool")
        start_ms = self._tool_start_ms.pop(key, time.monotonic() * 1000)
        duration_ms = time.monotonic() * 1000 - start_ms
        tool_args = self._tool_inputs.pop(key, {})

        logger.debug("[Handlebar] on_tool_error tool=%s run_id=%s", tool_name, run_id)
        await run.after_tool(
            tool_name=tool_name,
            args=tool_args,
            result=None,
            duration_ms=duration_ms,
            error=error,
        )
