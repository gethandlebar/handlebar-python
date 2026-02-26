"""Run — mirrors packages/core/src/run.ts."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from ._sync import run_sync
from .api.manager import ApiManager, EvaluateAfterRequest, EvaluateBeforeRequest
from .metrics.hooks import AgentMetricHookRegistry
from .schema.events import AuditEvent
from .schema.governance import Decision
from .sinks.bus import SinkBus
from .subjects import SubjectRegistry, sanitise_subjects
from .types import (
    FAILOPEN_DECISION,
    Actor,
    LLMMessage,
    LLMResponse,
    ModelInfo,
    RunConfig,
    RunEndStatus,
    Tool,
    ToolResult,
    derive_output_text,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("handlebar")

# Module-level adapter — built once, reused for every event emission.
_audit_event_adapter: TypeAdapter[AuditEvent] = TypeAdapter(AuditEvent)

# ---------------------------------------------------------------------------
# Internal config
# ---------------------------------------------------------------------------


class RunInternalConfig:
    def __init__(
        self,
        run_config: RunConfig,
        agent_id: str | None,
        tools: list[Tool] | None,
        enforce_mode: str,
        fail_closed: bool,
        api: ApiManager,
        bus: SinkBus,
        metric_registry: AgentMetricHookRegistry | None = None,
        subject_registry: SubjectRegistry | None = None,
    ) -> None:
        self.run_config = run_config
        self.agent_id = agent_id
        self.tools = tools
        self.enforce_mode = enforce_mode
        self.fail_closed = fail_closed
        self.api = api
        self.bus = bus
        self.metric_registry = metric_registry
        self.subject_registry = subject_registry


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run:
    """Encapsulates lifecycle hooks for a single agent run."""

    def __init__(self, config: RunInternalConfig) -> None:
        self.run_id: str = config.run_config.run_id
        self.session_id: str | None = config.run_config.session_id
        self.actor: Actor | None = config.run_config.actor
        self.tags: dict[str, str] = config.run_config.tags or {}

        self._state: str = "active"  # "active" | "ended"
        self._step_index: int = 0
        self._history: list[ToolResult] = []
        self._pending_llm_tokens_in: int = 0
        self._pending_llm_tokens_out: int = 0

        self._agent_id = config.agent_id
        self._tools = config.tools
        self._enforce_mode = config.enforce_mode
        self._api = config.api
        self._bus = config.bus
        self._metric_registry = config.metric_registry
        self._subject_registry = config.subject_registry

        self._ttl_task: asyncio.TimerHandle | None = None

        # Schedule auto-close if TTL is configured.
        ttl = config.run_config.run_ttl_ms
        if ttl is not None and ttl > 0:
            try:
                loop = asyncio.get_running_loop()
                self._ttl_task = loop.call_later(ttl / 1000, lambda: asyncio.ensure_future(self.end("timeout")))
            except RuntimeError:
                pass  # No running loop at construction time; TTL not supported in sync mode.

        self._emit_run_started()

    # ------------------------------------------------------------------
    # Lifecycle hooks — async
    # ------------------------------------------------------------------

    async def before_tool(
        self,
        tool_name: str,
        args: Any,
        tool_tags: list[str] | None = None,
    ) -> Decision:
        """Call before invoking a tool. Returns the Decision from the server.

        In shadow/off mode, always returns ALLOW without enforcing.
        """
        if self._state != "active":
            return FAILOPEN_DECISION
        if self._enforce_mode == "off":
            return FAILOPEN_DECISION

        before_metrics: dict[str, float | int] = {}

        bytes_in = _approx_bytes(args)
        if bytes_in is not None:
            before_metrics["bytes_in"] = bytes_in

        if self._pending_llm_tokens_in > 0:
            before_metrics["llm_tokens_in"] = self._pending_llm_tokens_in
            self._pending_llm_tokens_in = 0
        if self._pending_llm_tokens_out > 0:
            before_metrics["llm_tokens_out"] = self._pending_llm_tokens_out
            self._pending_llm_tokens_out = 0

        if self._metric_registry:
            from .metrics.types import AgentMetricInputToolBefore

            await self._metric_registry.run_phase(
                "tool.before",
                AgentMetricInputToolBefore(tool_name=tool_name, args=args, run=self),
                lambda key, value, unit=None: before_metrics.__setitem__(key, value),
            )

        subjects = None
        if self._subject_registry:
            from .tool import ToolMeta

            raw = await self._subject_registry.extract(
                tool=ToolMeta(tags=tool_tags),
                tool_name=tool_name,
                tool_args=args,
                run=self,
            )
            if raw:
                subjects = sanitise_subjects(raw)

        req = EvaluateBeforeRequest(
            agent_id=self._agent_id or "",
            tool_name=tool_name,
            tool_tags=tool_tags,
            args=args,
            actor_external_id=self.actor.external_id if self.actor else None,
            tags=self.tags or None,
            subjects=subjects,
            metrics=before_metrics if before_metrics else None,
        )

        decision = await self._api.evaluate(self.run_id, req)
        registered_tags = self._get_tool_tags(tool_name)

        self._emit_tool_decision(decision, tool_name, tool_tags or registered_tags)

        if self._enforce_mode == "shadow":
            return FAILOPEN_DECISION
        return decision

    async def after_tool(
        self,
        tool_name: str,
        args: Any,
        result: Any,
        duration_ms: float | None = None,
        error: Any = None,
        tool_tags: list[str] | None = None,
    ) -> Decision:
        """Call after a tool returns (or throws).

        Increments ``step_index``, emits ``tool.result``, records to history.
        """
        if self._state != "active":
            return FAILOPEN_DECISION

        tool_result = ToolResult(
            tool_name=tool_name,
            args=args,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )
        self._history.append(tool_result)

        metrics: dict[str, float | int] = {}
        bytes_in = _approx_bytes(args)
        if bytes_in is not None:
            metrics["bytes_in"] = bytes_in
        bytes_out = _approx_bytes(result)
        if bytes_out is not None:
            metrics["bytes_out"] = bytes_out
        if duration_ms is not None:
            metrics["duration_ms"] = duration_ms

        if self._metric_registry:
            from .metrics.types import AgentMetricInputToolAfter

            await self._metric_registry.run_phase(
                "tool.after",
                AgentMetricInputToolAfter(
                    tool_name=tool_name, args=args, run=self, result=result, error=error
                ),
                lambda key, value, unit=None: metrics.__setitem__(key, value),
            )

        subjects_after = None
        if self._subject_registry:
            from .tool import ToolMeta

            raw = await self._subject_registry.extract(
                tool=ToolMeta(tags=tool_tags),
                tool_name=tool_name,
                tool_args=args,
                run=self,
            )
            if raw:
                subjects_after = sanitise_subjects(raw)

        req = EvaluateAfterRequest(
            agent_id=self._agent_id or "",
            tool_name=tool_name,
            tool_tags=tool_tags,
            args=args,
            result=result,
            actor_external_id=self.actor.external_id if self.actor else None,
            tags=self.tags or None,
            subjects=subjects_after,
            metrics=metrics if metrics else None,
        )

        decision = (
            FAILOPEN_DECISION
            if self._enforce_mode == "off"
            else await self._api.evaluate(self.run_id, req)
        )

        registered_tags = self._get_tool_tags(tool_name)
        self._emit_tool_result(
            tool_name=tool_name,
            tool_tags=tool_tags or registered_tags,
            duration_ms=duration_ms,
            error=error,
        )

        self._step_index += 1
        return FAILOPEN_DECISION if self._enforce_mode == "shadow" else decision

    async def before_llm(
        self,
        messages: list[LLMMessage],
        meta: dict | None = None,
    ) -> list[LLMMessage]:
        """Call before sending messages to the LLM.

        Emits ``message.raw.created`` for each message; returns (possibly
        modified) messages — surface for future PII redaction.
        """
        if self._state != "active":
            return messages

        from uuid6 import uuid7

        for msg in messages:
            content_str = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content)
            )
            self._emit_event(
                kind="message.raw.created",
                data={
                    "messageId": str(uuid7()),
                    "role": msg.role,
                    "kind": _llm_role_to_kind(msg.role),
                    "content": content_str,
                    "contentTruncated": False,
                },
            )

        return messages

    async def after_llm(self, response: LLMResponse) -> LLMResponse:
        """Call after the LLM responds.

        Re-derives ``output_text``, accumulates token counts, emits
        ``llm.result`` and ``message.raw.created``.
        """
        if self._state != "active":
            return response

        derived = derive_output_text(response)
        response.output_text = derived or response.output_text

        in_tokens = response.usage.input_tokens if response.usage else None
        out_tokens = response.usage.output_tokens if response.usage else None

        if in_tokens is not None:
            self._pending_llm_tokens_in += in_tokens
        if out_tokens is not None:
            self._pending_llm_tokens_out += out_tokens

        if in_tokens is not None or out_tokens is not None:
            self._emit_event(
                kind="llm.result",
                data={
                    "model": {
                        "name": response.model.name,
                        "provider": response.model.provider,
                    },
                    "tokens": {"in": in_tokens or 0, "out": out_tokens or 0},
                    "messageCount": len(response.content),
                    "durationMs": response.duration_ms,
                },
            )

        from uuid6 import uuid7

        self._emit_event(
            kind="message.raw.created",
            data={
                "messageId": str(uuid7()),
                "role": "assistant",
                "kind": "output",
                "content": response.output_text or json.dumps(response.content),
                "contentTruncated": False,
            },
        )

        return response

    async def end(self, status: RunEndStatus = "success") -> None:
        """End this run. Idempotent — calling end() twice is a no-op."""
        if self._state == "ended":
            return
        self._state = "ended"

        if self._ttl_task is not None:
            self._ttl_task.cancel()
            self._ttl_task = None

        await self._api.end_run(self.run_id, self._agent_id, status)
        self._emit_event(
            kind="run.ended",
            data={"status": status, "totalSteps": self._step_index},
        )
        await self._bus.drain()

    # ------------------------------------------------------------------
    # Sync convenience wrappers
    # ------------------------------------------------------------------

    def before_tool_sync(
        self,
        tool_name: str,
        args: Any,
        tool_tags: list[str] | None = None,
    ) -> Decision:
        return run_sync(self.before_tool(tool_name, args, tool_tags))

    def after_tool_sync(
        self,
        tool_name: str,
        args: Any,
        result: Any,
        duration_ms: float | None = None,
        error: Any = None,
        tool_tags: list[str] | None = None,
    ) -> Decision:
        return run_sync(self.after_tool(tool_name, args, result, duration_ms, error, tool_tags))

    def before_llm_sync(
        self,
        messages: list[LLMMessage],
        meta: dict | None = None,
    ) -> list[LLMMessage]:
        return run_sync(self.before_llm(messages, meta))

    def after_llm_sync(self, response: LLMResponse) -> LLMResponse:
        return run_sync(self.after_llm(response))

    def end_sync(self, status: RunEndStatus = "success") -> None:
        run_sync(self.end(status))

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def is_ended(self) -> bool:
        return self._state == "ended"

    @property
    def current_step_index(self) -> int:
        return self._step_index

    def get_history(self) -> list[ToolResult]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tool_tags(self, tool_name: str) -> list[str] | None:
        if self._tools is None:
            return None
        for t in self._tools:
            if t.name == tool_name:
                return t.tags
        return None

    def _emit_event(self, kind: str, data: dict) -> None:
        if not self._agent_id:
            return
        event_dict = {
            "schema": "handlebar.audit.v1",
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            "runId": self.run_id,
            "sessionId": self.session_id or "",
            "actorExternalId": self.actor.external_id if self.actor else "", # TODO: set to None once nullables allowed
            "stepIndex": self._step_index,
            "data": data,
        }

        # Build validated event and pass to bus using the module-level cached adapter.
        try:
            event = _audit_event_adapter.validate_python(event_dict)
            self._bus.emit(self._agent_id, event)
            logger.debug("[Handlebar] Emitted event %s (run=%s)", kind, self.run_id)
        except Exception as exc:
            logger.warning("[Handlebar] Failed to build/emit event %s: %s", kind, exc)

    def _emit_run_started(self) -> None:
        data = {
            "agent": {"id": self._agent_id or None},
            "adapter": {"name": "core"},
        }
        if self.actor and self.actor.external_id:
            data["actor"] = {
                "externalId": self.actor.external_id,
                "metadata": self.actor.metadata or {},
            }
        self._emit_event(
            kind="run.started",
            data=data,
        )

    def _emit_tool_decision(
        self, decision: Decision, tool_name: str, tool_tags: list[str] | None
    ) -> None:
        self._emit_event(
            kind="tool.decision",
            data={
                "verdict": decision.verdict.value,
                "control": decision.control.value,
                "cause": decision.cause.model_dump(by_alias=True),
                "message": decision.message,
                "evaluatedRules": [r.model_dump(by_alias=True) for r in decision.evaluated_rules],
                "finalRuleId": decision.final_rule_id or "", # TODO: nulls vs undefineds in JS.
                "tool": {"name": tool_name, "categories": tool_tags or []},
            },
        )

    def _emit_tool_result(
        self,
        tool_name: str,
        tool_tags: list[str] | None,
        duration_ms: float | None,
        error: Any,
    ) -> None:
        error_data = None
        if isinstance(error, Exception):
            error_data = {"name": type(error).__name__, "message": str(error)}
        self._emit_event(
            kind="tool.result",
            data={
                "tool": {"name": tool_name, "categories": tool_tags or []},
                "outcome": "error" if error else "success",
                "durationMs": duration_ms or 0,
                # TODO: reimplement when None data is allowed
                #"error": error_data or "",
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _approx_bytes(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(json.dumps(value).encode("utf-8"))
    except Exception:
        return None


def _llm_role_to_kind(role: str) -> str:
    return {
        "user": "input",
        "assistant": "output",
        "tool": "tool_result",
    }.get(role, "observation")
