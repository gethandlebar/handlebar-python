"""Framework-agnostic type definitions for the Handlebar core SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .schema.enduser import EndUserConfig
from .schema.governance import Decision, DecisionCause, RuleEval, RunControl, Verdict

if TYPE_CHECKING:
    pass

# Re-export decision types from schema for convenience.
__all__ = [
    "Decision",
    "DecisionCause",
    "RuleEval",
    "RunControl",
    "Verdict",
    # local
    "Actor",
    "EnforceMode",
    "Tool",
    "InsertableTool",
    "ToolCall",
    "ToolResult",
    "HttpSinkConfig",
    "ConsoleSinkConfig",
    "SinkConfig",
    "HandlebarConfig",
    "RunConfig",
    "RunEndStatus",
    "ModelInfo",
    "TokenUsage",
    "LLMMessagePart",
    "LLMMessage",
    "LLMResponsePart",
    "LLMResponse",
    "FAILOPEN_DECISION",
    "FAILCLOSED_DECISION",
    "derive_output_text",
]

# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

# Forward-compatible term for the human/system/agent the run acts on behalf of.
Actor = EndUserConfig


# ---------------------------------------------------------------------------
# Enforce mode
# ---------------------------------------------------------------------------

EnforceMode = Literal["enforce", "shadow", "off"]

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class Tool:
    """Framework-agnostic tool descriptor."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.tags = tags

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, tags={self.tags!r})"


class InsertableTool:
    """Tool shape the Handlebar server expects."""

    def __init__(
        self,
        key: str,
        name: str,
        version: int,
        kind: Literal["function", "mcp"],
        description: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.key = key
        self.name = name
        self.version = version
        self.kind = kind
        self.description = description
        self.metadata = metadata

    def to_dict(self) -> dict:
        d: dict = {"key": self.key, "name": self.name, "version": self.version, "kind": self.kind}
        if self.description:
            d["description"] = self.description
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class ToolCall:
    def __init__(self, tool_name: str, args: object) -> None:
        self.tool_name = tool_name
        self.args = args


class ToolResult:
    def __init__(
        self,
        tool_name: str,
        args: object,
        result: object,
        error: object = None,
        duration_ms: float | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.args = args
        self.result = result
        self.error = error
        self.duration_ms = duration_ms


# ---------------------------------------------------------------------------
# Sink configuration
# ---------------------------------------------------------------------------


class HttpSinkConfig:
    type: Literal["http"] = "http"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        queue_depth: int = 500,
        flush_interval_ms: int = 1000,
        max_batch_size: int = 50,
        flush_timeout_ms: int = 5000,
    ) -> None:
        self.type: Literal["http"] = "http"
        self.endpoint = endpoint
        self.api_key = api_key
        self.queue_depth = queue_depth
        self.flush_interval_ms = flush_interval_ms
        self.max_batch_size = max_batch_size
        self.flush_timeout_ms = flush_timeout_ms


class ConsoleSinkConfig:
    def __init__(self, format: Literal["pretty", "json"] = "json") -> None:
        self.type: Literal["console"] = "console"
        self.format = format


SinkConfig = HttpSinkConfig | ConsoleSinkConfig


# ---------------------------------------------------------------------------
# HandlebarConfig
# ---------------------------------------------------------------------------


class AgentDescriptor:
    def __init__(
        self,
        slug: str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.slug = slug
        self.name = name
        self.description = description
        self.tags = tags


class HandlebarConfig:
    def __init__(
        self,
        agent: AgentDescriptor,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        fail_closed: bool = False,
        enforce_mode: EnforceMode = "enforce",
        sinks: list[SinkConfig] | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self.agent = agent
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.fail_closed = fail_closed
        self.enforce_mode = enforce_mode
        self.sinks = sinks
        self.tools = tools


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class ModelInfo:
    def __init__(self, name: str, provider: str | None = None) -> None:
        self.name = name
        self.provider = provider

    def to_dict(self) -> dict:
        d: dict = {"name": self.name}
        if self.provider:
            d["provider"] = self.provider
        return d


class RunConfig:
    def __init__(
        self,
        run_id: str,
        session_id: str | None = None,
        actor: Actor | None = None,
        model: ModelInfo | None = None,
        tags: dict[str, str] | None = None,
        run_ttl_ms: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.actor = actor
        self.model = model
        self.tags = tags or {}
        self.run_ttl_ms = run_ttl_ms


# ---------------------------------------------------------------------------
# Decision defaults
# ---------------------------------------------------------------------------

FAILOPEN_DECISION = Decision.model_validate(
    {
        "verdict": "ALLOW",
        "control": "CONTINUE",
        "cause": {"kind": "ALLOW"},
        "message": "API unavailable; failing open",
        "evaluatedRules": [],
    }
)

FAILCLOSED_DECISION = Decision.model_validate(
    {
        "verdict": "BLOCK",
        "control": "TERMINATE",
        "cause": {"kind": "LOCKDOWN"},
        "message": "API unavailable; failing closed",
        "evaluatedRules": [],
    }
)

# ---------------------------------------------------------------------------
# Run end status
# ---------------------------------------------------------------------------

RunEndStatus = Literal["success", "error", "timeout", "interrupted"]

# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------


class TokenUsage:
    def __init__(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


# Provider-agnostic message part shapes (input to the LLM).
LLMMessagePart = (
    dict  # {type: "text", text: str}
    # | {type: "tool_use", toolUseId: str, toolName: str, input: Any}
    # | {type: "tool_result", toolUseId: str, content: str | list}
    # | {type: "thinking", thinking: str}
    # Using plain dict preserves flexibility across providers.
)


class LLMMessage:
    def __init__(
        self,
        role: Literal["system", "user", "assistant", "tool"],
        content: str | list[LLMMessagePart],
    ) -> None:
        self.role = role
        self.content = content


# Provider-agnostic response part shapes (output from the LLM).
LLMResponsePart = dict  # {type: "text"/"tool_call"/"refusal", ...}


class LLMResponse:
    def __init__(
        self,
        content: list[LLMResponsePart],
        model: ModelInfo,
        output_text: str | None = None,
        usage: TokenUsage | None = None,
        duration_ms: float | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.output_text = output_text
        self.usage = usage
        self.duration_ms = duration_ms


def derive_output_text(response: LLMResponse) -> str:
    """Derive outputText from LLMResponse.content text parts."""
    parts = [p["text"] for p in response.content if p.get("type") == "text" and "text" in p]
    return "".join(parts)
