"""handlebar-core — framework-agnostic governance engine for AI agents."""

from .budget_manager import BudgetGrant, BudgetManager
from .client import (
    Handlebar,
    HandlebarClient,
    HandlebarClientConfig,
    get_current_run,
    with_run,
)
from .metrics import (
    AgentMetricCollector,
    AgentMetricHook,
    AgentMetricHookPhase,
    AgentMetricHookRegistry,
    MetricInfo,
    approx_bytes,
    approx_records,
)
from .run import Run, RunInternalConfig
from .schema import (
    AuditEvent,
    Decision,
    DecisionCause,
    InbuiltAgentMetricKind,
    LlmResultEvent,
    MessageRawCreatedEvent,
    Rule,
    RuleCondition,
    RuleEval,
    RunControl,
    RunEndedEvent,
    RunStartedEvent,
    ToolDecisionEvent,
    ToolResultEvent,
    Verdict,
)
from .sinks import Sink, SinkBus, create_console_sink, create_http_sink
from .subjects import SubjectExtractor, SubjectRef, SubjectRegistry, sanitise_subjects
from .tokens import tokenise_count
from .tool import ToolMeta, define_tool, wrap_tool
from .types import (
    Actor,
    AgentDescriptor,
    ConsoleSinkConfig,
    EnforceMode,
    FAILCLOSED_DECISION,
    FAILOPEN_DECISION,
    HandlebarConfig,
    HttpSinkConfig,
    InsertableTool,
    LLMMessage,
    LLMMessagePart,
    LLMResponse,
    LLMResponsePart,
    ModelInfo,
    RunConfig,
    RunEndStatus,
    SinkConfig,
    TokenUsage,
    Tool,
    ToolCall,
    ToolResult,
    derive_output_text,
)
from .utils import generate_slug

__all__ = [
    # client
    "Handlebar",
    "HandlebarClient",
    "HandlebarClientConfig",
    "get_current_run",
    "with_run",
    # run
    "Run",
    "RunInternalConfig",
    # types
    "Actor",
    "AgentDescriptor",
    "ConsoleSinkConfig",
    "Decision",
    "DecisionCause",
    "EnforceMode",
    "FAILCLOSED_DECISION",
    "FAILOPEN_DECISION",
    "HandlebarConfig",
    "HttpSinkConfig",
    "InsertableTool",
    "LLMMessage",
    "LLMMessagePart",
    "LLMResponse",
    "LLMResponsePart",
    "ModelInfo",
    "RunConfig",
    "RunEndStatus",
    "SinkConfig",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolResult",
    "derive_output_text",
    # schema / governance
    "AuditEvent",
    "InbuiltAgentMetricKind",
    "LlmResultEvent",
    "MessageRawCreatedEvent",
    "Rule",
    "RuleCondition",
    "RuleEval",
    "RunControl",
    "RunEndedEvent",
    "RunStartedEvent",
    "ToolDecisionEvent",
    "ToolResultEvent",
    "Verdict",
    # sinks
    "Sink",
    "SinkBus",
    "create_console_sink",
    "create_http_sink",
    # metrics
    "AgentMetricCollector",
    "AgentMetricHook",
    "AgentMetricHookPhase",
    "AgentMetricHookRegistry",
    "MetricInfo",
    "approx_bytes",
    "approx_records",
    # subjects
    "SubjectExtractor",
    "SubjectRef",
    "SubjectRegistry",
    "sanitise_subjects",
    # tools
    "ToolMeta",
    "define_tool",
    "wrap_tool",
    # budget
    "BudgetGrant",
    "BudgetManager",
    # utils
    "generate_slug",
    "tokenise_count",
]
