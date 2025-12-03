from __future__ import annotations

import importlib
from typing import Any, Dict, List

# Public surface re-exports (lazy via __getattr__)
_EXPORTS: Dict[str, str] = {
    # core engine
    "GovernanceEngine": "handlebar_core.engine",
    # rule builders
    "and_": "handlebar_core.rules",
    "or_": "handlebar_core.rules",
    "not_": "handlebar_core.rules",
    "tool_name": "handlebar_core.rules",
    "tool_tag": "handlebar_core.rules",
    "exec_time": "handlebar_core.rules",
    "sequence": "handlebar_core.rules",
    "max_calls": "handlebar_core.rules",
    "custom": "handlebar_core.rules",
    "block": "handlebar_core.rules",
    "allow": "handlebar_core.rules",
    "rule": "handlebar_core.rules",
    "config_to_rule": "handlebar_core.rules",
    # types
    "Tool": "handlebar_core.types",
    "ToolMeta": "handlebar_core.types",
    "ToolCall": "handlebar_core.types",
    "ToolResult": "handlebar_core.types",
    "CustomCheck": "handlebar_core.types",
    "GovernanceConfig": "handlebar_core.types",
    "RunContext": "handlebar_core.types",
    "Id": "handlebar_core.types",
    "ISO8601": "handlebar_core.types",
    # audit
    "emit": "handlebar_core.telemetry",
    "with_run_context": "handlebar_core.telemetry",
    "Telemetry": "handlebar_core.telemetry",
    # API
    "ApiManager": "handlebar_core.api.manager",
    "ApiConfig": "handlebar_core.api.types",
}

__all__: List[str] = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'handlebar_core' has no attribute '{name}'")
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise AttributeError(
            f"attribute '{name}' not found in lazily imported module '{module_path}'"
        ) from exc


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
