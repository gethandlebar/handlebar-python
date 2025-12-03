from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
from typing_extensions import NotRequired, TypedDict

from ..types import Rule


class ApiConfig(TypedDict, total=False):
    api_endpoint: NotRequired[str]
    api_key: NotRequired[str]


@runtime_checkable
class ApiManagerProtocol(Protocol):
    def headers(self) -> dict[str, str]: ...
    def fetch_agent_rules(self, agent_name: str) -> Optional[list[Rule]]: ...


__all__ = ["ApiConfig", "ApiManagerProtocol"]
