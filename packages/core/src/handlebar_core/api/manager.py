from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .types import ApiConfig
from ..types import Rule


class ApiManager:
    """
    Minimal API client to fetch governance rules for an agent.

    Behavior:
    - Uses api_endpoint and api_key from the provided config or environment variables:
      - HANDLEBAR_API_ENDPOINT
      - HANDLEBAR_API_KEY
    - If neither endpoint nor api key is provided, the client is disabled and returns None.
    - Expects the API to accept a JSON body: {"agentName": "<name>"} and return either a single
      Rule object or a list of Rule objects. Results are normalized to a list of Rule dicts.

    No external dependencies; uses urllib from the stdlib.
    """

    def __init__(self, config: Optional[ApiConfig] = None) -> None:
        cfg = config or {}
        self.api_endpoint: str = (
            cfg.get("api_endpoint")
            or os.getenv("HANDLEBAR_API_ENDPOINT")
            or "http://localhost:8000"
        )
        self.api_key: Optional[str] = cfg.get("api_key") or os.getenv("HANDLEBAR_API_KEY")
        self._use_api: bool = bool(self.api_endpoint or self.api_key)

    def headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def fetch_agent_rules(self, agent_name: str, timeout: float = 5.0) -> Optional[List[Rule]]:
        if not self._use_api:
            return None

        payload = json.dumps({"agentName": agent_name}).encode("utf-8")
        headers = {"content-type": "application/json", **self.headers()}
        req = urllib.request.Request(self.api_endpoint, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except (urllib.error.URLError, ValueError, TimeoutError):
            return None

        rules = self._normalize_rules(data)
        return rules if rules else None

    def _normalize_rules(self, data: Any) -> List[Rule]:
        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return []

        out: List[Rule] = []
        for item in items:
            if isinstance(item, dict) and self._is_rule_like(item):
                # Best-effort normalization; we trust upstream to provide correct shapes.
                out.append(item)  # type: ignore[arg-type]
        return out

    def _is_rule_like(self, obj: Dict[str, Any]) -> bool:
        # Minimal structural validation
        required_top = {"id", "policy_id", "priority", "when", "condition", "actions"}
        if not required_top.issubset(obj.keys()):
            return False
        if obj.get("when") not in ("pre", "post", "both"):
            return False
        if not isinstance(obj.get("priority"), int):
            return False
        if not isinstance(obj.get("actions"), list):
            return False
        return True


__all__ = ["ApiManager"]
