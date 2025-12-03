
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Literal

import httpx

from handlebar_core.types import GovernanceConfig, Rule, ToolMeta


class HandlebarApiError(Exception):
    pass


async def fetch_rules(
    *,
    api_key: str,
    base_url: str,
    app_name: str,
    org_id: Optional[str],
) -> List[Rule]:
    # TODO: get rules
    return []

async def upsert_agent(
    *,
    api_key: str,
    base_url: str,
    slug: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """
    PUT /v1/agent to upsert agent metadata and return agentId.

    Request body:
        {
          "slug": "...",
          "name": "...",          # optional
          "description": "...",   # optional
          "tags": ["..."]         # optional
        }

    Returns:
        agentId (str)

    TODO:
        - Move this function to `core`
        - Error handling with fallback
    """
    url = f"{base_url.rstrip('/')}/v1/agent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {"slug": slug}
    if name:
        payload["name"] = name
    if description:
        payload["description"] = description
    if tags:
        payload["tags"] = tags

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.put(url, json=payload, headers=headers)
        if resp.status_code not in (200, 201):
            raise HandlebarApiError(f"Failed to upsert agent: {resp.status_code} {resp.text}")
        data = resp.json()
        agent_id = data.get("id") or data.get("agentId")
        if not agent_id:
            raise HandlebarApiError("Agent upsert response missing 'id'/'agentId'")
        return str(agent_id)

async def fetch_governance_config(
    *,
    api_key: str,
    base_url: str,
    app_name: str,
    org_id: Optional[str],
    tools: List[ToolMeta],
    default_uncategorised: Literal["allow", "block"] = "allow",
    agent_id: Optional[str] = None,
) -> GovernanceConfig:
    rules = await fetch_rules(
        api_key=api_key,
        base_url=base_url,
        app_name=app_name,
        org_id=org_id,
    )
    cfg: GovernanceConfig = {
        "tools": tools,
        "rules": rules,
        "defaultUncategorised": default_uncategorised,
        "mode": "enforce",
        "verbose": False,
    }

    if agent_id:
            cfg["agentId"] = agent_id

    return cfg
