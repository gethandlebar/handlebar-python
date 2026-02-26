"""API manager — mirrors packages/core/src/api/manager.ts."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from ..schema.governance import Decision
from ..subjects import SubjectRef
from ..types import (
    FAILCLOSED_DECISION,
    FAILOPEN_DECISION,
    Actor,
    ModelInfo,
    RunEndStatus,
    Tool,
)
from ..utils import tool_to_insertable_tool

logger = logging.getLogger("handlebar")

DEFAULT_ENDPOINT = "https://api.gethandlebar.com"

_RETRY_DEFAULTS = {
    "max_retries": 3,
    "base_ms": 200,
    "cap_ms": 5_000,
}


@dataclass
class LockdownStatus:
    active: bool
    reason: str | None = None
    until: int | None = None  # Unix timestamp (ms), None = indefinite


@dataclass
class _BaseEvaluateRequest:
    agent_id: str
    tool_name: str
    tool_tags: list[str] | None
    args: Any
    actor_external_id: str | None
    tags: dict[str, str] | None
    subjects: list[SubjectRef] | None
    metrics: dict[str, float | int] | None


@dataclass
class EvaluateBeforeRequest(_BaseEvaluateRequest):
    phase: str = "tool.before"


@dataclass
class EvaluateAfterRequest(_BaseEvaluateRequest):
    phase: str = "tool.after"
    result: Any = None


EvaluateRequest = EvaluateBeforeRequest | EvaluateAfterRequest


class ApiManager:
    """Communicates with the Handlebar API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        fail_closed: bool = False,
        _retry_base_ms: int = _RETRY_DEFAULTS["base_ms"],
    ) -> None:
        self._endpoint = (
            api_endpoint
            or os.environ.get("HANDLEBAR_API_ENDPOINT")
            or DEFAULT_ENDPOINT
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("HANDLEBAR_API_KEY")
        self._fail_closed = fail_closed
        self._retry_base_ms = _retry_base_ms
        self._active = bool(self._api_key)

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    async def upsert_agent(
        self,
        slug: str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        tools: list[Tool] | None = None,
    ) -> str | None:
        """Upsert agent and (optionally) register tools atomically.

        Returns the server-assigned agentId or ``None`` if inactive / error.
        """
        if not self._active:
            return None

        url = self._url("/v1/agents")
        body: dict[str, Any] = {"slug": slug}
        if name:
            body["name"] = name
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "tags": t.tags}
                for t in tools
            ]

        try:
            resp = await self._post(url, body)
            if not resp.is_success:
                logger.error("[Handlebar] Agent upsert failed: %s", resp.status_code)
                return None
            data = resp.json()
            return data["agentId"]
        except Exception as exc:
            logger.error("[Handlebar] Agent upsert error: %s", exc)
            return None

    async def register_tools(self, agent_id: str, tools: list[Tool]) -> bool:
        """Register or update tools on an existing agent."""
        if not self._active or not tools:
            return True

        url = self._url(f"/v1/agents/{agent_id}/tools")
        body = {"tools": [tool_to_insertable_tool(t) for t in tools]}

        try:
            resp = await self._put(url, body)
            if not resp.is_success:
                logger.error("[Handlebar] Tool registration failed: %s", resp.status_code)
                return False
            return True
        except Exception as exc:
            logger.error("[Handlebar] Tool registration error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def start_run(
        self,
        run_id: str,
        agent_id: str,
        session_id: str | None = None,
        actor: Actor | None = None,
        model: ModelInfo | None = None,
    ) -> LockdownStatus:
        """Start a run on the server. Returns lockdown status."""
        if not self._active:
            return LockdownStatus(active=False)

        url = self._url(f"/v1/runs/{run_id}/start")
        body: dict[str, Any] = {"agentId": agent_id}
        if session_id:
            body["sessionId"] = session_id
        if actor:
            body["actor"] = {"externalId": actor.external_id, **({"metadata": actor.metadata} if actor.metadata else {})}
        if model:
            body["model"] = model.to_dict()

        try:
            resp = await self._post(url, body)
            if not resp.is_success:
                logger.warning(
                    "[Handlebar] Run start returned %s; assuming no lockdown", resp.status_code
                )
                return LockdownStatus(active=False)
            data = resp.json()
            ld = data.get("lockdown", {})
            return LockdownStatus(
                active=ld.get("active", False),
                reason=ld.get("reason"),
                until=ld.get("until_ts"),
            )
        except Exception as exc:
            logger.error("[Handlebar] Run start error: %s", exc)
            return LockdownStatus(active=False)

    async def end_run(
        self, run_id: str, agent_id: str | None, status: RunEndStatus
    ) -> None:
        if not self._active or agent_id is None:
            return

        url = self._url(f"/v1/runs/{run_id}/end")
        body = {"agentId": agent_id, "status": status}

        try:
            resp = await self._post_with_retry(url, body)
            if not resp.is_success:
                logger.warning("[Handlebar] Run end returned %s", resp.status_code)
        except Exception as exc:
            logger.error("[Handlebar] Run end error: %s", exc)

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, run_id: str, req: EvaluateRequest) -> Decision:
        """Evaluate a tool call against active rules. Returns a Decision."""
        if not self._active:
            return self._fail_decision()

        url = self._url(f"/v1/runs/{run_id}/evaluate")
        body = self._evaluate_request_to_dict(req)

        try:
            resp = await self._post_with_retry(url, body)
            if not resp.is_success:
                logger.error("[Handlebar] Evaluate returned %s", resp.status_code)
                return self._fail_decision()
            raw = resp.json()
            return Decision.model_validate(raw)
        except Exception as exc:
            logger.error("[Handlebar] Evaluate error: %s", exc)
            return self._fail_decision()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def _fail_decision(self) -> Decision:
        return FAILCLOSED_DECISION if self._fail_closed else FAILOPEN_DECISION

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _post(self, url: str, body: Any) -> httpx.Response:
        import json

        async with httpx.AsyncClient() as client:
            return await client.post(
                url,
                content=json.dumps(body),
                headers=self._headers(),
            )

    async def _put(self, url: str, body: Any) -> httpx.Response:
        import json

        async with httpx.AsyncClient() as client:
            return await client.put(
                url,
                content=json.dumps(body),
                headers=self._headers(),
            )

    async def _post_with_retry(self, url: str, body: Any) -> httpx.Response:
        import json

        base_ms = self._retry_base_ms
        max_retries: int = _RETRY_DEFAULTS["max_retries"]
        cap_ms: int = _RETRY_DEFAULTS["cap_ms"]
        attempt = 0

        while True:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        url,
                        content=json.dumps(body),
                        headers=self._headers(),
                    )
                # Don't retry on 4xx.
                if resp.is_success or (400 <= resp.status_code < 500):
                    return resp
                raise Exception(f"HTTP {resp.status_code}")
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                backoff_s = min(base_ms * (2**attempt), cap_ms) / 1000
                await asyncio.sleep(backoff_s)
                attempt += 1

    @staticmethod
    def _evaluate_request_to_dict(req: EvaluateRequest) -> dict:
        body: dict[str, Any] = {
            "phase": req.phase,
            "agentId": req.agent_id,
            "tool": {"name": req.tool_name},
            "args": req.args,
        }
        if req.tool_tags:
            body["tool"]["tags"] = req.tool_tags
        if req.actor_external_id:
            body["actor"] = {"externalId": req.actor_external_id}
        if req.tags:
            body["tags"] = req.tags
        if req.subjects:
            body["subjects"] = [s.to_dict() for s in req.subjects]
        if req.metrics:
            body["metrics"] = req.metrics
        if isinstance(req, EvaluateAfterRequest) and req.result is not None:
            body["result"] = req.result
        return body
