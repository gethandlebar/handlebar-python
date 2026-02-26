"""HTTP sink with background flush thread — mirrors packages/core/src/sinks/http.ts."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from .types import Sink

if TYPE_CHECKING:
    from ..schema.events import AuditEvent

logger = logging.getLogger("handlebar")

_DEFAULTS = {
    "queue_depth": 500,
    "flush_interval_ms": 1000,
    "max_batch_size": 50,
    "flush_timeout_ms": 5000,
    "max_retries": 3,
    "retry_base_ms": 500,
    "retry_cap_ms": 10_000,
}


class _QueuedEvent:
    __slots__ = ("agent_id", "event")

    def __init__(self, agent_id: str, event: "AuditEvent") -> None:
        self.agent_id = agent_id
        self.event = event


class HttpSink(Sink):
    """Buffers events and flushes them to the Handlebar HTTP API in background.

    Uses a daemon thread with a recurring timer so it works transparently in
    both synchronous and async caller contexts.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None,
        queue_depth: int = _DEFAULTS["queue_depth"],
        flush_interval_ms: int = _DEFAULTS["flush_interval_ms"],
        max_batch_size: int = _DEFAULTS["max_batch_size"],
        flush_timeout_ms: int = _DEFAULTS["flush_timeout_ms"],
        _retry_base_ms: int = _DEFAULTS["retry_base_ms"],
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._queue_depth = queue_depth
        self._flush_interval_s = flush_interval_ms / 1000
        self._max_batch_size = max_batch_size
        self._flush_timeout_s = flush_timeout_ms / 1000
        self._retry_base_ms = _retry_base_ms

        self._queue: list[_QueuedEvent] = []
        self._lock = threading.Lock()
        self._closed = False
        self._timer: threading.Timer | None = None
        self._flush_event = threading.Event()

    # ------------------------------------------------------------------
    # Sink interface
    # ------------------------------------------------------------------

    async def init(self) -> None:
        self._schedule()

    def write_batch(self, agent_id: str, events: list["AuditEvent"]) -> None:
        with self._lock:
            for event in events:
                if self._closed:
                    return
                if len(self._queue) >= self._queue_depth:
                    self._queue.pop(0)  # drop oldest
                self._queue.append(_QueuedEvent(agent_id, event))

    async def drain(self) -> None:
        """Flush the queue and wait up to ``flush_timeout_s``."""
        done = asyncio.Event()

        def _flush_and_signal() -> None:
            self._flush_sync()
            done.set()

        t = threading.Thread(target=_flush_and_signal, daemon=True)
        t.start()
        try:
            await asyncio.wait_for(done.wait(), timeout=self._flush_timeout_s)
        except asyncio.TimeoutError:
            pass

    async def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        await self.drain()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule(self) -> None:
        if self._closed:
            return
        self._timer = threading.Timer(self._flush_interval_s, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        self._flush_sync()
        self._schedule()

    def _flush_sync(self) -> None:
        with self._lock:
            if not self._queue:
                return
            snapshot = self._queue[:]
            self._queue.clear()

        # Group by agent_id.
        by_agent: dict[str, list[Any]] = {}
        for item in snapshot:
            by_agent.setdefault(item.agent_id, []).append(item.event)

        import httpx

        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._endpoint}/v1/runs/events"

        for agent_id, events in by_agent.items():
            for i in range(0, len(events), self._max_batch_size):
                batch = events[i : i + self._max_batch_size]
                self._send_batch_sync(
                    url=url,
                    headers=headers,
                    agent_id=agent_id,
                    events=batch,
                )

    def _send_batch_sync(
        self,
        url: str,
        headers: dict[str, str],
        agent_id: str,
        events: list[Any],
    ) -> None:
        import httpx
        import json

        # Serialise events using Pydantic's JSON encoder (camelCase aliases).
        events_payload = []
        for e in events:
            try:
                events_payload.append(e.model_dump(by_alias=True, mode="json"))
            except Exception:
                events_payload.append(str(e))

        body = json.dumps({"agentId": agent_id, "events": events_payload})

        attempt = 0
        max_retries: int = _DEFAULTS["max_retries"]
        retry_cap_ms: int = _DEFAULTS["retry_cap_ms"]

        while attempt <= max_retries:
            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.post(url, content=body, headers=headers)
                if resp.is_success:
                    return
                if 400 <= resp.status_code < 500:
                    logger.error(
                        "[Handlebar] HttpSink: non-retryable %s from %s", resp.status_code, url
                    )
                    return
                raise Exception(f"HTTP {resp.status_code}")
            except Exception as exc:
                if attempt >= max_retries:
                    logger.error(
                        "[Handlebar] HttpSink: giving up after %d attempts: %s",
                        attempt + 1,
                        exc,
                    )
                    return
                backoff_s = min(self._retry_base_ms * (2**attempt), retry_cap_ms) / 1000
                time.sleep(backoff_s)
                attempt += 1


def create_http_sink(
    endpoint: str,
    api_key: str | None,
    *,
    queue_depth: int = _DEFAULTS["queue_depth"],
    flush_interval_ms: int = _DEFAULTS["flush_interval_ms"],
    max_batch_size: int = _DEFAULTS["max_batch_size"],
    flush_timeout_ms: int = _DEFAULTS["flush_timeout_ms"],
    _retry_base_ms: int = _DEFAULTS["retry_base_ms"],
) -> Sink:
    """Factory — equivalent to JS ``createHttpSink``."""
    return HttpSink(
        endpoint=endpoint,
        api_key=api_key,
        queue_depth=queue_depth,
        flush_interval_ms=flush_interval_ms,
        max_batch_size=max_batch_size,
        flush_timeout_ms=flush_timeout_ms,
        _retry_base_ms=_retry_base_ms,
    )
