from __future__ import annotations
from handlebar_core.telemetry.types import AuditEvent, AuditKind
from handlebar_core.types import JSONValue, GovernanceDecision
from typing import Literal, List, Dict, Any, Optional

import contextvars
import datetime as dt
import json
import logging
import os
import threading
import typing as t
import urllib.error
import urllib.request
import queue

from ..types import RunContext

RunId = str
T = t.TypeVar("T")

logger = logging.getLogger("handlebar. " + __name__)


# ==== Run context ============================================================

_run_ctx: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "handlebar_run_ctx", default=None
)


def with_run_context(ctx: RunContext, fn: t.Callable[[], T]) -> T:
    token = _run_ctx.set(ctx)
    try:
        return fn()
    finally:
        _run_ctx.reset(token)


def get_run_context() -> RunContext | None:
    return _run_ctx.get()


def push_run_context(ctx: RunContext) -> contextvars.Token[RunContext | None]:
    return _run_ctx.set(ctx)


def pop_run_context(token: contextvars.Token[RunContext | None]) -> None:
    _run_ctx.reset(token)


def inc_step() -> None:
    ctx = _run_ctx.get()
    if ctx is not None:
        ctx["stepIndex"] = int(ctx.get("stepIndex", 0)) + 1


class AuditSink:
    def init(self) -> None:  # optional
        ...

    def write(self, event: AuditEvent) -> None:
        raise NotImplementedError

    def flush(self) -> None:  # optional
        ...

    def close(self) -> None:  # optional
        ...


class AuditBus:
    def __init__(self) -> None:
        self._sinks: list[AuditSink] = []
        self._closed = False

    def use(self, *sinks: AuditSink) -> None:
        for s in sinks:
            try:
                s.init()
            except Exception as e:
                logger.error("Audit sink init error: %s", e)
        self._sinks.extend(sinks)

    def emit(self, event: AuditEvent) -> None:
        if self._closed:
            logger.debug("Not emitting event. Bus is closed")
            return

        for s in list(self._sinks):
            try:
                s.write(event)
            except Exception as e:
                logger.error("Audit sink write error: %s", e)

    def shutdown(self) -> None:
        self._closed = True
        for s in list(self._sinks):
            try:
                s.flush()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass


def _json_default(obj: t.Any) -> t.Any:
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        if isinstance(obj, dt.datetime):
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=dt.timezone.utc)
            return obj.astimezone(dt.timezone.utc).isoformat()
        return obj.isoformat()
    return str(obj)


class ConsoleSink(AuditSink):
    def __init__(self, mode: t.Literal["json", "pretty"] = "json") -> None:
        self.mode = mode

    def write(self, event: AuditEvent) -> None:
        if self.mode == "json":
            print(json.dumps(event, default=_json_default, separators=(",", ":")))
        else:
            kind = event.get("kind", "-")
            run_id = event.get("runId", "-")
            step = event.get("stepIndex", "-")
            print(f"[{kind}] run={run_id} step={step}")


class FileSink(AuditSink):
    def __init__(self, path: str) -> None:
        self._path = path
        self._fh: t.TextIO | None = None

    def init(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")

    def write(self, event: AuditEvent) -> None:
        if not self._fh:
            return
        self._fh.write(json.dumps(event, default=_json_default) + "\n")

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None


class HttpSink(AuditSink):
    """
    HTTP sink with a single background worker and a queue.

    - Non-blocking for emitters.
    - Bounded queue to avoid unbounded memory/thread growth.
    - Best-effort: drops events if the queue is full.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Dict[str, str] | None = None,
        timeout: float = 5.0,
        max_queue_size: int = 10000,
    ) -> None:
        self._endpoint = endpoint
        self._headers = {"content-type": "application/json", **(headers or {})}
        self._timeout = timeout
        logger.debug(f"Connected http sink with ${self._endpoint}")
        self._queue: "queue.Queue[AuditEvent]" = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def init(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _post(self, payload: str) -> None:
        req = urllib.request.Request(
            self._endpoint,
            data=payload.encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
        except urllib.error.URLError as e:
            logger.error("Audit HTTP sink error: %s", e)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                agent_id = event.get("agentId")
                if not agent_id:
                    logger.warning("Audit HTTP sink: event missing agentId; dropping")
                else:
                    del event["agentId"]
                    payload_obj = {
                        "agentId": agent_id,
                        "events": [event],
                    }
                    payload = json.dumps(
                        payload_obj,
                        default=_json_default,
                        separators=(",", ":"),
                    )
                    self._post(payload)
            except Exception as e:  # noqa: BLE001
                logger.error("Audit HTTP worker error: %s", e)
            finally:
                self._queue.task_done()

    def write(self, event: AuditEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Audit HTTP queue full; dropping event")

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# ==== Telemetry singleton ====================================================

class TelemetryOptions(t.TypedDict, total=False):
    api_key: str
    endpoint: str
    default_sinks: list[t.Literal["http", "console", "file"]]
    file_path: str
    headers: Dict[str, str]


class _TelemetrySingleton:
    def __init__(self) -> None:
        self._bus: AuditBus | None = None
        self._inited = False

    def init(self, opts: TelemetryOptions | None = None) -> None:
        if self._inited:
            return
        self._inited = True
        self._bus = AuditBus()

        endpoint = (opts or {}).get("endpoint") or os.getenv("HANDLEBAR_ENDPOINT")
        api_key = (opts or {}).get("api_key") or os.getenv("HANDLEBAR_API_KEY")

        defaults = (opts or {}).get("default_sinks")
        if not defaults:
            defaults = ["http"] if endpoint else ["console"]

        sinks: list[AuditSink] = []
        if "console" in defaults:
            sinks.append(ConsoleSink("json"))

        if "file" in defaults and (opts or {}).get("file_path"):
            sinks.append(FileSink((opts or {})["file_path"]))  # type: ignore[index]

        if "http" in defaults and endpoint:
            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            headers.update((opts or {}).get("headers", {}))
            sinks.append(HttpSink(endpoint, headers=headers))

        self._bus.use(*sinks)

    def bus(self) -> AuditBus | None:
        if not self._inited:
            self.init()
        return self._bus

    def add_sink(self, sink: AuditSink) -> None:
        if not self._inited or not self._bus:
            self.init()
        if self._bus:
            self._bus.use(sink)


Telemetry = _TelemetrySingleton()


# ==== Emission ==============================================================

def emit(kind: AuditKind, data: Dict[str, Any], extras: Optional[Dict[str, Any]] = None) -> None:
    ctx = get_run_context()
    if not ctx or not ctx.get("runId"):
        return

    event: AuditEvent = {
        "schema": "handlebar.audit.v1",
        "kind": kind,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runId": ctx["runId"],
        "stepIndex": ctx.get("stepIndex"),
        "decisionId": ctx.get("decisionId", ""),
        "otel": ctx.get("otel", {}),
        "agentId": ctx.get("agentId"),
        "data": data,
        **(extras or {}),
    }  # type: ignore[assignment]
    bus = Telemetry.bus()
    if bus:
        bus.emit(event)


__all__ = [
    "with_run_context",
    "get_run_context",
    "push_run_context",
    "pop_run_context",
    "inc_step",
    "AuditSink",
    "AuditBus",
    "ConsoleSink",
    "FileSink",
    "HttpSink",
    "Telemetry",
    "TelemetryOptions",
    "emit",
]
