"""HandlebarClient — mirrors packages/core/src/client.ts."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from ._sync import run_sync
from .api.manager import ApiManager, resolve_api_endpoint
from .metrics.hooks import AgentMetricHookRegistry
from .metrics.types import AgentMetricHook
from .run import Run, RunInternalConfig
from .sinks.bus import SinkBus
from .sinks.console import create_console_sink
from .sinks.http import create_http_sink
from .subjects import SubjectRegistry
from .types import HandlebarConfig, RunConfig, SinkConfig, Tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger("handlebar")

# ---------------------------------------------------------------------------
# Context var for implicit run propagation (mirrors AsyncLocalStorage).
# ---------------------------------------------------------------------------

_run_context: ContextVar[Run | None] = ContextVar("_run_context", default=None)


async def with_run(run: Run, fn) -> Any:
    """Bind ``run`` to the current async context, then await ``fn()``."""
    token = _run_context.set(run)
    try:
        return await fn()
    finally:
        _run_context.reset(token)


def get_current_run() -> Run | None:
    """Return the run bound to the current async context, or None."""
    return _run_context.get()


# ---------------------------------------------------------------------------
# HandlebarClientConfig
# ---------------------------------------------------------------------------


class HandlebarClientConfig(HandlebarConfig):
    """HandlebarConfig extended with optional metric hooks and subject registry."""

    def __init__(
        self,
        *,
        agent,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        fail_closed: bool = False,
        enforce_mode: str = "enforce",
        sinks: list[SinkConfig] | None = None,
        tools: list[Tool] | None = None,
        metric_hooks: list[AgentMetricHook] | None = None,
        subject_registry: SubjectRegistry | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            api_key=api_key,
            api_endpoint=api_endpoint,
            fail_closed=fail_closed,
            enforce_mode=enforce_mode,  # type: ignore[arg-type]
            sinks=sinks,
            tools=tools,
        )
        self.metric_hooks = metric_hooks
        self.subject_registry = subject_registry


# ---------------------------------------------------------------------------
# HandlebarClient
# ---------------------------------------------------------------------------


class HandlebarClient:
    """Governance client. Use ``HandlebarClient.init()`` — do not call directly."""

    def __init__(self, config: HandlebarClientConfig) -> None:
        self._config = config
        self._api = ApiManager(
            api_key=config.api_key,
            api_endpoint=config.api_endpoint,
            fail_closed=config.fail_closed,
        )
        self._bus = SinkBus()
        self._agent_id: str | None = None
        self._tools: list[Tool] | None = config.tools
        self._active_runs: dict[str, Run] = {}
        self._init_task: Any = None  # asyncio.Task

        metric_registry: AgentMetricHookRegistry | None = None
        if config.metric_hooks:
            metric_registry = AgentMetricHookRegistry()
            for hook in config.metric_hooks:
                metric_registry.register_hook(hook)
        self._metric_registry = metric_registry
        self._subject_registry = config.subject_registry

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def init(cls, config: HandlebarClientConfig) -> HandlebarClient:
        """Async factory — equivalent to JS ``HandlebarClient.init(config)``."""
        client = cls(config)
        await client._init_sinks(config.sinks)
        import asyncio

        client._init_task = asyncio.ensure_future(client._init_agent(config))
        client._init_task.add_done_callback(
            lambda t: (
                logger.error("[Handlebar] Async init error: %s", t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )
        return client

    @classmethod
    def init_sync(cls, config: HandlebarClientConfig) -> HandlebarClient:
        """Sync factory — wraps ``init()`` for use in synchronous contexts."""
        return run_sync(cls.init(config))

    # ------------------------------------------------------------------
    # Ready
    # ------------------------------------------------------------------

    async def ready(self) -> None:
        """Await completion of the async agent registration."""
        if self._init_task:
            await self._init_task

    def ready_sync(self) -> None:
        run_sync(self.ready())

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    async def register_tools(self, tools: list[Tool]) -> None:
        """Register tools added after init."""
        await self.ready()
        self._tools = tools
        if self._agent_id:
            await self._api.register_tools(self._agent_id, tools)

    def register_tools_sync(self, tools: list[Tool]) -> None:
        run_sync(self.register_tools(tools))

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    async def start_run(self, config: RunConfig) -> Run:
        """Start a new run. Idempotent — same runId returns the same run."""
        await self.ready()

        existing = self._active_runs.get(config.run_id)
        if existing and not existing.is_ended:
            return existing

        lockdown = await self._api.start_run(
            run_id=config.run_id,
            agent_id=self._agent_id or "",
            session_id=config.session_id,
            actor=config.actor,
            model=config.model,
        )

        if lockdown.active:
            logger.warning(
                "[Handlebar] Agent is under lockdown%s",
                f": {lockdown.reason}" if lockdown.reason else "",
            )

        run = Run(
            RunInternalConfig(
                run_config=config,
                agent_id=self._agent_id,
                tools=self._tools,
                enforce_mode=self._config.enforce_mode,
                fail_closed=self._config.fail_closed,
                api=self._api,
                bus=self._bus,
                metric_registry=self._metric_registry,
                subject_registry=self._subject_registry,
            )
        )

        self._active_runs[config.run_id] = run
        return run

    def start_run_sync(self, config: RunConfig) -> Run:
        return run_sync(self.start_run(config))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Flush all pending audit events and release resources."""
        await self._bus.close()

    def shutdown_sync(self) -> None:
        run_sync(self.shutdown())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _init_sinks(self, sinks: list[SinkConfig] | None) -> None:
        if not sinks:
            endpoint = resolve_api_endpoint(self._config.api_endpoint)
            api_key = self._config.api_key or os.environ.get("HANDLEBAR_API_KEY")
            self._bus.add(create_http_sink(endpoint, api_key))
        else:
            for sink_cfg in sinks:
                if sink_cfg.type == "console":
                    self._bus.add(create_console_sink(sink_cfg.format))  # type: ignore[arg-type]
                elif sink_cfg.type == "http":
                    endpoint = resolve_api_endpoint(
                        getattr(sink_cfg, "endpoint", None) or self._config.api_endpoint
                    )
                    api_key = (
                        getattr(sink_cfg, "api_key", None)
                        or self._config.api_key
                        or os.environ.get("HANDLEBAR_API_KEY")
                    )
                    self._bus.add(create_http_sink(endpoint, api_key))
        await self._bus.init()

    async def _init_agent(self, config: HandlebarConfig) -> None:
        agent = config.agent
        agent_id = await self._api.upsert_agent(
            slug=agent.slug,
            name=agent.name,
            description=agent.description,
            tags=agent.tags,
            tools=config.tools,
        )
        self._agent_id = agent_id


# ---------------------------------------------------------------------------
# Convenience factory — mirrors JS `Handlebar.init(config)`
# ---------------------------------------------------------------------------


class Handlebar:
    """Top-level namespace — mirrors the JS ``Handlebar`` convenience object."""

    @staticmethod
    async def init(config: HandlebarClientConfig) -> HandlebarClient:
        return await HandlebarClient.init(config)

    @staticmethod
    def init_sync(config: HandlebarClientConfig) -> HandlebarClient:
        return HandlebarClient.init_sync(config)
