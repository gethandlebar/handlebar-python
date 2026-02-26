"""Utility for running async coroutines from synchronous contexts."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Works in both sync contexts (no running event loop) and from within a
    running event loop (e.g. a Jupyter notebook or an async framework) by
    delegating to a fresh thread in the latter case.

    Note: calling sync wrappers from within an async context is generally
    discouraged — prefer the async API there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop — straightforward case.
        return asyncio.run(coro)
    else:
        # A loop is already running (e.g. inside an async framework).
        # Spin up a worker thread with its own fresh event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()  # ty:ignore[invalid-return-type]
