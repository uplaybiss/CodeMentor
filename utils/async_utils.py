"""Async helpers: resilient retries and safe sync<->async bridging.

Design rationale
----------------
LLM calls are the most failure-prone dependency in the system. ``async_retry``
gives every agent a uniform, configurable retry policy (exponential backoff +
jitter) so transient upstream failures do not fail a whole tutoring run.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from utils.errors import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def async_retry(
    fn: Callable[..., Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    exceptions: tuple[type[BaseException], ...] = (LLMError, TimeoutError),
    label: str = "operation",
) -> T:
    """Await ``fn`` retrying on ``exceptions`` with exponential backoff + jitter.

    Raises the last exception once ``attempts`` are exhausted.
    """
    delay = base_delay
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except exceptions as exc:  # type: ignore[valid-type]
            last_exc = exc
            if attempt >= attempts:
                break
            jittered = delay * (1 + random.random() * 0.25)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                label,
                attempt,
                attempts,
                exc,
                jittered,
            )
            await asyncio.sleep(jittered)
            delay = min(delay * 2, max_delay)
    assert last_exc is not None
    raise last_exc


def retry_decorator(
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (LLMError, TimeoutError),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator form of :func:`async_retry` for instance methods."""

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await async_retry(
                functools.partial(fn, *args, **kwargs),
                attempts=attempts,
                base_delay=base_delay,
                exceptions=exceptions,
                label=fn.__name__,
            )

        return wrapper

    return decorator


def run_coro(coro: Awaitable[T]) -> T:
    """Run a coroutine from synchronous code, tolerating a running loop.

    Used by the (synchronous) Streamlit dashboard when it talks to the async
    API client via ``asyncio.run``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # An event loop is already running (e.g. inside a notebook): do our best.
    return asyncio.new_event_loop().run_until_complete(coro)
