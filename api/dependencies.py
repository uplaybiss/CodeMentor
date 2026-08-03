"""FastAPI dependency-injection helpers."""

from __future__ import annotations

from functools import lru_cache

from agents.context import AgentContext
from services.container import Container, get_container


@lru_cache(maxsize=1)
def get_container_cache() -> Container:
    """Lazy singleton used by all endpoint dependencies."""
    return get_container()


async def get_ctx() -> AgentContext:
    """Yield the shared :class:`AgentContext` for graph-based endpoints."""
    return get_container_cache().context()


def get_container_dep() -> Container:
    """FastAPI dependency for endpoints that need the full container."""
    return get_container_cache()
