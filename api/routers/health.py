"""Health-check endpoints — no authentication required."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_container_dep
from services.container import Container

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    demo_mode: bool = False


@router.get("/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container_dep)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=container.settings.app.version,
        demo_mode=container.settings.llm.provider == "demo",
    )
