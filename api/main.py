"""FastAPI application entry point for EduAgent.

The app is a thin HTTP layer over the LangGraph workflows and the services
layer. Dependency injection flows through :func:`api.dependencies.get_ctx` so
tests can swap the container without touching the route logic.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.exception_handlers import register_exception_handlers
from api.routers import chat, dashboard, documents, health, quizzes, students
from config.settings import get_settings
from services.container import get_container

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure DB schema exists. Shutdown: release connection pools."""
    container = get_container()
    if container.settings.database.create_on_startup:
        container.db.create_schema()
    logger.info("EduAgent API ready (LLM provider=%s)", container.settings.llm.provider)
    yield
    container.dispose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=(
            "EduAgent — an AI tutoring assistant backed by LangGraph, RAG and a "
            "persistent student model."
        ),
        lifespan=lifespan,
    )

    # CORS for the Streamlit frontend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Map the EduAgentError tree to HTTP responses.
    register_exception_handlers(app)

    # Routers.
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(students.router)
    app.include_router(quizzes.router)
    app.include_router(documents.router)
    app.include_router(dashboard.router)

    return app


app = create_app()
