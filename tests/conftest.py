"""Shared pytest fixtures: isolated settings, DB, container and API client.

Every fixture uses ``tmp_path`` so tests never touch the real ``./data``
directory or the real (env-driven) settings. The LLM is always the offline
``demo`` provider and the vector store always uses :class:`FakeEmbedder`, so the
whole suite is deterministic and needs no API keys or network access.
"""

from __future__ import annotations

import pytest

from config.settings import (
    ApiSettings,
    AppSettings,
    DatabaseSettings,
    LLMSettings,
    LoggingSettings,
    Settings,
    VectorStoreSettings,
)
from database.connection import Database
from services.container import Container
from tests.fakes import FakeEmbedder


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fully isolated settings pointed at per-test temp directories."""
    return Settings(
        _env_file=None,
        app=AppSettings(name="test-app", version="0.0.0-test", debug=False),
        llm=LLMSettings(provider="demo"),
        database=DatabaseSettings(
            url=f"sqlite:///{tmp_path / 'test.db'}",
            create_on_startup=True,
        ),
        vector_store=VectorStoreSettings(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_notes",
            top_k=4,
            embedding_model="fake",
        ),
        api=ApiSettings(upload_dir=str(tmp_path / "uploads")),
        logging=LoggingSettings(log_dir=str(tmp_path / "logs")),
    )


@pytest.fixture
def db(settings) -> Database:
    database = Database(settings.database)
    database.create_schema()
    yield database
    database.dispose()


@pytest.fixture
def container(settings, db, monkeypatch) -> Container:
    """Container wired to the temp DB and a fake embedder."""
    monkeypatch.setattr(
        "services.rag.vector_store.get_embedding_provider",
        lambda _settings: FakeEmbedder(),
    )
    c = Container(settings, database=db)
    yield c
    c.dispose()


@pytest.fixture
def ctx(container):
    """The AgentContext the graph nodes receive."""
    return container.context()


@pytest.fixture
def client(container):
    """FastAPI TestClient with the real container injected via overrides.

    The app's lifespan is *not* run (we don't enter the context manager) because
    it would build the process-wide, real-settings container; the test container
    has already created its schema, so requests are fully functional.
    """
    from fastapi.testclient import TestClient

    from api.dependencies import get_container_dep, get_ctx
    from api.main import create_app

    app = create_app()
    app.dependency_overrides[get_container_dep] = lambda: container
    app.dependency_overrides[get_ctx] = container.context
    return TestClient(app)
