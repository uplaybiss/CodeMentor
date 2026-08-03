"""Centralised, typed configuration for EduAgent.

Design rationale
----------------
* One ``Settings`` object is built once per process from environment variables
  (with ``.env`` support) and passed around via dependency injection. Nothing in
  the codebase reaches for ``os.environ`` directly — this keeps configuration
  testable and keeps every component decoupled from its environment.
* Each concern gets its own nested settings model with an ``env_prefix`` so the
  env namespace is flat (e.g. ``LLM_OPENAI_API_KEY``) while the Python objects
  stay grouped.
* Safe defaults mean the project boots without a real API key — useful for the
  test-suite and for demo mode.

Note on ``env_file``: pydantic-settings only applies ``env_file`` to the model
that declares it. Nested models with their own ``model_config`` (as here, for
``env_prefix``) do **not** inherit the parent's ``env_file``, so every nested
model declares the same ``env_file=".env"`` to keep ``.env`` values flowing to
all groups.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Shared by every nested settings model. pydantic-settings does not propagate
# the parent's ``env_file`` to nested models that define their own config, so
# each group declares it explicitly to honour ``.env`` values.
_ENV_FILE = ".env"


class AppSettings(BaseSettings):
    """Application-level metadata and runtime behaviour."""

    model_config = SettingsConfigDict(
        env_prefix="APP_", extra="ignore", env_file=_ENV_FILE
    )

    name: str = "EduAgent"
    version: str = "1.0.0"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://localhost:8000"]
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        """Allow CORS origins to be supplied as a CSV env var."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class LLMSettings(BaseSettings):
    """LLM provider selection. Provider is ``openai`` or ``gemini``."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", extra="ignore", env_file=_ENV_FILE
    )

    provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.4
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.4
    timeout_seconds: float = 60.0
    max_retries: int = 2


class DatabaseSettings(BaseSettings):
    """SQLAlchemy connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_", extra="ignore", env_file=_ENV_FILE
    )

    url: str = "sqlite:///./data/edueagent.db"
    echo: bool = False
    create_on_startup: bool = True


class VectorStoreSettings(BaseSettings):
    """ChromaDB + embedding model settings."""

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_", extra="ignore", env_file=_ENV_FILE
    )

    persist_directory: str = "./data/chroma"
    collection_name: str = "edueagent_notes"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4


class ApiSettings(BaseSettings):
    """REST API surface settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_", extra="ignore", env_file=_ENV_FILE
    )

    prefix: str = "/api/v1"
    upload_dir: str = "./data/uploads"
    pagination_default_size: int = 20


class LoggingSettings(BaseSettings):
    """Centralised logging behaviour."""

    model_config = SettingsConfigDict(
        env_prefix="LOGGING_", extra="ignore", env_file=_ENV_FILE
    )

    level: str = "INFO"
    json_format: bool = False
    log_dir: str = "./logs"
    rotation_bytes: int = 5 * 1024 * 1024  # 5 MB
    backup_count: int = 3


class FrontendSettings(BaseSettings):
    """Where the Streamlit dashboard reaches the API."""

    model_config = SettingsConfigDict(
        env_prefix="FRONTEND_", extra="ignore", env_file=_ENV_FILE
    )

    api_base_url: str = "http://localhost:8000"


class Settings(BaseSettings):
    """Aggregate settings object injected across the whole application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    frontend: FrontendSettings = Field(default_factory=FrontendSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` instance (cached)."""
    return Settings()
