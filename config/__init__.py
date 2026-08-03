"""Configuration package for EduAgent.

Exposes a lazily-initialised, cached ``Settings`` singleton via :func:`get_settings`.
Importing ``from config import settings`` gives a fully-populated settings object
without triggering any I/O until the first attribute access.
"""

from config.settings import (
    ApiSettings,
    AppSettings,
    DatabaseSettings,
    FrontendSettings,
    LLMSettings,
    LoggingSettings,
    Settings,
    VectorStoreSettings,
    get_settings,
)

__all__ = [
    "ApiSettings",
    "AppSettings",
    "DatabaseSettings",
    "FrontendSettings",
    "LLMSettings",
    "LoggingSettings",
    "Settings",
    "VectorStoreSettings",
    "get_settings",
    "settings",
]

settings = get_settings()
