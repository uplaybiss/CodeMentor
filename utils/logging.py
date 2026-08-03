"""Centralised logging for EduAgent.

Design rationale
----------------
Every module logs through :func:`get_logger`, which hands back a child of the
root logger configured here once per process. Two output sinks are attached:
a human-readable console stream and a rotating file for forensic debugging.
Optionally the console formatter is switched to JSON so logs can be shipped
straight into a log aggregator in production.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any

from config.settings import LoggingSettings

_configured: bool = False


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("student_id", "agent", "attempt", "document_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def _build_formatter(json_format: bool) -> logging.Formatter:
    if json_format:
        return JsonFormatter()
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(settings: LoggingSettings) -> None:
    """Configure the root logger once. Idempotent across repeated calls."""
    global _configured  # noqa: PLW0603 - intentional idempotent setup guard
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(settings.level.upper())

    formatter = _build_formatter(settings.json_format)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if settings.log_dir:
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "edueagent.log",
            maxBytes=settings.rotation_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence noisy third-party loggers.
    for noisy in ("httpx", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger named relative to the package root."""
    if not _configured:
        # Safe fallback so tests/logging work before explicit setup.
        setup_logging(LoggingSettings())
    return logging.getLogger(name)
