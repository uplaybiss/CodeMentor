"""Database engine + session management.

Design rationale
----------------
* One ``Database`` object owns the SQLAlchemy ``Engine`` and a session factory.
  It is constructed once from settings and injected everywhere — repositories
  and the API never build their own engines.
* ``session()`` is a commit-on-success context manager, so callers are shielded
  from manual ``commit``/``rollback``/``close`` and transactions always resolve.
* SQLite foreign-key enforcement is switched on via a connection event so
  referential integrity holds even on the lightweight default store.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import DatabaseSettings
from database.models import Base
from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class Database:
    """Owns an engine and session factory; exposes a transactional session."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._active_session: ContextVar[Session | None] = ContextVar(
            f"database_session_{id(self)}", default=None
        )
        self._url = settings.url
        self._engine: Engine = self._build_engine(settings)
        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )

    # ------------------------------------------------------------------ #
    def _build_engine(self, settings: DatabaseSettings) -> Engine:
        url = settings.url
        connect_args: dict = {}
        if url.startswith("sqlite"):
            # SQLite objects cannot be shared across threads by default.
            connect_args["check_same_thread"] = False
            db_path = url.replace("sqlite:///", "", 1)
            if db_path and db_path != ":memory:":
                ensure_dir(Path(db_path).parent)
        engine = create_engine(url, echo=settings.echo, connect_args=connect_args)

        if url.startswith("sqlite"):

            @event.listens_for(engine, "connect")
            def _enable_sqlite_fk(dbapi_connection, _record):  # pragma: no cover
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        return engine

    # ------------------------------------------------------------------ #
    def create_schema(self) -> None:
        """Create all tables that do not yet exist."""
        Base.metadata.create_all(self._engine)
        logger.info("Database schema ensured for %s", self._url)

    def drop_schema(self) -> None:
        """Drop all tables (used by tests / reset tooling only)."""
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a bound session; commit on success, rollback on error."""
        existing = self._active_session.get()
        if existing is not None:
            yield existing
            return
        session = self._session_factory()
        token = self._active_session.set(session)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._active_session.reset(token)
            session.close()

    def dispose(self) -> None:
        """Release connection pools (used at shutdown / in tests)."""
        self._engine.dispose()


@lru_cache(maxsize=1)
def get_database(settings: DatabaseSettings | None = None) -> Database:
    """Return the process-wide :class:`Database` (cached)."""
    from config import get_settings

    resolved = settings or get_settings().database
    return Database(resolved)
