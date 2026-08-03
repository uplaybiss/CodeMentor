"""Base repository providing shared session access.

All repositories inherit :class:`BaseRepository`, which wraps a
:class:`~database.connection.Database` and exposes the transactional
``session()`` context manager. This keeps every repository unit-testable with
an in-memory SQLite database and keeps SQL out of the agent layer entirely.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from database.connection import Database


class BaseRepository:
    """Shared plumbing for data-access repositories."""

    def __init__(self, db: Database) -> None:
        self._db = db

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Open a transactional session bound to this repository's database."""
        with self._db.session() as session:
            yield session

    @property
    def database(self) -> Database:
        return self._db
