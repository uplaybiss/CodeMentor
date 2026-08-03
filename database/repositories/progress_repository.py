"""Persistence for progress snapshots (the learning-curve chart)."""

from __future__ import annotations

from sqlalchemy import func, select

from database.models import ProgressSnapshot
from database.repositories.base_repository import BaseRepository
from models.schemas import ProgressPoint
from utils.helpers import utc_now_iso


class ProgressRepository(BaseRepository):
    """Append-only log of (topic, score, timestamp) points."""

    def add_snapshot(
        self, student_id: str, topic: str, score: float
    ) -> int:
        with self.session() as session:
            row = ProgressSnapshot(
                student_id=student_id,
                topic=topic,
                score=float(score),
                created_at=utc_now_iso(),
            )
            session.add(row)
            session.flush()
            return row.id

    def list_snapshots(
        self, student_id: str, limit: int = 200
    ) -> list[ProgressPoint]:
        """Chronological progress points (oldest first)."""
        with self.session() as session:
            rows = session.execute(
                select(ProgressSnapshot)
                .where(ProgressSnapshot.student_id == student_id)
                .order_by(ProgressSnapshot.id.asc())
                .limit(limit)
            ).scalars().all()
            return [
                ProgressPoint(
                    date=r.created_at, score=r.score, topic=r.topic
                )
                for r in rows
            ]

    def latest_score(self, student_id: str) -> ProgressPoint | None:
        with self.session() as session:
            row = session.execute(
                select(ProgressSnapshot)
                .where(ProgressSnapshot.student_id == student_id)
                .order_by(ProgressSnapshot.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return ProgressPoint(date=row.created_at, score=row.score, topic=row.topic)

    def count(self, student_id: str) -> int:
        with self.session() as session:
            return session.execute(
                select(func.count()).select_from(ProgressSnapshot).where(
                    ProgressSnapshot.student_id == student_id
                )
            ).scalar_one()
