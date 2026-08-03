"""Persistence for delivered lessons."""

from __future__ import annotations

from database.models import Lesson
from database.repositories.base_repository import BaseRepository
from models.schemas import Explanation
from utils.helpers import utc_now_iso


class LessonRepository(BaseRepository):
    """Append-only store of Teacher-agent lessons."""

    def save_lesson(self, student_id: str, lesson: Explanation) -> int:
        with self.session() as session:
            row = Lesson(
                student_id=student_id,
                topic=lesson.topic,
                summary=lesson.summary,
                explanation=lesson.explanation,
                examples=list(lesson.examples),
                analogies=list(lesson.analogies),
                recap=lesson.recap,
                created_at=utc_now_iso(),
            )
            session.add(row)
            session.flush()
            return row.id

    def list_lessons(self, student_id: str, limit: int = 50) -> list[Explanation]:
        from sqlalchemy import select

        with self.session() as session:
            rows = session.execute(
                select(Lesson)
                .where(Lesson.student_id == student_id)
                .order_by(Lesson.id.desc())
                .limit(limit)
            ).scalars().all()
            return [
                Explanation(
                    topic=r.topic,
                    summary=r.summary,
                    explanation=r.explanation,
                    examples=list(r.examples or []),
                    analogies=list(r.analogies or []),
                    recap=r.recap,
                )
                for r in reversed(rows)
            ]
