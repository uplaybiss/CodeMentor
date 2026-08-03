"""Persistence for long-term memory facts and recommendations."""

from __future__ import annotations

from sqlalchemy import select

from database.models import MemoryEntry, Recommendation
from database.repositories.base_repository import BaseRepository
from models.schemas import Recommendation as RecommendationSchema
from utils.helpers import utc_now_iso


class MemoryRepository(BaseRepository):
    """Stores the facts the Memory agent extracts, plus recommendations."""

    def add_fact(
        self, student_id: str, fact_type: str, content: str, importance: float = 0.5
    ) -> int:
        with self.session() as session:
            row = MemoryEntry(
                student_id=student_id,
                fact_type=fact_type,
                content=content,
                importance=importance,
                created_at=utc_now_iso(),
            )
            session.add(row)
            session.flush()
            return row.id

    def list_facts(
        self, student_id: str, fact_type: str | None = None, limit: int = 50
    ) -> list[dict]:
        query = select(MemoryEntry).where(MemoryEntry.student_id == student_id)
        if fact_type:
            query = query.where(MemoryEntry.fact_type == fact_type)
        query = query.order_by(MemoryEntry.id.desc()).limit(limit)
        with self.session() as session:
            rows = session.execute(query).scalars().all()
            return [
                {
                    "id": r.id,
                    "fact_type": r.fact_type,
                    "content": r.content,
                    "importance": r.importance,
                    "created_at": r.created_at,
                }
                for r in reversed(rows)
            ]

    def latest_facts_by_type(
        self, student_id: str, fact_type: str, limit: int = 5
    ) -> list[dict]:
        return self.list_facts(student_id, fact_type=fact_type, limit=limit)

    # ----------------------------------------------------- recommendations --
    def save_recommendation(self, student_id: str, rec: RecommendationSchema) -> int:
        with self.session() as session:
            row = Recommendation(
                student_id=student_id,
                next_topic=rec.next_topic,
                rationale=rec.rationale,
                revision_schedule=list(rec.revision_schedule),
                daily_plan=list(rec.daily_plan),
                weak_areas=list(rec.weak_areas),
                suggested_quiz_topics=list(rec.suggested_quiz_topics),
                created_at=utc_now_iso(),
            )
            session.add(row)
            session.flush()
            return row.id

    def latest_recommendation(self, student_id: str) -> RecommendationSchema | None:
        with self.session() as session:
            row = session.execute(
                select(Recommendation)
                .where(Recommendation.student_id == student_id)
                .order_by(Recommendation.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return RecommendationSchema(
                next_topic=row.next_topic,
                rationale=row.rationale,
                revision_schedule=list(row.revision_schedule or []),
                daily_plan=list(row.daily_plan or []),
                weak_areas=list(row.weak_areas or []),
                suggested_quiz_topics=list(row.suggested_quiz_topics or []),
            )
