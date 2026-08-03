"""Persistence for student profiles, concept scores and topic history."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from database.models import ConceptScoreRow, Student, Topic
from database.repositories.base_repository import BaseRepository
from models.enums import StudentLevel
from models.schemas import ConceptScore, StudentProfile
from utils.errors import ResourceNotFoundError
from utils.helpers import utc_now_iso


class StudentRepository(BaseRepository):
    """CRUD + aggregates around :class:`Student`."""

    # ------------------------------------------------------------- create --
    def create(
        self,
        *,
        student_id: str,
        name: str = "Learner",
        level: str = StudentLevel.BEGINNER.value,
    ) -> StudentProfile:
        now = utc_now_iso()
        student = Student(
            id=student_id,
            name=name,
            level=level,
            created_at=now,
            updated_at=now,
        )
        with self.session() as session:
            session.add(student)
            session.flush()
        return self._to_profile(student)

    def get_or_create(self, student_id: str, name: str = "Learner") -> StudentProfile:
        existing = self.get(student_id)
        if existing is not None:
            return existing
        return self.create(student_id=student_id, name=name)

    # -------------------------------------------------------------- read --
    def get(self, student_id: str) -> StudentProfile | None:
        with self.session() as session:
            student = session.get(Student, student_id)
            return self._to_profile(student) if student else None

    def get_or_404(self, student_id: str) -> StudentProfile:
        profile = self.get(student_id)
        if profile is None:
            raise ResourceNotFoundError(f"Student {student_id!r} not found")
        return profile

    # ------------------------------------------------------------ update --
    def update_profile(self, student_id: str, **fields: Any) -> StudentProfile:
        allowed = {
            "name",
            "level",
            "learning_style",
            "learning_speed",
            "preferred_difficulty",
        }
        clean = {k: v for k, v in fields.items() if k in allowed and v is not None}
        with self.session() as session:
            student = session.get(Student, student_id)
            if student is None:
                raise ResourceNotFoundError(f"Student {student_id!r} not found")
            for key, value in clean.items():
                setattr(student, key, value)
            student.updated_at = utc_now_iso()
            session.flush()
        return self._to_profile(student)

    # ------------------------------------------------------- aggregates --
    def record_quiz_score(self, student_id: str, score: float) -> None:
        """Append one quiz score to the student's rolling history."""
        with self.session() as session:
            student = session.get(Student, student_id)
            if student is None:
                raise ResourceNotFoundError(f"Student {student_id!r} not found")
            scores = list(student.quiz_scores or [])
            scores.append(float(score))
            student.quiz_scores = scores[-200:]  # cap history length
            student.updated_at = utc_now_iso()

    def add_weak_concept(self, student_id: str, concept: str) -> None:
        with self.session() as session:
            student = session.get(Student, student_id)
            if student is None:
                raise ResourceNotFoundError(f"Student {student_id!r} not found")
            weak = list(student.weak_concepts or [])
            if concept not in weak:
                weak.append(concept)
                student.weak_concepts = weak
                student.updated_at = utc_now_iso()

    def add_topic_learned(self, student_id: str, topic: str) -> None:
        with self.session() as session:
            student = session.get(Student, student_id)
            if student is None:
                raise ResourceNotFoundError(f"Student {student_id!r} not found")
            learned = list(student.topics_learned or [])
            if topic not in learned:
                learned.append(topic)
                student.topics_learned = learned
                student.updated_at = utc_now_iso()
            session.add(
                Topic(student_id=student_id, name=topic, created_at=utc_now_iso())
            )

    def list_topics(self, student_id: str, limit: int = 100) -> list[str]:
        with self.session() as session:
            rows = session.execute(
                select(Topic.name)
                .where(Topic.student_id == student_id)
                .order_by(Topic.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return list(dict.fromkeys(rows))

    # --------------------------------------------------- concept scores --
    def upsert_concept_score(
        self, student_id: str, concept: str, proficiency: float
    ) -> ConceptScore:
        """Weighted update of a concept's proficiency based on the new score."""
        with self.session() as session:
            row = session.execute(
                select(ConceptScoreRow).where(
                    ConceptScoreRow.student_id == student_id,
                    ConceptScoreRow.concept == concept,
                )
            ).scalar_one_or_none()
            if row is None:
                row = ConceptScoreRow(
                    student_id=student_id, concept=concept, proficiency=0.0, attempts=0
                )
                session.add(row)
            prev = row.proficiency
            row.attempts += 1
            # Exponential moving average — recent attempts matter more.
            alpha = 0.6 if row.attempts > 1 else 1.0
            row.proficiency = round(prev * (1 - alpha) + proficiency * alpha, 4)
            session.flush()
            return ConceptScore(
                concept=row.concept, proficiency=row.proficiency, attempts=row.attempts
            )

    def list_concept_scores(self, student_id: str) -> list[ConceptScore]:
        with self.session() as session:
            rows = session.execute(
                select(ConceptScoreRow)
                .where(ConceptScoreRow.student_id == student_id)
                .order_by(ConceptScoreRow.proficiency.asc())
            ).scalars().all()
            return [
                ConceptScore(
                    concept=r.concept, proficiency=r.proficiency, attempts=r.attempts
                )
                for r in rows
            ]

    def weakest_concepts(self, student_id: str, limit: int = 5) -> list[str]:
        scores = self.list_concept_scores(student_id)
        weak = [s for s in scores if s.proficiency < 0.6]
        weak.sort(key=lambda s: s.proficiency)
        return [s.concept for s in weak[:limit]]

    # ---------------------------------------------------------- helpers --
    @staticmethod
    def _to_profile(student: Student) -> StudentProfile:
        return StudentProfile(
            student_id=student.id,
            name=student.name,
            level=StudentLevel(student.level) if student.level else StudentLevel.BEGINNER,
            learning_style=student.learning_style or "reading_writing",
            learning_speed=student.learning_speed or "normal",
            preferred_difficulty=student.preferred_difficulty or "beginner",
            topics_learned=list(student.topics_learned or []),
            weak_concepts=list(student.weak_concepts or []),
            quiz_scores=list(student.quiz_scores or []),
        )
