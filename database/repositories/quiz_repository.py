"""Persistence for quizzes, questions and attempts."""

from __future__ import annotations

from sqlalchemy import select

from database.models import Quiz, QuizAttempt, QuizQuestion
from database.repositories.base_repository import BaseRepository
from models.enums import DifficultyLevel, QuestionType
from models.schemas import EvaluationResult
from models.schemas import Quiz as QuizSchema
from models.schemas import QuizQuestion as QuizQuestionSchema
from utils.errors import ResourceNotFoundError
from utils.helpers import utc_now_iso


class QuizRepository(BaseRepository):
    """CRUD + history for quizzes and quiz attempts."""

    # ------------------------------------------------------------ create --
    def create_quiz(self, quiz: QuizSchema, student_id: str) -> QuizSchema:
        """Persist a generated quiz (header + questions) idempotently."""
        now = utc_now_iso()
        with self.session() as session:
            existing = session.get(Quiz, quiz.id)
            if existing is not None:
                if existing.student_id != student_id:
                    from harness.errors import ToolPermissionError

                    raise ToolPermissionError("Quiz ID belongs to another student")
                return self.get_quiz(quiz.id)
            if existing is None:
                session.add(
                    Quiz(
                        id=quiz.id,
                        student_id=student_id,
                        topic=quiz.topic,
                        difficulty=quiz.difficulty.value,
                        created_at=now,
                    )
                )
                session.flush()  # parent must exist before inserting FK-bound questions
            for question in quiz.questions:
                session.add(
                    QuizQuestion(
                        quiz_id=quiz.id,
                        qtype=question.type.value,
                        topic=question.topic,
                        question=question.question,
                        options=list(question.options),
                        correct_answer=question.correct_answer,
                        explanation=question.explanation,
                        points=question.points,
                    )
                )
            session.flush()  # visible to a reload within the guard's outer transaction
        return quiz

    # -------------------------------------------------------------- read --
    def get_quiz(self, quiz_id: str) -> QuizSchema | None:
        with self.session() as session:
            header = session.get(Quiz, quiz_id)
            if header is None:
                return None
            questions = (
                session.execute(
                    select(QuizQuestion)
                    .where(QuizQuestion.quiz_id == quiz_id)
                    .order_by(QuizQuestion.id)
                )
                .scalars()
                .all()
            )
            return QuizSchema(
                id=header.id,
                topic=header.topic,
                difficulty=DifficultyLevel(header.difficulty),
                questions=[
                    QuizQuestionSchema(
                        id=f"{quiz_id}q{idx}",
                        type=QuestionType(q.qtype),
                        topic=q.topic,
                        question=q.question,
                        options=list(q.options or []),
                        correct_answer=q.correct_answer,
                        explanation=q.explanation,
                        points=q.points,
                    )
                    for idx, q in enumerate(questions, start=1)
                ],
            )

    def list_quizzes(self, student_id: str, limit: int = 20) -> list[str]:
        """Return quiz ids for a student, newest first."""
        with self.session() as session:
            return list(
                session.execute(
                    select(Quiz.id)
                    .where(Quiz.student_id == student_id)
                    .order_by(Quiz.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )

    # ---------------------------------------------------------- attempts --
    def save_attempt(
        self,
        *,
        attempt_id: str,
        student_id: str,
        quiz_id: str,
        answers: dict,
        evaluation: EvaluationResult,
    ) -> None:
        with self.session() as session:
            session.add(
                QuizAttempt(
                    id=attempt_id,
                    student_id=student_id,
                    quiz_id=quiz_id,
                    answers=answers,
                    score=evaluation.score,
                    max_score=evaluation.max_score,
                    percent=evaluation.percent,
                    weak_concepts=list(evaluation.weak_concepts),
                    revision_topics=list(evaluation.revision_topics),
                    next_lesson=evaluation.next_lesson,
                    feedback=evaluation.feedback,
                    created_at=utc_now_iso(),
                )
            )

    def get_attempt(self, attempt_id: str) -> dict | None:
        with self.session() as session:
            attempt = session.get(QuizAttempt, attempt_id)
            if attempt is None:
                return None
            return {
                "attempt_id": attempt.id,
                "student_id": attempt.student_id,
                "quiz_id": attempt.quiz_id,
                "score": attempt.score,
                "max_score": attempt.max_score,
                "percent": attempt.percent,
                "weak_concepts": list(attempt.weak_concepts or []),
                "revision_topics": list(attempt.revision_topics or []),
                "next_lesson": attempt.next_lesson,
                "feedback": attempt.feedback,
                "created_at": attempt.created_at,
            }

    def get_attempt_or_404(self, attempt_id: str) -> dict:
        attempt = self.get_attempt(attempt_id)
        if attempt is None:
            raise ResourceNotFoundError(f"Attempt {attempt_id!r} not found")
        return attempt

    def list_attempts(self, student_id: str, limit: int = 50) -> list[dict]:
        """Attempt history for the progress chart (chronological)."""
        with self.session() as session:
            rows = (
                session.execute(
                    select(QuizAttempt)
                    .where(QuizAttempt.student_id == student_id)
                    .order_by(QuizAttempt.created_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "attempt_id": r.id,
                    "topic": r.quiz_id,  # overwritten by caller with topic if known
                    "score": r.score,
                    "percent": r.percent,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
