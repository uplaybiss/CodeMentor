"""SQLAlchemy ORM models — the physical schema for EduAgent.

Design rationale
----------------
* SQLite is the default backing store (zero setup, file-based), but every
  column is portable so ``DATABASE_URL`` can point at Postgres later.
* List-valued columns use SQLAlchemy's ``JSON`` type; SQLite persists these as
  TEXT natively while other dialects keep their native JSON column.
* All timestamps are ISO-8601 strings (UTC) to stay timezone-safe and
  JSON-ready with no conversion step.
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Student(Base):
    """Core student profile row."""

    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="Learner")
    level: Mapped[str] = mapped_column(String(32), default="beginner")
    learning_style: Mapped[str] = mapped_column(String(32), default="reading_writing")
    learning_speed: Mapped[str] = mapped_column(String(32), default="normal")
    preferred_difficulty: Mapped[str] = mapped_column(String(32), default="beginner")
    topics_learned: Mapped[list[str]] = mapped_column(JSON, default=list)
    weak_concepts: Mapped[list[str]] = mapped_column(JSON, default=list)
    quiz_scores: Mapped[list[float]] = mapped_column(JSON, default=list)
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))


class Topic(Base):
    """A topic the student has encountered, with mastery status."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    name: Mapped[str] = mapped_column(String(256))
    mastered: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[str] = mapped_column(String(32))


class ConceptScoreRow(Base):
    """Rolling proficiency score per concept."""

    __tablename__ = "concept_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    concept: Mapped[str] = mapped_column(String(256))
    proficiency: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)


class Lesson(Base):
    """One Teacher-agent lesson delivered to a student."""

    __tablename__ = "lessons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    topic: Mapped[str] = mapped_column(String(256))
    summary: Mapped[str] = mapped_column(Text, default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    examples: Mapped[list[str]] = mapped_column(JSON, default=list)
    analogies: Mapped[list[str]] = mapped_column(JSON, default=list)
    recap: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(32))


class Quiz(Base):
    """A generated quiz header."""

    __tablename__ = "quizzes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    topic: Mapped[str] = mapped_column(String(256))
    difficulty: Mapped[str] = mapped_column(String(32), default="beginner")
    created_at: Mapped[str] = mapped_column(String(32))


class QuizQuestion(Base):
    """One question belonging to a quiz."""

    __tablename__ = "quiz_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quiz_id: Mapped[str] = mapped_column(ForeignKey("quizzes.id"))
    qtype: Mapped[str] = mapped_column(String(32))
    topic: Mapped[str] = mapped_column(String(256))
    question: Mapped[str] = mapped_column(Text)
    options: Mapped[list[str]] = mapped_column(JSON, default=list)
    correct_answer: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text, default="")
    points: Mapped[int] = mapped_column(Integer, default=1)


class QuizAttempt(Base):
    """One submitted-and-evaluated quiz attempt."""

    __tablename__ = "quiz_attempts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    quiz_id: Mapped[str] = mapped_column(ForeignKey("quizzes.id"))
    answers: Mapped[dict] = mapped_column(JSON, default=dict)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)
    percent: Mapped[float] = mapped_column(Float, default=0.0)
    weak_concepts: Mapped[list[str]] = mapped_column(JSON, default=list)
    revision_topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    next_lesson: Mapped[str] = mapped_column(Text, default="")
    feedback: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(32))


class Conversation(Base):
    """One message in a student conversation transcript."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(32))


class MemoryEntry(Base):
    """A long-term memory fact extracted from the conversation."""

    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    fact_type: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[str] = mapped_column(String(32))


class Document(Base):
    """Metadata about an uploaded document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    filename: Mapped[str] = mapped_column(String(256))
    filepath: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(32))


class Recommendation(Base):
    """Latest recommendation payload generated for a student."""

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    next_topic: Mapped[str] = mapped_column(String(256))
    rationale: Mapped[str] = mapped_column(Text, default="")
    revision_schedule: Mapped[list[str]] = mapped_column(JSON, default=list)
    daily_plan: Mapped[list[str]] = mapped_column(JSON, default=list)
    weak_areas: Mapped[list[str]] = mapped_column(JSON, default=list)
    suggested_quiz_topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[str] = mapped_column(String(32))


class ProgressSnapshot(Base):
    """One point on the learning-curve chart (topic + score + timestamp)."""

    __tablename__ = "progress_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    topic: Mapped[str] = mapped_column(String(256))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[str] = mapped_column(String(32))


class HarnessRun(Base):
    """Run ownership/status; graph snapshots belong to LangGraph's own saver."""

    __tablename__ = "harness_runs"
    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(String(64))
    workflow: Mapped[str] = mapped_column(String(16))
    input_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="running")
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))


class HarnessEffect(Base):
    """Atomic local write + cached result, scoped to a graph node invocation."""

    __tablename__ = "harness_effects"
    effect_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    tool_name: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(32))


class WrongAnswer(Base):
    """Bounded topic-specific context can query actual mistakes, not all history."""

    __tablename__ = "wrong_answers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    topic: Mapped[str] = mapped_column(String(256), index=True)
    quiz_id: Mapped[str] = mapped_column(ForeignKey("quizzes.id"))
    question: Mapped[str] = mapped_column(Text)
    submitted: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(32))
