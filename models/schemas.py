"""Pydantic domain models.

These are the *contracts* shared by the API layer, the LangGraph state, the
agents and the dashboard. Every object an agent produces or the API exposes is
a validated Pydantic model, so bad LLM output is caught early (at parse time)
rather than surfacing as a crash deep inside a workflow run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.enums import (
    DifficultyLevel,
    LearningSpeed,
    LearningStyle,
    MemoryFactType,
    QuestionType,
    StudentLevel,
)


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #
class StudentProfile(BaseModel):
    """Stable + evolving attributes of a learner."""

    model_config = ConfigDict(from_attributes=True)

    student_id: str
    name: str = "Learner"
    level: StudentLevel = StudentLevel.BEGINNER
    learning_style: LearningStyle = LearningStyle.READING_WRITING
    learning_speed: LearningSpeed = LearningSpeed.NORMAL
    preferred_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    topics_learned: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    quiz_scores: list[float] = Field(default_factory=list)


class ConceptScore(BaseModel):
    """Rolling proficiency of a single concept."""

    concept: str
    proficiency: float = Field(ge=0.0, le=1.0)
    attempts: int = 0


# --------------------------------------------------------------------------- #
# Lesson / explanation
# --------------------------------------------------------------------------- #
class Explanation(BaseModel):
    """The Teacher agent's output for a single concept."""

    topic: str
    summary: str
    explanation: str = Field(description="Simple-language explanation")
    examples: list[str] = Field(default_factory=list)
    analogies: list[str] = Field(default_factory=list)
    recap: str = Field(default="", description="One-paragraph recap")


class LessonPlan(BaseModel):
    """Planner output: the goal of the current tutoring run."""

    topic: str
    learning_goal: str
    level: StudentLevel
    prerequisites: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Quiz
# --------------------------------------------------------------------------- #
class QuizQuestion(BaseModel):
    """A single question inside a quiz."""

    id: str
    type: QuestionType
    topic: str
    question: str
    options: list[str] = Field(default_factory=list)  # MCQ only
    correct_answer: str
    explanation: str = ""
    points: int = 1


class Quiz(BaseModel):
    """A generated quiz. Composition: 5 MCQ + 2 numerical + 1 subjective."""

    id: str
    topic: str
    difficulty: DifficultyLevel
    questions: list[QuizQuestion] = Field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(q.points for q in self.questions)


class AnswerSubmission(BaseModel):
    """One student answer to one question."""

    question_id: str = Field(min_length=1, max_length=80)
    answer: str = Field(max_length=8000)


class QuizSubmission(BaseModel):
    """Full submission for a quiz."""

    quiz_id: str
    student_id: str
    answers: list[AnswerSubmission] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class EvaluationResult(BaseModel):
    """Evaluation agent output for one quiz attempt."""

    score: float
    max_score: float
    percent: float = Field(ge=0.0, le=100.0)
    weak_concepts: list[str] = Field(default_factory=list)
    revision_topics: list[str] = Field(default_factory=list)
    next_lesson: str = ""
    feedback: str = ""


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
class Recommendation(BaseModel):
    """Recommendation agent output."""

    next_topic: str
    rationale: str
    revision_schedule: list[str] = Field(default_factory=list)
    daily_plan: list[str] = Field(default_factory=list)
    weak_areas: list[str] = Field(default_factory=list)
    suggested_quiz_topics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """A retrieved document fragment, with citation info."""

    content: str
    source: str
    page: int | None = None
    score: float = 0.0


class RAGContext(BaseModel):
    """Everything the RAG agent passes to Teacher/Recommendation."""

    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    used: bool = False

    def as_pretty_string(self) -> str:
        """Flatten chunks into a single labelled string for prompt injection."""
        parts = []
        for i, chunk in enumerate(self.retrieved_chunks, start=1):
            loc = f" (p.{chunk.page})" if chunk.page is not None else ""
            parts.append(f"[{i}] Source: {chunk.source}{loc}\n{chunk.content}")
        return "\n\n".join(parts)


class DocumentRecord(BaseModel):
    """Metadata about an uploaded/processed document."""

    id: str
    student_id: str
    filename: str
    status: str
    chunk_count: int = 0
    created_at: str = ""


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class MemoryFact(BaseModel):
    """A single fact persisted to long-term memory."""

    fact_type: MemoryFactType
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class ProgressPoint(BaseModel):
    """A single point on the progress/time chart."""

    date: str
    score: float
    topic: str = ""


class DashboardSummary(BaseModel):
    """Aggregated payload for the dashboard home screen."""

    profile: StudentProfile
    topics_learned: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    quiz_scores: list[float] = Field(default_factory=list)
    progress: list[ProgressPoint] = Field(default_factory=list)
    recommendations: Recommendation | None = None
    conversation_count: int = 0
    document_count: int = 0


# --------------------------------------------------------------------------- #
# API contracts
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    """Body for POST /chat — one student turn in a tutoring conversation."""

    student_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=8000)
    topic: str | None = None
    with_quiz: bool = False
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    run_id: str | None = Field(default=None, min_length=1, max_length=64)
    pause_before: str | None = None


class ChatResponse(BaseModel):
    """Full result of a tutoring run (post-recommendation)."""

    student_id: str
    intent: str
    topic: str
    lesson: Explanation | None = None
    quiz: Quiz | None = None
    evaluation: EvaluationResult | None = None
    recommendations: Recommendation | None = None
    rag_sources: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    runtime: dict = Field(default_factory=dict)
    harness_metrics: dict = Field(default_factory=dict)
    context_metrics: dict = Field(default_factory=dict)
    pending_nodes: list[str] = Field(default_factory=list)
    replayed: bool = False
