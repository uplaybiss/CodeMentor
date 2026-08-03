"""Immutable constants shared across EduAgent.

Anything that is *not* environment-specific and is referenced in more than one
module lives here so the rest of the codebase never contains magic strings.
Environment-specific values belong in :mod:`config.settings`.
"""

from __future__ import annotations


# --- Agent identity -----------------------------------------------------------
class Agents:
    """Canonical names for every graph node / agent."""

    PLANNER = "planner"
    TEACHER = "teacher"
    QUIZ = "quiz"
    EVALUATION = "evaluation"
    MEMORY = "memory"
    RECOMMENDATION = "recommendation"
    RAG = "rag"
    DASHBOARD = "dashboard"

    ALL = (
        PLANNER,
        TEACHER,
        QUIZ,
        EVALUATION,
        MEMORY,
        RECOMMENDATION,
        RAG,
        DASHBOARD,
    )


# --- Graph routing -------------------------------------------------------------
class Routes:
    """Returned by the planner to drive LangGraph conditional edges."""

    TEACH = "teach"
    QUIZ = "quiz"
    ANSWER_QUESTION = "answer_question"
    RAG_ONLY = "rag_only"
    END = "end"


# --- Quiz composition -----------------------------------------------------------
class QuizComposition:
    """Fixed composition of a generated quiz."""

    MCQ_COUNT = 5
    NUMERICAL_COUNT = 2
    SUBJECTIVE_COUNT = 1
    TOTAL = MCQ_COUNT + NUMERICAL_COUNT + SUBJECTIVE_COUNT


# --- RAG -------------------------------------------------------------------------
# Canonical lifecycle states and difficulty levels live in models/enums.py;
# only pure numbers / name strings shared across modules belong here.
SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})

# --- Misc -------------------------------------------------------------------------
DEFAULT_STUDENT_NAME = "Learner"
MAX_CHUNKS_PER_DOCUMENT = 2000
