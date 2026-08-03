"""Quiz agent — generates a fixed-composition quiz for the current topic."""

from __future__ import annotations

import logging

from agents.base import (
    memory_prompt_section,
    profile_prompt_section,
    rag_prompt_section,
    student_profile_from_state,
)
from agents.context import AgentContext
from models.enums import QuestionType
from models.schemas import Quiz
from prompts import render_prompt
from utils.helpers import generate_id

logger = logging.getLogger(__name__)


class QuizAgent:
    """Quiz node: produces a validated :class:`Quiz` and persists it."""

    name = "quiz"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    async def run(self, state: dict) -> dict:
        topic = (state.get("topic") or "general").strip()
        student_id = state.get("student_id") or "anonymous"
        profile = student_profile_from_state(state, self._ctx)

        user_prompt = (
            f"Topic: {topic}\n"
            f"Requested difficulty: {profile.preferred_difficulty.value}\n"
            f"Student level: {state.get('level') or profile.level.value}\n\n"
            f"{profile_prompt_section(profile)}\n\n"
            f"{memory_prompt_section(self._ctx, student_id)}\n\n"
            f"{rag_prompt_section(state)}"
        )
        quiz = await self._ctx.llm.generate_json(
            render_prompt("quiz"),
            user_prompt,
            model_class=Quiz,
        )

        # Guarantee a unique id per generation (demo/provider ids may collide).
        quiz.id = generate_id("quiz")

        self._ctx.quiz_repo.create_quiz(quiz, student_id)
        # Reload from the repo so question ids match what the evaluation agent
        # will load when grading a submission ({quiz_id}q{idx}). Returning the
        # generator's own ids (demoq1, ...) would make every submitted answer
        # miss when the quiz is later graded.
        quiz = self._ctx.quiz_repo.get_quiz(quiz.id) or quiz
        message = self._format_message(quiz)
        logger.info("Quiz agent generated %d questions on %r", len(quiz.questions), topic)
        return {
            "quiz": quiz.model_dump(),
            "messages": [{"role": "assistant", "content": message, "node": self.name}],
        }

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _format_message(quiz: Quiz) -> str:
        """Render the quiz for the student (answers never shown)."""
        lines = [
            f"### Quiz: {quiz.topic}",
            f"Difficulty: {quiz.difficulty.value} · {len(quiz.questions)} questions",
            "",
        ]
        for idx, q in enumerate(quiz.questions, start=1):
            lines.append(f"**Q{idx}.** {q.question}")
            if q.type == QuestionType.MCQ and q.options:
                for opt_idx, option in enumerate(q.options):
                    letter = chr(ord("A") + opt_idx)
                    lines.append(f"  - {letter}) {option}")
            elif q.type == QuestionType.NUMERICAL:
                lines.append("  - (enter a numeric answer)")
            else:
                lines.append("  - (answer in your own words)")
            lines.append("")
        lines.append(
            "Type your answers or submit them in the quiz tab — I'll mark them and "
            "show your weak points."
        )
        return "\n".join(lines)
