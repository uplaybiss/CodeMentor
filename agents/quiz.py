"""Quiz agent — generates a fixed-composition quiz for the current topic."""

from __future__ import annotations

import logging

from agents.base import (
    call_tool,
    memory_prompt_section,
    profile_prompt_section,
    rag_prompt_section,
    student_profile_from_state,
)
from agents.context import AgentContext
from config.curriculum import canonical_topic, difficulty_for
from config.presentation import DIFFICULTY_LABELS, topic_label
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
        profile = await student_profile_from_state({**state, "_task": self.name}, self._ctx)
        difficulty = profile.preferred_difficulty
        programming_topic = canonical_topic(topic)
        if programming_topic:
            scores = await call_tool(self._ctx, self.name, "student.scores", student_id=student_id)
            mastery = next(
                (s.proficiency for s in scores if canonical_topic(s.concept) == programming_topic),
                None,
            )
            difficulty = difficulty_for(mastery)
            topic = programming_topic

        user_prompt = (
            f"Topic: {topic}\n"
            f"Requested difficulty: {difficulty.value}\n"
            f"Student level: {state.get('level') or profile.level.value}\n\n"
            f"{profile_prompt_section(profile)}\n\n"
            f"{memory_prompt_section(self._ctx, student_id)}\n\n"
            f"{rag_prompt_section(state)}"
        )
        quiz = await call_tool(
            self._ctx,
            self.name,
            "llm.json",
            student_id=student_id,
            system=render_prompt("quiz"),
            user=user_prompt,
            model_class=Quiz,
        )

        # Guarantee a unique id per generation (demo/provider ids may collide).
        quiz.id = generate_id("quiz")
        quiz.difficulty = difficulty

        saved = await call_tool(
            self._ctx, self.name, "quiz.create", student_id=student_id, quiz=quiz
        )
        quiz = Quiz.model_validate(saved)
        # Reload from the repo so question ids match what the evaluation agent
        # will load when grading a submission ({quiz_id}q{idx}). Returning the
        # generator's own ids (demoq1, ...) would make every submitted answer
        # miss when the quiz is later graded.
        quiz = await call_tool(
            self._ctx, self.name, "quiz.get", student_id=student_id, quiz_id=quiz.id
        )
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
            f"### 专项练习：{topic_label(quiz.topic)}",
            f"难度：{DIFFICULTY_LABELS[quiz.difficulty.value]} · 共{len(quiz.questions)}题",
            "",
        ]
        for idx, q in enumerate(quiz.questions, start=1):
            lines.append(f"**第{idx}题** {q.question}")
            if q.type == QuestionType.MCQ and q.options:
                for opt_idx, option in enumerate(q.options):
                    letter = chr(ord("A") + opt_idx)
                    lines.append(f"  - {letter}) {option}")
            elif q.type == QuestionType.NUMERICAL:
                lines.append("  - 请填写数值答案")
            else:
                lines.append("  - 请用自己的话回答")
            lines.append("")
        lines.append("请前往专项练习页面作答，提交后即可查看评分与错题反馈。")
        return "\n".join(lines)
