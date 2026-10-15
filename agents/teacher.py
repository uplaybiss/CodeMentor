"""Teacher agent — generates the lesson/explanation for the current topic."""

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
from config.curriculum import canonical_topic
from config.presentation import topic_label
from models.schemas import Explanation
from prompts import render_prompt

logger = logging.getLogger(__name__)


class TeacherAgent:
    """Lesson node: produces a validated :class:`Explanation` and persists it."""

    name = "teacher"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    async def run(self, state: dict) -> dict:
        topic = (state.get("topic") or "general").strip()
        student_id = state.get("student_id") or "anonymous"
        profile = await student_profile_from_state({**state, "_task": self.name}, self._ctx)

        user_prompt = (
            f"Topic: {topic}\n"
            f"Student level: {state.get('level') or profile.level.value}\n"
            f"Learning goal: {state.get('learning_goal') or 'understand the basics'}\n\n"
            f"{profile_prompt_section(profile)}\n\n"
            f"{memory_prompt_section(self._ctx, student_id)}\n\n"
            f"{rag_prompt_section(state)}"
        )
        lesson = await call_tool(
            self._ctx,
            self.name,
            "llm.json",
            student_id=student_id,
            system=render_prompt("teacher"),
            user=user_prompt,
            model_class=Explanation,
        )

        # Persist the lesson and record the topic as covered.
        saved = await call_tool(
            self._ctx, self.name, "lesson.save", student_id=student_id, lesson=lesson
        )
        lesson = Explanation.model_validate(saved)

        message = self._format_message(lesson)
        logger.info("Teacher delivered lesson on %r", lesson.topic)
        return {
            "lesson": lesson.model_dump(),
            "messages": [{"role": "assistant", "content": message, "node": self.name}],
        }

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _format_message(lesson: Explanation) -> str:
        blocks = [
            f"### {topic_label(lesson.topic)}",
            lesson.summary,
            "",
            lesson.explanation,
        ]
        if lesson.examples:
            blocks += ["", "**代码示例：**"]
            blocks += [
                f"```python\n{ex}\n```" if canonical_topic(lesson.topic) else f"- {ex}"
                for ex in lesson.examples
            ]
        if lesson.analogies:
            blocks += ["", "**换个角度理解：**"]
            blocks += [f"- {an}" for an in lesson.analogies]
        if lesson.recap:
            blocks += ["", f"**要点回顾：** {lesson.recap}"]
        return "\n".join(blocks)
