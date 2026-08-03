"""Recommendation agent — proposes the student's next learning step."""

from __future__ import annotations

import logging

from agents.base import (
    memory_prompt_section,
    profile_prompt_section,
    student_profile_from_state,
)
from agents.context import AgentContext
from models.schemas import Recommendation
from prompts import render_prompt

logger = logging.getLogger(__name__)


class RecommendationAgent:
    """Recommendation node: produces a :class:`Recommendation` and persists it."""

    name = "recommendation"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    async def run(self, state: dict) -> dict:
        student_id = state.get("student_id") or "anonymous"
        profile = student_profile_from_state(state, self._ctx)
        concept_scores = self._ctx.student_repo.list_concept_scores(student_id)
        weak = list(profile.weak_concepts or []) or [
            c.concept for c in concept_scores if c.proficiency < 0.6
        ]
        evaluation = state.get("evaluation") or {}

        score_lines = "\n".join(
            f"- {s.concept}: {s.proficiency:.0%} ({s.attempts} attempts)"
            for s in concept_scores[:10]
        ) or "- (no concept scores yet)"

        user_prompt = (
            f"{profile_prompt_section(profile)}\n\n"
            f"Concept scores:\n{score_lines}\n\n"
            f"Weak concepts: {', '.join(weak) or 'none identified'}\n"
            f"Latest quiz: {evaluation.get('percent') if evaluation else 'none'}"
            f" ({evaluation.get('weak_concepts') if evaluation else ''})\n"
            f"Feedback: {evaluation.get('feedback') or 'none'}\n\n"
            f"{memory_prompt_section(self._ctx, student_id)}"
        )
        rec = await self._ctx.llm.generate_json(
            render_prompt("recommendation"),
            user_prompt,
            model_class=Recommendation,
        )
        self._ctx.memory_repo.save_recommendation(student_id, rec)

        message = self._format_message(rec)
        logger.info("Recommendation agent suggested next topic: %r", rec.next_topic)
        return {
            "recommendations": rec.model_dump(),
            "messages": [
                {"role": "assistant", "content": message, "node": self.name}
            ],
        }

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _format_message(rec: Recommendation) -> str:
        lines = [f"### Recommended next step: {rec.next_topic}", rec.rationale]
        if rec.revision_schedule:
            lines += ["", "**Revision schedule:**"]
            lines += [f"- {item}" for item in rec.revision_schedule]
        if rec.daily_plan:
            lines += ["", "**Daily plan:**"]
            lines += [f"- {item}" for item in rec.daily_plan]
        if rec.suggested_quiz_topics:
            lines += [
                "",
                f"**Suggested quiz topics:** {', '.join(rec.suggested_quiz_topics)}",
            ]
        return "\n".join(lines)
