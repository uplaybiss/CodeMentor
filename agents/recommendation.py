"""Recommendation agent — proposes the student's next learning step."""

from __future__ import annotations

import logging

from agents.base import (
    call_tool,
    memory_prompt_section,
    profile_prompt_section,
    student_profile_from_state,
)
from agents.context import AgentContext
from config.curriculum import canonical_topic, recommend_next
from config.presentation import DIFFICULTY_LABELS, policy_rationale, topic_label
from harness.control import current_projection
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
        profile = await student_profile_from_state({**state, "_task": self.name}, self._ctx)
        concept_scores = await call_tool(
            self._ctx, self.name, "student.scores", student_id=student_id
        )
        weak = list(profile.weak_concepts or []) or [
            c.concept for c in concept_scores if c.proficiency < 0.6
        ]
        evaluation = state.get("evaluation") or {}

        score_lines = (
            "\n".join(
                f"- {s.concept}: {s.proficiency:.0%} ({s.attempts} attempts)"
                for s in concept_scores[:10]
            )
            or "- (no concept scores yet)"
        )

        user_prompt = (
            f"{profile_prompt_section(profile)}\n\n"
            f"Concept scores:\n{score_lines}\n\n"
            f"Weak concepts: {', '.join(weak) or 'none identified'}\n"
            f"Latest quiz: {evaluation.get('percent') if evaluation else 'none'}"
            f" ({evaluation.get('weak_concepts') if evaluation else ''})\n"
            f"Feedback: {evaluation.get('feedback') or 'none'}\n\n"
            f"{memory_prompt_section(self._ctx, student_id)}"
        )
        if current_projection.get() is not None:
            user_prompt = profile_prompt_section(profile)
        rec = await call_tool(
            self._ctx,
            self.name,
            "llm.json",
            student_id=student_id,
            system=render_prompt("recommendation"),
            user=user_prompt,
            model_class=Recommendation,
        )
        if canonical_topic(state.get("topic")) or any(
            canonical_topic(s.concept) for s in concept_scores
        ):
            policy = recommend_next({s.concept: s.proficiency for s in concept_scores})
            rec.next_topic = policy["next_topic"]
            rec.rationale = policy_rationale(
                policy,
                {canonical_topic(s.concept) or s.concept: s.proficiency for s in concept_scores},
            )
            rec.suggested_quiz_topics = [policy["next_topic"]]
            rec.weak_areas = [s.concept for s in concept_scores if s.proficiency < 0.6]
            rec.daily_plan = [
                f"先复习{topic_label(policy['next_topic'])}，再完成{DIFFICULTY_LABELS[policy['difficulty']]}练习。"
            ]
            rec.revision_schedule = [
                f"完成基础复习后，再测验{topic_label(policy['target_topic'])}。"
            ]
        saved = await call_tool(
            self._ctx, self.name, "recommendation.save", student_id=student_id, recommendation=rec
        )
        rec = Recommendation.model_validate(saved)

        message = self._format_message(rec)
        logger.info("Recommendation agent suggested next topic: %r", rec.next_topic)
        return {
            "recommendations": rec.model_dump(),
            "messages": [{"role": "assistant", "content": message, "node": self.name}],
        }

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _format_message(rec: Recommendation) -> str:
        lines = [f"### 下一步学习建议：{topic_label(rec.next_topic)}", rec.rationale]
        if rec.revision_schedule:
            lines += ["", "**复习安排：**"]
            lines += [f"- {item}" for item in rec.revision_schedule]
        if rec.daily_plan:
            lines += ["", "**今日计划：**"]
            lines += [f"- {item}" for item in rec.daily_plan]
        if rec.suggested_quiz_topics:
            lines += [
                "",
                f"**建议练习：** {'、'.join(topic_label(x) for x in rec.suggested_quiz_topics)}",
            ]
        return "\n".join(lines)
