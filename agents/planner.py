"""Planner agent — classifies intent, picks a topic and routes the graph."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agents.base import (
    memory_prompt_section,
    profile_prompt_section,
    student_profile_from_state,
)
from agents.context import AgentContext
from config.constants import Routes
from models.enums import ChatIntent, StudentLevel
from prompts import render_prompt

logger = logging.getLogger(__name__)


class PlannerOutput(BaseModel):
    """Structured output of the Planner agent."""

    intent: ChatIntent
    topic: str
    learning_goal: str
    level: StudentLevel = StudentLevel.BEGINNER
    prerequisites: list[str] = Field(default_factory=list)


class PlannerAgent:
    """First node in the tutoring graph."""

    name = "planner"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    # ------------------------------------------------------------- routing --
    @staticmethod
    def route(intent: ChatIntent) -> str:
        """Map a classified intent to the next graph node (config.constants.Routes)."""
        if intent == ChatIntent.PRACTICE:
            return Routes.QUIZ
        if intent == ChatIntent.ANSWER_QUESTION:
            return Routes.ANSWER_QUESTION
        if intent == ChatIntent.LEARN:
            return Routes.TEACH
        # GENERAL / UPLOAD terminate the graph; the planner supplies the reply.
        return Routes.END

    # ------------------------------------------------------------------ run --
    async def run(self, state: dict) -> dict:
        """Classify the student's message and decide the next node."""
        user_input = (state.get("user_input") or "").strip()
        student_id = state.get("student_id") or "anonymous"
        profile = student_profile_from_state(state, self._ctx)

        user_prompt = (
            f"Student message:\n{user_input}\n\n"
            f"{profile_prompt_section(profile)}\n\n"
            f"{memory_prompt_section(self._ctx, student_id)}"
        )
        output = await self._ctx.llm.generate_json(
            render_prompt("planner"),
            user_prompt,
            model_class=PlannerOutput,
        )

        next_agent = self.route(output.intent)
        updates: dict = {
            "intent": output.intent.value,
            "topic": state.get("topic") or output.topic,
            "learning_goal": output.learning_goal,
            "level": output.level.value,
            "next_agent": next_agent,
        }

        if next_agent == Routes.END:
            # GENERAL / UPLOAD: answer conversationally and finish the turn.
            reply = await self._ctx.llm.generate(
                "You are a warm, friendly AI tutor. Reply to the student's "
                "message briefly (1-3 sentences) and naturally, without "
                "structured lesson output.",
                f"Student: {user_input}",
            )
            updates["messages"] = [
                {
                    "role": "assistant",
                    "content": reply,
                    "node": self.name,
                }
            ]
            updates["lesson"] = None
            logger.info(
                "Planner routed %r -> %s (END, general reply)", user_input[:60], next_agent
            )
        else:
            logger.info(
                "Planner routed %r -> %s (intent=%s, topic=%s)",
                user_input[:60],
                next_agent,
                output.intent.value,
                output.topic,
            )
        return updates
