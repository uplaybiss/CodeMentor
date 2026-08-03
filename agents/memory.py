"""Memory agent — extracts durable facts and persists the conversation."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agents.base import (
    memory_prompt_section,
    profile_prompt_section,
    student_profile_from_state,
)
from agents.context import AgentContext
from models.schemas import MemoryFact
from prompts import render_prompt

logger = logging.getLogger(__name__)


class MemoryExtraction(BaseModel):
    """Structured output of the Memory agent."""

    facts: list[MemoryFact] = Field(default_factory=list)


class MemoryAgent:
    """Memory node: extracts facts, stores them and logs the transcript."""

    name = "memory"

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    async def run(self, state: dict) -> dict:
        student_id = state.get("student_id") or "anonymous"
        user_input = (state.get("user_input") or "").strip()
        profile = student_profile_from_state(state, self._ctx)

        assistant_text = "\n".join(
            m.get("content", "")
            for m in (state.get("messages") or [])
            if m.get("role") == "assistant" and m.get("content")
        )
        evaluation = state.get("evaluation") or {}

        # Graceful no-op when called with no meaningful content to extract.
        has_content = user_input or assistant_text or evaluation.get("feedback")
        if not has_content:
            logger.debug("Memory agent: nothing to extract; skipping")
            return {"memory_updates": []}

        user_prompt = (
            f"Student message:\n{user_input}\n\n"
            f"Assistant response:\n{assistant_text[:2000]}\n\n"
            f"Latest evaluation: {evaluation.get('feedback') or 'none'}\n\n"
            f"{profile_prompt_section(profile)}\n\n"
            f"{memory_prompt_section(self._ctx, student_id, limit=6)}"
        )
        output = await self._ctx.llm.generate_json(
            render_prompt("memory"),
            user_prompt,
            model_class=MemoryExtraction,
        )

        # Persist extracted facts.
        payload: list[dict] = []
        for fact in output.facts:
            self._ctx.memory_repo.add_fact(
                student_id,
                fact.fact_type.value,
                fact.content,
                fact.importance,
            )
            payload.append(fact.model_dump())
            logger.debug("Stored memory fact [%s]: %s", fact.fact_type.value, fact.content)

        # Persist the conversation transcript (only for real user turns).
        if user_input:
            self._ctx.conversation_repo.add_message(student_id, "user", user_input)
        for message in state.get("messages") or []:
            if message.get("role") == "assistant" and message.get("content"):
                self._ctx.conversation_repo.add_message(
                    student_id, "assistant", message["content"]
                )

        logger.info("Memory agent stored %d fact(s) for student %s", len(payload), student_id)
        return {"memory_updates": payload}
