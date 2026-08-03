"""Shared helpers for agent nodes.

LangGraph merges returned dicts back into the state. For ``Annotated[list,
operator.add]`` fields (``messages``, ``memory_updates``, ``errors``) a node
must return a *list containing the new items* so they append instead of
replacing.
"""

from __future__ import annotations

import logging
from typing import Any

from models.schemas import StudentProfile
from models.state import EduAgentState

logger = logging.getLogger(__name__)


def append_message(role: str, content: str, **extra: Any) -> list[dict]:
    """Build the ``messages`` accumulator payload for one assistant/user turn."""
    message = {"role": role, "content": content, **extra}
    return [message]


def student_profile_from_state(
    state: EduAgentState, ctx: Any
) -> StudentProfile:
    """Load (creating if needed) the student profile referenced by the state."""
    student_id = state.get("student_id") or "anonymous"
    return ctx.student_repo.get_or_create(student_id)


def profile_prompt_section(profile: StudentProfile | None) -> str:
    """Render the student-profile context block used in agent prompts."""
    if profile is None:
        return "Student profile: unknown (new learner)."
    return (
        f"Student profile:\n"
        f"- name: {profile.name}\n"
        f"- level: {profile.level.value}\n"
        f"- learning style: {profile.learning_style.value}\n"
        f"- topics learned: {', '.join(profile.topics_learned) or 'none'}\n"
        f"- weak concepts: {', '.join(profile.weak_concepts) or 'none'}"
    )


def memory_prompt_section(ctx: Any, student_id: str, limit: int = 8) -> str:
    """Render recent long-term memory facts for prompt context."""
    facts = ctx.memory_repo.list_facts(student_id, limit=limit)
    if not facts:
        return "Memory: no stored facts yet."
    lines = [f"- ({f['fact_type']}) {f['content']}" for f in facts]
    return "Memory:\n" + "\n".join(lines)


def rag_prompt_section(state: EduAgentState) -> str:
    """Render retrieved chunks (if any) as grounded context for a node."""
    rag_context = state.get("rag_context") or {}
    chunks = rag_context.get("retrieved_chunks") or []
    if not chunks:
        return "Retrieved notes: none."
    parts = [
        f"[{i}] Source: {c.get('source')}"
        + (f" (p.{c['page']})" if c.get("page") is not None else "")
        + f"\n{c.get('content', '')}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "Retrieved notes (ground answers in these):\n\n" + "\n\n".join(parts)
