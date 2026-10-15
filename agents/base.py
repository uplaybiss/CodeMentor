"""Shared helpers for agent nodes.

LangGraph merges returned dicts back into the state. For ``Annotated[list,
operator.add]`` fields (``messages``, ``memory_updates``, ``errors``) a node
must return a *list containing the new items* so they append instead of
replacing.
"""

from __future__ import annotations

import logging
from typing import Any

from harness.control import current_projection
from models.schemas import StudentProfile
from models.state import EduAgentState

logger = logging.getLogger(__name__)


def append_message(role: str, content: str, **extra: Any) -> list[dict]:
    """Build the ``messages`` accumulator payload for one assistant/user turn."""
    message = {"role": role, "content": content, **extra}
    return [message]


async def call_tool(ctx: Any, task: str, name: str, **params: Any) -> Any:
    """All agent model/repository operations enter the same typed dispatch guard."""
    if ctx.harness is None:
        from harness.agent_harness import AgentHarness

        ctx.harness = AgentHarness(ctx.student_repo.database, ctx.settings.harness)
    return await ctx.harness.call(ctx, task, name, **params)


async def student_profile_from_state(state: EduAgentState, ctx: Any) -> StudentProfile:
    """Load (creating if needed) the student profile referenced by the state."""
    student_id = state.get("student_id") or "anonymous"
    ctx.student_repo.get_or_create(student_id)  # bootstrap, not an agent tool
    from harness.control import current_control

    task = current_control.get().node if current_control.get() else state.get("_task", "planner")
    return await call_tool(ctx, task, "student.profile", student_id=student_id)


def profile_prompt_section(profile: StudentProfile | None) -> str:
    """Render the student-profile context block used in agent prompts."""
    projection = current_projection.get()
    if projection is not None:
        return "Task learning context:\n" + projection["text"]
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
    if current_projection.get() is not None:
        return ""  # relevant memories are already in the bounded task projection
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
