"""Chat endpoints — tutoring conversation and quiz submission."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.context import AgentContext
from api.dependencies import get_ctx
from models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _initial_state(req: ChatRequest) -> dict[str, Any]:
    """Build the LangGraph state for a fresh tutoring turn."""
    return {
        "student_id": req.student_id,
        "user_input": req.message,
        "topic": req.topic or None,
        "with_quiz": req.with_quiz,
        "intent": "general",
        "messages": [],
        "errors": [],
        "memory_updates": [],
        "lesson": None,
        "quiz": None,
        "evaluation": None,
        "recommendations": None,
        "rag_context": None,
        "rag_sources": [],
    }


def _last_assistant_message(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages") or []):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return ""


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, ctx: AgentContext = Depends(get_ctx)) -> ChatResponse:
    """Send a message through the tutoring graph and get a reply."""
    from graphs.workflow import build_tutoring_graph

    graph = build_tutoring_graph(ctx)
    final = await graph.ainvoke(_initial_state(req))
    return ChatResponse(
        student_id=req.student_id,
        intent=final.get("intent", "general"),
        topic=final.get("topic") or "",
        lesson=final.get("lesson"),
        quiz=final.get("quiz"),
        evaluation=final.get("evaluation"),
        recommendations=final.get("recommendations"),
        rag_sources=list(final.get("rag_sources") or []),
        messages=list(final.get("messages") or []),
    )


# ── POST /chat/submit-quiz ────────────────────────────────────────────────────


class SubmitQuizRequest(BaseModel):
    quiz_id: str
    student_id: str
    answers: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/chat/submit-quiz")
async def submit_quiz(
    req: SubmitQuizRequest, ctx: AgentContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Submit quiz answers and receive evaluation results."""
    from graphs.workflow import build_evaluation_graph

    graph = build_evaluation_graph(ctx)
    state: dict[str, Any] = {
        "student_id": req.student_id,
        "quiz_id": req.quiz_id,
        "answers": req.answers,
        "messages": [],
        "errors": [],
        "memory_updates": [],
    }

    final = await graph.ainvoke(state)
    evaluation = final.get("evaluation")
    if evaluation is None:
        return {"error": "Evaluation could not be completed."}

    return {
        "evaluation": evaluation,
        "recommendations": final.get("recommendations"),
        "messages": list(final.get("messages") or []),
    }
