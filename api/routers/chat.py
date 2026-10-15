"""Chat endpoints — tutoring conversation and quiz submission."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.context import AgentContext
from api.dependencies import get_ctx
from models.schemas import AnswerSubmission, ChatRequest, ChatResponse

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
    final = await ctx.runtime.execute(
        ctx,
        "tutoring",
        _initial_state(req),
        session_id=req.session_id,
        run_id=req.run_id,
        pause_before=req.pause_before,
    )
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
        runtime=final["runtime"],
        harness_metrics=final.get("harness_metrics") or {},
        context_metrics=final.get("context_metrics") or {},
        pending_nodes=final.get("pending_nodes") or [],
        replayed=final.get("replayed", False),
    )


# ── POST /chat/submit-quiz ────────────────────────────────────────────────────


class SubmitQuizRequest(BaseModel):
    quiz_id: str = Field(min_length=1, max_length=64)
    student_id: str = Field(min_length=1, max_length=64)
    answers: list[AnswerSubmission] = Field(default_factory=list, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    run_id: str | None = Field(default=None, min_length=1, max_length=64)
    pause_before: str | None = None


@router.post("/chat/submit-quiz")
async def submit_quiz(
    req: SubmitQuizRequest, ctx: AgentContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Submit quiz answers and receive evaluation results."""
    state: dict[str, Any] = {
        "student_id": req.student_id,
        "quiz_id": req.quiz_id,
        "answers": [answer.model_dump() for answer in req.answers],
        "messages": [],
        "errors": [],
        "memory_updates": [],
    }

    final = await ctx.runtime.execute(
        ctx,
        "evaluation",
        state,
        session_id=req.session_id,
        run_id=req.run_id,
        pause_before=req.pause_before,
    )
    evaluation = final.get("evaluation")
    if evaluation is None:
        return {"runtime": final["runtime"], "pending_nodes": final.get("pending_nodes", [])}

    return {
        "evaluation": evaluation,
        "question_feedback": final.get("question_feedback") or [],
        "recommendations": final.get("recommendations"),
        "messages": list(final.get("messages") or []),
        "runtime": final["runtime"],
        "harness_metrics": final.get("harness_metrics") or {},
        "context_metrics": final.get("context_metrics") or {},
        "replayed": final.get("replayed", False),
    }


class ResumeRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    run_id: str = Field(min_length=1, max_length=64)
    workflow: Literal["tutoring", "evaluation"] = "tutoring"


@router.post("/chat/resume")
async def resume(req: ResumeRequest, ctx: AgentContext = Depends(get_ctx)) -> dict:
    return await ctx.runtime.resume(ctx, req.student_id, req.session_id, req.run_id, req.workflow)


@router.get("/chat/runs/{session_id}/{run_id}")
async def run_status(
    session_id: str,
    run_id: str,
    student_id: str,
    workflow: Literal["tutoring", "evaluation"] = "tutoring",
    ctx: AgentContext = Depends(get_ctx),
) -> dict:
    return ctx.runtime.status(student_id, session_id, run_id, workflow)
