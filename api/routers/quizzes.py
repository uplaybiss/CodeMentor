"""Quiz endpoints — list quizzes and view individual quizzes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_container_dep
from services.container import Container
from utils.errors import ResourceNotFoundError

router = APIRouter(prefix="/api/v1", tags=["quizzes"])


@router.get("/quizzes/{student_id}")
async def list_quizzes(
    student_id: str, container: Container = Depends(get_container_dep)
) -> dict:
    """List quiz ids for a student, newest first."""
    quiz_ids = container.quiz_repo.list_quizzes(student_id)
    return {"student_id": student_id, "quiz_ids": quiz_ids}


@router.get("/quiz/{quiz_id}")
async def get_quiz(
    quiz_id: str, container: Container = Depends(get_container_dep)
) -> dict:
    """Get a specific quiz (with questions) by ID."""
    quiz = container.quiz_repo.get_quiz(quiz_id)
    if quiz is None:
        raise ResourceNotFoundError(f"Quiz {quiz_id!r} not found")
    return quiz.model_dump()
