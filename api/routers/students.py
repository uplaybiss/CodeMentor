"""Student profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_container_dep
from models.schemas import StudentProfile
from services.container import Container

router = APIRouter(prefix="/api/v1/students", tags=["students"])


class StudentProfileUpdate(BaseModel):
    name: str | None = None
    level: str | None = None
    learning_style: str | None = None


def _to_response(profile: StudentProfile) -> dict:
    return {
        "student_id": profile.student_id,
        "name": profile.name,
        "level": profile.level.value if hasattr(profile.level, "value") else profile.level,
        "learning_style": profile.learning_style.value
        if hasattr(profile.learning_style, "value")
        else profile.learning_style,
        "learning_speed": profile.learning_speed,
        "preferred_difficulty": profile.preferred_difficulty,
        "weak_concepts": profile.weak_concepts or [],
        "topics_learned": profile.topics_learned or [],
        "quiz_scores": profile.quiz_scores or [],
    }


@router.get("/{student_id}")
async def get_student(
    student_id: str, container: Container = Depends(get_container_dep)
) -> dict:
    """Get (or lazily create) a student profile."""
    profile = container.student_repo.get_or_create(student_id)
    return _to_response(profile)


@router.put("/{student_id}")
async def update_student(
    student_id: str,
    update: StudentProfileUpdate,
    container: Container = Depends(get_container_dep),
) -> dict:
    """Update editable profile fields."""
    container.student_repo.get_or_create(student_id)
    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    profile = container.student_repo.update_profile(student_id, **updates)
    return _to_response(profile)
