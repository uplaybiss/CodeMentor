"""Dashboard summary endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_container_dep
from services.container import Container

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/{student_id}")
async def get_dashboard(
    student_id: str, container: Container = Depends(get_container_dep)
) -> dict:
    """Get a summary dashboard for the student."""
    profile = container.student_repo.get_or_create(student_id)
    attempts = container.quiz_repo.list_attempts(student_id, limit=50)
    memory_facts = container.memory_repo.list_facts(student_id, limit=10)
    recommendation = container.memory_repo.latest_recommendation(student_id)
    doc_count = container.document_repo.count_for_student(student_id)

    return {
        "student_id": student_id,
        "name": profile.name,
        "level": profile.level.value if hasattr(profile.level, "value") else profile.level,
        "learning_style": profile.learning_style.value
        if hasattr(profile.learning_style, "value")
        else profile.learning_style,
        "total_attempts": len(attempts),
        "total_documents": doc_count,
        "quiz_scores": profile.quiz_scores or [],
        "attempt_history": attempts,
        "weak_concepts": profile.weak_concepts or [],
        "topics_learned": profile.topics_learned or [],
        "recent_memory_facts": [
            {"type": f["fact_type"], "content": f["content"]} for f in memory_facts
        ],
        "latest_recommendation": (
            recommendation.model_dump() if recommendation is not None else None
        ),
    }
