"""Typed adapters for existing Python repositories/services; not an LLM tool agent."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from database.models import Quiz as QuizRow
from database.models import WrongAnswer
from harness.errors import ToolPermissionError
from models.schemas import EvaluationResult, Explanation, MemoryFact, Quiz, Recommendation
from utils.errors import ResourceNotFoundError
from utils.helpers import utc_now_iso


class StudentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    student_id: str = Field(min_length=1, max_length=64)


class JsonInput(StudentInput):
    system: str
    user: str
    model_class: type[BaseModel]


class TextInput(StudentInput):
    system: str
    user: str


class QuizLookupInput(StudentInput):
    quiz_id: str = Field(min_length=1, max_length=64)


class RetrievalInput(StudentInput):
    query: str = Field(min_length=1, max_length=8000)


class LessonInput(StudentInput):
    lesson: Explanation


class QuizInput(StudentInput):
    quiz: Quiz


class EvaluationInput(StudentInput):
    quiz: Quiz
    answers: dict[str, str]
    evaluation: EvaluationResult
    per_topic: dict[str, list[tuple[bool, float]]]
    wrong_answers: list[dict] = Field(default_factory=list)


class MemoryInput(StudentInput):
    facts: list[MemoryFact]
    messages: list[dict]
    user_input: str = ""
    topic: str = "general"


class RecommendationInput(StudentInput):
    recommendation: Recommendation


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable
    schema: type[BaseModel]
    tasks: frozenset[str]
    readonly: bool = False
    retry_safe: bool = False
    model_call: bool = False


def register_tools(ctx) -> dict[str, ToolSpec]:
    """Reuse the original repositories and score update; expose explicit task policies."""
    all_nodes = frozenset(("planner", "teacher", "quiz", "evaluation", "memory", "recommendation"))
    db = ctx.student_repo.database

    async def llm_json(student_id, system, user, model_class):
        return await ctx.llm.generate_json(system, user, model_class=model_class)

    async def llm_text(student_id, system, user):
        return await ctx.llm.generate(system, user)

    def quiz_get(student_id, quiz_id):
        with db.session() as session:
            owner = session.execute(
                select(QuizRow.student_id).where(QuizRow.id == quiz_id)
            ).scalar_one_or_none()
        if owner is None:
            raise ResourceNotFoundError(f"Quiz {quiz_id!r} not found")
        if owner != student_id:
            raise ToolPermissionError("The quiz belongs to another student")
        return ctx.quiz_repo.get_quiz(quiz_id)

    def lesson_save(student_id, lesson):
        ctx.lesson_repo.save_lesson(student_id, lesson)
        ctx.student_repo.add_topic_learned(student_id, lesson.topic)
        return lesson.model_dump(mode="json")

    def quiz_create(student_id, quiz):
        ctx.quiz_repo.create_quiz(quiz, student_id)
        return ctx.quiz_repo.get_quiz(quiz.id).model_dump(mode="json")

    def evaluation_save(student_id, quiz, answers, evaluation, per_topic, wrong_answers):
        from agents.evaluation import EvaluationAgent

        # Repository sessions join the guard's transaction. Preserve the original EMA logic.
        quiz_get(student_id, quiz.id)
        EvaluationAgent(ctx)._persist(student_id, quiz.id, quiz, answers, evaluation, per_topic)
        with db.session() as session:
            for mistake in wrong_answers:
                session.add(
                    WrongAnswer(
                        student_id=student_id,
                        topic=mistake["topic"],
                        quiz_id=quiz.id,
                        question=mistake["question"],
                        submitted=mistake["submitted"],
                        explanation=mistake["explanation"],
                        created_at=utc_now_iso(),
                    )
                )
        return evaluation.model_dump(mode="json")

    def memory_save(student_id, facts, messages, user_input, topic):
        for fact in facts:
            ctx.memory_repo.add_fact(
                student_id, fact.fact_type.value, fact.content, fact.importance
            )
        if user_input:
            ctx.conversation_repo.add_message(student_id, "user", user_input, {"topic": topic})
        for message in messages:
            if message.get("role") == "assistant" and message.get("content"):
                ctx.conversation_repo.add_message(
                    student_id, "assistant", message["content"], {"topic": topic}
                )
        return {"facts": [fact.model_dump(mode="json") for fact in facts]}

    def recommendation_save(student_id, recommendation):
        ctx.memory_repo.save_recommendation(student_id, recommendation)
        return recommendation.model_dump(mode="json")

    specs = [
        ToolSpec(
            "student.profile",
            ctx.student_repo.get_or_404,
            StudentInput,
            all_nodes,
            True,
            True,
        ),
        ToolSpec(
            "student.scores",
            ctx.student_repo.list_concept_scores,
            StudentInput,
            frozenset(("quiz", "recommendation", "planner")),
            True,
            True,
        ),
        ToolSpec("llm.json", llm_json, JsonInput, all_nodes, model_call=True),
        ToolSpec("llm.text", llm_text, TextInput, frozenset(("planner",)), model_call=True),
        ToolSpec(
            "rag.context",
            lambda student_id, query: ctx.rag.build_context(query),
            RetrievalInput,
            frozenset(("rag",)),
            True,
            True,
        ),
        ToolSpec("lesson.save", lesson_save, LessonInput, frozenset(("teacher",))),
        ToolSpec("quiz.create", quiz_create, QuizInput, frozenset(("quiz",))),
        ToolSpec(
            "quiz.get", quiz_get, QuizLookupInput, frozenset(("quiz", "evaluation")), True, True
        ),
        ToolSpec("evaluation.save", evaluation_save, EvaluationInput, frozenset(("evaluation",))),
        ToolSpec("memory.save", memory_save, MemoryInput, frozenset(("memory",))),
        ToolSpec(
            "recommendation.save",
            recommendation_save,
            RecommendationInput,
            frozenset(("recommendation",)),
        ),
    ]
    return {spec.name: spec for spec in specs}
