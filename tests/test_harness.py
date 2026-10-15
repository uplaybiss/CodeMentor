"""Acceptance tests exercise the dispatched workflow and observable side effects."""

import asyncio
import threading
import time

import pytest
from sqlalchemy import func, select

from config.curriculum import CURRICULUM, recommend_next
from config.settings import HarnessSettings
from database.models import ConceptScoreRow, HarnessEffect, WrongAnswer
from graphs.workflow import build_tutoring_graph
from harness.context_builder import ContextBuilder, count_tokens
from harness.control import RunControl, current_control
from harness.errors import (
    ContextBudgetError,
    RunConflictError,
    ToolBudgetError,
    ToolExecutionError,
    ToolParameterError,
    ToolPermissionError,
    ToolRepeatError,
    ToolTimeoutError,
)
from harness.tool_guard import ToolGuard
from harness.tools import StudentInput, ToolSpec
from models.schemas import Quiz, QuizQuestion
from services.demo_content_zh import CONTENT_ZH
from utils.helpers import utc_now_iso


def state(student="alice", topic="Recursion"):
    return {
        "student_id": student,
        "topic": topic,
        "user_input": f"teach me {topic}",
        "messages": [],
        "errors": [],
        "memory_updates": [],
    }


def seed_quiz(ctx, student="alice"):
    ctx.student_repo.get_or_create(student)
    quiz = Quiz(
        id=f"quiz_{student}",
        topic="Recursion",
        difficulty="beginner",
        questions=[
            QuizQuestion(
                id="q1",
                type="mcq",
                topic="Recursion",
                question="What ends recursion?",
                options=["base case", "print"],
                correct_answer="base case",
                explanation="A reachable base case.",
            ),
            QuizQuestion(
                id="q2",
                type="numerical",
                topic="Recursion",
                question="factorial(4)?",
                correct_answer="24",
            ),
        ],
    )
    ctx.quiz_repo.create_quiz(quiz, student)
    return ctx.quiz_repo.get_quiz(quiz.id)


def guarded_fixture(ctx, settings=None):
    return ToolGuard(
        ctx.student_repo.database, settings or HarnessSettings(), ctx.harness.guard.tools.copy()
    )


async def test_context_sql_filter_priority_and_budget(ctx):
    quiz = seed_quiz(ctx)
    ctx.student_repo.get_or_create("bob", name="BOB_SECRET")
    ctx.student_repo.upsert_concept_score("alice", "Recursion", 0.25)
    for index in range(60):
        ctx.memory_repo.add_fact(
            "alice", "concept", f"Recursion note {index}: " + "base case " * 60
        )
        ctx.memory_repo.add_fact("alice", "concept", "UNRELATED_BINARY_TREE " * 60)
        ctx.memory_repo.add_fact("bob", "concept", "Recursion BOB_SECRET " * 60)
        ctx.conversation_repo.add_message(
            "alice", "user", "Recursion history " * 60, {"topic": "Recursion"}
        )
    with ctx.student_repo.database.session() as session:
        for index in range(8):
            session.add(
                WrongAnswer(
                    student_id="alice",
                    quiz_id=quiz.id,
                    topic="Recursion",
                    question=f"RELEVANT_MISTAKE_{index}",
                    submitted="no base case",
                    explanation="Add a terminating case.",
                    created_at=utc_now_iso(),
                )
            )
    projection = ContextBuilder(
        ctx.student_repo.database, HarnessSettings(context_tokens=256, recent_mistakes=2)
    ).build(state(), "teacher")
    assert projection["metrics"]["candidate_tokens"] > 256
    assert count_tokens(projection["text"]) <= 256
    assert projection["payload"]["topic"] == "Recursion"
    assert projection["payload"]["mastery"] == {"score": 0.25, "attempts": 1}
    assert projection["payload"]["goal"]
    assert "RELEVANT_MISTAKE_7" in projection["text"]
    assert len(projection["payload"]["mistakes"]) <= 2
    assert "BOB_SECRET" not in projection["text"]
    assert "UNRELATED_BINARY_TREE" not in projection["text"]
    assert not projection["payload"]["recent"]  # lowest priority dropped first


async def test_required_context_overflow_is_explicit(ctx):
    ctx.student_repo.get_or_create("alice")
    with pytest.raises(ContextBudgetError):
        ContextBuilder(ctx.student_repo.database, HarnessSettings(context_tokens=128)).build(
            {**state(), "learning_goal": "required goal " * 500}, "teacher"
        )


async def test_current_mastery_is_not_lost_in_a_large_profile(ctx):
    ctx.student_repo.get_or_create("alice")
    ctx.student_repo.upsert_concept_score("alice", "Recursion", 0.95)
    with ctx.student_repo.database.session() as session:
        session.add_all(
            [
                ConceptScoreRow(
                    student_id="alice", concept=f"other_{index}", proficiency=0.1, attempts=1
                )
                for index in range(80)
            ]
        )
    projected = ContextBuilder(ctx.student_repo.database, HarnessSettings()).build(
        state(), "teacher"
    )
    assert projected["payload"]["mastery"] == {"score": 0.95, "attempts": 1}


async def test_projection_reaches_actual_llm_prompt(ctx, monkeypatch):
    ctx.student_repo.get_or_create("alice")
    ctx.memory_repo.add_fact("alice", "concept", "Recursion ALICE_MARKER")
    ctx.student_repo.get_or_create("bob")
    ctx.memory_repo.add_fact("bob", "concept", "Recursion BOB_SECRET")
    prompts = []
    original = ctx.llm.generate_json

    async def capture(system, user, **kwargs):
        prompts.append((kwargs["model_class"].__name__, user))
        return await original(system, user, **kwargs)

    monkeypatch.setattr(ctx.llm, "generate_json", capture)
    final = await build_tutoring_graph(ctx).ainvoke(state())
    teacher_prompt = next(prompt for model, prompt in prompts if model == "Explanation")
    assert "Task learning context:" in teacher_prompt
    assert "ALICE_MARKER" in teacher_prompt
    assert "BOB_SECRET" not in teacher_prompt
    assert final["harness_metrics"]["tool_calls"] > 0
    assert any(e["tool"] == "lesson.save" for e in final["harness_metrics"]["events"])


@pytest.mark.parametrize("kind", ["forbidden", "parameter", "student_mismatch"])
async def test_invalid_dispatch_never_calls_tool(ctx, kind):
    guard = guarded_fixture(ctx)
    control = RunControl("alice", "teacher")
    token = current_control.set(control)
    try:
        error = ToolParameterError if kind == "parameter" else ToolPermissionError
        with pytest.raises(error):
            if kind == "forbidden":
                await guard.call("teacher", "memory.save", student_id="alice")
            elif kind == "parameter":
                await guard.call("teacher", "student.profile", student_id="alice", invalid=True)
            else:
                await guard.call("teacher", "student.profile", student_id="bob")
        assert control.tool_calls == 0
        assert ctx.memory_repo.list_facts("alice") == []
    finally:
        current_control.reset(token)


async def test_repeat_limit_and_readonly_retry(ctx):
    calls = []

    def unstable(student_id):
        calls.append(student_id)
        if len(calls) == 1:
            raise ConnectionError("transient local read")
        return "ok"

    spec = ToolSpec("test.read", unstable, StudentInput, frozenset({"teacher"}), True, True)
    guard = ToolGuard(ctx.student_repo.database, HarnessSettings(), {spec.name: spec})
    control = RunControl("alice", "teacher")
    token = current_control.set(control)
    try:
        assert await guard.call("teacher", "test.read", student_id="alice") == "ok"
        assert control.tool_calls == 2
        assert await guard.call("teacher", "test.read", student_id="alice") == "ok"
        with pytest.raises(ToolRepeatError):
            await guard.call("teacher", "test.read", student_id="alice")
        assert len(calls) == 3
    finally:
        current_control.reset(token)


async def test_write_failure_rolls_back_and_is_not_retried(ctx):
    ctx.student_repo.get_or_create("alice")
    calls = []

    def failing_write(student_id):
        calls.append(student_id)
        ctx.memory_repo.add_fact(student_id, "concept", "MUST_ROLL_BACK")
        raise ConnectionError("failure after insert")

    spec = ToolSpec("test.write", failing_write, StudentInput, frozenset({"teacher"}))
    guard = ToolGuard(
        ctx.student_repo.database, HarnessSettings(readonly_retries=2), {spec.name: spec}
    )
    with pytest.raises(ToolExecutionError):
        await guard.call("teacher", "test.write", student_id="alice")
    assert calls == ["alice"]
    assert not ctx.memory_repo.list_facts("alice")


async def test_read_timeout_does_not_claim_worker_stopped(ctx):
    completed = threading.Event()

    def slow_read(student_id):
        time.sleep(0.06)
        completed.set()
        return "late"

    spec = ToolSpec("test.read", slow_read, StudentInput, frozenset({"teacher"}), True)
    guard = ToolGuard(
        ctx.student_repo.database, HarnessSettings(tool_timeout_seconds=0.01), {spec.name: spec}
    )
    with pytest.raises(ToolTimeoutError) as failure:
        await guard.call("teacher", "test.read", student_id="alice")
    assert failure.value.detail["downstream_stopped"] is False
    assert await asyncio.to_thread(completed.wait, 0.5)  # worker completes after caller timed out


async def test_model_timeout_requests_coroutine_cancellation(ctx):
    cancelled = []

    async def slow_model(student_id):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(True)

    spec = ToolSpec("test.model", slow_model, StudentInput, frozenset({"teacher"}), model_call=True)
    guard = ToolGuard(
        ctx.student_repo.database, HarnessSettings(tool_timeout_seconds=0.01), {spec.name: spec}
    )
    with pytest.raises(ToolTimeoutError):
        await guard.call("teacher", "test.model", student_id="alice")
    assert cancelled == [True]


async def test_soft_write_timeout_then_replay_keeps_single_effect(ctx):
    ctx.student_repo.get_or_create("alice")

    def slow_write(student_id):
        ctx.memory_repo.add_fact(student_id, "concept", "SINGLE_EFFECT")
        time.sleep(0.03)
        return {"saved": True}

    spec = ToolSpec("test.write", slow_write, StudentInput, frozenset({"teacher"}))
    guard = ToolGuard(
        ctx.student_repo.database, HarnessSettings(tool_timeout_seconds=0.005), {spec.name: spec}
    )
    token = current_control.set(RunControl("alice", "teacher", thread_id="test-thread"))
    try:
        with pytest.raises(ToolTimeoutError) as failure:
            await guard.call("teacher", "test.write", student_id="alice")
        assert failure.value.detail["operation_status"] == "completed"
    finally:
        current_control.reset(token)
    token = current_control.set(RunControl("alice", "teacher", thread_id="test-thread"))
    try:
        assert await guard.call("teacher", "test.write", student_id="alice") == {"saved": True}
    finally:
        current_control.reset(token)
    assert len(ctx.memory_repo.list_facts("alice")) == 1
    with ctx.student_repo.database.session() as session:
        assert session.scalar(select(func.count()).select_from(HarnessEffect)) == 1


async def test_actual_graph_budget_stops_before_lesson_write(ctx):
    ctx.settings.harness.max_tool_calls = 3
    with pytest.raises(ToolBudgetError):
        await ctx.runtime.execute(ctx, "tutoring", state(), session_id="budget", run_id="one")
    assert not ctx.lesson_repo.list_lessons("alice")
    status = ctx.runtime.status("alice", "budget", "one", "tutoring")
    assert status["tool_calls"] == 3
    assert status["status"] == "failed"
    # Resume cannot refund already consumed calls.
    with pytest.raises(ToolBudgetError):
        await ctx.runtime.resume(ctx, "alice", "budget", "one", "tutoring")
    assert ctx.runtime.status("alice", "budget", "one", "tutoring")["tool_calls"] == 3


async def test_atomic_evaluation_rollback(ctx, monkeypatch):
    quiz = seed_quiz(ctx)

    def failed_progress(*args, **kwargs):
        raise RuntimeError("injected failure after attempt and mastery updates")

    monkeypatch.setattr(ctx.progress_repo, "add_snapshot", failed_progress)
    with pytest.raises(ToolExecutionError):
        await ctx.runtime.execute(
            ctx,
            "evaluation",
            {
                "student_id": "alice",
                "quiz_id": quiz.id,
                "answers": [],
                "messages": [],
            },
        )
    assert ctx.quiz_repo.list_attempts("alice") == []
    assert ctx.student_repo.get_or_404("alice").quiz_scores == []
    assert ctx.student_repo.list_concept_scores("alice") == []
    with ctx.student_repo.database.session() as session:
        assert session.scalar(select(func.count()).select_from(WrongAnswer)) == 0


async def test_duplicate_submission_across_sessions_and_answer_order(ctx):
    quiz = seed_quiz(ctx)
    answers = [
        {"question_id": quiz.questions[0].id, "answer": "base case"},
        {"question_id": quiz.questions[1].id, "answer": "wrong"},
    ]
    first = await ctx.runtime.execute(
        ctx,
        "evaluation",
        {"student_id": "alice", "quiz_id": quiz.id, "answers": answers, "messages": []},
        session_id="one",
    )
    second = await ctx.runtime.execute(
        ctx,
        "evaluation",
        {"student_id": "alice", "quiz_id": quiz.id, "answers": answers[::-1], "messages": []},
        session_id="two",
    )
    assert first["evaluation"] == second["evaluation"]
    assert second["replayed"] is True
    assert len(ctx.quiz_repo.list_attempts("alice")) == 1
    assert ctx.student_repo.list_concept_scores("alice")[0].attempts == 1
    assert len(ctx.student_repo.get_or_404("alice").quiz_scores) == 1
    assert ctx.progress_repo.count("alice") == 1


async def test_cross_student_quiz_submission_is_denied(ctx):
    quiz = seed_quiz(ctx)
    with pytest.raises(ToolPermissionError):
        await ctx.runtime.execute(
            ctx,
            "evaluation",
            {"student_id": "bob", "quiz_id": quiz.id, "answers": [], "messages": []},
        )
    assert not ctx.quiz_repo.list_attempts("bob")
    assert not ctx.student_repo.list_concept_scores("bob")


async def test_run_id_conflict_does_not_write_another_lesson(ctx):
    await ctx.runtime.execute(ctx, "tutoring", state(), session_id="session", run_id="same")
    with pytest.raises(RunConflictError):
        await ctx.runtime.execute(
            ctx, "tutoring", state(topic="DFS"), session_id="session", run_id="same"
        )
    assert len(ctx.lesson_repo.list_lessons("alice")) == 1


async def test_unknown_or_duplicate_answer_ids_fail_without_attempt(ctx):
    quiz = seed_quiz(ctx)
    for index, answers in enumerate(
        (
            [{"question_id": "missing", "answer": "x"}],
            [{"question_id": quiz.questions[0].id, "answer": "x"}] * 2,
        )
    ):
        with pytest.raises(ToolParameterError):
            await ctx.runtime.execute(
                ctx,
                "evaluation",
                {"student_id": "alice", "quiz_id": quiz.id, "answers": answers},
                run_id=str(index),
            )
    assert not ctx.quiz_repo.list_attempts("alice")


@pytest.mark.parametrize("topic", list(CURRICULUM))
async def test_six_programming_topics_complete_workflow(ctx, topic):
    final = await ctx.runtime.execute(ctx, "tutoring", {**state(topic=topic), "with_quiz": True})
    assert final["lesson"]["topic"] == topic
    assert final["lesson"]["examples"] == [
        CURRICULUM[topic]["example"].replace("# a becomes Fibonacci(n)", "# a是Fibonacci(n)")
    ]
    quiz = Quiz.model_validate(final["quiz"])
    assert len(quiz.questions) == 8
    assert quiz.questions[0].correct_answer == CONTENT_ZH[topic]["options"][0]
    assert quiz.questions[0].correct_answer in quiz.questions[0].options
    assert quiz.questions[5].correct_answer == CURRICULUM[topic]["answer"]
    assert final["runtime"]["status"] == "completed"


async def test_ema_and_prerequisite_policy_are_explainable(ctx):
    ctx.student_repo.get_or_create("alice")
    first = ctx.student_repo.upsert_concept_score("alice", "Recursion", 0.25)
    second = ctx.student_repo.upsert_concept_score("alice", "Recursion", 1.0)
    assert first.proficiency == 0.25
    assert second.proficiency == 0.7  # original EMA: .25*.4+1*.6
    plan = recommend_next({"Python Basics": 0.9, "Recursion": 0.4, "Binary Trees": 0.2})
    assert plan["target_topic"] == "Binary Trees"
    assert plan["next_topic"] == "Recursion"
    assert "prerequisite" in plan["rationale"]
    final = await ctx.runtime.execute(ctx, "tutoring", {**state(), "with_quiz": True})
    assert final["quiz"]["difficulty"] == "intermediate"
    assert (
        "尚未完成测验" in final["recommendations"]["rationale"]
        or "掌握度" in final["recommendations"]["rationale"]
    )


async def test_llm_recommendation_cannot_override_policy(ctx, monkeypatch):
    from models.schemas import Recommendation

    original = ctx.llm.generate_json

    async def arbitrary(system, user, **kwargs):
        if kwargs["model_class"] is Recommendation:
            return Recommendation(next_topic="Quantum Physics", rationale="arbitrary")
        return await original(system, user, **kwargs)

    monkeypatch.setattr(ctx.llm, "generate_json", arbitrary)
    final = await ctx.runtime.execute(ctx, "tutoring", state())
    assert final["recommendations"]["next_topic"] == "Python Basics"
    assert final["recommendations"]["rationale"] != "arbitrary"


async def test_question_and_general_branches_enter_guard(ctx, monkeypatch):
    from agents.planner import PlannerOutput

    question = await ctx.runtime.execute(
        ctx, "tutoring", {**state(), "user_input": "what is Recursion?"}
    )
    assert any(e["tool"] == "rag.context" for e in question["harness_metrics"]["events"])
    original = ctx.llm.generate_json

    async def general_intent(system, user, **kwargs):
        if kwargs["model_class"] is PlannerOutput:
            return PlannerOutput(intent="general", topic="general", learning_goal="greet")
        return await original(system, user, **kwargs)

    monkeypatch.setattr(ctx.llm, "generate_json", general_intent)
    general = await ctx.runtime.execute(ctx, "tutoring", {**state(), "user_input": "hello"})
    assert any(e["tool"] == "llm.text" for e in general["harness_metrics"]["events"])
    assert general.get("lesson") is None
    assert general["runtime"]["status"] == "completed"
