"""Persistent graph state, run ownership and recoverable side effects."""

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from harness.errors import GraphLimitError, ToolExecutionError
from services.container import Container
from tests.test_harness import state
from utils.errors import ResourceNotFoundError


async def test_fresh_container_resumes_saved_checkpoint(settings, ctx):
    interrupted = await ctx.runtime.execute(
        ctx, "tutoring", state(), session_id="session", run_id="restart", pause_before="teacher"
    )
    assert interrupted["runtime"]["status"] == "interrupted"
    assert interrupted["pending_nodes"] == ["teacher"]
    assert not ctx.lesson_repo.list_lessons("alice")
    replacement = Container(settings)
    try:
        restored_ctx = replacement.context()
        completed = await restored_ctx.runtime.resume(
            restored_ctx, "alice", "session", "restart", "tutoring"
        )
        assert completed["lesson"]["topic"] == "Recursion"
        assert completed["runtime"]["status"] == "completed"
        assert completed["runtime"]["tool_calls"] > interrupted["runtime"]["tool_calls"]
        again = await restored_ctx.runtime.resume(
            restored_ctx, "alice", "session", "restart", "tutoring"
        )
        assert again["replayed"] is True
        assert len(restored_ctx.lesson_repo.list_lessons("alice")) == 1
    finally:
        replacement.dispose()


async def test_student_threads_with_identical_session_ids_are_isolated(ctx):
    alice = await ctx.runtime.execute(
        ctx,
        "tutoring",
        state("alice", "Recursion"),
        session_id="same",
        run_id="same",
        pause_before="teacher",
    )
    bob = await ctx.runtime.execute(
        ctx,
        "tutoring",
        state("bob", "DFS"),
        session_id="same",
        run_id="same",
        pause_before="teacher",
    )
    assert alice["runtime"]["thread_id"] != bob["runtime"]["thread_id"]
    final = await ctx.runtime.resume(ctx, "bob", "same", "same", "tutoring")
    assert final["student_id"] == "bob" and final["lesson"]["topic"] == "DFS"
    assert not ctx.lesson_repo.list_lessons("alice")
    assert ctx.runtime.status("alice", "same", "same", "tutoring")["status"] == "interrupted"
    with pytest.raises(ResourceNotFoundError):
        await ctx.runtime.resume(ctx, "mallory", "same", "same", "tutoring")


async def test_failed_node_after_committed_write_replays_without_duplicate(ctx, monkeypatch):
    from agents.teacher import TeacherAgent

    original = TeacherAgent.run

    async def crash_after_write(self, node_state):
        await original(self, node_state)
        raise ToolExecutionError("Simulated failure between local write and node checkpoint")

    monkeypatch.setattr(TeacherAgent, "run", crash_after_write)
    with pytest.raises(ToolExecutionError):
        await ctx.runtime.execute(ctx, "tutoring", state(), session_id="crash", run_id="one")
    assert len(ctx.lesson_repo.list_lessons("alice")) == 1
    monkeypatch.setattr(TeacherAgent, "run", original)
    final = await ctx.runtime.resume(ctx, "alice", "crash", "one", "tutoring")
    assert final["runtime"]["status"] == "completed"
    assert len(ctx.lesson_repo.list_lessons("alice")) == 1
    assert any(
        e["tool"] == "lesson.save" and e["status"] == "replayed"
        for e in final["harness_metrics"]["events"]
    )


async def test_recursion_limit_and_resume_actual_graph(ctx):
    ctx.settings.harness.recursion_limit = 2
    with pytest.raises(GraphLimitError):
        await ctx.runtime.execute(ctx, "tutoring", state(), session_id="limit", run_id="one")
    ctx.settings.harness.recursion_limit = 20
    completed = await ctx.runtime.resume(ctx, "alice", "limit", "one", "tutoring")
    assert completed["runtime"]["status"] == "completed"
    assert len(ctx.lesson_repo.list_lessons("alice")) == 1


async def test_langgraph_self_loop_is_bounded():
    graph = StateGraph(dict)
    graph.add_node("loop", lambda data: data)
    graph.add_edge(START, "loop")
    graph.add_edge("loop", "loop")
    with pytest.raises(GraphRecursionError):
        await graph.compile().ainvoke({"student_id": "alice"}, {"recursion_limit": 3})


async def test_completed_run_idempotency_and_business_memory_across_sessions(ctx):
    ctx.student_repo.get_or_create("alice")
    ctx.student_repo.upsert_concept_score("alice", "Recursion", 0.25)
    first = await ctx.runtime.execute(ctx, "tutoring", state(), session_id="first", run_id="one")
    repeat = await ctx.runtime.execute(ctx, "tutoring", state(), session_id="first", run_id="one")
    second = await ctx.runtime.execute(ctx, "tutoring", state(), session_id="second", run_id="one")
    assert repeat["replayed"] is True
    assert first["runtime"]["thread_id"] != second["runtime"]["thread_id"]
    assert second["task_context"]["mastery"]["score"] == 0.25
    assert len(ctx.lesson_repo.list_lessons("alice")) == 2


def test_http_resume_status_and_conflict(client):
    request = {
        "student_id": "alice",
        "topic": "Recursion",
        "message": "teach recursion",
        "session_id": "http",
        "run_id": "one",
        "pause_before": "teacher",
    }
    paused = client.post("/api/v1/chat", json=request)
    assert paused.status_code == 200
    assert paused.json()["runtime"]["status"] == "interrupted"
    status = client.get("/api/v1/chat/runs/http/one", params={"student_id": "alice"})
    assert status.json()["status"] == "interrupted"
    denied = client.post(
        "/api/v1/chat/resume", json={"student_id": "bob", "session_id": "http", "run_id": "one"}
    )
    assert denied.status_code == 404
    resumed = client.post(
        "/api/v1/chat/resume", json={"student_id": "alice", "session_id": "http", "run_id": "one"}
    )
    assert resumed.status_code == 200 and resumed.json()["runtime"]["status"] == "completed"
    conflict = client.post("/api/v1/chat", json={**request, "message": "changed input"})
    assert conflict.status_code == 409 and conflict.json()["error"] == "RUN_CONFLICT"


def test_http_malformed_answer_is_rejected_before_grading(client, ctx):
    from tests.test_harness import seed_quiz

    quiz = seed_quiz(ctx)
    response = client.post(
        "/api/v1/chat/submit-quiz",
        json={
            "student_id": "alice",
            "quiz_id": quiz.id,
            "answers": [{"question_id": ["invalid"], "answer": {"invalid": True}}],
        },
    )
    assert response.status_code == 422
    assert not ctx.quiz_repo.list_attempts("alice")
