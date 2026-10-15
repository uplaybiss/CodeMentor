"""Reproducible mock acceptance, with two distinct processes for checkpoint proof.

Run: python -m demo.acceptance --out-dir evidence/manual-run
Creates a fresh subdirectory; never erases an existing database or artifact.
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import platform
import sqlite3
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from demo.local import offline_environment


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def process_phase(args):
    offline_environment(args.data_dir.resolve())
    from services.container import Container

    container = Container()
    ctx = container.context()
    try:
        if args.phase == "interrupt":
            final = await ctx.runtime.execute(
                ctx,
                "tutoring",
                {
                    "student_id": "restart_student",
                    "topic": "Binary Trees",
                    "user_input": "teach me Binary Trees",
                    "with_quiz": True,
                    "messages": [],
                    "memory_updates": [],
                    "errors": [],
                },
                session_id="restart_session",
                run_id="restart_run",
                pause_before="teacher",
            )
            assert final["pending_nodes"] == ["teacher"]
            assert len(ctx.lesson_repo.list_lessons("restart_student")) == 0
            value = {
                "pid": os.getpid(),
                "runtime": final["runtime"],
                "pending_nodes": final["pending_nodes"],
                "lesson_count": 0,
                "checkpoint_path": ctx.runtime.checkpoint_path,
            }
        else:
            before = ctx.runtime.status(
                "restart_student", "restart_session", "restart_run", "tutoring"
            )
            final = await ctx.runtime.resume(
                ctx, "restart_student", "restart_session", "restart_run", "tutoring"
            )
            assert final["lesson"]["topic"] == "Binary Trees"
            repeat = await ctx.runtime.resume(
                ctx, "restart_student", "restart_session", "restart_run", "tutoring"
            )
            lesson_count = len(ctx.lesson_repo.list_lessons("restart_student"))
            assert lesson_count == 1 and repeat["replayed"] is True
            value = {
                "pid": os.getpid(),
                "before": before,
                "runtime": final["runtime"],
                "lesson_count": lesson_count,
                "quiz_count": len(ctx.quiz_repo.list_quizzes("restart_student")),
                "repeat_replayed": repeat["replayed"],
                "lesson_topic": final["lesson"]["topic"],
            }
        save(args.phase_output, value)
        print(json.dumps(value, ensure_ascii=False))
    finally:
        container.dispose()


async def sample(out):
    offline_environment((out / "sample_db").resolve())
    from sqlalchemy import select

    from config.curriculum import CURRICULUM
    from config.settings import HarnessSettings
    from database.models import Conversation, MemoryEntry, WrongAnswer
    from graphs.workflow import build_evaluation_graph, build_tutoring_graph
    from harness.context_builder import ContextBuilder, count_tokens, render_context
    from harness.control import RunControl, current_control
    from harness.tool_guard import ToolGuard
    from harness.tools import StudentInput, ToolSpec
    from services.container import Container
    from utils.errors import EduAgentError

    container = Container()
    ctx = container.context()
    ctx.settings.harness.context_tokens = 256
    logging.basicConfig(
        level=logging.INFO, filename=out / "harness-execution.log", encoding="utf-8", force=True
    )
    try:
        for student in ("alice", "bob"):
            ctx.student_repo.get_or_create(student, name=student.upper())
            ctx.student_repo.upsert_concept_score(
                student, "Python Basics", 0.9 if student == "alice" else 0.1
            )
            ctx.student_repo.upsert_concept_score(
                student, "Recursion", 0.25 if student == "alice" else 0.85
            )
        initial = {
            "student_id": "alice",
            "topic": "Recursion",
            "user_input": "teach me Recursion",
            "learning_goal": "explain a reachable base case",
            "messages": [],
            "memory_updates": [],
            "errors": [],
        }
        quiz_final = await ctx.runtime.execute(
            ctx, "tutoring", {**initial, "with_quiz": True}, run_id="sample_quiz"
        )
        quiz_id = quiz_final["quiz"]["id"]
        from models.schemas import Quiz

        bob_quiz = Quiz.model_validate(quiz_final["quiz"])
        bob_quiz.id = "fixture_bob_quiz"
        ctx.quiz_repo.create_quiz(bob_quiz, "bob")
        # Fixed, synthetic history. All-history baseline is explicitly a
        # measurement counterfactual, not a claim about upstream's old prompts.
        with ctx.student_repo.database.session() as session:
            for student in ("alice", "bob"):
                for index in range(120):
                    topic = "Recursion" if index % 2 == 0 else "Binary Trees"
                    marker = "ALICE_RECURSION" if student == "alice" else "BOB_SECRET"
                    body = (
                        f"{marker} {topic} history {index}: "
                        + "review the terminating case and step toward it. " * 35
                    )
                    session.add(
                        MemoryEntry(
                            student_id=student,
                            fact_type="concept",
                            content=body,
                            importance=0.6,
                            created_at="2026-10-13T00:00:00Z",
                        )
                    )
                    session.add(
                        Conversation(
                            student_id=student,
                            role="user",
                            content=body,
                            metadata_json={"topic": topic},
                            created_at="2026-10-13T00:00:00Z",
                        )
                    )
                for index in range(30):
                    topic = "Recursion" if index % 2 == 0 else "Binary Trees"
                    session.add(
                        WrongAnswer(
                            student_id=student,
                            topic=topic,
                            quiz_id=quiz_id if student == "alice" else bob_quiz.id,
                            question=f"{student} {topic} mistake {index}: What ensures termination?",
                            submitted="always recurse",
                            explanation="Use a reachable base case and a decreasing argument.",
                            created_at="2026-10-13T00:00:00Z",
                        )
                    )
        projection = ContextBuilder(ctx.student_repo.database, ctx.settings.harness).build(
            initial, "teacher"
        )
        with ctx.student_repo.database.session() as session:
            baseline = copy.deepcopy(projection["payload"])
            baseline["memory"] = [
                dict(r._mapping)
                for r in session.execute(
                    select(MemoryEntry.fact_type, MemoryEntry.content).where(
                        MemoryEntry.student_id == "alice"
                    )
                ).all()
            ]
            baseline["recent"] = [
                dict(r._mapping)
                for r in session.execute(
                    select(Conversation.role, Conversation.content).where(
                        Conversation.student_id == "alice"
                    )
                ).all()
            ]
            baseline["mistakes"] = [
                dict(r._mapping)
                for r in session.execute(
                    select(
                        WrongAnswer.question,
                        WrongAnswer.submitted,
                        WrongAnswer.explanation,
                        WrongAnswer.created_at,
                    ).where(WrongAnswer.student_id == "alice")
                ).all()
            ]
        baseline_text = render_context(baseline)
        (out / "context-all-history.json").write_text(baseline_text, encoding="utf-8")
        (out / "context-candidate.json").write_text(projection["candidate_text"], encoding="utf-8")
        (out / "context-projected.json").write_text(projection["text"], encoding="utf-8")
        context_result = {
            **projection["metrics"],
            "all_history_tokens": count_tokens(baseline_text),
            "all_history_records": {k: len(baseline[k]) for k in ("mistakes", "memory", "recent")},
            "candidate_to_projected_reduction_percent": round(
                (
                    1
                    - projection["metrics"]["context_tokens"]
                    / projection["metrics"]["candidate_tokens"]
                )
                * 100,
                2,
            ),
            "all_history_to_projected_reduction_percent": round(
                (1 - projection["metrics"]["context_tokens"] / count_tokens(baseline_text)) * 100, 2
            ),
            "necessary_fields_retained": all(
                projection["payload"].get(k) for k in ("topic", "mastery", "goal", "state")
            ),
            "student_isolation": "BOB_SECRET" not in projection["text"],
            "irrelevant_topic_removed": "Binary Trees" not in projection["text"],
            "mistakes_kept": len(projection["payload"]["mistakes"]),
            "comparison_note": "Same fixed fixture; all-history baseline is hypothetical, not upstream behavior. Candidate is actual SQL-filtered context before priority trimming. cl100k_base is exact tokenizer count, not provider billing.",
        }
        assert (
            context_result["necessary_fields_retained"]
            and context_result["student_isolation"]
            and context_result["irrelevant_topic_removed"]
        )
        assert context_result["context_tokens"] <= 256
        save(out / "context-measurement.json", context_result)

        captured = []
        original_json = ctx.llm.generate_json

        async def capture(system, user, **kwargs):
            captured.append({"schema": kwargs["model_class"].__name__, "user": user})
            return await original_json(system, user, **kwargs)

        ctx.llm.generate_json = capture
        started = time.perf_counter()
        learned = await ctx.runtime.execute(
            ctx, "tutoring", initial, session_id="later_session", run_id="history_read"
        )
        elapsed = (time.perf_counter() - started) * 1000
        ctx.llm.generate_json = original_json
        teacher_prompt = next(p["user"] for p in captured if p["schema"] == "Explanation")
        assert "Task learning context:" in teacher_prompt and "BOB_SECRET" not in teacher_prompt
        save(out / "actual-prompts.json", captured)
        save(
            out / "learning-workflow.json",
            {
                "result": learned,
                "mock_elapsed_ms": round(elapsed, 3),
                "prompt_projection_verified": True,
            },
        )

        quiz = ctx.quiz_repo.get_quiz(quiz_id)
        answers = [
            {"question_id": q.id, "answer": q.correct_answer if index == 0 else ""}
            for index, q in enumerate(quiz.questions)
        ]
        evaluation_state = {
            "student_id": "alice",
            "quiz_id": quiz_id,
            "answers": answers,
            "messages": [],
        }
        graded = await ctx.runtime.execute(ctx, "evaluation", evaluation_state)
        duplicate = await ctx.runtime.execute(
            ctx,
            "evaluation",
            {**evaluation_state, "answers": answers[::-1]},
            session_id="different",
        )
        alice_scores = [s.model_dump() for s in ctx.student_repo.list_concept_scores("alice")]
        bob_scores = [s.model_dump() for s in ctx.student_repo.list_concept_scores("bob")]
        assert len(ctx.quiz_repo.list_attempts("alice")) == 1 and duplicate["replayed"]
        assert next(s["proficiency"] for s in alice_scores if s["concept"] == "Recursion") == 0.175
        assert next(s["proficiency"] for s in bob_scores if s["concept"] == "Recursion") == 0.85
        save(
            out / "evaluation-memory.json",
            {
                "evaluation": graded["evaluation"],
                "runtime": graded["runtime"],
                "harness_metrics": graded["harness_metrics"],
                "duplicate_replayed": duplicate["replayed"],
                "attempt_count": 1,
                "progress_count": ctx.progress_repo.count("alice"),
                "alice_scores": alice_scores,
                "bob_scores": bob_scores,
                "ema_calculation": "0.25 * 0.4 + (1/8) * 0.6 = 0.175",
                "recommendation": graded["recommendations"],
            },
        )

        fault_results = []
        expected_codes = {
            "task_whitelist": "TOOL_FORBIDDEN",
            "parameter_validation": "TOOL_PARAMETER_ERROR",
            "repeated_call": "TOOL_REPEAT_LIMIT",
            "execution_failure": "TOOL_EXECUTION_FAILED",
            "read_timeout": "TOOL_TIMEOUT",
            "real_graph_budget": "TOOL_BUDGET_EXHAUSTED",
            "real_graph_recursion": "GRAPH_RECURSION_LIMIT",
            "context_required_overflow": "CONTEXT_BUDGET_EXHAUSTED",
            "run_conflict": "RUN_CONFLICT",
            "cross_student_quiz": "TOOL_FORBIDDEN",
        }

        async def expect_error(name, function):
            try:
                await function()
            except EduAgentError as exc:
                assert exc.code == expected_codes[name], (name, exc.code)
                fault_results.append(
                    {
                        "scenario": name,
                        "code": exc.code,
                        "message": exc.message,
                        "detail": exc.detail,
                    }
                )
            else:
                raise AssertionError(f"Expected guard failure: {name}")

        guard = ctx.harness.guard
        control_token = current_control.set(RunControl("alice", "teacher"))
        try:
            await expect_error(
                "task_whitelist", lambda: guard.call("teacher", "memory.save", student_id="alice")
            )
            await expect_error(
                "parameter_validation",
                lambda: guard.call("teacher", "student.profile", student_id="alice", invalid=True),
            )
            await guard.call("teacher", "student.profile", student_id="alice")
            await guard.call("teacher", "student.profile", student_id="alice")
            await expect_error(
                "repeated_call",
                lambda: guard.call("teacher", "student.profile", student_id="alice"),
            )
        finally:
            current_control.reset(control_token)

        def failing_read(student_id):
            raise RuntimeError("injected tool failure")

        def slow_read(student_id):
            time.sleep(0.03)
            return "late"

        custom = ToolGuard(
            ctx.student_repo.database,
            HarnessSettings(tool_timeout_seconds=1),
            {
                "fail": ToolSpec("fail", failing_read, StudentInput, frozenset({"teacher"}), True),
                "slow": ToolSpec("slow", slow_read, StudentInput, frozenset({"teacher"}), True),
            },
        )
        await expect_error(
            "execution_failure", lambda: custom.call("teacher", "fail", student_id="alice")
        )
        custom.settings.tool_timeout_seconds = 0.005
        await expect_error(
            "read_timeout", lambda: custom.call("teacher", "slow", student_id="alice")
        )
        saved_budget = ctx.settings.harness.max_tool_calls
        ctx.settings.harness.max_tool_calls = 3
        await expect_error(
            "real_graph_budget",
            lambda: ctx.runtime.execute(
                ctx, "tutoring", initial, session_id="faults", run_id="budget"
            ),
        )
        ctx.settings.harness.max_tool_calls = saved_budget
        ctx.settings.harness.recursion_limit = 1
        await expect_error(
            "real_graph_recursion",
            lambda: ctx.runtime.execute(
                ctx, "tutoring", initial, session_id="faults", run_id="recursion"
            ),
        )
        ctx.settings.harness.recursion_limit = 20
        await expect_error(
            "context_required_overflow",
            lambda: ctx.runtime.execute(
                ctx,
                "tutoring",
                {**initial, "learning_goal": "goal " * 1000},
                session_id="faults",
                run_id="context",
            ),
        )
        await expect_error(
            "run_conflict",
            lambda: ctx.runtime.execute(
                ctx,
                "tutoring",
                {**initial, "topic": "DFS"},
                session_id="later_session",
                run_id="history_read",
            ),
        )
        await expect_error(
            "cross_student_quiz",
            lambda: ctx.runtime.execute(
                ctx, "evaluation", {**evaluation_state, "student_id": "bob"}
            ),
        )
        save(out / "guard-scenarios.json", fault_results)
        topics = []
        for title in CURRICULUM:
            final = await ctx.runtime.execute(
                ctx,
                "tutoring",
                {
                    **initial,
                    "student_id": "curriculum_student",
                    "topic": title,
                    "user_input": f"teach me {title}",
                    "with_quiz": True,
                },
            )
            assert len(final["quiz"]["questions"]) == 8
            topics.append(
                {
                    "topic": title,
                    "question_count": 8,
                    "quiz_id": final["quiz"]["id"],
                    "difficulty": final["quiz"]["difficulty"],
                    "status": final["runtime"]["status"],
                    "tool_calls": final["runtime"]["tool_calls"],
                }
            )
        save(out / "curriculum.json", topics)
        question = await ctx.runtime.execute(
            ctx, "tutoring", {**initial, "user_input": "what is Recursion?"}
        )
        from agents.planner import PlannerOutput

        async def general_intent(system, user, **kwargs):
            if kwargs["model_class"] is PlannerOutput:
                return PlannerOutput(intent="general", topic="general", learning_goal="greet")
            return await original_json(system, user, **kwargs)

        ctx.llm.generate_json = general_intent
        general = await ctx.runtime.execute(ctx, "tutoring", {**initial, "user_input": "hello"})
        ctx.llm.generate_json = original_json
        save(
            out / "route-coverage.json",
            {
                "question": {
                    "runtime": question["runtime"],
                    "events": question["harness_metrics"]["events"],
                },
                "general": {
                    "runtime": general["runtime"],
                    "events": general["harness_metrics"]["events"],
                },
                "note": "GENERAL intent supplied by deterministic mock schema override; RAG is disabled in local profile but its adapter is dispatched.",
            },
        )
        graph_counts = {
            "tutoring_nodes": sorted(
                n for n in build_tutoring_graph(ctx).get_graph().nodes if not n.startswith("__")
            ),
            "evaluation_nodes": sorted(
                n for n in build_evaluation_graph(ctx).get_graph().nodes if not n.startswith("__")
            ),
            "registered_operations": sorted(ctx.harness.guard.tools),
            "python_tools": 9,
            "llm_operations": 2,
            "observed_tools_in_learning_and_evaluation": sorted(
                {
                    e["tool"]
                    for e in learned["harness_metrics"]["events"]
                    + graded["harness_metrics"]["events"]
                    + quiz_final["harness_metrics"]["events"]
                    + question["harness_metrics"]["events"]
                    + general["harness_metrics"]["events"]
                }
            ),
        }
        assert len(graph_counts["observed_tools_in_learning_and_evaluation"]) == 11
        save(out / "graph-tools.json", graph_counts)
        summary = {
            "python": platform.python_version(),
            "os": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "packages": {
                package: version(package)
                for package in (
                    "fastapi",
                    "langgraph",
                    "langgraph-checkpoint-sqlite",
                    "aiosqlite",
                    "sqlalchemy",
                    "tiktoken",
                )
            },
            "provider": "demo",
            "paid_model_calls": 0,
            "rag_enabled": False,
            "context": context_result,
            "guard_scenarios": len(fault_results),
            "guard_codes": sorted({r["code"] for r in fault_results}),
            "curriculum_topics": len(topics),
            "distinct_question_templates": 18,
            "generated_question_instances": 48,
            "max_tool_calls": ctx.settings.harness.max_tool_calls,
            "mock_learning_elapsed_ms": round(elapsed, 3),
            "business_db": str(out / "sample_db" / "codementor.sqlite"),
        }
        save(out / "summary.json", summary)
        return summary
    finally:
        container.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("evidence/manual"))
    parser.add_argument("--phase", choices=("interrupt", "resume"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--phase-output", type=Path)
    args = parser.parse_args()
    if args.phase:
        asyncio.run(process_phase(args))
        return
    out = args.out_dir.resolve() / ("run-" + uuid4().hex[:8])
    out.mkdir(parents=True)
    summary = asyncio.run(sample(out))
    phases = []
    for phase in ("interrupt", "resume"):
        command = [
            sys.executable,
            "-m",
            "demo.acceptance",
            "--phase",
            phase,
            "--data-dir",
            str(out / "restart_db"),
            "--phase-output",
            str(out / f"restart-{phase}.json"),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", check=False
        )
        (out / f"restart-{phase}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Separate-process {phase} failed; inspect saved log")
        phases.append(json.loads((out / f"restart-{phase}.json").read_text(encoding="utf-8")))
    assert phases[0]["pid"] != phases[1]["pid"]
    assert phases[1]["before"]["status"] == "interrupted"
    summary["process_restart"] = {
        "distinct_pids": [p["pid"] for p in phases],
        "status": phases[1]["runtime"]["status"],
        "lesson_count": phases[1]["lesson_count"],
        "repeat_replayed": phases[1]["repeat_replayed"],
    }
    save(out / "summary.json", summary)
    print(json.dumps({"evidence_dir": str(out), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
