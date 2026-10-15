"""Single-process local runtime: graph checkpoints and independent business memory.

One session groups runs; each (student, session, run, workflow) owns a checkpoint
thread. The original graphs are acyclic, so guarded write slots are stable on
node replay. This does not claim distributed exactly-once execution.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from database.models import HarnessRun
from harness.errors import GraphLimitError, RunConflictError, ToolParameterError
from utils.errors import EduAgentError, ResourceNotFoundError
from utils.helpers import utc_now_iso


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class SessionRuntime:
    def __init__(self, database, settings):
        self.db = database
        self.settings = settings
        business_path = database._engine.url.database
        self.checkpoint_path = settings.checkpoint_path or (
            str(Path(business_path).with_suffix(".checkpoints.sqlite"))
            if business_path and business_path != ":memory:"
            else ":memory:"
        )
        if self.checkpoint_path != ":memory:":
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        self._locks = WeakValueDictionary()

    @staticmethod
    def thread_id(student_id, session_id, run_id, workflow):
        return digest([student_id, session_id, run_id, workflow])

    def _lock(self, thread_id):
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock

    def status(self, student_id, session_id, run_id, workflow):
        thread_id = self.thread_id(student_id, session_id, run_id, workflow)
        with self.db.session() as session:
            row = session.get(HarnessRun, thread_id)
            if row is None or row.student_id != student_id:
                raise ResourceNotFoundError("Run not found for this student/session")
            return {
                "student_id": row.student_id,
                "session_id": row.session_id,
                "run_id": row.run_id,
                "thread_id": row.thread_id,
                "workflow": row.workflow,
                "status": row.status,
                "tool_calls": row.tool_calls,
                "persistent": self.checkpoint_path != ":memory:",
                "error": row.error,
            }

    def _update(self, thread_id, **values):
        with self.db.session() as session:
            row = session.get(HarnessRun, thread_id)
            for name, value in values.items():
                setattr(row, name, value)
            row.updated_at = utc_now_iso()

    def _builder(self, workflow):
        from graphs.workflow import build_evaluation_graph, build_tutoring_graph

        if workflow == "tutoring":
            return build_tutoring_graph
        if workflow == "evaluation":
            return build_evaluation_graph
        raise ToolParameterError("Unknown workflow")

    async def execute(
        self, ctx, workflow, state, *, session_id=None, run_id=None, pause_before=None
    ):
        builder = self._builder(workflow)
        allowed_pause = (
            {"teacher", "quiz", "memory", "recommendation"}
            if workflow == "tutoring"
            else {"evaluation", "memory", "recommendation"}
        )
        if pause_before and pause_before not in allowed_pause:
            raise ToolParameterError("Breakpoint is not a node in this workflow")
        student_id = state["student_id"]
        if workflow == "evaluation":
            state = {
                **state,
                "answers": sorted(
                    state.get("answers") or [], key=lambda x: str(x.get("question_id"))
                ),
            }
        ctx.student_repo.get_or_create(student_id)
        session_id = session_id or "default"
        # Identical quiz submissions without an explicit run id deduplicate even
        # across sessions. Supplying a new run id deliberately records a retake.
        if run_id is None and workflow == "evaluation":
            canonical_answers = state.get("answers") or []
            run_id = digest([student_id, state.get("quiz_id"), canonical_answers])
            session_id = "submissions"
        run_id = run_id or uuid4().hex
        thread_id = self.thread_id(student_id, session_id, run_id, workflow)
        input_hash = digest(state)
        async with self._lock(thread_id):
            with self.db.session() as session:
                row = session.get(HarnessRun, thread_id)
                if row:
                    if row.input_hash != input_hash:
                        raise RunConflictError("This run id already belongs to a different request")
                    if row.status == "completed":
                        return {
                            **row.result,
                            "runtime": self.status(student_id, session_id, run_id, workflow),
                            "replayed": True,
                        }
                    raise RunConflictError(
                        "Run already exists; use the resume endpoint",
                        detail=self.status(student_id, session_id, run_id, workflow),
                    )
                session.add(
                    HarnessRun(
                        thread_id=thread_id,
                        student_id=student_id,
                        session_id=session_id,
                        run_id=run_id,
                        workflow=workflow,
                        input_hash=input_hash,
                        status="running",
                        tool_calls=0,
                        result={},
                        error={},
                        created_at=utc_now_iso(),
                        updated_at=utc_now_iso(),
                    )
                )
            state = {
                **state,
                "harness_thread_id": thread_id,
                "session_id": session_id,
                "run_id": run_id,
            }
            return await self._invoke(
                ctx, builder, workflow, student_id, session_id, run_id, state, pause_before
            )

    async def resume(self, ctx, student_id, session_id, run_id, workflow):
        self.status(student_id, session_id, run_id, workflow)  # owned run, before opening saver
        thread_id = self.thread_id(student_id, session_id, run_id, workflow)
        async with self._lock(thread_id):
            with self.db.session() as session:
                row = session.get(HarnessRun, thread_id)
                if row.status == "completed":
                    return {
                        **row.result,
                        "runtime": self.status(student_id, session_id, run_id, workflow),
                        "replayed": True,
                    }
            return await self._invoke(
                ctx, self._builder(workflow), workflow, student_id, session_id, run_id, None, None
            )

    async def _invoke(
        self, ctx, builder, workflow, student_id, session_id, run_id, initial, pause_before
    ):
        thread_id = self.thread_id(student_id, session_id, run_id, workflow)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.recursion_limit,
        }
        try:
            async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as saver:
                graph = builder(
                    ctx,
                    checkpointer=saver,
                    interrupt_before=[pause_before] if pause_before else None,
                )
                if initial is None:
                    checkpoint = await graph.aget_state(config)
                    if not checkpoint.values:
                        raise RunConflictError("No saved graph state is available to resume")
                    if not checkpoint.next:
                        final = checkpoint.values
                    else:
                        self._update(thread_id, status="running", error={})
                        final = await graph.ainvoke(None, config)
                else:
                    final = await graph.ainvoke(initial, config)
                checkpoint = await graph.aget_state(config)
                status = "interrupted" if checkpoint.next else "completed"
                self._update(thread_id, status=status, result=final, error={})
                return {
                    **final,
                    "runtime": self.status(student_id, session_id, run_id, workflow),
                    "pending_nodes": list(checkpoint.next),
                    "replayed": False,
                }
        except GraphRecursionError as exc:
            error = GraphLimitError(
                "LangGraph recursion limit reached; saved progress can be inspected"
            )
            self._update(thread_id, status="failed", error={"code": error.code})
            error.detail = self.status(student_id, session_id, run_id, workflow)
            raise error from exc
        except EduAgentError as exc:
            self._update(
                thread_id, status="failed", error={"code": exc.code, "message": exc.message}
            )
            exc.detail = {
                **exc.detail,
                "runtime": self.status(student_id, session_id, run_id, workflow),
            }
            raise
        except Exception as exc:
            self._update(thread_id, status="failed", error={"code": type(exc).__name__})
            raise
