"""Node wrapper connects bounded context and guarded tools to existing LangGraph."""

from functools import wraps

from harness.context_builder import ContextBuilder
from harness.control import RunControl, current_control, current_projection
from harness.tool_guard import ToolGuard
from harness.tools import register_tools


class AgentHarness:
    def __init__(self, database, settings):
        self.db = database
        self.settings = settings
        self.context_builder = ContextBuilder(database, settings)
        self.guard = None

    def bind(self, ctx):
        if self.guard is None:
            self.guard = ToolGuard(self.db, self.settings, register_tools(ctx))

    def wrap_node(self, name, function):
        @wraps(function)
        async def guarded_node(state):
            if self.guard is None:
                raise RuntimeError("Bind AgentContext before compiling the graph")
            # Session bootstrap is intentionally outside the dispatched-tool budget.
            from database.repositories.student_repository import StudentRepository

            StudentRepository(self.db).get_or_create(state["student_id"])
            previous = state.get("harness_metrics") or {}
            control = RunControl(
                student_id=state["student_id"],
                node=name,
                thread_id=state.get("harness_thread_id", ""),
                tool_calls=previous.get("tool_calls", 0),
                events=list(previous.get("events", [])),
            )
            control_token = current_control.set(control)
            try:
                projection = self.context_builder.build(state, name)
                projection_token = current_projection.set(projection)
                try:
                    updates = await function({**state, "_task": name})
                finally:
                    current_projection.reset(projection_token)
                return {
                    **updates,
                    "task_context": projection["payload"],
                    "context_metrics": projection["metrics"],
                    "harness_metrics": control.metrics(),
                }
            finally:
                current_control.reset(control_token)

        return guarded_node

    async def call(self, ctx, task, name, **params):
        self.bind(ctx)
        return await self.guard.call(task, name, **params)
