"""Request-local control, shared by graph wrappers and actual tool calls."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunControl:
    student_id: str
    node: str
    thread_id: str = ""
    tool_calls: int = 0
    write_index: int = 0
    repeats: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def metrics(self) -> dict[str, Any]:
        return {"tool_calls": self.tool_calls, "events": self.events[-50:]}


current_control: ContextVar[RunControl | None] = ContextVar("harness_control", default=None)
current_projection: ContextVar[dict[str, Any] | None] = ContextVar(
    "harness_projection", default=None
)
