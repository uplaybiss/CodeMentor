"""Typed execution failures consumed by the existing API error mapper."""

from utils.errors import EduAgentError


class ToolParameterError(EduAgentError):
    code = "TOOL_PARAMETER_ERROR"
    status_code = 422


class ToolPermissionError(EduAgentError):
    code = "TOOL_FORBIDDEN"
    status_code = 403


class ToolTimeoutError(EduAgentError):
    code = "TOOL_TIMEOUT"
    status_code = 504


class ToolExecutionError(EduAgentError):
    code = "TOOL_EXECUTION_FAILED"
    status_code = 502


class ToolBudgetError(EduAgentError):
    code = "TOOL_BUDGET_EXHAUSTED"
    status_code = 429


class ToolRepeatError(ToolBudgetError):
    code = "TOOL_REPEAT_LIMIT"


class ContextBudgetError(ToolParameterError):
    code = "CONTEXT_BUDGET_EXHAUSTED"


class RunConflictError(EduAgentError):
    code = "RUN_CONFLICT"
    status_code = 409


class GraphLimitError(ToolBudgetError):
    code = "GRAPH_RECURSION_LIMIT"
