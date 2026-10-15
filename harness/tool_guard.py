"""Guard the actual dispatched tools, including durable per-run call accounting."""

import asyncio
import hashlib
import json
import logging
import time

from pydantic import ValidationError
from sqlalchemy import update

from database.models import HarnessEffect, HarnessRun
from harness.control import RunControl, current_control
from harness.errors import (
    ToolBudgetError,
    ToolExecutionError,
    ToolParameterError,
    ToolPermissionError,
    ToolRepeatError,
    ToolTimeoutError,
)
from utils.errors import EduAgentError, LLMError, LLMTimeoutError
from utils.helpers import utc_now_iso

logger = logging.getLogger(__name__)


class ToolGuard:
    def __init__(self, database, settings, tools):
        self.db = database
        self.settings = settings
        self.tools = tools

    def _record(self, control, name, status, started, **extra):
        event = {
            "node": control.node,
            "tool": name,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            **extra,
        }
        control.events.append(event)
        logger.info(
            "harness.tool %s",
            json.dumps({"thread_id": control.thread_id, "student_id": control.student_id, **event}),
        )

    def _reserve(self, control):
        if control.tool_calls >= self.settings.max_tool_calls:
            raise ToolBudgetError("Single-run tool call budget exhausted")
        if control.thread_id:
            with self.db.session() as session:
                row = session.get(HarnessRun, control.thread_id)
                if row is not None:
                    result = session.execute(
                        update(HarnessRun)
                        .where(
                            HarnessRun.thread_id == control.thread_id,
                            HarnessRun.student_id == control.student_id,
                            HarnessRun.tool_calls < self.settings.max_tool_calls,
                        )
                        .values(tool_calls=HarnessRun.tool_calls + 1)
                    )
                    if result.rowcount != 1:
                        raise ToolBudgetError("Persistent tool budget exhausted")
                    session.refresh(row)
                    control.tool_calls = row.tool_calls
                    return
        control.tool_calls += 1

    def _write_once(self, spec, params, control, effect_id):
        with self.db.session() as session:
            if effect_id:
                cached = session.get(HarnessEffect, effect_id)
                if cached:
                    return cached.result["value"], True
            result = spec.function(**params)
            if effect_id:
                session.add(
                    HarnessEffect(
                        effect_id=effect_id,
                        student_id=control.student_id,
                        tool_name=spec.name,
                        result={"value": result},
                        created_at=utc_now_iso(),
                    )
                )
            return result, False

    async def call(self, task: str, name: str, **params):
        started = time.perf_counter()
        control = current_control.get() or RunControl(
            student_id=params.get("student_id", ""), node=task
        )
        spec = self.tools.get(name)
        if spec is None or task not in spec.tasks or task != control.node:
            self._record(control, name, "denied", started, code="TOOL_FORBIDDEN")
            raise ToolPermissionError(f"Tool {name!r} is not allowed for task {task!r}")
        try:
            parsed = spec.schema.model_validate(params)
        except ValidationError as exc:
            self._record(control, name, "invalid", started, code="TOOL_PARAMETER_ERROR")
            raise ToolParameterError(f"Invalid parameters for {name}") from exc
        # Keep Python objects (e.g. Explanation and model classes) intact.
        params = {key: getattr(parsed, key) for key in type(parsed).model_fields}
        if params["student_id"] != control.student_id:
            raise ToolPermissionError("Tool student_id differs from the active run")
        signature = hashlib.sha256(
            json.dumps({"tool": name, "params": params}, sort_keys=True, default=str).encode()
        ).hexdigest()
        repeats = control.repeats.get(signature, 0) + 1
        control.repeats[signature] = repeats
        if repeats > self.settings.max_repeat_calls:
            self._record(control, name, "limited", started, code="TOOL_REPEAT_LIMIT")
            raise ToolRepeatError("Repeated identical tool invocation limit exceeded")

        effect_id = ""
        if not spec.readonly and not spec.model_call:
            control.write_index += 1
            if control.thread_id:
                effect_id = hashlib.sha256(
                    f"{control.thread_id}:{task}:{control.write_index}:{name}".encode()
                ).hexdigest()
                with self.db.session() as session:
                    cached = session.get(HarnessEffect, effect_id)
                    if cached:
                        self._record(control, name, "replayed", started)
                        return cached.result["value"]
        attempts = 1 + (self.settings.readonly_retries if spec.readonly and spec.retry_safe else 0)
        for attempt in range(attempts):
            try:
                self._reserve(control)
                if spec.readonly:
                    # Cancelling a read's await does NOT kill the worker thread.
                    result = await asyncio.wait_for(
                        asyncio.to_thread(spec.function, **params),
                        timeout=self.settings.tool_timeout_seconds,
                    )
                elif spec.model_call:
                    result = await asyncio.wait_for(
                        spec.function(**params), timeout=self.settings.tool_timeout_seconds
                    )
                else:
                    # Local synchronous writes commit together with their effect record.
                    # Their timeout is SOFT: measured after completion, never retried.
                    write_started = time.perf_counter()
                    result, _ = self._write_once(spec, params, control, effect_id)
                    if time.perf_counter() - write_started > self.settings.tool_timeout_seconds:
                        raise ToolTimeoutError(
                            "Local write exceeded its soft deadline after completion",
                            detail={"operation_status": "completed", "timeout_kind": "soft"},
                        )
                self._record(control, name, "success", started, attempt=attempt + 1)
                return result
            except (TimeoutError, ConnectionError, OSError) as exc:
                if attempt + 1 < attempts:
                    self._record(control, name, "retry", started, attempt=attempt + 1)
                    continue
                if isinstance(exc, TimeoutError):
                    self._record(control, name, "timeout", started, code="TOOL_TIMEOUT")
                    raise ToolTimeoutError(
                        f"Tool {name!r} exceeded its await deadline",
                        detail={"timeout_kind": "await", "downstream_stopped": False},
                    ) from exc
                self._record(control, name, "failed", started, code="TOOL_EXECUTION_FAILED")
                raise ToolExecutionError(f"Tool {name!r} failed after bounded attempts") from exc
            except LLMTimeoutError as exc:
                self._record(control, name, "timeout", started, code="TOOL_TIMEOUT")
                raise ToolTimeoutError(
                    "Model provider reported a timeout", detail={"downstream_stopped": False}
                ) from exc
            except LLMError as exc:
                self._record(control, name, "failed", started, code="TOOL_EXECUTION_FAILED")
                raise ToolExecutionError(f"Model operation {name!r} failed") from exc
            except EduAgentError as exc:
                self._record(control, name, "failed", started, code=exc.code)
                raise
            except Exception as exc:
                self._record(control, name, "failed", started, code="TOOL_EXECUTION_FAILED")
                raise ToolExecutionError(f"Tool {name!r} execution failed") from exc
