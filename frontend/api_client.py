"""HTTP client with safe Chinese errors and explicit run identifiers."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

ERROR_MESSAGES = {
    "TOOL_FORBIDDEN": "这项学习内容不属于当前档案，请切换档案后重试。",
    "TOOL_PARAMETER_ERROR": "提交内容不完整，请检查后重试。",
    "TOOL_TIMEOUT": "本次辅导等待时间较长，可以稍后继续同一次请求。",
    "TOOL_EXECUTION_FAILED": "辅导暂时未完成，可以继续同一次请求。",
    "TOOL_BUDGET_EXHAUSTED": "本次辅导已达到执行上限，请开始新的学习请求。",
    "TOOL_REPEAT_LIMIT": "本次辅导出现重复操作，已停止，请开始新的学习请求。",
    "CONTEXT_BUDGET_EXHAUSTED": "学习目标过长，请缩短描述后重新发送。",
    "GRAPH_RECURSION_LIMIT": "本次辅导步骤超过上限，请开始新的学习请求。",
    "RUN_CONFLICT": "这次请求已存在。请继续原请求，或发送一条新的消息。",
    "NOT_FOUND": "没有找到对应的学习记录，请重新选择。",
}


class EduAgentClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        status: int | None = None,
        runtime: dict | None = None,
    ):
        super().__init__(message)
        self.code, self.status, self.runtime = code, status, runtime


class EduAgentClient:
    """UI import never loads provider credentials; environment overrides default8013."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("FRONTEND_API_BASE_URL") or "http://127.0.0.1:8013"
        ).rstrip("/")
        try:
            self._client = httpx.Client(
                base_url=self.base_url, timeout=timeout, transport=transport, trust_env=False
            )
        except (ValueError, httpx.InvalidURL) as exc:
            raise EduAgentClientError(
                "学习服务地址配置有误，请检查启动时的连接地址。", code="INVALID_ADDRESS"
            ) from exc

    def close(self):
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any):
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise EduAgentClientError(
                "请求等待时间较长，请稍后继续。", code="NETWORK_TIMEOUT"
            ) from exc
        except httpx.HTTPError as exc:
            raise EduAgentClientError(
                "暂时连接不上学习服务，请确认后端已启动后重新连接。", code="NETWORK_UNAVAILABLE"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise EduAgentClientError(
                "学习服务返回了异常内容，请稍后重试。", code="INVALID_RESPONSE"
            ) from exc
        if response.is_error:
            code = payload.get("error", "") if isinstance(payload, dict) else ""
            runtime = None
            if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
                detail = payload["detail"]
                runtime = detail.get("runtime") or (detail if "run_id" in detail else None)
            fallback = {
                404: "没有找到对应的学习记录，请重新选择。",
                422: "提交内容不完整，请检查后重试。",
                503: "学习服务暂时不可用，请稍后重试。",
            }.get(response.status_code, "这次操作暂时未完成，请稍后重试。")
            raise EduAgentClientError(
                ERROR_MESSAGES.get(code, fallback),
                code=code,
                status=response.status_code,
                runtime=runtime,
            )
        return payload

    @staticmethod
    def _id(value: str):
        return quote(value, safe="")

    def health(self):
        return self._request("GET", "/health", timeout=2.0) or {}

    def chat(
        self,
        student_id: str,
        message: str,
        topic: str | None = None,
        with_quiz: bool = False,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        payload = {
            "student_id": student_id,
            "message": message,
            "topic": topic,
            "with_quiz": with_quiz,
        }
        if session_id:
            payload["session_id"] = session_id
        if run_id:
            payload["run_id"] = run_id
        return self._request("POST", "/api/v1/chat", json=payload)

    def submit_quiz(
        self,
        student_id: str,
        quiz_id: str,
        answers: list[dict],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        payload = {"quiz_id": quiz_id, "student_id": student_id, "answers": answers}
        if session_id:
            payload["session_id"] = session_id
        if run_id:
            payload["run_id"] = run_id
        return self._request("POST", "/api/v1/chat/submit-quiz", json=payload)

    def run_status(self, student_id: str, session_id: str, run_id: str, workflow: str = "tutoring"):
        return self._request(
            "GET",
            f"/api/v1/chat/runs/{self._id(session_id)}/{self._id(run_id)}",
            params={"student_id": student_id, "workflow": workflow},
        )

    def resume(self, student_id: str, session_id: str, run_id: str, workflow: str = "tutoring"):
        return self._request(
            "POST",
            "/api/v1/chat/resume",
            json={
                "student_id": student_id,
                "session_id": session_id,
                "run_id": run_id,
                "workflow": workflow,
            },
        )

    def get_student(self, student_id):
        return self._request("GET", f"/api/v1/students/{self._id(student_id)}")

    def update_student(self, student_id, **fields):
        return self._request("PUT", f"/api/v1/students/{self._id(student_id)}", json=fields)

    def list_quizzes(self, student_id):
        return self._request("GET", f"/api/v1/quizzes/{self._id(student_id)}")

    def get_quiz(self, quiz_id):
        return self._request("GET", f"/api/v1/quiz/{self._id(quiz_id)}")

    def get_dashboard(self, student_id):
        return self._request("GET", f"/api/v1/dashboard/{self._id(student_id)}")

    def upload_document(self, student_id, filename, content):
        return self._request(
            "POST", f"/api/v1/documents/{self._id(student_id)}", files={"file": (filename, content)}
        )

    def list_documents(self, student_id):
        return self._request("GET", f"/api/v1/documents/{self._id(student_id)}") or []

    def delete_document(self, doc_id):
        return self._request("DELETE", f"/api/v1/documents/{self._id(doc_id)}")
