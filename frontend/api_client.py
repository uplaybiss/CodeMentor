"""Thin httpx client for the EduAgent REST API.

Kept deliberately dumb — the Streamlit app maps these calls to widgets and
charts. Every method raises :class:`EduAgentClientError` on transport or HTTP
errors so the UI can show a friendly message instead of a stack trace.
"""

from __future__ import annotations

from typing import Any

import httpx

from config.settings import get_settings


class EduAgentClientError(RuntimeError):
    """Raised when the API is unreachable or returns an error."""


class EduAgentClient:
    """HTTP client wrapping the EduAgent API."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.frontend.api_base_url).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------- helpers --
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise EduAgentClientError(
                f"Cannot reach the EduAgent API at {url!r}: {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.is_error:
            message = ""
            if isinstance(payload, dict):
                message = payload.get("detail") or payload.get("message") or ""
            raise EduAgentClientError(
                f"API error {response.status_code} on {method.upper()} {path}"
                + (f": {message}" if message else "")
            )
        return payload

    # -------------------------------------------------------------- health --
    def health(self) -> dict:
        return self._request("GET", "/health") or {}

    # ---------------------------------------------------------------- chat --
    def chat(
        self,
        student_id: str,
        message: str,
        topic: str | None = None,
        with_quiz: bool = False,
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/chat",
            json={
                "student_id": student_id,
                "message": message,
                "topic": topic,
                "with_quiz": with_quiz,
            },
        )

    def submit_quiz(
        self, student_id: str, quiz_id: str, answers: list[dict]
    ) -> dict:
        return self._request(
            "POST",
            "/api/v1/chat/submit-quiz",
            json={
                "quiz_id": quiz_id,
                "student_id": student_id,
                "answers": answers,
            },
        )

    # ------------------------------------------------------------- students --
    def get_student(self, student_id: str) -> dict:
        return self._request("GET", f"/api/v1/students/{student_id}")

    def update_student(self, student_id: str, **fields: Any) -> dict:
        return self._request(
            "PUT", f"/api/v1/students/{student_id}", json=fields
        )

    # -------------------------------------------------------------- quizzes --
    def list_quizzes(self, student_id: str) -> dict:
        return self._request("GET", f"/api/v1/quizzes/{student_id}")

    def get_quiz(self, quiz_id: str) -> dict:
        return self._request("GET", f"/api/v1/quiz/{quiz_id}")

    # ----------------------------------------------------------- dashboard --
    def get_dashboard(self, student_id: str) -> dict:
        return self._request("GET", f"/api/v1/dashboard/{student_id}")

    # ------------------------------------------------------------ documents --
    def upload_document(
        self, student_id: str, filename: str, content: bytes
    ) -> dict:
        files = {"file": (filename, content)}
        return self._request(
            "POST", f"/api/v1/documents/{student_id}", files=files
        )

    def list_documents(self, student_id: str) -> list[dict]:
        return self._request("GET", f"/api/v1/documents/{student_id}") or []

    def delete_document(self, doc_id: str) -> dict:
        return self._request("DELETE", f"/api/v1/documents/{doc_id}")
