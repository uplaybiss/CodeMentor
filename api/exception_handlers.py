"""Global exception handler — maps EduAgentError tree to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils.errors import EduAgentError


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the error-to-HTTP mapper to ``app``."""

    @app.exception_handler(EduAgentError)
    async def eduagent_error_handler(
        request: Request, exc: EduAgentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": str(exc.message),
                "detail": exc.detail,
            },
        )
