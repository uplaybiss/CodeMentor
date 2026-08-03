"""Typed exception hierarchy for EduAgent.

Design rationale
----------------
A small, purpose-built exception tree lets every layer raise *domain-meaningful*
errors and lets the API layer translate them to HTTP responses with a single
registry. Each exception carries an ``HTTPStatus``-compatible ``status_code`` and
a stable machine-readable ``code`` so clients can branch without string-matching.
"""

from __future__ import annotations


class EduAgentError(Exception):
    """Base class for all EduAgent-specific exceptions."""

    status_code: int = 500
    code: str = "EDUAGENT_ERROR"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConfigurationError(EduAgentError):
    """Missing/invalid runtime configuration (e.g. no LLM API key set)."""

    status_code = 500
    code = "CONFIGURATION_ERROR"


class ProviderConfigurationError(ConfigurationError):
    """The selected LLM provider is unsupported or misconfigured."""

    code = "LLM_PROVIDER_CONFIG"


class DatabaseError(EduAgentError):
    """Any persistence failure (connection, integrity, constraint)."""

    status_code = 500
    code = "DATABASE_ERROR"


class LLMError(EduAgentError):
    """Failure while calling the LLM provider (upstream / network)."""

    status_code = 502
    code = "LLM_ERROR"


class LLMTimeoutError(LLMError):
    """The LLM call exceeded the configured timeout window."""

    code = "LLM_TIMEOUT"


class LLMOutputParsingError(LLMError):
    """The model returned output that could not be validated/parsed."""

    code = "LLM_OUTPUT_PARSE"


class IngestionError(EduAgentError):
    """A document could not be loaded, chunked or embedded."""

    status_code = 422
    code = "INGESTION_ERROR"


class RetrievalError(EduAgentError):
    """Semantic retrieval failed (collection missing, embedder error)."""

    status_code = 500
    code = "RETRIEVAL_ERROR"


class ResourceNotFoundError(EduAgentError):
    """Requested entity (student, quiz, document, ...) does not exist."""

    status_code = 404
    code = "NOT_FOUND"


class ValidationError(EduAgentError):
    """Input payload failed domain validation."""

    status_code = 422
    code = "VALIDATION_ERROR"
