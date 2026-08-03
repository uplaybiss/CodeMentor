"""Unified async client for the supported LLM providers.

Design rationale
----------------
* Every agent talks to the LLM *only* through :class:`LLMClient`. The provider
  (``openai``, ``gemini`` or ``demo``) is chosen once from ``LLMSettings`` at
  construction, so switching providers is a one-line config change — the agents
  never know which backend they are on.
* ``generate_json`` is the workhorse: it asks the model for JSON and validates
  the result against a Pydantic schema *before* the caller sees it, so malformed
  LLM output surfaces as a typed :class:`LLMOutputParsingError` at the boundary
  instead of a crash deep inside a graph node.
* All upstream failures (network, rate-limit, timeout) are wrapped into
  :class:`LLMError` and retried with exponential backoff via
  :func:`utils.async_utils.async_retry`.
* The model is built *lazily* so the app boots without an API key configured —
  the key is only required when a call is actually attempted. The ``demo``
  provider returns deterministic, topic-aware content so the whole system is
  runnable offline (CI, demos, first-run experience).
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from config.settings import LLMSettings
from utils.async_utils import async_retry
from utils.errors import (
    LLMError,
    LLMOutputParsingError,
    LLMTimeoutError,
    ProviderConfigurationError,
)
from utils.helpers import safe_json_parse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_REAL_PROVIDERS = ("openai", "gemini")


class LLMClient:
    """Async wrapper around a langchain chat model with JSON validation."""

    provider: str

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        provider = (settings.provider or "openai").strip().lower()
        if provider not in _REAL_PROVIDERS:
            raise ProviderConfigurationError(
                f"Unsupported LLM_PROVIDER={provider!r}; expected one of "
                f"{(*_REAL_PROVIDERS, 'demo')}"
            )
        self.provider = provider
        self._model: Any | None = None  # lazily constructed on first call

    # ------------------------------------------------------------- factory --
    def _get_model(self) -> Any:
        """Build (once) the langchain chat model for the selected provider."""
        if self._model is not None:
            return self._model
        settings = self._settings
        temperature = (
            settings.openai_temperature
            if self.provider == "openai"
            else settings.gemini_temperature
        )
        if self.provider == "openai":
            if not settings.openai_api_key:
                raise ProviderConfigurationError(
                    "LLM_PROVIDER=openai but LLM_OPENAI_API_KEY is not set. "
                    "Set it in .env (or switch LLM_PROVIDER=demo) before chatting."
                )
            self._model = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=temperature,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
            )
        else:
            if not settings.gemini_api_key:
                raise ProviderConfigurationError(
                    "LLM_PROVIDER=gemini but LLM_GEMINI_API_KEY is not set. "
                    "Set it in .env (or switch LLM_PROVIDER=demo) before chatting."
                )
            self._model = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=temperature,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
            )
        logger.info("LLM provider ready: %s (%s)", self.provider, self._model.model)
        return self._model

    # ------------------------------------------------------------ generate --
    async def generate(
        self, system: str, user: str, *, temperature: float | None = None
    ) -> str:
        """Return the raw text completion for a system+user message pair."""
        attempts = self._settings.max_retries + 1
        model = self._get_model()
        if temperature is not None and self.provider == "openai":
            model = ChatOpenAI(
                model=self._settings.openai_model,
                api_key=self._settings.openai_api_key,
                temperature=temperature,
                timeout=self._settings.timeout_seconds,
                max_retries=self._settings.max_retries,
            )

        async def _call() -> str:
            try:
                response = await model.ainvoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
            except Exception as exc:  # langchain raises provider-specific errors
                if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
                    raise LLMTimeoutError(f"LLM call timed out: {exc}") from exc
                raise LLMError(f"LLM call failed: {exc}") from exc
            content = getattr(response, "content", None)
            if not content:
                raise LLMError("LLM returned an empty response")
            return str(content)

        return await async_retry(_call, attempts=attempts, label="llm.generate")

    # -------------------------------------------------------- generate_json --
    async def generate_json(
        self,
        system: str,
        user: str,
        *,
        model_class: type[T],
        temperature: float | None = None,
    ) -> T:
        """Ask the model for JSON and return a validated ``model_class`` instance.

        Fenced ```json blocks are stripped before parsing. Any parse or
        validation failure raises :class:`LLMOutputParsingError`.
        """
        text = await self.generate(system, user, temperature=temperature)
        payload = safe_json_parse(text)
        if payload is None:
            raise LLMOutputParsingError(
                "Model did not return parseable JSON. "
                f"First 200 chars: {text[:200]!r}"
            )
        try:
            return model_class.model_validate(payload)
        except ValidationError as exc:
            raise LLMOutputParsingError(
                f"Model JSON failed {model_class.__name__} validation: {exc}"
            ) from exc

    # -------------------------------------------------------------- access --
    @property
    def settings(self) -> LLMSettings:
        return self._settings


def build_llm_client(settings: LLMSettings) -> LLMClient:
    """Return the appropriate client for ``settings.provider``.

    The ``demo`` provider short-circuits to a deterministic, offline client so
    the entire application can be exercised without any external API key.
    """
    provider = (settings.provider or "openai").strip().lower()
    if provider == "demo":
        from services.llm_demo import DemoLLMClient

        return DemoLLMClient(settings)
    return LLMClient(settings)
