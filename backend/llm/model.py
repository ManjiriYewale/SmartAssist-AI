"""LLM access (Groq through LangChain) with centralised error translation."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from backend.config.settings import get_settings
from backend.exceptions import LLMConfigurationError, LLMServiceError, LLMToolCallError

logger = logging.getLogger(__name__)

_override: BaseChatModel | None = None


def set_llm_override(llm: BaseChatModel | None) -> None:
    """Inject a different chat model (unit tests / offline demos)."""
    global _override
    _override = llm


@lru_cache(maxsize=8)
def _build_groq(model: str, temperature: float, key_fingerprint: int, timeout: int, retries: int) -> BaseChatModel:
    from langchain_groq import ChatGroq

    s = get_settings()
    return ChatGroq(
        model=model,
        api_key=s.groq_api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=retries,
    )


def get_llm(temperature: float | None = None) -> BaseChatModel:
    if _override is not None:
        return _override
    s = get_settings()
    if not s.llm_configured:
        raise LLMConfigurationError()
    temp = s.llm_temperature if temperature is None else temperature
    return _build_groq(s.groq_model, float(temp), hash(s.groq_api_key), s.llm_timeout_seconds, s.llm_max_retries)


def translate_llm_error(exc: Exception) -> LLMServiceError:
    """Map provider exceptions to safe, actionable messages (details go to the log only)."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    status = getattr(exc, "status_code", None)
    if "tool_use_failed" in low or "failed to call a function" in low or "failed_generation" in low:
        return LLMToolCallError(detail=text)
    if name == "RateLimitError" or status == 429 or "rate limit" in low:
        return LLMServiceError("The AI service is rate-limited right now. Please wait a few seconds and try again.", detail=text)
    if name == "AuthenticationError" or status in (401, 403):
        return LLMServiceError("The Groq API key was rejected. Check GROQ_API_KEY in your .env file.", detail=text)
    if "model_decommissioned" in low or "model_not_found" in low or status == 404 or "decommissioned" in low:
        return LLMServiceError(
            f"The configured model '{get_settings().groq_model}' is not available. Change GROQ_MODEL in .env.", detail=text
        )
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout"} or "connection" in low or "timed out" in low:
        return LLMServiceError("Could not reach the AI service. Check your internet connection and try again.", detail=text)
    if status and status >= 500:
        return LLMServiceError("The AI service reported a temporary problem. Please try again shortly.", detail=text)
    return LLMServiceError(detail=text)


def invoke_llm(runnable: Any, payload: Any, **kwargs: Any):
    """Invoke any runnable, translating provider errors into LLMServiceError."""
    try:
        return runnable.invoke(payload, **kwargs)
    except LLMServiceError:
        raise
    except Exception as exc:  # noqa: BLE001
        err = translate_llm_error(exc)
        logger.error("LLM call failed (%s): %s", type(exc).__name__, exc)
        raise err from exc


def check_llm_connection() -> tuple[bool, str]:
    """Tiny live call used by the Settings page. Never raises."""
    try:
        llm = get_llm(temperature=0)
        invoke_llm(llm, "Reply with the single word: ok")
        return True, f"Connected to Groq model '{get_settings().groq_model}'."
    except LLMConfigurationError as exc:
        return False, exc.user_message
    except LLMServiceError as exc:
        return False, exc.user_message
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error in check_llm_connection")
        return False, "Unexpected error while contacting the LLM."
