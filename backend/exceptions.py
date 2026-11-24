"""Application-specific exceptions.

Every exception carries a ``user_message`` that is safe to show in the UI / API.
Internal details (stack traces, SQL, API responses) are only written to the log.
"""
from __future__ import annotations


class SmartAssistError(Exception):
    """Base class. ``user_message`` is always safe to display to end users."""

    default_message = "Something went wrong. Please try again."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None):
        self.user_message = user_message or self.default_message
        self.detail = detail
        super().__init__(self.user_message)


class UploadValidationError(SmartAssistError):
    default_message = "The uploaded file is not valid."


class DocumentProcessingError(SmartAssistError):
    default_message = "The document could not be processed."


class NotFoundError(SmartAssistError):
    default_message = "The requested item was not found."


class ToolInputError(SmartAssistError):
    default_message = "The tool received invalid input."


class DatabaseError(SmartAssistError):
    default_message = "A database error occurred. Check that PostgreSQL is running and DATABASE_URL is correct."


class LLMConfigurationError(SmartAssistError):
    default_message = "The LLM is not configured. Set GROQ_API_KEY in your .env file."


class LLMServiceError(SmartAssistError):
    default_message = "The AI service is temporarily unavailable. Please try again."


class LLMToolCallError(LLMServiceError):
    default_message = "The model produced an invalid tool call."
