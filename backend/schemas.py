"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[int] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    document_id: Optional[int] = None
    file_type: Optional[str] = None
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class ToolRunRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
