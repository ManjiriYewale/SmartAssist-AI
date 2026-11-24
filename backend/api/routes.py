"""REST API (FastAPI). Mirrors what the Streamlit UI does through ``backend.services``."""
from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from backend import services
from backend.config.settings import get_settings
from backend.database import repository as repo
from backend.database.db import check_connection
from backend.exceptions import (DatabaseError, LLMConfigurationError, LLMServiceError, NotFoundError,
                                SmartAssistError, ToolInputError, UploadValidationError)
from backend.schemas import ChatRequest, SearchRequest, ToolRunRequest
from backend.tools import tool_catalog

logger = logging.getLogger(__name__)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    token = get_settings().api_auth_token
    if token and not (x_api_key and hmac.compare_digest(x_api_key, token)):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


router = APIRouter(dependencies=[Depends(require_api_key)])
public = APIRouter()


def http_error(exc: SmartAssistError) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(404, exc.user_message)
    if isinstance(exc, (UploadValidationError, ToolInputError)):
        return HTTPException(422 if isinstance(exc, ToolInputError) else 400, exc.user_message)
    if isinstance(exc, LLMConfigurationError):
        return HTTPException(503, exc.user_message)
    if isinstance(exc, LLMServiceError):
        return HTTPException(502, exc.user_message)
    if isinstance(exc, DatabaseError):
        return HTTPException(503, exc.user_message)
    return HTTPException(500, exc.user_message)


@public.get("/health")
def health() -> dict:
    ok, msg = check_connection()
    s = get_settings()
    return {"status": "ok" if ok else "degraded", "database": msg, "llm_configured": s.llm_configured, "model": s.groq_model}


@router.get("/documents")
def list_documents(search: str | None = None, status: str | None = None) -> list[dict]:
    try:
        return repo.list_documents(search=search, status=status)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    max_bytes = get_settings().max_upload_bytes
    data = await file.read(max_bytes + 1)  # never buffer more than limit+1 bytes
    try:
        return services.upload_document(file.filename or "", data)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.get("/documents/{doc_id}")
def get_document(doc_id: int) -> dict:
    try:
        return repo.get_document(doc_id)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.post("/documents/{doc_id}/reprocess")
def reprocess(doc_id: int) -> dict:
    try:
        repo.get_document(doc_id)
        return services.reprocess_document(doc_id)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int) -> dict:
    try:
        services.delete_document(doc_id)
        return {"deleted": doc_id}
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.post("/search")
def search(req: SearchRequest) -> dict:
    try:
        return services.explore_retrieval(req.query, top_k=req.top_k, doc_id=req.document_id,
                                          file_type=req.file_type, min_score=req.min_score)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    try:
        return services.ask(req.message, req.conversation_id)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.get("/conversations")
def conversations() -> list[dict]:
    return repo.list_conversations()


@router.get("/conversations/{conv_id}/messages")
def messages(conv_id: int) -> list[dict]:
    try:
        repo.get_conversation(conv_id)
        return repo.get_messages(conv_id)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int) -> dict:
    try:
        repo.delete_conversation(conv_id)
        return {"deleted": conv_id}
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.get("/tools")
def tools() -> list[dict]:
    return tool_catalog()


@router.post("/tools/{name}/run")
def run_tool(name: str, req: ToolRunRequest) -> dict:
    try:
        return services.run_tool_manually(name, req.arguments)
    except SmartAssistError as exc:
        raise http_error(exc) from exc


@router.get("/stats")
def stats() -> dict:
    return {"dashboard": repo.get_dashboard_stats(), "tools": repo.get_tool_stats(),
            "intents": repo.get_intent_distribution(), "documents_by_status": repo.get_document_status_counts()}
