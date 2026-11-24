"""TOOL 6 – Document management: list, metadata, status, availability."""
from __future__ import annotations

from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.database import repository as repo
from backend.exceptions import ToolInputError
from backend.tools.common import resolve_document, safe_tool_call


def _brief(d: dict) -> dict:
    return {"id": d["id"], "filename": d["filename"], "type": d["file_type"], "status": d["status"],
            "size_kb": round(d["size_bytes"] / 1024, 1), "pages": d["page_count"], "chunks": d["chunk_count"],
            "uploaded_at": d["created_at"]}


def library_overview(max_items: int = 40) -> str:
    """One-line-per-document text used inside the agent's system prompt."""
    docs = repo.list_documents()
    if not docs:
        return "(no documents uploaded yet)"
    lines = [f"- #{d['id']} {d['filename']} [{d['file_type']}, {d['status']}]" for d in docs[:max_items]]
    if len(docs) > max_items:
        lines.append(f"- … and {len(docs) - max_items} more")
    return "\n".join(lines)


def run_document_manager(action: str, document: str | None = None, file_type: str | None = None) -> dict:
    if action == "list":
        docs = repo.list_documents(file_type=file_type.lower().lstrip(".") if file_type else None)
        return {"action": action, "count": len(docs), "documents": [_brief(d) for d in docs]}
    if action == "available":
        docs = repo.list_documents(status="processed", file_type=file_type.lower().lstrip(".") if file_type else None)
        return {"action": action, "count": len(docs), "documents": [_brief(d) for d in docs]}
    if action in {"metadata", "status"}:
        if not document:
            raise ToolInputError("Please provide the 'document' name or id.")
        d = resolve_document(document, only_processed=False)
        if action == "status":
            return {"action": action, "id": d["id"], "filename": d["filename"], "status": d["status"],
                    "error": d["error_message"]}
        return {"action": action, "document": {**_brief(d), "characters": d["char_count"], "processed_at": d["processed_at"],
                                               "processing_settings": d["extra_metadata"], "error": d["error_message"]}}
    raise ToolInputError(f"Unknown action '{action}'.")


class DocumentManagerInput(BaseModel):
    action: Literal["list", "available", "metadata", "status"] = Field(
        description="list: all uploaded files; available: only processed files ready for questions; "
                    "metadata: details of one file; status: processing status of one file.")
    document: Optional[str] = Field(None, description="File name (or id) – required for metadata / status.")
    file_type: Optional[str] = Field(None, description="Optional filter for list/available: pdf, docx, txt, csv, xlsx.")


@tool("document_manager", args_schema=DocumentManagerInput)
def document_manager(action: str, document: Optional[str] = None, file_type: Optional[str] = None) -> str:
    """Inspect the document library: list uploaded documents, list processed/available ones,
    get a document's metadata (type, size, pages, chunks) or check its processing status."""
    return safe_tool_call("document_manager", lambda: run_document_manager(action, document, file_type))
