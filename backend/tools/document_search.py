"""TOOL 2 – Document search (semantic retrieval with source information)."""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.rag.retriever import retrieve
from backend.tools.common import resolve_document, safe_tool_call, to_json, truncate


class DocumentSearchInput(BaseModel):
    query: str = Field(description="What to look for, phrased as a search query.", min_length=2, max_length=500)
    document: Optional[str] = Field(None, description="Optional file name or id to restrict the search to one document.")
    file_type: Optional[str] = Field(None, description="Optional filter: pdf, docx, txt, csv or xlsx.")
    top_k: int = Field(5, ge=1, le=10, description="How many passages to return.")


def run_document_search(query: str, document: str | None = None, file_type: str | None = None, top_k: int = 5) -> dict:
    doc_ids = None
    if document:
        doc_ids = [resolve_document(document)["id"]]
    if file_type:
        file_type = file_type.lower().lstrip(".")
    res = retrieve(query, top_k=top_k or get_settings().top_k, doc_ids=doc_ids, file_type=file_type)
    return {
        "query": query,
        "found": res.found,
        "message": "" if res.found else "No sufficiently relevant passages were found in the uploaded documents.",
        "results": [
            {"rank": i, "source": c.citation(), "document": c.filename, "page": c.page or None,
             "score": c.score, "text": truncate(c.content, 1200)}
            for i, c in enumerate(res.chunks, 1)
        ],
        "chunks": [c.model_dump() for c in res.chunks],
    }


@tool("document_search", args_schema=DocumentSearchInput, response_format="content_and_artifact")
def document_search(query: str, document: Optional[str] = None, file_type: Optional[str] = None, top_k: int = 5):
    """Semantic search over the user's uploaded documents. Returns the most relevant passages with source file,
    page number and a relevance score. Use it to answer any question about document content."""
    holder: dict = {}

    def body():
        r = run_document_search(query, document, file_type, top_k)
        holder["chunks"] = r.pop("chunks")
        return r

    content = safe_tool_call("document_search", body)
    return content, holder.get("chunks", [])
