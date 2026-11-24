"""Semantic retrieval with metadata filtering and relevance scores."""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.rag import vector_store


class RetrievedChunk(BaseModel):
    content: str
    doc_id: int
    filename: str
    file_type: str = ""
    page: int = 0
    sheet: str = ""
    chunk_index: int = 0
    score: float = Field(0.0, description="Cosine similarity, 0..1 (higher = more relevant)")

    def citation(self) -> str:
        parts = [self.filename]
        if self.page:
            parts.append(f"p. {self.page}")
        if self.sheet:
            parts.append(f"sheet '{self.sheet}'")
        return ", ".join(parts)


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    min_score: float
    chunks: list[RetrievedChunk] = []
    below_threshold: int = 0

    @property
    def found(self) -> bool:
        return bool(self.chunks)


def retrieve(query: str, *, top_k: int | None = None, doc_ids: list[int] | None = None,
             file_type: str | None = None, min_score: float | None = None) -> RetrievalResult:
    s = get_settings()
    k = top_k or s.top_k
    threshold = s.min_relevance_score if min_score is None else min_score
    query = (query or "").strip()
    if not query:
        return RetrievalResult(query="", top_k=k, min_score=threshold)
    pairs = vector_store.similarity_search(query, k, doc_ids=doc_ids, file_type=file_type)
    kept: list[RetrievedChunk] = []
    for doc, score in pairs:
        if score < threshold:
            continue
        m = doc.metadata
        kept.append(
            RetrievedChunk(
                content=doc.page_content, doc_id=int(m.get("doc_id", 0)), filename=str(m.get("filename", "")),
                file_type=str(m.get("file_type", "")), page=int(m.get("page", 0) or 0), sheet=str(m.get("sheet", "") or ""),
                chunk_index=int(m.get("chunk_index", 0)), score=round(score, 4),
            )
        )
    return RetrievalResult(query=query, top_k=k, min_score=threshold, chunks=kept, below_threshold=len(pairs) - len(kept))


def format_context(chunks: list[RetrievedChunk | dict], max_chars: int = 9000) -> str:
    """Numbered context block for prompts: ``[1] (file, p. 3) text``."""
    out: list[str] = []
    used = 0
    for i, c in enumerate(chunks, 1):
        c = c if isinstance(c, RetrievedChunk) else RetrievedChunk(**c)
        block = f"[{i}] ({c.citation()}, relevance {c.score:.2f})\n{c.content.strip()}"
        if used + len(block) > max_chars and out:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)
