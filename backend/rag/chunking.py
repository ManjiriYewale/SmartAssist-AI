"""Text cleaning and chunking."""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config.settings import get_settings


def clean_text(text: str) -> str:
    """Normalise extracted text without destroying structure."""
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # re-join hyphenated line breaks
    text = re.sub(r"[ \t\f\v\u00a0]+", " ", text)            # collapse spaces
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)                   # collapse blank lines
    return text.strip()


def get_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    s = get_settings()
    size = chunk_size or s.chunk_size
    overlap = s.chunk_overlap if chunk_overlap is None else chunk_overlap
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(
    docs: list[Document],
    *,
    doc_id: int,
    filename: str,
    file_type: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Clean + split into chunks, attaching Chroma-safe (primitive) metadata."""
    cleaned: list[Document] = []
    for d in docs:
        text = clean_text(d.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=dict(d.metadata)))
    chunks = get_splitter(chunk_size, chunk_overlap).split_documents(cleaned)
    out: list[Document] = []
    for i, c in enumerate(chunks):
        md = c.metadata
        out.append(
            Document(
                page_content=c.page_content,
                metadata={
                    "doc_id": int(doc_id),
                    "filename": filename,
                    "file_type": file_type,
                    "page": int(md.get("page") or 0),          # 0 = not applicable
                    "sheet": str(md.get("sheet") or ""),
                    "row_start": int(md.get("row_start") or 0),
                    "row_end": int(md.get("row_end") or 0),
                    "chunk_index": i,
                    "char_count": len(c.page_content),
                },
            )
        )
    return out
