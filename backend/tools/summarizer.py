"""TOOL 4 – Document summarisation (map-reduce for long documents)."""
from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from backend.chains.prompts import SUMMARY_FINAL_PROMPT, SUMMARY_MAP_PROMPT, SUMMARY_MODE_INSTRUCTIONS
from backend.llm.model import get_llm, invoke_llm
from backend.rag.document_loader import load_text
from backend.rag.chunking import clean_text
from backend.rag.ingestion import stored_path
from backend.tools.common import resolve_document, safe_tool_call

logger = logging.getLogger(__name__)

SEGMENT_CHARS = 9000
MAX_SEGMENTS = 10


def condense_text(text: str, filename: str, *, segment_chars: int = SEGMENT_CHARS, max_segments: int = MAX_SEGMENTS) -> tuple[str, dict]:
    """Return text small enough for one LLM call, plus coverage info (map-reduce)."""
    if len(text) <= segment_chars:
        return text, {"coverage": "full", "sections": 1}
    splitter = RecursiveCharacterTextSplitter(chunk_size=segment_chars, chunk_overlap=0)
    parts = splitter.split_text(text)
    total = len(parts)
    if total > max_segments:  # sample evenly so every part of the document is represented
        step = total / max_segments
        parts = [parts[int(i * step)] for i in range(max_segments)]
    chain = SUMMARY_MAP_PROMPT | get_llm(temperature=0) | StrOutputParser()
    notes = []
    for i, p in enumerate(parts, 1):
        notes.append(invoke_llm(chain, {"filename": filename, "i": i, "n": len(parts), "text": p}))
    coverage = "full" if total <= max_segments else f"sampled {len(parts)} of {total} sections"
    return "\n\n".join(notes), {"coverage": coverage, "sections": len(parts)}


def run_summarizer(document: str | None, mode: str = "short") -> dict:
    if mode not in SUMMARY_MODE_INSTRUCTIONS:
        mode = "short"
    doc = resolve_document(document)
    text, _pages = load_text(stored_path(doc["stored_name"]), doc["file_type"])
    text = clean_text(text)
    condensed, info = condense_text(text, doc["filename"])
    chain = SUMMARY_FINAL_PROMPT | get_llm(temperature=0.1) | StrOutputParser()
    summary = invoke_llm(chain, {"filename": doc["filename"], "mode_instruction": SUMMARY_MODE_INSTRUCTIONS[mode], "text": condensed})
    return {"document": doc["filename"], "mode": mode, "summary": summary.strip(), "characters_read": len(text), **info}


class SummarizerInput(BaseModel):
    document: Optional[str] = Field(None, description="File name or id. Optional if only one document exists.")
    mode: Literal["short", "detailed", "key_points", "findings"] = Field(
        "short", description="short: brief summary; detailed: structured long summary; key_points: bullet list; findings: important findings/figures.")


@tool("document_summarizer", args_schema=SummarizerInput)
def document_summarizer(document: Optional[str] = None, mode: str = "short") -> str:
    """Summarise ONE uploaded document: short summary, detailed summary, key points or important findings.
    Handles long documents by condensing them section by section."""
    return safe_tool_call("document_summarizer", lambda: run_summarizer(document, mode))
