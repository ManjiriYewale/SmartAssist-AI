"""TOOL 5 – Compare two documents (computed overlap metrics + LLM-written report)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.chains.prompts import COMPARE_PROMPT
from backend.config.settings import TABULAR_TYPES
from backend.exceptions import ToolInputError
from backend.llm.model import get_llm, invoke_llm
from backend.rag.chunking import clean_text
from backend.rag.document_loader import load_text
from backend.rag.embeddings import _STOP
from backend.rag.ingestion import stored_path
from backend.tools.common import resolve_document, safe_tool_call
from backend.tools.csv_analyzer import load_dataframe
from backend.tools.summarizer import condense_text

COMPARE_CHARS = 9000
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}|\d[\d,\.]*")


def term_overlap(text_a: str, text_b: str) -> dict:
    """Deterministic similarity metrics (Jaccard on vocabulary + distinctive terms)."""
    ca = Counter(w.lower() for w in _WORD.findall(text_a) if w.lower() not in _STOP)
    cb = Counter(w.lower() for w in _WORD.findall(text_b) if w.lower() not in _STOP)
    sa, sb = set(ca), set(cb)
    union = sa | sb
    jaccard = len(sa & sb) / len(union) if union else 0.0
    only_a = [w for w, _ in (ca - cb).most_common(40) if w not in sb][:12]
    only_b = [w for w, _ in (cb - ca).most_common(40) if w not in sa][:12]
    shared = [w for w, _ in (ca & cb).most_common(12)]
    return {"vocabulary_jaccard_similarity": round(jaccard, 3), "shared_terms": shared,
            "terms_only_in_a": only_a, "terms_only_in_b": only_b}


def _schema_diff(a: dict, b: dict) -> dict | None:
    if a["file_type"] not in TABULAR_TYPES or b["file_type"] not in TABULAR_TYPES:
        return None
    ca = set(load_dataframe(a)[0].columns)
    cb = set(load_dataframe(b)[0].columns)
    return {"common_columns": sorted(ca & cb), "only_in_a": sorted(ca - cb), "only_in_b": sorted(cb - ca)}


def run_comparison(document_a: str, document_b: str, focus: str | None = None) -> dict:
    if not document_a or not document_b:
        raise ToolInputError("Two documents are required: 'document_a' and 'document_b'.")
    a = resolve_document(document_a)
    b = resolve_document(document_b)
    if a["id"] == b["id"]:
        raise ToolInputError("Please choose two different documents.")
    text_a = clean_text(load_text(stored_path(a["stored_name"]), a["file_type"])[0])
    text_b = clean_text(load_text(stored_path(b["stored_name"]), b["file_type"])[0])
    metrics = term_overlap(text_a, text_b)
    schema = _schema_diff(a, b)
    if schema:
        metrics["column_comparison"] = schema
    ta, info_a = condense_text(text_a, a["filename"], segment_chars=COMPARE_CHARS)
    tb, info_b = condense_text(text_b, b["filename"], segment_chars=COMPARE_CHARS)
    chain = COMPARE_PROMPT | get_llm(temperature=0.1) | StrOutputParser()
    report = invoke_llm(chain, {
        "name_a": a["filename"], "name_b": b["filename"], "text_a": ta, "text_b": tb,
        "metrics": str(metrics), "focus": f"\nFocus especially on: {focus}" if focus else "",
    })
    return {"document_a": a["filename"], "document_b": b["filename"], "metrics": metrics, "report": report.strip(),
            "coverage": {"a": info_a["coverage"], "b": info_b["coverage"]}}


class ComparatorInput(BaseModel):
    document_a: str = Field(description="First document: file name or id.")
    document_b: str = Field(description="Second document: file name or id.")
    focus: Optional[str] = Field(None, description="Optional aspect to focus on, e.g. 'revenue and headcount'.")


@tool("document_comparator", args_schema=ComparatorInput)
def document_comparator(document_a: str, document_b: str, focus: Optional[str] = None) -> str:
    """Compare TWO uploaded documents: similarities, differences, information present in only one of them,
    plus computed overlap metrics. Returns a structured comparison report."""
    return safe_tool_call("document_comparator", lambda: run_comparison(document_a, document_b, focus))
