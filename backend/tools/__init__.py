"""Tool registry. These are real LangChain tools bound to the LLM (native tool calling)."""
from __future__ import annotations

from backend.tools.calculator import calculator
from backend.tools.comparator import document_comparator
from backend.tools.csv_analyzer import csv_excel_analyzer
from backend.tools.document_manager import document_manager
from backend.tools.document_search import document_search
from backend.tools.summarizer import document_summarizer

ALL_TOOLS = [calculator, document_search, csv_excel_analyzer, document_summarizer, document_comparator, document_manager]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def tool_catalog() -> list[dict]:
    """Metadata used by the Tool Center page and the API."""
    out = []
    for t in ALL_TOOLS:
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        out.append({"name": t.name, "description": (t.description or "").strip(), "parameters": schema.get("properties", {}),
                    "required": schema.get("required", [])})
    return out
