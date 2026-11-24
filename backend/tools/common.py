"""Shared helpers for tools: safe execution, JSON serialisation, document resolution."""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

from backend.database import repository as repo
from backend.exceptions import NotFoundError, SmartAssistError, ToolInputError

logger = logging.getLogger(__name__)


def py(value: Any) -> Any:
    """Convert numpy / pandas / Decimal values into JSON-safe Python objects."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # pragma: no cover
        np = pd = None  # type: ignore
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [py(v) for v in value]
    if pd is not None and isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def to_json(obj: Any) -> str:
    return json.dumps(py(obj), ensure_ascii=False, default=str)


def safe_tool_call(name: str, fn: Callable[[], dict]) -> str:
    """Run a tool body; always return a JSON string with ``status`` ok|error.

    Expected errors (bad input, missing document, LLM trouble) become readable
    messages the agent can react to; unexpected ones are logged and masked.
    """
    try:
        result = fn()
        return to_json({"status": "ok", **result})
    except SmartAssistError as exc:
        logger.info("Tool %s rejected request: %s", name, exc.user_message)
        return to_json({"status": "error", "error": exc.user_message})
    except Exception:  # noqa: BLE001
        logger.exception("Tool %s crashed", name)
        return to_json({"status": "error", "error": f"The {name} tool failed unexpectedly."})


def resolve_document(reference: str | None, *, file_types: set[str] | None = None,
                     only_processed: bool = True, what: str = "document") -> dict:
    """Resolve a document by id / (partial) file name. Gives helpful errors listing choices."""
    candidates = repo.list_documents()
    pool = [d for d in candidates if (not file_types or d["file_type"] in file_types)
            and (not only_processed or d["status"] == "processed")]
    if not reference or not str(reference).strip():
        if len(pool) == 1:
            return pool[0]
        if not pool:
            raise NotFoundError(f"No {what}s are available. Upload and process one first.")
        names = ", ".join(d["filename"] for d in pool[:10])
        raise ToolInputError(f"Please specify which {what} to use. Available: {names}")
    hits = repo.find_documents(str(reference), file_types=file_types, only_processed=only_processed)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        names = ", ".join(d["filename"] for d in pool[:10]) or "none"
        raise NotFoundError(f"No {what} matches '{reference}'. Available: {names}")
    names = ", ".join(d["filename"] for d in hits[:10])
    raise ToolInputError(f"'{reference}' matches several documents: {names}. Please be more specific.")


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + " …[truncated]"
