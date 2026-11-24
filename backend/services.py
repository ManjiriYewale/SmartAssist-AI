"""Service layer used by BOTH the FastAPI routes and the Streamlit frontend."""
from __future__ import annotations

import json
import logging
import time

from backend.agents.graph import run_agent
from backend.config.settings import get_settings, setup_logging
from backend.database import repository as repo
from backend.database.db import init_db
from backend.exceptions import SmartAssistError, ToolInputError
from backend.rag import ingestion
from backend.rag.retriever import retrieve
from backend.tools import TOOLS_BY_NAME, tool_catalog

logger = logging.getLogger(__name__)


def init_app() -> None:
    """Idempotent start-up: logging, folders, tables."""
    setup_logging()
    s = get_settings()
    s.upload_path.mkdir(parents=True, exist_ok=True)
    s.chroma_path.mkdir(parents=True, exist_ok=True)
    init_db()


# ───────────── documents ─────────────
def upload_document(filename: str, data: bytes) -> dict:
    record, duplicate = ingestion.process_upload(filename, data)
    return {**record, "duplicate": duplicate}


def reprocess_document(doc_id: int) -> dict:
    return ingestion.ingest_document(doc_id)


def delete_document(doc_id: int) -> None:
    ingestion.delete_document(doc_id)


# ───────────── chat ─────────────
def ask(message: str, conversation_id: int | None = None) -> dict:
    """Run the agent for one message, persisting messages, tool logs and query history."""
    message = (message or "").strip()
    if not message:
        raise ToolInputError("Please enter a question.")
    if conversation_id is None:
        conversation_id = repo.create_conversation(message[:60])["id"]
    else:
        repo.get_conversation(conversation_id)  # raises NotFoundError
    history = [{"role": m["role"], "content": m["content"]} for m in
               repo.get_messages(conversation_id, limit=get_settings().max_history_messages)]
    first_message = not history
    repo.add_message(conversation_id, "user", message)

    start = time.perf_counter()
    state = run_agent(message, history=history, conversation_id=conversation_id)
    latency = int((time.perf_counter() - start) * 1000)

    answer = state.get("final_response") or "No answer was produced."
    tools = [{"name": o["tool"], "args": o["args"], "status": o["status"], "duration_ms": o["duration_ms"],
              "output": o["output"][:2000]} for o in state.get("tool_outputs", [])]
    assistant = repo.add_message(conversation_id, "assistant", answer, sources=state.get("sources", []),
                                 tool_calls=tools, intent=state.get("intent"))
    try:
        repo.log_query(query=message, rewritten_query=state.get("rewritten_query"), intent=state.get("intent"),
                       selected_action=state.get("selected_action"), tools_used=[t["name"] for t in tools],
                       num_sources=len(state.get("sources", [])), status=state.get("status", "completed"),
                       latency_ms=latency, error_message=state.get("error"), conversation_id=conversation_id)
        if first_message:
            repo.rename_conversation(conversation_id, message[:60])
    except SmartAssistError:
        logger.warning("Could not record query history")
    return {"conversation_id": conversation_id, "message": assistant, "answer": answer, "intent": state.get("intent"),
            "action": state.get("selected_action"), "status": state.get("status"), "sources": state.get("sources", []),
            "tools": tools, "rewritten_query": state.get("rewritten_query"), "trace": state.get("trace", []),
            "warnings": state.get("warnings", []), "latency_ms": latency}


# ───────────── retrieval explorer ─────────────
def explore_retrieval(query: str, *, top_k: int, doc_id: int | None, file_type: str | None, min_score: float) -> dict:
    res = retrieve(query, top_k=top_k, doc_ids=[doc_id] if doc_id else None, file_type=file_type, min_score=min_score)
    return res.model_dump()


# ───────────── tools ─────────────
def run_tool_manually(name: str, args: dict) -> dict:
    """Run one tool directly (Tool Center). Arguments are validated by the tool's pydantic schema."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise ToolInputError(f"Unknown tool '{name}'.")
    if not isinstance(args, dict):
        raise ToolInputError("Tool arguments must be a JSON object.")
    start = time.perf_counter()
    try:
        result = tool.invoke({"name": name, "args": args, "id": "manual", "type": "tool_call"})
        content = result.content if isinstance(result.content, str) else json.dumps(result.content, default=str)
        status, err = "success", None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("status") == "error":
                status, err = "error", parsed.get("error")
        except json.JSONDecodeError:
            parsed = content
    except Exception as exc:  # noqa: BLE001 - schema validation errors
        content = json.dumps({"status": "error", "error": "Invalid arguments for this tool."})
        parsed, status, err = json.loads(content), "error", str(exc)[:300]
    duration = int((time.perf_counter() - start) * 1000)
    repo.log_tool_execution(tool_name=name, input_args=args, output=content, status=status, error_message=err,
                            duration_ms=duration, source="manual")
    return {"tool": name, "status": status, "duration_ms": duration, "output": parsed}


__all__ = ["init_app", "upload_document", "reprocess_document", "delete_document", "ask", "explore_retrieval",
           "run_tool_manually", "tool_catalog"]
