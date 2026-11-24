"""CRUD operations and aggregate queries. All functions return plain dicts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func, select

from backend.database.db import get_session
from backend.database.models import Conversation, Document, Message, QueryHistory, ToolExecution, utcnow
from backend.exceptions import DatabaseError, NotFoundError


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


# ───────────────────────────── documents ─────────────────────────────
def _doc(d: Document) -> dict:
    return {
        "id": d.id,
        "filename": d.filename,
        "stored_name": d.stored_name,
        "file_hash": d.file_hash,
        "file_type": d.file_type,
        "size_bytes": d.size_bytes,
        "status": d.status,
        "error_message": d.error_message,
        "page_count": d.page_count,
        "chunk_count": d.chunk_count,
        "char_count": d.char_count,
        "extra_metadata": d.extra_metadata or {},
        "created_at": _iso(d.created_at),
        "processed_at": _iso(d.processed_at),
    }


def create_document(*, filename: str, stored_name: str, file_hash: str, file_type: str, size_bytes: int) -> dict:
    try:
        with get_session() as s:
            d = Document(
                filename=filename, stored_name=stored_name, file_hash=file_hash,
                file_type=file_type, size_bytes=size_bytes, status="uploaded",
            )
            s.add(d)
            s.flush()
            return _doc(d)
    except DatabaseError as exc:
        if exc.detail and ("UNIQUE" in exc.detail.upper() or "duplicate key" in exc.detail.lower()):
            raise DatabaseError("A document with identical content already exists.") from exc
        raise


def get_document(doc_id: int) -> dict:
    with get_session() as s:
        d = s.get(Document, doc_id)
        if not d:
            raise NotFoundError(f"Document #{doc_id} was not found.")
        return _doc(d)


def get_document_by_hash(file_hash: str) -> dict | None:
    with get_session() as s:
        d = s.scalar(select(Document).where(Document.file_hash == file_hash))
        return _doc(d) if d else None


def list_documents(search: str | None = None, status: str | None = None, file_type: str | None = None) -> list[dict]:
    with get_session() as s:
        q = select(Document).order_by(Document.created_at.desc(), Document.id.desc())
        if search:
            q = q.where(Document.filename.ilike(f"%{search}%"))
        if status:
            q = q.where(Document.status == status)
        if file_type:
            q = q.where(Document.file_type == file_type)
        return [_doc(d) for d in s.scalars(q)]


def update_document(doc_id: int, **fields: Any) -> dict:
    allowed = {"status", "error_message", "page_count", "chunk_count", "char_count", "extra_metadata", "processed_at"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"Cannot update fields: {sorted(bad)}")
    with get_session() as s:
        d = s.get(Document, doc_id)
        if not d:
            raise NotFoundError(f"Document #{doc_id} was not found.")
        for k, v in fields.items():
            setattr(d, k, v)
        s.flush()
        return _doc(d)


def delete_document_record(doc_id: int) -> None:
    with get_session() as s:
        d = s.get(Document, doc_id)
        if not d:
            raise NotFoundError(f"Document #{doc_id} was not found.")
        s.delete(d)


def find_documents(reference: str, *, file_types: set[str] | None = None, only_processed: bool = False) -> list[dict]:
    """Resolve a user/LLM supplied reference (id, exact name, partial name)."""
    ref = (reference or "").strip().strip('"\'')
    if not ref:
        return []
    docs = list_documents()
    if file_types:
        docs = [d for d in docs if d["file_type"] in file_types]
    if only_processed:
        docs = [d for d in docs if d["status"] == "processed"]
    if ref.lstrip("#").isdigit():
        hit = [d for d in docs if d["id"] == int(ref.lstrip("#"))]
        if hit:
            return hit
    low = ref.lower()
    exact = [d for d in docs if d["filename"].lower() == low]
    if exact:
        return exact
    stem = [d for d in docs if d["filename"].lower().rsplit(".", 1)[0] == low]
    if stem:
        return stem
    return [d for d in docs if low in d["filename"].lower()]


# ───────────────────────── conversations / messages ─────────────────────────
def _conv(c: Conversation, n_messages: int | None = None) -> dict:
    out = {"id": c.id, "title": c.title, "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at)}
    if n_messages is not None:
        out["message_count"] = n_messages
    return out


def create_conversation(title: str = "New conversation") -> dict:
    with get_session() as s:
        c = Conversation(title=(title or "New conversation")[:200])
        s.add(c)
        s.flush()
        return _conv(c, 0)


def get_conversation(conv_id: int) -> dict:
    with get_session() as s:
        c = s.get(Conversation, conv_id)
        if not c:
            raise NotFoundError(f"Conversation #{conv_id} was not found.")
        n = s.scalar(select(func.count(Message.id)).where(Message.conversation_id == conv_id)) or 0
        return _conv(c, n)


def list_conversations(limit: int = 50) -> list[dict]:
    with get_session() as s:
        counts = dict(s.execute(select(Message.conversation_id, func.count(Message.id)).group_by(Message.conversation_id)).all())
        rows = s.scalars(select(Conversation).order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit))
        return [_conv(c, counts.get(c.id, 0)) for c in rows]


def rename_conversation(conv_id: int, title: str) -> dict:
    with get_session() as s:
        c = s.get(Conversation, conv_id)
        if not c:
            raise NotFoundError(f"Conversation #{conv_id} was not found.")
        c.title = title[:200]
        s.flush()
        return _conv(c)


def delete_conversation(conv_id: int) -> None:
    with get_session() as s:
        c = s.get(Conversation, conv_id)
        if not c:
            raise NotFoundError(f"Conversation #{conv_id} was not found.")
        s.delete(c)


def add_message(conversation_id: int, role: str, content: str, *, sources=None, tool_calls=None, intent=None) -> dict:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    with get_session() as s:
        c = s.get(Conversation, conversation_id)
        if not c:
            raise NotFoundError(f"Conversation #{conversation_id} was not found.")
        m = Message(conversation_id=conversation_id, role=role, content=content,
                    sources=sources, tool_calls=tool_calls, intent=intent)
        s.add(m)
        c.updated_at = utcnow()
        s.flush()
        return _msg(m)


def _msg(m: Message) -> dict:
    return {
        "id": m.id, "conversation_id": m.conversation_id, "role": m.role, "content": m.content,
        "sources": m.sources or [], "tool_calls": m.tool_calls or [], "intent": m.intent,
        "created_at": _iso(m.created_at),
    }


def get_messages(conversation_id: int, limit: int | None = None) -> list[dict]:
    """Oldest → newest. With ``limit`` returns the most recent N messages."""
    with get_session() as s:
        q = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id.desc())
        if limit:
            q = q.limit(limit)
        rows = list(s.scalars(q))
        rows.reverse()
        return [_msg(m) for m in rows]


def clear_messages(conversation_id: int) -> int:
    with get_session() as s:
        rows = list(s.scalars(select(Message).where(Message.conversation_id == conversation_id)))
        for m in rows:
            s.delete(m)
        return len(rows)


# ───────────────────────────── logs ─────────────────────────────
def log_tool_execution(*, tool_name: str, input_args: dict | None, output: str | None, status: str,
                       error_message: str | None = None, duration_ms: int = 0,
                       conversation_id: int | None = None, source: str = "agent") -> dict:
    with get_session() as s:
        t = ToolExecution(
            conversation_id=conversation_id, tool_name=tool_name, input_args=input_args,
            output=(output or "")[:20000], status=status, error_message=error_message,
            duration_ms=int(duration_ms), source=source,
        )
        s.add(t)
        s.flush()
        return _tool(t)


def _tool(t: ToolExecution) -> dict:
    return {
        "id": t.id, "conversation_id": t.conversation_id, "tool_name": t.tool_name,
        "input_args": t.input_args or {}, "output": t.output, "status": t.status,
        "error_message": t.error_message, "duration_ms": t.duration_ms, "source": t.source,
        "created_at": _iso(t.created_at),
    }


def list_tool_executions(limit: int = 100, tool_name: str | None = None, conversation_id: int | None = None) -> list[dict]:
    with get_session() as s:
        q = select(ToolExecution).order_by(ToolExecution.id.desc()).limit(limit)
        if tool_name:
            q = q.where(ToolExecution.tool_name == tool_name)
        if conversation_id is not None:
            q = q.where(ToolExecution.conversation_id == conversation_id)
        return [_tool(t) for t in s.scalars(q)]


def log_query(*, query: str, rewritten_query: str | None, intent: str | None, selected_action: str | None,
              tools_used: list[str] | None, num_sources: int, status: str, latency_ms: int,
              error_message: str | None = None, conversation_id: int | None = None) -> dict:
    with get_session() as s:
        q = QueryHistory(
            conversation_id=conversation_id, query=query, rewritten_query=rewritten_query, intent=intent,
            selected_action=selected_action, tools_used=tools_used or [], num_sources=num_sources,
            status=status, error_message=error_message, latency_ms=int(latency_ms),
        )
        s.add(q)
        s.flush()
        return _query(q)


def _query(q: QueryHistory) -> dict:
    return {
        "id": q.id, "conversation_id": q.conversation_id, "query": q.query,
        "rewritten_query": q.rewritten_query, "intent": q.intent, "selected_action": q.selected_action,
        "tools_used": q.tools_used or [], "num_sources": q.num_sources, "status": q.status,
        "error_message": q.error_message, "latency_ms": q.latency_ms, "created_at": _iso(q.created_at),
    }


def list_queries(limit: int = 100) -> list[dict]:
    with get_session() as s:
        return [_query(q) for q in s.scalars(select(QueryHistory).order_by(QueryHistory.id.desc()).limit(limit))]


# ───────────────────────────── statistics ─────────────────────────────
def get_dashboard_stats() -> dict:
    with get_session() as s:
        total_docs = s.scalar(select(func.count(Document.id))) or 0
        processed = s.scalar(select(func.count(Document.id)).where(Document.status == "processed")) or 0
        failed = s.scalar(select(func.count(Document.id)).where(Document.status == "failed")) or 0
        chunks = s.scalar(select(func.coalesce(func.sum(Document.chunk_count), 0))) or 0
        queries = s.scalar(select(func.count(QueryHistory.id))) or 0
        ok_queries = s.scalar(select(func.count(QueryHistory.id)).where(QueryHistory.status == "completed")) or 0
        avg_latency = s.scalar(select(func.coalesce(func.avg(QueryHistory.latency_ms), 0))) or 0
        tool_calls = s.scalar(select(func.count(ToolExecution.id))) or 0
        convs = s.scalar(select(func.count(Conversation.id))) or 0
        msgs = s.scalar(select(func.count(Message.id))) or 0
    return {
        "total_documents": total_docs, "processed_documents": processed, "failed_documents": failed,
        "total_chunks": int(chunks), "total_queries": queries, "successful_queries": ok_queries,
        "avg_latency_ms": int(avg_latency), "total_tool_executions": tool_calls,
        "total_conversations": convs, "total_messages": msgs,
    }


def get_tool_stats() -> list[dict]:
    with get_session() as s:
        rows = s.execute(
            select(
                ToolExecution.tool_name,
                func.count(ToolExecution.id),
                func.sum(case((ToolExecution.status == "error", 1), else_=0)),
                func.avg(ToolExecution.duration_ms),
            ).group_by(ToolExecution.tool_name)
        ).all()
    out = [
        {"tool": r[0], "calls": int(r[1]), "errors": int(r[2] or 0), "avg_ms": int(r[3] or 0)}
        for r in rows
    ]
    return sorted(out, key=lambda x: -x["calls"])


def get_document_status_counts() -> dict[str, int]:
    with get_session() as s:
        return {k: v for k, v in s.execute(select(Document.status, func.count(Document.id)).group_by(Document.status)).all()}


def get_file_type_counts() -> dict[str, int]:
    with get_session() as s:
        return {k: v for k, v in s.execute(select(Document.file_type, func.count(Document.id)).group_by(Document.file_type)).all()}


def get_intent_distribution() -> dict[str, int]:
    with get_session() as s:
        rows = s.execute(select(QueryHistory.intent, func.count(QueryHistory.id)).group_by(QueryHistory.intent)).all()
    return {(k or "unknown"): v for k, v in rows}


def get_queries_per_day(days: int = 14) -> list[dict]:
    since = utcnow() - timedelta(days=days)
    with get_session() as s:
        stamps = [r for r in s.scalars(select(QueryHistory.created_at).where(QueryHistory.created_at >= since))]
    counter: Counter = Counter()
    for ts in stamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        counter[ts.date().isoformat()] += 1
    today = utcnow().date()
    return [
        {"date": (today - timedelta(days=i)).isoformat(), "queries": counter.get((today - timedelta(days=i)).isoformat(), 0)}
        for i in range(days - 1, -1, -1)
    ]


def get_recent_activity(limit: int = 12) -> list[dict]:
    events: list[dict] = []
    with get_session() as s:
        for d in s.scalars(select(Document).order_by(Document.id.desc()).limit(limit)):
            events.append({"time": _iso(d.created_at), "type": "document",
                           "text": f"Uploaded {d.filename} ({d.status})"})
        for q in s.scalars(select(QueryHistory).order_by(QueryHistory.id.desc()).limit(limit)):
            events.append({"time": _iso(q.created_at), "type": "query",
                           "text": f"Query: {q.query[:80]}" + (" …" if len(q.query) > 80 else "")})
        for t in s.scalars(select(ToolExecution).order_by(ToolExecution.id.desc()).limit(limit)):
            events.append({"time": _iso(t.created_at), "type": "tool",
                           "text": f"Tool {t.tool_name} → {t.status} ({t.duration_ms} ms)"})
    events.sort(key=lambda e: e["time"] or "", reverse=True)
    return events[:limit]
