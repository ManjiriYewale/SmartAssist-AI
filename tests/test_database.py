import pytest

from backend.database import db, repository as repo
from backend.exceptions import DatabaseError, NotFoundError


def mk(name="a.txt", h="h1", **kw):
    return repo.create_document(filename=name, stored_name=f"s-{h}.txt", file_hash=h, file_type="txt", size_bytes=10, **kw)


def test_connection_and_tables():
    ok, msg = db.check_connection()
    assert ok and msg == "Connected"
    from sqlalchemy import inspect
    assert {"documents", "conversations", "messages", "tool_executions", "query_history"} <= set(inspect(db.get_engine()).get_table_names())


def test_connection_failure_hides_credentials():
    db.configure_engine("postgresql+psycopg2://user:SECRETPW@127.0.0.1:1/none")
    ok, msg = db.check_connection()
    assert not ok and "SECRETPW" not in msg
    db.configure_engine()


def test_document_crud():
    d = mk()
    assert repo.get_document(d["id"])["status"] == "uploaded"
    repo.update_document(d["id"], status="processed", chunk_count=4)
    assert repo.list_documents(status="processed")[0]["chunk_count"] == 4
    assert repo.list_documents(search="A.TX")[0]["id"] == d["id"]       # case-insensitive search
    assert repo.get_document_by_hash("h1")["id"] == d["id"]
    repo.delete_document_record(d["id"])
    with pytest.raises(NotFoundError):
        repo.get_document(d["id"])


def test_unique_hash_enforced_and_bad_update_rejected():
    d = mk()
    with pytest.raises(DatabaseError):
        mk("b.txt", h="h1")
    with pytest.raises(ValueError):
        repo.update_document(d["id"], file_hash="x")


def test_find_documents_by_reference():
    a, b = mk("Report 2023.pdf", "x1"), mk("Report 2024.pdf", "x2")
    assert [d["id"] for d in repo.find_documents(str(a["id"]))] == [a["id"]]
    assert len(repo.find_documents("report")) == 2
    assert len(repo.find_documents("report 2024")) == 1
    assert repo.find_documents("nothing") == []


def test_conversation_and_messages():
    c = repo.create_conversation("Hello")
    repo.add_message(c["id"], "user", "hi")
    repo.add_message(c["id"], "assistant", "hello", sources=[{"filename": "a"}], tool_calls=[{"name": "calculator"}], intent="general_chat")
    msgs = repo.get_messages(c["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"] == [{"filename": "a"}] and msgs[1]["intent"] == "general_chat"
    assert [m["content"] for m in repo.get_messages(c["id"], limit=1)] == ["hello"]
    assert repo.list_conversations()[0]["message_count"] == 2
    assert repo.clear_messages(c["id"]) == 2 and repo.get_messages(c["id"]) == []
    with pytest.raises(ValueError):
        repo.add_message(c["id"], "system", "x")
    with pytest.raises(NotFoundError):
        repo.add_message(999, "user", "x")


def test_conversation_delete_cascades_messages():
    c = repo.create_conversation()
    repo.add_message(c["id"], "user", "hi")
    repo.delete_conversation(c["id"])
    assert repo.get_messages(c["id"]) == []


def test_tool_execution_logging_and_stats():
    repo.log_tool_execution(tool_name="calculator", input_args={"a": 1}, output="{}", status="success", duration_ms=10)
    repo.log_tool_execution(tool_name="calculator", input_args={}, output="{}", status="error", error_message="bad", duration_ms=30)
    repo.log_tool_execution(tool_name="document_search", input_args={}, output="{}", status="success", duration_ms=5, source="manual")
    stats = {s["tool"]: s for s in repo.get_tool_stats()}
    assert stats["calculator"] == {"tool": "calculator", "calls": 2, "errors": 1, "avg_ms": 20}
    assert repo.list_tool_executions(tool_name="document_search")[0]["source"] == "manual"


def test_query_history_and_dashboard_stats():
    d = mk(); repo.update_document(d["id"], status="processed", chunk_count=7)
    repo.log_query(query="q1", rewritten_query="q1", intent="document_qa", selected_action="rag", tools_used=["document_search"],
                   num_sources=2, status="completed", latency_ms=100)
    repo.log_query(query="q2", rewritten_query="q2", intent="general_chat", selected_action="general", tools_used=[],
                   num_sources=0, status="error", latency_ms=300, error_message="x")
    s = repo.get_dashboard_stats()
    assert (s["total_documents"], s["processed_documents"], s["total_chunks"]) == (1, 1, 7)
    assert (s["total_queries"], s["successful_queries"], s["avg_latency_ms"]) == (2, 1, 200)
    assert repo.get_intent_distribution() == {"document_qa": 1, "general_chat": 1}
    assert sum(r["queries"] for r in repo.get_queries_per_day(7)) == 2
    assert repo.get_document_status_counts() == {"processed": 1}
    assert {e["type"] for e in repo.get_recent_activity()} == {"document", "query"}


def test_session_rolls_back_on_error():
    with pytest.raises(RuntimeError):
        with db.get_session() as s:
            from backend.database.models import Conversation
            s.add(Conversation(title="x")); s.flush()
            raise RuntimeError("boom")
    assert repo.list_conversations() == []
