"""PostgreSQL-specific tests. Run with:  RUN_PG_TESTS=1 POSTGRES_PASSWORD=... pytest tests/test_postgres.py
(skipped automatically when no PostgreSQL is configured)."""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_PG_TESTS"), reason="set RUN_PG_TESTS=1 to run PostgreSQL tests")


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_DB", "smartassist_pytest")
    from backend.config.settings import reload_settings
    from backend.database import db
    reload_settings()
    db.configure_engine()
    yield db
    db.get_engine().dispose()
    from sqlalchemy import create_engine, text
    admin = create_engine(reload_settings().sqlalchemy_url().set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text("DROP DATABASE IF EXISTS smartassist_pytest WITH (FORCE)"))
    admin.dispose()


def test_database_is_created_and_initialised(pg):
    created = pg.ensure_database_exists()
    pg.configure_engine()
    assert pg.check_connection() == (True, "Connected")
    pg.init_db()
    from sqlalchemy import inspect
    assert {"documents", "messages", "tool_executions"} <= set(inspect(pg.get_engine()).get_table_names())
    assert pg.ensure_database_exists() is False            # second call: already exists
    assert created in (True, False)


def test_repository_roundtrip_on_postgres(pg):
    pg.ensure_database_exists(); pg.configure_engine(); pg.init_db()
    from backend.database import repository as repo
    d = repo.create_document(filename="a.txt", stored_name="s.txt", file_hash="h", file_type="txt", size_bytes=1)
    c = repo.create_conversation("t")
    repo.add_message(c["id"], "assistant", "hi", sources=[{"a": 1}], tool_calls=[{"name": "x"}])
    repo.log_tool_execution(tool_name="calculator", input_args={"a": 1}, output="o", status="error", duration_ms=5)
    assert repo.get_messages(c["id"])[0]["sources"] == [{"a": 1}]
    assert repo.get_tool_stats()[0]["errors"] == 1
    assert repo.get_dashboard_stats()["total_documents"] == 1
    from backend.exceptions import DatabaseError
    with pytest.raises(DatabaseError):
        repo.create_document(filename="b.txt", stored_name="s2.txt", file_hash="h", file_type="txt", size_bytes=1)
    repo.delete_conversation(c["id"])
    assert repo.get_document(d["id"])["filename"] == "a.txt"
