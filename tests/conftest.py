"""Shared fixtures: an isolated SQLite DB, Chroma dir and upload dir per test, offline hash embeddings."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "sample_data"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    import os
    # Set TEST_DATABASE_URL to run the whole suite against PostgreSQL instead of SQLite.
    monkeypatch.setenv("DATABASE_URL", os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("API_AUTH_TOKEN", "")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.15")
    from backend.config.settings import reload_settings
    from backend.database import db
    from backend.llm.model import set_llm_override
    from backend.rag import embeddings, vector_store

    reload_settings()
    embeddings.clear_embedding_cache()
    vector_store.reset_vector_store()
    db.configure_engine()
    from backend.database.models import Base
    Base.metadata.drop_all(db.get_engine())      # clean slate (matters for a shared PostgreSQL DB)
    db.init_db()
    set_llm_override(None)
    yield
    set_llm_override(None)
    vector_store.reset_vector_store()
    db._engine.dispose()


@pytest.fixture
def samples():
    return SAMPLES


@pytest.fixture
def upload():
    """Upload a sample file through the real ingestion pipeline."""
    from backend.rag.ingestion import process_upload

    def _up(name: str) -> dict:
        rec, _dup = process_upload(name, (SAMPLES / name).read_bytes())
        assert rec["status"] == "processed", rec
        return rec

    return _up


@pytest.fixture
def all_docs(upload):
    return {n: upload(n) for n in ["employee_handbook.pdf", "annual_report_2023.docx", "annual_report_2024.docx",
                                    "sales_data.csv", "expenses_q1.xlsx", "project_notes.txt"]}
