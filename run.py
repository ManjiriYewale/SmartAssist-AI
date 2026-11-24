"""SmartAssist AI launcher.

    python run.py setup      create folders, create the PostgreSQL database if needed, create tables, init ChromaDB
    python run.py check      diagnose configuration (DB, LLM key, embeddings)
    python run.py ui         start the Streamlit frontend           http://localhost:8501
    python run.py api        start the FastAPI backend              http://localhost:8000/docs
    python run.py all        start both (default)
    python run.py test       run the test suite
    python run.py eval       run the evaluation harness (add --live for LLM metrics)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def cmd_setup(create_db: bool = True) -> int:
    from backend.config.settings import get_settings, setup_logging
    from backend.database import db
    from backend.exceptions import SmartAssistError

    setup_logging()
    s = get_settings()
    print("SmartAssist AI - setup")
    s.upload_path.mkdir(parents=True, exist_ok=True)
    s.chroma_path.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    _ok(f"folders ready ({s.upload_path.name}, {s.chroma_path.name}, logs)")

    if not (ROOT / ".env").exists():
        _warn(".env not found - copy .env.example to .env and fill in your values")
    try:
        if create_db and s.database_backend == "postgresql":
            created = db.ensure_database_exists()
            _ok(f"PostgreSQL database '{s.postgres_db}' " + ("created" if created else "already exists"))
        db.init_db()
        ok, msg = db.check_connection()
        (_ok if ok else _fail)(f"database tables ready at {s.masked_database_url()}" if ok else msg)
        if not ok:
            return 1
    except SmartAssistError as exc:
        _fail(exc.user_message)
        if exc.detail:
            print("         detail:", exc.detail.splitlines()[0][:200])
        return 1

    try:
        from backend.rag import vector_store
        print("  ...  loading embedding model (first run downloads ~90 MB)")
        vector_store.get_vector_store()._embedding_function.embed_query("warm-up")
        _ok(f"ChromaDB ready ({vector_store.collection_count()} chunks) with {s.embedding_backend}:{s.embedding_model}")
    except SmartAssistError as exc:
        _fail(exc.user_message)
        return 1
    except Exception as exc:  # noqa: BLE001
        _fail(f"ChromaDB / embeddings: {str(exc)[:200]}")
        return 1

    (_ok if s.llm_configured else _warn)("GROQ_API_KEY is set" if s.llm_configured else "GROQ_API_KEY is empty - the assistant needs it (documents and tools still work)")
    print("\nSetup finished.  Next:  python run.py all")
    return 0


def cmd_check() -> int:
    from backend.config.settings import get_settings
    from backend.database.db import check_connection
    from backend.llm.model import check_llm_connection

    s = get_settings()
    print("Configuration check")
    ok, msg = check_connection()
    (_ok if ok else _fail)(f"database: {msg} ({s.masked_database_url()})")
    _ok(f"model: {s.groq_model}") if s.llm_configured else _warn("GROQ_API_KEY missing")
    if s.llm_configured:
        ok2, msg2 = check_llm_connection()
        (_ok if ok2 else _fail)(f"LLM: {msg2}")
    return 0 if ok else 1


def _spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, *args], cwd=ROOT)


def cmd_serve(api: bool, ui: bool) -> int:
    from backend.config.settings import get_settings

    s = get_settings()
    rc = cmd_setup()
    if rc:
        return rc
    procs: list[subprocess.Popen] = []
    try:
        if api:
            procs.append(_spawn(["-m", "uvicorn", "backend.main:app", "--host", s.api_host, "--port", str(s.api_port)]))
            print(f"API docs:   http://{s.api_host}:{s.api_port}/docs")
        if ui:
            procs.append(_spawn(["-m", "streamlit", "run", "frontend/app.py", "--browser.gatherUsageStats", "false"]))
            print("Frontend:   http://localhost:8501")
        print("Press Ctrl+C to stop.")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        return next((p.returncode for p in procs if p.returncode), 0)
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


def main() -> int:
    ap = argparse.ArgumentParser(description="SmartAssist AI launcher")
    ap.add_argument("command", nargs="?", default="all", choices=["setup", "check", "ui", "api", "all", "test", "eval"])
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="extra args for test/eval")
    a = ap.parse_args()
    if a.command == "setup":
        return cmd_setup()
    if a.command == "check":
        return cmd_check()
    if a.command == "ui":
        return cmd_serve(api=False, ui=True)
    if a.command == "api":
        return cmd_serve(api=True, ui=False)
    if a.command == "test":
        return subprocess.call([sys.executable, "-m", "pytest", *a.rest], cwd=ROOT)
    if a.command == "eval":
        return subprocess.call([sys.executable, "evaluation/evaluate.py", *a.rest], cwd=ROOT)
    return cmd_serve(api=True, ui=True)


if __name__ == "__main__":
    raise SystemExit(main())
