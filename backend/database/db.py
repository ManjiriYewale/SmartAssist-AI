"""Database connection management."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.config.settings import get_settings
from backend.database.models import Base
from backend.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _build_engine(url) -> Engine:
    url_obj = make_url(url) if isinstance(url, str) else url
    kwargs: dict = {"pool_pre_ping": True}
    if url_obj.get_backend_name() == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=5, max_overflow=10, pool_recycle=1800)
        kwargs["connect_args"] = {"connect_timeout": 10}
    return create_engine(url_obj, **kwargs)


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine(get_settings().sqlalchemy_url())
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


def configure_engine(url=None) -> Engine:
    """(Re)create the engine. ``url=None`` re-reads settings. Used by tests."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = _build_engine(url or get_settings().sqlalchemy_url())
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


@contextmanager
def get_session() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback + wrap errors on failure."""
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Database error")
        raise DatabaseError(detail=str(exc)) from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (idempotent)."""
    try:
        Base.metadata.create_all(get_engine())
    except SQLAlchemyError as exc:
        logger.exception("init_db failed")
        raise DatabaseError(detail=str(exc)) from exc
    logger.info("Database initialised (%s)", get_settings().masked_database_url())


def check_connection() -> tuple[bool, str]:
    """Return (ok, human readable message). Never raises, never leaks the password."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected"
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB connection check failed: %s", exc)
        return False, "Cannot connect to the database. Check DATABASE_URL / POSTGRES_* settings and that PostgreSQL is running."


def ensure_database_exists() -> bool:
    """For PostgreSQL: create the target database if it does not exist yet.

    Connects to the maintenance database ``postgres`` with the same credentials.
    Returns True if the database was created, False if it already existed.
    """
    raw = get_settings().sqlalchemy_url()
    url = make_url(raw) if isinstance(raw, str) else raw
    if url.get_backend_name() != "postgresql":
        return False
    target = url.database
    admin_engine = create_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 10}
    )
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target}).scalar()
            if exists:
                return False
            safe = target.replace('"', '""')
            conn.execute(text(f'CREATE DATABASE "{safe}"'))
            logger.info("Created PostgreSQL database %s", target)
            return True
    except SQLAlchemyError as exc:
        raise DatabaseError(detail=str(exc)) from exc
    finally:
        admin_engine.dispose()
