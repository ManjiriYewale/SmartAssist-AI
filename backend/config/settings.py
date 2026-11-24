"""Central, environment-driven configuration (pydantic-settings).

Nothing secret is hard-coded: API keys and database passwords come from the
process environment or from the ``.env`` file in the project root.
"""
from __future__ import annotations

import logging
import logging.handlers
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "xlsx"}
TABULAR_TYPES = {"csv", "xlsx"}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM (Groq) -------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_timeout_seconds: int = Field(60, ge=5, le=600)
    llm_max_retries: int = Field(2, ge=0, le=6)

    # --- PostgreSQL -------------------------------------------------------
    database_url: str = ""  # full SQLAlchemy URL; overrides the POSTGRES_* parts
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "smartassist"

    # --- Storage ----------------------------------------------------------
    upload_dir: str = "data/uploads"
    chroma_dir: str = "data/chroma_db"
    chroma_collection: str = "smartassist_documents"
    max_upload_mb: int = Field(25, ge=1, le=500)

    # --- RAG --------------------------------------------------------------
    embedding_backend: Literal["huggingface", "hash"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    top_k: int = Field(5, ge=1, le=20)
    min_relevance_score: float = Field(0.20, ge=0.0, le=1.0)
    max_indexed_rows: int = Field(5000, ge=100)

    # --- Agent ------------------------------------------------------------
    max_agent_steps: int = Field(4, ge=1, le=10)
    max_tool_calls_per_step: int = Field(4, ge=1, le=10)
    max_history_messages: int = Field(8, ge=0, le=40)

    # --- API / misc -------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_auth_token: str = ""  # if set, API requests must send "X-API-Key: <token>"
    log_level: str = "INFO"

    @field_validator("groq_api_key", "api_auth_token", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # ---- derived values --------------------------------------------------
    @property
    def upload_path(self) -> Path:
        return _resolve(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return _resolve(self.chroma_dir)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key)

    def sqlalchemy_url(self) -> str | URL:
        """URL used by SQLAlchemy. The password never appears in source code."""
        if self.database_url:
            return self.database_url
        return URL.create(
            "postgresql+psycopg2",
            username=self.postgres_user,
            password=self.postgres_password or None,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    def masked_database_url(self) -> str:
        url = self.sqlalchemy_url()
        if isinstance(url, str):
            from sqlalchemy.engine import make_url

            url = make_url(url)
        return url.render_as_string(hide_password=True)

    @property
    def database_backend(self) -> str:
        url = self.sqlalchemy_url()
        if isinstance(url, str):
            return url.split(":", 1)[0].split("+", 1)[0]
        return url.get_backend_name()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Re-read the environment (used by tests and the Settings page)."""
    get_settings.cache_clear()
    return get_settings()


_LOGGING_READY = False


def setup_logging() -> None:
    """Console + rotating file logging. Safe to call many times."""
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    s = get_settings()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, s.log_level.upper(), logging.INFO))
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "smartassist.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:  # read-only file systems etc.
        pass
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _LOGGING_READY = True
