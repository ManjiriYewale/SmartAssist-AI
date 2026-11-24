"""Upload validation, secure storage and the document ingestion pipeline.

Pipeline: validate → store → extract → clean → chunk → embed → Chroma → metadata (PostgreSQL)
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import uuid
from pathlib import Path

from pydantic import BaseModel

from backend.config.settings import ALLOWED_EXTENSIONS, get_settings
from backend.database import repository as repo
from backend.database.models import utcnow
from backend.exceptions import DocumentProcessingError, SmartAssistError, UploadValidationError
from backend.rag import vector_store
from backend.rag.chunking import split_documents
from backend.rag.document_loader import load_document

logger = logging.getLogger(__name__)

_ZIP_MAGIC = b"PK\x03\x04"


class ValidatedUpload(BaseModel):
    original_name: str
    safe_name: str
    extension: str
    size_bytes: int
    sha256: str


def sanitize_filename(name: str) -> str:
    """Strip directories (path traversal), control characters and odd symbols."""
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().strip(".")
    name = re.sub(r"[^\w.\- ()]", "_", name)
    return name[:200]


def validate_upload(filename: str, data: bytes) -> ValidatedUpload:
    s = get_settings()
    safe = sanitize_filename(filename)
    if not safe or "." not in safe:
        raise UploadValidationError("The file must have a name and an extension (PDF, DOCX, TXT, CSV or XLSX).")
    ext = safe.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(e.upper() for e in ALLOWED_EXTENSIONS))}.")
    size = len(data)
    if size == 0:
        raise UploadValidationError("The file is empty.")
    if size > s.max_upload_bytes:
        raise UploadValidationError(f"The file is too large ({size / 1_048_576:.1f} MB). Maximum is {s.max_upload_mb} MB.")
    # content sniffing – extension alone is not trusted
    if ext == "pdf" and not data[:1024].lstrip().startswith(b"%PDF"):
        raise UploadValidationError("This file does not look like a real PDF.")
    if ext in {"docx", "xlsx"} and not data.startswith(_ZIP_MAGIC):
        raise UploadValidationError(f"This file does not look like a real {ext.upper()} document.")
    if ext in {"txt", "csv"} and b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UploadValidationError(f"This file looks binary, not a {ext.upper()} text file.")
    return ValidatedUpload(original_name=filename, safe_name=safe, extension=ext, size_bytes=size,
                           sha256=hashlib.sha256(data).hexdigest())


def stored_path(stored_name: str) -> Path:
    """Resolve a stored file name, guaranteeing it stays inside the upload directory."""
    base = get_settings().upload_path.resolve()
    p = (base / stored_name).resolve()
    if base != p.parent:
        raise UploadValidationError("Invalid file path.")
    return p


def _save_bytes(stored_name: str, data: bytes) -> Path:
    base = get_settings().upload_path
    base.mkdir(parents=True, exist_ok=True)
    path = stored_path(stored_name)
    path.write_bytes(data)
    return path


def ingest_document(doc_id: int) -> dict:
    """Run the pipeline for an already-registered document. Updates status in PostgreSQL."""
    doc = repo.get_document(doc_id)
    repo.update_document(doc_id, status="processing", error_message=None)
    try:
        path = stored_path(doc["stored_name"])
        raw_docs = load_document(path, doc["file_type"])
        page_count = raw_docs[0].metadata.get("page_count") if doc["file_type"] == "pdf" else None
        chunks = split_documents(raw_docs, doc_id=doc_id, filename=doc["filename"], file_type=doc["file_type"])
        if not chunks:
            raise DocumentProcessingError("The document contains no usable text.")
        vector_store.delete_document_chunks(doc_id)  # idempotent re-processing
        n = vector_store.add_chunks(chunks, doc_id)
        s = get_settings()
        meta = {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "embedding_model": s.embedding_model
                if s.embedding_backend == "huggingface" else "hash-offline"}
        if doc["file_type"] in {"csv", "xlsx"}:
            meta["rows_indexed_cap"] = s.max_indexed_rows
        updated = repo.update_document(
            doc_id, status="processed", error_message=None, page_count=page_count, chunk_count=n,
            char_count=sum(len(c.page_content) for c in chunks), extra_metadata=meta, processed_at=utcnow(),
        )
        logger.info("Ingested doc %s (%s): %d chunks", doc_id, doc["filename"], n)
        return updated
    except SmartAssistError as exc:
        logger.warning("Ingestion failed for doc %s: %s | %s", doc_id, exc.user_message, exc.detail)
        vector_store.delete_document_chunks(doc_id)
        return repo.update_document(doc_id, status="failed", error_message=exc.user_message, chunk_count=0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected ingestion error for doc %s", doc_id)
        vector_store.delete_document_chunks(doc_id)
        return repo.update_document(doc_id, status="failed", error_message="Unexpected error while processing the document.", chunk_count=0)


def process_upload(filename: str, data: bytes) -> tuple[dict, bool]:
    """Validate, store and ingest an uploaded file.

    Returns ``(document_record, is_duplicate)``. Identical content (SHA-256) is never stored twice.
    """
    v = validate_upload(filename, data)
    existing = repo.get_document_by_hash(v.sha256)
    if existing:
        return existing, True
    stored_name = f"{uuid.uuid4().hex}.{v.extension}"
    path = _save_bytes(stored_name, data)
    try:
        record = repo.create_document(filename=v.safe_name, stored_name=stored_name, file_hash=v.sha256,
                                      file_type=v.extension, size_bytes=v.size_bytes)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return ingest_document(record["id"]), False


def delete_document(doc_id: int) -> None:
    """Remove DB record, vectors and the stored file."""
    doc = repo.get_document(doc_id)
    vector_store.delete_document_chunks(doc_id)
    try:
        stored_path(doc["stored_name"]).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete stored file for doc %s", doc_id)
    repo.delete_document_record(doc_id)
