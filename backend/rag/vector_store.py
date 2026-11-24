"""ChromaDB persistence layer (via ``langchain-chroma``)."""
from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.exceptions import DocumentProcessingError
from backend.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_stores: dict[tuple, Any] = {}
_BATCH = 128


def get_vector_store():
    """Return a cached persistent Chroma store for the configured directory/collection."""
    s = get_settings()
    key = (str(s.chroma_path), s.chroma_collection, s.embedding_backend, s.embedding_model)
    with _lock:
        if key not in _stores:
            import chromadb
            from langchain_chroma import Chroma

            s.chroma_path.mkdir(parents=True, exist_ok=True)
            _stores[key] = Chroma(
                collection_name=s.chroma_collection,
                embedding_function=get_embeddings(),
                persist_directory=str(s.chroma_path),
                collection_metadata={"hnsw:space": "cosine"},
                client_settings=chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True,
                                                         persist_directory=str(s.chroma_path)),
            )
        return _stores[key]


def reset_vector_store() -> None:
    """Forget cached stores (tests / after changing settings)."""
    with _lock:
        _stores.clear()


def add_chunks(chunks: list[Document], doc_id: int) -> int:
    if not chunks:
        return 0
    store = get_vector_store()
    try:
        for start in range(0, len(chunks), _BATCH):
            part = chunks[start : start + _BATCH]
            ids = [f"doc{doc_id}-chunk{c.metadata['chunk_index']}" for c in part]
            store.add_documents(part, ids=ids)
    except DocumentProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chroma add failed")
        delete_document_chunks(doc_id)
        raise DocumentProcessingError("Could not store the document in the vector database.", detail=str(exc)) from exc
    return len(chunks)


def delete_document_chunks(doc_id: int) -> None:
    try:
        get_vector_store()._collection.delete(where={"doc_id": int(doc_id)})
    except Exception:  # noqa: BLE001
        logger.exception("Could not delete chunks for doc %s", doc_id)


def build_filter(doc_ids: list[int] | None = None, file_type: str | None = None) -> dict | None:
    conds: list[dict] = []
    if doc_ids:
        conds.append({"doc_id": int(doc_ids[0])} if len(doc_ids) == 1 else {"doc_id": {"$in": [int(i) for i in doc_ids]}})
    if file_type:
        conds.append({"file_type": file_type})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def similarity_search(query: str, k: int, doc_ids: list[int] | None = None,
                      file_type: str | None = None) -> list[tuple[Document, float]]:
    """Return (chunk, cosine_similarity) pairs, best first."""
    store = get_vector_store()
    try:
        count = store._collection.count()
        if count == 0:
            return []
        pairs = store.similarity_search_with_score(query, k=min(k, count), filter=build_filter(doc_ids, file_type))
    except DocumentProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chroma query failed")
        raise DocumentProcessingError("The vector database query failed.", detail=str(exc)) from exc
    # Chroma returns cosine *distance*; similarity = 1 - distance
    return [(doc, max(0.0, min(1.0, 1.0 - float(dist)))) for doc, dist in pairs]


def get_document_chunks(doc_id: int) -> list[Document]:
    store = get_vector_store()
    res = store.get(where={"doc_id": int(doc_id)}, include=["documents", "metadatas"])
    items = [Document(page_content=t, metadata=m) for t, m in zip(res.get("documents", []), res.get("metadatas", []))]
    return sorted(items, key=lambda d: d.metadata.get("chunk_index", 0))


def collection_count() -> int:
    return int(get_vector_store()._collection.count())
