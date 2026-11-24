"""Embedding model factory.

Production: Hugging Face ``sentence-transformers/all-MiniLM-L6-v2`` (configurable).
``EMBEDDING_BACKEND=hash`` selects a tiny deterministic bag-of-words embedder that
needs no download. It exists for offline unit tests / CI only – retrieval quality is
far below a real embedding model.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from backend.config.settings import get_settings
from backend.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "by",
    "with", "as", "at", "it", "this", "that", "from", "what", "which", "who", "how", "do", "does", "my",
}
_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embedder (offline testing only)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _stem(tok: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
        return tok

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = [self._stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        for t in toks:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=4)
def _build(backend: str, model_name: str) -> Embeddings:
    if backend == "hash":
        logger.warning("Using HashEmbeddings (offline testing backend) – not for production use")
        return HashEmbeddings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not load embedding model")
        raise DocumentProcessingError(
            "The embedding model could not be loaded. The first run needs an internet connection to download "
            f"'{model_name}'. Check your connection and that 'sentence-transformers' is installed.",
            detail=str(exc),
        ) from exc


def get_embeddings() -> Embeddings:
    s = get_settings()
    return _build(s.embedding_backend, s.embedding_model)


def clear_embedding_cache() -> None:
    _build.cache_clear()
