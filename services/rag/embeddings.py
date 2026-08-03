"""Local embedding providers (no API key needed).

Design rationale
----------------
The configured embedding model (``all-MiniLM-L6-v2`` by default) runs locally.
We prefer ``sentence-transformers`` (best quality, batched) but fall back to
ChromaDB's built-in ONNX implementation of the same MiniLM model when PyTorch is
unavailable — this keeps the pipeline working on minimal installs and on Python
versions where ``torch`` wheels are not yet published.
"""

from __future__ import annotations

import logging
from typing import Protocol

from config.settings import VectorStoreSettings
from utils.errors import RetrievalError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - fallback path
    SentenceTransformer = None


class EmbeddingProvider(Protocol):
    """Anything that turns a list of texts into a list of vectors."""

    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerProvider:
    """Batch embeddings via ``sentence-transformers``."""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RetrievalError(
                "sentence-transformers is not installed; cannot use "
                "SentenceTransformerProvider"
            )
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in embeddings]


class ChromaOnnxProvider:
    """Embeddings via ChromaDB's bundled ONNX MiniLM runtime."""

    name = "chromadb-onnx"

    def __init__(self) -> None:
        try:
            from chromadb.utils import embedding_functions
        except ImportError as exc:  # pragma: no cover - dependency guaranteed
            raise RetrievalError(
                "chromadb is not installed; cannot use ChromaOnnxProvider"
            ) from exc
        self._fn = embedding_functions.ONNXMiniLM_L6_V2()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(map(float, vec)) for vec in self._fn(texts)]


def get_embedding_provider(settings: VectorStoreSettings) -> EmbeddingProvider:
    """Return the best available embedding provider for the configured model."""
    model_name = settings.embedding_model or "all-MiniLM-L6-v2"
    if SentenceTransformer is not None:
        try:
            provider = SentenceTransformerProvider(model_name)
            logger.info("Using sentence-transformers embeddings (%s)", model_name)
            return provider
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning(
                "sentence-transformers failed to load (%s); falling back to ONNX", exc
            )
    provider = ChromaOnnxProvider()
    logger.info("Using chromadb ONNX MiniLM embeddings")
    return provider
