"""Token-aware text chunking with a no-tiktoken fallback.

Design rationale
----------------
The vector store's ``chunk_size`` / ``chunk_overlap`` settings are expressed in
*tokens* because that is what the embedding model's context window cares about.
We approximate tokens with OpenAI's ``cl100k_base`` BPE via ``tiktoken`` (a good
proxy for modern embedders). If ``tiktoken`` is unavailable we fall back to a
whitespace word count so the pipeline degrades gracefully.
"""

from __future__ import annotations

import logging

from config.constants import MAX_CHUNKS_PER_DOCUMENT

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly
    import tiktoken
except ImportError:  # pragma: no cover - fallback path
    tiktoken = None

_ENCODING_NAME = "cl100k_base"
_encoder = None


def _get_encoder():
    """Lazily load the BPE encoder once per process."""
    global _encoder  # noqa: PLW0603 - intentional process-wide lazy cache
    if tiktoken is None:
        return None
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoder


def count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``."""
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text.split())


def chunk_text(
    text: str, chunk_size: int = 600, overlap: int = 100
) -> list[str]:
    """Split ``text`` into overlapping, token-bounded chunks.

    Overlap is capped at ``chunk_size - 1`` so chunks always make progress.
    Returns a list of non-empty, whitespace-stripped chunks.
    """
    if not text or not text.strip():
        return []
    chunk_size = max(int(chunk_size), 1)
    overlap = max(int(overlap), 0)
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    encoder = _get_encoder()
    if encoder is not None:
        tokens = encoder.encode(text)
        chunks: list[str] = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + chunk_size, n)
            piece = encoder.decode(tokens[start:end]).strip()
            if piece:
                chunks.append(piece)
            if end >= n:
                break
            start = max(start + chunk_size - overlap, start + 1)
        return chunks

    # Fallback: chunk by words when tiktoken is unavailable.
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i : i + chunk_size]).strip()
        if piece:
            chunks.append(piece)
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_document_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int = 600,
    overlap: int = 100,
    source: str,
) -> list[dict]:
    """Chunk a list of ``(page_number, text)`` pairs, tagging each chunk.

    Returns a list of ``{"text", "page", "source"}`` dicts — page boundaries are
    preserved per-chunk (a chunk never spans two pages), which keeps citations
    accurate. The number of returned chunks is capped by
    :data:`config.constants.MAX_CHUNKS_PER_DOCUMENT`.
    """
    chunks: list[dict] = []
    for page_number, text in pages:
        for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                {"text": piece, "page": page_number, "source": source}
            )
            if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT:
                logger.warning(
                    "Reached chunk cap (%d) while chunking %s", MAX_CHUNKS_PER_DOCUMENT, source
                )
                return chunks
    return chunks
