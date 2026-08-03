"""Alternative chunking strategies for the retrieval experiments.

The application chunks documents with a fixed token window
(``services.rag.chunking.chunk_text`` with ``chunk_size=600, overlap=100``).
These experiments compare that baseline against smaller / larger token windows
and a sentence-bounded strategy.

The baseline and token variants re-use the application's real chunker (read-
only import); the sentence strategy is an additional approach implemented here
so the comparison covers structurally different chunking, not just different
sizes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Make the application importable when this module is imported directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.rag.chunking import (  # noqa: E402  (application code, read-only)
    chunk_text,
    count_tokens,
)


# --------------------------------------------------------------------------- #
# Token-window strategies (reuse the application chunker with different sizes)
# --------------------------------------------------------------------------- #
def chunk_baseline(text: str) -> list[str]:
    """Application default: 600-token window, 100-token overlap."""
    return chunk_text(text, chunk_size=600, overlap=100)


def chunk_small(text: str) -> list[str]:
    """Small 300-token window, 50-token overlap (finer granularity)."""
    return chunk_text(text, chunk_size=300, overlap=50)


def chunk_large(text: str) -> list[str]:
    """Large 1000-token window, 200-token overlap (more context per chunk)."""
    return chunk_text(text, chunk_size=1000, overlap=200)


# --------------------------------------------------------------------------- #
# Sentence-bounded strategy
# --------------------------------------------------------------------------- #
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")


def chunk_sentences(text: str, max_tokens: int = 150) -> list[str]:
    """Greedily group sentences so each chunk stays under ``max_tokens``.

    A sentence is never split across chunks (keeps semantic units intact), so
    chunks are much smaller and cleaner than a fixed token window when the
    source is naturally sentence-shaped.
    """
    raw_sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in raw_sentences:
        n = max(count_tokens(sentence), 1)
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += n

    if current:
        chunks.append(" ".join(current))
    return chunks


CHUNKERS = [
    ("baseline",     "tokens 600/100",      chunk_baseline),
    ("small",        "tokens 300/50",       chunk_small),
    ("large",        "tokens 1000/200",     chunk_large),
    ("sentence",     "sentences ~150 tok",  chunk_sentences),
]
