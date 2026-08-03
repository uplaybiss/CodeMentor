"""Deterministic stand-ins for heavy/external components used in tests.

The one thing tests must not depend on is the real embedding model
(``sentence-transformers`` / ONNX), which is slow and downloads weights. Instead
the vector-store tests use :class:`FakeEmbedder` — a tiny deterministic hasher
that produces stable, unit-norm vectors whose *similarity* tracks token overlap,
so semantic-search behaviour (ranking by relevance) is still exercised for real.
"""

from __future__ import annotations

import math
import zlib


class FakeEmbedder:
    """Trigram-hash embedder: identical/similar texts yield similar vectors."""

    name = "fake"
    dim = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        lowered = (text or "").lower()
        for i in range(len(lowered) - 2):
            trigram = lowered[i : i + 3]
            idx = zlib.crc32(trigram.encode("utf-8")) % self.dim
            vec[idx] += 1.0
        if not any(vec):  # too short for a trigram — hash the whole string
            vec[zlib.crc32(lowered.encode("utf-8")) % self.dim] = 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec
