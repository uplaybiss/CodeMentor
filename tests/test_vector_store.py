"""Tests for the ChromaDB-backed vector store (real store, fake embedder)."""

from __future__ import annotations

import pytest

from services.rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder


@pytest.fixture
def store(settings) -> VectorStore:
    vs = VectorStore(settings.vector_store, embedder=FakeEmbedder())
    yield vs
    vs._client.delete_collection(vs._settings.collection_name)  # clean up per-test


def _chunks() -> list[dict]:
    return [
        {"text": "Photosynthesis converts sunlight into chemical energy.", "page": 1, "source": "bio.txt"},
        {"text": "The water cycle describes evaporation and precipitation.", "page": 2, "source": "bio.txt"},
        {"text": "Newton's laws describe the motion of objects.", "page": 3, "source": "phys.txt"},
    ]


class TestIngest:
    def test_add_returns_count(self, store):
        assert store.add_document("doc_a", _chunks(), source="bio.txt") == 3
        assert store.count() == 3

    def test_add_empty_returns_zero(self, store):
        assert store.add_document("doc_b", [], source="x.txt") == 0

    def test_reingest_replaces(self, store):
        store.add_document("doc_a", _chunks()[:2], source="bio.txt")
        store.add_document("doc_a", _chunks(), source="bio.txt")
        assert store.count() == 3

    def test_has_document(self, store):
        assert not store.has_document("doc_a")
        store.add_document("doc_a", _chunks(), source="bio.txt")
        assert store.has_document("doc_a")


class TestSearch:
    def test_empty_collection_returns_empty(self, store):
        assert store.search("anything") == []

    def test_returns_chunks_with_metadata(self, store):
        store.add_document("doc_a", _chunks(), source="bio.txt")
        results = store.search("photosynthesis energy", top_k=4)
        assert results
        chunk = results[0]
        assert chunk.content
        assert chunk.source == "bio.txt"
        assert chunk.page == 1
        assert 0.0 <= chunk.score <= 1.0

    def test_most_relevant_ranks_first(self, store):
        store.add_document("doc_a", _chunks(), source="bio.txt")
        results = store.search("evaporation precipitation rain", top_k=4)
        assert results[0].content  # expected to be the water-cycle chunk


class TestDelete:
    def test_delete_removes_document(self, store):
        store.add_document("doc_a", _chunks(), source="bio.txt")
        assert store.count() == 3
        store.delete_document("doc_a")
        assert store.count() == 0
        assert not store.has_document("doc_a")
