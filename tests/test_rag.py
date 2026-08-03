"""Tests for the RAG service orchestration and the RAG agent."""

from __future__ import annotations

from agents.rag import RAGAgent
from utils.errors import RetrievalError


class TestRAGService:
    def test_build_context_empty_when_no_docs(self, ctx):
        context = ctx.rag.build_context("photosynthesis")
        assert context.retrieved_chunks == []
        assert context.used is False

    def test_build_context_used_after_ingest(self, ctx):
        ctx.rag.ingest_chunks(
            "doc_x",
            [
                {"text": "Photosynthesis converts sunlight to energy.", "page": 1, "source": "bio.txt"},
                {"text": "Enzymes speed up chemical reactions.", "page": 2, "source": "bio.txt"},
            ],
        )
        context = ctx.rag.build_context("photosynthesis energy")
        assert context.used is True
        assert len(context.retrieved_chunks) >= 1
        assert context.retrieved_chunks[0].source == "bio.txt"

    def test_build_context_degrades_gracefully_on_error(self, ctx, monkeypatch):
        def boom(*args, **kwargs):
            raise RetrievalError("store is down")

        monkeypatch.setattr(ctx.rag, "search", boom)
        # Must not raise — returns an empty context instead.
        context = ctx.rag.build_context("anything")
        assert context.retrieved_chunks == []
        assert context.used is False

    def test_count_and_has_document(self, ctx):
        assert ctx.rag.count() == 0
        ctx.rag.ingest_chunks("doc_y", [{"text": "hello world", "page": 1, "source": "a.md"}])
        assert ctx.rag.count() == 1
        assert ctx.rag.has_document("doc_y")
        ctx.rag.delete_document("doc_y")
        assert ctx.rag.count() == 0


class TestRAGAgent:
    async def test_empty_state(self, ctx):
        agent = RAGAgent(ctx)
        out = await agent.run({})
        assert out["rag_context"]["retrieved_chunks"] == []
        assert out["rag_sources"] == []

    async def test_with_topic_queries_store(self, ctx):
        ctx.rag.ingest_chunks(
            "doc_z", [{"text": "Rockets use Newton's third law.", "page": 1, "source": "space.txt"}]
        )
        agent = RAGAgent(ctx)
        out = await agent.run({"student_id": "s1", "topic": "rockets newton"})
        assert out["rag_sources"] == ["space.txt"]
        assert len(out["rag_context"]["retrieved_chunks"]) == 1
