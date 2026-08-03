"""Tests for token-aware chunking and ingestion helpers."""

from __future__ import annotations

from services.rag.chunking import chunk_document_pages, chunk_text, count_tokens

SAMPLE = (
    "The mitochondria is the powerhouse of the cell. Cells are the basic "
    "building blocks of all living organisms. Each organism is made of trillions "
    "of cells, each carrying out specialised functions. "
    "Photosynthesis in plants converts sunlight into chemical energy, "
    "producing glucose and oxygen as by-products. "
    "The Krebs cycle, also known as the citric acid cycle, is central to aerobic "
    "respiration in eukaryotic cells."
) * 3


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_positive_count(self):
        assert count_tokens("hello world how are you today") > 0


class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_returns_nonempty_stripped_chunks(self):
        chunks = chunk_text(SAMPLE, chunk_size=100, overlap=20)
        assert chunks
        assert all(isinstance(c, str) and c.strip() for c in chunks)

    def test_multi_chunk_splits(self):
        chunks = chunk_text(SAMPLE, chunk_size=40, overlap=10)
        assert len(chunks) > 1

    def test_content_preserved_roughly(self):
        chunks = chunk_text(SAMPLE, chunk_size=100, overlap=20)
        joined = " ".join(chunks).lower()
        # Key words must survive the split.
        assert "mitochondria" in joined
        assert "photosynthesis" in joined

    def test_overlap_capped(self):
        # overlap >= chunk_size is clamped to chunk_size - 1 (no infinite loop).
        chunks = chunk_text("a b c d e f g h i j k l m n o p", chunk_size=4, overlap=10)
        assert chunks
        assert len(chunks) <= 16  # must terminate and not blow up

    def test_single_word_chunk_size(self):
        chunks = chunk_text("one two three four", chunk_size=1, overlap=0)
        assert chunks


class TestChunkDocumentPages:
    def test_tags_chunks_with_page_and_source(self):
        pages = [(1, "Page one text " * 10), (2, "Page two text " * 10)]
        chunks = chunk_document_pages(
            pages, chunk_size=50, overlap=10, source="notes.md"
        )
        assert chunks
        assert all(c["source"] == "notes.md" for c in chunks)
        pages_seen = {c["page"] for c in chunks}
        assert pages_seen == {1, 2}

    def test_empty_pages(self):
        assert chunk_document_pages([], source="x.md") == []

    def test_cap_not_exceeded(self):
        from config.constants import MAX_CHUNKS_PER_DOCUMENT

        pages = [(1, "word " * 100000)]
        chunks = chunk_document_pages(pages, chunk_size=10, overlap=0, source="big.txt")
        assert len(chunks) <= MAX_CHUNKS_PER_DOCUMENT
