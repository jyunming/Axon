"""Tests for text splitters."""

import pytest

from axon.splitters import RecursiveCharacterTextSplitter, SemanticTextSplitter, TableSplitter


class TestSemanticTextSplitter:
    def test_sentence_boundaries_preserved(self):
        # 10 tokens (~40 chars). Enough for 1-2 short sentences.
        splitter = SemanticTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "This is the first sentence. This is the second sentence! Is this the third?"
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert "first sentence." in chunks[0]
        assert "second sentence!" in chunks[1]

    def test_abbreviations_not_split(self):
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "Dr. Smith went to Washington, D.C. He bought a 1.5 L bottle."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_overlap(self):
        # 12 tokens per chunk, 4 token overlap
        splitter = SemanticTextSplitter(chunk_size=12, chunk_overlap=4)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Overlap should ensure a sentence is repeated in subsequent chunks
        assert "Sentence four." in chunks[0]
        assert "Sentence four." in chunks[1]

    def test_giant_sentence_fallback(self):
        # A single sentence larger than chunk_size should not be split mid-word.
        # It should just be yielded as a single oversized chunk to preserve semantic integrity.
        splitter = SemanticTextSplitter(chunk_size=5, chunk_overlap=1)
        text = "This is a ridiculously long single sentence that goes on forever without any punctuation marks whatsoever"
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestRecursiveCharacterTextSplitter:
    """Test the RecursiveCharacterTextSplitter class."""

    def test_basic_split(self):
        """Test basic text splitting."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 20
        chunks = splitter.split(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some variance

    def test_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split("")
        assert chunks == []

    def test_short_text(self):
        """Test text shorter than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        text = "Short text"
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_transform_documents(self):
        """Test document transformation with metadata."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        docs = [{"id": "doc1", "text": "A" * 100, "metadata": {"source": "test"}}]

        result = splitter.transform_documents(docs)

        assert len(result) > 1
        assert all("_chunk_" in doc["id"] for doc in result)
        assert all("chunk" in doc["metadata"] for doc in result)
        assert all(doc["metadata"]["source"] == "test" for doc in result)


class TestSemanticTextSplitterValidation:
    """Cover ValueError branches (lines 18, 20, 22) and tiktoken fallback (lines 31-35, 40)."""

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            SemanticTextSplitter(chunk_size=0)

    def test_chunk_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=10, chunk_overlap=10)

    def test_tiktoken_fallback_get_length(self, monkeypatch):
        """When tiktoken is unavailable, _get_length falls back to len(text)//4."""
        import sys

        # Patch tiktoken import to fail
        original = sys.modules.get("tiktoken")
        sys.modules["tiktoken"] = None  # type: ignore[assignment]
        try:
            splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=0)
            splitter.encoder = None
            length = splitter._get_length("hello world test")
            assert length == len("hello world test") // 4
        finally:
            if original is None:
                sys.modules.pop("tiktoken", None)
            else:
                sys.modules["tiktoken"] = original

    def test_empty_text_returns_empty(self):
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=0)
        assert splitter.split("") == []

    def test_abbreviation_in_sentence_continues(self):
        """Abbreviation mid-sentence (line 91-93 branch) should not split."""
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=0)
        text = "Send it to Mr. Jones tomorrow."
        chunks = splitter.split(text)
        assert len(chunks) == 1

    def test_oversized_sentence_flushes_current_chunk(self):
        """If there's already content and then an oversized sentence, flush first (lines 117-120)."""
        splitter = SemanticTextSplitter(chunk_size=8, chunk_overlap=0)
        # Short sentence + very long sentence — forces the flush-then-yield path
        short = "Hi."
        long_sent = "This is an extremely long single sentence that will exceed any reasonable chunk size limit."
        chunks = splitter.split(f"{short} {long_sent}")
        assert any(short.rstrip(".") in c for c in chunks)
        assert any(long_sent in c for c in chunks)

    def test_transform_documents_preserves_metadata(self):
        """SemanticTextSplitter.transform_documents copies metadata correctly (line 76)."""
        splitter = SemanticTextSplitter(chunk_size=5, chunk_overlap=0)
        docs = [{"id": "d1", "text": "Sentence one. Sentence two.", "metadata": {"src": "x"}}]
        result = splitter.transform_documents(docs)
        assert all(r["metadata"]["src"] == "x" for r in result)
        assert all("_chunk_" in r["id"] for r in result)

    def test_trailing_sentence_appended(self):
        """Text not ending in punct is collected and appended (line 95-96)."""
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=0)
        text = "First sentence. No final punctuation here"
        chunks = splitter.split(text)
        combined = " ".join(chunks)
        assert "No final punctuation here" in combined


class TestRecursiveCharacterTextSplitterValidation:
    """Cover ValueError branches (lines 207, 209, 211)."""

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveCharacterTextSplitter(chunk_size=0)

    def test_chunk_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)


class TestTableSplitter:
    """Cover TableSplitter (lines 163-196)."""

    def test_single_row(self):
        splitter = TableSplitter(table_name="Users", batch_size=1)
        rows = [{"name": "Alice", "age": "30"}]
        headers = ["name", "age"]
        result = splitter.transform_rows(rows, headers)
        assert len(result) == 1
        assert "Alice" in result[0]
        assert "Users" in result[0]

    def test_batch_grouping(self):
        splitter = TableSplitter(table_name="Orders", batch_size=2)
        rows = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        headers = ["id"]
        result = splitter.transform_rows(rows, headers)
        # 3 rows in batches of 2 → 2 chunks
        assert len(result) == 2

    def test_empty_value_skipped(self):
        """Cells with empty string or None are excluded from row string (line 183)."""
        splitter = TableSplitter(batch_size=1)
        rows = [{"a": "hello", "b": "", "c": None, "d": "world"}]
        headers = ["a", "b", "c", "d"]
        result = splitter.transform_rows(rows, headers)
        assert "b:" not in result[0]
        assert "c:" not in result[0]
        assert "hello" in result[0]
        assert "world" in result[0]

    def test_partial_batch_flushed(self):
        """Remainder rows below batch_size are still returned (line 193-194)."""
        splitter = TableSplitter(batch_size=5)
        rows = [{"x": str(i)} for i in range(3)]
        headers = ["x"]
        result = splitter.transform_rows(rows, headers)
        assert len(result) == 1
