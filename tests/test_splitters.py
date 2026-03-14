"""Tests for text splitters."""

from axon.splitters import RecursiveCharacterTextSplitter, SemanticTextSplitter


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
