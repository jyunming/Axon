"""Tests for text splitters."""

import pytest
from rag_brain.splitters import RecursiveCharacterTextSplitter


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
        docs = [
            {"id": "doc1", "text": "A" * 100, "metadata": {"source": "test"}}
        ]

        result = splitter.transform_documents(docs)

        assert len(result) > 1
        assert all("_chunk_" in doc["id"] for doc in result)
        assert all("chunk" in doc["metadata"] for doc in result)
        assert all(doc["metadata"]["source"] == "test" for doc in result)
