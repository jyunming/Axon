"""Tests for retriever modules."""

import pytest
import tempfile
import os
from rag_brain.retrievers import BM25Retriever, reciprocal_rank_fusion


class TestBM25Retriever:
    """Test the BM25Retriever class."""

    def test_add_and_search(self):
        """Test adding documents and searching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = BM25Retriever(storage_path=tmpdir)

            docs = [
                {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog", "metadata": {}},
                {"id": "doc2", "text": "Python is a great programming language", "metadata": {}},
                {"id": "doc3", "text": "Machine learning with Python is powerful", "metadata": {}}
            ]

            retriever.add_documents(docs)

            results = retriever.search("Python programming", top_k=2)

            assert len(results) > 0
            assert results[0]["id"] in ["doc2", "doc3"]
            assert "score" in results[0]

    def test_empty_search(self):
        """Test searching with no documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = BM25Retriever(storage_path=tmpdir)
            results = retriever.search("test query")
            assert results == []

    def test_persistence(self):
        """Test that index is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add documents
            retriever1 = BM25Retriever(storage_path=tmpdir)
            docs = [{"id": "doc1", "text": "Test document", "metadata": {}}]
            retriever1.add_documents(docs)

            # Load in new instance
            retriever2 = BM25Retriever(storage_path=tmpdir)
            assert len(retriever2.corpus) == 1
            assert retriever2.corpus[0]["id"] == "doc1"


class TestReciprocalRankFusion:
    """Test reciprocal rank fusion function."""

    def test_fusion_basic(self):
        """Test basic fusion of two result sets."""
        vector_results = [
            {"id": "doc1", "text": "text1", "score": 0.9},
            {"id": "doc2", "text": "text2", "score": 0.8}
        ]
        bm25_results = [
            {"id": "doc2", "text": "text2", "score": 5.0},
            {"id": "doc3", "text": "text3", "score": 4.0}
        ]

        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        assert len(fused) == 3
        assert all("score" in doc for doc in fused)
        # doc2 should be ranked higher as it appears in both
        assert fused[0]["id"] == "doc2"

    def test_fusion_empty_bm25(self):
        """Test fusion with empty BM25 results."""
        vector_results = [{"id": "doc1", "text": "text1", "score": 0.9}]
        bm25_results = []

        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        assert len(fused) == 1
        assert fused[0]["id"] == "doc1"
