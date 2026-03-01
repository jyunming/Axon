"""Tests for retriever modules."""

import pytest
import tempfile
import os
from rag_brain.retrievers import BM25Retriever, reciprocal_rank_fusion


class TestBM25Retriever:
    """Test the BM25Retriever class."""

    def test_add_and_search(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [
            {"id": "doc1", "text": "python programming language", "metadata": {}},
            {"id": "doc2", "text": "javascript web development", "metadata": {}},
            {"id": "doc3", "text": "python data science machine learning", "metadata": {}},
        ]
        r.add_documents(docs)
        results = r.search("python", top_k=2)
        assert len(results) >= 1
        ids = [res["id"] for res in results]
        assert "doc1" in ids or "doc3" in ids

    def test_add_and_search_with_tempdir(self):
        """Test adding documents and searching (tempfile style)."""
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

    def test_search_empty_returns_empty(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        results = r.search("anything")
        assert results == []

    def test_empty_search(self):
        """Test searching with no documents (tempfile style)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = BM25Retriever(storage_path=tmpdir)
            results = retriever.search("test query")
            assert results == []

    def test_persistence_roundtrip(self, tmp_path):
        r1 = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "doc1", "text": "test content", "metadata": {}}]
        r1.add_documents(docs)

        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert len(r2.corpus) == 1
        assert r2.corpus[0]["id"] == "doc1"

    def test_persistence(self):
        """Test that index is saved and loaded correctly (tempfile style)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever1 = BM25Retriever(storage_path=tmpdir)
            docs = [{"id": "doc1", "text": "Test document", "metadata": {}}]
            retriever1.add_documents(docs)

            retriever2 = BM25Retriever(storage_path=tmpdir)
            assert len(retriever2.corpus) == 1
            assert retriever2.corpus[0]["id"] == "doc1"

    def test_scores_above_zero_only(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "doc1", "text": "apple fruit", "metadata": {}}]
        r.add_documents(docs)
        # Query totally unrelated — may return empty
        results = r.search("zzzzquux")
        # All returned results should have score > 0
        for res in results:
            assert res["score"] > 0

    def test_add_empty_does_not_rebuild(self, tmp_path):
        # Adding empty list should not raise and should not rebuild (corpus stays empty)
        r = BM25Retriever(storage_path=str(tmp_path))
        r.add_documents([])
        assert r.bm25 is None  # no rebuild happened
        assert r.corpus == []

    def test_batch_add_rebuilds_once(self, tmp_path):
        # Adding a batch of docs in one call correctly populates index
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": f"d{i}", "text": f"word{i}", "metadata": {}} for i in range(5)]
        r.add_documents(docs)
        assert len(r.corpus) == 5
        assert r.bm25 is not None


class TestReciprocalRankFusion:
    """Test reciprocal rank fusion function."""

    def test_merges_overlapping_results(self):
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}},
               {"id": "b", "text": "t", "score": 0.7, "metadata": {}}]
        bm25 = [{"id": "b", "text": "t", "score": 5.0, "metadata": {}},
                {"id": "c", "text": "t", "score": 3.0, "metadata": {}}]
        fused = reciprocal_rank_fusion(vec, bm25)
        ids = [r["id"] for r in fused]
        assert "a" in ids and "b" in ids and "c" in ids
        # "b" appears in both, so should score higher than "c" (only in bm25)
        b_score = next(r["score"] for r in fused if r["id"] == "b")
        c_score = next(r["score"] for r in fused if r["id"] == "c")
        assert b_score > c_score

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

    def test_fused_only_flag(self):
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}}]
        bm25 = [{"id": "z", "text": "t", "score": 5.0, "metadata": {}}]
        fused = reciprocal_rank_fusion(vec, bm25)
        z = next(r for r in fused if r["id"] == "z")
        assert z.get("fused_only") is True

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}}]
        result = reciprocal_rank_fusion(vec, [])
        assert len(result) == 1

    def test_fusion_empty_bm25(self):
        """Test fusion with empty BM25 results."""
        vector_results = [{"id": "doc1", "text": "text1", "score": 0.9}]
        bm25_results = []
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        assert len(fused) == 1
        assert fused[0]["id"] == "doc1"
