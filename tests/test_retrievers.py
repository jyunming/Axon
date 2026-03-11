"""Tests for retriever modules."""

import pytest
import tempfile
import os
from axon.retrievers import BM25Retriever, reciprocal_rank_fusion


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


# ---------------------------------------------------------------------------
# New tests: delete_documents, save/load
# ---------------------------------------------------------------------------

class TestBM25RetrieverDelete:
    def test_delete_removes_document_from_search(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [
            {"id": "keep", "text": "python programming language", "metadata": {}},
            {"id": "remove", "text": "javascript web development", "metadata": {}},
        ]
        r.add_documents(docs)

        r.delete_documents(["remove"])

        results = r.search("javascript web", top_k=5)
        ids = [res["id"] for res in results]
        assert "remove" not in ids

    def test_delete_keeps_other_documents(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [
            {"id": "d1", "text": "alpha beta gamma", "metadata": {}},
            {"id": "d2", "text": "delta epsilon zeta", "metadata": {}},
            {"id": "d3", "text": "alpha delta omega", "metadata": {}},
        ]
        r.add_documents(docs)
        r.delete_documents(["d2"])

        assert len(r.corpus) == 2
        remaining_ids = [d["id"] for d in r.corpus]
        assert "d1" in remaining_ids
        assert "d3" in remaining_ids
        assert "d2" not in remaining_ids

    def test_delete_all_documents_results_in_empty_index(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "only", "text": "some content", "metadata": {}}]
        r.add_documents(docs)
        r.delete_documents(["only"])

        assert r.corpus == []
        assert r.bm25 is None
        assert r.search("some content") == []

    def test_delete_nonexistent_id_is_noop(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "real", "text": "hello world", "metadata": {}}]
        r.add_documents(docs)

        r.delete_documents(["ghost_id"])

        assert len(r.corpus) == 1
        assert r.corpus[0]["id"] == "real"

    def test_delete_persists_to_disk(self, tmp_path):
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [
            {"id": "a", "text": "aardvark", "metadata": {}},
            {"id": "b", "text": "baboon", "metadata": {}},
        ]
        r.add_documents(docs)
        r.delete_documents(["a"])

        # Reload from disk
        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert len(r2.corpus) == 1
        assert r2.corpus[0]["id"] == "b"


class TestBM25RetrieverSaveLoad:
    def test_save_and_load_preserves_corpus(self, tmp_path):
        r1 = BM25Retriever(storage_path=str(tmp_path))
        docs = [
            {"id": "x1", "text": "machine learning basics", "metadata": {"src": "a"}},
            {"id": "x2", "text": "deep neural networks", "metadata": {"src": "b"}},
        ]
        r1.add_documents(docs)
        r1.save()

        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert len(r2.corpus) == 2
        loaded_ids = {d["id"] for d in r2.corpus}
        assert loaded_ids == {"x1", "x2"}

    def test_loaded_retriever_can_search(self, tmp_path):
        # BM25Okapi IDF = log(N-df+0.5) - log(df+0.5).
        # With only 2 docs and the term in 1, IDF = 0 → score = 0 → filtered out.
        # Use 3 docs so the unique term gets positive IDF.
        r1 = BM25Retriever(storage_path=str(tmp_path))
        docs = [
            {"id": "q1", "text": "quantum computing qubits superposition", "metadata": {}},
            {"id": "q2", "text": "classical computing transistors", "metadata": {}},
            {"id": "q3", "text": "classical computing binary gates", "metadata": {}},
        ]
        r1.add_documents(docs)

        r2 = BM25Retriever(storage_path=str(tmp_path))
        results = r2.search("quantum", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "q1"

    def test_empty_corpus_saves_and_loads(self, tmp_path):
        r1 = BM25Retriever(storage_path=str(tmp_path))
        r1.save()

        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert r2.corpus == []
        assert r2.bm25 is None
