import pytest
import os


class TestBM25Retriever:
    def test_add_and_search(self, tmp_path):
        from rag_brain.retrievers import BM25Retriever
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

    def test_search_empty_returns_empty(self, tmp_path):
        from rag_brain.retrievers import BM25Retriever
        r = BM25Retriever(storage_path=str(tmp_path))
        results = r.search("anything")
        assert results == []

    def test_persistence_roundtrip(self, tmp_path):
        from rag_brain.retrievers import BM25Retriever
        r1 = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "doc1", "text": "test content", "metadata": {}}]
        r1.add_documents(docs)

        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert len(r2.corpus) == 1
        assert r2.corpus[0]["id"] == "doc1"

    def test_scores_above_zero_only(self, tmp_path):
        from rag_brain.retrievers import BM25Retriever
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "doc1", "text": "apple fruit", "metadata": {}}]
        r.add_documents(docs)
        # Query totally unrelated — may return empty
        results = r.search("zzzzquux")
        # All returned results should have score > 0
        for res in results:
            assert res["score"] > 0


    def test_add_empty_does_not_rebuild(self, tmp_path):
        from rag_brain.retrievers import BM25Retriever
        # Adding empty list should not raise and should not rebuild (corpus stays empty)
        r = BM25Retriever(storage_path=str(tmp_path))
        r.add_documents([])
        assert r.bm25 is None  # no rebuild happened
        assert r.corpus == []

    def test_batch_add_rebuilds_once(self, tmp_path):
        from rag_brain.retrievers import BM25Retriever
        # Adding a batch of docs in one call correctly populates index
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": f"d{i}", "text": f"word{i}", "metadata": {}} for i in range(5)]
        r.add_documents(docs)
        assert len(r.corpus) == 5
        assert r.bm25 is not None
    def test_merges_overlapping_results(self):
        from rag_brain.retrievers import reciprocal_rank_fusion
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

    def test_fused_only_flag(self):
        from rag_brain.retrievers import reciprocal_rank_fusion
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}}]
        bm25 = [{"id": "z", "text": "t", "score": 5.0, "metadata": {}}]
        fused = reciprocal_rank_fusion(vec, bm25)
        z = next(r for r in fused if r["id"] == "z")
        assert z.get("fused_only") is True

    def test_empty_inputs(self):
        from rag_brain.retrievers import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([], []) == []
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}}]
        result = reciprocal_rank_fusion(vec, [])
        assert len(result) == 1
