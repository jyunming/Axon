"""Tests for axon.sparse_retrieval (Epic 4 Story 4.3)."""

import pytest

from axon.sparse_retrieval import (
    NoOpSparseRetriever,
    SparseRetriever,
    SparseVector,
    empty_sparse_vector,
    fuse_sparse,
)

# ---------------------------------------------------------------------------
# SparseVector contract
# ---------------------------------------------------------------------------


class TestSparseVector:
    def test_valid_construction(self):
        sv = SparseVector(indices=[0, 5, 42], values=[0.8, 0.3, 0.1])
        assert len(sv.indices) == 3
        assert sv.dim == 0  # unset by default
        assert sv.model == ""

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="indices length"):
            SparseVector(indices=[0, 1], values=[0.5])

    def test_empty_vector_valid(self):
        sv = SparseVector(indices=[], values=[])
        assert len(sv.indices) == 0

    def test_as_dict_fields(self):
        sv = SparseVector(indices=[1, 2], values=[0.9, 0.5], dim=30000, model="test")
        d = sv.as_dict()
        assert d["nnz"] == 2
        assert d["dim"] == 30000
        assert d["model"] == "test"
        assert d["indices"] == [1, 2]
        assert d["values"] == [0.9, 0.5]

    def test_empty_sparse_vector_helper(self):
        sv = empty_sparse_vector(dim=50000, model="splade")
        assert sv.indices == []
        assert sv.values == []
        assert sv.dim == 50000
        assert sv.model == "splade"


# ---------------------------------------------------------------------------
# SparseRetriever Protocol
# ---------------------------------------------------------------------------


class TestSparseRetrieverProtocol:
    def test_noop_satisfies_protocol(self):
        r = NoOpSparseRetriever()
        assert isinstance(r, SparseRetriever)

    def test_noop_encode_query_returns_empty(self):
        r = NoOpSparseRetriever(model="noop-test")
        sv = r.encode_query("what is the capital of France?")
        assert isinstance(sv, SparseVector)
        assert sv.indices == []
        assert sv.model == "noop-test"

    def test_noop_search_returns_empty(self):
        r = NoOpSparseRetriever()
        sv = empty_sparse_vector()
        results = r.search(sv, top_k=5)
        assert results == []

    def test_noop_search_with_filter_still_empty(self):
        r = NoOpSparseRetriever()
        sv = empty_sparse_vector()
        results = r.search(sv, top_k=5, filter_dict={"source": "foo"})
        assert results == []

    def test_custom_class_satisfies_protocol(self):
        """A minimal class with encode_query + search satisfies SparseRetriever."""

        class MinimalSparse:
            def encode_query(self, query: str) -> SparseVector:
                return empty_sparse_vector()

            def search(self, qv: SparseVector, top_k: int = 10, filter_dict=None):
                return []

        assert isinstance(MinimalSparse(), SparseRetriever)


# ---------------------------------------------------------------------------
# fuse_sparse
# ---------------------------------------------------------------------------


def _dense_result(doc_id: str, text: str, score: float, **meta) -> dict:
    return {"id": doc_id, "text": text, "score": score, "metadata": meta}


class TestFuseSparse:
    def test_noop_retriever_returns_dense_results_unchanged(self):
        dense = [
            _dense_result("d1", "text one", 0.9),
            _dense_result("d2", "text two", 0.7),
        ]
        result = fuse_sparse(NoOpSparseRetriever(), "query", dense, top_k=5)
        # No sparse hits → dense results returned as-is
        assert [r["id"] for r in result] == ["d1", "d2"]

    def test_sparse_hit_boosts_overlapping_doc(self):
        dense = [_dense_result("d1", "text one", 0.5)]
        sparse_hit = {"id": "d1", "text": "text one", "score": 1.0, "metadata": {}}

        class FixedSparse:
            def encode_query(self, q):
                return empty_sparse_vector()

            def search(self, qv, top_k=10, filter_dict=None):
                return [sparse_hit]

        result = fuse_sparse(FixedSparse(), "q", dense, top_k=5, sparse_weight=0.3)
        assert len(result) == 1
        assert result[0]["id"] == "d1"
        # Score must be a blend, not just the dense score
        assert result[0]["score"] != 0.5

    def test_sparse_introduces_new_doc(self):
        dense = [_dense_result("d1", "text one", 0.9)]
        sparse_hit = {"id": "d_new", "text": "extra doc", "score": 0.8, "metadata": {}}

        class FixedSparse:
            def encode_query(self, q):
                return empty_sparse_vector()

            def search(self, qv, top_k=10, filter_dict=None):
                return [sparse_hit]

        result = fuse_sparse(FixedSparse(), "q", dense, top_k=10, sparse_weight=0.3)
        ids = [r["id"] for r in result]
        assert "d1" in ids
        assert "d_new" in ids

    def test_top_k_limit_enforced(self):
        dense = [_dense_result(f"d{i}", f"text {i}", 1.0 - i * 0.05) for i in range(8)]
        result = fuse_sparse(NoOpSparseRetriever(), "q", dense, top_k=3)
        assert len(result) == 3

    def test_sparse_failure_falls_back_to_dense(self):
        dense = [_dense_result("d1", "text", 0.9)]

        class FailingSparse:
            def encode_query(self, q):
                raise RuntimeError("sparse index unavailable")

            def search(self, qv, top_k=10, filter_dict=None):
                return []

        result = fuse_sparse(FailingSparse(), "q", dense, top_k=5)
        assert result[0]["id"] == "d1"

    def test_sparse_metadata_tag_added(self):
        dense = [_dense_result("d1", "text", 0.5)]
        sparse_hit = {"id": "d1", "text": "text", "score": 0.8, "metadata": {}}

        class FixedSparse:
            def encode_query(self, q):
                return empty_sparse_vector()

            def search(self, qv, top_k=10, filter_dict=None):
                return [sparse_hit]

        result = fuse_sparse(FixedSparse(), "q", dense, top_k=5)
        assert "sparse_score" in result[0]["metadata"]

    def test_empty_dense_with_sparse_hits(self):
        sparse_hit = {"id": "s1", "text": "sparse only doc", "score": 1.0, "metadata": {}}

        class FixedSparse:
            def encode_query(self, q):
                return empty_sparse_vector()

            def search(self, qv, top_k=10, filter_dict=None):
                return [sparse_hit]

        result = fuse_sparse(FixedSparse(), "q", [], top_k=5)
        assert result[0]["id"] == "s1"

    def test_results_sorted_by_score_descending(self):
        dense = [
            _dense_result("low", "low", 0.3),
            _dense_result("high", "high", 0.9),
        ]
        result = fuse_sparse(NoOpSparseRetriever(), "q", dense, top_k=5)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)
