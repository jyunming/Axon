"""Tests for axon.sparse_retrieval (Epic 4 Story 4.3 + audit batch E1)."""

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


# ---------------------------------------------------------------------------
# SpladeSparseRetriever (Phase 1 backend)
#
# These tests exercise the persistence + scoring interface of the SPLADE
# backend WITHOUT loading the real ~440 MB model. We monkeypatch
# `_ensure_model_loaded` and `_encode` so the suite is fast and deterministic
# even when transformers is installed. A separate end-to-end test below is
# guarded with importorskip + a marker so CI without the optional extra
# installed still passes.
# ---------------------------------------------------------------------------


from axon.sparse_retrieval import SpladeSparseRetriever  # noqa: E402


def _stub_encode_factory(weight_map_per_text: dict):
    """Build a deterministic _encode stub that returns SparseVectors based on the input text.

    weight_map_per_text: dict[text -> dict[token_id -> weight]]
    """

    def _encode(self, text):
        weights = weight_map_per_text.get(text, {})
        indices = list(weights.keys())
        values = [float(weights[i]) for i in indices]
        return SparseVector(indices=indices, values=values, dim=100, model="stub")

    return _encode


class TestSpladeSparseRetrieverNoModel:
    """Behavioural tests that bypass the heavy SPLADE model load."""

    def _build(self, tmp_path, monkeypatch, encode_map):
        # Bypass model load entirely.
        monkeypatch.setattr(SpladeSparseRetriever, "_ensure_model_loaded", lambda self: None)
        monkeypatch.setattr(SpladeSparseRetriever, "_encode", _stub_encode_factory(encode_map))
        r = SpladeSparseRetriever(
            storage_path=str(tmp_path / "sparse"),
            model="stub-model",
            eager_load_model=False,
        )
        # Bake the dim so persistence round-trip exercises the field.
        r.dim = 100
        return r

    def test_encode_query_returns_sparse_vector(self, tmp_path, monkeypatch):
        encode_map = {"hello world": {1: 0.9, 5: 0.4}}
        r = self._build(tmp_path, monkeypatch, encode_map)
        sv = r.encode_query("hello world")
        assert isinstance(sv, SparseVector)
        assert sv.indices == [1, 5]
        assert sv.values == [0.9, 0.4]

    def test_add_and_search_returns_relevant_doc(self, tmp_path, monkeypatch):
        encode_map = {
            "doc about cats": {1: 0.9, 2: 0.5},
            "doc about dogs": {3: 0.9, 4: 0.5},
            "doc about birds": {5: 0.9, 6: 0.5},
            "tell me about cats": {1: 0.8, 2: 0.4},
        }
        r = self._build(tmp_path, monkeypatch, encode_map)
        added = r.add(
            [
                {"id": "d1", "text": "doc about cats", "metadata": {}},
                {"id": "d2", "text": "doc about dogs", "metadata": {}},
                {"id": "d3", "text": "doc about birds", "metadata": {}},
            ]
        )
        assert added == 3
        hits = r.search("tell me about cats", top_k=3)
        assert len(hits) >= 1
        # The cat doc must be the top hit.
        assert hits[0]["id"] == "d1"
        # Dogs / birds share no tokens with the query, so they should not appear.
        assert all(h["id"] != "d3" for h in hits)

    def test_search_top_k_limit(self, tmp_path, monkeypatch):
        encode_map = {f"text {i}": {1: 1.0, i + 10: 0.5} for i in range(5)}
        encode_map["query"] = {1: 1.0}
        r = self._build(tmp_path, monkeypatch, encode_map)
        r.add([{"id": f"d{i}", "text": f"text {i}", "metadata": {}} for i in range(5)])
        hits = r.search("query", top_k=2)
        assert len(hits) == 2

    def test_search_filter_dict_applied(self, tmp_path, monkeypatch):
        encode_map = {
            "alpha doc": {1: 1.0},
            "beta doc": {1: 1.0},
            "query": {1: 1.0},
        }
        r = self._build(tmp_path, monkeypatch, encode_map)
        r.add(
            [
                {"id": "a", "text": "alpha doc", "metadata": {"kind": "alpha"}},
                {"id": "b", "text": "beta doc", "metadata": {"kind": "beta"}},
            ]
        )
        hits = r.search("query", top_k=10, filter_dict={"kind": "alpha"})
        assert len(hits) == 1
        assert hits[0]["id"] == "a"

    def test_persist_and_reload_returns_same_results(self, tmp_path, monkeypatch):
        encode_map = {
            "first doc": {1: 0.9, 2: 0.5},
            "second doc": {3: 0.9, 4: 0.5},
            "query": {1: 0.8, 3: 0.2},
        }
        # Build, add, save.
        r1 = self._build(tmp_path, monkeypatch, encode_map)
        r1.add(
            [
                {"id": "d1", "text": "first doc", "metadata": {"k": 1}},
                {"id": "d2", "text": "second doc", "metadata": {"k": 2}},
            ]
        )
        r1.save()
        hits1 = r1.search("query", top_k=5)

        # Build a fresh retriever pointing at the same dir; it must rehydrate.
        r2 = self._build(tmp_path, monkeypatch, encode_map)
        assert set(r2.docs.keys()) == {"d1", "d2"}
        assert r2.docs["d1"]["metadata"] == {"k": 1}
        # Token-id keys must come back as ints, not strings.
        assert all(isinstance(t, int) for t in r2.docs["d1"]["vec"].keys())

        hits2 = r2.search("query", top_k=5)
        assert [(h["id"], round(h["score"], 6)) for h in hits1] == [
            (h["id"], round(h["score"], 6)) for h in hits2
        ]

    def test_empty_index_search_returns_empty(self, tmp_path, monkeypatch):
        encode_map = {"query": {1: 1.0}}
        r = self._build(tmp_path, monkeypatch, encode_map)
        assert r.search("query", top_k=5) == []

    def test_add_skips_documents_with_no_id_or_text(self, tmp_path, monkeypatch):
        encode_map = {"good": {1: 1.0}}
        r = self._build(tmp_path, monkeypatch, encode_map)
        added = r.add(
            [
                {"id": "d1", "text": "good", "metadata": {}},
                {"id": "", "text": "skipme", "metadata": {}},
                {"id": "d3", "text": "", "metadata": {}},
            ]
        )
        assert added == 1
        assert set(r.docs.keys()) == {"d1"}

    def test_encode_query_failure_returns_empty_vector(self, tmp_path, monkeypatch):
        # Build with a stub _encode that always raises.
        monkeypatch.setattr(SpladeSparseRetriever, "_ensure_model_loaded", lambda self: None)

        def _boom(self, text):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(SpladeSparseRetriever, "_encode", _boom)
        r = SpladeSparseRetriever(
            storage_path=str(tmp_path / "sparse"),
            model="stub-model",
            eager_load_model=False,
        )
        sv = r.encode_query("anything")
        assert isinstance(sv, SparseVector)
        assert sv.indices == []
        assert sv.values == []

    def test_add_skips_failed_encodes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SpladeSparseRetriever, "_ensure_model_loaded", lambda self: None)

        def _selective(self, text):
            if text == "fail":
                raise RuntimeError("nope")
            return SparseVector(indices=[1], values=[1.0], dim=100, model="stub")

        monkeypatch.setattr(SpladeSparseRetriever, "_encode", _selective)
        r = SpladeSparseRetriever(
            storage_path=str(tmp_path / "sparse"),
            model="stub-model",
            eager_load_model=False,
        )
        added = r.add(
            [
                {"id": "ok", "text": "ok", "metadata": {}},
                {"id": "bad", "text": "fail", "metadata": {}},
            ]
        )
        assert added == 1
        assert "ok" in r.docs and "bad" not in r.docs

    def test_save_format_uses_string_token_keys(self, tmp_path, monkeypatch):
        import json

        encode_map = {"hello": {7: 0.5, 13: 0.3}}
        r = self._build(tmp_path, monkeypatch, encode_map)
        r.add([{"id": "h", "text": "hello", "metadata": {}}])
        r.save()
        with open(r.index_path, encoding="utf-8") as fh:
            blob = json.load(fh)
        assert blob["model"] == "stub-model"
        # JSON requires string keys; we round-trip them to int on load.
        vec_keys = list(blob["docs"]["h"]["vec"].keys())
        assert all(isinstance(k, str) for k in vec_keys)
        assert sorted(vec_keys) == sorted([str(k) for k in encode_map["hello"]])


class TestSpladeSparseRetrieverGracefulFallback:
    """SPLADE init must not abort the brain when the optional extra is missing."""

    def test_missing_transformers_raises_importerror_with_install_hint(self, tmp_path, monkeypatch):
        # Force the transformers / torch import inside _ensure_model_loaded to fail.
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name in {"transformers", "torch"} or name.startswith("transformers."):
                raise ImportError(f"forced: {name} unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="axon-rag\\[sparse\\]"):
            SpladeSparseRetriever(
                storage_path=str(tmp_path / "sparse"),
                model="stub-model",
                eager_load_model=True,
            )


# ---------------------------------------------------------------------------
# End-to-end SPLADE encoding (only runs when the optional extra is installed
# AND the env opts in — model download is ~440 MB).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    True,  # skip by default; flip to env-gated check when wiring CI
    reason=(
        "End-to-end SPLADE test requires `pip install axon-rag[sparse]` and a "
        "~440 MB model download. Skipped in the default test run."
    ),
)
def test_splade_real_model_smoke(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from axon.sparse_retrieval import SpladeSparseRetriever as _RealSplade

    r = _RealSplade(storage_path=str(tmp_path / "splade"))
    r.add(
        [
            {"id": "d1", "text": "the cat sat on the mat", "metadata": {}},
            {"id": "d2", "text": "machine learning models for retrieval", "metadata": {}},
        ]
    )
    hits = r.search("retrieval models", top_k=2)
    assert hits and hits[0]["id"] == "d2"
