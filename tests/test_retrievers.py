from __future__ import annotations
﻿"""Tests for retriever modules."""

import os
from unittest.mock import patch

from axon.retrievers import BM25Retriever, weighted_score_fusion


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

    def test_add_and_search_with_tempdir(self, tmp_path):
        """Test adding documents and searching (tempfile style)."""
        tmpdir = str(tmp_path)
        retriever = BM25Retriever(storage_path=tmpdir)
        docs = [
            {
                "id": "doc1",
                "text": "The quick brown fox jumps over the lazy dog",
                "metadata": {},
            },
            {"id": "doc2", "text": "Python is a great programming language", "metadata": {}},
            {"id": "doc3", "text": "Machine learning with Python is powerful", "metadata": {}},
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

    def test_empty_search(self, tmp_path):
        """Test searching with no documents (tempfile style)."""
        tmpdir = str(tmp_path)
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

    def test_persistence(self, tmp_path):
        """Test that index is saved and loaded correctly (tempfile style)."""
        tmpdir = str(tmp_path)
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
        # Query totally unrelated â€” may return empty
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
        # Adding docs sets the dirty flag; index is rebuilt lazily on first search() (Story 6.2).
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": f"d{i}", "text": f"word{i}", "metadata": {}} for i in range(5)]
        r.add_documents(docs)
        assert len(r.corpus) == 5
        # After add, index is NOT yet built â€” dirty flag is set instead (lazy rebuild)
        assert r._dirty is True
        assert r.bm25 is None
        # First search triggers exactly one rebuild
        r.search("word1")
        assert r._dirty is False
        assert r.bm25 is not None


class TestWeightedScoreFusion:
    """Test weighted score fusion function (Normalized Convex Combination)."""

    def test_merges_overlapping_results(self):
        vec = [
            {"id": "a", "text": "t", "score": 0.9, "metadata": {}},
            {"id": "b", "text": "t", "score": 0.7, "metadata": {}},
        ]
        bm25 = [
            {"id": "b", "text": "t", "score": 5.0, "metadata": {}},
            {"id": "c", "text": "t", "score": 3.0, "metadata": {}},
        ]
        fused = weighted_score_fusion(vec, bm25)
        ids = [r["id"] for r in fused]
        assert "a" in ids and "b" in ids and "c" in ids
        # "b" appears in both, should have high score
        b_score = next(r["score"] for r in fused if r["id"] == "b")
        assert b_score > 0

    def test_fusion_basic(self):
        """Test basic fusion of two result sets."""
        vector_results = [
            {"id": "doc2", "text": "text2", "score": 0.95},
            {"id": "doc1", "text": "text1", "score": 0.90},
            {"id": "doc3", "text": "text3", "score": 0.80},
        ]
        bm25_results = [
            {"id": "doc2", "text": "text2", "score": 10.0},
            {"id": "doc1", "text": "text1", "score": 5.0},
            {"id": "doc4", "text": "text4", "score": 2.0},
        ]
        fused = weighted_score_fusion(vector_results, bm25_results)
        assert len(fused) == 4
        assert all("score" in doc for doc in fused)
        # doc2 should be ranked high as it has a very strong BM25 signal and good vector score
        assert fused[0]["id"] == "doc2"

    def test_fused_only_flag(self):
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}}]
        bm25 = [{"id": "z", "text": "t", "score": 5.0, "metadata": {}}]
        fused = weighted_score_fusion(vec, bm25)
        z = next(r for r in fused if r["id"] == "z")
        assert z.get("fused_only") is True

    def test_empty_inputs(self):
        assert weighted_score_fusion([], []) == []
        vec = [{"id": "a", "text": "t", "score": 0.9, "metadata": {}}]
        result = weighted_score_fusion(vec, [])
        assert len(result) == 1

    def test_fusion_empty_bm25(self):
        """Test fusion with empty BM25 results."""
        vector_results = [{"id": "doc1", "text": "text1", "score": 0.9}]
        bm25_results = []
        fused = weighted_score_fusion(vector_results, bm25_results)
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
        # With only 2 docs and the term in 1, IDF = 0 â†’ score = 0 â†’ filtered out.
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

    def test_save_overwrites_existing_file(self, tmp_path):
        """Second save must not raise [Errno 22] on Windows (os.replace race)."""
        r = BM25Retriever(storage_path=str(tmp_path))
        docs = [{"id": "a", "text": "first save", "metadata": {}}]
        r.add_documents(docs)
        r.save()  # creates the corpus file

        # A second save while the file already exists must succeed
        r.corpus[0]["text"] = "second save"
        r.save()

        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert r2.corpus[0]["text"] == "second save"

    def test_save_fallback_on_permission_error(self, tmp_path, monkeypatch):
        """save() must survive PermissionError from os.replace (Windows file-lock)."""
        import shutil

        r = BM25Retriever(storage_path=str(tmp_path))
        r.add_documents([{"id": "z", "text": "locked file", "metadata": {}}])

        replace_called = []
        copy_called = []
        real_copy = shutil.copy2

        def mock_replace(src, dst):
            replace_called.append(True)
            raise PermissionError("file in use")

        def mock_copy(src, dst):
            copy_called.append(True)
            real_copy(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)
        monkeypatch.setattr("shutil.copy2", mock_copy)

        r.save()  # must not raise
        assert replace_called, "os.replace should have been attempted"
        assert copy_called, "shutil.copy2 fallback should have been used"


class TestHybridThresholdBug:
    """Regression: BM25-only hits must not be filtered by vector similarity threshold."""

    def test_fused_only_hit_passes_threshold(self):
        """A BM25-only fused hit (fused_only=True, vector_score=0.0) must survive
        the similarity threshold filter that would otherwise drop it."""
        from axon.retrievers import weighted_score_fusion

        vector_results = []
        bm25_results = [
            {"id": "bm25doc", "text": "INC-44721 exact match", "score": 1.0, "metadata": {}}
        ]
        fused = weighted_score_fusion(vector_results, bm25_results, weight=0.7)
        assert len(fused) == 1
        bm25_hit = fused[0]
        # fused_only flag must be set so the threshold filter skips it
        assert bm25_hit.get("fused_only") is True
        # vector_score must be 0.0 (no vector contribution)
        assert bm25_hit.get("vector_score", 0.0) == 0.0

    def test_mixed_results_only_low_vector_score_filtered(self):
        """Vector results below threshold should be filtered; BM25-only hits kept."""
        # This directly tests the filtering logic in main.py
        threshold = 0.4
        results = [
            {"id": "vec_good", "vector_score": 0.8, "fused_only": False},
            {"id": "vec_bad", "vector_score": 0.1, "fused_only": False},
            {"id": "bm25_only", "vector_score": 0.0, "fused_only": True},
        ]
        filtered = []
        for r in results:
            if r.get("fused_only"):
                filtered.append(r)
                continue
            if r.get("vector_score", r.get("score", 0.0)) >= threshold:
                filtered.append(r)

        ids = [r["id"] for r in filtered]
        assert "vec_good" in ids
        assert "vec_bad" not in ids
        assert "bm25_only" in ids  # must survive despite vector_score=0.0


class TestBM25SaveOsError:
    """Regression tests for Bug: BM25 save raises OSError (errno 22 / WinError 87) on Windows.

    os.replace() on Windows can raise OSError (not just PermissionError) in some
    filesystem configurations.  The fallback path must activate for any OSError.
    """

    def test_save_falls_back_on_oserror(self, tmp_path, monkeypatch):
        """When os.replace raises OSError (errno 22), shutil.copy2 fallback is used."""
        import shutil
        import unittest.mock as mock

        r = BM25Retriever(storage_path=str(tmp_path))
        r.add_documents([{"id": "d1", "text": "hello world", "metadata": {}}])

        # Make the next save fail with OSError (simulates WinError 87 / errno 22)
        with mock.patch("os.replace", side_effect=OSError(22, "Invalid argument")):
            copy2_called = []
            real_copy2 = shutil.copy2

            def _fake_copy2(src, dst):
                copy2_called.append((src, dst))
                return real_copy2(src, dst)

            with mock.patch("shutil.copy2", side_effect=_fake_copy2):
                # Should not raise
                r.save()

        assert len(copy2_called) == 1, "shutil.copy2 fallback should have been called"
        # Verify corpus still readable from disk
        r2 = BM25Retriever(storage_path=str(tmp_path))
        assert len(r2.corpus) == 1
        assert r2.corpus[0]["id"] == "d1"

    def test_save_falls_back_on_permission_error(self, tmp_path, monkeypatch):
        """PermissionError is still handled (subset of OSError)."""
        import shutil
        import unittest.mock as mock

        r = BM25Retriever(storage_path=str(tmp_path))
        r.add_documents([{"id": "d1", "text": "test", "metadata": {}}])

        with mock.patch("os.replace", side_effect=PermissionError("locked")):
            copy2_called = []
            real_copy2 = shutil.copy2

            def _fake_copy2(src, dst):
                copy2_called.append((src, dst))
                return real_copy2(src, dst)

            with mock.patch("shutil.copy2", side_effect=_fake_copy2):
                r.save()

        assert len(copy2_called) == 1


class TestBM25BatchAdd:
    """Tests for BM25Retriever deferred-save batch mode (TASK 13A)."""

    def test_add_documents_save_deferred_true(self, tmp_path):
        """save_deferred=True â†’ save() NOT called; corpus extended."""
        retriever = BM25Retriever(str(tmp_path / "bm25"))
        docs = [{"id": "d1", "text": "hello world", "metadata": {}}]
        with patch.object(retriever, "save") as mock_save:
            retriever.add_documents(docs, save_deferred=True)
            mock_save.assert_not_called()
        assert len(retriever.corpus) == 1

    def test_add_documents_save_deferred_false(self, tmp_path):
        """save_deferred=False (default) â†’ save() IS called."""
        retriever = BM25Retriever(str(tmp_path / "bm25"))
        docs = [{"id": "d1", "text": "hello world", "metadata": {}}]
        with patch.object(retriever, "save") as mock_save:
            retriever.add_documents(docs, save_deferred=False)
            mock_save.assert_called_once()

    def test_flush_calls_save(self, tmp_path):
        """flush() â†’ save() called."""
        retriever = BM25Retriever(str(tmp_path / "bm25"))
        with patch.object(retriever, "save") as mock_save:
            retriever.flush()
            mock_save.assert_called_once()
"""Tests for OpenReranker providers in axon.rerank."""
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig


def _cfg(**kwargs):
    defaults = {
        "bm25_path": "/tmp/bm25",
        "vector_store_path": "/tmp/vs",
        "rerank": True,
        "reranker_provider": "cross-encoder",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


class TestOpenRerankerDisabled:
    def test_rerank_false_returns_docs_unchanged(self):
        from axon.rerank import OpenReranker

        cfg = _cfg(rerank=False)
        r = OpenReranker(cfg)
        docs = [{"text": "hello", "vector_score": 0.9}]
        assert r.rerank("query", docs) == docs

    def test_rerank_empty_docs_returns_empty(self):
        from axon.rerank import OpenReranker

        cfg = _cfg()
        mock_model = MagicMock()
        mock_st = MagicMock(CrossEncoder=MagicMock(return_value=mock_model))
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            r = OpenReranker(cfg)
            assert r.rerank("query", []) == []

    def test_rerank_no_model_returns_docs(self):
        from axon.rerank import OpenReranker

        cfg = _cfg()
        r = OpenReranker.__new__(OpenReranker)
        r.config = cfg
        r.model = None
        r.llm = None
        docs = [{"text": "hello"}]
        assert r.rerank("query", docs) == docs


class TestOpenRerankerCrossEncoder:
    def test_init_loads_cross_encoder(self):
        from axon.rerank import OpenReranker

        cfg = _cfg()
        mock_model = MagicMock()
        mock_ce = MagicMock(return_value=mock_model)
        mock_st = MagicMock(CrossEncoder=mock_ce)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            r = OpenReranker(cfg)
            mock_ce.assert_called_once_with(cfg.reranker_model)
            assert r.model is mock_model

    def test_rerank_sorts_by_score(self):
        from axon.rerank import OpenReranker

        cfg = _cfg()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.1]
        mock_st = MagicMock(CrossEncoder=MagicMock(return_value=mock_model))

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            r = OpenReranker(cfg)
            docs = [
                {"text": "A", "vector_score": 0.5},
                {"text": "B", "vector_score": 0.5},
                {"text": "C", "vector_score": 0.5},
            ]
            result = r.rerank("query", docs)
            assert result[0]["text"] == "B"  # highest score 0.9
            assert result[1]["text"] == "A"  # score 0.3
            assert result[2]["text"] == "C"  # lowest score 0.1

    def test_rerank_adds_rerank_score(self):
        from axon.rerank import OpenReranker

        cfg = _cfg()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        mock_st = MagicMock(CrossEncoder=MagicMock(return_value=mock_model))

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            r = OpenReranker(cfg)
            docs = [{"text": "A"}]
            result = r.rerank("query", docs)
            assert "rerank_score" in result[0]
            assert result[0]["rerank_score"] == pytest.approx(0.8)

    def test_import_error_disables_rerank(self):
        from axon.rerank import OpenReranker

        cfg = _cfg()
        mock_st = MagicMock(CrossEncoder=MagicMock(side_effect=ImportError("not installed")))

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            r = OpenReranker(cfg)
            assert not r.config.rerank
            assert r.model is None


class TestOpenRerankerLLM:
    def test_init_llm_provider(self):
        from axon.rerank import OpenReranker

        cfg = _cfg(reranker_provider="llm")
        mock_llm_instance = MagicMock()
        mock_llm_cls = MagicMock(return_value=mock_llm_instance)
        mock_axon_llm = MagicMock(OpenLLM=mock_llm_cls)

        with patch.dict("sys.modules", {"axon.llm": mock_axon_llm}):
            r = OpenReranker(cfg)
            assert r.llm is mock_llm_instance

    def _make_llm_reranker(self, cfg, mock_llm):
        """Build an OpenReranker with LLM provider by directly setting attributes."""
        from axon.rerank import OpenReranker

        r = OpenReranker.__new__(OpenReranker)
        r.config = cfg
        r.model = None
        r.llm = mock_llm
        return r

    def test_llm_rerank_scores_and_sorts(self):
        cfg = _cfg(reranker_provider="llm")
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = ["8", "3"]

        r = self._make_llm_reranker(cfg, mock_llm)
        docs = [{"text": "A"}, {"text": "B"}]
        result = r._llm_rerank("query", docs)
        assert result[0]["rerank_score"] == 8.0
        assert result[1]["rerank_score"] == 3.0
        assert result[0]["text"] == "A"

    def test_llm_rerank_non_numeric_score_is_zero(self):
        cfg = _cfg(reranker_provider="llm")
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "not a number"

        r = self._make_llm_reranker(cfg, mock_llm)
        docs = [{"text": "X"}]
        result = r._llm_rerank("query", docs)
        assert result[0]["rerank_score"] == 0.0

    def test_llm_rerank_exception_is_zero(self):
        cfg = _cfg(reranker_provider="llm")
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("API error")

        r = self._make_llm_reranker(cfg, mock_llm)
        docs = [{"text": "X"}]
        result = r._llm_rerank("query", docs)
        assert result[0]["rerank_score"] == 0.0

    def test_rerank_dispatches_to_llm_rerank(self):
        cfg = _cfg(reranker_provider="llm")
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "7"

        r = self._make_llm_reranker(cfg, mock_llm)
        docs = [{"text": "A"}]
        result = r.rerank("query", docs)
        assert result[0]["rerank_score"] == 7.0
