"""Tests for OpenReranker providers in axon.rerank."""
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig


def _cfg(**kwargs):
    defaults = dict(
        bm25_path="/tmp/bm25",
        vector_store_path="/tmp/vs",
        rerank=True,
        reranker_provider="cross-encoder",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
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
