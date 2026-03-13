"""
tests/test_new_features.py

Tests for new features added in sprint: axon-integration-alternatives.
Covers _KNOWN_DIMS, dataset type detection, doc versions, hybrid_mode=rrf,
and health endpoint behaviour.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _KNOWN_DIMS
# ---------------------------------------------------------------------------


class TestKnownDims:
    def test_bge_large_dim(self):
        from axon.main import _KNOWN_DIMS

        assert _KNOWN_DIMS["BAAI/bge-large-en-v1.5"] == 1024

    def test_bge_small_dim(self):
        from axon.main import _KNOWN_DIMS

        assert _KNOWN_DIMS["BAAI/bge-small-en-v1.5"] == 384

    def test_minilm_dim(self):
        from axon.main import _KNOWN_DIMS

        assert _KNOWN_DIMS["all-MiniLM-L6-v2"] == 384

    def test_nomic_dim(self):
        from axon.main import _KNOWN_DIMS

        assert _KNOWN_DIMS["nomic-embed-text"] == 768

    def test_mxbai_dim(self):
        from axon.main import _KNOWN_DIMS

        assert _KNOWN_DIMS["mxbai-embed-large"] == 1024

    def test_ollama_known_model_uses_known_dims(self):
        """Ollama provider dimension should use _KNOWN_DIMS lookup."""
        from axon.main import AxonConfig, OpenEmbedding

        config = AxonConfig(embedding_provider="ollama", embedding_model="nomic-embed-text")
        emb = OpenEmbedding.__new__(OpenEmbedding)
        emb.config = config
        emb.provider = "ollama"
        emb.model = None
        emb.dimension = 0
        emb._load_model()
        assert emb.dimension == 768

    def test_ollama_unknown_model_defaults_768(self):
        """Ollama provider with unknown model should default to 768."""
        from axon.main import AxonConfig, OpenEmbedding

        config = AxonConfig(embedding_provider="ollama", embedding_model="some-unknown-model-xyz")
        emb = OpenEmbedding.__new__(OpenEmbedding)
        emb.config = config
        emb.provider = "ollama"
        emb.model = None
        emb.dimension = 0
        emb._load_model()
        assert emb.dimension == 768


# ---------------------------------------------------------------------------
# AxonConfig defaults
# ---------------------------------------------------------------------------


class TestAxonConfigNewDefaults:
    def test_embedding_provider_default(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.embedding_provider == "sentence_transformers"
        assert cfg.embedding_model == "all-MiniLM-L6-v2"

    def test_max_workers_default(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.max_workers == 8

    def test_hybrid_mode_default(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.hybrid_mode == "weighted"

    def test_dataset_type_default(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.dataset_type == "auto"

    def test_smart_ingest_default(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.smart_ingest is False

    def test_qdrant_url_default_empty(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.qdrant_url == ""
        assert cfg.qdrant_api_key == ""


# ---------------------------------------------------------------------------
# Dataset type detection
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestDetectDatasetType:
    def _brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        return AxonBrain(AxonConfig(hybrid_search=False, rerank=False))

    def test_code_extension(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        doc = {"id": "x", "text": "def foo(): pass", "metadata": {"source": "app.py"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "codebase"
        assert has_code is False

    def test_ts_extension(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        doc = {"id": "x", "text": "const x = 1", "metadata": {"source": "index.ts"}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "codebase"

    def test_discussion_json_list(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        import json

        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        data = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        doc = {"id": "x", "text": json.dumps(data), "metadata": {}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "discussion"

    def test_discussion_json_dict(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        import json

        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        data = {"messages": [{"role": "user", "content": "hi"}]}
        doc = {"id": "x", "text": json.dumps(data), "metadata": {}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "discussion"

    def test_tabular_csv(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        rows = ["a,b,c,d,e", "1,2,3,4,5", "6,7,8,9,10", "11,12,13,14,15"]
        doc = {"id": "x", "text": "\n".join(rows), "metadata": {}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "knowledge"

    def test_paper_signals(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        text = "Abstract\nThis is a paper. Introduction\nSection 1. Conclusion\nDOI: 10.1000/x"
        doc = {"id": "x", "text": text, "metadata": {}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "paper"

    def test_markdown_doc(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        text = "# Title\n\nStep 1: do this\n\n## Section\n\nNote: important"
        doc = {"id": "x", "text": text, "metadata": {}}
        dtype, _ = brain._detect_dataset_type(doc)
        assert dtype == "doc"

    def test_empty_text_returns_doc(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        doc = {"id": "x", "text": "", "metadata": {}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "doc"

    def test_configured_override(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, dataset_type="paper")
        brain = AxonBrain(config)
        doc = {"id": "x", "text": "def foo(): pass", "metadata": {"source": "app.py"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "paper"
        assert has_code is False


# ---------------------------------------------------------------------------
# Doc versions
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestDocVersions:
    def test_get_doc_versions_returns_dict(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig(hybrid_search=False, rerank=False))
        assert isinstance(brain.get_doc_versions(), dict)

    def test_load_doc_versions_empty_when_no_file(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, bm25_path=str(tmp_path / "bm25"))
        brain = AxonBrain(config)
        assert brain._doc_versions == {}

    def test_save_and_reload_doc_versions(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        from axon.main import AxonBrain, AxonConfig

        bm25_dir = tmp_path / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        config = AxonConfig(hybrid_search=False, rerank=False, bm25_path=str(bm25_dir))
        brain = AxonBrain(config)
        brain._doc_versions = {"file.txt": {"content_hash": "deadbeef"}}
        brain._save_doc_versions()

        brain2 = AxonBrain(config)
        assert brain2._doc_versions.get("file.txt", {}).get("content_hash") == "deadbeef"


# ---------------------------------------------------------------------------
# Health endpoint (503/200)
# ---------------------------------------------------------------------------


from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

_client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_brain_for_health():
    """Isolate brain state for each health test."""
    original = api_module.brain
    yield
    api_module.brain = original


def test_health_returns_503_when_brain_none():
    api_module.brain = None
    resp = _client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "initializing"


def test_health_returns_200_when_brain_initialized():
    mock_brain = MagicMock()
    mock_brain.config.project = "default"
    api_module.brain = mock_brain
    resp = _client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Hybrid mode RRF
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestHybridModeRRF:
    def test_rrf_mode_calls_rrf_function(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        import axon.retrievers as _ret
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=True, hybrid_mode="rrf", rerank=False, similarity_threshold=0.0
        )
        brain = AxonBrain(config)
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9, "metadata": {}}]
        )
        brain.bm25.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 5.0, "metadata": {}}]
        )
        with patch.object(
            _ret, "reciprocal_rank_fusion", wraps=_ret.reciprocal_rank_fusion
        ) as mock_rrf:
            brain._execute_retrieval("test query")
            assert mock_rrf.called

    def test_weighted_mode_calls_weighted_function(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        import axon.retrievers as _ret
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=True, hybrid_mode="weighted", rerank=False, similarity_threshold=0.0
        )
        brain = AxonBrain(config)
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9, "metadata": {}}]
        )
        brain.bm25.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 5.0, "metadata": {}}]
        )
        with patch.object(
            _ret, "weighted_score_fusion", wraps=_ret.weighted_score_fusion
        ) as mock_wsf:
            brain._execute_retrieval("test query")
            assert mock_wsf.called


# ---------------------------------------------------------------------------
# /docs and /ingest/refresh endpoints
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_api_brain():
    original = api_module.brain
    api_module.brain = None
    yield
    api_module.brain = original


def test_tracked_docs_endpoint_503_when_no_brain():
    api_module.brain = None
    resp = _client.get("/tracked-docs")
    assert resp.status_code == 503


def test_tracked_docs_endpoint_returns_dict_when_brain_set():
    mock_brain = MagicMock()
    mock_brain.get_doc_versions.return_value = {"file.txt": {"content_hash": "abc"}}
    api_module.brain = mock_brain
    resp = _client.get("/tracked-docs")
    assert resp.status_code == 200
    assert "docs" in resp.json()


def test_ingest_refresh_503_when_no_brain():
    api_module.brain = None
    resp = _client.post("/ingest/refresh")
    assert resp.status_code == 503


def test_ingest_refresh_returns_results_when_brain_set(tmp_path):
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hello world", encoding="utf-8")

    mock_brain = MagicMock()
    mock_brain.get_doc_versions.return_value = {}
    api_module.brain = mock_brain

    resp = _client.post("/ingest/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert "skipped" in data
    assert "reingest_needed" in data
    assert "missing" in data
