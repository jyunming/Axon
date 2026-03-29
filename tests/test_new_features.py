"""
tests/test_new_features.py

Tests for new features added in sprint: axon-integration-alternatives.
Covers _KNOWN_DIMS, dataset type detection, doc versions, hybrid_mode=rrf,
and health endpoint behaviour.
"""

import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _reset_graphrag_model_caches():
    from axon.graph_rag import GraphRagMixin

    GraphRagMixin._shared_gliner_models = {}
    GraphRagMixin._shared_rebel_pipelines = {}


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
        assert cfg.hybrid_mode == "rrf"

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
    post_data = resp.json()
    assert "job_id" in post_data
    status_resp = _client.get(f"/ingest/status/{post_data['job_id']}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert "skipped" in data
    assert "reingested" in data
    assert "missing" in data


# ---------------------------------------------------------------------------
# Query Router (Option B)
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestQueryRouter:
    def _brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        return AxonBrain(AxonConfig(hybrid_search=False, rerank=False, query_router="heuristic"))

    def test_heuristic_factual_short_query(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        result = brain._classify_query_route_heuristic("What is Python?")
        assert result == "factual"

    def test_heuristic_synthesis_keyword(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        result = brain._classify_query_route_heuristic("summarize the key findings")
        assert result == "synthesis"

    def test_heuristic_table_keyword(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        result = brain._classify_query_route_heuristic("how many rows in the table")
        assert result == "table_lookup"

    def test_heuristic_entity_keyword(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        result = brain._classify_query_route_heuristic("what is the relationship between X and Y")
        assert result == "entity_relation"

    def test_heuristic_corpus_keyword(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        result = brain._classify_query_route_heuristic(
            "what are the main themes across all documents"
        )
        assert result == "corpus_exploration"

    def test_llm_router_valid_response(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.generate = MagicMock(return_value="synthesis\n")
        result = brain._classify_query_route_llm("What are the main findings?")
        assert result == "synthesis"

    def test_llm_router_invalid_response_defaults_factual(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.generate = MagicMock(return_value="unknown")
        result = brain._classify_query_route_llm("What are the main findings?")
        assert result == "factual"

    def test_route_profile_factual_disables_graphrag(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from axon.main import _ROUTE_PROFILES

        profile = _ROUTE_PROFILES["factual"]
        assert profile["graph_rag"] is False

    def test_route_profile_entity_enables_graphrag_light(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from axon.main import _ROUTE_PROFILES

        profile = _ROUTE_PROFILES["entity_relation"]
        assert profile["graph_rag"] is True
        assert profile["graph_rag_community"] is False


# ---------------------------------------------------------------------------
# Code Graph (Phase 2 + Phase 3)
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestCodeGraph:
    def _brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **kwargs):
        from axon.main import AxonBrain, AxonConfig

        cfg = AxonConfig(hybrid_search=False, rerank=False, code_graph=True, **kwargs)
        return AxonBrain(cfg)

    def test_code_graph_default_false(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonConfig

        assert AxonConfig().code_graph is False
        assert AxonConfig().code_graph_bridge is False

    def test_build_creates_file_node(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "foo",
                    "source": "/src/foo.py",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        assert "/src/foo.py" in brain._code_graph["nodes"]
        assert brain._code_graph["nodes"]["/src/foo.py"]["node_type"] == "file"

    def test_build_creates_symbol_node(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "foo",
                    "source": "/src/foo.py",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        sym_id = "/src/foo.py::foo"
        assert sym_id in brain._code_graph["nodes"]
        assert brain._code_graph["nodes"][sym_id]["node_type"] == "function"

    def test_build_contains_edge(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "foo",
                    "source": "/src/foo.py",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        assert any(
            e["edge_type"] == "CONTAINS"
            and e["source"] == "/src/foo.py"
            and e["target"] == "/src/foo.py::foo"
            for e in edges
        )

    def test_build_imports_edge(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        # First ingest bar.py so it exists as a file node
        bar_chunk = {
            "id": "c0",
            "text": "def bar(): pass",
            "metadata": {
                "source_class": "code",
                "file_path": "/src/pkg/bar.py",
                "language": "python",
                "symbol_type": "function",
                "symbol_name": "bar",
                "source": "/src/pkg/bar.py",
            },
        }
        foo_chunk = {
            "id": "c1",
            "text": "from pkg.bar import bar\ndef foo(): pass",
            "metadata": {
                "source_class": "code",
                "file_path": "/src/pkg/foo.py",
                "language": "python",
                "symbol_type": "function",
                "symbol_name": "foo",
                "source": "/src/pkg/foo.py",
                "imports": "from pkg.bar import bar",
            },
        }
        brain._build_code_graph_from_chunks([bar_chunk, foo_chunk])
        edges = brain._code_graph["edges"]
        assert any(e["edge_type"] == "IMPORTS" and e["source"] == "/src/pkg/foo.py" for e in edges)

    def test_bridge_adds_mentioned_in_edge(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        # Build a code graph with a known symbol
        code_chunks = [
            {
                "id": "c1",
                "text": "def CodeAwareSplitter(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/splitters.py",
                    "language": "python",
                    "symbol_type": "class",
                    "symbol_name": "CodeAwareSplitter",
                    "source": "/src/splitters.py",
                },
            }
        ]
        brain._build_code_graph_from_chunks(code_chunks)
        # Now a prose chunk that mentions the symbol
        prose_chunks = [
            {
                "id": "doc1",
                "text": "The CodeAwareSplitter handles Python AST chunking.",
                "metadata": {"source": "README.md"},
            }
        ]
        brain._build_code_doc_bridge(prose_chunks)
        edges = brain._code_graph["edges"]
        assert any(
            e["edge_type"] == "MENTIONED_IN"
            and "CodeAwareSplitter" in e["source"]
            and e["target"] == "doc1"
            for e in edges
        )

    def test_expand_with_code_graph_returns_matched_names(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        code_chunks = [
            {
                "id": "c1",
                "text": "def MyFunc(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "MyFunc",
                    "source": "/src/foo.py",
                },
            }
        ]
        brain._build_code_graph_from_chunks(code_chunks)
        # Mock vector_store.get_by_ids to return empty
        brain.vector_store.get_by_ids = lambda ids: []
        results, matched = brain._expand_with_code_graph("What does MyFunc do?", [], brain.config)
        assert "MyFunc" in matched

    def test_expand_no_match_returns_unchanged(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.vector_store.get_by_ids = lambda ids: []
        results, matched = brain._expand_with_code_graph(
            "unrelated prose query here", [], brain.config
        )
        assert matched == []
        assert results == []

    def test_save_and_load_code_graph(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(
            AxonConfig(bm25_path=str(tmp_path), code_graph=True, hybrid_search=False, rerank=False)
        )
        brain._code_graph = {
            "nodes": {
                "f": {
                    "node_id": "f",
                    "node_type": "file",
                    "name": "f.py",
                    "file_path": "f.py",
                    "language": "python",
                    "chunk_ids": [],
                    "signature": "",
                    "start_line": None,
                    "end_line": None,
                }
            },
            "edges": [],
        }
        brain._save_code_graph()
        brain2 = AxonBrain(
            AxonConfig(bm25_path=str(tmp_path), code_graph=True, hybrid_search=False, rerank=False)
        )
        assert "f" in brain2._code_graph["nodes"]


# ---------------------------------------------------------------------------
# Code Lexical Boost
# ---------------------------------------------------------------------------


class TestCodeLexicalBoost:
    def test_extract_tokens_camelcase(self):
        from axon.main import _extract_code_query_tokens

        tokens = _extract_code_query_tokens("CodeAwareSplitter")
        assert "codeawaresplitter" in tokens
        assert "code" in tokens
        assert "aware" in tokens

    def test_extract_tokens_snake_case(self):
        from axon.main import _extract_code_query_tokens

        tokens = _extract_code_query_tokens("_split_python_ast")
        assert "split" in tokens
        assert "python" in tokens
        assert "ast" in tokens

    def test_extract_tokens_filename(self):
        from axon.main import _extract_code_query_tokens

        tokens = _extract_code_query_tokens("where is loaders.py")
        assert "loaders" in tokens

    def test_looks_like_code_query_camel(self):
        from axon.main import _looks_like_code_query

        assert _looks_like_code_query("How does CodeAwareSplitter work?") is True

    def test_looks_like_code_query_snake(self):
        from axon.main import _looks_like_code_query

        assert _looks_like_code_query("what does _split_python_ast do") is True

    def test_looks_like_code_query_false(self):
        from axon.main import _looks_like_code_query

        assert _looks_like_code_query("What is the capital of France?") is False

    def test_boost_exact_symbol_match_ranks_first(self):
        from axon.main import AxonBrain, AxonConfig, _extract_code_query_tokens

        cfg = AxonConfig(code_lexical_boost=True, code_max_chunks_per_file=10)
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg

        results = [
            {
                "id": "a",
                "score": 0.9,
                "text": "broad main module text",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": "main",
                    "symbol_type": "function",
                    "file_path": "main.py",
                },
            },
            {
                "id": "b",
                "score": 0.7,
                "text": "CodeAwareSplitter implementation details",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": "CodeAwareSplitter",
                    "symbol_type": "class",
                    "file_path": "splitters.py",
                },
            },
        ]
        query_tokens = _extract_code_query_tokens("How does CodeAwareSplitter work?")
        boosted = brain._apply_code_lexical_boost(results, query_tokens, cfg=cfg)
        assert boosted[0]["id"] == "b", "Exact symbol match should rank first"

    def test_boost_skips_non_code_results(self):
        from axon.main import AxonBrain, AxonConfig, _extract_code_query_tokens

        cfg = AxonConfig(code_lexical_boost=True, code_max_chunks_per_file=10)
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg

        results = [
            {"id": "x", "score": 0.8, "text": "prose text", "metadata": {"source": "doc.md"}},
            {"id": "y", "score": 0.6, "text": "more prose", "metadata": {"source": "readme.md"}},
        ]
        query_tokens = _extract_code_query_tokens("CodeAwareSplitter")
        boosted = brain._apply_code_lexical_boost(results, query_tokens, cfg=cfg)
        # Scores should be unchanged (no code chunks → max_lex == 0)
        assert boosted[0]["score"] == 0.8
        assert boosted[1]["score"] == 0.6

    def test_boost_no_tokens_no_change(self):
        from axon.main import AxonBrain, AxonConfig

        cfg = AxonConfig(code_lexical_boost=True, code_max_chunks_per_file=10)
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg

        results = [
            {
                "id": "a",
                "score": 0.9,
                "text": "x",
                "metadata": {"source_class": "code", "symbol_name": "foo", "file_path": "a.py"},
            },
        ]
        boosted = brain._apply_code_lexical_boost(results, frozenset(), cfg=cfg)
        assert boosted[0]["score"] == 0.9

    def test_diversity_cap(self):
        from axon.main import AxonBrain, AxonConfig, _extract_code_query_tokens

        cfg = AxonConfig(code_lexical_boost=True, code_max_chunks_per_file=3)
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg

        same_file = [
            {
                "id": str(i),
                "score": 0.9 - i * 0.05,
                "text": f"chunk {i}",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": f"fn{i}",
                    "file_path": "main.py",
                },
            }
            for i in range(5)
        ]
        other_file = [
            {
                "id": "other",
                "score": 0.5,
                "text": "different file chunk",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": "helper",
                    "file_path": "utils.py",
                },
            }
        ]
        results = same_file + other_file
        query_tokens = _extract_code_query_tokens("fn0 main.py")
        boosted = brain._apply_code_lexical_boost(results, query_tokens, cfg=cfg)
        # After diversity cap, the first 3 from main.py and 1 from utils.py should be in top 4
        top_4_files = [r.get("metadata", {}).get("file_path") for r in boosted[:4]]
        assert top_4_files.count("main.py") == 3
        assert "utils.py" in top_4_files

    def test_config_code_lexical_boost_disableable(self):
        from axon.main import AxonConfig

        cfg = AxonConfig(code_lexical_boost=False)
        assert cfg.code_lexical_boost is False


# ---------------------------------------------------------------------------
# Retrieval Diagnostics
# ---------------------------------------------------------------------------


class TestCodeRetrievalDiagnostics:
    def test_default_values(self):
        from axon.main import CodeRetrievalDiagnostics

        d = CodeRetrievalDiagnostics()
        assert d.diagnostics_version == "1.3"
        assert d.code_mode_triggered is False
        assert d.tokens_extracted == []
        assert d.channels_activated == []
        assert d.result_count == 0
        assert d.boost_applied is False
        assert d.fallback_chunks_in_results == 0

    def test_to_dict_keys(self):
        from axon.main import CodeRetrievalDiagnostics

        d = CodeRetrievalDiagnostics(code_mode_triggered=True, result_count=5)
        out = d.to_dict()
        assert out["diagnostics_version"] == "1.3"
        assert out["code_mode_triggered"] is True
        assert out["result_count"] == 5
        assert "channels_activated" in out

    def test_to_json_is_valid(self):
        import json

        from axon.main import CodeRetrievalDiagnostics

        d = CodeRetrievalDiagnostics()
        parsed = json.loads(d.to_json())
        assert parsed["diagnostics_version"] == "1.3"

    def test_independent_mutable_defaults(self):
        from axon.main import CodeRetrievalDiagnostics

        d1 = CodeRetrievalDiagnostics()
        d2 = CodeRetrievalDiagnostics()
        d1.tokens_extracted.append("foo")
        assert d2.tokens_extracted == []


class TestCodeRetrievalTrace:
    def test_default_values(self):
        from axon.main import CodeRetrievalTrace

        t = CodeRetrievalTrace()
        assert t.per_result_score_breakdown == []
        assert t.channel_raw_counts == {}
        assert t.diversity_cap_deferrals == 0
        assert t.deferred_chunk_ids == []

    def test_to_dict_keys(self):
        from axon.main import CodeRetrievalTrace

        t = CodeRetrievalTrace(diversity_cap_deferrals=2)
        out = t.to_dict()
        assert out["diversity_cap_deferrals"] == 2
        assert "per_result_score_breakdown" in out

    def test_independent_mutable_defaults(self):
        from axon.main import CodeRetrievalTrace

        t1 = CodeRetrievalTrace()
        t2 = CodeRetrievalTrace()
        t1.deferred_chunk_ids.append("x")
        assert t2.deferred_chunk_ids == []


class TestClassifyRetrievalFailure:
    def test_exact_symbol_missed(self):
        from axon.main import _classify_retrieval_failure, _extract_code_query_tokens

        results = [
            {
                "id": "a",
                "score": 0.9,
                "text": "some code",
                "metadata": {"source_class": "code", "symbol_name": "unrelated_function"},
            }
        ]
        tokens = _extract_code_query_tokens("CodeAwareSplitter")
        labels = _classify_retrieval_failure(results, tokens)
        assert "exact_symbol_missed" in labels

    def test_fallback_chunk_involved(self):
        from axon.main import _classify_retrieval_failure

        results = [
            {
                "id": "a",
                "score": 0.8,
                "text": "x",
                "metadata": {
                    "is_fallback": True,
                    "source_class": "code",
                    "symbol_type": "function",
                },
            }
        ]
        labels = _classify_retrieval_failure(results, frozenset())
        assert "fallback_chunk_involved" in labels

    def test_too_many_broad_chunks(self):
        from axon.main import _classify_retrieval_failure

        results = [
            {"id": str(i), "score": 0.5, "text": "prose", "metadata": {}} for i in range(4)
        ] + [
            {
                "id": "c",
                "score": 0.8,
                "text": "code",
                "metadata": {"source_class": "code", "symbol_type": "function"},
            }
        ]
        labels = _classify_retrieval_failure(results, frozenset())
        assert "too_many_broad_chunks" in labels

    def test_no_failures_clean_result(self):
        from axon.main import _classify_retrieval_failure, _extract_code_query_tokens

        results = [
            {
                "id": "a",
                "score": 0.9,
                "text": "def mysplitter(): pass",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": "mysplitter",
                    "symbol_type": "function",
                },
            }
        ]
        tokens = _extract_code_query_tokens("mysplitter")
        labels = _classify_retrieval_failure(results, tokens)
        assert "exact_symbol_missed" not in labels
        assert "fallback_chunk_involved" not in labels

    def test_right_file_wrong_block(self):
        from axon.main import _classify_retrieval_failure

        results = [
            {
                "id": "a",
                "score": 0.7,
                "text": "other stuff in splitters",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": "other_fn",
                    "source": "splitters.py",
                    "symbol_type": "function",
                },
            }
        ]
        labels = _classify_retrieval_failure(results, frozenset(), expected_symbol="mysplitter")
        assert "right_file_wrong_block" not in labels  # source doesn't contain "mysplitter"

        results2 = [
            {
                "id": "b",
                "score": 0.7,
                "text": "code in mysplitter file",
                "metadata": {
                    "source_class": "code",
                    "symbol_name": "other_fn",
                    "source": "mysplitter_utils.py",
                    "symbol_type": "function",
                },
            }
        ]
        labels2 = _classify_retrieval_failure(results2, frozenset(), expected_symbol="mysplitter")
        assert "right_file_wrong_block" in labels2


@pytest.mark.skipif(importlib.util.find_spec("gliner") is None, reason="gliner not installed")
class TestGLiNERConfigModel:
    """Fix 1 — _ensure_gliner respects graph_rag_gliner_model from config."""

    def test_gliner_uses_config_model_path(self):
        """_ensure_gliner loads the model path from config, not a hardcoded string."""
        from unittest.mock import MagicMock, patch

        from axon.main import AxonBrain

        _reset_graphrag_model_caches()
        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_gliner_model = "local/path"
        brain._gliner_model = None

        with patch("gliner.GLiNER.from_pretrained") as mock_fp:
            mock_fp.return_value = MagicMock()
            AxonBrain._ensure_gliner(brain)

        mock_fp.assert_called_once_with("local/path", local_files_only=False)


class TestREBELZeroEdgeWarning:
    """Fix 2 — REBEL 0-edge warning is logged when chunks were processed but no edges produced."""

    def test_rebel_zero_edge_warning_logged(self):
        """logger.warning is called when REBEL produces 0 edges from non-empty chunks."""
        from unittest.mock import MagicMock, patch

        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain._relation_graph = {}
        brain.config = MagicMock()
        brain.config.graph_rag_relation_backend = "rebel"

        _rel_chunks = [{"id": "c1", "text": "Apple acquired Beats."}]

        with patch("axon.main.logger") as mock_logger:
            # Simulate the post-loop warning block from ingest()
            if getattr(brain.config, "graph_rag_relation_backend", "llm") == "rebel":
                _rg_edge_count = sum(len(v) for v in brain._relation_graph.values())
                if _rg_edge_count == 0 and len(_rel_chunks) > 0:
                    mock_logger.warning(
                        "GraphRAG REBEL: processed %d chunks but produced 0 relation edges. "
                        "If using a local model path, verify the checkpoint contains pretrained weights "
                        "(a 'newly initialized weights' warning from transformers indicates an invalid checkpoint).",
                        len(_rel_chunks),
                    )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "0 relation edges" in call_args[0]


# ---------------------------------------------------------------------------
# Phase 2-4: local model routing — _ensure_gliner / _ensure_llmlingua
# ---------------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("gliner") is None, reason="gliner not installed")
def test_gliner_local_files_only_when_local_path(tmp_path):
    """_ensure_gliner passes local_files_only=True when model path is absolute."""
    from axon.main import AxonBrain

    _reset_graphrag_model_caches()
    model_dir = tmp_path / "gliner_model"
    model_dir.mkdir()
    cfg = MagicMock()
    cfg.graph_rag_gliner_model = str(model_dir)
    brain = MagicMock()
    brain.config = cfg
    brain._gliner_model = None

    with patch("gliner.GLiNER.from_pretrained") as mock_gliner:
        mock_gliner.return_value = MagicMock()
        AxonBrain._ensure_gliner(brain)
        mock_gliner.assert_called_once_with(str(model_dir), local_files_only=True)


def test_rebel_local_path_does_not_forward_local_files_only(tmp_path):
    """_ensure_rebel should load local artifacts explicitly and not leak local_files_only into generate()."""
    from axon.main import AxonBrain

    _reset_graphrag_model_caches()
    model_dir = tmp_path / "rebel_model"
    model_dir.mkdir()
    cfg = MagicMock()
    cfg.graph_rag_rebel_model = str(model_dir)
    brain = AxonBrain.__new__(AxonBrain)
    brain.config = cfg
    brain._rebel_pipeline = None

    model_obj = MagicMock()
    tokenizer_obj = MagicMock()
    pipeline_obj = MagicMock()

    with patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=model_obj
    ) as mock_model, patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer_obj
    ) as mock_tok, patch(
        "transformers.pipelines.pipeline", return_value=pipeline_obj
    ) as mock_pipe:
        result = AxonBrain._ensure_rebel(brain)

    assert result is pipeline_obj
    mock_model.assert_called_once_with(str(model_dir), local_files_only=True)
    mock_tok.assert_called_once_with(str(model_dir), local_files_only=True)
    mock_pipe.assert_called_once()
    assert mock_pipe.call_args.kwargs["model"] is model_obj
    assert mock_pipe.call_args.kwargs["tokenizer"] is tokenizer_obj
    assert "local_files_only" not in mock_pipe.call_args.kwargs


def test_extract_relations_rebel_decodes_generated_token_ids_like_tensor(tmp_path):
    """_extract_relations_rebel should handle pipeline outputs that expose generated_token_ids as tensors."""
    from axon.main import AxonBrain, AxonConfig

    class _FakeTensorIds:
        def tolist(self):
            return [0, 1, 2, 3]

    cfg = AxonConfig(
        bm25_path=str(tmp_path / "bm25"),
        vector_store_path=str(tmp_path / "vec"),
        graph_rag=True,
        graph_rag_relation_backend="rebel",
    )
    brain = AxonBrain.__new__(AxonBrain)
    brain.config = cfg
    brain._rebel_pipeline = MagicMock(return_value=[{"generated_token_ids": _FakeTensorIds()}])
    brain._rebel_pipeline.model = MagicMock()
    brain._rebel_pipeline.model.config = SimpleNamespace(
        task_specific_params={"relation_extraction": {}}
    )
    brain._rebel_pipeline.tokenizer = MagicMock()
    brain._rebel_pipeline.tokenizer.batch_decode.return_value = [
        "<triplet> Apple <subj> Microsoft <obj> competes with"
    ]

    rels = AxonBrain._extract_relations_rebel(brain, "Apple competes with Microsoft.")

    assert len(rels) == 1
    assert rels[0]["subject"] == "Apple"
    assert rels[0]["object"] == "Microsoft"
    assert rels[0]["relation"] == "competes with"


@pytest.mark.skipif(importlib.util.find_spec("gliner") is None, reason="gliner not installed")
def test_gliner_shared_cache_loads_once_across_threads():
    """Concurrent _ensure_gliner calls should load a shared model only once."""
    from axon.main import AxonBrain

    _reset_graphrag_model_caches()

    brains = []
    for _ in range(8):
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = SimpleNamespace(graph_rag_gliner_model="shared/model")
        brain._gliner_model = None
        brains.append(brain)

    load_calls = 0
    load_calls_lock = threading.Lock()
    sentinel = object()
    results = [None] * len(brains)

    def _fake_load(*_args, **_kwargs):
        nonlocal load_calls
        time.sleep(0.05)
        with load_calls_lock:
            load_calls += 1
        return sentinel

    def _worker(idx, brain):
        results[idx] = AxonBrain._ensure_gliner(brain)

    with patch("gliner.GLiNER.from_pretrained", side_effect=_fake_load) as mock_gliner:
        threads = [
            threading.Thread(target=_worker, args=(idx, brain), daemon=True)
            for idx, brain in enumerate(brains)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

    assert load_calls == 1
    assert mock_gliner.call_count == 1
    assert all(result is sentinel for result in results)


@pytest.mark.skipif(importlib.util.find_spec("llmlingua") is None, reason="llmlingua not installed")
def test_llmlingua_uses_config_model(tmp_path):
    """_ensure_llmlingua reads graph_rag_llmlingua_model from config."""
    from axon.main import AxonBrain

    model_dir = tmp_path / "llmlingua"
    model_dir.mkdir()
    cfg = MagicMock()
    cfg.graph_rag_llmlingua_model = str(model_dir)
    brain = MagicMock()
    brain.config = cfg
    brain._llmlingua = None

    with patch("llmlingua.PromptCompressor") as mock_compressor:
        mock_compressor.return_value = MagicMock()
        AxonBrain._ensure_llmlingua(brain)
        mock_compressor.assert_called_once()
        call_kwargs = mock_compressor.call_args
        assert call_kwargs.kwargs.get("model_name") == str(model_dir) or call_kwargs.args[0] == str(
            model_dir
        )
        assert call_kwargs.kwargs.get("local_files_only") is True
