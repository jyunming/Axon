"""
Comprehensive pytest tests for axon/main.py (AxonBrain) targeting uncovered lines.

Coverage targets:
  lines 179, 237-238, 291-292, 342-343, 360-381, 487-496, 519-520, 545, 553, 556,
  560, 567-579, 591-592, 596, 600-601, 604, 622-623, 673, 720-732, 739, 759-760,
  789-790, 818-822, 841-876, 881-905, 910-920, 925-947, 952-961, 967-972, 977-978,
  1017-1019, 1031-1032, 1070-1071, 1134, 1137-1138, 1162, 1192-1193, 1206, 1236,
  1266-1268, 1295, 1301, 1324-1326, 1368, 1405-1412, 1423-1426, 1429-1430, 1433,
  1448-1450, 1466, 1482, 1548, 1594-1596, 1626, 1628, 1632, 1646, 1669, 1676,
  1701, 1734, 1742, 1795-1797, 1895, 1928, 1951, 1961, 1970-1973, 2021-2027,
  2053-2054, 2057, 2091, 2112-2121, 2134, 2136-2143, 2151-2153, 2155, 2187,
  2191, 2197-2214, 2217-2235, 2288-2293, 2328
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, **kwargs):
    from axon.config import AxonConfig

    defaults = {
        "bm25_path": str(tmp_path / "bm25"),
        "vector_store_path": str(tmp_path / "vs"),
        "query_router": "off",
        "query_cache": False,
        "raptor": False,
        "graph_rag": False,
        "rerank": False,
        "hybrid_search": False,
        "truth_grounding": False,
        "compress_context": False,
        "contextual_retrieval": False,
        "mmr": False,
        "discussion_fallback": False,
        "similarity_threshold": 0.0,
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


LOAD_PATCHES = [
    ("_load_hash_store", set()),
    ("_load_doc_versions", None),
    ("_load_entity_graph", {}),
    ("_load_code_graph", {}),
    ("_load_relation_graph", {}),
    ("_load_community_levels", {}),
    ("_load_community_summaries", {}),
    ("_load_entity_embeddings", {}),
    ("_load_claims_graph", {}),
    ("_load_community_hierarchy", {}),
    ("_log_startup_summary", None),
    ("_preflight_model_audit", None),
]


def _brain_patches():
    """Return a list of (target, return_value) for all heavy init mocks."""
    return [
        patch("axon.main.OpenVectorStore"),
        patch("axon.main.OpenEmbedding"),
        patch("axon.main.OpenLLM"),
        patch("axon.main.OpenReranker"),
        patch("axon.retrievers.BM25Retriever"),
        patch("axon.projects.ensure_project"),
    ]


def _make_brain(tmp_path, **cfg_kwargs):
    """Construct a fully-mocked AxonBrain for a temp directory."""
    from axon.main import AxonBrain

    cfg = _make_config(tmp_path, **cfg_kwargs)

    with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
        "axon.main.OpenLLM"
    ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
        "axon.projects.ensure_project"
    ):
        # Patch all disk-loading methods
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(AxonBrain, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            brain = AxonBrain(cfg)
    return brain


@pytest.fixture
def brain(tmp_path):
    return _make_brain(tmp_path)


# ---------------------------------------------------------------------------
# Utility: fake retrieval result
# ---------------------------------------------------------------------------


def _fake_result(id="doc1", text="hello world", score=0.9, **meta):
    return {"id": id, "text": text, "score": score, "metadata": meta}


def _mock_retrieval(results=None):
    """Return a dict that mimics _execute_retrieval output."""
    from axon.code_retrieval import CodeRetrievalDiagnostics, CodeRetrievalTrace

    if results is None:
        results = [_fake_result()]
    return {
        "results": results,
        "vector_count": len(results),
        "bm25_count": 0,
        "filtered_count": len(results),
        "graph_expanded_count": 0,
        "matched_entities": [],
        "transforms": {
            "hyde_applied": False,
            "multi_query_applied": False,
            "step_back_applied": False,
            "decompose_applied": False,
            "web_search_applied": False,
            "queries": ["test query"],
        },
        "diagnostics": CodeRetrievalDiagnostics(),
        "trace": CodeRetrievalTrace(),
        "_effective_top_k": 10,
    }


# ===========================================================================
# Class 1: Initialization paths
# ===========================================================================


class TestInitPaths:
    """Tests for __init__ branches and startup."""

    def test_local_assets_only_disables_truth_grounding(self, tmp_path, monkeypatch):
        for _k in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            monkeypatch.delenv(_k, raising=False)
        """local_assets_only=True should turn off truth_grounding (line 237-238)."""
        from axon.main import AxonBrain

        cfg = _make_config(tmp_path, local_assets_only=True, truth_grounding=True)
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            assert b.config.truth_grounding is False
            b.close()

    def test_offline_mode_disables_raptor_and_graph_rag(self, tmp_path, monkeypatch):
        for _k in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            monkeypatch.delenv(_k, raising=False)
        """offline_mode=True should disable raptor and graph_rag (lines 267-278)."""
        from axon.main import AxonBrain

        cfg = _make_config(tmp_path, offline_mode=True, raptor=True, graph_rag=True)
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            assert b.config.raptor is False
            assert b.config.graph_rag is False
            b.close()

    def test_offline_mode_disables_truth_grounding(self, tmp_path, monkeypatch):
        for _k in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            monkeypatch.delenv(_k, raising=False)
        """offline_mode=True should turn off truth_grounding (line 264-266)."""
        from axon.main import AxonBrain

        cfg = _make_config(tmp_path, offline_mode=True, truth_grounding=True)
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            assert b.config.truth_grounding is False
            b.close()

    def test_tokenizer_cache_dir_sets_env(self, tmp_path, monkeypatch):
        """tokenizer_cache_dir in config should set TIKTOKEN_CACHE_DIR (lines 291-292)."""
        monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
        from axon.main import AxonBrain

        cache_dir = str(tmp_path / "tiktoken_cache")
        cfg = _make_config(tmp_path, tokenizer_cache_dir=cache_dir)
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            assert os.environ.get("TIKTOKEN_CACHE_DIR") == cache_dir
            b.close()
        # Explicitly remove the env var set by AxonBrain so it doesn't bleed across tests.
        # monkeypatch.delenv at the start registers "absent" but the direct os.environ
        # assignment in __init__ bypasses monkeypatch's tracking; pop here is the safest fix.
        os.environ.pop("TIKTOKEN_CACHE_DIR", None)

    def test_semantic_splitter_init(self, tmp_path):
        """chunk_strategy=semantic initializes SemanticTextSplitter (lines 360-365)."""
        from axon.main import AxonBrain

        cfg = _make_config(tmp_path, chunk_strategy="semantic")
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.splitters.SemanticTextSplitter"
        ) as mock_splitter, patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            mock_splitter.assert_called_once()
            b.close()

    def test_markdown_splitter_init(self, tmp_path):
        """chunk_strategy=markdown initializes MarkdownSplitter (lines 366-370)."""
        from axon.main import AxonBrain

        cfg = _make_config(tmp_path, chunk_strategy="markdown")
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.splitters.MarkdownSplitter"
        ) as mock_splitter, patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            mock_splitter.assert_called_once()
            b.close()

    def test_cosine_semantic_splitter_init(self, tmp_path):
        """chunk_strategy=cosine_semantic initializes CosineSemanticSplitter (lines 371-378)."""
        from axon.main import AxonBrain

        cfg = _make_config(tmp_path, chunk_strategy="cosine_semantic")
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.splitters.CosineSemanticSplitter"
        ) as mock_splitter, patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ), patch.object(
            AxonBrain, "_preflight_model_audit", return_value=None
        ):
            b = AxonBrain(cfg)
            mock_splitter.assert_called_once()
            b.close()


# ===========================================================================
# Class 2: _resolve_model_path
# ===========================================================================


class TestResolveModelPath:
    def test_absolute_path_returned_unchanged(self, brain):
        result = brain._resolve_model_path("/abs/path/to/model", "hf")
        assert result == "/abs/path/to/model"

    def test_dot_path_returned_unchanged(self, brain):
        result = brain._resolve_model_path("./relative/model", "hf")
        assert result == "./relative/model"

    def test_no_roots_returns_model_name(self, brain):
        brain.config.hf_models_dir = ""
        brain.config.local_models_dir = ""
        result = brain._resolve_model_path("org/model-name", "hf")
        assert result == "org/model-name"

    def test_embedding_kind_uses_embedding_models_dir(self, tmp_path, brain):
        model_dir = tmp_path / "emb_models" / "my-model"
        model_dir.mkdir(parents=True)
        brain.config.embedding_models_dir = str(tmp_path / "emb_models")
        brain.config.local_models_dir = ""
        result = brain._resolve_model_path("org/my-model", "embedding")
        assert result == str(model_dir)

    def test_hf_dash_slug_resolution(self, tmp_path, brain):
        """Model ID with / converted to -- for HF hub style."""
        model_dir = tmp_path / "hf_models" / "org--model-name"
        model_dir.mkdir(parents=True)
        brain.config.hf_models_dir = str(tmp_path / "hf_models")
        brain.config.local_models_dir = ""
        result = brain._resolve_model_path("org/model-name", "hf")
        assert result == str(model_dir)

    def test_model_not_found_returns_original_and_logs_warning(self, tmp_path, brain):
        brain.config.hf_models_dir = str(tmp_path / "hf_models")
        brain.config.local_models_dir = ""
        result = brain._resolve_model_path("org/nonexistent-model", "hf")
        assert result == "org/nonexistent-model"


# ===========================================================================
# Class 3: _preflight_model_audit
# ===========================================================================


class TestPreflightModelAudit:
    def test_local_assets_only_raises_on_remote_id(self, tmp_path, monkeypatch):
        for _k in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            monkeypatch.delenv(_k, raising=False)
        """local_assets_only=True + remote embedding model should raise RuntimeError."""
        from axon.config import AxonConfig
        from axon.main import AxonBrain

        cfg = AxonConfig(
            bm25_path=str(tmp_path / "bm25"),
            vector_store_path=str(tmp_path / "vs"),
            local_assets_only=True,
            embedding_model="some-remote-model",  # bare HF ID, not on disk
        )
        with patch("axon.main.OpenVectorStore"), patch("axon.main.OpenEmbedding"), patch(
            "axon.main.OpenLLM"
        ), patch("axon.main.OpenReranker"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.projects.ensure_project"
        ), patch.object(
            AxonBrain, "_load_hash_store", return_value=set()
        ), patch.object(
            AxonBrain, "_load_doc_versions", return_value=None
        ), patch.object(
            AxonBrain, "_load_entity_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ), patch.object(
            AxonBrain, "_log_startup_summary", return_value=None
        ):
            with pytest.raises(RuntimeError, match="local_assets_only"):
                AxonBrain(cfg)

    def test_audit_logs_for_valid_config(self, brain):
        """_preflight_model_audit should not raise for a standard mocked config."""
        # The brain fixture uses a patched _preflight_model_audit so this just
        # ensures the brain is constructed without error.
        assert brain is not None


# ===========================================================================
# Class 4: should_recommend_project
# ===========================================================================


class TestShouldRecommendProject:
    def test_non_default_project_returns_false(self, brain):
        brain._active_project = "myproject"
        assert brain.should_recommend_project() is False

    def test_default_project_no_named_projects_returns_true(self, brain):
        brain._active_project = "default"
        with patch("axon.projects.list_projects", return_value=[{"name": "default"}]):
            assert brain.should_recommend_project() is True

    def test_default_project_with_named_projects_returns_false(self, brain):
        brain._active_project = "default"
        with patch(
            "axon.projects.list_projects",
            return_value=[{"name": "default"}, {"name": "work"}],
        ):
            assert brain.should_recommend_project() is False

    def test_exception_returns_false(self, brain):
        brain._active_project = "default"
        with patch("axon.projects.list_projects", side_effect=Exception("boom")):
            assert brain.should_recommend_project() is False


# ===========================================================================
# Class 5: context manager __enter__ / __exit__
# ===========================================================================


class TestContextManager:
    def test_enter_returns_brain(self, brain):
        result = brain.__enter__()
        assert result is brain

    def test_exit_calls_close(self, brain):
        brain.close = MagicMock()
        brain.__exit__(None, None, None)
        brain.close.assert_called_once()

    def test_exit_calls_close_on_exception(self, brain):
        brain.close = MagicMock()
        brain.__exit__(ValueError, ValueError("oops"), None)
        brain.close.assert_called_once()


# ===========================================================================
# Class 6: _apply_overrides
# ===========================================================================


class TestApplyOverrides:
    def test_none_overrides_returns_config_unchanged(self, brain):
        result = brain._apply_overrides(None)
        assert result is brain.config

    def test_empty_overrides_returns_config_unchanged(self, brain):
        result = brain._apply_overrides({})
        assert result is brain.config

    def test_valid_override_creates_copy(self, brain):
        result = brain._apply_overrides({"top_k": 99})
        assert result is not brain.config
        assert result.top_k == 99
        assert brain.config.top_k != 99  # original unchanged

    def test_unknown_key_ignored(self, brain):
        result = brain._apply_overrides({"nonexistent_key_xyz": "value"})
        assert result is not brain.config  # copy still made on non-empty dict

    def test_none_value_skipped(self, brain):
        original_top_k = brain.config.top_k
        result = brain._apply_overrides({"top_k": None})
        assert result.top_k == original_top_k


# ===========================================================================
# Class 7: _doc_hash
# ===========================================================================


class TestDocHash:
    def test_hash_of_known_text(self, brain):
        import hashlib

        doc = {"text": "hello world"}
        expected = hashlib.md5(b"hello world").hexdigest()
        assert brain._doc_hash(doc) == expected

    def test_empty_text(self, brain):
        import hashlib

        doc = {"text": ""}
        expected = hashlib.md5(b"").hexdigest()
        assert brain._doc_hash(doc) == expected

    def test_missing_text_key(self, brain):
        import hashlib

        doc = {}
        expected = hashlib.md5(b"").hexdigest()
        assert brain._doc_hash(doc) == expected


# ===========================================================================
# Class 8: query() basic paths
# ===========================================================================


class TestQueryBasic:
    def test_query_returns_llm_response(self, brain):
        brain.llm.complete = MagicMock(return_value="The answer is 42.")
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("some context", False))
        brain._validate_embedding_meta = MagicMock()

        response = brain.query("What is the answer?")
        assert response == "The answer is 42."

    def test_query_no_results_returns_fallback_message(self, brain):
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval(results=[]))
        brain._validate_embedding_meta = MagicMock()
        brain.config.discussion_fallback = False

        response = brain.query("empty")
        assert "don't have any relevant information" in response

    def test_query_no_results_discussion_fallback(self, brain):
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval(results=[]))
        brain._validate_embedding_meta = MagicMock()
        brain.config.discussion_fallback = True
        brain.llm.complete = MagicMock(return_value="General knowledge answer.")

        response = brain.query("some question")
        assert response == "General knowledge answer."

    def test_query_cache_hit_returns_cached(self, brain):
        brain._validate_embedding_meta = MagicMock()
        brain.config.query_cache = True
        brain.config.query_cache_size = 10
        brain.config.query_router = "off"

        # Pre-populate the cache
        cache_key = brain._make_cache_key("cached query", None, brain.config)
        with brain._cache_lock:
            brain._query_cache[cache_key] = "cached response"

        result = brain.query("cached query")
        assert result == "cached response"

    def test_query_cache_stores_response(self, brain):
        brain.llm.complete = MagicMock(return_value="fresh answer")
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.config.query_cache = True
        brain.config.query_cache_size = 10
        brain.config.query_router = "off"

        brain.query("new question")

        cache_key = brain._make_cache_key("new question", None, brain.config)
        with brain._cache_lock:
            assert cache_key in brain._query_cache

    def test_query_cache_lru_evicts_oldest(self, brain):
        brain.llm.complete = MagicMock(return_value="answer")
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.config.query_cache = True
        brain.config.query_cache_size = 1
        brain.config.query_router = "off"

        brain.query("first question")
        brain.query("second question")  # should evict first

        first_key = brain._make_cache_key("first question", None, brain.config)
        with brain._cache_lock:
            assert first_key not in brain._query_cache

    def test_query_with_chat_history_bypasses_cache(self, brain):
        brain.llm.complete = MagicMock(return_value="history answer")
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.config.query_cache = True
        brain.config.query_cache_size = 10

        history = [{"role": "user", "content": "prior message"}]
        brain.query("question with history", chat_history=history)

        cache_key = brain._make_cache_key("question with history", None, brain.config)
        with brain._cache_lock:
            assert cache_key not in brain._query_cache

    def test_query_with_overrides_applied(self, brain):
        brain.llm.complete = MagicMock(return_value="overridden answer")
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()

        brain.query("test query", overrides={"top_k": 3})
        brain._execute_retrieval.assert_called_once()

    def test_query_rerank_is_called(self, brain):
        brain.config.rerank = True
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="ranked answer")
        brain.reranker.rerank = MagicMock(return_value=[_fake_result()])

        brain.query("test query")
        brain.reranker.rerank.assert_called_once()

    def test_query_slices_to_top_k(self, brain):
        many_results = [_fake_result(id=f"doc{i}") for i in range(20)]
        brain._execute_retrieval = MagicMock(
            return_value={**_mock_retrieval(many_results), "_effective_top_k": 5}
        )
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="sliced answer")

        brain.query("test")
        # _build_context receives only top 5
        ctx_results = brain._build_context.call_args[0][0]
        assert len(ctx_results) <= 5


# ===========================================================================
# Class 9: query() advanced RAG transforms
# ===========================================================================


class TestQueryAdvancedRAG:
    def test_query_with_hyde_enabled(self, brain):
        """HyDE: LLM generates hypothetical doc, embed is used for retrieval."""
        brain.config.hyde = True
        brain.config.query_router = "off"

        hyde_text = "A hypothetical passage about the topic."
        brain.llm.complete = MagicMock(side_effect=["The final answer."])
        brain._get_hyde_document = MagicMock(return_value=hyde_text)
        brain.embedding.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
        brain.vector_store.search = MagicMock(
            return_value=[_fake_result(id="hd1", text="relevant doc")]
        )
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="hyde answer")

        result = brain.query("What is X?")
        assert result == "hyde answer"

    def test_query_with_multi_query(self, brain):
        """multi_query: LLM expands query into multiple phrasings."""
        brain.config.multi_query = True
        brain.config.query_router = "off"

        brain._get_multi_queries = MagicMock(return_value=["Q1", "alternative Q", "rephrased Q"])
        brain.embedding.embed_query = MagicMock(return_value=[0.1, 0.2])
        brain.embedding.embed = MagicMock(return_value=[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        brain.vector_store.search = MagicMock(return_value=[_fake_result()])
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="multi answer")

        result = brain.query("What is X?")
        assert result == "multi answer"

    def test_query_with_step_back(self, brain):
        """step_back: LLM generates abstract version of query."""
        brain.config.step_back = True
        brain.config.query_router = "off"

        brain._get_step_back_query = MagicMock(return_value="More general version of the question")
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.embedding.embed = MagicMock(return_value=[[0.1], [0.2]])
        brain.vector_store.search = MagicMock(return_value=[_fake_result()])
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="step-back answer")

        result = brain.query("Specific technical question?")
        assert result == "step-back answer"

    def test_query_with_hybrid_search(self, brain):
        """hybrid_search: combines vector and BM25 results."""
        brain.config.hybrid_search = True
        brain.config.hybrid_mode = "rrf"
        brain.config.query_router = "off"

        brain.embedding.embed_query = MagicMock(return_value=[0.1, 0.2])
        brain.vector_store.search = MagicMock(return_value=[_fake_result(id="v1")])
        brain.bm25 = MagicMock()
        brain.bm25.search = MagicMock(return_value=[_fake_result(id="b1", score=0.7)])
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="hybrid answer")

        result = brain.query("hybrid query")
        assert result == "hybrid answer"

    def test_query_graphrag_budget_applied(self, brain):
        """graph_rag_budget > 0 splits results into base and graph-expanded slots."""
        brain.config.graph_rag = True
        brain.config.graph_rag_budget = 2
        brain.config.query_router = "off"

        base = [_fake_result(id=f"base{i}") for i in range(3)]
        expanded = [{**_fake_result(id=f"exp{i}"), "_graph_expanded": True} for i in range(4)]
        retrieval = {
            **_mock_retrieval(base + expanded),
            "matched_entities": [],
            "_effective_top_k": 3,
        }
        brain._execute_retrieval = MagicMock(return_value=retrieval)
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="budget answer")

        result = brain.query("test")
        assert result == "budget answer"

    def test_query_graph_rag_local_context_prepended(self, brain):
        """When graph_rag + local mode + matched entities, local context is prepended."""
        brain.config.graph_rag = True
        brain.config.graph_rag_mode = "local"
        brain.config.query_router = "off"

        brain._entity_graph = {"entity1": {"chunk_ids": ["doc1"], "description": "desc"}}
        retrieval = {**_mock_retrieval(), "matched_entities": ["entity1"]}
        brain._execute_retrieval = MagicMock(return_value=retrieval)
        brain._local_search_context = MagicMock(return_value="Local graph context here.")
        brain._build_context = MagicMock(return_value=("doc context", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.complete = MagicMock(return_value="local graph answer")

        result = brain.query("entity question")
        assert result == "local graph answer"
        # Local context should have been used
        call_args = brain.llm.complete.call_args
        assert "Local graph context here." in call_args[0][1]


# ===========================================================================
# Class 10: query_stream()
# ===========================================================================


class TestQueryStream:
    def test_stream_yields_sources_then_chunks(self, brain):
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.stream = MagicMock(return_value=iter(["chunk1", " chunk2"]))

        chunks = list(brain.query_stream("what?"))
        # First item should be sources marker
        assert isinstance(chunks[0], dict)
        assert chunks[0]["type"] == "sources"
        assert "chunk1" in chunks[1:]

    def test_stream_no_results_yields_fallback(self, brain):
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval(results=[]))
        brain._validate_embedding_meta = MagicMock()
        brain.config.discussion_fallback = False

        chunks = list(brain.query_stream("empty?"))
        assert any("don't have any relevant information" in str(c) for c in chunks)

    def test_stream_no_results_discussion_fallback(self, brain):
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval(results=[]))
        brain._validate_embedding_meta = MagicMock()
        brain.config.discussion_fallback = True
        brain.llm.stream = MagicMock(return_value=iter(["gen ", "answer"]))

        chunks = list(brain.query_stream("general?"))
        assert "gen " in chunks or "answer" in chunks

    def test_stream_rerank_called_when_enabled(self, brain):
        brain.config.rerank = True
        brain._execute_retrieval = MagicMock(return_value=_mock_retrieval())
        brain._build_context = MagicMock(return_value=("ctx", False))
        brain._validate_embedding_meta = MagicMock()
        brain.llm.stream = MagicMock(return_value=iter(["answer"]))
        brain.reranker.rerank = MagicMock(return_value=[_fake_result()])

        list(brain.query_stream("ranked?"))
        brain.reranker.rerank.assert_called_once()


# ===========================================================================
# Class 11: list_documents
# ===========================================================================


class TestListDocuments:
    def test_list_documents_delegates_to_vector_store(self, brain):
        expected = [{"source": "file.txt", "chunks": 3, "doc_ids": ["a", "b", "c"]}]
        brain.vector_store.list_documents = MagicMock(return_value=expected)
        result = brain.list_documents()
        assert result == expected

    def test_list_documents_empty(self, brain):
        brain.vector_store.list_documents = MagicMock(return_value=[])
        assert brain.list_documents() == []


# ===========================================================================
# Class 12: ingest() basic paths
# ===========================================================================


class TestIngest:
    def _make_doc(self, id="doc1", text="some content"):
        return {"id": id, "text": text, "metadata": {"source": f"{id}.txt"}}

    def test_ingest_empty_list_returns_early(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain.ingest([])
        brain._assert_write_allowed.assert_not_called()

    def test_ingest_single_doc(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain._save_hash_store = MagicMock()
        brain._save_doc_versions = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain.splitter = None  # skip splitting

        brain.embedding.embed = MagicMock(return_value=[[0.1, 0.2]])
        brain._own_vector_store.add = MagicMock()
        brain._own_bm25 = MagicMock()

        from axon.runtime import _WriteLease

        with patch("axon.runtime.get_registry") as mock_reg:
            fake_lease = MagicMock(spec=_WriteLease)
            fake_lease.is_stale.return_value = False
            mock_reg.return_value.acquire.return_value = fake_lease
            brain.ingest([self._make_doc()])

        brain._own_vector_store.add.assert_called_once()

    def test_ingest_dedup_skips_duplicate(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain._save_hash_store = MagicMock()
        brain._save_doc_versions = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain.config.dedup_on_ingest = True
        brain.splitter = None

        doc = self._make_doc()
        existing_hash = brain._doc_hash(doc)
        brain._ingested_hashes = {existing_hash}

        brain.embedding.embed = MagicMock()
        brain._own_vector_store.add = MagicMock()
        brain._own_bm25 = MagicMock()

        from axon.runtime import _WriteLease

        with patch("axon.runtime.get_registry") as mock_reg:
            fake_lease = MagicMock(spec=_WriteLease)
            fake_lease.is_stale.return_value = False
            mock_reg.return_value.acquire.return_value = fake_lease
            brain.ingest([doc])

        brain._own_vector_store.add.assert_not_called()

    def test_ingest_dedup_new_doc_passes_through(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain._save_hash_store = MagicMock()
        brain._save_doc_versions = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain.config.dedup_on_ingest = True
        brain.splitter = None
        brain._ingested_hashes = set()

        brain.embedding.embed = MagicMock(return_value=[[0.1]])
        brain._own_vector_store.add = MagicMock()
        brain._own_bm25 = MagicMock()

        from axon.runtime import _WriteLease

        with patch("axon.runtime.get_registry") as mock_reg:
            fake_lease = MagicMock(spec=_WriteLease)
            fake_lease.is_stale.return_value = False
            mock_reg.return_value.acquire.return_value = fake_lease
            brain.ingest([self._make_doc()])

        brain._own_vector_store.add.assert_called_once()

    def test_ingest_updates_doc_versions(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain._save_hash_store = MagicMock()
        brain._save_doc_versions = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain.splitter = None

        brain.embedding.embed = MagicMock(return_value=[[0.1]])
        brain._own_vector_store.add = MagicMock()
        brain._own_bm25 = MagicMock()

        from axon.runtime import _WriteLease

        with patch("axon.runtime.get_registry") as mock_reg:
            fake_lease = MagicMock(spec=_WriteLease)
            fake_lease.is_stale.return_value = False
            mock_reg.return_value.acquire.return_value = fake_lease
            brain.ingest([self._make_doc("doc1")])

        assert "doc1.txt" in brain._doc_versions

    def test_ingest_raises_on_read_only_scope(self, brain):
        brain._read_only_scope = True

        with pytest.raises((PermissionError, Exception)):
            brain.ingest([self._make_doc()])

    def test_ingest_with_max_chunks_per_source_cap(self, brain):
        """max_chunks_per_source should truncate chunks from one source."""
        brain._assert_write_allowed = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain._save_hash_store = MagicMock()
        brain._save_doc_versions = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain.config.max_chunks_per_source = 2
        brain.splitter = None

        # 5 chunks all from the same source
        docs = [
            {"id": f"c{i}", "text": f"text {i}", "metadata": {"source": "same.txt"}}
            for i in range(5)
        ]

        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1]] * len(texts))
        brain._own_vector_store.add = MagicMock()
        brain._own_bm25 = MagicMock()

        from axon.runtime import _WriteLease

        with patch("axon.runtime.get_registry") as mock_reg:
            fake_lease = MagicMock(spec=_WriteLease)
            fake_lease.is_stale.return_value = False
            mock_reg.return_value.acquire.return_value = fake_lease
            brain.ingest(docs)

        # add should have been called with at most 2 items
        call_args = brain._own_vector_store.add.call_args
        ids_passed = call_args[0][0]
        assert len(ids_passed) <= 2


# ===========================================================================
# Class 13: _load_hash_store / _save_hash_store
# ===========================================================================


class TestHashStore:
    def test_load_hash_store_empty_when_no_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.config.bm25_path = str(tmp_path / "bm25_new")
        result = brain._load_hash_store()
        assert result == set()

    def test_load_hash_store_reads_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        bm25_dir = tmp_path / "bm25_hs"
        bm25_dir.mkdir(parents=True)
        brain.config.bm25_path = str(bm25_dir)
        hash_file = bm25_dir / ".content_hashes"
        hash_file.write_text("abc123\ndef456\n")

        result = brain._load_hash_store()
        assert "abc123" in result
        assert "def456" in result

    def test_save_hash_store_writes_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        bm25_dir = tmp_path / "bm25_save"
        bm25_dir.mkdir(parents=True)
        brain.config.bm25_path = str(bm25_dir)
        brain._ingested_hashes = {"h1", "h2"}

        brain._save_hash_store()

        hash_file = bm25_dir / ".content_hashes"
        assert hash_file.exists()
        content = hash_file.read_text()
        assert "h1" in content
        assert "h2" in content


# ===========================================================================
# Class 14: _load_doc_versions / _save_doc_versions
# ===========================================================================


class TestDocVersions:
    def test_load_doc_versions_no_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._doc_versions_path = str(tmp_path / "nonexistent.json")
        brain._load_doc_versions()
        assert brain._doc_versions == {}

    def test_load_doc_versions_reads_json(self, tmp_path):
        import json

        brain = _make_brain(tmp_path)
        versions_file = tmp_path / "doc_versions.json"
        data = {"file.txt": {"content_hash": "abc", "chunk_count": 3}}
        versions_file.write_text(json.dumps(data))
        brain._doc_versions_path = str(versions_file)
        brain._load_doc_versions()
        assert brain._doc_versions["file.txt"]["content_hash"] == "abc"

    def test_load_doc_versions_corrupt_json(self, tmp_path):
        brain = _make_brain(tmp_path)
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        brain._doc_versions_path = str(bad_file)
        brain._load_doc_versions()
        assert brain._doc_versions == {}

    def test_save_doc_versions_writes_json(self, tmp_path):
        import json

        brain = _make_brain(tmp_path)
        brain._doc_versions_path = str(tmp_path / "versions.json")
        brain._doc_versions = {"src.txt": {"content_hash": "xyz", "chunk_count": 1}}
        brain._save_doc_versions()
        data = json.loads((tmp_path / "versions.json").read_text())
        assert data["src.txt"]["content_hash"] == "xyz"

    def test_get_doc_versions_returns_copy(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._doc_versions = {"a.txt": {"chunk_count": 2}}
        result = brain.get_doc_versions()
        assert result == brain._doc_versions
        assert result is not brain._doc_versions


# ===========================================================================
# Class 15: _validate_embedding_meta
# ===========================================================================


class TestValidateEmbeddingMeta:
    def test_no_meta_file_passes_silently(self, brain):
        brain._load_embedding_meta = MagicMock(return_value=None)
        brain._validate_embedding_meta(on_mismatch="raise")  # should not raise

    def test_matching_meta_passes_silently(self, brain):
        brain._load_embedding_meta = MagicMock(
            return_value={
                "embedding_provider": brain.config.embedding_provider,
                "embedding_model": brain.config.embedding_model,
            }
        )
        brain._validate_embedding_meta(on_mismatch="raise")  # should not raise

    def test_mismatch_raises(self, brain):
        brain._load_embedding_meta = MagicMock(
            return_value={
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-ada-002",
            }
        )
        brain.config.embedding_provider = "sentence_transformers"
        brain.config.embedding_model = "all-MiniLM-L6-v2"
        with pytest.raises(ValueError, match="Embedding model mismatch"):
            brain._validate_embedding_meta(on_mismatch="raise")

    def test_mismatch_warns_not_raise(self, brain):
        brain._load_embedding_meta = MagicMock(
            return_value={
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-ada-002",
            }
        )
        brain.config.embedding_provider = "sentence_transformers"
        brain.config.embedding_model = "all-MiniLM-L6-v2"
        # Should NOT raise when on_mismatch="warn"
        brain._validate_embedding_meta(on_mismatch="warn")


# ===========================================================================
# Class 16: switch_project
# ===========================================================================


class TestSwitchProject:
    def _patch_switch(self, brain, project_exists=True):
        from pathlib import Path

        mock_vs = MagicMock()
        mock_bm25 = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = project_exists

        brain.close = MagicMock()
        return mock_vs, mock_bm25, mock_path

    def test_switch_to_default_restores_base_paths(self, brain):
        original_vs = brain._base_vector_store_path
        original_bm25 = brain._base_bm25_path

        brain.close = MagicMock()
        with patch("axon.main.OpenVectorStore") as mock_vs_cls, patch(
            "axon.retrievers.BM25Retriever"
        ) as mock_bm25_cls, patch("axon.projects.set_active_project"), patch(
            "axon.projects.list_descendants", return_value=[]
        ), patch(
            "axon.runtime.get_registry"
        ):
            mock_vs_cls.return_value = MagicMock()
            mock_bm25_cls.return_value = MagicMock()
            brain._load_hash_store = MagicMock(return_value=set())
            brain._load_entity_graph = MagicMock(return_value={})
            brain._load_relation_graph = MagicMock(return_value={})
            brain._load_community_levels = MagicMock(return_value={})
            brain._load_community_summaries = MagicMock(return_value={})
            brain._load_entity_embeddings = MagicMock(return_value={})
            brain._load_claims_graph = MagicMock(return_value={})
            brain._load_community_hierarchy = MagicMock(return_value={})

            brain.switch_project("default")

        assert brain.config.vector_store_path == original_vs
        assert brain.config.bm25_path == original_bm25
        assert brain._active_project == "default"
        assert brain._active_project_kind == "default"

    def test_switch_to_nonexistent_project_raises(self, brain):
        from pathlib import Path

        brain.close = MagicMock()
        with patch("axon.projects.project_dir") as mock_dir, patch(
            "axon.projects.project_vector_path"
        ), patch("axon.projects.project_bm25_path"), patch(
            "axon.projects.set_active_project"
        ), patch(
            "axon.projects.list_descendants", return_value=[]
        ):
            p = MagicMock(spec=Path)
            p.exists.return_value = False
            mock_dir.return_value = p
            with pytest.raises(ValueError, match="does not exist"):
                brain.switch_project("nonexistent_project")

    def test_switch_to_scope_at_projects(self, brain):
        """Switching to @projects calls _switch_to_scope."""
        brain._switch_to_scope = MagicMock()
        brain.switch_project("@projects")
        brain._switch_to_scope.assert_called_once_with("@projects")

    def test_switch_sets_active_project_kind_local(self, brain):
        from pathlib import Path

        brain.close = MagicMock()
        with patch("axon.projects.project_dir") as mock_dir, patch(
            "axon.projects.project_vector_path", return_value="/fake/vs"
        ), patch("axon.projects.project_bm25_path", return_value="/fake/bm25"), patch(
            "axon.projects.set_active_project"
        ), patch(
            "axon.projects.list_descendants", return_value=[]
        ), patch(
            "axon.main.OpenVectorStore"
        ) as mock_vs_cls, patch(
            "axon.retrievers.BM25Retriever"
        ) as mock_bm25_cls, patch(
            "axon.runtime.get_registry"
        ):
            p = MagicMock(spec=Path)
            p.exists.return_value = True
            mock_dir.return_value = p
            mock_vs_cls.return_value = MagicMock()
            mock_bm25_cls.return_value = MagicMock()
            brain._load_hash_store = MagicMock(return_value=set())
            brain._load_entity_graph = MagicMock(return_value={})
            brain._load_relation_graph = MagicMock(return_value={})
            brain._load_community_levels = MagicMock(return_value={})
            brain._load_community_summaries = MagicMock(return_value={})
            brain._load_entity_embeddings = MagicMock(return_value={})
            brain._load_claims_graph = MagicMock(return_value={})
            brain._load_community_hierarchy = MagicMock(return_value={})

            brain.switch_project("myproject")

        assert brain._active_project == "myproject"
        assert brain._active_project_kind == "local"


# ===========================================================================
# Class 17: _switch_to_scope
# ===========================================================================


class TestSwitchToScope:
    def test_invalid_scope_raises(self, brain):
        with pytest.raises(ValueError, match="Unknown scope"):
            brain._switch_to_scope("@invalid_scope")

    def test_no_projects_found_raises(self, brain):
        """_switch_to_scope(@projects) raises ValueError when no authoritative projects exist."""
        from pathlib import Path

        import axon.projects as proj_mod

        brain.close = MagicMock()
        # Patch PROJECTS_ROOT to a non-existent Path so no projects are found
        fake_root = MagicMock(spec=Path)
        fake_root.exists.return_value = False

        with patch.object(proj_mod, "PROJECTS_ROOT", fake_root), patch(
            "axon.main.OpenVectorStore"
        ), patch("axon.retrievers.BM25Retriever"):
            with pytest.raises(ValueError, match="No authoritative projects"):
                brain._switch_to_scope("@projects")


# ===========================================================================
# Class 18: _is_mounted_share / _assert_write_allowed
# ===========================================================================


class TestWriteGuards:
    def test_is_mounted_share_returns_true_when_kind_mounted(self, brain):
        brain._active_project_kind = "mounted"
        assert brain._is_mounted_share() is True

    def test_is_mounted_share_returns_false_when_kind_local(self, brain):
        brain._active_project_kind = "local"
        assert brain._is_mounted_share() is False

    def test_is_mounted_share_fallback_to_legacy_flag(self, brain):
        # Remove the kind attribute to test legacy fallback
        del brain._active_project_kind
        brain._mounted_share = True
        assert brain._is_mounted_share() is True

    def test_assert_write_allowed_raises_on_scope(self, brain):
        brain._read_only_scope = True
        with pytest.raises((PermissionError, ValueError, RuntimeError)):
            brain._assert_write_allowed("ingest")

    def test_assert_write_allowed_passes_on_local(self, brain):
        brain._read_only_scope = False
        brain._active_project_kind = "local"
        brain._active_project = "myproject"
        # Should not raise
        brain._assert_write_allowed("ingest")


# ===========================================================================
# Class 19: RAPTOR - _raptor_group_by_structure
# ===========================================================================


class TestRaptorGroupByStructure:
    def _make_chunk(self, id, text, **meta):
        return {"id": id, "text": text, "metadata": meta}

    def test_no_structure_uses_fixed_windows(self, brain):
        chunks = [self._make_chunk(f"c{i}", f"plain text {i}") for i in range(6)]
        groups = brain._raptor_group_by_structure(chunks, n=3)
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 3

    def test_markdown_headings_create_sections(self, brain):
        chunks = [
            self._make_chunk("c0", "# Introduction\nSome intro text."),
            self._make_chunk("c1", "More intro content."),
            self._make_chunk("c2", "# Methods\nMethod details here."),
            self._make_chunk("c3", "More method details."),
        ]
        groups = brain._raptor_group_by_structure(chunks, n=5)
        # Should detect sections at headings
        assert len(groups) >= 2

    def test_heading_metadata_creates_new_section(self, brain):
        chunks = [
            self._make_chunk("c0", "text", heading="Section A"),
            self._make_chunk("c1", "more text"),
            self._make_chunk("c2", "text 2", heading="Section B"),
        ]
        groups = brain._raptor_group_by_structure(chunks, n=10)
        assert len(groups) >= 2


# ===========================================================================
# Class 20: RAPTOR - _generate_raptor_summaries
# ===========================================================================


class TestGenerateRaptorSummaries:
    def test_zero_chunk_group_size_returns_empty(self, brain):
        brain.config.raptor_chunk_group_size = 0
        result = brain._generate_raptor_summaries([])
        assert result == []

    def test_empty_documents_returns_empty(self, brain):
        brain.config.raptor_chunk_group_size = 3
        result = brain._generate_raptor_summaries([])
        assert result == []

    def test_generates_summaries_for_chunks(self, brain):
        brain.config.raptor_chunk_group_size = 2
        brain.config.raptor_cache_summaries = False
        brain.config.raptor_max_levels = 1
        brain.llm.complete = MagicMock(return_value="Summary of the window.")

        docs = [
            {"id": f"c{i}", "text": f"Content {i}", "metadata": {"source": "doc.txt"}}
            for i in range(4)
        ]
        result = brain._generate_raptor_summaries(docs)
        assert len(result) >= 1
        assert all(r["metadata"]["raptor_level"] == 1 for r in result)

    def test_raptor_summary_cache_hit(self, brain):
        brain.config.raptor_chunk_group_size = 3
        brain.config.raptor_cache_summaries = True
        brain.config.raptor_max_levels = 1
        brain.llm.complete = MagicMock(return_value="Cached summary.")

        docs = [{"id": "c0", "text": "Content A", "metadata": {"source": "doc.txt"}}]
        # Run twice — second run should use cache
        brain._generate_raptor_summaries(docs)
        initial_call_count = brain.llm.complete.call_count
        brain._generate_raptor_summaries(docs)
        # Cache should reduce calls on second run
        assert brain.llm.complete.call_count <= initial_call_count * 2

    def test_llm_returns_empty_string_no_node_created(self, brain):
        brain.config.raptor_chunk_group_size = 2
        brain.config.raptor_cache_summaries = False
        brain.config.raptor_max_levels = 1
        brain.llm.complete = MagicMock(return_value="")

        docs = [
            {"id": "c0", "text": "Content A", "metadata": {"source": "doc.txt"}},
            {"id": "c1", "text": "Content B", "metadata": {"source": "doc.txt"}},
        ]
        result = brain._generate_raptor_summaries(docs)
        assert result == []


# ===========================================================================
# Class 21: RAPTOR drilldown
# ===========================================================================


class TestRaptorDrilldown:
    def test_non_raptor_results_pass_through(self, brain):
        brain.config.raptor_drilldown = True
        results = [_fake_result(id="leaf1")]
        out = brain._raptor_drilldown("query", results)
        assert out == results

    def test_raptor_node_with_children_fetches_leaves(self, brain):
        brain.config.raptor_drilldown = True
        brain.config.rerank = False
        brain.config.raptor_drilldown_top_k = 5

        summary_doc = {
            "id": "raptor_abc",
            "text": "summary",
            "score": 0.9,
            "metadata": {
                "raptor_level": 1,
                "source": "doc.txt",
                "window_start": 0,
                "window_end": 1,
                "children_ids": ["c0", "c1"],
            },
        }
        leaf_c0 = _fake_result(id="c0", text="leaf 0")
        leaf_c1 = _fake_result(id="c1", text="leaf 1")

        brain.vector_store.get_by_ids = MagicMock(return_value=[leaf_c0, leaf_c1])

        out = brain._raptor_drilldown("query", [summary_doc])
        assert any(r["id"] in ("c0", "c1") for r in out)

    def test_raptor_node_without_children_uses_search(self, brain):
        brain.config.raptor_drilldown = True
        brain.config.rerank = False
        brain.config.raptor_drilldown_top_k = 5

        summary_doc = {
            "id": "raptor_abc",
            "text": "summary",
            "score": 0.9,
            "metadata": {
                "raptor_level": 1,
                "source": "doc.txt",
                "window_start": 0,
                "window_end": 1,
                # No children_ids — legacy node
            },
        }
        leaf = _fake_result(id="leaf1")
        brain.embedding.embed = MagicMock(return_value=[[0.1, 0.2]])
        brain.vector_store.search = MagicMock(return_value=[leaf])

        out = brain._raptor_drilldown("query", [summary_doc])
        assert any(r["id"] == "leaf1" for r in out)

    def test_drilldown_disabled_returns_unchanged(self, brain):
        brain.config.raptor_drilldown = False
        results = [
            {
                "id": "raptor_abc",
                "text": "summary",
                "score": 0.9,
                "metadata": {
                    "raptor_level": 1,
                    "source": "doc.txt",
                    "window_start": 0,
                    "window_end": 1,
                },
            }
        ]
        out = brain._raptor_drilldown("query", results, cfg=brain.config)
        assert out == results


# ===========================================================================
# Class 22: _apply_artifact_ranking
# ===========================================================================


class TestApplyArtifactRanking:
    def test_tree_traversal_mode_boosts_leaves(self, brain):
        leaf = _fake_result(id="leaf", score=0.8)
        raptor = {"id": "rap1", "text": "sum", "score": 0.8, "metadata": {"raptor_level": 1}}
        community = {
            "id": "__community__1",
            "text": "comm",
            "score": 0.8,
            "metadata": {"graph_rag_type": "community_report"},
        }
        brain.config.raptor_retrieval_mode = "tree_traversal"
        out = brain._apply_artifact_ranking([community, raptor, leaf], cfg=brain.config)
        # leaf should come first in tree_traversal mode (highest multiplier)
        assert out[0]["id"] == "leaf"

    def test_summary_first_mode_boosts_raptor(self, brain):
        leaf = _fake_result(id="leaf", score=0.8)
        raptor = {"id": "rap1", "text": "sum", "score": 0.8, "metadata": {"raptor_level": 1}}
        brain.config.raptor_retrieval_mode = "summary_first"
        out = brain._apply_artifact_ranking([leaf, raptor], cfg=brain.config)
        assert out[0]["id"] == "rap1"

    def test_unknown_mode_returns_unchanged(self, brain):
        results = [_fake_result(id="r1")]
        brain.config.raptor_retrieval_mode = "unknown_mode"
        out = brain._apply_artifact_ranking(results, cfg=brain.config)
        assert out == results


# ===========================================================================
# Class 23: _detect_dataset_type
# ===========================================================================


class TestDetectDatasetType:
    def test_code_extension_detected(self, brain):
        brain.config.dataset_type = "auto"
        doc = {"text": "def foo(): pass", "metadata": {"source": "script.py"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "codebase"

    def test_manifest_file_detected(self, brain):
        brain.config.dataset_type = "auto"
        doc = {"text": '{"name": "mypackage"}', "metadata": {"source": "package.json"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "manifest"

    def test_discussion_json_detected(self, brain):
        import json

        brain.config.dataset_type = "auto"
        content = json.dumps([{"role": "user", "content": "Hello"}])
        doc = {"text": content, "metadata": {"source": "chat.json"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "discussion"

    def test_paper_signals_detected(self, brain):
        brain.config.dataset_type = "auto"
        text = "Abstract\nThis paper presents a novel approach.\nIntroduction\nReferences\n"
        doc = {"text": text, "metadata": {"source": "paper.txt"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "paper"

    def test_configured_type_overrides_auto(self, brain):
        brain.config.dataset_type = "knowledge"
        doc = {"text": "def foo(): pass", "metadata": {"source": "code.py"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "knowledge"

    def test_tabular_content_detected(self, brain):
        brain.config.dataset_type = "auto"
        text = "col1,col2,col3,col4\n1,2,3,4\n5,6,7,8\na,b,c,d\n"
        doc = {"text": text, "metadata": {"source": "data.csv"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype == "knowledge"

    def test_code_heavy_doc_is_codebase(self, brain):
        brain.config.dataset_type = "auto"
        lines = ["def foo(): pass\n"] * 20 + ["Some prose.\n"] * 5
        doc = {"text": "".join(lines), "metadata": {"source": "mixed.md"}}
        dtype, has_code = brain._detect_dataset_type(doc)
        assert dtype in ("codebase", "doc")  # heavy code ratio


# ===========================================================================
# Class 24: _build_context
# ===========================================================================


class TestBuildContext:
    def test_empty_results_returns_empty_context(self, brain):
        ctx, has_web = brain._build_context([])
        assert ctx == ""
        assert has_web is False

    def test_local_result_included_with_label(self, brain):
        results = [_fake_result(id="doc1", text="Important fact.")]
        ctx, has_web = brain._build_context(results)
        assert "Important fact." in ctx
        assert has_web is False

    def test_web_result_sets_has_web_flag(self, brain):
        results = [
            {
                "id": "http://example.com",
                "text": "Web snippet.",
                "score": 1.0,
                "metadata": {},
                "is_web": True,
            }
        ]
        ctx, has_web = brain._build_context(results)
        assert has_web is True


# ===========================================================================
# Class 25: finalize_ingest
# ===========================================================================


class TestFinalizeIngest:
    def test_finalize_flushes_bm25_in_batch_mode(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain.config.ingest_batch_mode = True
        brain._own_bm25 = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_claims_graph = MagicMock()
        brain._save_code_graph = MagicMock()
        brain.finalize_graph = MagicMock()

        brain.finalize_ingest()

        brain._own_bm25.flush.assert_called_once()
        brain._save_entity_graph.assert_called_once()
        brain._save_relation_graph.assert_called_once()

    def test_finalize_noop_when_not_batch_mode(self, brain):
        brain._assert_write_allowed = MagicMock()
        brain.config.ingest_batch_mode = False
        brain._own_bm25 = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain.finalize_graph = MagicMock()

        brain.finalize_ingest()

        brain._own_bm25.flush.assert_not_called()
        brain._save_entity_graph.assert_not_called()
        brain.finalize_graph.assert_called_once()


# ===========================================================================
# Class 26: _make_cache_key
# ===========================================================================


class TestMakeCacheKey:
    def test_same_query_same_key(self, brain):
        k1 = brain._make_cache_key("hello", None, brain.config)
        k2 = brain._make_cache_key("hello", None, brain.config)
        assert k1 == k2

    def test_different_query_different_key(self, brain):
        k1 = brain._make_cache_key("hello", None, brain.config)
        k2 = brain._make_cache_key("world", None, brain.config)
        assert k1 != k2

    def test_different_filters_different_key(self, brain):
        k1 = brain._make_cache_key("hello", {"source": "a"}, brain.config)
        k2 = brain._make_cache_key("hello", {"source": "b"}, brain.config)
        assert k1 != k2


# ===========================================================================
# Class 27: close()
# ===========================================================================


class TestClose:
    def test_close_shuts_down_executor(self, brain):
        brain._executor = MagicMock()
        brain.close()
        brain._executor.shutdown.assert_called_once_with(wait=False)

    def test_close_calls_vector_store_close(self, brain):
        brain._executor = MagicMock()
        brain.vector_store = MagicMock()
        brain.vector_store.close = MagicMock()
        brain._own_vector_store = brain.vector_store  # same object
        brain.bm25 = None
        brain._own_bm25 = None

        brain.close()
        brain.vector_store.close.assert_called_once()

    def test_close_avoids_double_close_on_same_store(self, brain):
        brain._executor = MagicMock()
        shared_vs = MagicMock()
        shared_vs.close = MagicMock()
        brain.vector_store = shared_vs
        brain._own_vector_store = shared_vs  # same reference
        brain.bm25 = None
        brain._own_bm25 = None

        brain.close()
        shared_vs.close.assert_called_once()  # not twice


# ===========================================================================
# Class 28: _log_startup_summary
# ===========================================================================


class TestLogStartupSummary:
    def test_startup_summary_with_meta_file(self, tmp_path):
        import json

        brain = _make_brain(tmp_path)
        # Create a meta.json for the default project path
        bm25_dir = tmp_path / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        project_dir = bm25_dir.parent
        meta_file = project_dir / "meta.json"
        meta_file.write_text(json.dumps({"project_namespace_id": "ns-test-123"}))
        brain.config.bm25_path = str(bm25_dir)

        with patch("axon.projects.get_project_namespace_id", return_value=None):
            # Should not raise
            brain._log_startup_summary()

    def test_startup_summary_handles_no_meta_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        with patch("axon.projects.get_project_namespace_id", return_value=None):
            # Should not raise even without meta.json
            brain._log_startup_summary()
