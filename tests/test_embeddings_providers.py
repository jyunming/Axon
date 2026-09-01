"""Tests for OpenEmbedding providers in axon.embeddings."""
import os
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig


def _make_config(**kwargs):
    defaults = {
        "bm25_path": "/tmp/bm25",
        "vector_store_path": "/tmp/vs",
        "embedding_provider": "sentence_transformers",
        "embedding_model": "all-MiniLM-L6-v2",
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


class TestOpenEmbeddingSentenceTransformers:
    def test_init_loads_model(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
        ):
            emb = OpenEmbedding(cfg)
            assert emb.dimension == 384
            assert emb.model is mock_model

    def test_init_uses_model_path_when_set(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config()
        cfg.embedding_model_path = "/local/model"
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
        ):
            emb = OpenEmbedding(cfg)
            # An explicit embedding_model_path always loads online (local_files_only
            # is only inferred from the HF hub cache for bare model ids).
            mock_st.assert_called_once_with("/local/model", local_files_only=False)
            assert emb.dimension == 768

    def test_embed_returns_list(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
        ):
            emb = OpenEmbedding(cfg)
            result = emb.embed(["hello", "world"])
            assert isinstance(result, list)
            assert len(result) == 2

    def test_embed_without_tolist(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]  # plain list
        mock_st = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
        ):
            emb = OpenEmbedding(cfg)
            result = emb.embed(["hello"])
            assert isinstance(result, list)

    def test_embed_query_returns_single(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1, 0.2]]
        mock_st = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
        ):
            emb = OpenEmbedding(cfg)
            result = emb.embed_query("hello")
            assert isinstance(result, list)


class TestOpenEmbeddingOllama:
    def test_init_known_model(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="ollama", embedding_model="nomic-embed-text")
        emb = OpenEmbedding(cfg)
        assert emb.dimension == 768
        assert emb.provider == "ollama"

    def test_init_unknown_model_defaults_768(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="ollama", embedding_model="custom-unknown-model")
        emb = OpenEmbedding(cfg)
        assert emb.dimension == 768

    def test_embed_calls_ollama_client(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="ollama", embedding_model="nomic-embed-text")
        emb = OpenEmbedding(cfg)

        mock_client = MagicMock()
        # New batch API: client.embed(model=..., input=texts) -> {"embeddings": [...]}
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_ollama = MagicMock(Client=MagicMock(return_value=mock_client))

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = emb.embed(["test text"])
            assert result == [[0.1, 0.2, 0.3]]

    def test_embed_multiple_texts(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="ollama", embedding_model="nomic-embed-text")
        emb = OpenEmbedding(cfg)

        mock_client = MagicMock()
        # New batch API: single call returns all embeddings
        mock_client.embed.return_value = {"embeddings": [[0.1], [0.2]]}
        mock_ollama = MagicMock(Client=MagicMock(return_value=mock_client))

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = emb.embed(["a", "b"])
            assert result == [[0.1], [0.2]]


class TestOpenEmbeddingFastembed:
    def test_init_loads_model(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-small-en-v1.5")
        mock_te = MagicMock()
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            # No explicit embedding_model_path -> cache_dir defaults to
            # <axon_store_base>/model_cache/fastembed; assert the model name and
            # that some stable cache_dir was passed, not the exact machine path.
            call_kwargs = mock_te_cls.call_args.kwargs
            assert call_kwargs["model_name"] == "BAAI/bge-small-en-v1.5"
            assert call_kwargs["cache_dir"].endswith(os.path.join("model_cache", "fastembed"))
            assert emb.dimension == 384

    def test_init_with_cache_dir(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-small-en-v1.5")
        cfg.embedding_model_path = "/cache/dir"
        mock_te = MagicMock()
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            OpenEmbedding(cfg)
            mock_te_cls.assert_called_once_with(
                model_name="BAAI/bge-small-en-v1.5", cache_dir="/cache/dir"
            )

    def test_embed(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-small-en-v1.5")
        mock_te = MagicMock()
        vec = MagicMock()
        vec.tolist.return_value = [0.1, 0.2]
        mock_te.embed.return_value = [vec]
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            result = emb.embed(["hello"])
            assert result == [[0.1, 0.2]]


class TestOpenEmbeddingOpenAI:
    def test_init_sets_dimension(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="openai", embedding_model="text-embedding-3-small")
        mock_client = MagicMock()
        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            emb = OpenEmbedding(cfg)
            assert emb.dimension == 1536

    def test_init_with_api_key(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="openai", embedding_model="text-embedding-3-small")
        cfg.api_key = "sk-test-key"
        mock_client = MagicMock()
        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            OpenEmbedding(cfg)
            call_kwargs = mock_openai_cls.call_args[1]
            assert call_kwargs.get("api_key") == "sk-test-key"

    def test_init_with_custom_base_url(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="openai", embedding_model="text-embedding-3-small")
        cfg.ollama_base_url = "http://custom:8080"
        mock_client = MagicMock()
        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            OpenEmbedding(cfg)
            call_kwargs = mock_openai_cls.call_args[1]
            assert call_kwargs.get("base_url") == "http://custom:8080"

    def test_embed_calls_api(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="openai", embedding_model="text-embedding-3-small")
        mock_client = MagicMock()
        data_item = MagicMock()
        data_item.embedding = [0.1, 0.2]
        mock_client.embeddings.create.return_value.data = [data_item]
        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            emb = OpenEmbedding(cfg)
            result = emb.embed(["hello"])
            assert result == [[0.1, 0.2]]


class TestBGEM3FastEmbed:
    """Story 4.1 — BGE-M3 dense support hardening."""

    def test_bge_m3_resolves_1024_dim(self):
        """BAAI/bge-m3 is in _KNOWN_DIMS and must always resolve to 1024."""
        from axon.embeddings import _KNOWN_DIMS, OpenEmbedding

        assert _KNOWN_DIMS["BAAI/bge-m3"] == 1024

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-m3")
        mock_te = MagicMock()
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            # Dimension must come from registry, not from a probe call
            mock_te.embed.assert_not_called()
            assert emb.dimension == 1024

    def test_bge_m3_embed_produces_1024_dim_vectors(self):
        """embed() for BGE-M3 must return 1024-element vectors."""
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-m3")
        mock_te = MagicMock()
        vec = MagicMock()
        vec.tolist.return_value = list(range(1024))
        mock_te.embed.return_value = [vec]
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            result = emb.embed(["test sentence"])
            assert len(result[0]) == 1024

    def test_bge_m3_embed_query_returns_single_vector(self):
        """embed_query() for BGE-M3 returns a flat list, not a list of lists."""
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-m3")
        mock_te = MagicMock()
        vec = MagicMock()
        vec.tolist.return_value = list(range(1024))
        mock_te.embed.return_value = [vec]
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            result = emb.embed_query("query")
            assert isinstance(result, list)
            assert len(result) == 1024

    def test_unknown_fastembed_model_auto_detects_dimension(self):
        """A model not in _KNOWN_DIMS auto-detects its dimension via a probe embedding."""
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="custom/my-model-512")
        mock_te = MagicMock()
        probe_vec = MagicMock()
        probe_vec.__len__ = lambda self: 512
        # probe call: embed(["dim-probe"]) → list with one vector of length 512
        mock_te.embed.return_value = iter([[0.0] * 512])
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            assert emb.dimension == 512

    def test_unknown_fastembed_model_probe_fallback_on_empty(self):
        """If the probe returns an empty list, dimension falls back to 384 gracefully."""
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="custom/empty-probe")
        mock_te = MagicMock()
        mock_te.embed.return_value = iter([])  # empty probe result
        mock_te_cls = MagicMock(return_value=mock_te)

        with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_te_cls)}):
            emb = OpenEmbedding(cfg)
            assert emb.dimension == 384

    def test_fastembed_import_error_gives_actionable_message(self):
        """Missing fastembed dependency raises ImportError with install hint."""
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="fastembed", embedding_model="BAAI/bge-m3")

        with patch.dict("sys.modules", {"fastembed": None}):
            with pytest.raises(ImportError, match="pip install"):
                OpenEmbedding(cfg)


class TestOllamaUnknownModelWarning:
    def test_unknown_ollama_model_logs_warning(self, caplog):
        """Ollama with a model not in _KNOWN_DIMS must log a warning, not fail silently."""
        import logging

        from axon.embeddings import OpenEmbedding

        cfg = _make_config(
            embedding_provider="ollama", embedding_model="totally-unknown-embed-model"
        )
        with caplog.at_level(logging.WARNING, logger="Axon"):
            emb = OpenEmbedding(cfg)
        assert emb.dimension == 768  # fallback
        assert any("not in the dimension registry" in r.message for r in caplog.records)

    def test_known_ollama_model_no_warning(self, caplog):
        """Ollama with a known model must not log the unknown-model warning."""
        import logging

        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="ollama", embedding_model="nomic-embed-text")
        with caplog.at_level(logging.WARNING, logger="Axon"):
            emb = OpenEmbedding(cfg)
        assert emb.dimension == 768
        assert not any("not in the dimension registry" in r.message for r in caplog.records)


class TestOpenEmbeddingUnknown:
    def test_unknown_provider_raises(self):
        from axon.embeddings import OpenEmbedding

        cfg = _make_config(embedding_provider="unknown_provider")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            OpenEmbedding(cfg)


class TestEmbeddingIdentity:
    """embedding_identity() must only equate provider/model pairs actually
    verified numerically identical — see _VERIFIED_CROSS_PROVIDER_MODELS."""

    def test_verified_model_equivalent_across_providers(self):
        from axon.embeddings import embedding_identity

        assert embedding_identity(
            "sentence_transformers", "all-MiniLM-L6-v2"
        ) == embedding_identity("fastembed", "sentence-transformers/all-MiniLM-L6-v2")

    def test_unverified_model_not_equivalent_across_providers(self):
        """A bare sentence-transformers-org model that was never verified
        against its fastembed ONNX export must NOT be silently treated as the
        same embedding — that would let _validate_embedding_meta wave through
        a real mismatch."""
        from axon.embeddings import embedding_identity

        assert embedding_identity(
            "sentence_transformers", "all-mpnet-base-v2"
        ) != embedding_identity("fastembed", "sentence-transformers/all-mpnet-base-v2")

    def test_different_models_not_equivalent(self):
        from axon.embeddings import embedding_identity

        assert embedding_identity(
            "sentence_transformers", "all-MiniLM-L6-v2"
        ) != embedding_identity("fastembed", "BAAI/bge-small-en-v1.5")

    def test_unrelated_provider_unaffected(self):
        from axon.embeddings import embedding_identity

        assert embedding_identity("ollama", "nomic-embed-text") == (
            "ollama",
            "nomic-embed-text",
        )


class TestIsHfModelCached:
    def test_guess_st_prefix_default_true(self, tmp_path, monkeypatch):
        from axon.embeddings import is_hf_model_cached

        monkeypatch.setenv("HF_HOME", str(tmp_path))
        (tmp_path / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir(parents=True)
        assert is_hf_model_cached("all-MiniLM-L6-v2") is True

    def test_guess_st_prefix_false_misses_bare_name(self, tmp_path, monkeypatch):
        """With guess_st_prefix=False, a bare id that only exists under the
        sentence-transformers org slug must NOT be reported as cached — this
        is what keeps the reranker/gliner/rebel/llmlingua audit rows honest."""
        from axon.embeddings import is_hf_model_cached

        monkeypatch.setenv("HF_HOME", str(tmp_path))
        (tmp_path / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir(parents=True)
        assert is_hf_model_cached("all-MiniLM-L6-v2", guess_st_prefix=False) is False

    def test_exact_slug_match_regardless_of_guess_flag(self, tmp_path, monkeypatch):
        from axon.embeddings import is_hf_model_cached

        monkeypatch.setenv("HF_HOME", str(tmp_path))
        (tmp_path / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2").mkdir(parents=True)
        assert (
            is_hf_model_cached("cross-encoder/ms-marco-MiniLM-L-6-v2", guess_st_prefix=False)
            is True
        )


class TestIsFastembedModelCached:
    def test_known_default_model_maps_to_qdrant_repo(self, tmp_path):
        from axon.embeddings import is_fastembed_model_cached

        (tmp_path / "models--qdrant--all-MiniLM-L6-v2-onnx").mkdir(parents=True)
        assert (
            is_fastembed_model_cached("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
            is True
        )

    def test_known_default_model_uses_static_map_without_importing_fastembed(
        self, tmp_path, monkeypatch
    ):
        """The static _KNOWN_FASTEMBED_SOURCES entry must short-circuit before
        ever touching fastembed's catalog — that import is exactly the ~1s
        cost this map exists to avoid paying on every boot."""
        import sys

        from axon.embeddings import is_fastembed_model_cached

        (tmp_path / "models--qdrant--all-MiniLM-L6-v2-onnx").mkdir(parents=True)
        monkeypatch.setitem(sys.modules, "fastembed", None)  # import fastembed -> ImportError
        assert (
            is_fastembed_model_cached("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
            is True
        )

    def test_missing_cache_dir_returns_false(self, tmp_path):
        from axon.embeddings import is_fastembed_model_cached

        assert (
            is_fastembed_model_cached(
                "sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "nope")
            )
            is False
        )

    def test_unknown_model_not_in_cache_returns_false(self, tmp_path):
        from axon.embeddings import is_fastembed_model_cached

        (tmp_path / "models--qdrant--all-MiniLM-L6-v2-onnx").mkdir(parents=True)
        assert is_fastembed_model_cached("some/other-model", str(tmp_path)) is False

    def test_malformed_catalog_entry_does_not_crash(self, tmp_path, monkeypatch):
        """A non-dict entry from a future fastembed version must not raise —
        the audit falls back to the public model_id rather than crashing
        AxonBrain startup (fastembed is the default provider now, so this
        path runs on every boot)."""
        import sys
        import types

        from axon.embeddings import is_fastembed_model_cached

        # Nothing under this cache_dir matches "some/other-model" itself (the
        # fallback repo when the catalog lookup can't resolve a real source) —
        # only a cache dir for an unrelated model exists, so a correct,
        # non-crashing fallback must still return False.
        (tmp_path / "models--unrelated--model").mkdir(parents=True)
        fake_fastembed = types.ModuleType("fastembed")
        fake_fastembed.TextEmbedding = type(
            "FakeTextEmbedding",
            (),
            {"list_supported_models": staticmethod(lambda: ["not-a-dict"])},
        )
        monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
        # "some/other-model" isn't in the static map, so this exercises the
        # dynamic catalog lookup with a malformed (non-dict) entry.
        assert is_fastembed_model_cached("some/other-model", str(tmp_path)) is False
