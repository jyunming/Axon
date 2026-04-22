"""Tests for OpenEmbedding providers in axon.embeddings."""
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
            mock_st.assert_called_once_with("/local/model")
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
            mock_te_cls.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")
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
