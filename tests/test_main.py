import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, MagicMock


class TestOpenStudioConfig:
    def test_defaults(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig()
        assert config.embedding_provider == "sentence_transformers"
        assert config.llm_provider == "ollama"
        assert config.vector_store == "chroma"
        assert config.hybrid_search is True
        assert config.top_k == 10
        assert config.chunk_size == 1000

    def test_load_from_yaml(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        cfg = {
            "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
            "llm": {"provider": "ollama", "model": "phi3:mini", "temperature": 0.5},
            "vector_store": {"provider": "chroma", "path": str(tmp_path / "chroma")},
            "bm25": {"path": str(tmp_path / "bm25")},
            "rag": {"top_k": 5, "similarity_threshold": 0.6, "hybrid_search": False},
            "chunk": {"size": 500, "overlap": 50},
            "rerank": {"enabled": False, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        config = OpenStudioConfig.load(str(cfg_path))
        assert config.embedding_provider == "fastembed"
        assert config.embedding_model == "BAAI/bge-small-en-v1.5"
        assert config.llm_model == "phi3:mini"
        assert config.llm_temperature == 0.5
        assert config.top_k == 5
        assert config.similarity_threshold == 0.6
        assert config.hybrid_search is False
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50

    def test_load_missing_file_uses_defaults(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig.load("/nonexistent/path/config.yaml")
        assert config.embedding_provider == "sentence_transformers"


@patch("rag_brain.retrievers.BM25Retriever")
@patch("rag_brain.main.OpenVectorStore")
@patch("rag_brain.main.OpenLLM")
@patch("rag_brain.main.OpenEmbedding")
@patch("rag_brain.main.OpenReranker")
class TestOpenStudioBrainQuery:
    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0, top_k=3)
        brain = OpenStudioBrain(config)
        return brain

    def test_query_returns_string(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = OpenStudioBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1, 0.2])
        brain.vector_store.search = MagicMock(return_value=[
            {"id": "doc1", "text": "hello world", "score": 0.9, "metadata": {}}
        ])
        brain.llm.complete = MagicMock(return_value="Test answer")

        result = brain.query("test question")
        assert isinstance(result, str)
        assert result == "Test answer"

    def test_query_returns_no_results_message_when_empty(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = OpenStudioBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1, 0.2])
        brain.vector_store.search = MagicMock(return_value=[])

        result = brain.query("something with no results")
        assert "don't have" in result.lower() or "no" in result.lower()

    def test_ingest_calls_vector_store_add(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False, chunk_size=10000)
        brain = OpenStudioBrain(config)
        brain.splitter = None  # skip splitting

        docs = [{"id": f"doc{i}", "text": f"text {i}", "metadata": {}} for i in range(5)]
        brain.embedding.embed = MagicMock(return_value=[[0.1] * 3] * 5)
        brain.vector_store.add = MagicMock()

        brain.ingest(docs)
        brain.vector_store.add.assert_called_once()
        call_args = brain.vector_store.add.call_args[0]
        assert len(call_args[0]) == 5  # ids
        assert len(call_args[1]) == 5  # texts
