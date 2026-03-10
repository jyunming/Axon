"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path
from rag_brain.main import OpenStudioConfig


class TestOpenStudioConfig:
    """Test the OpenStudioConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OpenStudioConfig()

        assert config.embedding_provider == "sentence_transformers"
        assert config.llm_provider == "ollama"
        assert config.vector_store == "chroma"
        assert config.top_k == 10
        assert config.chunk_size == 1000

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "embedding": {
                    "provider": "ollama",
                    "model": "nomic-embed-text"
                },
                "llm": {
                    "provider": "ollama",
                    "model": "llama3.1",
                    "temperature": 0.5
                },
                "rag": {
                    "top_k": 5,
                    "hybrid_search": False
                }
            }
            yaml.dump(config_data, f)
            f.flush()

            config = OpenStudioConfig.load(f.name)

            assert config.embedding_provider == "ollama"
            assert config.embedding_model == "nomic-embed-text"
            assert config.llm_temperature == 0.5
            assert config.top_k == 5
            assert config.hybrid_search is False

    def test_load_nonexistent_config(self):
        """Test loading from nonexistent file returns defaults."""
        config = OpenStudioConfig.load("nonexistent_config.yaml")

        assert isinstance(config, OpenStudioConfig)
        assert config.embedding_provider == "sentence_transformers"

    def test_yaml_query_transformations_step_back(self):
        """step_back is loaded from query_transformations section in YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"query_transformations": {"step_back": True, "hyde": True}}, f)
            f.flush()
            config = OpenStudioConfig.load(f.name)
        assert config.step_back is True
        assert config.hyde is True

    def test_yaml_rag_section_parent_chunk_size(self):
        """parent_chunk_size is loaded from rag section in YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"rag": {"parent_chunk_size": 2000, "top_k": 5}}, f)
            f.flush()
            config = OpenStudioConfig.load(f.name)
        assert config.parent_chunk_size == 2000
        assert config.top_k == 5

    def test_yaml_rag_section_caching_and_dedup(self):
        """query_cache and dedup_on_ingest are loaded from rag section in YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"rag": {"query_cache": True, "query_cache_size": 64, "dedup_on_ingest": False}}, f)
            f.flush()
            config = OpenStudioConfig.load(f.name)
        assert config.query_cache is True
        assert config.query_cache_size == 64
        assert config.dedup_on_ingest is False

    def test_yaml_query_decompose_and_compress(self):
        """query_decompose and compress_context are loaded from their YAML sections."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "query_transformations": {"query_decompose": True},
                "context_compression": {"enabled": True},
            }, f)
            f.flush()
            config = OpenStudioConfig.load(f.name)
        assert config.query_decompose is True
        assert config.compress_context is True

    def test_yaml_rerank_model_bge(self):
        """reranker_model is loaded from rerank.model in YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"rerank": {"enabled": True, "model": "BAAI/bge-reranker-v2-m3"}}, f)
            f.flush()
            config = OpenStudioConfig.load(f.name)
        assert config.rerank is True
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
