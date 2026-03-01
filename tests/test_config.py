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
