"""Tests for configuration management."""


import yaml

from axon.main import AxonConfig


class TestAxonConfig:

    """Test the AxonConfig class."""

    def test_default_config(self):
        """Test default configuration values."""

        config = AxonConfig()

        assert config.embedding_provider == "sentence_transformers"

        assert config.embedding_model == "all-MiniLM-L6-v2"

        assert config.llm_provider == "ollama"

        assert config.vector_store == "lancedb"

        assert config.top_k == 10

        assert config.chunk_size == 1000

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""

        config_path = tmp_path / "config.yaml"

        config_data = {
            "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
            "llm": {"provider": "ollama", "model": "llama3.1", "temperature": 0.5},
            "rag": {"top_k": 5, "hybrid_search": False},
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = AxonConfig.load(str(config_path))

        assert config.embedding_provider == "ollama"

        assert config.embedding_model == "nomic-embed-text"

        assert config.llm_temperature == 0.5

        assert config.top_k == 5

        assert config.hybrid_search is False

    def test_load_nonexistent_config(self):
        """Test loading from nonexistent file returns defaults."""

        config = AxonConfig.load("nonexistent_config.yaml")

        assert isinstance(config, AxonConfig)

        assert config.embedding_provider == "sentence_transformers"

    def test_yaml_query_transformations_step_back(self, tmp_path):
        """step_back is loaded from query_transformations section in YAML."""

        config_path = tmp_path / "config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"query_transformations": {"step_back": True, "hyde": True}}, f)

        config = AxonConfig.load(str(config_path))

        assert config.step_back is True

        assert config.hyde is True

    def test_yaml_rag_section_parent_chunk_size(self, tmp_path):
        """parent_chunk_size is loaded from rag section in YAML."""

        config_path = tmp_path / "config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"rag": {"parent_chunk_size": 2000, "top_k": 5}}, f)

        config = AxonConfig.load(str(config_path))

        assert config.parent_chunk_size == 2000

        assert config.top_k == 5

    def test_yaml_rag_section_caching_and_dedup(self, tmp_path):
        """query_cache and dedup_on_ingest are loaded from rag section in YAML."""

        config_path = tmp_path / "config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"rag": {"query_cache": True, "query_cache_size": 64, "dedup_on_ingest": False}}, f
            )

        config = AxonConfig.load(str(config_path))

        assert config.query_cache is True

        assert config.query_cache_size == 64

        assert config.dedup_on_ingest is False

    def test_yaml_query_decompose_and_compress(self, tmp_path):
        """query_decompose and compress_context are loaded from their YAML sections."""

        config_path = tmp_path / "config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "query_transformations": {"query_decompose": True},
                    "context_compression": {"enabled": True},
                },
                f,
            )

        config = AxonConfig.load(str(config_path))

        assert config.query_decompose is True

        assert config.compress_context is True

    def test_yaml_rerank_model_bge(self, tmp_path):
        """reranker_model is loaded from rerank.model in YAML."""

        config_path = tmp_path / "config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"rerank": {"enabled": True, "model": "BAAI/bge-reranker-v2-m3"}}, f)

        config = AxonConfig.load(str(config_path))

        assert config.rerank is True

        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
