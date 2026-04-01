from __future__ import annotations

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


"""


tests/test_config_extra.py


Comprehensive tests for src/axon/config.py covering missed lines:


- 97, 112, 116, 125: __post_init__ env-var branches


- 162-163, 171, 173-181: __post_init__ WSL/AxonStore branches


- 587-593: load() default-path creation / permission error


- 657, 659, 661, 665, 684, 687, 690, 693, 696-698, 708, 711: load() nested-section parsing


- 729-821: save() method


"""


import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------


# Helpers


# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ===========================================================================


# 1. save() method --" lines 727-821


# ===========================================================================


class TestSave:

    """Tests for AxonConfig.save()."""

    def test_save_to_explicit_path(self, tmp_path):
        """save(path) writes a valid YAML file at the given path."""

        cfg = AxonConfig()

        target = tmp_path / "config.yaml"

        cfg.save(str(target))

        assert target.exists()

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "embedding" in data

        assert data["embedding"]["provider"] == "sentence_transformers"

    def test_save_creates_parent_directory(self, tmp_path):
        """save() calls os.makedirs to create missing parent dirs."""

        cfg = AxonConfig()

        nested = tmp_path / "deep" / "nested" / "config.yaml"

        cfg.save(str(nested))

        assert nested.exists()

    def test_save_then_load_round_trip(self, tmp_path):
        """save() + load() round-trip preserves core fields."""

        cfg = AxonConfig(
            embedding_provider="fastembed",
            embedding_model="BAAI/bge-small-en",
            llm_provider="openai",
            llm_model="gpt-4o",
            top_k=20,
            chunk_size=512,
            chunk_overlap=64,
            rerank=True,
            multi_query=True,
            hyde=True,
        )

        target = tmp_path / "config.yaml"

        cfg.save(str(target))

        loaded = AxonConfig.load(str(target))

        assert loaded.embedding_provider == "fastembed"

        assert loaded.embedding_model == "BAAI/bge-small-en"

        assert loaded.llm_provider == "openai"

        assert loaded.llm_model == "gpt-4o"

        assert loaded.top_k == 20

        assert loaded.chunk_size == 512

        assert loaded.chunk_overlap == 64

        assert loaded.rerank is True

        assert loaded.multi_query is True

        assert loaded.hyde is True

    def test_save_to_loaded_path_when_no_arg(self, tmp_path):
        """When no path argument is given, save() uses _loaded_path."""

        target = tmp_path / "myconfig.yaml"

        # Manually set _loaded_path to our temp file instead

        cfg2 = AxonConfig()

        cfg2._loaded_path = str(target)

        cfg2.save()

        assert target.exists()

    def test_save_writes_llm_section(self, tmp_path):
        """save() writes a nested llm section with provider/model/temperature/max_tokens."""

        cfg = AxonConfig(
            llm_provider="gemini", llm_model="gemini-pro", llm_temperature=0.2, llm_max_tokens=1024
        )

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["provider"] == "gemini"

        assert data["llm"]["model"] == "gemini-pro"

        assert data["llm"]["temperature"] == pytest.approx(0.2)

        assert data["llm"]["max_tokens"] == 1024

    def test_save_writes_vector_store_section(self, tmp_path):
        """save() writes the vector_store nested block."""

        cfg = AxonConfig(vector_store="qdrant")

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["vector_store"]["provider"] == "qdrant"

    def test_save_writes_store_section(self, tmp_path):
        """save() writes store.base (not bm25.path which is derived from it)."""

        cfg = AxonConfig()

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "store" in data

        assert "base" in data["store"]

        # bm25.path is not persisted -- derived at runtime from store.base

        if "bm25" in data:
            assert "path" not in data["bm25"]

    def test_save_writes_rag_section(self, tmp_path):
        """save() writes rag block with top_k, similarity_threshold, hybrid_search."""

        cfg = AxonConfig(top_k=15, similarity_threshold=0.5, hybrid_search=False)

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["rag"]["top_k"] == 15

        assert data["rag"]["similarity_threshold"] == pytest.approx(0.5)

        assert data["rag"]["hybrid_search"] is False

    def test_save_writes_chunk_section(self, tmp_path):
        """save() writes chunk block with strategy/size/overlap."""

        cfg = AxonConfig(chunk_strategy="markdown", chunk_size=800, chunk_overlap=100)

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["chunk"]["strategy"] == "markdown"

        assert data["chunk"]["size"] == 800

        assert data["chunk"]["overlap"] == 100

    def test_save_writes_rerank_section(self, tmp_path):
        """save() writes rerank block with enabled/provider/model."""

        cfg = AxonConfig(
            rerank=True, reranker_provider="cross-encoder", reranker_model="BAAI/bge-reranker-v2-m3"
        )

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["rerank"]["enabled"] is True

        assert data["rerank"]["provider"] == "cross-encoder"

        assert data["rerank"]["model"] == "BAAI/bge-reranker-v2-m3"

    def test_save_writes_query_transformations_section(self, tmp_path):
        """save() writes query_transformations block."""

        cfg = AxonConfig(multi_query=True, hyde=True, step_back=True, query_decompose=True)

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        qt = data["query_transformations"]

        assert qt["multi_query"] is True

        assert qt["hyde"] is True

        assert qt["step_back"] is True

        assert qt["query_decompose"] is True

    def test_save_writes_repl_section(self, tmp_path):
        """save() writes repl.shell_passthrough."""

        cfg = AxonConfig(repl_shell_passthrough="always")

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["repl"]["shell_passthrough"] == "always"

    def test_save_writes_context_compression_section(self, tmp_path):
        """save() writes context_compression block."""

        cfg = AxonConfig(compress_context=True)

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["context_compression"]["enabled"] is True

    def test_save_writes_web_search_section(self, tmp_path):
        """save() writes web_search block with enabled and brave_api_key."""

        cfg = AxonConfig(truth_grounding=True, brave_api_key="test-key")

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["web_search"]["enabled"] is True

        assert data["web_search"]["brave_api_key"] == "test-key"

    def test_save_writes_offline_section(self, tmp_path):
        """save() writes offline block with all sub-fields."""

        cfg = AxonConfig(
            offline_mode=True,
            local_models_dir="/models",
            local_assets_only=True,
            embedding_models_dir="/em",
            hf_models_dir="/hf",
            tokenizer_cache_dir="/tok",
        )

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        ol = data["offline"]

        assert ol["enabled"] is True

        assert ol["local_models_dir"] == "/models"

        assert ol["local_assets_only"] is True

        assert ol["embedding_models_dir"] == "/em"

        assert ol["hf_models_dir"] == "/hf"

        assert ol["tokenizer_cache_dir"] == "/tok"

    def test_save_includes_api_key_when_set(self, tmp_path):
        """save() includes llm.api_key only when non-empty."""

        cfg = AxonConfig(api_key="sk-test")

        # Force both fields so env vars don't pollute

        cfg.api_key = "sk-test"

        cfg.openai_api_key = "sk-test"

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["api_key"] == "sk-test"

    def test_save_omits_api_key_when_empty(self, tmp_path):
        """save() does NOT include llm.api_key when it is empty string."""

        cfg = AxonConfig(api_key="")

        # Force both fields so env vars don't pollute

        cfg.api_key = ""

        cfg.openai_api_key = ""

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "api_key" not in data["llm"]

    def test_save_includes_gemini_api_key(self, tmp_path):
        """save() serialises gemini_api_key under llm section."""

        cfg = AxonConfig()

        cfg.gemini_api_key = "gemini-abc"

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["gemini_api_key"] == "gemini-abc"

    def test_save_includes_ollama_cloud_key(self, tmp_path):
        """save() serialises ollama_cloud_key under llm section."""

        cfg = AxonConfig()

        cfg.ollama_cloud_key = "cloud-key-xyz"

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["ollama_cloud_key"] == "cloud-key-xyz"

    def test_save_includes_ollama_cloud_url(self, tmp_path):
        """save() serialises ollama_cloud_url under llm section."""

        cfg = AxonConfig()

        cfg.ollama_cloud_url = "https://custom.ollama.com/api"

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["ollama_cloud_url"] == "https://custom.ollama.com/api"

    def test_save_includes_vllm_base_url(self, tmp_path):
        """save() serialises vllm_base_url under llm section."""

        cfg = AxonConfig(vllm_base_url="http://vllm-host:8000/v1")

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["vllm_base_url"] == "http://vllm-host:8000/v1"

    def test_save_includes_llm_timeout(self, tmp_path):
        """save() serialises llm_timeout under llm.timeout."""

        cfg = AxonConfig(llm_timeout=120)

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["llm"]["timeout"] == 120

    def test_save_axon_store_base_adds_store_section(self, tmp_path):
        """When axon_store_base is set, save() writes store.base, removes projects_root,


        and omits vector_store.path / bm25.path so stale hardcoded paths are never


        persisted to config.yaml (they are always derived fresh from axon_store_base)."""

        cfg = AxonConfig()

        cfg.axon_store_base = str(tmp_path / "shared")

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "store" in data

        assert data["store"]["base"] == str(tmp_path / "shared")

        assert "projects_root" not in data

        assert "path" not in data.get("vector_store", {})

        assert "path" not in data.get("bm25", {})

    def test_save_no_store_section_when_axon_store_base_empty(self, tmp_path):
        """When axon_store_base is empty, save() writes projects_root and no store section."""

        cfg = AxonConfig()

        cfg.axon_store_base = ""

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "store" not in data

        assert "projects_root" in data

    def test_save_writes_store_base(self, tmp_path):
        """save() writes store.base (projects_root is derived, not persisted)."""

        cfg = AxonConfig()

        target = tmp_path / "c.yaml"

        cfg.save(str(target))

        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "store" in data

        assert "base" in data["store"]


# ===========================================================================


# 2. load() edge cases --" lines 585-725


# ===========================================================================


class TestLoad:

    """Tests for AxonConfig.load() including edge cases."""

    def test_load_nonexistent_explicit_path_returns_defaults(self, tmp_path):
        """load(path) where path does not exist returns a default AxonConfig."""

        nonexistent = str(tmp_path / "does_not_exist.yaml")

        cfg = AxonConfig.load(nonexistent)

        assert isinstance(cfg, AxonConfig)

        assert cfg.embedding_provider == "sentence_transformers"

    def test_load_empty_yaml_returns_defaults(self, tmp_path):
        """An empty YAML file (None result from safe_load) returns default config."""

        p = tmp_path / "empty.yaml"

        p.write_text("", encoding="utf-8")

        cfg = AxonConfig.load(str(p))

        assert isinstance(cfg, AxonConfig)

    def test_load_valid_yaml_parses_embedding_section(self, tmp_path):
        """load() correctly parses embedding.provider and embedding.model."""

        data = {"embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.embedding_provider == "fastembed"

        assert cfg.embedding_model == "BAAI/bge-small-en"

    def test_load_valid_yaml_parses_llm_section(self, tmp_path):
        """load() correctly parses llm section."""

        data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 512,
            }
        }

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.llm_provider == "openai"

        assert cfg.llm_model == "gpt-4o-mini"

        assert cfg.llm_temperature == pytest.approx(0.3)

        assert cfg.llm_max_tokens == 512

    def test_load_parses_vector_store_provider(self, tmp_path):
        """load() maps vector_store.provider to vector_store (path is always derived)."""

        data = {"vector_store": {"provider": "lancedb"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.vector_store == "lancedb"

        # vector_store_path is always derived from AxonStore, not from config.yaml

        assert cfg.vector_store_path != ""

    def test_load_paths_always_derived_from_store(self, tmp_path):
        """Paths in config.yaml are ignored -- always derived from AxonStore layout."""

        data = {"bm25": {"path": "/old/path/bm25"}, "vector_store": {"path": "/old/chroma"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert "AxonStore" in cfg.bm25_path or ".axon" in cfg.bm25_path

        assert "AxonStore" in cfg.vector_store_path or ".axon" in cfg.vector_store_path

    def test_load_parses_rag_section(self, tmp_path):
        """load() reads rag keys directly into config_dict."""

        data = {"rag": {"top_k": 25, "similarity_threshold": 0.4, "hybrid_search": False}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.top_k == 25

        assert cfg.similarity_threshold == pytest.approx(0.4)

        assert cfg.hybrid_search is False

    def test_load_parses_chunk_section(self, tmp_path):
        """load() maps chunk.size/overlap/strategy."""

        data = {"chunk": {"size": 600, "overlap": 50, "strategy": "markdown"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.chunk_size == 600

        assert cfg.chunk_overlap == 50

        assert cfg.chunk_strategy == "markdown"

    def test_load_parses_rerank_section(self, tmp_path):
        """load() maps rerank.enabled/provider/model."""

        data = {
            "rerank": {
                "enabled": True,
                "provider": "cross-encoder",
                "model": "BAAI/bge-reranker-v2-m3",
            }
        }

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.rerank is True

        assert cfg.reranker_provider == "cross-encoder"

        assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_load_parses_query_transformations_section(self, tmp_path):
        """load() reads multi_query, hyde, step_back, query_decompose, discussion_fallback."""

        data = {
            "query_transformations": {
                "multi_query": True,
                "hyde": True,
                "step_back": True,
                "query_decompose": True,
                "discussion_fallback": False,
            }
        }

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.multi_query is True

        assert cfg.hyde is True

        assert cfg.step_back is True

        assert cfg.query_decompose is True

        assert cfg.discussion_fallback is False

    def test_load_parses_repl_shell_passthrough(self, tmp_path):
        """load() maps repl.shell_passthrough -> repl_shell_passthrough."""

        data = {"repl": {"shell_passthrough": "off"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.repl_shell_passthrough == "off"

    def test_load_parses_context_compression_section(self, tmp_path):
        """load() maps context_compression.enabled â†' compress_context."""

        data = {"context_compression": {"enabled": True}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.compress_context is True

    def test_load_parses_web_search_section(self, tmp_path):
        """load() reads truth_grounding and brave_api_key from web_search section."""

        data = {"web_search": {"enabled": True, "brave_api_key": "bk-abc123"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.truth_grounding is True

        assert cfg.brave_api_key == "bk-abc123"

    def test_load_parses_offline_section_local_models_dir(self, tmp_path):
        """load() reads offline.local_models_dir."""

        data = {"offline": {"enabled": True, "local_models_dir": "/offline/models"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.offline_mode is True

        assert cfg.local_models_dir == "/offline/models"

    def test_load_parses_offline_section_local_assets_only(self, tmp_path):
        """load() reads offline.local_assets_only."""

        data = {"offline": {"local_assets_only": True}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.local_assets_only is True

    def test_load_parses_offline_section_embedding_models_dir(self, tmp_path):
        """load() reads offline.embedding_models_dir."""

        data = {"offline": {"embedding_models_dir": "/em/dir"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.embedding_models_dir == "/em/dir"

    def test_load_parses_offline_section_hf_models_dir(self, tmp_path):
        """load() reads offline.hf_models_dir."""

        data = {"offline": {"hf_models_dir": "/hf/dir"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.hf_models_dir == "/hf/dir"

    def test_load_parses_offline_section_tokenizer_cache_dir(self, tmp_path):
        """load() reads offline.tokenizer_cache_dir."""

        data = {"offline": {"tokenizer_cache_dir": "/tok/dir"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.tokenizer_cache_dir == "/tok/dir"

    def test_load_llm_base_url_mapped_to_ollama_base_url(self, tmp_path):
        """load() maps llm.base_url â†' ollama_base_url when no explicit ollama_base_url given."""

        data = {"llm": {"base_url": "http://custom-ollama:11434"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.ollama_base_url == "http://custom-ollama:11434"

    def test_load_llm_models_dir_mapped_to_ollama_models_dir(self, tmp_path, monkeypatch):
        """load() maps llm.models_dir â†' ollama_models_dir."""
        monkeypatch.delenv("OLLAMA_MODELS", raising=False)

        data = {"llm": {"models_dir": "/ollama/models"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.ollama_models_dir == "/ollama/models"

    def test_load_llm_api_key_mapped_to_api_key(self, tmp_path):
        """load() maps llm.api_key â†' api_key when no top-level api_key given."""

        data = {"llm": {"api_key": "sk-from-llm-section"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.api_key == "sk-from-llm-section"

    def test_load_llm_vllm_base_url_mapped(self, tmp_path):
        """load() maps llm.vllm_base_url â†' vllm_base_url."""

        data = {"llm": {"vllm_base_url": "http://vllm:9000/v1"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.vllm_base_url == "http://vllm:9000/v1"

    def test_load_store_base_top_level(self, tmp_path):
        """load() reads store.base and derives projects_root from it."""

        store_base = str(tmp_path / "mystore")

        data = {"store": {"base": store_base}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.axon_store_base == store_base

        assert "AxonStore" in cfg.projects_root

    def test_load_max_workers_top_level(self, tmp_path):
        """load() reads top-level max_workers."""

        data = {"max_workers": 16}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.max_workers == 16

    def test_load_ingest_batch_mode_top_level(self, tmp_path):
        """load() reads top-level ingest_batch_mode."""

        data = {"ingest_batch_mode": True}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.ingest_batch_mode is True

    def test_load_max_chunks_per_source_top_level(self, tmp_path):
        """load() reads top-level max_chunks_per_source."""

        data = {"max_chunks_per_source": 50}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.max_chunks_per_source == 50

    def test_load_source_policy_enabled_top_level(self, tmp_path):
        """load() reads top-level source_policy_enabled."""

        data = {"source_policy_enabled": True}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.source_policy_enabled is True

    def test_load_store_section_sets_axon_store_base(self, tmp_path):
        """load() reads store.base â†' axon_store_base."""

        data = {"store": {"base": "/shared/axon"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.axon_store_base == "/shared/axon"

    def test_load_unknown_keys_are_ignored(self, tmp_path):
        """load() silently ignores YAML keys that are not dataclass fields."""

        data = {"totally_unknown_key": "some_value", "embedding": {"provider": "ollama"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg.embedding_provider == "ollama"

        assert not hasattr(cfg, "totally_unknown_key")

    def test_load_sets_loaded_path(self, tmp_path):
        """load() stores the resolved path in _loaded_path."""

        data = {"embedding": {"provider": "ollama"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        cfg = AxonConfig.load(str(p))

        assert cfg._loaded_path == str(p)

    def test_load_env_ollama_host_overrides_config(self, tmp_path, monkeypatch):
        """OLLAMA_HOST env var overrides ollama_base_url even when yaml specifies a different URL."""

        data = {"llm": {"base_url": "http://yaml-ollama:11434"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        monkeypatch.setenv("OLLAMA_HOST", "http://env-ollama:11434")

        cfg = AxonConfig.load(str(p))

        assert cfg.ollama_base_url == "http://env-ollama:11434"

    def test_load_env_vllm_base_url_overrides(self, tmp_path, monkeypatch):
        """VLLM_BASE_URL env var overrides vllm_base_url from yaml."""

        data = {"llm": {"vllm_base_url": "http://yaml-vllm:8000/v1"}}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        monkeypatch.setenv("VLLM_BASE_URL", "http://env-vllm:9000/v1")

        cfg = AxonConfig.load(str(p))

        assert cfg.vllm_base_url == "http://env-vllm:9000/v1"

    def test_load_env_axon_store_base_overrides(self, tmp_path, monkeypatch):
        """AXON_STORE_BASE env var overrides store base and derives projects_root."""

        env_base = str(tmp_path / "env_store")

        data = {}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        monkeypatch.setenv("AXON_STORE_BASE", env_base)

        cfg = AxonConfig.load(str(p))

        assert cfg.axon_store_base == env_base

        assert "AxonStore" in cfg.projects_root

    def test_load_env_ollama_models_overrides(self, tmp_path, monkeypatch):
        """OLLAMA_MODELS env var overrides ollama_models_dir."""

        data = {}

        p = tmp_path / "c.yaml"

        _write_yaml(p, data)

        monkeypatch.setenv("OLLAMA_MODELS", "/env/ollama/models")

        cfg = AxonConfig.load(str(p))

        assert cfg.ollama_models_dir == "/env/ollama/models"

    def test_load_default_path_missing_creates_file(self, tmp_path, monkeypatch):
        """When load(None) is called and default path is missing, it attempts to create it."""

        fake_cfg_dir = tmp_path / ".config" / "axon"

        fake_cfg_path = fake_cfg_dir / "config.yaml"

        monkeypatch.setattr("axon.config._USER_CONFIG_PATH", str(fake_cfg_path))

        cfg = AxonConfig.load(None)

        assert isinstance(cfg, AxonConfig)

        # File should have been created

        assert fake_cfg_path.exists()

    def test_load_default_path_permission_error_returns_defaults(self, tmp_path, monkeypatch):
        """When creating the default config fails with PermissionError, defaults are returned."""

        fake_cfg_path = tmp_path / "axon" / "config.yaml"

        monkeypatch.setattr("axon.config._USER_CONFIG_PATH", str(fake_cfg_path))

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("no write")):
            cfg = AxonConfig.load(None)

        assert isinstance(cfg, AxonConfig)


# ===========================================================================


# 3. __post_init__ env-var and WSL branches --" lines 83-181


# ===========================================================================


class TestPostInit:

    """Tests for __post_init__ environment-variable handling."""

    def test_api_key_from_env_api_key(self, monkeypatch):
        """api_key field is populated from API_KEY env var."""

        monkeypatch.setenv("API_KEY", "from-api-key-env")

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        cfg = AxonConfig()

        assert cfg.api_key == "from-api-key-env"

    def test_api_key_from_env_openai_api_key(self, monkeypatch):
        """api_key field falls back to OPENAI_API_KEY env var."""

        monkeypatch.delenv("API_KEY", raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "from-openai-env")

        cfg = AxonConfig()

        assert cfg.api_key == "from-openai-env"

    def test_gemini_api_key_from_env(self, monkeypatch):
        """gemini_api_key is populated from GEMINI_API_KEY env var."""

        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")

        cfg = AxonConfig()

        assert cfg.gemini_api_key == "gemini-env-key"

    def test_ollama_cloud_key_from_env(self, monkeypatch):
        """ollama_cloud_key is populated from OLLAMA_CLOUD_KEY env var."""

        monkeypatch.setenv("OLLAMA_CLOUD_KEY", "oc-key-123")

        cfg = AxonConfig()

        assert cfg.ollama_cloud_key == "oc-key-123"

    def test_ollama_cloud_url_default_from_env(self, monkeypatch):
        """ollama_cloud_url defaults to OLLAMA_CLOUD_URL env var when field is empty."""

        monkeypatch.setenv("OLLAMA_CLOUD_URL", "https://custom.ollama.com/api")

        cfg = AxonConfig()

        assert cfg.ollama_cloud_url == "https://custom.ollama.com/api"

    def test_vllm_base_url_from_env(self, monkeypatch):
        """vllm_base_url is set from VLLM_BASE_URL env var when field is the default."""

        monkeypatch.setenv("VLLM_BASE_URL", "http://env-vllm:9999/v1")

        cfg = AxonConfig()

        assert cfg.vllm_base_url == "http://env-vllm:9999/v1"

    def test_vllm_base_url_not_overridden_when_non_default(self, monkeypatch):
        """vllm_base_url is NOT overridden by env var when field has a non-default value."""

        monkeypatch.setenv("VLLM_BASE_URL", "http://env-vllm:9999/v1")

        cfg = AxonConfig(vllm_base_url="http://my-custom:1111/v1")

        # The field was already non-default so env should NOT override it in __post_init__

        assert cfg.vllm_base_url == "http://my-custom:1111/v1"

    def test_axon_store_base_env_derives_projects_root(self, monkeypatch, tmp_path):
        """AXON_STORE_BASE env var sets the store base and derives projects_root."""  # noqa

        monkeypatch.setenv("AXON_STORE_BASE", str(tmp_path))

        cfg = AxonConfig()

        assert "AxonStore" in cfg.projects_root

        import getpass

        assert getpass.getuser() in cfg.projects_root

    def test_axon_store_base_field_derives_projects_root(self, tmp_path, monkeypatch):
        """axon_store_base field in constructor derives projects_root under AxonStore/."""

        monkeypatch.delenv("AXON_STORE_BASE", raising=False)

        cfg = AxonConfig(axon_store_base=str(tmp_path))

        import getpass

        assert getpass.getuser() in cfg.projects_root

        assert "AxonStore" in cfg.projects_root

    def test_paths_always_derived_from_store(self, monkeypatch):
        """vector_store_path and bm25_path are always derived from the store layout."""

        monkeypatch.delenv("AXON_STORE_BASE", raising=False)

        cfg = AxonConfig()

        assert cfg.vector_store_path != ""

        assert "lancedb_data" in cfg.vector_store_path

        assert "bm25_index" in cfg.bm25_path


# ---------------------------------------------------------------------------


# First-run config creation: starter YAML values must be returned, not defaults


# ---------------------------------------------------------------------------


class TestFirstRunConfigCreation:

    """P0-1: AxonConfig.load() must return starter file values on first run.


    The starter YAML ships with raptor=false, graph_rag=false,


    graph_rag_community=false.  The dataclass defaults are True.  Before the


    bug fix, load() always called cls() after creating the file, which silently


    ignored the file it just wrote and returned the (wrong) dataclass defaults.


    First-run creation only triggers when path=None (uses _USER_CONFIG_PATH).


    Tests redirect _USER_CONFIG_PATH to a tmp directory via monkeypatch.


    """

    def _patch_config_path(self, monkeypatch, tmp_path):
        """Redirect _USER_CONFIG_PATH to tmp_path/config.yaml."""

        import axon.config as _cfg_mod

        config_path = str(tmp_path / "config.yaml")

        monkeypatch.setattr(_cfg_mod, "_USER_CONFIG_PATH", config_path)

        monkeypatch.delenv("AXON_PROJECTS_ROOT", raising=False)

        monkeypatch.delenv("AXON_STORE_BASE", raising=False)

        return config_path

    def test_first_run_raptor_disabled(self, tmp_path, monkeypatch):
        """First-run config: raptor must be False (file value), not True (dataclass default)."""

        from axon.config import AxonConfig

        self._patch_config_path(monkeypatch, tmp_path)

        cfg = AxonConfig.load()

        assert cfg.raptor is False, (
            "First-run config must disable RAPTOR (starter YAML has raptor: false); "
            "got True -- load() is returning the dataclass default instead of the file value"
        )

    def test_first_run_graph_rag_disabled(self, tmp_path, monkeypatch):
        """First-run config: graph_rag must be False (file value), not True (dataclass default)."""

        from axon.config import AxonConfig

        self._patch_config_path(monkeypatch, tmp_path)

        cfg = AxonConfig.load()

        assert (
            cfg.graph_rag is False
        ), "First-run config must disable GraphRAG (starter YAML has graph_rag: false)"

    def test_first_run_graph_rag_community_disabled(self, tmp_path, monkeypatch):
        """First-run config: graph_rag_community must be False."""

        from axon.config import AxonConfig

        self._patch_config_path(monkeypatch, tmp_path)

        cfg = AxonConfig.load()

        assert cfg.graph_rag_community is False

    def test_first_run_creates_the_file(self, tmp_path, monkeypatch):
        """load() creates the config file on first run."""

        from axon.config import AxonConfig

        config_path = self._patch_config_path(monkeypatch, tmp_path)

        assert not os.path.exists(config_path)

        AxonConfig.load()

        assert os.path.exists(config_path)

    def test_first_run_file_then_roundtrip(self, tmp_path, monkeypatch):
        """save() + load() round-trip preserves disabled flags."""

        from axon.config import AxonConfig

        config_path = self._patch_config_path(monkeypatch, tmp_path)

        cfg = AxonConfig.load()

        assert cfg.raptor is False

        cfg.save(config_path)

        reloaded = AxonConfig.load()

        assert reloaded.raptor is False

        assert reloaded.graph_rag is False


class TestStoreDerivedPaths:
    def test_axon_store_base_derives_all_paths(self, tmp_path):
        """axon_store_base derives projects_root, vector_store_path, and bm25_path."""

        import getpass

        user = getpass.getuser()

        cfg = AxonConfig(axon_store_base=str(tmp_path))

        assert str(tmp_path) in cfg.projects_root

        assert "AxonStore" in cfg.projects_root

        assert user in cfg.projects_root

        assert "lancedb_data" in cfg.vector_store_path

        assert "bm25_index" in cfg.bm25_path

    def test_default_store_base_is_axon_home(self, monkeypatch):
        """Default store base is ~/.axon when not configured."""

        monkeypatch.delenv("AXON_STORE_BASE", raising=False)

        cfg = AxonConfig()

        assert ".axon" in cfg.axon_store_base or "axon" in cfg.axon_store_base.lower()

        assert "AxonStore" in cfg.projects_root
