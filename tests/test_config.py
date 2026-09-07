from __future__ import annotations

"""Tests for configuration management."""


import yaml

from axon.main import AxonConfig


class TestAxonConfig:
    """Test the AxonConfig class."""

    def test_default_config(self):
        """Test default configuration values."""

        config = AxonConfig()

        assert config.embedding_provider == "fastembed"

        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

        assert config.llm_provider == "ollama"

        assert config.vector_store == "turboquantdb"

        assert config.top_k == 10

        assert config.chunk_size == 1000

    def test_share_mount_defaults(self):
        """Share-mount config knobs added by #53 must keep their advertised defaults
        — silent drift would change runtime behaviour for grantees on a mount."""

        config = AxonConfig()

        # Default refresh policy: cache marker on switch, no per-query overhead.
        assert config.mount_refresh_mode == "switch"
        assert config.mount_sync_retry_max == 5
        assert config.mount_sync_retry_backoff_s == 0.5

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

        assert config.embedding_provider == "fastembed"

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

    def test_yaml_rag_rust_engines(self, tmp_path):
        """Rust engine toggles are loaded from rag section in YAML."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "rag": {
                        "ingest_engine": "rust",
                        "bm25_engine": "rust",
                        "symbol_index_engine": "rust",
                        "rust_fallback_enabled": False,
                        "rust_batch_size": 1024,
                    }
                },
                f,
            )

        config = AxonConfig.load(str(config_path))
        assert config.ingest_engine == "rust"
        assert config.bm25_engine == "rust"
        assert config.symbol_index_engine == "rust"
        assert config.rust_fallback_enabled is False
        assert config.rust_batch_size == 1024

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

        assert data["embedding"]["provider"] == "fastembed"

    def test_save_creates_parent_directory(self, tmp_path):
        """save() calls os.makedirs to create missing parent dirs."""

        cfg = AxonConfig()

        nested = tmp_path / "deep" / "nested" / "config.yaml"

        cfg.save(str(nested))

        assert nested.exists()

    def test_save_writes_atomically_no_stray_tmp_file(self, tmp_path):
        """save() now goes through write_text_if_changed() (temp file +
        os.replace) instead of a bare open().write() — a crash mid-write
        must never leave a truncated config.yaml, and a completed save must
        never leave the .tmp file behind."""

        cfg = AxonConfig()
        target = tmp_path / "config.yaml"

        cfg.save(str(target))

        assert target.exists()
        assert not target.with_suffix(target.suffix + ".tmp").exists()

    def test_save_twice_unchanged_is_a_no_op_write(self, tmp_path):
        """Saving identical content twice in a row must skip the second
        write (the skip-if-unchanged path) rather than always rewriting."""

        cfg = AxonConfig()
        target = tmp_path / "config.yaml"

        cfg.save(str(target))
        mtime_before = target.stat().st_mtime_ns
        cfg.save(str(target))

        assert target.stat().st_mtime_ns == mtime_before

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

        assert cfg.embedding_provider == "fastembed"

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


class TestRemovedFields:
    """Keys removed in a past release must announce themselves.

    Regression: dropping the sparse_retrieval / sparse_model / sparse_weight
    dataclass fields made load() filter them out against valid_fields with no
    log at all, so a user upgrading with ``rag.sparse_retrieval: true`` silently
    lost SPLADE and saw retrieval behaviour change with zero signal. validate()
    also reported it as an unknown key with a difflib "Did you mean...?", which
    reads like a typo rather than a removed feature.
    """

    def _cfg(self, tmp_path, body: str) -> str:
        p = tmp_path / "config.yaml"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_load_still_succeeds_with_a_removed_key(self, tmp_path):
        from axon.config import AxonConfig

        path = self._cfg(tmp_path, "rag:\n  sparse_retrieval: true\n  top_k: 5\n")
        cfg = AxonConfig.load(path)
        assert cfg.top_k == 5

    def test_load_warns_that_the_key_was_removed(self, tmp_path, caplog):
        import logging

        from axon.config import AxonConfig

        path = self._cfg(tmp_path, "rag:\n  sparse_retrieval: true\n")
        with caplog.at_level(logging.WARNING):
            AxonConfig.load(path)
        assert any("removed in 0.5.0" in r.message for r in caplog.records)

    def test_validate_says_removed_not_did_you_mean(self, tmp_path):
        from axon.config import AxonConfig

        path = self._cfg(tmp_path, "rag:\n  sparse_retrieval: true\n")
        match = [i for i in AxonConfig.validate(path) if i.field == "sparse_retrieval"]
        assert match, "removed key produced no issue"
        assert "no longer a valid key" in match[0].message
        assert "Did you mean" not in (match[0].suggestion or "")

    def test_a_genuine_typo_still_gets_a_suggestion(self, tmp_path):
        """The removed-key branch must not swallow the typo branch."""
        from axon.config import AxonConfig

        path = self._cfg(tmp_path, "rag:\n  top_kk: 5\n")
        match = [i for i in AxonConfig.validate(path) if i.field == "top_kk"]
        assert match and "Did you mean" in (match[0].suggestion or "")


class TestDemotedGraphTuning:
    """GraphRAG context-assembly knobs demoted to constants in 0.5.0.

    Regression: before the demotion these were dataclass fields that ``load()``
    accepted but ``_KNOWN_YAML_KEYS`` did not list, so ``validate()`` called a
    *working* key unknown and offered a difflib suggestion naming a different
    real field — e.g. ``graph_rag_local_entity_weight`` (which took effect) was
    reported as "Did you mean 'graph_federation_weights'?". Following that
    advice would have turned a working config into a broken one.
    """

    def _cfg(self, tmp_path, body: str) -> str:
        p = tmp_path / "config.yaml"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_every_demoted_key_is_registered_as_removed(self):
        """No demoted field may fall through to the typo branch."""
        from axon.config import _DEMOTED_GRAPH_TUNING, _REMOVED_FIELDS

        missing = [f for f in _DEMOTED_GRAPH_TUNING if f not in _REMOVED_FIELDS]
        assert not missing, f"demoted but not registered as removed: {missing}"

    def test_no_demoted_key_is_still_a_dataclass_field(self):
        """A field left on AxonConfig would silently shadow its constant."""
        from axon.config import _DEMOTED_GRAPH_TUNING, AxonConfig

        still = [f for f in _DEMOTED_GRAPH_TUNING if f in AxonConfig.__dataclass_fields__]
        assert not still, f"demoted but still a dataclass field: {still}"

    def test_demoted_key_reports_removal_not_a_typo(self, tmp_path):
        path = self._cfg(tmp_path, "rag:\n  graph_rag_local_entity_weight: 9.5\n")
        match = [i for i in AxonConfig.validate(path) if i.field == "graph_rag_local_entity_weight"]
        assert match, "demoted key produced no issue"
        assert "no longer a valid key" in match[0].message
        assert "Did you mean" not in (match[0].suggestion or "")
        assert "graph_defaults" in (match[0].suggestion or "")

    def test_demoted_key_does_not_break_load(self, tmp_path):
        """An upgraded config carrying a demoted key must still load."""
        path = self._cfg(tmp_path, "rag:\n  graph_rag_local_entity_weight: 9.5\n  top_k: 7\n")
        cfg = AxonConfig.load(path)
        assert cfg.top_k == 7
        assert not hasattr(cfg, "graph_rag_local_entity_weight")

    def test_constants_match_the_defaults_they_replaced(self):
        """The demotion was a move, not a retune."""
        from axon import graph_defaults as _gd

        assert _gd.GLOBAL_MIN_SCORE == 20
        assert _gd.GLOBAL_TOP_POINTS == 50
        assert _gd.GLOBAL_REDUCE_MAX_TOKENS == 8000
        assert _gd.LOCAL_MAX_CONTEXT_TOKENS == 8000
        assert _gd.LOCAL_ENTITY_WEIGHT == 3.0
        assert _gd.LOCAL_RELATION_WEIGHT == 2.0
        assert _gd.LOCAL_COMMUNITY_WEIGHT == 1.5
        assert _gd.LOCAL_TEXT_UNIT_WEIGHT == 1.0
        assert _gd.LOCAL_EARLY_CUTOFF_FACTOR == 1.5
        # Group B — community clustering. USE_LCC is the one that mattered:
        # its old getattr fallback said True, which would have silently
        # dropped every component outside the largest.
        assert _gd.COMMUNITY_USE_LCC is False
        assert _gd.COMMUNITY_MIN_SIZE == 3
        assert _gd.COMMUNITY_LLM_MAX_TOTAL == 30
        assert _gd.LEIDEN_SEED == 42
        # Group C — extraction and relation persistence. MSGPACK_PERSIST is
        # the inverted-fallback case here.
        assert _gd.RELATION_MSGPACK_PERSIST is False
        assert _gd.ENTITY_RESOLVE_THRESHOLD == 0.92
        assert _gd.ENTITY_RESOLVE_MAX == 5000
        assert _gd.ENTITY_MATCH_THRESHOLD == 0.5
        assert _gd.RELATION_SHARD_COUNT == 16

    def test_no_module_reaches_demoted_fields_via_getattr(self):
        """A stale getattr would resurrect the old fallback, which often differed.

        Several fields moved to graph_defaults.py had a pre-0.5.0 getattr
        fallback that disagreed with the dataclass default it shadowed —
        ``graph_rag_community_use_lcc``'s said True where the field said False.
        Across the wider graph_rag_* surface the audit found 24 such drifts
        among 143 call sites. Every one was dead only because the field always
        existed, so deleting a field without deleting its getattr would make
        that stale fallback live.

        Scans every module in the package rather than a fixed list: a demoted
        field's read can live anywhere, and group B's did move beyond
        graph_rag.py into query_router.py and graph_backends/.
        """
        hits = self._scan_package(lambda f: rf"getattr\([^)]*['\"]{f}['\"]")
        assert not hits, f"getattr still reaching demoted fields: {hits}"

    def test_no_module_reads_a_demoted_field_as_an_attribute(self):
        """`cfg.graph_rag_community_min_size` would now raise AttributeError."""
        hits = self._scan_package(lambda f: rf"\.{f}\b")
        assert not hits, f"attribute reads of demoted fields: {hits}"

    @staticmethod
    def _scan_package(pattern_for):
        """Search every module in the installed `axon` package for a pattern.

        Resolves through the imported package rather than the cwd, so the scan
        works from any working directory and against an installed copy. Reports
        package-relative path plus line number — bare filenames are ambiguous
        in a tree with several `__init__.py`.
        """
        import re
        from pathlib import Path

        import axon
        import axon.config
        from axon.config import _DEMOTED_GRAPH_TUNING

        root = Path(axon.__file__).parent
        # config.py itself holds every demoted name, by definition.
        skip = Path(axon.config.__file__).resolve()
        hits = []
        for path in sorted(root.rglob("*.py")):
            if path.resolve() == skip:
                continue
            src = path.read_text(encoding="utf-8", errors="replace")
            for field in _DEMOTED_GRAPH_TUNING:
                for m in re.finditer(pattern_for(field), src):
                    line = src[: m.start()].count("\n") + 1
                    rel = path.relative_to(root).as_posix()
                    hits.append(f"{rel}:{line}:{field}")
        return hits


class TestRagSectionSchemaMatchesLoad:
    """`validate()` must accept every key `load()` accepts under `rag:`.

    Regression: `load()` does `config_dict.update(data["rag"])`, taking that
    section's keys verbatim as dataclass field names, and `save()` parks every
    field without a bespoke section mapping there. `validate()` checked a
    hand-written `_KNOWN_YAML_KEYS["rag"]` set instead, which had drifted 69
    keys behind — so a config Axon wrote itself failed its own validation, and
    documented, API-exposed keys like `graph_rag_relation_backend` were
    reported as typos with a suggestion naming an unrelated field. Following
    that advice would replace a working setting with a different one.
    """

    def _cfg(self, tmp_path, body: str) -> str:
        p = tmp_path / "config.yaml"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_every_dataclass_field_validates_under_rag(self, tmp_path):
        """The whole surface, not a sample — this is what drifted before.

        Each key carries its own default so the config stays semantically
        valid; only the structural unknown-key pass is under test here.
        """
        import yaml as _yaml

        cfg = AxonConfig()
        rag = {}
        for name in AxonConfig.__dataclass_fields__:
            if name.startswith("_"):
                continue
            value = getattr(cfg, name, None)
            if value is None or isinstance(value, (str, int, float, bool, list, dict)):
                rag[name] = value
        path = self._cfg(tmp_path, _yaml.safe_dump({"rag": rag}, sort_keys=True))
        unknown = [i.field for i in AxonConfig.validate(path) if "Unknown key" in i.message]
        assert not unknown, f"validate() rejects keys load() accepts: {unknown}"
        assert len(rag) > 150, f"only {len(rag)} fields exercised — the sweep stopped working"

    def test_documented_kept_keys_are_not_called_typos(self, tmp_path):
        """The keys three PRs of collapsing deliberately kept, spot-checked."""
        kept = [
            "graph_rag_relation_backend",
            "graph_rag_ner_backend",
            "graph_rag_claims",
            "graph_rag_canonicalize",
            "graph_rag_community_lazy",
            "graph_rag_community_levels",
            "graph_rag_entity_resolve",
            "graph_rag_min_entities_for_relations",
        ]
        path = self._cfg(tmp_path, "rag:\n" + "".join(f"  {k}: null\n" for k in kept))
        bad = [i.field for i in AxonConfig.validate(path) if "Unknown key" in i.message]
        assert not bad, f"documented keys reported as unknown: {bad}"

    def test_a_genuine_typo_is_still_caught(self, tmp_path):
        """Widening the accepted set must not blunt the typo branch."""
        path = self._cfg(tmp_path, "rag:\n  top_kk: 5\n")
        match = [i for i in AxonConfig.validate(path) if i.field == "top_kk"]
        assert match, "typo produced no issue"
        assert "Did you mean 'top_k'?" in (match[0].suggestion or "")

    def test_a_removed_key_still_reports_removal(self, tmp_path):
        """Removed keys must not be silently swallowed by the widened set."""
        path = self._cfg(tmp_path, "rag:\n  graph_rag_local_entity_weight: 9.5\n")
        match = [i for i in AxonConfig.validate(path) if i.field == "graph_rag_local_entity_weight"]
        assert match and "no longer a valid key" in match[0].message
        assert "Did you mean" not in (match[0].suggestion or "")


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

        assert "vector_store_data" in cfg.vector_store_path

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

    def test_first_run_max_tokens_matches_dataclass_default(self, tmp_path, monkeypatch):
        """First-run config must not truncate reasoning models.

        Regression: _DEFAULT_CONFIG_YAML hardcoded ``max_tokens: 2048`` while the
        dataclass default is 8192 (bumped in 0.4.3 specifically because 2048
        truncates reasoning models mid-thought). Unlike raptor/graph_rag above,
        this one is a values-must-MATCH check, not a values-must-DIFFER check --
        the file is the single source of truth on first run either way, so a
        stale literal in the template silently regresses a shipped fix. Every
        fresh install and every ``/config reset`` hit this.
        """

        from axon.config import AxonConfig

        self._patch_config_path(monkeypatch, tmp_path)

        cfg = AxonConfig.load()

        assert cfg.llm_max_tokens == AxonConfig().llm_max_tokens == 8192

    def test_first_run_embedding_provider_matches_dataclass_default(self, tmp_path, monkeypatch):
        """First-run config's embedding provider/model must match the dataclass
        default (fastembed / sentence-transformers/all-MiniLM-L6-v2 since 0.4.6).
        Same values-must-MATCH shape as the max_tokens regression above — a stale
        literal in the starter YAML would silently ship the old ~20s-cold-start
        provider to every fresh install.
        """

        from axon.config import AxonConfig

        self._patch_config_path(monkeypatch, tmp_path)

        cfg = AxonConfig.load()

        default = AxonConfig()

        assert cfg.embedding_provider == default.embedding_provider == "fastembed"

        assert (
            cfg.embedding_model
            == default.embedding_model
            == "sentence-transformers/all-MiniLM-L6-v2"
        )

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

        assert "vector_store_data" in cfg.vector_store_path

        assert "bm25_index" in cfg.bm25_path

    def test_default_store_base_is_axon_home(self, monkeypatch):
        """Default store base is ~/.axon when not configured."""

        monkeypatch.delenv("AXON_STORE_BASE", raising=False)

        cfg = AxonConfig()

        assert ".axon" in cfg.axon_store_base or "axon" in cfg.axon_store_base.lower()

        assert "AxonStore" in cfg.projects_root
