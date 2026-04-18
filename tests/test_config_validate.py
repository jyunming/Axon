import json
import os
import tempfile
from unittest.mock import patch

import yaml

from axon.config import AxonConfig


def test_validate_missing_file():
    config = AxonConfig()
    issues = config.validate(path="some_nonexistent_file.yaml")
    assert any(i.level == "info" and "Config file not found" in i.message for i in issues)


def test_validate_default_config(tmp_path):
    config = AxonConfig()
    config_path = tmp_path / "config.yaml"
    config.save(str(config_path))
    issues = config.validate(path=str(config_path))
    for issue in issues:
        assert issue.level in ["info", "warn", "warning", "error", "success"]
        # Cover to_dict
        assert isinstance(issue.to_dict(), dict)


def test_validate_all_providers(tmp_path):
    config = AxonConfig()
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "llm": {"provider": "openai", "api_key": "test"},
                "embedding": {"provider": "openai", "api_key": "test"},
                "vector_store": {"provider": "chroma", "collection": "test"},
                "reranker": {"provider": "cohere", "api_key": "test"},
                "web_search": {"provider": "tavily", "api_key": "test"},
            },
            f,
        )

    issues = config.validate(path=str(config_path))
    assert isinstance(issues, list)


def test_validate_missing_keys(tmp_path):
    config = AxonConfig()
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "llm": {"provider": "openai", "api_key": ""},
                "embedding": {"provider": "openai", "api_key": ""},
                "vector_store": {"provider": "qdrant", "url": ""},
                "reranker": {"provider": "cohere", "api_key": ""},
                "web_search": {"provider": "tavily", "api_key": ""},
            },
            f,
        )

    issues = config.validate(path=str(config_path))
    assert isinstance(issues, list)


def test_validate_offline_models(tmp_path):
    config = AxonConfig()
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "offline": {"enabled": True, "local_assets_only": True},
                "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
                "reranker": {
                    "provider": "cross-encoder",
                    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                },
                "llm": {"provider": "ollama"},
            },
            f,
        )

    issues = config.validate(path=str(config_path))
    # Ensure no "warn"-level issues are raised
    assert not [i for i in issues if i.level == "warn"]


def test_validate_provider_warnings(tmp_path):
    config_path = tmp_path / "config_warn.yaml"

    # Test OpenAI missing key
    with open(config_path, "w") as f:
        yaml.dump({"llm": {"provider": "openai", "api_key": ""}}, f)
    with patch.dict("os.environ", {}, clear=True):
        issues = AxonConfig().validate(path=str(config_path))
        assert any(i.level == "warn" and "openai_api_key" in i.field for i in issues)

    # Test Gemini missing key
    with open(config_path, "w") as f:
        yaml.dump({"llm": {"provider": "gemini", "api_key": ""}}, f)
    with patch.dict("os.environ", {}, clear=True):
        issues = AxonConfig().validate(path=str(config_path))
        assert any(i.level == "warn" and "gemini_api_key" in i.field for i in issues)

    # Test Grok missing key
    with open(config_path, "w") as f:
        yaml.dump({"llm": {"provider": "grok", "api_key": ""}}, f)
    with patch.dict("os.environ", {}, clear=True):
        issues = AxonConfig().validate(path=str(config_path))
        assert any(i.level == "warn" and "grok_api_key" in i.field for i in issues)


def test_validate_store_overrides(tmp_path):
    config_path = tmp_path / "config_store.yaml"
    env_base = str(tmp_path / "env_base")
    cfg_base = str(tmp_path / "cfg_base")
    os.makedirs(env_base, exist_ok=True)
    os.makedirs(cfg_base, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump({"store": {"base": cfg_base}}, f)

    with patch.dict("os.environ", {"AXON_STORE_BASE": env_base}):
        issues = AxonConfig().validate(path=str(config_path))
        assert any("overrides" in i.message for i in issues)


def test_validate_nonexistent_custom_base(tmp_path):
    config_path = tmp_path / "config_bad_base.yaml"
    bad_base = str(tmp_path / "nonexistent_base")
    with open(config_path, "w") as f:
        yaml.dump({"store": {"base": bad_base}}, f)

    issues = AxonConfig().validate(path=str(config_path))
    assert any("does not exist" in i.message for i in issues)


def test_validate_corrupted_yaml(tmp_path):
    config_path = tmp_path / "corrupted.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: :")

    issues = AxonConfig().validate(path=str(config_path))
    assert any(i.level == "error" and "Could not read config file" in i.message for i in issues)

    # Also cover load() with malformed yaml
    config = AxonConfig.load(str(config_path))
    assert config is not None


def test_validate_store_meta_check(tmp_path):
    # Test path exists but no store_meta.json
    base_path = tmp_path / "axon_store"
    os.makedirs(base_path, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"store": {"base": str(base_path)}}, f)

    issues = AxonConfig().validate(path=str(config_path))
    assert any("No store_meta.json found" in i.message for i in issues)

    # Test path exists with store_meta.json
    import getpass

    username = getpass.getuser()
    meta_dir = base_path / "AxonStore" / username
    os.makedirs(meta_dir, exist_ok=True)
    with open(meta_dir / "store_meta.json", "w") as f:
        json.dump({"store_id": "test-id"}, f)

    issues = AxonConfig().validate(path=str(config_path))
    assert any("AxonStore initialised" in i.message for i in issues)


def test_load_legacy_advanced_section(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"advanced": {"raptor": True, "graph_rag": True}}, f)
    config = AxonConfig.load(str(config_path))
    assert config.raptor is True
    assert config.graph_rag is True


def test_validate_unknown_keys(tmp_path):
    config_path = tmp_path / "unknown.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"llm": {"unknown_key": "val"}}, f)
    issues = AxonConfig().validate(path=str(config_path))
    assert any("Unknown key 'unknown_key'" in i.message for i in issues)


def test_save_blocked_in_tmp(tmp_path):
    # This might already be covered but good to be sure
    config = AxonConfig()
    with patch("os.path.commonpath", return_value=tempfile.gettempdir()):
        config.save(os.path.join(tempfile.gettempdir(), "config.yaml"))


def test_validate_non_dict_section(tmp_path):
    config_path = tmp_path / "bad_struct.yaml"
    with open(config_path, "w") as f:
        f.write("llm: some_string\n")
    issues = AxonConfig().validate(path=str(config_path))
    # Should skip the non-dict section without crashing
    assert isinstance(issues, list)


def test_validate_invalid_values(tmp_path):
    config_path = tmp_path / "invalid_vals.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "rag": {
                    "graph_rag_mode": "invalid",
                    "graph_rag_depth": "invalid",
                    "query_router": "invalid",
                    "top_k": 0,
                    "similarity_threshold": 1.1,
                    "sentence_window_size": 0,
                }
            },
            f,
        )

    issues = AxonConfig().validate(path=str(config_path))
    errors = [i for i in issues if i.level == "error"]
    assert any("Invalid graph_rag_mode" in i.message for i in errors)
    assert any("Invalid graph_rag_depth" in i.message for i in errors)
    assert any("Invalid query_router" in i.message for i in errors)
    assert any("top_k must be >= 1" in i.message for i in errors)
    assert any("similarity_threshold must be between" in i.message for i in errors)
    assert any("sentence_window_size must be >= 1" in i.message for i in errors)


def test_save_all_extras(tmp_path):
    config = AxonConfig()
    config.grok_api_key = "grok-key"
    config.gemini_api_key = "gemini-key"
    config.ollama_cloud_key = "ollama-key"
    config.ollama_cloud_url = "https://ollama.cloud"
    config.vllm_base_url = "http://vllm:8000"
    config.llm_timeout = 60
    config.axon_store_base = str(tmp_path / "custom_store")

    config_path = tmp_path / "full_config.yaml"
    config.save(str(config_path))

    with open(config_path) as f:
        data = yaml.safe_load(f)
        assert data["llm"]["grok_api_key"] == "grok-key"
        assert data["llm"]["gemini_api_key"] == "gemini-key"
        assert data["llm"]["ollama_cloud_key"] == "ollama-key"
        assert data["llm"]["timeout"] == 60
        assert data["store"]["base"] == config.axon_store_base


def test_load_all_compat_keys(tmp_path):
    config_path = tmp_path / "compat.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "llm": {
                    "openai_api_key": "okey",
                    "grok_api_key": "gkey",
                },
                "projects_root": "/tmp/projs",
                "source_policy_enabled": True,
                "qdrant_url": "qurl",
                "qdrant_api_key": "qkey",
                "repl_shell_passthrough": True,
            },
            f,
        )

    # We must ensure AXON_STORE_BASE is NOT set to avoid re-deriving projects_root
    with patch.dict("os.environ", {}, clear=True):
        config = AxonConfig.load(str(config_path))
        assert config.openai_api_key == "okey"
        assert config.grok_api_key == "gkey"
        assert config.source_policy_enabled is True
        assert config.qdrant_url == "qurl"
        assert config.qdrant_api_key == "qkey"
        assert config.repl_shell_passthrough is True
