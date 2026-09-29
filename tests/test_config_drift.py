"""Config keys that used to be accepted but ignored, and offline-mode provider enforcement."""

from unittest.mock import MagicMock

import pytest

from axon.config import (
    AxonConfig,
    OfflineModeError,
    enforce_offline_mode,
    offline_violations,
)


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestVectorStoreYamlKeys:
    def test_nested_qdrant_keys_load(self, tmp_path):
        path = _write(
            tmp_path,
            "vector_store:\n  provider: qdrant\n"
            "  qdrant_url: http://localhost:6333\n  qdrant_api_key: k\n",
        )
        cfg = AxonConfig.load(path)
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.qdrant_api_key == "k"

    def test_top_level_qdrant_keys_still_win(self, tmp_path):
        path = _write(
            tmp_path,
            "vector_store:\n  qdrant_url: http://nested:6333\nqdrant_url: http://top:6333\n",
        )
        assert AxonConfig.load(path).qdrant_url == "http://top:6333"

    @pytest.mark.parametrize(
        "yaml_text,field",
        [
            ("vector_store:\n  qdrant_collection: c\n", "qdrant_collection"),
            ("vector_store:\n  lancedb_path: /x\n", "lancedb_path"),
            ("web_search:\n  safe_search: true\n", "safe_search"),
        ],
    )
    def test_keys_that_were_never_read_are_reported_unknown(self, tmp_path, yaml_text, field):
        issues = AxonConfig.validate(_write(tmp_path, yaml_text))
        assert [i for i in issues if i.field == field and "Unknown key" in i.message]

    def test_nested_qdrant_keys_validate_clean(self, tmp_path):
        issues = AxonConfig.validate(
            _write(tmp_path, "vector_store:\n  qdrant_url: http://h:6333\n")
        )
        assert not [i for i in issues if i.field == "qdrant_url"]


class TestProjectsRootEnvRemoved:
    def test_env_var_no_longer_changes_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AXON_PROJECTS_ROOT", str(tmp_path / "elsewhere"))
        cfg = AxonConfig.load(_write(tmp_path, "llm:\n  provider: ollama\n"))
        assert "elsewhere" not in cfg.projects_root

    def test_projects_module_ignores_env_var(self, tmp_path, monkeypatch):
        import axon.projects as proj_mod

        monkeypatch.setenv("AXON_PROJECTS_ROOT", str(tmp_path))
        assert proj_mod._resolve_projects_root() != tmp_path


class TestOfflineViolations:
    def _cfg(self, **kw):
        cfg = AxonConfig()
        cfg.offline_mode = True
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.mark.parametrize(
        "provider", ["openai", "gemini", "grok", "ollama_cloud", "copilot", "github_copilot"]
    )
    def test_cloud_llm_is_a_violation(self, provider):
        cfg = self._cfg(llm_provider=provider)
        assert offline_violations(cfg)
        with pytest.raises(OfflineModeError, match=provider):
            enforce_offline_mode(cfg)

    @pytest.mark.parametrize("provider", ["ollama", "vllm", "local"])
    def test_local_llm_is_fine(self, provider):
        assert offline_violations(self._cfg(llm_provider=provider)) == []

    def test_openai_embedding_is_a_violation(self):
        cfg = self._cfg(llm_provider="ollama", embedding_provider="openai")
        assert any("embedding.provider" in m for m in offline_violations(cfg))

    @pytest.mark.parametrize("provider", ["fastembed", "sentence_transformers", "ollama"])
    def test_local_embedding_is_fine(self, provider):
        cfg = self._cfg(llm_provider="ollama", embedding_provider=provider)
        assert offline_violations(cfg) == []

    def test_nothing_is_enforced_when_offline_is_off(self):
        cfg = AxonConfig()
        cfg.llm_provider = "openai"
        cfg.embedding_provider = "openai"
        assert offline_violations(cfg) == []

    def test_a_duck_typed_config_never_trips(self):
        assert offline_violations(MagicMock()) == []

    def test_prospective_provider_does_not_mutate(self):
        cfg = self._cfg(llm_provider="ollama")
        assert offline_violations(cfg, llm_provider="openai", only="llm")
        assert cfg.llm_provider == "ollama"

    def test_only_restricts_the_side_checked(self):
        cfg = self._cfg(llm_provider="openai", embedding_provider="fastembed")
        assert offline_violations(cfg, only="embedding") == []
        assert offline_violations(cfg, only="llm")


class TestOfflineEnforcedWhereProvidersAreBuilt:
    def test_openllm_refuses_cloud_provider(self):
        from axon.llm import OpenLLM

        cfg = AxonConfig()
        cfg.offline_mode = True
        cfg.llm_provider = "gemini"
        with pytest.raises(OfflineModeError):
            OpenLLM(cfg)

    def test_openllm_allows_local_provider(self):
        from axon.llm import OpenLLM

        cfg = AxonConfig()
        cfg.offline_mode = True
        cfg.llm_provider = "ollama"
        assert OpenLLM(cfg) is not None

    def test_openembedding_refuses_openai(self):
        from axon.embeddings import OpenEmbedding

        cfg = AxonConfig()
        cfg.offline_mode = True
        cfg.embedding_provider = "openai"
        with pytest.raises(OfflineModeError):
            OpenEmbedding(cfg)

    def test_validate_reports_an_error(self, tmp_path):
        path = _write(tmp_path, "offline:\n  enabled: true\nllm:\n  provider: openai\n")
        issues = AxonConfig.validate(path)
        errors = [i for i in issues if i.section == "offline" and i.level == "error"]
        assert errors and "openai" in errors[0].message

    def test_validate_is_quiet_for_a_local_setup(self, tmp_path):
        path = _write(tmp_path, "offline:\n  enabled: true\nllm:\n  provider: ollama\n")
        assert not [i for i in AxonConfig.validate(path) if i.section == "offline"]
