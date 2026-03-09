"""Tests for offline mode: config loading, model path resolution, and runtime guards."""

import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(config, *, mock_bm25=True):
    """Construct an OpenStudioBrain with all heavy dependencies mocked."""
    from rag_brain.main import OpenStudioBrain
    with patch("rag_brain.main.OpenEmbedding"), \
         patch("rag_brain.main.OpenLLM"), \
         patch("rag_brain.main.OpenVectorStore"), \
         patch("rag_brain.main.OpenReranker"), \
         patch("rag_brain.retrievers.BM25Retriever"):
        return OpenStudioBrain(config)


# ---------------------------------------------------------------------------
# Config: new fields and YAML loading
# ---------------------------------------------------------------------------

class TestOfflineConfig:
    def test_default_offline_fields(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig()
        assert config.offline_mode is False
        assert config.local_models_dir == ""

    def test_load_offline_section_enabled(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        data = {"offline": {"enabled": True, "local_models_dir": "/srv/models"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(data))

        config = OpenStudioConfig.load(str(cfg_path))
        assert config.offline_mode is True
        assert config.local_models_dir == "/srv/models"

    def test_load_offline_section_disabled(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        data = {"offline": {"enabled": False, "local_models_dir": "/srv/models"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(data))

        config = OpenStudioConfig.load(str(cfg_path))
        assert config.offline_mode is False
        # local_models_dir still loaded even if offline is disabled
        assert config.local_models_dir == "/srv/models"

    def test_load_offline_section_missing(self, tmp_path):
        """YAML without offline: section → defaults."""
        from rag_brain.main import OpenStudioConfig
        data = {"llm": {"provider": "ollama", "model": "gemma"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(data))

        config = OpenStudioConfig.load(str(cfg_path))
        assert config.offline_mode is False
        assert config.local_models_dir == ""

    def test_load_offline_no_local_models_dir(self, tmp_path):
        """offline.enabled without local_models_dir → empty string."""
        from rag_brain.main import OpenStudioConfig
        data = {"offline": {"enabled": True}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(data))

        config = OpenStudioConfig.load(str(cfg_path))
        assert config.offline_mode is True
        assert config.local_models_dir == ""


# ---------------------------------------------------------------------------
# _resolve_model_path
# ---------------------------------------------------------------------------

class TestResolveModelPath:
    def _brain_offline(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(
            offline_mode=True,
            local_models_dir=str(tmp_path),
        )
        return _make_brain(config)

    def test_offline_off_returns_unchanged(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(offline_mode=False, local_models_dir=str(tmp_path))
        brain = _make_brain(config)
        assert brain._resolve_model_path("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"

    def test_absolute_path_returned_unchanged(self, tmp_path):
        brain = self._brain_offline(tmp_path)
        abs_path = str(tmp_path / "some-model")
        assert brain._resolve_model_path(abs_path) == abs_path

    def test_relative_dot_path_returned_unchanged(self, tmp_path):
        brain = self._brain_offline(tmp_path)
        assert brain._resolve_model_path("./models/bge-reranker-base") == "./models/bge-reranker-base"

    def test_resolves_bare_name(self, tmp_path):
        """'all-MiniLM-L6-v2' → '<dir>/all-MiniLM-L6-v2' when that dir exists."""
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        brain = self._brain_offline(tmp_path)
        result = brain._resolve_model_path("all-MiniLM-L6-v2")
        assert result == str(model_dir)

    def test_resolves_hf_org_slash_name_by_short_name(self, tmp_path):
        """'sentence-transformers/all-MiniLM-L6-v2' → '<dir>/all-MiniLM-L6-v2'."""
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        brain = self._brain_offline(tmp_path)
        result = brain._resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
        assert result == str(model_dir)

    def test_resolves_hf_cache_style_name(self, tmp_path):
        """Falls back to 'BAAI--bge-reranker-base' style if short name not found."""
        model_dir = tmp_path / "BAAI--bge-reranker-base"
        model_dir.mkdir()
        brain = self._brain_offline(tmp_path)
        result = brain._resolve_model_path("BAAI/bge-reranker-base")
        assert result == str(model_dir)

    def test_short_name_preferred_over_hf_cache_style(self, tmp_path):
        """Short name wins if both directory styles exist."""
        short_dir = tmp_path / "bge-reranker-base"
        hf_dir    = tmp_path / "BAAI--bge-reranker-base"
        short_dir.mkdir()
        hf_dir.mkdir()
        brain = self._brain_offline(tmp_path)
        result = brain._resolve_model_path("BAAI/bge-reranker-base")
        assert result == str(short_dir)

    def test_not_found_returns_original_and_warns(self, tmp_path):
        """When model directory is missing the original name is returned (caller will fail clearly)."""
        brain = self._brain_offline(tmp_path)
        result = brain._resolve_model_path("BAAI/bge-reranker-base")
        assert result == "BAAI/bge-reranker-base"

    def test_empty_local_models_dir_returns_unchanged(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(offline_mode=True, local_models_dir="")
        brain = _make_brain(config)
        assert brain._resolve_model_path("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# OpenStudioBrain.__init__ — offline mode side-effects
# ---------------------------------------------------------------------------

class TestBrainOfflineInit:
    def test_env_vars_set_when_offline(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(offline_mode=True, local_models_dir=str(tmp_path))
        _make_brain(config)
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert os.environ.get("HF_DATASETS_OFFLINE") == "1"
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_env_vars_not_forced_when_online(self, tmp_path):
        # Remove any leftover from previous test
        for var in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            os.environ.pop(var, None)
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(offline_mode=False)
        _make_brain(config)
        # Must not be set to "1" by the brain itself
        assert os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        assert os.environ.get("HF_HUB_OFFLINE") != "1"

    def test_truth_grounding_disabled_offline(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(
            offline_mode=True,
            local_models_dir=str(tmp_path),
            truth_grounding=True,
            brave_api_key="some-key",
        )
        brain = _make_brain(config)
        assert brain.config.truth_grounding is False

    def test_truth_grounding_untouched_when_online(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(offline_mode=False, truth_grounding=True, brave_api_key="key")
        brain = _make_brain(config)
        assert brain.config.truth_grounding is True

    def test_embedding_model_resolved_on_init(self, tmp_path):
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(
            offline_mode=True,
            local_models_dir=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        brain = _make_brain(config)
        assert brain.config.embedding_model == str(model_dir)

    def test_reranker_model_resolved_on_init(self, tmp_path):
        model_dir = tmp_path / "bge-reranker-base"
        model_dir.mkdir()
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(
            offline_mode=True,
            local_models_dir=str(tmp_path),
            reranker_model="BAAI/bge-reranker-base",
        )
        brain = _make_brain(config)
        assert brain.config.reranker_model == str(model_dir)

    def test_already_absolute_reranker_model_unchanged(self, tmp_path):
        abs_model = str(tmp_path / "my-reranker")
        Path(abs_model).mkdir()
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig(
            offline_mode=True,
            local_models_dir=str(tmp_path),
            reranker_model=abs_model,
        )
        brain = _make_brain(config)
        assert brain.config.reranker_model == abs_model


# ---------------------------------------------------------------------------
# prefetch_models.py helper function
# ---------------------------------------------------------------------------

class TestPrefetchModelsHelper:
    def test_human_size_bytes(self):
        from scripts.prefetch_models import _human_size
        assert _human_size(512) == "512.0 B"

    def test_human_size_kb(self):
        from scripts.prefetch_models import _human_size
        assert _human_size(2048) == "2.0 KB"

    def test_human_size_mb(self):
        from scripts.prefetch_models import _human_size
        assert _human_size(1024 * 1024) == "1.0 MB"

    def test_human_size_gb(self):
        from scripts.prefetch_models import _human_size
        assert _human_size(1024 ** 3) == "1.0 GB"

    def test_default_models_list(self):
        from scripts.prefetch_models import DEFAULT_MODELS
        assert "sentence-transformers/all-MiniLM-L6-v2" in DEFAULT_MODELS
        assert "BAAI/bge-reranker-base" in DEFAULT_MODELS

    def test_download_skips_existing(self, tmp_path, capsys):
        """If model dir already exists and is non-empty, download is skipped."""
        from scripts.prefetch_models import download_model
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")   # non-empty

        # snapshot_download (imported lazily inside the function) must NOT be called
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            download_model("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
            mock_dl.assert_not_called()

        captured = capsys.readouterr()
        assert "Already exists" in captured.out

    def test_download_calls_snapshot_download(self, tmp_path):
        """Empty/missing dir triggers snapshot_download."""
        from scripts.prefetch_models import download_model

        def fake_download(repo_id, local_dir, **kwargs):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model.bin").write_bytes(b"x" * 1024)

        with patch("huggingface_hub.snapshot_download", side_effect=fake_download) as mock_dl:
            download_model("BAAI/bge-reranker-base", str(tmp_path))
            mock_dl.assert_called_once()
            call_kwargs = mock_dl.call_args
            assert call_kwargs.kwargs["repo_id"] == "BAAI/bge-reranker-base"
            assert call_kwargs.kwargs["local_dir_use_symlinks"] is False

    def test_download_short_name_used_as_dir(self, tmp_path):
        """Model stored as <dir>/<short-name>, not <dir>/<org>/<name>."""
        from scripts.prefetch_models import download_model

        def fake_download(repo_id, local_dir, **kwargs):
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        with patch("huggingface_hub.snapshot_download", side_effect=fake_download) as mock_dl:
            download_model("BAAI/bge-reranker-base", str(tmp_path))
            used_dir = mock_dl.call_args.kwargs["local_dir"]
            assert os.path.basename(used_dir) == "bge-reranker-base"
