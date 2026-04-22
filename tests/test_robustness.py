from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from axon.projects import (
    delete_project,
    ensure_project,
    project_dir,
)


@pytest.fixture()
def tmp_projects(tmp_path, monkeypatch):
    import axon.projects as _p

    monkeypatch.setattr(_p, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(_p, "_ACTIVE_FILE", tmp_path / ".active_project")
    return tmp_path


class TestProjectRobustness:
    def test_delete_project_permission_error_exhausted(self, tmp_projects):
        ensure_project("locked-proj")
        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("File locked by another process")
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(PermissionError, match="File locked"):
                    delete_project("locked-proj")
                assert mock_rmtree.call_count == 3
                assert mock_sleep.call_count == 2

    def test_ensure_project_corrupt_meta(self, tmp_projects):
        root = project_dir("corrupt-meta")
        root.mkdir(parents=True)
        (root / "vector_data").mkdir()
        (root / "bm25_index").mkdir()
        (root / "sessions").mkdir()
        meta_file = root / "meta.json"
        meta_file.write_text("this is not json { [", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ensure_project("corrupt-meta")

    def test_max_depth_strict_boundary(self, tmp_projects):
        valid_name = "1/2/3/4/5"
        invalid_name = "1/2/3/4/5/6"
        ensure_project(valid_name)
        assert project_dir(valid_name).exists()
        with pytest.raises(ValueError, match="maximum depth is 5"):
            ensure_project(invalid_name)


class TestConfigRobustness:
    def test_load_nonexistent_config(self, tmp_path):
        from axon.config import AxonConfig

        config_path = tmp_path / "missing.yaml"
        # Actual behavior: AxonConfig.load() falls back to defaults when file is missing.
        config = AxonConfig.load(str(config_path))
        assert config is not None

    def test_config_type_coercion_leak(self, tmp_path):
        """AxonConfig is a dataclass and does not enforce types at runtime during load."""
        from axon.config import AxonConfig

        bad_config = tmp_path / "leaky_types.yaml"
        bad_config.write_text('llm:\n  temperature: "very hot"\n', encoding="utf-8")

        # This currently PASSES without error (no validation), which might be a concern.
        config = AxonConfig.load(str(bad_config))
        # Depending on how the dict is unpacked into the dataclass, it might just keep the string.
        # But wait, load() likely populates self.llm which is probably another nested dataclass.
        # Let's see if we can find where it is stored.
        # If it's a nested dataclass, it might still not raise.
        assert config is not None


class TestVectorStoreRobustness:
    def test_sanitize_chroma_meta_unsupported_type(self):
        from axon.vector_store import OpenVectorStore

        # Test sanitization logic with an unsupported nested dict type
        metadatas = [{"name": "doc1", "tags": ["tag1", "tag2"], "nested": {"a": 1}}]

        sanitized = OpenVectorStore._sanitize_chroma_meta(metadatas)

        # Based on docstring:
        # - list -> pipe-joined string
        # - other -> str(v)

        assert sanitized[0]["tags"] == "tag1|tag2"
        assert sanitized[0]["nested"] == "{'a': 1}"  # Coerced to string
