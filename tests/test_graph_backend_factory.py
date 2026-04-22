"""Tests for GraphBackend factory and S3 wiring.

Covers:
  - get_graph_backend() returns the correct backend for each config value
  - Unknown config value raises ValueError with helpful message
  - AxonConfig.graph_backend field exists with correct default
  - projects.py: ensure_project persists graph_backend; immutability enforced
  - projects.py: get_project_graph_backend reads the field correctly
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig
from axon.graph_backends.base import GraphBackend
from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend
from axon.graph_backends.factory import get_graph_backend
from axon.graph_backends.graphrag_backend import GraphRagBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_brain(backend: str = "graphrag") -> MagicMock:
    brain = MagicMock()
    brain.config.graph_backend = backend
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_levels = {}
    brain._community_summaries = {}
    return brain


# ---------------------------------------------------------------------------
# AxonConfig field
# ---------------------------------------------------------------------------


class TestAxonConfigGraphBackendField:
    def test_default_is_graphrag(self):
        cfg = AxonConfig()
        assert cfg.graph_backend == "graphrag"

    def test_field_accepts_dynamic(self):
        cfg = AxonConfig(graph_backend="dynamic")
        assert cfg.graph_backend == "dynamic"

    def test_field_is_in_dataclass_fields(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(AxonConfig)}
        assert "graph_backend" in field_names


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetGraphBackend:
    def test_graphrag_config_returns_graphrag_backend(self):
        brain = _fake_brain("graphrag")
        backend = get_graph_backend(brain)
        assert isinstance(backend, GraphRagBackend)

    def test_dynamic_config_returns_dynamic_backend(self, tmp_path):
        brain = _fake_brain("dynamic_graph")
        brain.config.bm25_path = str(tmp_path)
        backend = get_graph_backend(brain)
        assert isinstance(backend, DynamicGraphBackend)

    def test_result_satisfies_protocol(self):
        brain = _fake_brain("graphrag")
        backend = get_graph_backend(brain)
        assert isinstance(backend, GraphBackend)

    def test_unknown_backend_raises_value_error(self):
        brain = _fake_brain("neo4j")
        with pytest.raises(ValueError, match="Unknown graph_backend 'neo4j'"):
            get_graph_backend(brain)

    def test_error_message_lists_valid_options(self):
        brain = _fake_brain("bad")
        with pytest.raises(ValueError, match="graphrag"):
            get_graph_backend(brain)

    def test_none_graph_backend_falls_back_to_graphrag(self):
        brain = MagicMock()
        brain.config.graph_backend = None
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        backend = get_graph_backend(brain)
        assert isinstance(backend, GraphRagBackend)

    def test_missing_config_falls_back_to_graphrag(self):
        brain = MagicMock(spec=[])  # no attributes at all
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        backend = get_graph_backend(brain)
        assert isinstance(backend, GraphRagBackend)

    def test_graphrag_backend_holds_brain_reference(self):
        brain = _fake_brain("graphrag")
        backend = get_graph_backend(brain)
        assert isinstance(backend, GraphRagBackend)
        assert backend._brain is brain


# ---------------------------------------------------------------------------
# projects.py: graph_backend persistence
# ---------------------------------------------------------------------------


class TestProjectsGraphBackendPersistence:
    def test_ensure_project_writes_graph_backend_to_meta(self):
        from axon.projects import _ensure_single_project

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=tmp / "p"):
                _ensure_single_project("p", description="", graph_backend="graphrag")

            meta = json.loads((tmp / "p" / "meta.json").read_text())
            assert meta["graph_backend"] == "graphrag"

    def test_ensure_project_persists_dynamic_backend(self):
        from axon.projects import _ensure_single_project

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=tmp / "p"):
                _ensure_single_project("p", description="", graph_backend="dynamic")

            meta = json.loads((tmp / "p" / "meta.json").read_text())
            assert meta["graph_backend"] == "dynamic"

    def test_ensure_project_immutability_raises_on_conflict(self):
        from axon.projects import _ensure_single_project

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=tmp / "p"):
                # First call — creates with "graphrag"
                _ensure_single_project("p", description="", graph_backend="graphrag")
                # Second call — attempts to switch to "dynamic" → must raise
                with pytest.raises(ValueError, match="immutable"):
                    _ensure_single_project("p", description="", graph_backend="dynamic")

    def test_ensure_project_same_backend_does_not_raise(self):
        from axon.projects import _ensure_single_project

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=tmp / "p"):
                _ensure_single_project("p", description="", graph_backend="graphrag")
                # Calling again with same value is idempotent
                _ensure_single_project("p", description="", graph_backend="graphrag")

    def test_legacy_project_backfilled_with_graphrag(self):
        """meta.json without graph_backend is backfilled on next ensure call."""
        from axon.projects import _ensure_single_project

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            proj = tmp / "p"
            proj.mkdir()
            (proj / "vector_data").mkdir()
            (proj / "bm25_index").mkdir()
            (proj / "sessions").mkdir()
            # Write legacy meta.json without graph_backend
            (proj / "meta.json").write_text(
                json.dumps({"name": "p", "project_id": "proj_old"}), encoding="utf-8"
            )

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=proj):
                _ensure_single_project("p", description="", graph_backend="graphrag")

            meta = json.loads((proj / "meta.json").read_text())
            assert meta["graph_backend"] == "graphrag"

    def test_get_project_graph_backend_reads_meta(self):
        from axon.projects import get_project_graph_backend

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            proj = tmp / "p"
            proj.mkdir()
            (proj / "meta.json").write_text(
                json.dumps({"graph_backend": "dynamic"}), encoding="utf-8"
            )

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=proj):
                result = get_project_graph_backend("p")

            assert result == "dynamic"

    def test_get_project_graph_backend_missing_project_returns_default(self):
        from axon.projects import get_project_graph_backend

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=tmp / "nonexistent"):
                result = get_project_graph_backend("nonexistent")

            assert result == "graphrag"

    def test_get_project_graph_backend_missing_field_returns_default(self):
        from axon.projects import get_project_graph_backend

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            proj = tmp / "p"
            proj.mkdir()
            (proj / "meta.json").write_text(json.dumps({"name": "p"}), encoding="utf-8")

            import axon.projects as proj_mod

            with patch.object(proj_mod, "project_dir", return_value=proj):
                result = get_project_graph_backend("p")

            assert result == "graphrag"
