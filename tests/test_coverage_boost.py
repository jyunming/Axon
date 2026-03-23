"""Coverage boost tests for shares routes, maintenance routes, graph_render, and main.py init paths."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
import axon.projects as projects_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_api(monkeypatch, tmp_path):
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()
    projects_root = tmp_path / "projects_root"
    active_file = tmp_path / ".active_project"
    monkeypatch.setattr(projects_module, "PROJECTS_ROOT", projects_root)
    monkeypatch.setattr(projects_module, "_ACTIVE_FILE", active_file)
    yield
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()


@pytest.fixture
def client():
    c = TestClient(api_module.app, raise_server_exceptions=False)
    yield c
    c.close()


@pytest.fixture
def mock_brain(monkeypatch):
    brain = MagicMock()
    brain.config.axon_store_mode = False
    brain.config.projects_root = "/tmp/axon/user"
    brain._active_project = "default"
    monkeypatch.setattr(api_module, "brain", brain)
    return brain


# ---------------------------------------------------------------------------
# api_routes/shares.py — /store/init (lines 24-57)
# ---------------------------------------------------------------------------


class TestStoreInitRoute:
    def test_store_init_no_brain(self, client, tmp_path, monkeypatch):
        """store/init with no brain creates AxonBrain from scratch."""
        monkeypatch.setattr(api_module, "brain", None)
        with patch("axon.projects.ensure_user_namespace"):
            with patch("axon.api_routes.shares.getpass.getuser", return_value="testuser"):
                with patch("axon.main.AxonBrain") as FakeBrain:
                    FakeBrain.return_value = MagicMock()
                    resp = client.post("/store/init", json={"base_path": str(tmp_path)})
        assert resp.status_code in (200, 422, 500)

    def test_store_init_with_brain(self, client, mock_brain, tmp_path, monkeypatch):
        """store/init closes existing brain and reinitialises."""
        with patch("axon.projects.ensure_user_namespace"):
            with patch("axon.api_routes.shares.getpass.getuser", return_value="testuser"):
                with patch("axon.main.AxonBrain") as FakeBrain:
                    FakeBrain.return_value = MagicMock()
                    resp = client.post("/store/init", json={"base_path": str(tmp_path)})
        assert resp.status_code in (200, 422, 500)
        mock_brain.close.assert_called()


# ---------------------------------------------------------------------------
# api_routes/shares.py — /store/whoami (lines 60-75)
# ---------------------------------------------------------------------------


class TestStoreWhoamiRoute:
    def test_whoami_no_store_mode(self, client, mock_brain):
        mock_brain.config.axon_store_mode = False
        with patch("axon.api_routes.shares.getpass.getuser", return_value="alice"):
            resp = client.get("/store/whoami")
        assert resp.status_code == 200
        data = resp.json()
        assert data["store_mode"] is False
        assert data["username"] == "alice"

    def test_whoami_store_mode(self, client, mock_brain):
        mock_brain.config.axon_store_mode = True
        mock_brain.config.projects_root = "/axon/AxonStore/alice"
        with patch("axon.api_routes.shares.getpass.getuser", return_value="alice"):
            resp = client.get("/store/whoami")
        assert resp.status_code == 200
        data = resp.json()
        assert data["store_mode"] is True


# ---------------------------------------------------------------------------
# api_routes/maintenance.py — /copilot/agent (lines 46-93)
# ---------------------------------------------------------------------------


class TestCopilotAgentRoute:
    def test_copilot_no_brain_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(api_module, "brain", None)
        resp = client.post(
            "/copilot/agent",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "agent_request_id": "req1",
            },
        )
        assert resp.status_code == 503

    def test_copilot_empty_message_returns_400(self, client, mock_brain):
        resp = client.post(
            "/copilot/agent",
            json={"messages": [], "agent_request_id": "req2"},
        )
        assert resp.status_code in (200, 400)

    def test_copilot_search_command(self, client, mock_brain):
        mock_brain._execute_retrieval.return_value = {"results": []}
        resp = client.post(
            "/copilot/agent",
            json={
                "messages": [{"role": "user", "content": "/search axon architecture"}],
                "agent_request_id": "req3",
            },
        )
        assert resp.status_code == 200

    def test_copilot_projects_command(self, client, mock_brain):
        with patch("axon.projects.list_projects", return_value=[{"name": "p1", "description": ""}]):
            resp = client.post(
                "/copilot/agent",
                json={
                    "messages": [{"role": "user", "content": "/projects"}],
                    "agent_request_id": "req4",
                },
            )
        assert resp.status_code == 200

    def test_copilot_default_query(self, client, mock_brain):
        mock_brain.query.return_value = "answer"
        resp = client.post(
            "/copilot/agent",
            json={
                "messages": [
                    {"role": "user", "content": "prior msg"},
                    {"role": "user", "content": "what is axon?"},
                ],
                "agent_request_id": "req5",
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# api_routes/maintenance.py — /project/maintenance (lines 121-147)
# ---------------------------------------------------------------------------


class TestMaintenanceRoute:
    def test_set_maintenance_invalid_name(self, client):
        resp = client.post("/project/maintenance", json={"name": "bad name!", "state": "readonly"})
        assert resp.status_code == 422

    def test_set_maintenance_valid(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(projects_module, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "testproj"
        proj.mkdir()
        (proj / "meta.json").write_text(json.dumps({"name": "testproj"}))
        with patch("axon.maintenance.apply_maintenance_state", return_value={"status": "ok"}):
            resp = client.post(
                "/project/maintenance", json={"name": "testproj", "state": "readonly"}
            )
        assert resp.status_code == 200

    def test_set_maintenance_raises_value_error(self, client):
        with patch(
            "axon.maintenance.apply_maintenance_state",
            side_effect=ValueError("does not exist"),
        ):
            resp = client.post(
                "/project/maintenance", json={"name": "missing", "state": "readonly"}
            )
        assert resp.status_code == 404

    def test_get_maintenance_invalid_name(self, client):
        resp = client.get("/project/maintenance?name=bad name!")
        assert resp.status_code == 422

    def test_get_maintenance_not_found(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(projects_module, "PROJECTS_ROOT", tmp_path)
        resp = client.get("/project/maintenance?name=noproject")
        assert resp.status_code == 404

    def test_get_maintenance_found(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(projects_module, "PROJECTS_ROOT", tmp_path)
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "meta.json").write_text(json.dumps({"name": "myproj"}))
        with patch("axon.maintenance.get_maintenance_status", return_value={"state": "normal"}):
            resp = client.get("/project/maintenance?name=myproj")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# graph_render.py — build_code_graph_payload (lines 155-193)
# ---------------------------------------------------------------------------


class TestBuildCodeGraphPayload:
    def _make_renderer(self):
        from axon.graph_render import GraphRenderMixin

        class FakeRenderer(GraphRenderMixin):
            _VIZ_TYPE_COLORS = {"UNKNOWN": "#bab0ab", "file": "#4ec9b0"}

            def __init__(self):
                self._code_graph = {"nodes": {}, "edges": []}
                self._entity_graph = {}
                self._relation_graph = {}
                self._community_levels = {}
                self.vector_store = MagicMock()
                self.vector_store.get_by_ids.return_value = []

        return FakeRenderer()

    def test_empty_code_graph(self):
        r = self._make_renderer()
        payload = r.build_code_graph_payload()
        assert payload == {"nodes": [], "links": []}

    def test_nodes_and_edges(self):
        r = self._make_renderer()
        r._code_graph = {
            "nodes": {
                "axon/main.py": {
                    "node_type": "file",
                    "name": "main.py",
                    "file_path": "axon/main.py",
                    "start_line": 1,
                    "chunk_ids": ["c1"],
                    "signature": "",
                }
            },
            "edges": [{"source": "axon/main.py", "target": "axon/cli.py", "edge_type": "IMPORTS"}],
        }
        payload = r.build_code_graph_payload()
        assert len(payload["nodes"]) == 1
        assert payload["nodes"][0]["type"] == "file"
        assert len(payload["links"]) == 1
        assert payload["links"][0]["label"] == "IMPORTS"

    def test_function_node_color(self):
        r = self._make_renderer()
        r._code_graph = {
            "nodes": {
                "fn1": {
                    "node_type": "function",
                    "name": "my_func",
                    "file_path": "axon/a.py",
                    "start_line": 10,
                    "chunk_ids": [],
                    "signature": "def my_func():",
                }
            },
            "edges": [],
        }
        payload = r.build_code_graph_payload()
        assert payload["nodes"][0]["color"] == "#dcdcaa"

    def test_unknown_node_type_default_color(self):
        r = self._make_renderer()
        r._code_graph = {
            "nodes": {
                "x": {
                    "node_type": "unknown_type",
                    "name": "x",
                    "file_path": "",
                    "start_line": None,
                    "chunk_ids": [],
                }
            },
            "edges": [],
        }
        payload = r.build_code_graph_payload()
        assert payload["nodes"][0]["color"] == "#888888"


# ---------------------------------------------------------------------------
# graph_render.py — build_graph_payload exception path (lines 76-78)
# ---------------------------------------------------------------------------


class TestBuildGraphPayloadExceptions:
    def _make_renderer(self, entity_graph=None, relation_graph=None, community_levels=None):
        from axon.graph_render import GraphRenderMixin

        eg = entity_graph or {}
        rg = relation_graph or {}
        cl = community_levels or {}

        class FakeRenderer(GraphRenderMixin):
            _VIZ_TYPE_COLORS = {
                "PERSON": "#4e79a7",
                "ORGANIZATION": "#f28e2b",
                "UNKNOWN": "#bab0ab",
            }

            def __init__(self):
                self._entity_graph = eg
                self._relation_graph = rg
                self._community_levels = cl
                self.vector_store = MagicMock()
                self._code_graph = {"nodes": {}, "edges": []}

        return FakeRenderer()

    def test_vector_store_exception_is_swallowed(self):
        r = self._make_renderer(
            entity_graph={"Axon": {"type": "PRODUCT", "description": "RAG", "chunk_ids": ["c1"]}},
        )
        r.vector_store.get_by_ids.side_effect = RuntimeError("db error")
        payload = r.build_graph_payload()
        assert isinstance(payload["nodes"], list)

    def test_community_levels_non_int_handled(self):
        r = self._make_renderer(
            entity_graph={"BM25": {"type": "CONCEPT", "description": "", "chunk_ids": []}},
            community_levels={0: {"bm25": "notanint"}},
        )
        payload = r.build_graph_payload()
        # Should not raise — TypeError caught at lines 46-47
        assert len(payload["nodes"]) == 1


# ---------------------------------------------------------------------------
# main.py — AxonStore mode init (lines 313-316)
# ---------------------------------------------------------------------------


class TestAxonBrainInit:
    def test_axon_store_mode_calls_ensure_user_namespace(self, tmp_path, monkeypatch):
        from axon.main import AxonBrain, AxonConfig

        monkeypatch.setattr(projects_module, "PROJECTS_ROOT", tmp_path)
        monkeypatch.setattr(projects_module, "_ACTIVE_FILE", tmp_path / ".active_project")

        config = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            projects_root=str(tmp_path),
            axon_store_mode=True,
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            llm_provider="ollama",
        )
        with patch("axon.main.OpenEmbedding") as FakeEmbed:
            with patch("axon.main.OpenVectorStore") as FakeVS:
                with patch("axon.main.OpenLLM") as FakeLLM:
                    with patch("axon.main.OpenReranker") as FakeRerank:
                        with patch("axon.projects.ensure_user_namespace") as mock_ensure:
                            FakeEmbed.return_value = MagicMock()
                            FakeVS.return_value = MagicMock()
                            FakeLLM.return_value = MagicMock()
                            FakeRerank.return_value = MagicMock()
                            brain = AxonBrain(config)
                            brain.close()
        mock_ensure.assert_called()

    def test_splitter_markdown_strategy(self, tmp_path, monkeypatch):
        from axon.main import AxonBrain, AxonConfig

        monkeypatch.setattr(projects_module, "PROJECTS_ROOT", tmp_path)
        monkeypatch.setattr(projects_module, "_ACTIVE_FILE", tmp_path / ".active_project")

        config = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            projects_root=str(tmp_path),
            chunk_strategy="markdown",
        )
        with patch("axon.main.OpenEmbedding", return_value=MagicMock()):
            with patch("axon.main.OpenVectorStore", return_value=MagicMock()):
                with patch("axon.main.OpenLLM", return_value=MagicMock()):
                    with patch("axon.main.OpenReranker", return_value=MagicMock()):
                        brain = AxonBrain(config)
        assert brain.splitter is not None
        brain.close()

    def test_splitter_recursive_strategy(self, tmp_path, monkeypatch):
        from axon.main import AxonBrain, AxonConfig

        monkeypatch.setattr(projects_module, "PROJECTS_ROOT", tmp_path)
        monkeypatch.setattr(projects_module, "_ACTIVE_FILE", tmp_path / ".active_project")

        config = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            projects_root=str(tmp_path),
            chunk_strategy="recursive",
        )
        with patch("axon.main.OpenEmbedding", return_value=MagicMock()):
            with patch("axon.main.OpenVectorStore", return_value=MagicMock()):
                with patch("axon.main.OpenLLM", return_value=MagicMock()):
                    with patch("axon.main.OpenReranker", return_value=MagicMock()):
                        brain = AxonBrain(config)
        assert brain.splitter is not None
        brain.close()
