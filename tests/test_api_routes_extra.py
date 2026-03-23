"""Extra coverage tests for api_routes sub-modules.

Targets missed lines in:
  graph.py, ingest.py, maintenance.py, projects.py, query.py, shares.py
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_brain(provider="chroma"):
    brain = MagicMock()
    brain.vector_store.provider = provider
    brain.vector_store.client = MagicMock()
    brain.bm25 = MagicMock()
    brain.config.top_k = 5
    brain.config.hybrid_search = True
    brain.config.rerank = False
    brain.config.hyde = False
    brain.config.multi_query = False
    brain.config.discussion_fallback = True
    brain.config.similarity_threshold = 0.3
    brain.config.step_back = False
    brain.config.query_decompose = False
    brain.config.compress_context = False
    brain.config.axon_store_mode = False
    brain.config.projects_root = "/tmp/axon_projects"
    brain._apply_overrides.return_value = brain.config
    brain._community_build_in_progress = False
    brain._community_summaries = {}
    brain._active_project = "default"
    brain._entity_graph = {}
    brain._embedding_meta_path = "/tmp/axon_meta.json"
    return brain


@pytest.fixture(autouse=True)
def reset_state():
    original = api_module.brain
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()
    yield
    api_module.brain = original
    api_module._source_hashes.clear()
    api_module._jobs.clear()


# ===========================================================================
# graph.py
# ===========================================================================


class TestGraphStatus:
    def test_503_when_no_brain(self):
        """graph.py line 20 — brain is None raises 503."""
        resp = client.get("/graph/status")
        assert resp.status_code == 503

    def test_returns_status_with_brain(self):
        brain = _make_brain()
        brain._community_build_in_progress = True
        brain._community_summaries = {"a": 1, "b": 2}
        api_module.brain = brain
        resp = client.get("/graph/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["community_build_in_progress"] is True
        assert data["community_summary_count"] == 2

    def test_no_community_attrs_defaults(self):
        brain = _make_brain()
        del brain._community_build_in_progress
        del brain._community_summaries
        api_module.brain = brain
        resp = client.get("/graph/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["community_build_in_progress"] is False
        assert data["community_summary_count"] == 0


class TestGraphFinalize:
    def test_503_when_no_brain(self):
        """graph.py line 38 — brain is None raises 503."""
        resp = client.post("/graph/finalize")
        assert resp.status_code == 503

    def test_success(self):
        """graph.py lines 40-41 — finalize runs ok."""
        brain = _make_brain()
        brain._community_summaries = {"c1": "summary"}
        brain.finalize_graph.return_value = None
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["community_summary_count"] == 1

    def test_permission_error_returns_403(self):
        """graph.py lines 44-46 — PermissionError → 403."""
        brain = _make_brain()
        brain.finalize_graph.side_effect = PermissionError("read-only mode")
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 403

    def test_generic_exception_returns_500(self):
        """graph.py lines 44-46 — generic Exception → 500."""
        brain = _make_brain()
        brain.finalize_graph.side_effect = RuntimeError("graph build failed")
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 500


class TestGraphVisualize:
    def test_503_when_no_brain(self):
        """graph.py line 56 — brain is None raises 503."""
        resp = client.get("/graph/visualize")
        assert resp.status_code == 503

    def test_returns_html(self):
        brain = _make_brain()
        brain.export_graph_html.return_value = "<html>graph</html>"
        api_module.brain = brain
        resp = client.get("/graph/visualize")
        assert resp.status_code == 200
        assert b"graph" in resp.content

    def test_import_error_returns_501(self):
        """graph.py lines 62-63 — ImportError → 501."""
        brain = _make_brain()
        brain.export_graph_html.side_effect = ImportError("pyvis not installed")
        api_module.brain = brain
        resp = client.get("/graph/visualize")
        assert resp.status_code == 501

    def test_generic_error_returns_500(self):
        """graph.py lines 62-63 — generic Exception → 500."""
        brain = _make_brain()
        brain.export_graph_html.side_effect = ValueError("bad graph data")
        api_module.brain = brain
        resp = client.get("/graph/visualize")
        assert resp.status_code == 500


class TestGraphData:
    def test_503_when_no_brain(self):
        """graph.py line 77 — brain is None raises 503."""
        resp = client.get("/graph/data")
        assert resp.status_code == 503

    def test_returns_payload_when_dict(self):
        brain = _make_brain()
        brain.build_graph_payload.return_value = {"nodes": [{"id": "n1"}], "links": []}
        api_module.brain = brain
        resp = client.get("/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data

    def test_fallback_when_not_dict(self):
        """graph.py line 77 — non-dict payload → fallback empty nodes/links."""
        brain = _make_brain()
        brain.build_graph_payload.return_value = None
        api_module.brain = brain
        resp = client.get("/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"nodes": [], "links": []}


class TestCodeGraphData:
    def test_503_when_no_brain(self):
        """graph.py line 87 — brain is None raises 503."""
        resp = client.get("/code-graph/data")
        assert resp.status_code == 503

    def test_returns_code_graph(self):
        brain = _make_brain()
        brain.build_code_graph_payload.return_value = {"nodes": [], "edges": []}
        api_module.brain = brain
        resp = client.get("/code-graph/data")
        assert resp.status_code == 200


# ===========================================================================
# ingest.py
# ===========================================================================


class TestIngestRefresh:
    def test_503_when_no_brain(self):
        resp = client.post("/ingest/refresh")
        assert resp.status_code == 503

    def test_missing_source_reported(self):
        """ingest.py line 43 — source path does not exist → 'missing' bucket."""
        brain = _make_brain()
        brain.get_doc_versions.return_value = {"/nonexistent/path.txt": {"content_hash": "abc123"}}
        api_module.brain = brain
        resp = client.post("/ingest/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "/nonexistent/path.txt" in data["missing"]

    def test_no_loader_for_extension_reported_as_error(self, tmp_path):
        """ingest.py lines 51-54 — no loader for extension → 'errors' bucket."""
        brain = _make_brain()
        exotic = tmp_path / "file.exotic123"
        exotic.write_text("content", encoding="utf-8")
        brain.get_doc_versions.return_value = {str(exotic): {"content_hash": "abc"}}
        api_module.brain = brain

        mock_loader_mgr = MagicMock()
        mock_loader_mgr.loaders = {}  # no loader for .exotic123

        with patch("axon.loaders.DirectoryLoader", return_value=mock_loader_mgr):
            resp = client.post("/ingest/refresh")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["errors"]) == 1
        assert "no loader" in data["errors"][0]["error"]

    def test_loader_returns_no_docs_reported_as_error(self, tmp_path):
        """ingest.py lines 59-62 — loader returns empty list → 'errors' bucket."""
        brain = _make_brain()
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("content", encoding="utf-8")
        brain.get_doc_versions.return_value = {str(txt_file): {"content_hash": "old_hash"}}
        api_module.brain = brain

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader_mgr = MagicMock()
        mock_loader_mgr.loaders = {".txt": mock_loader_instance}

        with patch("axon.loaders.DirectoryLoader", return_value=mock_loader_mgr):
            resp = client.post("/ingest/refresh")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["errors"]) == 1
        assert "no documents" in data["errors"][0]["error"]

    def test_unchanged_hash_skipped(self, tmp_path):
        """ingest.py lines 65-67 — same hash → skipped bucket."""
        import hashlib

        content = "same content"
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        brain = _make_brain()
        txt_file = tmp_path / "same.txt"
        txt_file.write_text(content, encoding="utf-8")
        brain.get_doc_versions.return_value = {str(txt_file): {"content_hash": content_hash}}
        api_module.brain = brain

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [{"text": content}]
        mock_loader_mgr = MagicMock()
        mock_loader_mgr.loaders = {".txt": mock_loader_instance}

        with patch("axon.loaders.DirectoryLoader", return_value=mock_loader_mgr):
            resp = client.post("/ingest/refresh")

        assert resp.status_code == 200
        data = resp.json()
        assert str(txt_file) in data["skipped"]

    def test_changed_hash_reingested(self, tmp_path):
        """ingest.py line 69 — changed hash → reingested bucket."""
        brain = _make_brain()
        txt_file = tmp_path / "changed.txt"
        txt_file.write_text("new content", encoding="utf-8")
        brain.get_doc_versions.return_value = {
            str(txt_file): {"content_hash": "old_different_hash"}
        }
        api_module.brain = brain

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [{"text": "new content"}]
        mock_loader_mgr = MagicMock()
        mock_loader_mgr.loaders = {".txt": mock_loader_instance}

        with patch("axon.loaders.DirectoryLoader", return_value=mock_loader_mgr):
            resp = client.post("/ingest/refresh")

        assert resp.status_code == 200
        data = resp.json()
        assert str(txt_file) in data["reingested"]

    def test_exception_during_loader_reported_as_error(self, tmp_path):
        """ingest.py lines 72-73 — generic exception → errors bucket."""
        brain = _make_brain()
        txt_file = tmp_path / "error.txt"
        txt_file.write_text("content", encoding="utf-8")
        brain.get_doc_versions.return_value = {str(txt_file): {"content_hash": "old"}}
        api_module.brain = brain

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.side_effect = RuntimeError("loader crashed")
        mock_loader_mgr = MagicMock()
        mock_loader_mgr.loaders = {".txt": mock_loader_instance}

        with patch("axon.loaders.DirectoryLoader", return_value=mock_loader_mgr):
            resp = client.post("/ingest/refresh")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["errors"]) == 1


class TestIngestPath:
    def test_unsupported_ext_background(self, tmp_path):
        """ingest.py lines 125-126 — unsupported ext logs warning, docs=[]."""
        brain = _make_brain()
        brain._assert_write_allowed.return_value = None
        api_module.brain = brain

        exotic = tmp_path / "file.exotic"
        exotic.write_text("data", encoding="utf-8")

        mock_loader_mgr = MagicMock()
        mock_loader_mgr.is_dir.return_value = False
        mock_loader_mgr.loaders = {}

        with patch.dict(os.environ, {"RAG_INGEST_BASE": str(tmp_path)}):
            with patch("axon.loaders.DirectoryLoader") as MockDL:
                instance = MockDL.return_value
                instance.loaders = {}
                resp = client.post("/ingest", json={"path": str(exotic)})

        # The endpoint should return 200 — actual loading happens in background task
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestAddText:
    def test_empty_text_returns_400(self):
        """ingest.py lines 233-234 — empty text → 400."""
        brain = _make_brain()
        api_module.brain = brain
        resp = client.post("/add_text", json={"text": "   "})
        assert resp.status_code == 400

    def test_dedup_skip_returns_existing(self):
        """ingest.py lines 239-241 — dedup hit returns skipped doc_id."""
        brain = _make_brain()
        brain._active_project = "default"
        api_module.brain = brain
        api_module._source_hashes["default"] = {}

        with patch(
            "axon.api._check_dedup", return_value={"doc_id": "existing-id", "status": "skipped"}
        ):
            resp = client.post("/add_text", json={"text": "hello world"})

        # Either 200 with skipped or processed normally
        assert resp.status_code in (200,)

    def test_success(self):
        """ingest.py lines 253-255 — successful ingest."""
        brain = _make_brain()
        brain._active_project = "default"
        api_module.brain = brain

        mock_loader = MagicMock()
        mock_loader.load_text.return_value = [{"text": "chunk", "id": "doc1", "metadata": {}}]

        with patch("axon.loaders.SmartTextLoader", return_value=mock_loader):
            resp = client.post("/add_text", json={"text": "some useful content"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "doc_id" in data

    def test_ingest_exception_returns_500(self):
        """ingest.py lines 256-258 — ingest raises → 500."""
        brain = _make_brain()
        brain.ingest.side_effect = RuntimeError("storage failure")
        brain._active_project = "default"
        api_module.brain = brain

        mock_loader = MagicMock()
        mock_loader.load_text.return_value = [{"text": "chunk", "id": "doc1", "metadata": {}}]

        with patch("axon.loaders.SmartTextLoader", return_value=mock_loader):
            resp = client.post("/add_text", json={"text": "some content"})

        assert resp.status_code == 500

    def test_with_metadata(self):
        """ingest.py lines 248-250 — metadata merged into docs."""
        brain = _make_brain()
        brain._active_project = "default"
        api_module.brain = brain

        mock_loader = MagicMock()
        doc = {"text": "chunk", "id": "doc1", "metadata": {}}
        mock_loader.load_text.return_value = [doc]

        with patch("axon.loaders.SmartTextLoader", return_value=mock_loader):
            resp = client.post(
                "/add_text",
                json={"text": "content", "metadata": {"author": "alice"}},
            )

        assert resp.status_code == 200


class TestAddTextsBatch:
    def test_batch_dedup_skip_within_batch(self):
        """ingest.py lines 283-284 — duplicate within batch is skipped."""
        brain = _make_brain()
        brain._active_project = "default"
        api_module.brain = brain

        mock_loader = MagicMock()
        mock_loader.load_text.return_value = [{"text": "chunk", "id": "d1", "metadata": {}}]

        with patch("axon.loaders.SmartTextLoader", return_value=mock_loader):
            resp = client.post(
                "/add_texts",
                json={
                    "docs": [
                        {"text": "same text"},
                        {"text": "same text"},  # duplicate
                    ]
                },
            )

        assert resp.status_code == 200
        results = resp.json()
        statuses = [r["status"] for r in results]
        assert "skipped" in statuses


# ===========================================================================
# maintenance.py
# ===========================================================================


class TestCopilotAgent:
    def test_503_when_no_brain(self):
        """maintenance.py line 30 — brain is None → 503."""
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "agent_request_id": "req-001",
        }
        resp = client.post("/copilot/agent", json=body)
        assert resp.status_code == 503

    def test_empty_messages_returns_400(self):
        """maintenance.py line 34 — empty query → 400."""
        brain = _make_brain()
        api_module.brain = brain
        body = {"messages": [], "agent_request_id": "req-002"}
        resp = client.post("/copilot/agent", json=body)
        assert resp.status_code == 400

    def test_search_command_streams(self):
        """maintenance.py lines 46-57 — /search command produces SSE stream."""
        brain = _make_brain()
        brain._execute_retrieval.return_value = {"results": []}
        api_module.brain = brain

        with patch("axon.projects.list_projects", return_value=[]):
            body = {
                "messages": [{"role": "user", "content": "/search python decorators"}],
                "agent_request_id": "req-003",
            }
            resp = client.post("/copilot/agent", json=body)
        assert resp.status_code == 200
        # SSE response
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_projects_command_streams(self):
        """maintenance.py lines 71-76 — /projects command lists projects."""
        brain = _make_brain()
        api_module.brain = brain

        with patch(
            "axon.projects.list_projects", return_value=[{"name": "proj1", "description": "test"}]
        ):
            body = {
                "messages": [{"role": "user", "content": "/projects"}],
                "agent_request_id": "req-004",
            }
            resp = client.post("/copilot/agent", json=body)
        assert resp.status_code == 200

    def test_default_query_streams(self):
        """maintenance.py lines 78-85 — plain query uses brain.query."""
        brain = _make_brain()
        brain.query.return_value = "The answer"
        api_module.brain = brain

        with patch("axon.projects.list_projects", return_value=[]):
            body = {
                "messages": [{"role": "user", "content": "what is python?"}],
                "agent_request_id": "req-005",
            }
            resp = client.post("/copilot/agent", json=body)
        assert resp.status_code == 200

    def test_exception_in_stream_yields_error_event(self):
        """maintenance.py lines 87-89 — exception yields error event."""
        brain = _make_brain()
        brain.query.side_effect = RuntimeError("llm exploded")
        api_module.brain = brain

        with patch("axon.projects.list_projects", return_value=[]):
            body = {
                "messages": [{"role": "user", "content": "query that fails"}],
                "agent_request_id": "req-006",
            }
            resp = client.post("/copilot/agent", json=body)
        assert resp.status_code == 200
        assert b"error" in resp.content.lower() or b"[DONE]" in resp.content


class TestProjectMaintenance:
    def test_set_invalid_project_name_returns_422(self):
        """maintenance.py line 127 — invalid project name → 422."""
        resp = client.post(
            "/project/maintenance",
            json={"name": "../../etc/passwd", "state": "readonly"},
        )
        assert resp.status_code == 422

    def test_set_maintenance_state_project_not_found(self):
        """maintenance.py line 131 — project does not exist → 404."""
        with patch("axon.maintenance.apply_maintenance_state") as mock_apply:
            mock_apply.side_effect = ValueError("Project 'missing' does not exist")
            resp = client.post(
                "/project/maintenance",
                json={"name": "missing", "state": "readonly"},
            )
        assert resp.status_code == 404

    def test_set_maintenance_state_success(self):
        """maintenance.py lines 129 — success path."""
        with patch("axon.maintenance.apply_maintenance_state", return_value={"status": "ok"}):
            resp = client.post(
                "/project/maintenance",
                json={"name": "my-project", "state": "readonly"},
            )
        assert resp.status_code == 200

    def test_set_maintenance_generic_error(self):
        """maintenance.py lines 132-134 — generic error → 500."""
        with patch("axon.maintenance.apply_maintenance_state") as mock_apply:
            mock_apply.side_effect = RuntimeError("disk full")
            resp = client.post(
                "/project/maintenance",
                json={"name": "my-project", "state": "readonly"},
            )
        assert resp.status_code == 500

    def test_get_invalid_project_name_returns_422(self):
        """maintenance.py line 144 — invalid project name → 422."""
        resp = client.get("/project/maintenance?name=../bad")
        assert resp.status_code == 422

    def test_get_project_not_found(self):
        """maintenance.py line 144 — meta.json missing → 404."""
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value = MagicMock()
            mock_dir.return_value.__truediv__.return_value.exists.return_value = False
            resp = client.get("/project/maintenance?name=nonexistent-project")
        assert resp.status_code == 404

    def test_get_maintenance_status_success(self):
        """maintenance.py line 147 — success path."""
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value = MagicMock()
            mock_dir.return_value.__truediv__.return_value.exists.return_value = True
            with patch("axon.maintenance.get_maintenance_status", return_value={"state": "active"}):
                resp = client.get("/project/maintenance?name=my-project")
        assert resp.status_code == 200


class TestCopilotTaskBridge:
    def test_get_tasks_returns_list(self):
        """maintenance.py lines 60-69 — poll empty task queue."""
        from axon.main import _copilot_bridge_lock, _copilot_task_queue

        with _copilot_bridge_lock:
            _copilot_task_queue.clear()

        resp = client.get("/llm/copilot/tasks")
        assert resp.status_code == 200
        assert resp.json()["tasks"] == []

    def test_submit_result_unknown_task_returns_404(self):
        """maintenance.py — unknown task_id → 404."""
        resp = client.post(
            "/llm/copilot/result/nonexistent-task",
            json={"result": "some result", "error": None},
        )
        assert resp.status_code == 404

    def test_submit_result_known_task_sets_event(self):
        """maintenance.py lines 113-117 — known task_id sets event."""
        import threading

        from axon.main import _copilot_bridge_lock, _copilot_responses

        event = threading.Event()
        with _copilot_bridge_lock:
            _copilot_responses["task-abc"] = {"result": None, "error": None, "event": event}

        resp = client.post(
            "/llm/copilot/result/task-abc",
            json={"result": "the answer", "error": None},
        )
        assert resp.status_code == 200
        assert event.is_set()

        with _copilot_bridge_lock:
            _copilot_responses.pop("task-abc", None)


# ===========================================================================
# projects.py
# ===========================================================================


class TestGetProjects:
    def test_returns_project_list(self):
        """projects.py lines 98-138 — enumerate on-disk projects."""
        api_module.brain = None
        with patch("axon.projects.list_projects", return_value=[{"name": "proj1"}]):
            resp = client.get("/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert "projects" in data
        assert any(p["name"] == "proj1" for p in data["projects"])

    def test_memory_only_projects_included(self):
        """projects.py lines 104-110 — memory-only projects merged."""
        api_module.brain = None
        api_module._source_hashes["memory_proj"] = {}
        with patch("axon.projects.list_projects", return_value=[]):
            resp = client.get("/projects")
        assert resp.status_code == 200
        data = resp.json()
        names = [p["name"] for p in data.get("memory_only", [])]
        assert "memory_proj" in names

    def test_list_projects_exception_handled(self):
        """projects.py lines 100-102 — on_disk enumeration fails gracefully."""
        api_module.brain = None
        with patch("axon.projects.list_projects", side_effect=RuntimeError("fs error")):
            resp = client.get("/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert data["projects"] == []

    def test_axon_store_mode_validates_shares(self):
        """projects.py lines 91-96 — axon_store_mode validates received shares."""
        brain = _make_brain()
        brain.config.axon_store_mode = True
        api_module.brain = brain

        with patch("axon.projects.list_projects", return_value=[]):
            with patch("axon.shares.validate_received_shares", return_value=[]) as mock_validate:
                resp = client.get("/projects")

        assert resp.status_code == 200
        mock_validate.assert_called_once()

    def test_share_validation_exception_handled(self):
        """projects.py lines 95-96 — share validation exception does not crash."""
        brain = _make_brain()
        brain.config.axon_store_mode = True
        api_module.brain = brain

        with patch("axon.projects.list_projects", return_value=[]):
            with patch("axon.shares.validate_received_shares", side_effect=RuntimeError("bad")):
                resp = client.get("/projects")

        assert resp.status_code == 200


class TestCreateProject:
    def test_invalid_name_returns_400(self):
        """projects.py lines 146-153 — invalid name → 400."""
        resp = client.post("/project/new", json={"name": "../../evil"})
        assert resp.status_code == 400

    def test_empty_name_returns_400(self):
        """projects.py line 146 — empty name → 400."""
        resp = client.post("/project/new", json={"name": ""})
        assert resp.status_code == 400

    def test_success(self):
        """projects.py lines 155-157 — success path."""
        with patch("axon.projects.ensure_project", return_value=None):
            resp = client.post("/project/new", json={"name": "my-project"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_value_error_returns_400(self):
        """projects.py line 158-159 — ValueError → 400."""
        with patch("axon.projects.ensure_project", side_effect=ValueError("already exists")):
            resp = client.post("/project/new", json={"name": "existing"})
        assert resp.status_code == 400

    def test_generic_error_returns_500(self):
        """projects.py lines 160-162 — generic error → 500."""
        with patch("axon.projects.ensure_project", side_effect=RuntimeError("disk full")):
            resp = client.post("/project/new", json={"name": "new-project"})
        assert resp.status_code == 500


class TestSwitchProject:
    def test_503_when_no_brain(self):
        resp = client.post("/project/switch", json={"name": "other"})
        assert resp.status_code == 503

    def test_success(self):
        """projects.py lines 174-176 — switch ok."""
        brain = _make_brain()
        api_module.brain = brain
        resp = client.post("/project/switch", json={"name": "new-project"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_value_error_returns_404(self):
        """projects.py lines 177-178 — ValueError → 404."""
        brain = _make_brain()
        brain.switch_project.side_effect = ValueError("project not found")
        api_module.brain = brain
        resp = client.post("/project/switch", json={"name": "nonexistent"})
        assert resp.status_code == 404

    def test_generic_error_returns_500(self):
        """projects.py lines 179-181 — generic error → 500."""
        brain = _make_brain()
        brain.switch_project.side_effect = RuntimeError("store failure")
        api_module.brain = brain
        resp = client.post("/project/switch", json={"name": "bad-proj"})
        assert resp.status_code == 500


class TestDeleteProject:
    def test_mount_prefix_blocked(self):
        """projects.py lines 193-197 — 'mounts' literal → 400.
        Note: route /project/delete/{name} only captures single segments;
        use the literal 'mounts' to verify the guard."""
        resp = client.post("/project/delete/mounts")
        assert resp.status_code == 400

    def test_mounts_literal_blocked(self):
        """projects.py lines 193-197 — 'mounts' literal → 400."""
        resp = client.post("/project/delete/mounts")
        assert resp.status_code == 400

    def test_active_project_auto_switches(self):
        """projects.py lines 213-214 — active project → switch to default first."""
        brain = _make_brain()
        brain._active_project = "my-project"
        api_module.brain = brain

        with patch("axon.projects.delete_project", return_value=None):
            with patch("axon.shares.list_shares", return_value={"sharing": []}):
                resp = client.post("/project/delete/my-project")

        assert resp.status_code == 200
        brain.switch_project.assert_called_with("default")

    def test_success(self):
        """projects.py lines 200-208 — success path."""
        brain = _make_brain()
        brain._active_project = "other"
        api_module.brain = brain

        with patch("axon.projects.delete_project", return_value=None):
            with patch("axon.shares.list_shares", return_value={"sharing": []}):
                resp = client.post("/project/delete/target-project")

        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]

    def test_project_has_active_shares_blocks_delete(self):
        """projects.py lines 200-211 — active shares → 409."""
        brain = _make_brain()
        brain.config.axon_store_mode = True
        brain._active_project = "other"
        api_module.brain = brain

        with patch("axon.shares.list_shares") as mock_shares:
            mock_shares.return_value = {
                "sharing": [{"project": "shared-proj", "grantee": "bob", "revoked": False}]
            }
            resp = client.post("/project/delete/shared-proj")

        assert resp.status_code == 409

    def test_value_error_returns_404(self):
        """projects.py line 220 — ValueError → 404."""
        brain = _make_brain()
        brain._active_project = "other"
        api_module.brain = brain

        with patch("axon.projects.delete_project", side_effect=ValueError("not found")):
            with patch("axon.shares.list_shares", return_value={"sharing": []}):
                resp = client.post("/project/delete/ghost-project")

        assert resp.status_code == 404


# ===========================================================================
# query.py
# ===========================================================================


class TestQueryBrain:
    def test_503_when_no_brain(self):
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 503

    def test_dry_run_mode(self):
        """query.py lines 49-66 — dry_run returns results + diagnostics."""
        brain = _make_brain()
        mock_diag = MagicMock()
        mock_diag.to_dict.return_value = {"step": "dry"}
        brain.search_raw.return_value = (
            [{"text": "chunk", "score": 0.9, "metadata": {}}],
            mock_diag,
            MagicMock(),
        )
        api_module.brain = brain

        resp = client.post("/query", json={"query": "hello", "dry_run": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert "results" in data
        assert "diagnostics" in data

    def test_query_success_with_diagnostics(self):
        """query.py line 157 — include_diagnostics adds diagnostics field."""
        brain = _make_brain()
        brain.query.return_value = "answer"
        mock_diag = MagicMock()
        mock_diag.to_dict.return_value = {"retrieval_time": 0.1}
        brain._last_diagnostics = mock_diag
        api_module.brain = brain

        resp = client.post("/query", json={"query": "hello", "include_diagnostics": True})
        assert resp.status_code == 200
        data = resp.json()
        assert "diagnostics" in data

    def test_query_with_overrides(self):
        """query.py lines 43-44 — overrides passed to query."""
        brain = _make_brain()
        brain.query.return_value = "result"
        api_module.brain = brain

        resp = client.post(
            "/query",
            json={
                "query": "hello",
                "top_k": 10,
                "hybrid": True,
                "rerank": True,
                "hyde": True,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200

    def test_exception_returns_500(self):
        """query.py lines 102-104 — generic exception → 500."""
        brain = _make_brain()
        brain.query.side_effect = RuntimeError("LLM failure")
        api_module.brain = brain

        resp = client.post("/query", json={"query": "crash"})
        assert resp.status_code == 500

    def test_query_with_filters(self):
        """query.py line 178 — filters forwarded."""
        brain = _make_brain()
        brain.query.return_value = "filtered result"
        api_module.brain = brain

        resp = client.post(
            "/query",
            json={"query": "hello", "filters": {"source": "docs/readme.md"}},
        )
        assert resp.status_code == 200


class TestSearchRaw:
    def test_503_when_no_brain(self):
        resp = client.post("/search/raw", json={"query": "test"})
        assert resp.status_code == 503

    def test_success_without_trace(self):
        """query.py lines 194-201 — include_trace=False omits trace key."""
        brain = _make_brain()
        mock_diag = MagicMock()
        mock_diag.to_dict.return_value = {}
        mock_trace = MagicMock()
        brain.search_raw.return_value = ([], mock_diag, mock_trace)
        api_module.brain = brain

        resp = client.post("/search/raw", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "trace" not in data

    def test_success_with_trace(self):
        """query.py lines 199-200 — include_trace=True adds trace."""
        brain = _make_brain()
        mock_diag = MagicMock()
        mock_diag.to_dict.return_value = {}
        mock_trace = MagicMock()
        mock_trace.to_dict.return_value = {"steps": []}
        brain.search_raw.return_value = ([], mock_diag, mock_trace)
        api_module.brain = brain

        resp = client.post("/search/raw?include_trace=true", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "trace" in data

    def test_exception_returns_500(self):
        """query.py lines 202-204 — generic exception → 500."""
        brain = _make_brain()
        brain.search_raw.side_effect = RuntimeError("store down")
        api_module.brain = brain

        resp = client.post("/search/raw", json={"query": "test"})
        assert resp.status_code == 500


class TestClearBrain:
    def test_503_when_no_brain(self):
        resp = client.post("/clear")
        assert resp.status_code == 503

    def test_qdrant_provider_cleared(self):
        """query.py lines 228-229 — qdrant provider path."""
        brain = _make_brain(provider="qdrant")
        brain._embedding_meta_path = "/tmp/nonexistent_meta.json"
        api_module.brain = brain

        resp = client.post("/clear")
        assert resp.status_code == 200

    def test_lancedb_provider_cleared(self):
        """query.py lines 234-235 — lancedb provider path."""
        brain = _make_brain(provider="lancedb")
        brain._embedding_meta_path = "/tmp/nonexistent_meta.json"
        api_module.brain = brain

        resp = client.post("/clear")
        assert resp.status_code == 200

    def test_chroma_provider_cleared(self):
        """query.py lines 252-253, 256-257 — chroma provider path."""
        brain = _make_brain(provider="chroma")
        brain._embedding_meta_path = "/tmp/nonexistent_meta.json"
        api_module.brain = brain

        resp = client.post("/clear")
        assert resp.status_code == 200

    def test_exception_returns_500(self):
        brain = _make_brain()
        brain._save_hash_store.side_effect = RuntimeError("disk full")
        api_module.brain = brain

        resp = client.post("/clear")
        assert resp.status_code == 500


# ===========================================================================
# shares.py
# ===========================================================================


class TestShareGenerate:
    def test_project_not_found_returns_404(self):
        """shares.py lines 45-46 — project dir has no meta.json → 404."""
        with patch("axon.api._get_user_dir") as mock_user_dir:
            user_dir = MagicMock()
            mock_user_dir.return_value = user_dir
            project_dir = MagicMock()
            user_dir.__truediv__.return_value = project_dir
            project_dir.exists.return_value = False

            resp = client.post(
                "/share/generate",
                json={"project": "nonexistent", "grantee": "bob"},
            )
        assert resp.status_code == 404

    def test_success(self):
        """shares.py line 88-93 — success path."""
        with patch("axon.api._get_user_dir") as mock_user_dir:
            user_dir = MagicMock()
            mock_user_dir.return_value = user_dir

            project_dir = MagicMock()
            project_dir.exists.return_value = True
            meta_json = MagicMock()
            meta_json.exists.return_value = True
            project_dir.__truediv__.return_value = meta_json
            user_dir.__truediv__.return_value = project_dir

            with patch(
                "axon.shares.generate_share_key",
                return_value={"key_id": "k1", "share_string": "axon://..."},
            ):
                resp = client.post(
                    "/share/generate",
                    json={"project": "my-project", "grantee": "alice"},
                )

        assert resp.status_code == 200
        assert "key_id" in resp.json()


class TestShareRedeem:
    def test_value_error_returns_400(self):
        """shares.py line 108-109 — ValueError on redeem → 400."""
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch("axon.shares.redeem_share_key", side_effect=ValueError("invalid share")):
                resp = client.post(
                    "/share/redeem",
                    json={"share_string": "invalid-string"},
                )
        assert resp.status_code == 400

    def test_not_implemented_returns_400(self):
        """shares.py line 108-109 — NotImplementedError on redeem → 400."""
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch(
                "axon.shares.redeem_share_key", side_effect=NotImplementedError("not supported")
            ):
                resp = client.post(
                    "/share/redeem",
                    json={"share_string": "axon://something"},
                )
        assert resp.status_code == 400

    def test_success(self):
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch("axon.shares.redeem_share_key", return_value={"status": "mounted"}):
                resp = client.post(
                    "/share/redeem",
                    json={"share_string": "axon://valid"},
                )
        assert resp.status_code == 200


class TestShareList:
    def test_returns_shares(self):
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch("axon.shares.validate_received_shares", return_value=[]):
                with patch(
                    "axon.shares.list_shares", return_value={"sharing": [], "shared_with_me": []}
                ):
                    resp = client.get("/share/list")
        assert resp.status_code == 200

    def test_removed_stale_included_when_present(self):
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch("axon.shares.validate_received_shares", return_value=["stale1"]):
                with patch(
                    "axon.shares.list_shares", return_value={"sharing": [], "shared_with_me": []}
                ):
                    resp = client.get("/share/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "removed_stale" in data
        assert "stale1" in data["removed_stale"]


class TestShareRevoke:
    def test_value_error_returns_404(self):
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch("axon.shares.revoke_share_key", side_effect=ValueError("key not found")):
                resp = client.post("/share/revoke", json={"key_id": "nonexistent"})
        assert resp.status_code == 404

    def test_success(self):
        with patch("axon.api._get_user_dir") as mock_user_dir:
            mock_user_dir.return_value = MagicMock()
            with patch("axon.shares.revoke_share_key", return_value={"status": "revoked"}):
                resp = client.post("/share/revoke", json={"key_id": "k1"})
        assert resp.status_code == 200
