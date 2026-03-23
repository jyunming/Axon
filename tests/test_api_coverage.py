"""
tests/test_api_coverage.py

Targeted coverage tests for api.py endpoints not covered by test_api.py.
"""

import asyncio
import hashlib
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


def _make_brain(provider="chroma"):
    brain = MagicMock()
    brain.vector_store.provider = provider
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
    brain._apply_overrides.return_value = brain.config
    return brain


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------


def test_get_config_success():
    api_module.brain = _make_brain()
    resp = client.get("/config")
    assert resp.status_code == 200


def test_get_config_503_no_brain():
    api_module.brain = None
    resp = client.get("/config")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /config/update
# ---------------------------------------------------------------------------


def test_update_config_sets_top_k():
    brain = _make_brain()
    api_module.brain = brain
    resp = client.post("/config/update", json={"top_k": 12})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_update_config_503_no_brain():
    api_module.brain = None
    resp = client.post("/config/update", json={"top_k": 5})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /sessions + GET /session/{id}
# ---------------------------------------------------------------------------


def test_list_sessions_returns_list():
    api_module.brain = _make_brain()
    with patch("axon.sessions._list_sessions", return_value=[{"id": "s1"}]):
        resp = client.get("/sessions")
    assert resp.status_code == 200
    assert "sessions" in resp.json()


def test_get_session_found():
    api_module.brain = _make_brain()
    with patch("axon.sessions._load_session", return_value={"id": "abc"}):
        resp = client.get("/session/abc")
    assert resp.status_code == 200


def test_get_session_not_found():
    api_module.brain = _make_brain()
    with patch("axon.sessions._load_session", return_value=None):
        resp = client.get("/session/missing")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /clear — additional branches
# ---------------------------------------------------------------------------


def test_clear_503_no_brain():
    api_module.brain = None
    resp = client.post("/clear")
    assert resp.status_code == 503


def test_clear_qdrant_branch():
    brain = _make_brain(provider="qdrant")
    api_module.brain = brain
    brain.vector_store.provider = "qdrant"
    brain.vector_store.client = MagicMock()
    brain._save_hash_store = MagicMock()
    brain._save_entity_graph = MagicMock()
    brain._ingested_hashes = set()
    brain._entity_graph = {}
    brain.bm25 = None
    resp = client.post("/clear")
    assert resp.status_code == 200
    brain.vector_store.client.delete_collection.assert_called()


def test_clear_lancedb_branch():
    brain = _make_brain(provider="lancedb")
    api_module.brain = brain
    brain.vector_store.provider = "lancedb"
    brain.vector_store.client = MagicMock()
    brain._save_hash_store = MagicMock()
    brain._save_entity_graph = MagicMock()
    brain._ingested_hashes = set()
    brain._entity_graph = {}
    brain.bm25 = None
    resp = client.post("/clear")
    assert resp.status_code == 200
    brain.vector_store.client.drop_table.assert_called_with("axon")


# ---------------------------------------------------------------------------
# GET /llm/copilot/tasks + POST /llm/copilot/result/{task_id}
# ---------------------------------------------------------------------------


def test_get_copilot_tasks_drains_queue():
    from axon.main import _copilot_bridge_lock, _copilot_task_queue

    with _copilot_bridge_lock:
        _copilot_task_queue.append({"task_id": "t1", "prompt": "hi"})
    resp = client.get("/llm/copilot/tasks")
    assert resp.status_code == 200
    assert "tasks" in resp.json()


def test_submit_copilot_result_not_found():
    resp = client.post("/llm/copilot/result/nonexistent", json={"result": "ok", "error": None})
    assert resp.status_code == 404


def test_submit_copilot_result_success():
    from axon.main import _copilot_bridge_lock, _copilot_responses

    event = threading.Event()
    with _copilot_bridge_lock:
        _copilot_responses["task_cov_1"] = {"result": None, "error": None, "event": event}
    resp = client.post("/llm/copilot/result/task_cov_1", json={"result": "answer", "error": None})
    assert resp.status_code == 200
    assert event.is_set()
    with _copilot_bridge_lock:
        _copilot_responses.pop("task_cov_1", None)


# ---------------------------------------------------------------------------
# GET /tracked-docs + POST /ingest/refresh
# ---------------------------------------------------------------------------


def test_tracked_docs_success():
    brain = _make_brain()
    api_module.brain = brain
    brain.get_doc_versions.return_value = {"file.txt": {"content_hash": "abc"}}
    resp = client.get("/tracked-docs")
    assert resp.status_code == 200
    assert "docs" in resp.json()


def test_tracked_docs_503_no_brain():
    api_module.brain = None
    resp = client.get("/tracked-docs")
    assert resp.status_code == 503


def test_ingest_refresh_skipped(tmp_path):
    brain = _make_brain()
    api_module.brain = brain
    real_file = tmp_path / "doc.txt"
    real_file.write_bytes(b"hello")
    content_hash = hashlib.md5(b"hello").hexdigest()
    brain.get_doc_versions.return_value = {str(real_file): {"content_hash": content_hash}}
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 200
    assert str(real_file) in resp.json()["skipped"]


def test_ingest_refresh_reingest_needed(tmp_path):
    brain = _make_brain()
    api_module.brain = brain
    real_file = tmp_path / "changed.txt"
    real_file.write_bytes(b"new content")
    brain.get_doc_versions.return_value = {str(real_file): {"content_hash": "old_hash"}}
    # Mock the loader so we don't touch the real filesystem loader internals
    fake_doc = [{"id": "doc1", "text": "new content", "metadata": {"source": str(real_file)}}]
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = fake_doc
    with patch("axon.loaders.DirectoryLoader") as mock_loader_cls:
        mock_loader_cls.return_value.loaders = {".txt": mock_loader_instance}
        resp = client.post("/ingest/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert str(real_file) in data["reingested"]


def test_ingest_refresh_missing_file():
    brain = _make_brain()
    api_module.brain = brain
    brain.get_doc_versions.return_value = {"/nonexistent/path.txt": {"content_hash": "x"}}
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 200
    assert "/nonexistent/path.txt" in resp.json()["missing"]


def test_ingest_refresh_503_no_brain():
    api_module.brain = None
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /collection
# ---------------------------------------------------------------------------


def test_get_collection_success():
    brain = _make_brain()
    api_module.brain = brain
    brain.list_documents.return_value = [
        {"source": "a.txt", "chunks": 3},
        {"source": "b.txt", "chunks": 2},
    ]
    resp = client.get("/collection")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_files"] == 2
    assert data["total_chunks"] == 5


def test_get_collection_503_no_brain():
    api_module.brain = None
    resp = client.get("/collection")
    assert resp.status_code == 503


def test_get_collection_error():
    brain = _make_brain()
    api_module.brain = brain
    brain.list_documents.side_effect = RuntimeError("db error")
    resp = client.get("/collection")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /project/switch
# ---------------------------------------------------------------------------


def test_switch_project_success():
    brain = _make_brain()
    api_module.brain = brain
    resp = client.post("/project/switch", json={"name": "research"})
    assert resp.status_code == 200


def test_switch_project_not_found():
    brain = _make_brain()
    api_module.brain = brain
    brain.switch_project.side_effect = ValueError("not found")
    resp = client.post("/project/switch", json={"name": "ghost"})
    assert resp.status_code == 404


def test_switch_project_error():
    brain = _make_brain()
    api_module.brain = brain
    brain.switch_project.side_effect = RuntimeError("crash")
    resp = client.post("/project/switch", json={"name": "bad"})
    assert resp.status_code == 500


def test_switch_project_503_no_brain():
    api_module.brain = None
    resp = client.post("/project/switch", json={"name": "x"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /project/delete/{name}
# ---------------------------------------------------------------------------


def test_delete_project_mounts_blocked():
    """mounts/ prefix is blocked — mounted share entries are read-only."""
    api_module.brain = _make_brain()
    resp = client.post("/project/delete/mounts")
    assert resp.status_code == 400


def test_delete_project_success():
    api_module.brain = None
    with patch("axon.projects.delete_project"):
        resp = client.post("/project/delete/myproject")
    assert resp.status_code == 200


def test_delete_project_not_found():
    api_module.brain = None
    with patch("axon.projects.delete_project", side_effect=ValueError("not found")):
        resp = client.post("/project/delete/ghost")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /add_texts
# ---------------------------------------------------------------------------


def test_add_texts_batch_creates_items():
    brain = _make_brain()
    api_module.brain = brain
    resp = client.post(
        "/add_texts",
        json={"docs": [{"text": "unique text alpha 1234"}, {"text": "unique text beta 5678"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert any(d["status"] == "created" for d in data)


def test_add_texts_ingest_error():
    brain = _make_brain()
    api_module.brain = brain
    brain.ingest.side_effect = RuntimeError("ingest boom")
    resp = client.post("/add_texts", json={"docs": [{"text": "unique text gamma 9999"}]})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /ingest_url — additional branches
# ---------------------------------------------------------------------------


def test_ingest_url_no_content():
    brain = _make_brain()
    api_module.brain = brain
    with patch("axon.loaders.URLLoader") as MockLoader:
        MockLoader.return_value.load.return_value = []
        resp = client.post("/ingest_url", json={"url": "http://example.com/empty"})
    assert resp.status_code == 422


def test_ingest_url_unexpected_error():
    brain = _make_brain()
    api_module.brain = brain
    with patch("axon.loaders.URLLoader") as MockLoader:
        MockLoader.return_value.load.side_effect = RuntimeError("network crash")
        resp = client.post("/ingest_url", json={"url": "http://example.com"})
    assert resp.status_code == 500


def test_ingest_url_ingest_error():
    brain = _make_brain()
    api_module.brain = brain
    brain.ingest.side_effect = RuntimeError("db crash")
    with patch("axon.loaders.URLLoader") as MockLoader:
        MockLoader.return_value.load.return_value = [
            {"id": "u1", "text": "unique url text cov 77777", "metadata": {"source": "u1"}}
        ]
        resp = client.post("/ingest_url", json={"url": "http://example.com/doc"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /project/new — error branches
# ---------------------------------------------------------------------------


def test_create_project_ensure_project_error():
    api_module.brain = _make_brain()
    with patch("axon.projects.ensure_project", side_effect=RuntimeError("disk full")):
        resp = client.post("/project/new", json={"name": "myproject"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /query — timeout + error
# ---------------------------------------------------------------------------


def test_query_timeout_returns_504():
    brain = _make_brain()
    api_module.brain = brain
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
        resp = client.post("/query", json={"query": "hello", "timeout": 0.01})
    assert resp.status_code == 504


def test_query_generic_error_returns_500():
    brain = _make_brain()
    api_module.brain = brain
    brain.query.side_effect = RuntimeError("model crash")
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /collection/stale
# ---------------------------------------------------------------------------


def test_stale_docs_returns_old_entry():
    api_module._source_hashes.clear()
    old_ts = datetime.fromtimestamp(0, tz=timezone.utc).isoformat()
    api_module._source_hashes["myproject"] = {"abc": {"doc_id": "doc1", "last_ingested_at": old_ts}}
    resp = client.get("/collection/stale?days=1")
    assert resp.status_code == 200
    assert any(d["doc_id"] == "doc1" for d in resp.json()["stale_docs"])
    api_module._source_hashes.clear()


def test_stale_docs_malformed_timestamp_skipped():
    api_module._source_hashes.clear()
    api_module._source_hashes["p1"] = {"h1": {"doc_id": "d1"}}  # no last_ingested_at
    resp = client.get("/collection/stale?days=1")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    api_module._source_hashes.clear()


# ---------------------------------------------------------------------------
# Dedup namespace regression: project fallback from _global to active project
# ---------------------------------------------------------------------------


def test_add_text_dedup_uses_active_project_not_global():
    """When project field is omitted, dedup key should use brain._active_project
    not the literal string '_global', preventing cross-project dedup skips.
    """
    brain = _make_brain()
    brain._active_project = "work"
    api_module.brain = brain

    import hashlib

    import axon.api as api_mod

    # Prime the dedup store with a hash under "work" namespace
    text = "unique dedup regression text xyz123"
    h = hashlib.sha256(text.strip().encode()).hexdigest()
    api_mod._source_hashes.setdefault("work", {})[h] = {
        "doc_id": "existing_doc",
        "ingested_at": "2026-01-01T00:00:00Z",
    }

    resp = client.post("/add_text", json={"text": text})
    # Should be detected as duplicate (skipped), not ingested again
    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"

    # Cleanup
    api_mod._source_hashes.pop("work", None)
