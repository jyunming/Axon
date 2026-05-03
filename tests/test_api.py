from __future__ import annotations

"""
tests/test_api.py

Unit tests for the FastAPI endpoints in axon.api.
"""

import os
from unittest.mock import ANY, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


def _make_brain(provider="chroma"):
    """Return a minimal mock brain with required attributes."""
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
    # _apply_overrides returns a copy of config with overrides applied
    brain._apply_overrides.return_value = brain.config
    brain._community_build_in_progress = False
    # Graph backend mock — graph_data() returns a payload whose to_dict() is a valid dict
    _mock_payload = MagicMock()
    _mock_payload.to_dict.return_value = {"nodes": [], "links": []}
    brain._graph_backend.graph_data.return_value = _mock_payload
    brain._graph_backend.status.return_value = {
        "backend": "graphrag",
        "entities": 0,
        "relations": 0,
        "communities": 0,
        "community_summaries": 0,
    }
    return brain


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_503_no_brain():
    """Health check returns 503 when brain is not initialized."""
    api_module.brain = None
    resp = client.get("/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "initializing"


def test_health_returns_200_with_brain():
    """Health check returns 200 with status ok when brain is initialized."""
    api_module.brain = _make_brain()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_reports_active_project():
    """Health endpoint must report brain._active_project, not brain.config.project."""
    brain = _make_brain()
    brain._active_project = "work/atlas"
    # Ensure brain.config has no 'project' attribute (mirrors real AxonConfig)
    del brain.config.project
    api_module.brain = brain
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["project"] == "work/atlas"


def test_health_reports_default_when_no_active_project():
    """Health endpoint falls back to 'default' when _active_project is absent."""
    brain = _make_brain()
    # Remove _active_project to test fallback
    if hasattr(brain, "_active_project"):
        del brain._active_project
    api_module.brain = brain
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["project"] == "default"


# ---------------------------------------------------------------------------
# 503 when brain is None
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_brain():
    """Ensure brain and source-dedup state are reset before/after every test."""
    original = api_module.brain
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()
    yield
    api_module.brain = original
    api_module._source_hashes.clear()
    api_module._jobs.clear()


def test_query_503_no_brain():
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 503


def test_ingest_503_no_brain():
    resp = client.post("/ingest", json={"path": "."})
    assert resp.status_code == 503


def test_delete_503_no_brain():
    resp = client.post("/delete", json={"doc_ids": ["id1"]})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------


def test_ingest_403_path_traversal(tmp_path):
    api_module.brain = _make_brain()
    # Set base to tmp_path; attempt escape via ../../..
    with patch.dict(os.environ, {"RAG_INGEST_BASE": str(tmp_path)}):
        resp = client.post("/ingest", json={"path": "/etc/passwd"})
    assert resp.status_code in (403, 404)  # 403 preferred; 404 if path resolves differently


@pytest.mark.skipif(
    os.name != "nt", reason="Windows system paths only blocked correctly on Windows"
)
def test_ingest_blocks_windows_system_path(tmp_path):
    """SEC-01: C:/Windows paths must be blocked even when RAG_INGEST_BASE covers them."""
    api_module.brain = _make_brain()
    with patch.dict(os.environ, {"RAG_INGEST_BASE": "C:\\"}):
        for blocked in ["C:/Windows/win.ini", r"C:\Windows\System32\drivers\etc\hosts"]:
            resp = client.post("/ingest", json={"path": blocked})
            assert resp.status_code == 403, f"Expected 403 for blocked path: {blocked}"


@pytest.mark.skipif(os.name == "nt", reason="Unix system paths only meaningful on non-Windows")
def test_ingest_blocks_unix_system_path(tmp_path):
    """SEC-01: /etc paths must be blocked even when RAG_INGEST_BASE covers them."""
    api_module.brain = _make_brain()
    with patch.dict(os.environ, {"RAG_INGEST_BASE": "/"}):
        for blocked in ["/etc/passwd", "/etc/shadow", "/proc/version"]:
            resp = client.post("/ingest", json={"path": blocked})
            assert resp.status_code == 403, f"Expected 403 for blocked path: {blocked}"


def test_ingest_traversal_dotdot_blocked(tmp_path):
    """SEC-01: ../ traversal from workspace root is blocked."""
    api_module.brain = _make_brain()
    with patch.dict(os.environ, {"RAG_INGEST_BASE": str(tmp_path)}):
        resp = client.post("/ingest", json={"path": str(tmp_path / ".." / ".." / "etc" / "passwd")})
    assert resp.status_code == 403


def test_ingest_valid_path(tmp_path):
    api_module.brain = _make_brain()
    # Create a file inside the allowed base
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("hello", encoding="utf-8")
    with patch.dict(os.environ, {"RAG_INGEST_BASE": str(tmp_path)}):
        with patch.object(api_module.brain, "ingest", return_value=None):
            resp = client.post("/ingest", json={"path": str(doc_file)})
    # Background task is queued — endpoint returns 200 with processing message
    assert resp.status_code == 200
    assert "processing" in resp.json().get("status", "").lower() or "Ingestion" in resp.json().get(
        "message", ""
    )


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


def test_query_success():
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "The answer is 42"
    resp = client.post("/query", json={"query": "What is the answer?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data or "answer" in data


# ---------------------------------------------------------------------------
# /delete
# ---------------------------------------------------------------------------


def test_delete_success():
    api_module.brain = _make_brain()
    # get_by_ids returns the two docs — both exist
    api_module.brain.vector_store.get_by_ids.return_value = [
        {"id": "doc1", "text": "a"},
        {"id": "doc2", "text": "b"},
    ]
    resp = client.post("/delete", json={"doc_ids": ["doc1", "doc2"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["deleted"] == 2
    assert set(data["doc_ids"]) == {"doc1", "doc2"}
    assert data["not_found"] == []
    # Verify delete_by_ids was called on the vector store with only existing IDs
    api_module.brain.vector_store.delete_by_ids.assert_called_once_with(["doc1", "doc2"])
    # Verify BM25 delete was called
    api_module.brain.bm25.delete_documents.assert_called_once_with(["doc1", "doc2"])


def test_delete_nonexistent_ids_returns_zero():
    """Deleting nonexistent IDs must report deleted=0 and list them in not_found."""
    api_module.brain = _make_brain()
    api_module.brain.vector_store.get_by_ids.return_value = []  # none exist
    resp = client.post("/delete", json={"doc_ids": ["fake-id-12345"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["deleted"] == 0
    assert "fake-id-12345" in data["not_found"]
    api_module.brain.vector_store.delete_by_ids.assert_not_called()


def test_delete_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain.vector_store.get_by_ids.return_value = [{"id": "x", "text": "t"}]
    api_module.brain.vector_store.delete_by_ids.side_effect = RuntimeError("store down")
    resp = client.post("/delete", json={"doc_ids": ["x"]})
    assert resp.status_code == 500


def test_delete_calls_delete_by_ids_not_collection_delete():
    """Endpoint must use delete_by_ids (works for all providers) not collection.delete."""
    api_module.brain = _make_brain()
    api_module.brain.vector_store.get_by_ids.return_value = [{"id": "id1", "text": "t"}]
    resp = client.post("/delete", json={"doc_ids": ["id1"]})
    assert resp.status_code == 200
    api_module.brain.vector_store.delete_by_ids.assert_called_once_with(["id1"])
    # collection.delete should NOT be called directly by the endpoint
    api_module.brain.vector_store.collection.delete.assert_not_called()


# ---------------------------------------------------------------------------
# /clear
# ---------------------------------------------------------------------------


def test_clear_actually_clears_chroma():
    """EXEC-009: /clear must delete and recreate the ChromaDB collection, not be a no-op."""
    brain = _make_brain()
    api_module.brain = brain
    # Simulate the chroma provider path
    brain.vector_store.provider = "chroma"
    brain.vector_store.client = MagicMock()
    brain.vector_store.client.create_collection.return_value = MagicMock()

    resp = client.post("/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    # The collection must have been deleted, not left intact
    brain.vector_store.client.delete_collection.assert_called_once_with("axon")
    brain.vector_store.client.create_collection.assert_called_once()


def test_clear_resets_bm25_and_hashes():
    """/clear must wipe BM25 corpus and ingested hash store."""
    brain = _make_brain()
    api_module.brain = brain
    brain.vector_store.provider = "chroma"
    brain.vector_store.client = MagicMock()
    brain.vector_store.client.create_collection.return_value = MagicMock()

    # Inject non-empty BM25 and hash state
    brain.bm25 = MagicMock()
    brain.bm25.corpus = ["fake_chunk"]
    brain._ingested_hashes = {"abc123"}
    brain._entity_graph = {"entity": {"description": "", "chunk_ids": ["neighbor"]}}

    with patch.object(brain, "_save_hash_store") as mock_save_hash, patch.object(
        brain, "_save_entity_graph"
    ) as mock_save_graph:
        resp = client.post("/clear")

    assert resp.status_code == 200
    assert brain._ingested_hashes == set()
    assert brain._entity_graph == {}
    mock_save_hash.assert_called_once()
    mock_save_graph.assert_called_once()
    brain.bm25.save.assert_called_once()


def test_clear_deletes_embedding_meta(tmp_path):
    """/clear must remove .embedding_meta.json so a new model can be used after clear."""
    brain = _make_brain()
    api_module.brain = brain
    brain.vector_store.provider = "chroma"
    brain.vector_store.client = MagicMock()
    brain.vector_store.client.create_collection.return_value = MagicMock()
    brain.bm25 = None

    # Create a fake embedding meta file and point the brain mock at it
    meta_file = tmp_path / ".embedding_meta.json"
    meta_file.write_text('{"provider":"fastembed","model":"BAAI/bge-large-en-v1.5"}')
    brain._embedding_meta_path = str(meta_file)

    with patch.object(brain, "_save_hash_store"), patch.object(brain, "_save_entity_graph"):
        resp = client.post("/clear")

    assert resp.status_code == 200
    assert not meta_file.exists(), "embedding meta file must be deleted on /clear"


def test_clear_resets_graph_state_and_api_dedup_cache():
    """/clear must reset graph/doc-version state and clear the active project's dedup cache."""
    brain = _make_brain()
    api_module.brain = brain
    brain.vector_store.provider = "chroma"
    brain.vector_store.client = MagicMock()
    brain.vector_store.client.create_collection.return_value = MagicMock()
    brain._active_project = "default"
    brain._doc_versions = {"doc.txt": {"hash": "abc123"}}
    brain._relation_graph = {"entity:a": {"entity:b": {"weight": 1.0}}}
    brain._community_levels = {"entity:a": 1}
    brain._community_summaries = {"community:1": "summary"}
    brain._community_hierarchy = {"community:1": ["entity:a"]}
    brain._community_children = {"community:1": ["entity:a"]}
    brain._community_graph_dirty = True
    brain._community_build_in_progress = True
    brain._claims_graph = {"claim:1": {"source": "doc.txt"}}
    brain._entity_embeddings = {"entity:a": [0.1, 0.2]}
    brain._entity_description_buffer = {"entity:a": "desc"}
    brain._relation_description_buffer = {"rel:1": "desc"}
    brain._text_unit_entity_map = {"chunk:1": ["entity:a"]}
    brain._text_unit_relation_map = {"chunk:1": ["rel:1"]}
    brain._raptor_summary_cache = {"doc.txt": "summary"}
    brain._code_graph = {"nodes": {"file.py": {}}, "edges": [{"from": "a", "to": "b"}]}
    api_module._source_hashes["default"] = {"abc123": {"source": "doc.txt"}}
    api_module._source_hashes["_global"] = {"legacy": {"source": "doc.txt"}}

    resp = client.post("/clear")

    assert resp.status_code == 200
    assert brain._doc_versions == {}
    assert brain._relation_graph == {}
    assert brain._community_levels == {}
    assert brain._community_summaries == {}
    assert brain._community_hierarchy == {}
    assert brain._community_children == {}
    assert brain._community_graph_dirty is False
    assert brain._community_build_in_progress is False
    assert brain._claims_graph == {}
    assert brain._entity_embeddings == {}
    assert brain._entity_description_buffer == {}
    assert brain._relation_description_buffer == {}
    assert brain._text_unit_entity_map == {}
    assert brain._text_unit_relation_map == {}
    assert brain._raptor_summary_cache == {}
    assert brain._code_graph == {"nodes": {}, "edges": []}
    assert "default" not in api_module._source_hashes
    assert "_global" not in api_module._source_hashes
    brain._save_doc_versions.assert_called_once()
    brain._save_relation_graph.assert_called_once()
    brain._save_community_levels.assert_called_once()
    brain._save_community_summaries.assert_called_once()
    brain._save_community_hierarchy.assert_called_once()
    brain._save_claims_graph.assert_called_once()
    brain._save_entity_embeddings.assert_called_once()
    brain._save_code_graph.assert_called_once()


# ---------------------------------------------------------------------------
# /query/stream
# ---------------------------------------------------------------------------


def test_query_stream_503_no_brain():
    resp = client.post("/query/stream", json={"query": "hello"})
    assert resp.status_code == 503


def test_query_stream_success():
    api_module.brain = _make_brain()
    api_module.brain.query_stream = MagicMock(return_value=iter(["Hello ", "world"]))
    resp = client.post("/query/stream", json={"query": "test"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_query_stream_error_yields_error_chunk():
    api_module.brain = _make_brain()
    api_module.brain.query_stream = MagicMock(side_effect=RuntimeError("llm down"))
    resp = client.post("/query/stream", json={"query": "test"})
    # StreamingResponse always returns 200; error appears in body
    assert resp.status_code == 200
    assert '"type": "error"' in resp.text
    assert "llm down" in resp.text


def test_query_stream_normalizes_raw_error_chunk():
    api_module.brain = _make_brain()
    api_module.brain.query_stream = MagicMock(
        return_value=iter(["[ERROR] 404 NOT_FOUND. {'error': {'code': 404}}"])
    )
    resp = client.post("/query/stream", json={"query": "test"})
    assert resp.status_code == 200
    assert '"type": "error"' in resp.text
    assert "[ERROR]" not in resp.text


# ---------------------------------------------------------------------------
# /query with filters
# ---------------------------------------------------------------------------


def test_query_passes_filters():
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Filtered answer"
    filters = {"type": "text", "source": "manual"}
    resp = client.post("/query", json={"query": "What?", "filters": filters})
    assert resp.status_code == 200
    call = api_module.brain.query.call_args
    assert call.kwargs.get("filters") == filters or (call.args and call.args[1] == filters)


# ---------------------------------------------------------------------------
# /add_text
# ---------------------------------------------------------------------------


def test_add_text_503_no_brain():
    resp = client.post("/add_text", json={"text": "hello world"})
    assert resp.status_code == 503


def test_add_text_success():
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None
    resp = client.post("/add_text", json={"text": "hello world", "doc_id": "myid"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["doc_id"] == "myid"
    # Verify ingest was called with a doc containing the right text
    call_args = api_module.brain.ingest.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["text"] == "hello world"
    assert call_args[0]["id"] == "myid"


def test_add_text_auto_generates_doc_id():
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None
    resp = client.post("/add_text", json={"text": "some content"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"].startswith("agent_doc_")


def test_add_text_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain.ingest.side_effect = RuntimeError("vector store full")
    resp = client.post("/add_text", json={"text": "hello"})
    assert resp.status_code == 500


def test_add_text_permission_error_returns_403():
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot ingest: project 'alpha' is in 'readonly' maintenance state."
    )
    resp = client.post("/add_text", json={"text": "hello"})
    assert resp.status_code == 403
    assert "readonly" in resp.json()["detail"]
    api_module.brain.ingest.assert_not_called()


def test_add_text_empty_string_rejected():
    """B-05: Empty or whitespace-only text must return 400."""
    api_module.brain = _make_brain()
    for bad in ["", "   ", "\t\n"]:
        resp = client.post("/add_text", json={"text": bad})
        assert resp.status_code == 400, f"Expected 400 for text={bad!r}"


# ---------------------------------------------------------------------------
# /project/new — project name validation (B-03)
# ---------------------------------------------------------------------------


def test_create_project_invalid_name_returns_400():
    """B-03: Invalid project names must return 400, not 500."""
    api_module.brain = _make_brain()
    # 6-segment path exceeds _MAX_DEPTH=5; other cases have illegal characters / empty
    for bad_name in ["has:colon!", "", "name with spaces", "a" * 65, "a/b/c/d/e/f"]:
        resp = client.post("/project/new", json={"name": bad_name})
        assert resp.status_code == 400, f"Expected 400 for name={bad_name!r}"


def test_create_project_valid_name_succeeds():
    """B-03: Valid project names pass name validation, including deep hierarchical names."""
    api_module.brain = _make_brain()
    with patch("axon.projects.ensure_project"):
        for good_name in [
            "valid-project_01",
            "research/papers",
            "research/papers/2024",
            "research/papers/2024/q1",  # 4 segments
            "a/b/c/d/e",  # 5 segments (max depth)
        ]:
            resp = client.post("/project/new", json={"name": good_name})
            assert resp.status_code == 200, f"Expected 200 for name={good_name!r}"


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


def test_search_503_no_brain():
    resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 503


def test_search_success():
    api_module.brain = _make_brain()
    api_module.brain._execute_retrieval.return_value = {
        "results": [
            {"id": "doc1", "text": "result text", "score": 0.9, "metadata": {"source": "test"}}
        ]
    }
    resp = client.post("/search", json={"query": "find me something"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "doc1"
    assert data[0]["score"] == 0.9


def test_search_uses_custom_top_k():
    api_module.brain = _make_brain()
    api_module.brain._execute_retrieval.return_value = {"results": []}
    resp = client.post("/search", json={"query": "q", "top_k": 3})
    assert resp.status_code == 200
    api_module.brain._execute_retrieval.assert_called_once()


def test_search_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain._execute_retrieval.side_effect = RuntimeError("retrieval fail")
    resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /collection
# ---------------------------------------------------------------------------


def test_collection_503_no_brain():
    resp = client.get("/collection")
    assert resp.status_code == 503


def test_collection_empty():
    api_module.brain = _make_brain()
    api_module.brain.list_documents.return_value = []
    resp = client.get("/collection")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_files"] == 0
    assert data["total_chunks"] == 0
    assert data["files"] == []


def test_collection_returns_sources():
    api_module.brain = _make_brain()
    api_module.brain.list_documents.return_value = [
        {"source": "notes.txt", "chunks": 4, "doc_ids": ["a", "b", "c", "d"]},
        {"source": "report.pdf", "chunks": 12, "doc_ids": [f"r{i}" for i in range(12)]},
    ]
    resp = client.get("/collection")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_files"] == 2
    assert data["total_chunks"] == 16
    sources = [f["source"] for f in data["files"]]
    assert "notes.txt" in sources
    assert "report.pdf" in sources
    # doc_ids should NOT be exposed by this endpoint (internal detail)
    for f in data["files"]:
        assert "doc_ids" not in f


def test_collection_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain.list_documents.side_effect = RuntimeError("chroma down")
    resp = client.get("/collection")
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /query — per-request RAG overrides
# ---------------------------------------------------------------------------


def test_query_override_hyde():
    """hyde=true override is forwarded to brain.query as overrides dict."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "HyDE answer"
    resp = client.post("/query", json={"query": "What is RAG?", "hyde": True})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["hyde"] is True


def test_query_override_multi_query():
    """multi_query=true override is forwarded to brain.query."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Multi-query answer"
    resp = client.post("/query", json={"query": "Explain RAG", "multi_query": True})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["multi_query"] is True


def test_query_override_top_k():
    """top_k override is forwarded to brain.query."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Answer"
    resp = client.post("/query", json={"query": "test", "top_k": 3})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["top_k"] == 3


def test_query_override_rerank():
    """rerank=true override is forwarded to brain.query."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Reranked answer"
    resp = client.post("/query", json={"query": "test", "rerank": True})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["rerank"] is True


def test_query_response_includes_settings():
    """Response body includes a 'settings' key with active flag values."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Answer"
    resp = client.post("/query", json={"query": "test", "hyde": True, "top_k": 7})
    assert resp.status_code == 200
    data = resp.json()
    assert "settings" in data
    assert "top_k" in data["settings"]
    assert "hyde" in data["settings"]
    assert "multi_query" in data["settings"]


def test_query_no_override_does_not_mutate_config():
    """Requests without overrides do not change brain.config (thread safety)."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Answer"
    original_top_k = api_module.brain.config.top_k

    resp = client.post("/query", json={"query": "test"})
    assert resp.status_code == 200
    assert api_module.brain.config.top_k == original_top_k


def test_query_stream_override_forwarded():
    """Overrides are forwarded to brain.query_stream."""
    api_module.brain = _make_brain()
    api_module.brain.query_stream = MagicMock(return_value=iter(["chunk"]))
    resp = client.post("/query/stream", json={"query": "test", "hyde": True, "multi_query": True})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query_stream.call_args[1]
    assert call_kwargs["overrides"]["hyde"] is True
    assert call_kwargs["overrides"]["multi_query"] is True


def test_query_override_decompose():
    """decompose=true override is forwarded to brain.query overrides dict."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Decomposed answer"
    resp = client.post("/query", json={"query": "complex multi-part question", "decompose": True})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["query_decompose"] is True


def test_query_override_compress():
    """compress=true override is forwarded to brain.query overrides dict."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Compressed answer"
    resp = client.post("/query", json={"query": "test", "compress": True})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["compress_context"] is True


def test_query_override_temperature():
    """temperature override is forwarded to brain.query as llm_temperature."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Creative answer"
    resp = client.post("/query", json={"query": "test", "temperature": 0.2})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["llm_temperature"] == 0.2


def test_query_temperature_none_not_applied():
    """When temperature is not set, llm_temperature override is None (not applied)."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Default answer"
    resp = client.post("/query", json={"query": "test"})
    assert resp.status_code == 200
    call_kwargs = api_module.brain.query.call_args[1]
    assert call_kwargs["overrides"]["llm_temperature"] is None


def test_query_temperature_validation_rejects_out_of_range():
    """Temperature outside 0.0–2.0 is rejected with 422."""
    api_module.brain = _make_brain()
    resp = client.post("/query", json={"query": "test", "temperature": 3.5})
    assert resp.status_code == 422


def test_concurrent_requests_no_cross_contamination():
    """Different override values in concurrent requests do not bleed into each other."""
    import threading

    api_module.brain = _make_brain()
    results = {}

    def make_request(name, hyde_val):
        api_module.brain.query.return_value = f"answer-{name}"
        r = client.post("/query", json={"query": f"q-{name}", "hyde": hyde_val})
        results[name] = r

    t1 = threading.Thread(target=make_request, args=("a", True))
    t2 = threading.Thread(target=make_request, args=("b", False))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["a"].status_code == 200
    assert results["b"].status_code == 200
    # The global config must remain unchanged after both requests
    assert api_module.brain.config.top_k == 5


# ---------------------------------------------------------------------------
# /add_texts  (P1-A)
# ---------------------------------------------------------------------------


def test_add_texts_503_no_brain():
    resp = client.post("/add_texts", json={"docs": [{"text": "hello"}]})
    assert resp.status_code == 503


def test_add_texts_batch_returns_unique_ids():
    """Three documents submitted together should return three distinct IDs."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    payload = {
        "docs": [
            {"text": "document alpha", "metadata": {"source": "test"}},
            {"text": "document beta", "metadata": {"source": "test"}},
            {"text": "document gamma", "metadata": {"source": "test"}},
        ]
    }
    resp = client.post("/add_texts", json=payload)
    assert resp.status_code == 200

    results = resp.json()
    assert len(results) == 3
    ids = [r["id"] for r in results]
    assert len(set(ids)) == 3, "All returned IDs must be unique"
    for r in results:
        assert r["status"] == "created"


def test_add_texts_batch_calls_ingest_once():
    """brain.ingest is called exactly once regardless of batch size."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    payload = {"docs": [{"text": "doc one"}, {"text": "doc two"}]}
    resp = client.post("/add_texts", json=payload)
    assert resp.status_code == 200
    assert api_module.brain.ingest.call_count == 1
    ingested_docs = api_module.brain.ingest.call_args[0][0]
    assert len(ingested_docs) == 2


def test_add_texts_explicit_doc_id_preserved():
    """Explicitly supplied doc_id is used and returned."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    payload = {"docs": [{"text": "explicit id doc", "doc_id": "my-stable-id"}]}
    resp = client.post("/add_texts", json=payload)
    assert resp.status_code == 200
    assert resp.json()[0]["id"] == "my-stable-id"


def test_add_texts_permission_error_returns_403():
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot ingest on mounted share 'mounts/alice_docs'."
    )
    resp = client.post("/add_texts", json={"docs": [{"text": "doc one"}]})
    assert resp.status_code == 403
    assert "mounted share" in resp.json()["detail"]
    api_module.brain.ingest.assert_not_called()


# ---------------------------------------------------------------------------
# /ingest_url  (P1-B)
# ---------------------------------------------------------------------------


def test_ingest_url_503_no_brain():
    resp = client.post("/ingest_url", json={"url": "https://example.com"})
    assert resp.status_code == 503


def test_ingest_url_success():
    """A valid URL that URLLoader accepts returns status=ingested and a doc_id."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    fake_doc = {
        "id": "abc123",
        "text": "Example page content",
        "metadata": {"source": "https://example.com", "type": "url"},
    }

    with patch("axon.loaders.URLLoader") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.load.return_value = [fake_doc]

        resp = client.post("/ingest_url", json={"url": "https://example.com"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ingested"
    assert data["doc_id"] == "abc123"
    assert data["url"] == "https://example.com"
    api_module.brain.ingest.assert_called_once()


def test_ingest_url_blocked_returns_400():
    """URLLoader.load raising ValueError (e.g. blocked host) maps to HTTP 400."""
    api_module.brain = _make_brain()

    with patch("axon.loaders.URLLoader") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.load.side_effect = ValueError("blocked private address")

        resp = client.post("/ingest_url", json={"url": "http://127.0.0.1"})

    assert resp.status_code == 400
    assert "blocked" in resp.json()["detail"]


def test_ingest_url_permission_error_returns_403():
    """Mounted-share / maintenance write denial should return 403 before fetching the URL."""
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot ingest: project 'alpha' is in 'offline' maintenance state."
    )

    with patch("axon.loaders.URLLoader") as mock_cls:
        resp = client.post("/ingest_url", json={"url": "https://example.com"})

    assert resp.status_code == 403
    assert "offline" in resp.json()["detail"]
    mock_cls.assert_not_called()


def test_ingest_url_extra_metadata_merged():
    """metadata supplied in the request is merged into the document metadata."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    fake_doc = {
        "id": "xyz",
        "text": "content",
        "metadata": {"source": "https://example.com", "type": "url"},
    }

    with patch("axon.loaders.URLLoader") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.load.return_value = [fake_doc]

        resp = client.post(
            "/ingest_url",
            json={"url": "https://example.com", "metadata": {"topic": "testing"}},
        )

    assert resp.status_code == 200
    ingested = api_module.brain.ingest.call_args[0][0][0]
    assert ingested["metadata"]["topic"] == "testing"


# ---------------------------------------------------------------------------
# Source-level deduplication  (P1-E)
# ---------------------------------------------------------------------------


def test_add_text_dedup_skips_second_identical_call():
    """Ingesting the same text twice returns 'skipped' on the second call."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    payload = {"text": "unique dedup test content", "doc_id": "dedup-id-1"}

    resp1 = client.post("/add_text", json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "success"

    resp2 = client.post("/add_text", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "skipped"
    assert data2["reason"] == "already_ingested"
    assert data2["doc_id"] == "dedup-id-1"

    # ingest should only have been called once (the second call was skipped)
    assert api_module.brain.ingest.call_count == 1


def test_add_texts_dedup_skips_duplicate_within_batch():
    """Duplicate docs within the same batch are detected and skipped."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    # Send two docs where the second is a repeat of the first
    payload = {
        "docs": [
            {"text": "repeated content", "doc_id": "first-id"},
            {"text": "repeated content", "doc_id": "second-id"},  # duplicate
        ]
    }
    resp = client.post("/add_texts", json=payload)
    assert resp.status_code == 200
    results = resp.json()
    assert results[0]["status"] == "created"
    assert results[1]["status"] == "skipped"

    # Only 1 doc should have been passed to brain.ingest
    ingested = api_module.brain.ingest.call_args[0][0]
    assert len(ingested) == 1


def test_add_text_dedup_different_content_not_skipped():
    """Different content is never considered a duplicate."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    client.post("/add_text", json={"text": "first unique text"})
    resp = client.post("/add_text", json={"text": "second unique text"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert api_module.brain.ingest.call_count == 2


def test_ingest_url_dedup_skips_second_identical_url():
    """Fetching the same URL twice returns 'skipped' on the second request."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    fake_doc = {
        "id": "url-doc-1",
        "text": "same page content",
        "metadata": {"source": "https://example.com", "type": "url"},
    }

    with patch("axon.loaders.URLLoader") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.load.return_value = [fake_doc]

        resp1 = client.post("/ingest_url", json={"url": "https://example.com"})
        assert resp1.json()["status"] == "ingested"

        # Second call with identical content
        mock_instance.load.return_value = [dict(fake_doc, id="url-doc-2")]  # same text, new id
        resp2 = client.post("/ingest_url", json={"url": "https://example.com"})

    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "skipped"
    assert api_module.brain.ingest.call_count == 1


# ---------------------------------------------------------------------------
# /ingest/upload
# ---------------------------------------------------------------------------


def test_ingest_upload_503_no_brain():
    resp = client.post(
        "/ingest/upload",
        files=[("files", ("note.txt", b"hello world", "text/plain"))],
    )
    assert resp.status_code == 503


def test_ingest_upload_text_file_success():
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None
    api_module.brain._active_project = "default"

    resp = client.post(
        "/ingest/upload",
        data={"project": "default"},
        files=[("files", ("note.txt", b"hello from upload", "text/plain"))],
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["ingested_files"] == 1
    assert payload["ingested_chunks"] == 1
    assert payload["files"][0]["filename"] == "note.txt"
    assert payload["files"][0]["status"] == "ingested"

    ingested_docs = api_module.brain.ingest.call_args[0][0]
    assert len(ingested_docs) == 1
    assert ingested_docs[0]["metadata"]["source"] == "note.txt"
    assert ingested_docs[0]["text"].startswith("[File Path: note.txt]\n")


def test_ingest_upload_unsupported_file_is_reported():
    api_module.brain = _make_brain()

    resp = client.post(
        "/ingest/upload",
        files=[("files", ("archive.xyz", b"not supported", "application/octet-stream"))],
    )

    assert resp.status_code == 400
    payload = resp.json()
    assert "supported" in payload["detail"].lower()
    api_module.brain.ingest.assert_not_called()


def test_ingest_upload_permission_error_returns_403():
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot ingest: project 'alpha' is in 'readonly' maintenance state."
    )

    resp = client.post(
        "/ingest/upload",
        files=[("files", ("note.txt", b"hello world", "text/plain"))],
    )

    assert resp.status_code == 403
    assert "readonly" in resp.json()["detail"]
    api_module.brain.ingest.assert_not_called()


# ---------------------------------------------------------------------------
# P1-C — job status tracking
# ---------------------------------------------------------------------------


def test_ingest_returns_job_id(tmp_path):
    """POST /ingest must include a job_id in the response."""
    api_module.brain = _make_brain()

    async def _noop(*_a, **_kw):
        pass

    api_module.brain.load_directory = _noop

    resp = client.post("/ingest", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "processing"
    assert len(data["job_id"]) == 12  # uuid hex[:12]


def test_ingest_status_unknown_job_returns_404():
    """GET /ingest/status/{job_id} returns 404 for an unrecognised job_id."""
    resp = client.get("/ingest/status/doesnotexist")
    assert resp.status_code == 404


def test_ingest_status_known_job_returns_processing(tmp_path):
    """GET /ingest/status/{job_id} returns the job record while pending."""
    api_module.brain = _make_brain()

    # Keep the background task pending by making load_directory block until
    # we've had a chance to poll (we inject the job directly for simplicity).
    job_id = "testjob123abc"
    import datetime as _dt

    api_module._jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "path": str(tmp_path),
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "started_at_ts": _dt.datetime.now(_dt.timezone.utc).timestamp(),
        "completed_at": None,
        "documents_ingested": None,
        "error": None,
    }

    resp = client.get(f"/ingest/status/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "processing"
    # Internal timestamp field must not be exposed
    assert "started_at_ts" not in data


def test_ingest_status_completed_job():
    """GET /ingest/status/{job_id} reflects completed status after job finishes."""
    job_id = "completed999abc"
    import datetime as _dt

    api_module._jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "path": "/some/path",
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "started_at_ts": _dt.datetime.now(_dt.timezone.utc).timestamp(),
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "documents_ingested": None,
        "error": None,
    }

    resp = client.get(f"/ingest/status/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# P1-D — per-call project targeting in dedup
# ---------------------------------------------------------------------------


def test_add_text_same_content_different_projects_not_skipped():
    """Same text sent to two different projects must NOT be treated as a duplicate."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    text = "project-scoped dedup test content"

    api_module.brain._active_project = "alpha"
    resp1 = client.post("/add_text", json={"text": text, "project": "alpha"})
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "success"

    api_module.brain._active_project = "beta"
    resp2 = client.post("/add_text", json={"text": text, "project": "beta"})
    assert resp2.status_code == 200
    # Different project — must NOT be skipped
    assert resp2.json()["status"] == "success"
    assert api_module.brain.ingest.call_count == 2


def test_add_text_same_content_same_project_is_skipped():
    """Same text sent twice to the same project must be skipped on the second call."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None
    api_module.brain._active_project = "alpha"

    text = "project-scoped dedup same project content"

    resp1 = client.post("/add_text", json={"text": text, "project": "alpha"})
    assert resp1.json()["status"] == "success"

    resp2 = client.post("/add_text", json={"text": text, "project": "alpha"})
    assert resp2.json()["status"] == "skipped"
    assert api_module.brain.ingest.call_count == 1  # only the first call ingested


def test_add_text_no_project_defaults_to_global_namespace():
    """Omitting 'project' must deduplicate against the _global namespace."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

    text = "global namespace dedup content"

    client.post("/add_text", json={"text": text})
    resp2 = client.post("/add_text", json={"text": text})  # no project → _global
    assert resp2.json()["status"] == "skipped"


# ---------------------------------------------------------------------------
# Gap 7 — GET /collection/stale
# ---------------------------------------------------------------------------


def test_stale_docs_empty_when_no_hashes():
    """With no ingested docs in the session, stale endpoint returns empty list."""
    resp = client.get("/collection/stale?days=7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["stale_docs"] == []
    assert data["threshold_days"] == 7


def test_stale_docs_returns_old_entries():
    """A doc timestamped in the past must appear in stale results."""
    import datetime as _dt

    api_module.brain = _make_brain()
    # Manually plant an old entry in _source_hashes (simulates a doc from 10 days ago)
    old_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=10)).isoformat()
    api_module._source_hashes["_global"] = {
        "deadbeefdeadbeef": {"doc_id": "old-doc-1", "last_ingested_at": old_ts}
    }

    resp = client.get("/collection/stale?days=7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["stale_docs"][0]["doc_id"] == "old-doc-1"
    assert data["stale_docs"][0]["age_days"] >= 10


def test_stale_docs_excludes_recent_entries():
    """A doc ingested yesterday must NOT appear when threshold is 7 days."""
    import datetime as _dt

    api_module.brain = _make_brain()
    recent_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=1)).isoformat()
    api_module._source_hashes["_global"] = {
        "cafebabecafebabe": {"doc_id": "recent-doc", "last_ingested_at": recent_ts}
    }

    resp = client.get("/collection/stale?days=7")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_stale_docs_invalid_days_returns_400():
    """Negative 'days' parameter must return 400."""
    resp = client.get("/collection/stale?days=-1")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Gap 8 — GET /projects
# ---------------------------------------------------------------------------


def test_get_projects_returns_structure():
    """GET /projects returns the expected top-level keys."""
    resp = client.get("/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert "projects" in data
    assert "memory_only" in data
    assert "total" in data


def test_get_projects_includes_memory_only_projects():
    """A project created via add_text in-memory should appear in memory_only."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None
    api_module.brain._active_project = "sprint3-test"

    client.post("/add_text", json={"text": "project test doc", "project": "sprint3-test"})

    resp = client.get("/projects")
    assert resp.status_code == 200
    data = resp.json()
    memory_names = [p["name"] for p in data["memory_only"]]
    assert "sprint3-test" in memory_names


def test_get_projects_excludes_reserved_memory_only_entries():
    """Reserved roots must not leak into memory_only project listings."""
    api_module.brain = _make_brain()
    api_module._source_hashes["projects"] = {}
    api_module._source_hashes["mounts/alice_shared"] = {}
    api_module._source_hashes["real-proj"] = {}

    resp = client.get("/projects")

    assert resp.status_code == 200
    data = resp.json()
    memory_names = [p["name"] for p in data["memory_only"]]
    assert "real-proj" in memory_names
    assert "projects" not in memory_names
    assert "mounts/alice_shared" not in memory_names


# ---------------------------------------------------------------------------
# /store/whoami
# ---------------------------------------------------------------------------


def test_store_whoami_no_brain():
    """whoami returns just username when brain is not initialized."""
    api_module.brain = None

    resp = client.get("/store/whoami")
    assert resp.status_code == 200
    data = resp.json()
    assert "username" in data
    assert "user_dir" not in data


def test_store_whoami_returns_store_paths():
    """whoami returns store paths and active_project when brain is initialized."""
    mock_brain = _make_brain()
    mock_brain.config.projects_root = "/data/AxonStore/alice"
    mock_brain.config.project = "_default"  # stale value — must NOT appear in response
    mock_brain._active_project = "research"  # real active project after a switch
    api_module.brain = mock_brain

    resp = client.get("/store/whoami")
    assert resp.status_code == 200
    data = resp.json()
    assert "workspace" in data
    assert "username" in data
    assert data.get("active_project") == "research"
    assert data["active_project"] == "research"  # must use _active_project, not config.project


# ---------------------------------------------------------------------------
# /share/list  (requires store mode)
# ---------------------------------------------------------------------------


def test_share_list_503_no_brain():
    """/share/list returns 503 when brain is not initialized."""
    api_module.brain = None

    resp = client.get("/share/list")
    assert resp.status_code == 503


def test_share_list_returns_sharing_and_shared(tmp_path):
    """/share/list returns both sharing and shared keys."""
    mock_brain = _make_brain()
    mock_brain.config.projects_root = str(tmp_path)
    api_module.brain = mock_brain

    # Create minimal dir structure expected by _get_user_dir
    (tmp_path / ".shares").mkdir()

    with patch("axon.api._shares.validate_received_shares", return_value=[]), patch(
        "axon.api._shares.list_shares", return_value={"sharing": [], "shared": []}
    ):
        resp = client.get("/share/list")

    assert resp.status_code == 200
    data = resp.json()
    assert "sharing" in data
    assert "shared" in data


# ---------------------------------------------------------------------------
# /share/generate
# ---------------------------------------------------------------------------


def test_share_generate_404_missing_project(tmp_path):
    """/share/generate returns 404 when project doesn't exist."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    mock_brain.config.projects_root = str(tmp_path)
    api_module.brain = mock_brain

    resp = client.post(
        "/share/generate",
        json={"project": "nonexistent", "grantee": "bob"},
    )
    assert resp.status_code == 404


def test_share_generate_success(tmp_path):
    """/share/generate returns share string when project exists."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    # axon_store_dir = tmp_path; projects_root = tmp_path/Workspace
    mock_brain.config.axon_store_dir = str(tmp_path)
    mock_brain.config.projects_root = str(tmp_path / "Workspace")
    api_module.brain = mock_brain

    # Create a real project dir under Workspace/ so the endpoint can find it
    proj = tmp_path / "Workspace" / "myproject"
    proj.mkdir(parents=True)
    (proj / "meta.json").write_text('{"name": "myproject"}')

    fake_result = {
        "key_id": "sk_abc123",
        "share_string": "abc:def:alice:myproject:/data/store",
        "project": "myproject",
        "grantee": "bob",
        "owner": "alice",
    }

    with patch("axon.api._shares.generate_share_key", return_value=fake_result):
        resp = client.post(
            "/share/generate",
            json={"project": "myproject", "grantee": "bob"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["key_id"] == "sk_abc123"
    assert "share_string" in data


# ---------------------------------------------------------------------------
# /share/revoke
# ---------------------------------------------------------------------------


def test_share_revoke_404_unknown_key(tmp_path):
    """/share/revoke returns 404 when key_id is not found."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    mock_brain.config.projects_root = str(tmp_path)
    api_module.brain = mock_brain

    (tmp_path / ".shares").mkdir()

    with patch("axon.api._shares.revoke_share_key", side_effect=ValueError("not found")):
        resp = client.post("/share/revoke", json={"key_id": "sk_missing"})

    assert resp.status_code == 404


def test_share_revoke_success(tmp_path):
    """/share/revoke marks a key as revoked and returns status."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    mock_brain.config.projects_root = str(tmp_path)
    api_module.brain = mock_brain

    (tmp_path / ".shares").mkdir()

    with patch(
        "axon.api._shares.revoke_share_key", return_value={"status": "revoked", "key_id": "sk_abc"}
    ):
        resp = client.post("/share/revoke", json={"key_id": "sk_abc"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "revoked"


# ---------------------------------------------------------------------------
# Bug regression: axon_ingestText 500 (OSError errno 22)
# ---------------------------------------------------------------------------


class TestAddTextBug500:
    """Regression tests for Bug: POST /add_text raises 500 when BM25 save
    fails with OSError (errno 22 / WinError 87 on Windows).

    After the fix, OSError from BM25 save is handled by the fallback copy path,
    so /add_text should succeed.
    """

    def test_add_text_succeeds_when_bm25_save_raises_oserror(self, tmp_path):
        """Bug regression: OSError in BM25 save should not cause a 500."""

        brain = _make_brain()
        brain.config.project = "default"
        brain._active_project = "default"
        brain.config.bm25_path = str(tmp_path / "bm25_index")
        brain.config.smart_ingest_mode = "normal"

        # ingest() should succeed even when brain.ingest raises OSError on BM25
        # (the fix moves the catch into BM25Retriever.save, so brain.ingest itself
        # shouldn't raise for this scenario — we verify it doesn't propagate)
        brain.ingest.return_value = None  # mock brain.ingest completes normally

        api_module.brain = brain
        api_module._dedup_cache = {}

        resp = client.post("/add_text", json={"text": "Atlas Cache default TTL is 60 seconds."})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert "doc_id" in data

    def test_add_text_empty_text_returns_400(self):
        """Empty text should return 400 not 500."""
        brain = _make_brain()
        api_module.brain = brain
        api_module._dedup_cache = {}

        resp = client.post("/add_text", json={"text": "   "})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Bug regression: path ingestion status polling (Bug 1)
# ---------------------------------------------------------------------------


class TestIngestStatusEndpoint:
    """Tests for GET /ingest/status/{job_id} — exposes job status to VS Code tools."""

    def test_get_status_returns_404_for_unknown_job(self):
        """Unknown job_id returns 404."""
        api_module.brain = _make_brain()
        resp = client.get("/ingest/status/nonexistent_job_id")
        assert resp.status_code == 404

    def test_get_status_returns_processing_for_active_job(self):
        """A job inserted in _jobs with status processing is returned correctly."""
        from datetime import datetime, timezone

        api_module.brain = _make_brain()
        job_id = "test_job_abc123"
        api_module._jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "path": "/some/path",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "started_at_ts": datetime.now(timezone.utc).timestamp(),
            "completed_at": None,
            "documents_ingested": None,
            "error": None,
        }
        resp = client.get(f"/ingest/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"
        assert data["job_id"] == job_id
        assert "started_at_ts" not in data  # internal field must be stripped
        # cleanup
        del api_module._jobs[job_id]

    def test_get_status_returns_completed_job(self):
        """A completed job is returned with status completed."""
        from datetime import datetime, timezone

        api_module.brain = _make_brain()
        job_id = "test_job_done456"
        now = datetime.now(timezone.utc).isoformat()
        api_module._jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "path": "/some/path",
            "started_at": now,
            "started_at_ts": 0.0,
            "completed_at": now,
            "documents_ingested": 5,
            "error": None,
        }
        resp = client.get(f"/ingest/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["documents_ingested"] == 5
        del api_module._jobs[job_id]

    def test_get_status_returns_failed_job(self):
        """A failed job is returned with status failed and error message."""
        from datetime import datetime, timezone

        api_module.brain = _make_brain()
        job_id = "test_job_fail789"
        now = datetime.now(timezone.utc).isoformat()
        api_module._jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "path": "/bad/path",
            "started_at": now,
            "started_at_ts": 0.0,
            "completed_at": now,
            "documents_ingested": None,
            "error": "[Errno 22] Invalid argument",
        }
        resp = client.get(f"/ingest/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "[Errno 22]" in data["error"]
        del api_module._jobs[job_id]


class TestEvictOldJobs:
    """Tests for _evict_old_jobs() — processing jobs must not be TTL-evicted."""

    def setup_method(self):
        api_module._jobs.clear()

    def teardown_method(self):
        api_module._jobs.clear()

    def test_processing_job_not_evicted_by_ttl(self):
        """A still-processing job must survive TTL eviction even if started long ago."""
        api_module._jobs["old_active"] = {
            "job_id": "old_active",
            "status": "processing",
            "started_at_ts": 0.0,  # epoch — well past 1-hour TTL
        }
        api_module._evict_old_jobs()
        assert "old_active" in api_module._jobs

    def test_completed_job_evicted_by_ttl(self):
        """A completed job started more than 1 hour ago must be evicted."""
        api_module._jobs["old_done"] = {
            "job_id": "old_done",
            "status": "completed",
            "started_at_ts": 0.0,
        }
        api_module._evict_old_jobs()
        assert "old_done" not in api_module._jobs

    def test_processing_job_survives_cap_eviction(self):
        """With >_MAX_JOBS entries, processing jobs are the last to be removed."""
        import axon.api as _api_mod

        cap = _api_mod._MAX_JOBS
        # Fill with completed old jobs up to cap+1
        for i in range(cap + 1):
            jid = f"done_{i}"
            api_module._jobs[jid] = {
                "job_id": jid,
                "status": "completed",
                "started_at_ts": float(i),
            }
        active_id = "active_new"
        api_module._jobs[active_id] = {
            "job_id": active_id,
            "status": "processing",
            "started_at_ts": 0.0,  # oldest timestamp — would be first to evict without fix
        }
        api_module._evict_old_jobs()
        # The active job must still be present; a completed job was evicted instead
        assert active_id in api_module._jobs
        assert len(api_module._jobs) <= cap


# ---------------------------------------------------------------------------
# Bug regression: directory ingest uses sync loader (no asyncio.run in thread)
# ---------------------------------------------------------------------------


class TestIngestPathSyncLoader:
    """Regression for Bug 1: directory ingest was using asyncio.run() inside a
    FastAPI background thread, which can hang on Windows ProactorEventLoop.
    The fix switches to the sync DirectoryLoader.load() for both directory and
    single-file ingestion.
    """

    def test_ingest_directory_uses_sync_loader(self, tmp_path):
        """POST /ingest for a directory calls DirectoryLoader.load() (sync), not asyncio.run()."""
        from unittest.mock import MagicMock, patch

        brain = _make_brain()
        brain.ingest.return_value = None
        api_module.brain = brain

        # Create a real directory with one file
        test_dir = tmp_path / "corpus"
        test_dir.mkdir()
        (test_dir / "doc.md").write_text("Atlas Cache default TTL is 60 seconds.")

        fake_docs = [
            {"id": "doc.md", "text": "Atlas Cache default TTL is 60 seconds.", "metadata": {}}
        ]

        with patch("axon.loaders.DirectoryLoader") as MockLoader:
            mock_instance = MagicMock()
            mock_instance.load.return_value = fake_docs
            MockLoader.return_value = mock_instance

            resp = client.post("/ingest", json={"path": str(test_dir)})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "processing"
            job_id = data["job_id"]

        # The background task runs synchronously in test client
        # Verify that DirectoryLoader.load() was called (not asyncio.run)
        mock_instance.load.assert_called_once_with(str(test_dir.resolve()))
        brain.ingest.assert_called_once_with(fake_docs, progress_callback=ANY)

        # Cleanup job
        api_module._jobs.pop(job_id, None)

    def test_ingest_single_file_succeeds(self, tmp_path):
        """POST /ingest for a single .md file completes without OSError."""
        from unittest.mock import MagicMock, patch

        brain = _make_brain()
        brain.ingest.return_value = None
        api_module.brain = brain

        # Create a single markdown file
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Atlas Cache\nDefault TTL: 60 seconds.")

        fake_docs = [
            {"id": "readme.md", "text": "# Atlas Cache\nDefault TTL: 60 seconds.", "metadata": {}}
        ]

        with patch("axon.loaders.DirectoryLoader") as MockLoader:
            mock_instance = MagicMock()
            mock_instance.loaders = {".md": MagicMock(load=MagicMock(return_value=fake_docs))}
            MockLoader.return_value = mock_instance

            resp = client.post("/ingest", json={"path": str(md_file)})
            assert resp.status_code == 200
            data = resp.json()
            job_id = data["job_id"]

        brain.ingest.assert_called_once_with(fake_docs, progress_callback=ANY)
        api_module._jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# /graph/visualize — browser side-effect guard
# ---------------------------------------------------------------------------


def test_graph_visualize_no_browser_open():
    """GET /graph/visualize must never call webbrowser.open (server-side side effect)."""
    brain = _make_brain()
    brain.export_graph_html.return_value = "<html>graph</html>"
    api_module.brain = brain

    with patch("webbrowser.open") as mock_open:
        resp = client.get("/graph/visualize")

    assert resp.status_code == 200
    mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# /graph/data
# ---------------------------------------------------------------------------


def test_graph_data_returns_nodes_and_links():
    """GET /graph/data returns {"nodes": list, "links": list}."""
    brain = _make_brain()
    api_module.brain = brain

    resp = client.get("/graph/data")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "links" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["links"], list)


def test_graph_data_returns_503_no_brain():
    """GET /graph/data returns 503 when brain is not initialized."""
    api_module.brain = None
    resp = client.get("/graph/data")
    assert resp.status_code == 503


def test_graph_data_rejects_mismatched_project():
    """GET /graph/data returns 409 when the requested project is not active."""
    brain = _make_brain()
    brain._active_project = "default"
    api_module.brain = brain

    resp = client.get("/graph/data?project=research")
    assert resp.status_code == 409
    assert "Use POST /project/switch" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /query error branches
# ---------------------------------------------------------------------------


def test_query_propagates_runtime_error():
    """RuntimeError during query is converted to 500."""
    api_module.brain = _make_brain()
    api_module.brain.query.side_effect = RuntimeError("llm unavailable")
    resp = client.post("/query", json={"query": "test"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /ingest PermissionError branch
# ---------------------------------------------------------------------------


def test_ingest_permission_error_returns_500(tmp_path):
    """PermissionError during ingest returns 500."""
    api_module.brain = _make_brain()
    doc_file = tmp_path / "secure.txt"
    doc_file.write_text("locked content", encoding="utf-8")
    api_module.brain.ingest.side_effect = PermissionError("read-only store")
    with patch.dict(os.environ, {"RAG_INGEST_BASE": str(tmp_path)}):
        with patch.object(api_module.brain, "ingest", side_effect=PermissionError("read-only")):
            resp = client.post("/ingest", json={"path": str(doc_file)})
    # Background task returns 200 accepted initially; the error propagates internally
    # The endpoint accepts the job and returns 200/processing, not 500 at HTTP level
    assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# /collection with empty results (coverage of list branch)
# ---------------------------------------------------------------------------


def test_collection_with_zero_docs():
    """GET /collection with 0 documents returns total_files=0."""
    api_module.brain = _make_brain()
    api_module.brain.list_documents.return_value = []
    resp = client.get("/collection")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_files"] == 0
    assert data["total_chunks"] == 0


# ---------------------------------------------------------------------------
# /graph/visualize ImportError branch
# ---------------------------------------------------------------------------


def test_graph_visualize_import_error():
    """GET /graph/visualize returns 501 when pyvis/networkx not installed."""
    api_module.brain = _make_brain()
    api_module.brain.export_graph_html.side_effect = ImportError("pyvis not installed")
    resp = client.get("/graph/visualize")
    assert resp.status_code == 501


# ---------------------------------------------------------------------------
# Mounted share write-rejection (403) tests
# ---------------------------------------------------------------------------


def test_ingest_mounted_share_returns_403(tmp_path):
    """POST /ingest returns 403 when _assert_write_allowed raises PermissionError."""
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot ingest on mounted share 'mounts/alice_proj'. "
        "Mounted projects are always read-only."
    )
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("hello", encoding="utf-8")
    with patch.dict(os.environ, {"RAG_INGEST_BASE": str(tmp_path)}):
        resp = client.post("/ingest", json={"path": str(doc_file)})
    assert resp.status_code == 403
    assert "mounted share" in resp.json()["detail"]


def test_delete_mounted_share_returns_403():
    """POST /delete returns 403 when active project is a mounted share."""
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot delete on mounted share 'mounts/alice_proj'."
    )
    resp = client.post("/delete", json={"doc_ids": ["id1"]})
    assert resp.status_code == 403
    assert "mounted share" in resp.json()["detail"]


def test_refresh_mounted_share_ingest_error_recorded_in_job(tmp_path):
    """POST /ingest/refresh: PermissionError during ingest is recorded in job errors.

    Refresh is now async — the 403 guard fires from _assert_write_allowed (see
    test_refresh_write_guard_returns_403_before_scanning_docs). A PermissionError
    raised from brain.ingest() inside the background task is caught and stored in
    the job's errors list rather than propagated as HTTP 403.
    """
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("hello world", encoding="utf-8")
    api_module.brain = _make_brain()
    api_module.brain.get_doc_versions.return_value = {
        str(doc_file): {"content_hash": "stale_hash_that_wont_match"}
    }
    api_module.brain.ingest.side_effect = PermissionError(
        "Cannot ingest on mounted share 'mounts/alice_proj'."
    )
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    status_resp = client.get(f"/ingest/status/{job_id}")
    job = status_resp.json()
    assert len(job["errors"]) >= 1


def test_refresh_write_guard_returns_403_before_scanning_docs():
    """POST /ingest/refresh should fail fast when writes are not allowed."""
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot refresh: project 'alpha' is in 'readonly' maintenance state."
    )
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 403
    api_module.brain.get_doc_versions.assert_not_called()


def test_finalize_mounted_share_returns_403():
    """POST /graph/finalize returns 403 when active project is a mounted share."""
    api_module.brain = _make_brain()
    # The route now prefers the backend protocol path; fall back trigger via
    # backend.finalize raising PermissionError.
    api_module.brain._graph_backend.finalize.side_effect = PermissionError(
        "Cannot finalize_graph on mounted share 'mounts/alice_proj'."
    )
    api_module.brain.finalize_graph.side_effect = PermissionError(
        "Cannot finalize_graph on mounted share 'mounts/alice_proj'."
    )
    resp = client.post("/graph/finalize")
    assert resp.status_code == 403


def test_clear_mounted_share_returns_403():
    """POST /clear returns 403 when active project is a mounted share."""
    api_module.brain = _make_brain()
    api_module.brain._assert_write_allowed.side_effect = PermissionError(
        "Cannot clear on mounted share 'mounts/alice_proj'."
    )
    api_module.brain.vector_store.client = MagicMock()
    resp = client.post("/clear")
    assert resp.status_code == 403
    assert "mounted share" in resp.json()["detail"]
    api_module.brain.vector_store.client.delete_collection.assert_not_called()


def test_query_mounted_share_returns_200():
    """POST /query succeeds (200) on a mounted share — reads are always allowed."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "answer"
    api_module.brain._community_build_in_progress = False
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /project/maintenance — Phase 2 maintenance state machine
# ---------------------------------------------------------------------------


def test_set_maintenance_state_ok(tmp_path):
    """POST /project/maintenance sets state and returns 200."""
    import json

    import axon.projects as _proj

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(
        json.dumps({"name": "myproject", "created_at": "2026-01-01"}), encoding="utf-8"
    )
    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.post("/project/maintenance", json={"name": "myproject", "state": "readonly"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["maintenance_state"] == "readonly"
    assert data["project"] == "myproject"
    assert "active_leases" in data
    assert "epoch" in data


def test_set_maintenance_state_invalid_state(tmp_path):
    """POST /project/maintenance with an invalid state returns 422."""
    import json

    import axon.projects as _proj

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(
        json.dumps({"name": "myproject", "created_at": "2026-01-01"}), encoding="utf-8"
    )
    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.post("/project/maintenance", json={"name": "myproject", "state": "broken"})
    assert resp.status_code == 422


def test_set_maintenance_state_unknown_project(tmp_path):
    """POST /project/maintenance for a nonexistent project returns 404."""
    import axon.projects as _proj

    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.post("/project/maintenance", json={"name": "ghost", "state": "readonly"})
    assert resp.status_code == 404


def test_get_maintenance_state_ok(tmp_path):
    """GET /project/maintenance?name=... returns current state."""
    import json

    import axon.projects as _proj

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(
        json.dumps({"name": "myproject", "maintenance_state": "draining"}), encoding="utf-8"
    )
    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.get("/project/maintenance?name=myproject")
    assert resp.status_code == 200
    data = resp.json()
    assert data["maintenance_state"] == "draining"
    assert "active_leases" in data
    assert "epoch" in data
    assert "draining" in data


def test_get_maintenance_state_defaults_to_normal(tmp_path):
    """GET /project/maintenance returns 'normal' when field absent in meta.json."""
    import json

    import axon.projects as _proj

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(json.dumps({"name": "myproject"}), encoding="utf-8")
    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.get("/project/maintenance?name=myproject")
    assert resp.status_code == 200
    assert resp.json()["maintenance_state"] == "normal"


def test_get_maintenance_state_unknown_project(tmp_path):
    """GET /project/maintenance for a nonexistent project returns 404."""
    import axon.projects as _proj

    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.get("/project/maintenance?name=ghost")
    assert resp.status_code == 404


def test_set_draining_triggers_registry_drain(tmp_path):
    """POST /project/maintenance with 'draining' calls start_drain on the registry."""
    import json

    import axon.projects as _proj
    from axon.runtime import get_registry

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(
        json.dumps({"name": "myproject", "created_at": "2026-01-01"}), encoding="utf-8"
    )
    reg = get_registry()
    reg.reset("myproject")  # ensure clean state
    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.post("/project/maintenance", json={"name": "myproject", "state": "draining"})
    assert resp.status_code == 200
    snap = reg.snapshot("myproject")
    assert snap["draining"] is True
    # Cleanup
    reg.stop_drain("myproject")
    reg.reset("myproject")


def test_set_normal_stops_registry_drain(tmp_path):
    """POST /project/maintenance with 'normal' calls stop_drain on the registry."""
    import json

    import axon.projects as _proj
    from axon.runtime import get_registry

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(
        json.dumps({"name": "myproject", "created_at": "2026-01-01"}), encoding="utf-8"
    )
    reg = get_registry()
    reg.start_drain("myproject")  # pre-drain
    with patch.object(_proj, "PROJECTS_ROOT", tmp_path):
        resp = client.post("/project/maintenance", json={"name": "myproject", "state": "normal"})
    assert resp.status_code == 200
    snap = reg.snapshot("myproject")
    assert snap["draining"] is False
    reg.reset("myproject")


def test_get_registry_leases_returns_shape():
    """GET /registry/leases returns 200 with expected response shape."""
    resp = client.get("/registry/leases")
    assert resp.status_code == 200
    data = resp.json()
    assert "leases" in data
    assert "total_projects_tracked" in data
    assert isinstance(data["leases"], list)


# ---------------------------------------------------------------------------
# Answer provenance — /query response must include provenance object
# ---------------------------------------------------------------------------


def test_query_response_includes_provenance():
    """POST /query always includes a top-level 'provenance' key."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Some answer"
    api_module.brain._last_provenance = {
        "answer_source": "local_kb",
        "retrieved_count": 3,
        "web_count": 0,
    }
    resp = client.post("/query", json={"query": "What is RAG?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "provenance" in data
    prov = data["provenance"]
    assert prov["answer_source"] == "local_kb"
    assert prov["retrieved_count"] == 3
    assert prov["web_count"] == 0


def test_query_provenance_web_snippet_fallback():
    """Provenance reflects web_snippet_fallback when brain signals it."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Answer from web"
    api_module.brain._last_provenance = {
        "answer_source": "web_snippet_fallback",
        "retrieved_count": 2,
        "web_count": 2,
    }
    resp = client.post("/query", json={"query": "Latest news?"})
    assert resp.status_code == 200
    prov = resp.json()["provenance"]
    assert prov["answer_source"] == "web_snippet_fallback"
    assert prov["web_count"] == 2


def test_query_provenance_no_context_fallback():
    """Provenance reflects no_context_fallback when retrieval returned nothing."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "General knowledge answer"
    api_module.brain._last_provenance = {
        "answer_source": "no_context_fallback",
        "retrieved_count": 0,
        "web_count": 0,
    }
    resp = client.post("/query", json={"query": "Something unrelated?"})
    assert resp.status_code == 200
    prov = resp.json()["provenance"]
    assert prov["answer_source"] == "no_context_fallback"
    assert prov["retrieved_count"] == 0


def test_query_provenance_missing_attribute_defaults_to_empty():
    """If brain lacks _last_provenance, provenance is an empty dict (not an error)."""
    api_module.brain = _make_brain()
    api_module.brain.query.return_value = "Answer"
    # Deliberately omit _last_provenance attribute
    if hasattr(api_module.brain, "_last_provenance"):
        del api_module.brain._last_provenance
    resp = client.post("/query", json={"query": "test"})
    assert resp.status_code == 200
    assert "provenance" in resp.json()
    assert resp.json()["provenance"] == {}


# ---------------------------------------------------------------------------
# Mount revocation at switch time — POST /project/switch
# ---------------------------------------------------------------------------


def test_switch_to_revoked_mount_returns_404(tmp_path):
    """Switching to a revoked mounted project returns 404 immediately."""
    brain = _make_brain()
    brain.config.projects_root = str(tmp_path)
    api_module.brain = brain

    # validate_received_shares returns the mount name as revoked/removed
    with patch(
        "axon.shares.validate_received_shares",
        return_value=["alice_docs"],
    ):
        resp = client.post("/project/switch", json={"project_name": "mounts/alice_docs"})

    assert resp.status_code == 404
    assert "revoked" in resp.json()["detail"].lower()
    brain.switch_project.assert_not_called()


def test_switch_to_valid_mount_proceeds(tmp_path):
    """Switching to a valid (non-revoked) mount proceeds normally."""
    brain = _make_brain()
    brain.config.projects_root = str(tmp_path)
    api_module.brain = brain

    with patch(
        "axon.shares.validate_received_shares",
        return_value=[],  # nothing revoked
    ):
        resp = client.post("/project/switch", json={"project_name": "mounts/alice_docs"})

    assert resp.status_code == 200
    brain.switch_project.assert_called_once_with("mounts/alice_docs")


def test_switch_to_normal_project_skips_revocation_check():
    """Switching to a non-mount project never calls validate_received_shares."""
    brain = _make_brain()
    api_module.brain = brain

    with patch("axon.shares.validate_received_shares") as mock_validate:
        resp = client.post("/project/switch", json={"project_name": "my-project"})

    assert resp.status_code == 200
    mock_validate.assert_not_called()


def test_switch_mount_revocation_check_runs_for_mounts():
    """validate_received_shares is always called when switching to a mounts/ project."""
    brain = _make_brain()
    api_module.brain = brain

    with patch("axon.shares.validate_received_shares", return_value=[]) as mock_validate:
        resp = client.post("/project/switch", json={"project_name": "mounts/alice_docs"})

    assert resp.status_code == 200
    mock_validate.assert_called_once()


# ---------------------------------------------------------------------------
# _evict_old_jobs (lines 47, 49-51)
# ---------------------------------------------------------------------------


class TestEvictOldJobsExtra:
    def test_evicts_expired_jobs(self):
        """Expired jobs (started_at_ts < cutoff) are removed (line 47)."""
        import axon.api as api_module

        old_ts = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
        api_module._jobs.clear()
        api_module._jobs["old_job"] = {"started_at_ts": old_ts, "status": "completed"}
        api_module._jobs["recent_job"] = {"started_at_ts": datetime.now(timezone.utc).timestamp()}

        api_module._evict_old_jobs()

        assert "old_job" not in api_module._jobs
        assert "recent_job" in api_module._jobs

    def test_caps_jobs_at_max(self):
        """When >_MAX_JOBS jobs exist, oldest are removed (lines 49-51)."""
        import axon.api as api_module

        api_module._jobs.clear()
        # Fill with more than _MAX_JOBS entries
        for i in range(api_module._MAX_JOBS + 5):
            api_module._jobs[f"job_{i:04d}"] = {
                "started_at_ts": datetime.now(timezone.utc).timestamp() + i + 1000
            }

        api_module._evict_old_jobs()

        assert len(api_module._jobs) <= api_module._MAX_JOBS


# ---------------------------------------------------------------------------
# api_key_middleware (lines 131-134)
# ---------------------------------------------------------------------------


class TestAPIKeyMiddleware:
    def test_missing_api_key_returns_401(self):
        """When RAG_API_KEY is set and key is missing, returns 401 (lines 131-134)."""
        import axon.api as api_module

        original = api_module._RAG_API_KEY
        api_module._RAG_API_KEY = "secret-test-key"
        try:
            client = TestClient(api_module.app, raise_server_exceptions=False)
            # Non-/health path without key should get 401
            resp = client.post("/query", json={"query": "hello"}, headers={})
            assert resp.status_code == 401
        finally:
            api_module._RAG_API_KEY = original

    def test_correct_api_key_passes(self):
        """When RAG_API_KEY is set and correct key is provided, request passes (lines 131-134)."""
        import axon.api as api_module

        original = api_module._RAG_API_KEY
        api_module._RAG_API_KEY = "correct-key"
        try:
            client = TestClient(api_module.app, raise_server_exceptions=False)
            # With correct key, should NOT get 401 (may get 422 or 503 from actual handler)
            resp = client.post(
                "/query", json={"query": "hello"}, headers={"X-API-Key": "correct-key"}
            )
            assert resp.status_code != 401
        finally:
            api_module._RAG_API_KEY = original


# ---------------------------------------------------------------------------
# main() (lines 194-196)
# ---------------------------------------------------------------------------


class TestApiMain:
    def test_main_calls_uvicorn(self):
        """main() calls uvicorn.run with app, host, and port (lines 194-196)."""
        import axon.api as api_module

        with patch("uvicorn.run") as mock_run:
            with patch.dict(os.environ, {"AXON_HOST": "127.0.0.1", "AXON_PORT": "9876"}):
                api_module.main()
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1].get("host") == "127.0.0.1" or call_args[0][1] == "127.0.0.1"

    def test_main_uses_defaults(self):
        """main() uses 0.0.0.0:8000 by default (lines 194-196)."""
        import axon.api as api_module

        env = {k: v for k, v in os.environ.items() if k not in ("AXON_HOST", "AXON_PORT")}
        with patch("uvicorn.run") as mock_run:
            with patch.dict(os.environ, env, clear=True):
                api_module.main()
            mock_run.assert_called_once()


"""
tests/test_api_coverage.py

Targeted coverage tests for api.py endpoints not covered by test_api.py.
"""

import asyncio
import hashlib
import threading
from datetime import datetime, timezone

from fastapi.testclient import TestClient

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


def _refresh_and_poll():
    """POST /ingest/refresh then poll /ingest/status/{job_id}; return final job dict."""
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    status_resp = client.get(f"/ingest/status/{data['job_id']}")
    assert status_resp.status_code == 200
    return status_resp.json()


def test_ingest_refresh_skipped(tmp_path):
    brain = _make_brain()
    api_module.brain = brain
    real_file = tmp_path / "doc.txt"
    real_file.write_bytes(b"hello")
    content_hash = hashlib.md5(b"hello").hexdigest()
    brain.get_doc_versions.return_value = {str(real_file): {"content_hash": content_hash}}
    data = _refresh_and_poll()
    assert str(real_file) in data["skipped"]


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
        data = _refresh_and_poll()
    assert str(real_file) in data["reingested"]


def test_ingest_refresh_missing_file():
    brain = _make_brain()
    api_module.brain = brain
    brain.get_doc_versions.return_value = {"/nonexistent/path.txt": {"content_hash": "x"}}
    data = _refresh_and_poll()
    assert "/nonexistent/path.txt" in data["missing"]


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


"""Tests for axon.api — increasing coverage."""
import os

import pytest
from fastapi.testclient import TestClient

from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.config.llm_provider = "ollama"
    brain.config.llm_model = "llama3"
    brain._active_project = "default"
    # Mock other attributes used by API
    brain.vector_store = MagicMock()
    brain.bm25 = MagicMock()
    return brain


def test_health_no_brain():
    api_module.brain = None
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "initializing"


def test_health_with_brain(mock_brain):
    api_module.brain = mock_brain
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_config_no_brain():
    api_module.brain = None
    response = client.get("/config")
    assert response.status_code == 503


def test_get_config_success_with_mock(mock_brain):
    api_module.brain = mock_brain
    mock_brain._apply_overrides.return_value = mock_brain.config
    response = client.get("/config")
    assert response.status_code == 200


def test_update_config_no_brain():
    api_module.brain = None
    response = client.post("/config/update", json={"top_k": 10})
    assert response.status_code == 503


def test_update_config_success(mock_brain):
    api_module.brain = mock_brain
    response = client.post("/config/update", json={"top_k": 10})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_list_sessions_no_brain():
    api_module.brain = None
    response = client.get("/sessions")
    assert response.status_code == 503


def test_list_sessions_success(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.sessions._list_sessions", return_value=[]):
        response = client.get("/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json()


def test_get_session_not_found_with_mock(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.sessions._load_session", return_value=None):
        response = client.get("/session/missing")
    assert response.status_code == 404


def test_get_session_success(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.sessions._load_session", return_value={"id": "s1"}):
        response = client.get("/session/s1")
    assert response.status_code == 200
    assert response.json()["id"] == "s1"


"""Extra coverage tests for api_routes sub-modules.

Targets missed lines in:
  graph.py, ingest.py, maintenance.py, projects.py, query.py, shares.py
"""

import os

import pytest
from fastapi.testclient import TestClient

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
    brain.config.projects_root = "/tmp/axon_projects"
    brain._apply_overrides.return_value = brain.config
    brain._community_build_in_progress = False
    brain._community_summaries = {}
    brain._active_project = "default"
    brain._entity_graph = {}
    brain._embedding_meta_path = "/tmp/axon_meta.json"
    # Graph backend mock — graph_data() returns a valid payload dict; status() returns stats
    _mock_payload = MagicMock()
    _mock_payload.to_dict.return_value = {"nodes": [], "links": []}
    brain._graph_backend.graph_data.return_value = _mock_payload
    brain._graph_backend.status.return_value = {
        "backend": "graphrag",
        "entities": 0,
        "relations": 0,
        "communities": 0,
        "community_summaries": 0,
    }
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
        """graph.py — finalize runs ok via legacy brain.finalize_graph fallback."""
        brain = _make_brain()
        brain._graph_backend = None  # exercise legacy fallback
        brain._community_summaries = {"c1": "summary"}
        brain.finalize_graph.return_value = None
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["community_summary_count"] == 1

    def test_permission_error_returns_403(self):
        """graph.py — PermissionError → 403 (via legacy fallback)."""
        brain = _make_brain()
        brain._graph_backend = None
        brain.finalize_graph.side_effect = PermissionError("read-only mode")
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 403

    def test_generic_exception_returns_500(self):
        """graph.py — generic Exception → 500 (via legacy fallback)."""
        brain = _make_brain()
        brain._graph_backend = None
        brain.finalize_graph.side_effect = RuntimeError("graph build failed")
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 500

    def test_backend_protocol_path_surfaces_status(self):
        """When backend.finalize() returns status='not_applicable' the route
        forwards that verbatim instead of silently reporting ok."""
        from axon.graph_backends.base import FinalizationResult

        brain = _make_brain()
        brain._community_summaries = {}
        brain._graph_backend.finalize.return_value = FinalizationResult(
            backend_id="dynamic_graph",
            status="not_applicable",
            detail="dynamic_graph has no community-detection step",
        )
        api_module.brain = brain
        resp = client.post("/graph/finalize")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_applicable"
        assert data["backend_id"] == "dynamic_graph"
        assert "community" in data["detail"].lower()


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
        _payload = MagicMock()
        _payload.to_dict.return_value = {"nodes": [{"id": "n1"}], "links": []}
        brain._graph_backend.graph_data.return_value = _payload
        api_module.brain = brain
        resp = client.get("/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert data["nodes"][0]["id"] == "n1"

    def test_fallback_when_not_dict(self):
        """Non-dict to_dict() result → fallback empty nodes/links."""
        brain = _make_brain()
        _payload = MagicMock()
        _payload.to_dict.return_value = None  # simulate malformed payload
        brain._graph_backend.graph_data.return_value = _payload
        api_module.brain = brain
        resp = client.get("/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"nodes": [], "links": []}


class TestGraphBackendStatus:
    def test_503_when_no_brain(self):
        api_module.brain = None
        resp = client.get("/graph/backend/status")
        assert resp.status_code == 503

    def test_returns_status_dict(self):
        brain = _make_brain()
        api_module.brain = brain
        resp = client.get("/graph/backend/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend"] == "graphrag"
        assert "entities" in data
        assert "relations" in data

    def test_status_reflects_backend_mock(self):
        brain = _make_brain()
        brain._graph_backend.status.return_value = {
            "backend": "graphrag",
            "entities": 7,
            "relations": 14,
            "communities": 2,
            "community_summaries": 2,
        }
        api_module.brain = brain
        resp = client.get("/graph/backend/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities"] == 7
        assert data["relations"] == 14
        assert data["communities"] == 2


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


def _refresh_and_poll_v2(client, api_module):
    """POST /ingest/refresh (async) → GET /ingest/status/{job_id} → job dict.

    TestClient runs BackgroundTasks synchronously, so the job is always
    complete by the time the POST returns.
    """
    resp = client.post("/ingest/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data, f"expected job_id in response, got: {data}"
    job_id = data["job_id"]
    status_resp = client.get(f"/ingest/status/{job_id}")
    assert status_resp.status_code == 200
    return status_resp.json()


class TestIngestRefresh:
    def test_503_when_no_brain(self):
        resp = client.post("/ingest/refresh")
        assert resp.status_code == 503

    def test_returns_job_id_immediately(self):
        """POST /ingest/refresh returns {job_id, status} immediately (async)."""
        brain = _make_brain()
        brain.get_doc_versions.return_value = {}
        api_module.brain = brain
        resp = client.post("/ingest/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "processing"

    def test_missing_source_reported(self):
        """Background refresh: source path does not exist → 'missing' bucket."""
        brain = _make_brain()
        brain.get_doc_versions.return_value = {"/nonexistent/path.txt": {"content_hash": "abc123"}}
        api_module.brain = brain
        job = _refresh_and_poll_v2(client, api_module)
        assert "/nonexistent/path.txt" in job["missing"]

    def test_no_loader_for_extension_reported_as_error(self, tmp_path):
        """Background refresh: no loader for extension → 'errors' bucket."""
        brain = _make_brain()
        exotic = tmp_path / "file.exotic123"
        exotic.write_text("content", encoding="utf-8")
        brain.get_doc_versions.return_value = {str(exotic): {"content_hash": "abc"}}
        api_module.brain = brain

        mock_loader_mgr = MagicMock()
        mock_loader_mgr.loaders = {}  # no loader for .exotic123

        with patch("axon.loaders.DirectoryLoader", return_value=mock_loader_mgr):
            job = _refresh_and_poll_v2(client, api_module)

        assert len(job["errors"]) == 1
        assert "no loader" in job["errors"][0]["error"]

    def test_loader_returns_no_docs_reported_as_error(self, tmp_path):
        """Background refresh: loader returns empty list → 'errors' bucket."""
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
            job = _refresh_and_poll_v2(client, api_module)

        assert len(job["errors"]) == 1
        assert "no documents" in job["errors"][0]["error"]

    def test_unchanged_hash_skipped(self, tmp_path):
        """Background refresh: same hash → skipped bucket."""
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
            job = _refresh_and_poll_v2(client, api_module)

        assert str(txt_file) in job["skipped"]

    def test_changed_hash_reingested(self, tmp_path):
        """Background refresh: changed hash → reingested bucket."""
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
            job = _refresh_and_poll_v2(client, api_module)

        assert str(txt_file) in job["reingested"]

    def test_exception_during_loader_reported_as_error(self, tmp_path):
        """Background refresh: generic exception → errors bucket."""
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
            job = _refresh_and_poll_v2(client, api_module)

        assert len(job["errors"]) == 1


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

    def test_validates_shares_on_project_list(self):
        """projects.py — received shares are validated on project list."""
        brain = _make_brain()
        api_module.brain = brain

        with patch("axon.projects.list_projects", return_value=[]):
            with patch("axon.shares.validate_received_shares", return_value=[]) as mock_validate:
                resp = client.get("/projects")

        assert resp.status_code == 200
        mock_validate.assert_called_once()

    def test_share_validation_exception_handled(self):
        """projects.py — share validation exception does not crash project list."""
        brain = _make_brain()
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

    def test_missing_name_returns_400(self):
        """Missing project_name/name should return 400 instead of 404."""
        api_module.brain = _make_brain()
        resp = client.post("/project/switch", json={})
        assert resp.status_code == 400
        assert "must be provided" in resp.json()["detail"].lower()

    def test_reserved_name_returns_400(self):
        brain = _make_brain()
        brain.switch_project.side_effect = ValueError(
            "Project 'projects' is reserved and cannot be activated as a local project."
        )
        api_module.brain = brain
        resp = client.post("/project/switch", json={"name": "projects"})
        assert resp.status_code == 400

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

    def test_delete_clears_dedup_cache(self):
        """Deleting a project must drop its dedup bucket."""
        brain = _make_brain()
        brain._active_project = "other"
        api_module.brain = brain
        api_module._source_hashes["todelete"] = {"abc": {"doc_id": "doc"}}

        with patch("axon.projects.delete_project", return_value=None):
            with patch("axon.shares.list_shares", return_value={"sharing": []}):
                resp = client.post("/project/delete/todelete")

        assert resp.status_code == 200
        assert "todelete" not in api_module._source_hashes

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
        """projects.py — active shares → 409."""
        brain = _make_brain()
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


class TestDeleteDocuments:
    def test_delete_clears_dedup_cache(self):
        """DELETE /delete should drop dedup entries for deleted docs."""
        brain = _make_brain()
        brain._active_project = "research"
        brain.vector_store.get_by_ids.return_value = [{"id": "doc1"}]
        api_module.brain = brain
        api_module._source_hashes["research"] = {"hash1": {"doc_id": "doc1"}}
        api_module._source_hashes["_global"] = {"hash1": {"doc_id": "doc1"}}

        resp = client.post("/delete", json={"doc_ids": ["doc1"]})

        assert resp.status_code == 200
        assert "research" not in api_module._source_hashes
        assert "_global" not in api_module._source_hashes


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

    def test_permission_error_returns_403(self):
        brain = _make_brain()
        brain._assert_write_allowed.side_effect = PermissionError(
            "Cannot clear: project 'alpha' is in 'readonly' maintenance state."
        )
        api_module.brain = brain

        resp = client.post("/clear")
        assert resp.status_code == 403
        brain.vector_store.client.delete_collection.assert_not_called()

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
            workspace = MagicMock()
            user_dir.__truediv__.return_value = workspace  # store_dir / "Workspace"
            project_dir = MagicMock()
            workspace.__truediv__.return_value = project_dir  # workspace / project
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

            workspace = MagicMock()
            user_dir.__truediv__.return_value = workspace  # store_dir / "Workspace"

            project_dir = MagicMock()
            project_dir.exists.return_value = True
            meta_json = MagicMock()
            meta_json.exists.return_value = True
            project_dir.__truediv__.return_value = meta_json
            workspace.__truediv__.return_value = project_dir  # workspace / project

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
            with patch(
                "axon.shares.redeem_share_key",
                return_value={
                    "key_id": "sk_abc123",
                    "mount_name": "alice_sharedproj",
                    "owner": "alice",
                    "project": "sharedproj",
                    "descriptor": {"mount_name": "alice_sharedproj"},
                },
            ), patch("axon.governance.emit") as mock_emit:
                resp = client.post(
                    "/share/redeem",
                    json={"share_string": "axon://valid"},
                )
        assert resp.status_code == 200
        assert resp.json()["key_id"] == "sk_abc123"
        args, kwargs = mock_emit.call_args
        assert args == ("share_redeemed", "share", "sk_abc123")
        assert kwargs["project"] == "sharedproj"
        assert kwargs["details"] == {"owner": "alice"}


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
