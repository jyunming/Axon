"""
tests/test_api.py

Unit tests for the FastAPI endpoints in axon.api.
"""

import os
from unittest.mock import MagicMock, patch

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
    brain._entity_graph = {"entity": ["neighbor"]}

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
    assert "[ERROR]" in resp.text


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
    for bad_name in ["has:colon!", "", "name with spaces", "a" * 65, "too/many/slash/parts"]:
        resp = client.post("/project/new", json={"name": bad_name})
        assert resp.status_code == 400, f"Expected 400 for name={bad_name!r}"


def test_create_project_valid_name_succeeds():
    """B-03: Valid project names pass name validation, including hierarchical names."""
    api_module.brain = _make_brain()
    with patch("axon.projects.ensure_project"):
        for good_name in ["valid-project_01", "research/papers", "research/papers/2024"]:
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

    resp1 = client.post("/add_text", json={"text": text, "project": "alpha"})
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "success"

    resp2 = client.post("/add_text", json={"text": text, "project": "beta"})
    assert resp2.status_code == 200
    # Different project — must NOT be skipped
    assert resp2.json()["status"] == "success"
    assert api_module.brain.ingest.call_count == 2


def test_add_text_same_content_same_project_is_skipped():
    """Same text sent twice to the same project must be skipped on the second call."""
    api_module.brain = _make_brain()
    api_module.brain.ingest.return_value = None

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

    client.post("/add_text", json={"text": "project test doc", "project": "sprint3-test"})

    resp = client.get("/projects")
    assert resp.status_code == 200
    data = resp.json()
    memory_names = [p["name"] for p in data["memory_only"]]
    assert "sprint3-test" in memory_names


# ---------------------------------------------------------------------------
# /store/whoami
# ---------------------------------------------------------------------------


def test_store_whoami_no_store_mode():
    """whoami returns store_mode=False when brain is not in store mode."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = False
    api_module.brain = mock_brain

    resp = client.get("/store/whoami")
    assert resp.status_code == 200
    data = resp.json()
    assert data["store_mode"] is False
    assert "username" in data


def test_store_whoami_store_mode_active():
    """whoami returns store paths when axon_store_mode is True."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    mock_brain.config.projects_root = "/data/AxonStore/alice"
    mock_brain.config.project = "_default"
    api_module.brain = mock_brain

    resp = client.get("/store/whoami")
    assert resp.status_code == 200
    data = resp.json()
    assert data["store_mode"] is True
    assert "user_dir" in data
    assert "username" in data


# ---------------------------------------------------------------------------
# /share/list  (requires store mode)
# ---------------------------------------------------------------------------


def test_share_list_503_no_store_mode():
    """/share/list returns 503 when axon_store_mode is off."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = False
    api_module.brain = mock_brain

    resp = client.get("/share/list")
    assert resp.status_code == 503


def test_share_list_returns_sharing_and_shared(tmp_path):
    """/share/list returns both sharing and shared keys."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    mock_brain.config.projects_root = str(tmp_path)
    api_module.brain = mock_brain

    # Create minimal dir structure expected by _get_user_dir
    (tmp_path / "ShareMount").mkdir()
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
        json={"project": "nonexistent", "grantee": "bob", "write_access": False},
    )
    assert resp.status_code == 404


def test_share_generate_success(tmp_path):
    """/share/generate returns share string when project exists."""
    mock_brain = _make_brain()
    mock_brain.config.axon_store_mode = True
    mock_brain.config.projects_root = str(tmp_path)
    api_module.brain = mock_brain

    # Create a real project dir so the endpoint can find it
    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text('{"name": "myproject"}')

    fake_result = {
        "key_id": "sk_abc123",
        "share_string": "abc:def:alice:myproject:/data/store",
        "project": "myproject",
        "grantee": "bob",
        "write_access": False,
        "owner": "alice",
    }

    with patch("axon.api._shares.generate_share_key", return_value=fake_result):
        resp = client.post(
            "/share/generate",
            json={"project": "myproject", "grantee": "bob", "write_access": False},
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

    (tmp_path / "ShareMount").mkdir()
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

    (tmp_path / "ShareMount").mkdir()
    (tmp_path / ".shares").mkdir()

    with patch(
        "axon.api._shares.revoke_share_key", return_value={"status": "revoked", "key_id": "sk_abc"}
    ):
        resp = client.post("/share/revoke", json={"key_id": "sk_abc"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "revoked"
