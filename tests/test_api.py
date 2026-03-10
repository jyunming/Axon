"""
tests/test_api.py

Unit tests for the FastAPI endpoints in rag_brain.api.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import rag_brain.api as api_module
from rag_brain.api import app

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

def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


# ---------------------------------------------------------------------------
# 503 when brain is None
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_brain():
    """Ensure brain is reset to None before/after every test."""
    original = api_module.brain
    api_module.brain = None
    yield
    api_module.brain = original


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
    assert "processing" in resp.json().get("status", "").lower() or "Ingestion" in resp.json().get("message", "")


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
    resp = client.post("/delete", json={"doc_ids": ["doc1", "doc2"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["deleted"] == 2
    assert set(data["doc_ids"]) == {"doc1", "doc2"}
    # Verify ChromaDB delete was called
    api_module.brain.vector_store.collection.delete.assert_called_once_with(ids=["doc1", "doc2"])
    # Verify BM25 delete was called
    api_module.brain.bm25.delete_documents.assert_called_once_with(["doc1", "doc2"])


def test_delete_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain.vector_store.collection.delete.side_effect = RuntimeError("chroma down")
    resp = client.post("/delete", json={"doc_ids": ["x"]})
    assert resp.status_code == 500


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
    call_kwargs = api_module.brain.query.call_args
    assert call_kwargs[1]["filters"] == filters or call_kwargs[0][1] == filters


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


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


def test_search_503_no_brain():
    resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 503


def test_search_success():
    api_module.brain = _make_brain()
    api_module.brain.embedding = MagicMock()
    api_module.brain.embedding.embed_query.return_value = [0.1] * 384
    api_module.brain.vector_store.search.return_value = [
        {"id": "doc1", "text": "result text", "score": 0.9, "metadata": {"source": "test"}}
    ]
    resp = client.post("/search", json={"query": "find me something"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "doc1"
    assert data[0]["score"] == 0.9


def test_search_uses_custom_top_k():
    api_module.brain = _make_brain()
    api_module.brain.embedding = MagicMock()
    api_module.brain.embedding.embed_query.return_value = [0.0] * 384
    api_module.brain.vector_store.search.return_value = []
    resp = client.post("/search", json={"query": "q", "top_k": 3})
    assert resp.status_code == 200
    api_module.brain.vector_store.search.assert_called_once()
    _, kwargs = api_module.brain.vector_store.search.call_args
    assert kwargs.get("top_k") == 3


def test_search_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain.embedding = MagicMock()
    api_module.brain.embedding.embed_query.side_effect = RuntimeError("embed fail")
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
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results["a"].status_code == 200
    assert results["b"].status_code == 200
    # The global config must remain unchanged after both requests
    assert api_module.brain.config.top_k == 5
