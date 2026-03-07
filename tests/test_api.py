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
    api_module.brain.query.assert_called_once_with("What?", filters=filters)


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
    client.post("/search", json={"query": "q", "top_k": 3})
    api_module.brain.vector_store.search.assert_called_once()
    _, kwargs = api_module.brain.vector_store.search.call_args
    assert kwargs.get("top_k") == 3


def test_search_propagates_error():
    api_module.brain = _make_brain()
    api_module.brain.embedding = MagicMock()
    api_module.brain.embedding.embed_query.side_effect = RuntimeError("embed fail")
    resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 500
