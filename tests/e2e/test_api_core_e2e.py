from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_core_query_search_stream_and_collection(api_client, make_brain):
    make_brain()

    first = api_client.post(
        "/add_text",
        json={
            "text": "Axon uses BM25 and vector search to retrieve grounded context.",
            "metadata": {"source": "retrieval_note.txt", "topic": "retrieval"},
            "doc_id": "retrieval_note",
        },
    )
    second = api_client.post(
        "/add_text",
        json={
            "text": "GraphRAG adds entity-centric expansion for connected concepts.",
            "metadata": {"source": "graph_note.txt", "topic": "graph"},
            "doc_id": "graph_note",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200

    health = api_client.get("/health")
    assert health.status_code == 200
    assert health.json()["project"] == "default"

    search = api_client.post("/search", json={"query": "Which technique uses BM25?", "top_k": 3})
    assert search.status_code == 200
    search_results = search.json()
    assert search_results
    assert search_results[0]["metadata"]["source"] == "retrieval_note.txt"

    raw = api_client.post("/search/raw", json={"query": "entity expansion", "top_k": 3})
    assert raw.status_code == 200
    raw_payload = raw.json()
    assert raw_payload["results"]
    assert raw_payload["diagnostics"]["result_count"] >= 1

    query = api_client.post(
        "/query",
        json={
            "query": "How does Axon retrieve context?",
            "include_diagnostics": True,
        },
    )
    assert query.status_code == 200
    query_payload = query.json()
    assert "BM25" in query_payload["response"]
    assert query_payload["diagnostics"]["result_count"] >= 1

    stream = api_client.post("/query/stream", json={"query": "How does Axon retrieve context?"})
    assert stream.status_code == 200
    stream_text = stream.text
    assert '"type": "sources"' in stream_text
    assert "BM25" in stream_text

    collection = api_client.get("/collection")
    assert collection.status_code == 200
    collection_payload = collection.json()
    assert collection_payload["total_files"] == 2
    assert collection_payload["total_chunks"] >= 2


def test_batch_ingest_tracked_docs_delete_clear_and_stale(api_client, make_brain):
    make_brain()

    batch = api_client.post(
        "/add_texts",
        json={
            "docs": [
                {
                    "doc_id": "agent_runbook",
                    "text": "The runbook explains how to reindex Axon after a schema update.",
                    "metadata": {"source": "runbook.txt"},
                },
                {
                    "doc_id": "ops_checklist",
                    "text": "Operators should clear stale indexes before a full rebuild.",
                    "metadata": {"source": "ops.txt"},
                },
            ]
        },
    )
    assert batch.status_code == 200
    batch_payload = batch.json()
    assert {item["status"] for item in batch_payload} == {"created"}

    tracked = api_client.get("/tracked-docs")
    assert tracked.status_code == 200
    tracked_docs = tracked.json()["docs"]
    assert "runbook.txt" in tracked_docs
    assert "ops.txt" in tracked_docs

    stale = api_client.get("/collection/stale?days=7")
    assert stale.status_code == 200
    assert stale.json()["total"] == 0

    delete = api_client.post("/delete", json={"doc_ids": ["agent_runbook_chunk_0"]})
    assert delete.status_code == 200
    assert delete.json()["deleted"] == 1

    collection_after_delete = api_client.get("/collection").json()
    assert collection_after_delete["total_files"] == 1

    clear = api_client.post("/clear")
    assert clear.status_code == 200
    assert clear.json()["status"] == "success"

    collection_after_clear = api_client.get("/collection").json()
    assert collection_after_clear["total_files"] == 0
    assert collection_after_clear["total_chunks"] == 0


def test_path_ingest_and_refresh_detects_changes(api_client, make_brain, sample_docs_dir: Path):
    make_brain()

    ingest = api_client.post("/ingest", json={"path": str(sample_docs_dir)})
    assert ingest.status_code == 200
    job_id = ingest.json()["job_id"]

    status = api_client.get(f"/ingest/status/{job_id}")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["status"] == "completed"
    assert status_payload["documents_ingested"] == 2

    tracked = api_client.get("/tracked-docs")
    tracked_docs = tracked.json()["docs"]
    assert str(sample_docs_dir / "overview.txt") in tracked_docs

    updated_path = sample_docs_dir / "overview.txt"
    updated_path.write_text(
        "Axon uses BM25, vector search, and reranking to retrieve grounded context.",
        encoding="utf-8",
    )

    refresh = api_client.post("/ingest/refresh")
    assert refresh.status_code == 200
    refresh_payload = refresh.json()
    assert str(updated_path) in refresh_payload["reingested"]

    query = api_client.post("/query", json={"query": "What does Axon use to retrieve context?"})
    assert query.status_code == 200
    assert "reranking" in query.json()["response"]
