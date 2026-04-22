import asyncio

import pytest
from httpx import AsyncClient

import axon.mcp_server as mcp_server

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


@pytest.mark.asyncio
async def test_hybrid_graphrag_pipeline(live_api_server, make_brain, monkeypatch, sample_docs_dir):
    make_brain()
    base_url = live_api_server()
    monkeypatch.setattr(mcp_server, "API_BASE", base_url)

    # 1. Initialize temporary project environment
    created = await mcp_server.create_project("hybrid_lab", "Hybrid RAG Testing")
    assert created["status"] == "success"
    await mcp_server.switch_project("hybrid_lab")

    # 2. Ingest mix of documents
    job = await mcp_server.ingest_path(str(sample_docs_dir))
    assert job["status"] == "processing"

    # Wait for ingestion to complete
    status = {}
    for _ in range(50):
        status = await mcp_server.get_job_status(job["job_id"])
        if status["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.1)

    assert status["status"] == "completed"
    assert status["documents_ingested"] > 0

    # 3. Validate specific query dispatches
    async with AsyncClient(base_url=base_url) as client:
        # Standard query (GraphRAG mode is config-driven, not a per-request type field)
        graph_res = await client.post(
            "/query",
            json={
                "query": "Identify the primary entities and their relationships.",
                "project": "hybrid_lab",
            },
        )
        assert graph_res.status_code == 200
        assert "response" in graph_res.json()

        # Hybrid BM25+vector query via the hybrid override field
        hybrid_res = await client.post(
            "/query",
            json={
                "query": "Summarize the technical implementation of the entities.",
                "project": "hybrid_lab",
                "hybrid": True,
            },
        )
        assert hybrid_res.status_code == 200
        data = hybrid_res.json()
        assert "response" in data
        assert len(data["response"]) > 0
