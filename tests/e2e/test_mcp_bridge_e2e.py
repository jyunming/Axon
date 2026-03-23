from __future__ import annotations

import asyncio

import pytest

import axon.mcp_server as mcp_server

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


async def _wait_for_job(job_id: str, attempts: int = 30) -> dict:
    status = {}
    for _ in range(attempts):
        status = await mcp_server.get_job_status(job_id)
        if status["status"] in {"completed", "failed"}:
            return status
        await asyncio.sleep(0.05)
    return status


def test_mcp_tools_bridge_project_text_query_and_settings(live_api_server, make_brain, monkeypatch):
    make_brain()
    base_url = live_api_server()

    monkeypatch.setattr(mcp_server, "API_BASE", base_url)
    monkeypatch.setattr(mcp_server, "API_KEY", None)

    async def _run():
        created = await mcp_server.create_project("mcp_lab", "MCP integration workspace")
        assert created["status"] == "success"

        switched = await mcp_server.switch_project("mcp_lab")
        assert switched["status"] == "success"
        assert switched["active_project"] == "mcp_lab"

        ingested = await mcp_server.ingest_text(
            text="Axon exposes MCP tools over FastMCP for agent workflows.",
            metadata={"source": "mcp_bridge.txt", "topic": "mcp"},
            project="mcp_lab",
        )
        assert ingested["status"] == "success"

        search = await mcp_server.search_knowledge("FastMCP tools", top_k=3)
        assert search
        assert search[0]["metadata"]["source"] == "mcp_bridge.txt"

        answer = await mcp_server.query_knowledge("What exposes MCP tools?", top_k=3)
        assert "FastMCP" in answer["response"]

        settings_before = await mcp_server.get_current_settings()
        assert settings_before["top_k"] == 5
        assert settings_before["hybrid_search"] is True

        settings_after = await mcp_server.update_settings(top_k=7, hybrid_search=False)
        assert settings_after["config"]["top_k"] == 7
        assert settings_after["config"]["hybrid_search"] is False

        listed = await mcp_server.list_knowledge()
        assert listed["total_files"] >= 1
        assert any(item["source"] == "mcp_bridge.txt" for item in listed["files"])

    asyncio.run(_run())


def test_mcp_tools_bridge_path_ingest_and_status_polling(
    live_api_server, make_brain, monkeypatch, sample_docs_dir
):
    make_brain()
    base_url = live_api_server()

    monkeypatch.setattr(mcp_server, "API_BASE", base_url)
    monkeypatch.setattr(mcp_server, "API_KEY", None)

    async def _run():
        job = await mcp_server.ingest_path(str(sample_docs_dir))
        assert job["status"] == "processing"
        assert job["job_id"]

        status = await _wait_for_job(job["job_id"])
        assert status["status"] == "completed"
        assert status["documents_ingested"] == 2

        documents = await mcp_server.list_knowledge()
        assert documents["total_files"] == 2
        assert documents["total_chunks"] >= 2

    asyncio.run(_run())
