from __future__ import annotations

import pytest

import axon.main as main_module

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_config_sessions_and_copilot_agent_query_flow(
    api_client, make_brain, monkeypatch, tmp_path
):
    make_brain()

    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(main_module, "_SESSIONS_DIR", str(sessions_dir))

    add_doc = api_client.post(
        "/add_text",
        json={
            "text": "GraphRAG adds entity-centric expansion for connected concepts.",
            "metadata": {"source": "graph_note.txt", "topic": "graph"},
            "doc_id": "graph_note",
        },
    )
    assert add_doc.status_code == 200

    config_before = api_client.get("/config")
    assert config_before.status_code == 200
    assert config_before.json()["top_k"] == 5
    assert config_before.json()["hybrid_search"] is True

    config_update = api_client.post(
        "/config/update",
        json={
            "top_k": 2,
            "hybrid_search": False,
            "discussion_fallback": False,
            "persist": False,
        },
    )
    assert config_update.status_code == 200
    updated = config_update.json()
    assert updated["status"] == "success"
    assert updated["persisted"] is False
    assert updated["config"]["top_k"] == 2
    assert updated["config"]["hybrid_search"] is False
    assert updated["config"]["discussion_fallback"] is False

    session = {
        "id": "sess-e2e-001",
        "started_at": "2026-03-19T00:00:00+00:00",
        "provider": "ollama",
        "model": "fake-llm",
        "project": "default",
        "history": [
            {"role": "user", "content": "What does GraphRAG add?"},
            {"role": "assistant", "content": "Entity-centric expansion."},
        ],
    }
    main_module._save_session(session)

    sessions = api_client.get("/sessions")
    assert sessions.status_code == 200
    session_ids = {item["id"] for item in sessions.json()["sessions"]}
    assert "sess-e2e-001" in session_ids

    loaded = api_client.get("/session/sess-e2e-001")
    assert loaded.status_code == 200
    assert loaded.json()["history"][0]["content"] == "What does GraphRAG add?"

    copilot = api_client.post(
        "/copilot/agent",
        json={
            "agent_request_id": "req-e2e-1",
            "messages": [{"role": "user", "content": "What does GraphRAG add?"}],
        },
    )
    assert copilot.status_code == 200
    body = copilot.text
    assert '"type": "created"' in body
    assert "GraphRAG adds entity-centric expansion" in body
    assert "[DONE]" in body


def test_copilot_agent_slash_commands_expose_search_and_projects(api_client, make_brain):
    make_brain()

    create_project = api_client.post(
        "/project/new", json={"name": "research", "description": "Research notes"}
    )
    assert create_project.status_code == 200

    api_client.post(
        "/add_text",
        json={
            "text": "Axon exposes MCP tools over FastMCP for agent mode integrations.",
            "metadata": {"source": "mcp_note.txt", "topic": "mcp"},
            "doc_id": "mcp_note",
        },
    )

    projects = api_client.post(
        "/copilot/agent",
        json={
            "agent_request_id": "req-e2e-2",
            "messages": [{"role": "user", "content": "/projects"}],
        },
    )
    assert projects.status_code == 200
    assert "Available Axon Projects" in projects.text
    assert "research" in projects.text

    search = api_client.post(
        "/copilot/agent",
        json={
            "agent_request_id": "req-e2e-3",
            "messages": [{"role": "user", "content": "/search FastMCP"}],
        },
    )
    assert search.status_code == 200
    assert "Search Results" in search.text
    assert "mcp_note.txt" in search.text
