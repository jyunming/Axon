import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio
async def test_session_context_management(live_api_server, make_brain):
    make_brain()
    base_url = live_api_server()

    async with AsyncClient(base_url=base_url) as client:
        # Verify session listing endpoint is reachable and returns expected shape.
        # The Axon session API exposes GET /sessions (list) and GET /session/{id} (fetch).
        # Multi-turn context is managed inside the brain; there is no POST /sessions
        # REST endpoint in v1 — see docs/API_REFERENCE.md.
        sessions_res = await client.get("/sessions")
        assert sessions_res.status_code == 200
        payload = sessions_res.json()
        assert "sessions" in payload
        assert isinstance(payload["sessions"], list)

        # Run a query so a session may be persisted by the brain
        query_res = await client.post(
            "/query",
            json={"query": "What is the core framework used in this project?"},
        )
        assert query_res.status_code == 200
        assert "response" in query_res.json()

        # After a query the session list should still return a valid structure
        sessions_after = await client.get("/sessions")
        assert sessions_after.status_code == 200
        assert "sessions" in sessions_after.json()

        # If any sessions were persisted, validate that GET /session/{id} works
        sessions = sessions_after.json()["sessions"]
        if sessions:
            session_id = sessions[0].get("id") or sessions[0].get("session_id")
            if session_id:
                detail_res = await client.get(f"/session/{session_id}")
                assert detail_res.status_code == 200
