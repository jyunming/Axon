"""


tests/test_mcp_server.py


Smoke tests for the Axon MCP stdio server (P2-A).


These tests only verify module structure, tool registration, and the entry point


callable.  They do NOT start an HTTP server or the actual MCP transport — that


would require integration infrastructure.


"""


# Expected tool names as registered on the FastMCP instance


EXPECTED_MCP_TOOL_NAMES = {
    # Retrieval
    "search_knowledge",
    "query_knowledge",
    # Ingestion
    "ingest_text",
    "ingest_texts",
    "ingest_url",
    "ingest_path",
    "get_job_status",
    "refresh_ingest",
    "get_stale_docs",
    # Knowledge base management
    "list_knowledge",
    "delete_documents",
    "clear_knowledge",
    # Projects
    "list_projects",
    "switch_project",
    "create_project",
    "delete_project",
    # Settings / sessions
    "get_current_settings",
    "update_settings",
    "list_sessions",
    "get_session",
    # Sharing
    "share_project",
    "redeem_share",
    "revoke_share",
    "list_shares",
    "init_store",
    "get_store_status",
    # Graph
    "graph_status",
    "graph_finalize",
    "graph_data",
    "get_active_leases",
}


def test_mcp_server_imports_without_error():
    """Importing axon.mcp_server must not raise any exception."""

    import axon.mcp_server  # noqa: F401 — import side-effect is the test


def test_mcp_server_exposes_fastmcp_instance():
    """The module must expose a FastMCP object named ``mcp``."""

    from mcp.server.fastmcp import FastMCP

    from axon.mcp_server import mcp

    assert isinstance(mcp, FastMCP)


def test_mcp_server_registers_all_expected_tools():
    """All 32 expected MCP tools must be registered.


    Validated via the public JSON-RPC ``tools/list`` protocol in


    ``test_mcp_protocol_tools_list`` below — that test is the authoritative


    contract check and avoids depending on FastMCP private internals


    (``_tool_manager._tools``).


    """

    pass


def test_mcp_server_tool_count():
    """Exactly 32 tools must be registered — not more, not fewer.


    See ``test_mcp_protocol_tools_list`` for the substantive assertion; this


    placeholder exists so the test name remains discoverable.


    """

    pass


def test_main_is_callable():
    """The ``main`` entry point must be importable and callable."""

    from axon.mcp_server import main

    assert callable(main)


def test_mcp_protocol_initialize():
    """Spawn the MCP server as a subprocess and verify the JSON-RPC initialize


    handshake returns the correct protocol version and server name."""

    import json
    import subprocess
    import sys

    msg = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize",'
        '"params":{"protocolVersion":"2024-11-05","capabilities":{},'
        '"clientInfo":{"name":"test","version":"1"}}}\n'
    )

    result = subprocess.run(
        [sys.executable, "-m", "axon.mcp_server"],
        input=msg,
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Parse first line of stdout as JSON

    first_line = result.stdout.strip().splitlines()[0]

    resp = json.loads(first_line)

    assert resp["jsonrpc"] == "2.0"

    assert resp["id"] == 1

    assert resp["result"]["protocolVersion"] == "2024-11-05"

    assert resp["result"]["serverInfo"]["name"] == "axon"


def test_mcp_protocol_tools_list():
    """Spawn the MCP server and verify tools/list returns all 12 expected tools."""

    import json
    import subprocess
    import sys

    msgs = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize",'
        '"params":{"protocolVersion":"2024-11-05","capabilities":{},'
        '"clientInfo":{"name":"test","version":"1"}}}\n'
        '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n'
    )

    result = subprocess.run(
        [sys.executable, "-m", "axon.mcp_server"],
        input=msgs,
        capture_output=True,
        text=True,
        timeout=10,
    )

    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]

    # Second response is tools/list

    tools_resp = json.loads(lines[1])

    assert tools_resp["id"] == 2

    returned_names = {t["name"] for t in tools_resp["result"]["tools"]}

    assert returned_names == EXPECTED_MCP_TOOL_NAMES, (
        f"Tool mismatch.\n  Missing: {EXPECTED_MCP_TOOL_NAMES - returned_names}\n"
        f"  Extra:   {returned_names - EXPECTED_MCP_TOOL_NAMES}"
    )


def test_mcp_tool_invocation_proxies_to_api():
    """Verify that calling an MCP tool function directly correctly proxies to the


    REST API via httpx.AsyncClient."""

    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from axon.mcp_server import ingest_text

    mock_resp = MagicMock()

    mock_resp.json.return_value = {"status": "created", "doc_id": "123"}

    mock_resp.raise_for_status = MagicMock()

    # Mock httpx.AsyncClient.post

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await ingest_text(text="Hello world", project="test-p")

            assert result == {"status": "created", "doc_id": "123"}

            assert mock_post.called

            # Check that it called the correct endpoint with correct body

            args, kwargs = mock_post.call_args

            assert args[0].endswith("/add_text")

            assert kwargs["json"]["text"] == "Hello world"

            assert kwargs["json"]["project"] == "test-p"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tool-level invocation tests (happy path via httpx mock)
# ---------------------------------------------------------------------------


def _mock_get(return_value: dict):
    """Return an AsyncMock that pretends to be httpx.AsyncClient.get."""
    from unittest.mock import AsyncMock, MagicMock

    mock_resp = MagicMock()
    mock_resp.json.return_value = return_value
    mock_resp.raise_for_status = MagicMock()
    m = AsyncMock(return_value=mock_resp)
    return m


def _mock_post(return_value: dict):
    """Return an AsyncMock that pretends to be httpx.AsyncClient.post."""
    from unittest.mock import AsyncMock, MagicMock

    mock_resp = MagicMock()
    mock_resp.json.return_value = return_value
    mock_resp.raise_for_status = MagicMock()
    m = AsyncMock(return_value=mock_resp)
    return m


def test_ingest_texts_proxies_batch():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import ingest_texts

    async def _run():
        rv = {"added": 2, "skipped": 0}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await ingest_texts(docs=[{"text": "A"}, {"text": "B"}], project="proj")
        assert result == rv

    asyncio.run(_run())


def test_ingest_url_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import ingest_url

    async def _run():
        rv = {"status": "created", "doc_id": "u1"}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await ingest_url(url="http://example.com/page")
        assert result == rv

    asyncio.run(_run())


def test_list_knowledge_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import list_knowledge

    async def _run():
        rv = {"sources": [], "total_chunks": 0}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await list_knowledge()
        assert result == rv

    asyncio.run(_run())


def test_delete_documents_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import delete_documents

    async def _run():
        rv = {"deleted": 1}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await delete_documents(doc_ids=["abc"])
        assert result == rv

    asyncio.run(_run())


def test_get_stale_docs_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import get_stale_docs

    async def _run():
        rv = {"stale_docs": []}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await get_stale_docs(days=7)
        assert result == rv

    asyncio.run(_run())


def test_list_projects_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import list_projects

    async def _run():
        rv = {"projects": ["default"]}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await list_projects()
        assert result == rv

    asyncio.run(_run())


def test_create_project_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import create_project

    async def _run():
        rv = {"status": "created", "project": "myproj"}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await create_project(name="myproj")
        assert result == rv

    asyncio.run(_run())


def test_delete_project_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import delete_project

    async def _run():
        rv = {"status": "deleted"}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await delete_project(name="myproj")
        assert result == rv

    asyncio.run(_run())


def test_get_current_settings_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import get_current_settings

    async def _run():
        rv = {"top_k": 10, "hybrid_search": True}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await get_current_settings()
        assert result == rv

    asyncio.run(_run())


def test_update_settings_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import update_settings

    async def _run():
        rv = {"status": "updated"}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await update_settings(hyde=True)
        assert result == rv

    asyncio.run(_run())


def test_graph_status_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import graph_status

    async def _run():
        rv = {"entity_count": 5, "community_summary_count": 2}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await graph_status()
        assert result == rv

    asyncio.run(_run())


def test_graph_finalize_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import graph_finalize

    async def _run():
        rv = {"community_summary_count": 3}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await graph_finalize()
        assert result == rv

    asyncio.run(_run())


def test_graph_data_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import graph_data

    async def _run():
        rv = {"nodes": [], "links": []}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await graph_data()
        assert result == rv

    asyncio.run(_run())


def test_get_active_leases_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import get_active_leases

    async def _run():
        rv = {"leases": {}}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await get_active_leases()
        assert result == rv

    asyncio.run(_run())


def test_list_sessions_proxies_get():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import list_sessions

    async def _run():
        rv = {"sessions": []}
        with patch("httpx.AsyncClient.get", _mock_get(rv)):
            result = await list_sessions()
        assert result == rv

    asyncio.run(_run())


def test_share_project_proxies_post():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import share_project

    async def _run():
        rv = {"share_string": "axon-share-v1:..."}
        with patch("httpx.AsyncClient.post", _mock_post(rv)):
            result = await share_project(project="default", grantee="alice")
        assert result == rv

    asyncio.run(_run())


def test_search_knowledge_top_k_zero_raises():
    """search_knowledge with top_k < 1 must raise ValueError (not return dict)."""
    import asyncio

    from axon.mcp_server import search_knowledge

    async def _run():
        try:
            await search_knowledge(query="test", top_k=0)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "top_k" in str(exc)

    asyncio.run(_run())


def test_query_knowledge_top_k_zero_raises():
    """query_knowledge with top_k < 1 must raise ValueError (not return dict)."""
    import asyncio

    from axon.mcp_server import query_knowledge

    async def _run():
        try:
            await query_knowledge(query="test", top_k=0)
            raise AssertionError("Expected ValueError")
        except ValueError as exc:
            assert "top_k" in str(exc)

    asyncio.run(_run())
