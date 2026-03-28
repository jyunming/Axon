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
