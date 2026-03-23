"""
tests/test_mcp_server.py

Smoke tests for the Axon MCP stdio server (P2-A).

These tests only verify module structure, tool registration, and the entry point
callable.  They do NOT start an HTTP server or the actual MCP transport — that
would require integration infrastructure.
"""


# Expected tool names as registered on the FastMCP instance
EXPECTED_MCP_TOOL_NAMES = {
    "ingest_text",
    "ingest_texts",
    "ingest_url",
    "ingest_path",
    "get_job_status",
    "search_knowledge",
    "query_knowledge",
    "list_knowledge",
    "switch_project",
    "delete_documents",
    "list_projects",
    "get_stale_docs",
    "create_project",
    "delete_project",
    "clear_knowledge",
    "get_current_settings",
    "update_settings",
    "list_sessions",
    "get_session",
    # AxonStore sharing tools
    "share_project",
    "redeem_share",
    "list_shares",
    "init_store",
    # Operator tooling (Phase 5)
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
    """All 12 expected MCP tools must be registered.

    Validated via the public JSON-RPC ``tools/list`` protocol in
    ``test_mcp_protocol_tools_list`` below — that test is the authoritative
    contract check and avoids depending on FastMCP private internals
    (``_tool_manager._tools``).
    """
    pass


def test_mcp_server_tool_count():
    """Exactly 12 tools must be registered — not more, not fewer.

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


# ---------------------------------------------------------------------------
# Coverage tests: exercise every tool function to reach ≥90% line coverage
# ---------------------------------------------------------------------------


def _make_mock_resp(data: dict):
    from unittest.mock import MagicMock

    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestMcpHeaders:
    """Unit tests for _headers() helper."""

    def test_headers_without_api_key(self, monkeypatch):
        import axon.mcp_server as ms

        monkeypatch.setattr(ms, "API_KEY", None)
        h = ms._headers()
        assert h["Content-Type"] == "application/json"
        assert h["X-Axon-Surface"] == "mcp"
        assert "X-API-Key" not in h

    def test_headers_with_api_key(self, monkeypatch):
        import axon.mcp_server as ms

        monkeypatch.setattr(ms, "API_KEY", "secret-key")
        h = ms._headers()
        assert h["Content-Type"] == "application/json"
        assert h["X-Axon-Surface"] == "mcp"
        assert h["X-API-Key"] == "secret-key"


class TestMcpGetHelper:
    """Unit tests for the _get() coroutine."""

    def test_get_calls_correct_url(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock, patch

        import axon.mcp_server as ms

        monkeypatch.setattr(ms, "API_BASE", "http://localhost:9999")
        mock_resp = _make_mock_resp({"result": "ok"})

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_resp
                result = await ms._get("/collection")
                args, _ = mock_get.call_args
                assert args[0] == "http://localhost:9999/collection"
                assert result == {"result": "ok"}

        asyncio.run(_run())


class TestMcpToolCoverage:
    """Exercise every MCP tool handler to maximise line coverage."""

    # ------------------------------------------------------------------
    # ingest_text
    # ------------------------------------------------------------------
    def test_ingest_text_no_optional_fields(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_text

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "created", "doc_id": "abc"})
                result = await ingest_text(text="plain text")
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["text"] == "plain text"
                assert "metadata" not in kwargs["json"]
                assert "project" not in kwargs["json"]
                assert result["status"] == "created"

        asyncio.run(_run())

    def test_ingest_text_with_metadata_and_project(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_text

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "created", "doc_id": "xyz"})
                await ingest_text(text="hello", metadata={"source": "wiki"}, project="research")
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["metadata"] == {"source": "wiki"}
                assert kwargs["json"]["project"] == "research"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # ingest_texts
    # ------------------------------------------------------------------
    def test_ingest_texts_batch(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_texts

        docs = [{"text": "doc1"}, {"text": "doc2"}]

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp(
                    [{"status": "created"}, {"status": "created"}]
                )
                result = await ingest_texts(docs=docs, project="eng")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/add_texts")
                assert kwargs["json"]["docs"] == docs
                assert kwargs["json"]["project"] == "eng"
                assert isinstance(result, list)

        asyncio.run(_run())

    def test_ingest_texts_no_project(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_texts

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp([])
                await ingest_texts(docs=[{"text": "x"}])
                _, kwargs = mock_post.call_args
                assert "project" not in kwargs["json"]

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # ingest_url
    # ------------------------------------------------------------------
    def test_ingest_url_basic(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_url

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp(
                    {"status": "ingested", "url": "http://example.com"}
                )
                result = await ingest_url(url="http://example.com")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/ingest_url")
                assert kwargs["json"]["url"] == "http://example.com"
                assert "metadata" not in kwargs["json"]
                assert result["status"] == "ingested"

        asyncio.run(_run())

    def test_ingest_url_with_metadata_and_project(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_url

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "ingested"})
                await ingest_url(url="http://x.com", metadata={"tag": "v"}, project="proj")
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["metadata"] == {"tag": "v"}
                assert kwargs["json"]["project"] == "proj"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # ingest_path
    # ------------------------------------------------------------------
    def test_ingest_path(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import ingest_path

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "processing", "job_id": "j1"})
                result = await ingest_path(path="/tmp/docs")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/ingest")
                assert kwargs["json"]["path"] == "/tmp/docs"
                assert result["job_id"] == "j1"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # get_job_status
    # ------------------------------------------------------------------
    def test_get_job_status(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import get_job_status

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"status": "completed", "job_id": "j42"})
                result = await get_job_status(job_id="j42")
                args, _ = mock_get.call_args
                assert "/ingest/status/j42" in args[0]
                assert result["status"] == "completed"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # search_knowledge
    # ------------------------------------------------------------------
    def test_search_knowledge_default_top_k(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import search_knowledge

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp([{"text": "chunk1"}])
                result = await search_knowledge(query="what is RAG?")
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["query"] == "what is RAG?"
                assert kwargs["json"]["top_k"] == 5
                assert "filters" not in kwargs["json"]
                assert isinstance(result, list)

        asyncio.run(_run())

    def test_search_knowledge_with_filters(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import search_knowledge

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp([])
                await search_knowledge(query="q", top_k=10, filters={"source": "wiki"})
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["top_k"] == 10
                assert kwargs["json"]["filters"] == {"source": "wiki"}

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # query_knowledge
    # ------------------------------------------------------------------
    def test_query_knowledge_minimal(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import query_knowledge

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"response": "answer", "sources": []})
                result = await query_knowledge(query="What is Axon?")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/query")
                assert kwargs["json"]["query"] == "What is Axon?"
                assert "top_k" not in kwargs["json"]
                assert "filters" not in kwargs["json"]
                assert result["response"] == "answer"

        asyncio.run(_run())

    def test_query_knowledge_with_top_k_and_filters(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import query_knowledge

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"response": "ok"})
                await query_knowledge(query="q", top_k=20, filters={"tag": "a"})
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["top_k"] == 20
                assert kwargs["json"]["filters"] == {"tag": "a"}

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # list_knowledge
    # ------------------------------------------------------------------
    def test_list_knowledge(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import list_knowledge

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"sources": [], "total_chunks": 0})
                result = await list_knowledge()
                args, _ = mock_get.call_args
                assert args[0].endswith("/collection")
                assert "total_chunks" in result

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # switch_project
    # ------------------------------------------------------------------
    def test_switch_project(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import switch_project

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"active_project": "eng"})
                result = await switch_project(project_name="eng")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/project/switch")
                assert kwargs["json"]["project_name"] == "eng"
                assert result["active_project"] == "eng"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # delete_documents
    # ------------------------------------------------------------------
    def test_delete_documents(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import delete_documents

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"deleted": 2})
                result = await delete_documents(doc_ids=["id1", "id2"])
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/delete")
                assert kwargs["json"]["doc_ids"] == ["id1", "id2"]
                assert result["deleted"] == 2

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # list_projects
    # ------------------------------------------------------------------
    def test_list_projects(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import list_projects

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"projects": ["default", "eng"]})
                result = await list_projects()
                args, _ = mock_get.call_args
                assert args[0].endswith("/projects")
                assert "projects" in result

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # get_stale_docs
    # ------------------------------------------------------------------
    def test_get_stale_docs_default_days(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import get_stale_docs

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"stale": []})
                await get_stale_docs()
                args, _ = mock_get.call_args
                assert "days=7" in args[0]

        asyncio.run(_run())

    def test_get_stale_docs_custom_days(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import get_stale_docs

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"stale": []})
                await get_stale_docs(days=30)
                args, _ = mock_get.call_args
                assert "days=30" in args[0]

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # create_project
    # ------------------------------------------------------------------
    def test_create_project_with_description(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import create_project

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "created", "project": "myproj"})
                result = await create_project(name="myproj", description="test project")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/project/new")
                assert kwargs["json"]["name"] == "myproj"
                assert kwargs["json"]["description"] == "test project"
                assert result["project"] == "myproj"

        asyncio.run(_run())

    def test_create_project_no_description(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import create_project

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "created", "project": "proj2"})
                await create_project(name="proj2")
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["description"] == ""

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # delete_project
    # ------------------------------------------------------------------
    def test_delete_project(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import delete_project

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "deleted"})
                result = await delete_project(name="old-proj")
                args, _ = mock_post.call_args
                assert "/project/delete/old-proj" in args[0]
                assert result["status"] == "deleted"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # clear_knowledge
    # ------------------------------------------------------------------
    def test_clear_knowledge(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import clear_knowledge

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "cleared"})
                result = await clear_knowledge()
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/clear")
                assert kwargs["json"] == {}
                assert result["status"] == "cleared"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # get_current_settings
    # ------------------------------------------------------------------
    def test_get_current_settings(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import get_current_settings

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"top_k": 5, "hybrid_search": True})
                result = await get_current_settings()
                args, _ = mock_get.call_args
                assert args[0].endswith("/config")
                assert result["top_k"] == 5

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # update_settings
    # ------------------------------------------------------------------
    def test_update_settings_sends_only_set_fields(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import update_settings

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "ok"})
                await update_settings(top_k=10, rerank=True)
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/config/update")
                # Only fields that were set should appear (None fields excluded)
                body = kwargs["json"]
                assert body["top_k"] == 10
                assert body["rerank"] is True
                # Fields not supplied must not appear
                assert "hyde" not in body
                assert "multi_query" not in body

        asyncio.run(_run())

    def test_update_settings_all_none_sends_empty_body(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import update_settings

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp({"status": "ok"})
                await update_settings()
                _, kwargs = mock_post.call_args
                # body should be empty when no args are set
                assert kwargs["json"] == {}

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # list_sessions
    # ------------------------------------------------------------------
    def test_list_sessions(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import list_sessions

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"sessions": ["s1", "s2"]})
                result = await list_sessions()
                args, _ = mock_get.call_args
                assert args[0].endswith("/sessions")
                assert result["sessions"] == ["s1", "s2"]

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # get_session
    # ------------------------------------------------------------------
    def test_get_session(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import get_session

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"session_id": "ses99", "messages": []})
                result = await get_session(session_id="ses99")
                args, _ = mock_get.call_args
                assert "/session/ses99" in args[0]
                assert result["session_id"] == "ses99"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # share_project (AxonStore MCP tool)
    # ------------------------------------------------------------------
    def test_share_project_mcp(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import share_project

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp(
                    {"share_string": "AXON.abcd", "key_id": "k1"}
                )
                result = await share_project(project="myproj", grantee="bob")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/share/generate")
                assert kwargs["json"]["project"] == "myproj"
                assert kwargs["json"]["grantee"] == "bob"
                assert "write_access" not in kwargs["json"]
                assert result["share_string"] == "AXON.abcd"

        asyncio.run(_run())

    def test_share_project_write_access(self):
        """write_access is always False — the flag is fully removed from the tool."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import share_project

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp(
                    {"share_string": "AXON.xyz", "key_id": "k2"}
                )
                await share_project(project="p", grantee="carol")
                _, kwargs = mock_post.call_args
                assert "write_access" not in kwargs["json"]

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # redeem_share
    # ------------------------------------------------------------------
    def test_redeem_share_mcp(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import redeem_share

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp(
                    {"status": "redeemed", "mount_name": "alice_myproj"}
                )
                result = await redeem_share(share_string="AXON.efgh")
                args, kwargs = mock_post.call_args
                assert args[0].endswith("/share/redeem")
                assert kwargs["json"]["share_string"] == "AXON.efgh"
                assert result["status"] == "redeemed"

        asyncio.run(_run())

    # ------------------------------------------------------------------
    # list_shares
    # ------------------------------------------------------------------
    def test_list_shares_mcp(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import list_shares

        async def _run():
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = _make_mock_resp({"sharing": [], "shared": []})
                result = await list_shares()
                args, _ = mock_get.call_args
                assert args[0].endswith("/share/list")
                assert "sharing" in result
                assert "shared" in result

        asyncio.run(_run())

    # init_store
    # ------------------------------------------------------------------
    def test_init_store_mcp(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.mcp_server import init_store

        async def _run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = _make_mock_resp(
                    {"status": "ok", "store_path": "/data/AxonStore", "username": "alice"}
                )
                result = await init_store(base_path="/data")
                _, kwargs = mock_post.call_args
                assert kwargs["json"]["base_path"] == "/data"
                assert result["status"] == "ok"

        asyncio.run(_run())
