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
    "query_knowledge",
    "search_knowledge",
    # Ingest (text | docs | url | path | refresh in one tool) + job polling
    "ingest_knowledge",
    "get_job_status",
    # Collection
    "list_knowledge",
    "delete_documents",
    # Projects
    "list_projects",
    "switch_project",
    "create_project",
    # Configuration
    "get_config",
    "set_config",
    # Graph
    "graph_retrieve",
    "update_fact",
    # Sharing (stays agent-callable; hard revoke / key rotation is human-only)
    "share_project",
    "redeem_share",
    "list_shares",
    "revoke_share",
    "extend_share",
}

# Removed from the agent surface in 0.5.0 (human-only now: REST / CLI / REPL).
REMOVED_MCP_TOOL_NAMES = {
    "ingest_text",
    "ingest_texts",
    "ingest_url",
    "ingest_path",
    "refresh_ingest",
    "get_stale_docs",
    "clear_knowledge",
    "delete_project",
    "get_current_settings",
    "update_settings",
    "update_config",
    "validate_config",
    "list_sessions",
    "get_session",
    "init_store",
    "get_store_status",
    "security_status",
    "security_bootstrap",
    "security_unlock",
    "security_lock",
    "security_change_passphrase",
    "suggest_passphrase",
    "set_keyring_mode",
    "wipe_sealed_cache",
    "seal_project",
    "pack_project",
    "unpack_project",
    "graph_status",
    "graph_finalize",
    "graph_data",
    "graph_backend_status",
    "graph_conflicts",
    "get_active_leases",
    "query_stream",
    "mount_refresh",
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
    """All 18 expected MCP tools must be registered.


    Validated via the public JSON-RPC ``tools/list`` protocol in


    ``test_mcp_protocol_tools_list`` below — that test is the authoritative


    contract check and avoids depending on FastMCP private internals


    (``_tool_manager._tools``).


    """

    pass


def test_mcp_server_tool_count():
    """Exactly 18 tools must be registered — not more, not fewer.


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
        timeout=30,
    )

    # Parse first line of stdout as JSON

    first_line = result.stdout.strip().splitlines()[0]

    resp = json.loads(first_line)

    assert resp["jsonrpc"] == "2.0"

    assert resp["id"] == 1

    assert resp["result"]["protocolVersion"] == "2024-11-05"

    assert resp["result"]["serverInfo"]["name"] == "axon"


def test_mcp_protocol_tools_list():
    """Spawn the MCP server and verify tools/list returns all expected tools.

    Uses interactive Popen communication (write → read → write → read) so each
    response is received before the next request is sent.  Sending all messages
    at once via subprocess.run() can race: the server may exit before flushing
    the tools/list response on some platforms / Python versions.
    """

    import json
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-m", "axon.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # 1. Send initialize and read its response.
        init_msg = (
            '{"jsonrpc":"2.0","id":1,"method":"initialize",'
            '"params":{"protocolVersion":"2024-11-05","capabilities":{},'
            '"clientInfo":{"name":"test","version":"1"}}}\n'
        )
        proc.stdin.write(init_msg)
        proc.stdin.flush()
        init_line = proc.stdout.readline()

        # 2. Send notifications/initialized (no response expected).
        proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n')
        proc.stdin.flush()

        # 3. Send tools/list and read its response.
        proc.stdin.write('{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n')
        proc.stdin.flush()
        tools_line = proc.stdout.readline()

        # Close stdin so the server can exit cleanly.
        proc.stdin.close()
        proc.wait(timeout=10)

    except Exception:
        proc.kill()
        proc.wait()
        raise

    stderr_output = proc.stderr.read()

    assert init_line.strip(), f"No initialize response.\nstderr: {stderr_output}"
    assert (
        tools_line.strip()
    ), f"No tools/list response.\ninit_line: {init_line!r}\nstderr: {stderr_output}"

    tools_resp = json.loads(tools_line.strip())

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

    from axon.mcp_server import ingest_knowledge

    mock_resp = MagicMock()

    mock_resp.json.return_value = {"status": "created", "doc_id": "123"}

    mock_resp.raise_for_status = MagicMock()

    # Mock httpx.AsyncClient.post

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            result = await ingest_knowledge(text="Hello world", project="test-p")

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


def test_removed_tools_are_not_module_attributes():
    """The dropped tools must be gone from the module, not just unregistered —
    a leftover coroutine is a trap for anyone re-adding a decorator."""
    import axon.mcp_server as mod

    leftovers = sorted(n for n in REMOVED_MCP_TOOL_NAMES if hasattr(mod, n))
    assert not leftovers, f"removed MCP tools still defined: {leftovers}"


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ingest_knowledge — one tool, exactly one source
# ---------------------------------------------------------------------------


class TestIngestKnowledge:
    def _call(self, **kwargs):
        from unittest.mock import patch

        from axon.mcp_server import ingest_knowledge

        mock = _mock_post({"status": "ok"})
        with patch("httpx.AsyncClient.post", mock):
            result = _run(ingest_knowledge(**kwargs))
        assert result == {"status": "ok"}
        args, kw = mock.call_args
        return args[0], kw["json"], mock

    def test_text_goes_to_add_text_with_metadata_doc_id_and_project(self):
        url, body, _ = self._call(
            text="Hello", metadata={"source": "s"}, doc_id="d1", project="proj"
        )
        assert url.endswith("/add_text")
        assert body == {
            "text": "Hello",
            "metadata": {"source": "s"},
            "doc_id": "d1",
            "project": "proj",
        }

    def test_text_omits_unset_optionals(self):
        url, body, _ = self._call(text="Hello")
        assert url.endswith("/add_text")
        assert body == {"text": "Hello"}

    def test_docs_go_to_add_texts(self):
        docs = [{"text": "A"}, {"text": "B", "doc_id": "b"}]
        url, body, _ = self._call(docs=docs, project="proj")
        assert url.endswith("/add_texts")
        assert body == {"docs": docs, "project": "proj"}

    def test_url_goes_to_ingest_url(self):
        url, body, _ = self._call(url="https://example.com/p", metadata={"topic": "x"})
        assert url.endswith("/ingest_url")
        assert body == {"url": "https://example.com/p", "metadata": {"topic": "x"}}

    def test_path_goes_to_ingest_with_project_assertion(self):
        url, body, _ = self._call(path="/data/docs", project="proj")
        assert url.endswith("/ingest")
        assert body == {"path": "/data/docs", "project": "proj"}

    def test_refresh_posts_refresh_and_never_switches_project(self):
        """The old refresh_ingest(project=...) silently POSTed /project/switch
        first; the project is now an assertion carried in the refresh body."""
        url, body, mock = self._call(refresh=True, project="proj")
        assert url.endswith("/ingest/refresh")
        assert body == {"project": "proj"}
        assert mock.call_count == 1
        called = [c.args[0] for c in mock.call_args_list]
        assert not any(u.endswith("/project/switch") for u in called)

    def test_refresh_without_project_sends_empty_body(self):
        url, body, _ = self._call(refresh=True)
        assert url.endswith("/ingest/refresh")
        assert body == {}

    def test_no_source_raises(self):
        import pytest

        from axon.mcp_server import ingest_knowledge

        with pytest.raises(ValueError, match="exactly one"):
            _run(ingest_knowledge())

    def test_two_sources_raise(self):
        import pytest

        from axon.mcp_server import ingest_knowledge

        with pytest.raises(ValueError, match="exactly one"):
            _run(ingest_knowledge(text="a", url="https://example.com"))
        with pytest.raises(ValueError, match="exactly one"):
            _run(ingest_knowledge(path="/x", refresh=True))

    def test_empty_text_still_counts_as_a_source(self):
        """text="" is a (bad) source, not "no source" — the server rejects it
        with a real error rather than the tool claiming nothing was given."""
        url, body, _ = self._call(text="")
        assert url.endswith("/add_text")
        assert body == {"text": ""}


def test_get_job_status_proxies_get():
    from unittest.mock import patch

    from axon.mcp_server import get_job_status

    rv = {"job_id": "j1", "status": "completed"}
    mock = _mock_get(rv)
    with patch("httpx.AsyncClient.get", mock):
        assert _run(get_job_status("j1")) == rv
    assert mock.call_args.args[0].endswith("/ingest/status/j1")


def test_list_knowledge_proxies_get():
    from unittest.mock import patch

    from axon.mcp_server import list_knowledge

    rv = {"sources": [], "total_chunks": 0}
    with patch("httpx.AsyncClient.get", _mock_get(rv)):
        assert _run(list_knowledge()) == rv


def test_delete_documents_proxies_post():
    from unittest.mock import patch

    from axon.mcp_server import delete_documents

    rv = {"deleted": 1}
    mock = _mock_post(rv)
    with patch("httpx.AsyncClient.post", mock):
        assert _run(delete_documents(doc_ids=["abc"])) == rv
    assert mock.call_args.kwargs["json"] == {"doc_ids": ["abc"]}


def test_list_projects_proxies_get():
    from unittest.mock import patch

    from axon.mcp_server import list_projects

    rv = {"projects": ["default"]}
    with patch("httpx.AsyncClient.get", _mock_get(rv)):
        assert _run(list_projects()) == rv


def test_switch_project_proxies_post():
    from unittest.mock import patch

    from axon.mcp_server import switch_project

    mock = _mock_post({"status": "success"})
    with patch("httpx.AsyncClient.post", mock):
        _run(switch_project("research"))
    assert mock.call_args.args[0].endswith("/project/switch")
    assert mock.call_args.kwargs["json"] == {"project_name": "research"}


def test_create_project_proxies_post():
    from unittest.mock import patch

    from axon.mcp_server import create_project

    rv = {"status": "created", "project": "myproj"}
    with patch("httpx.AsyncClient.post", _mock_post(rv)):
        assert _run(create_project(name="myproj")) == rv


def test_create_project_forwards_graph_backend_when_provided():
    from unittest.mock import patch

    from axon.mcp_server import create_project

    rv = {"status": "created", "project": "dgproj", "graph_backend": "dynamic_graph"}
    mock_post = _mock_post(rv)
    with patch("httpx.AsyncClient.post", mock_post):
        assert _run(create_project(name="dgproj", graph_backend="dynamic_graph")) == rv
    assert mock_post.call_args.kwargs["json"]["graph_backend"] == "dynamic_graph"


def test_create_project_omits_graph_backend_when_not_provided():
    from unittest.mock import patch

    from axon.mcp_server import create_project

    mock_post = _mock_post({"status": "created", "project": "myproj"})
    with patch("httpx.AsyncClient.post", mock_post):
        _run(create_project(name="myproj"))
    assert "graph_backend" not in mock_post.call_args.kwargs["json"]


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------


def test_get_config_returns_bare_config_by_default():
    from unittest.mock import patch

    from axon.mcp_server import get_config

    rv = {"top_k": 10}
    mock = _mock_get(rv)
    with patch("httpx.AsyncClient.get", mock):
        assert _run(get_config()) == rv
    assert mock.call_count == 1
    assert mock.call_args.args[0].endswith("/config")


def test_get_config_validate_also_fetches_validation():
    from unittest.mock import AsyncMock, patch

    from axon.mcp_server import get_config

    responses = {
        "/config": {"top_k": 10},
        "/config/validate": {"valid": True, "issue_count": 0, "issues": []},
    }
    mock = AsyncMock(side_effect=lambda path, params=None: responses[path])
    with patch("axon.mcp_server._get", mock):
        result = _run(get_config(validate=True))
    assert result == {"config": responses["/config"], "validation": responses["/config/validate"]}
    assert [c.args[0] for c in mock.call_args_list] == ["/config", "/config/validate"]


def test_set_config_sends_batch_with_persist_false_by_default():
    from unittest.mock import patch

    from axon.mcp_server import set_config

    rv = {"status": "success", "applied": []}
    mock = _mock_post(rv)
    with patch("httpx.AsyncClient.post", mock):
        assert _run(set_config({"top_k": 8, "rerank": True})) == rv
    assert mock.call_args.args[0].endswith("/config/set")
    assert mock.call_args.kwargs["json"] == {
        "settings": {"top_k": 8, "rerank": True},
        "persist": False,
    }


def test_set_config_persist_true_is_forwarded():
    from unittest.mock import patch

    from axon.mcp_server import set_config

    mock = _mock_post({"status": "success"})
    with patch("httpx.AsyncClient.post", mock):
        _run(set_config({"llm.model": "m"}, persist=True))
    assert mock.call_args.kwargs["json"]["persist"] is True


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def test_graph_retrieve_forwards_optional_fields():
    from unittest.mock import patch

    from axon.mcp_server import graph_retrieve

    mock = _mock_post({"contexts": []})
    with patch("httpx.AsyncClient.post", mock):
        _run(
            graph_retrieve(
                "who",
                top_k=3,
                point_in_time="2026-01-01T00:00:00Z",
                federation_weights={"graphrag": 1.0},
            )
        )
    assert mock.call_args.args[0].endswith("/graph/retrieve")
    assert mock.call_args.kwargs["json"] == {
        "query": "who",
        "top_k": 3,
        "point_in_time": "2026-01-01T00:00:00Z",
        "federation_weights": {"graphrag": 1.0},
    }


def test_update_fact_posts_graph_facts_minimal_body():
    """GraphFactRequest is extra=forbid, so only the given fields are sent."""
    from unittest.mock import patch

    from axon.mcp_server import update_fact

    rv = {"status": "created", "fact_id": "f1"}
    mock = _mock_post(rv)
    with patch("httpx.AsyncClient.post", mock):
        assert _run(update_fact("Alice", "WORKS_FOR", "Acme")) == rv
    assert mock.call_args.args[0].endswith("/graph/facts")
    assert mock.call_args.kwargs["json"] == {
        "subject": "Alice",
        "relation": "WORKS_FOR",
        "object": "Acme",
    }


def test_update_fact_forwards_all_optional_fields():
    from unittest.mock import patch

    from axon.mcp_server import update_fact

    mock = _mock_post({"status": "superseded"})
    with patch("httpx.AsyncClient.post", mock):
        _run(
            update_fact(
                "Alice",
                "IS_CEO_OF",
                "Acme",
                description="from the board minutes",
                confidence=0.8,
                replace=False,
                project="proj",
            )
        )
    assert mock.call_args.kwargs["json"] == {
        "subject": "Alice",
        "relation": "IS_CEO_OF",
        "object": "Acme",
        "description": "from the board minutes",
        "confidence": 0.8,
        "replace": False,
        "project": "proj",
    }


def test_update_fact_body_is_accepted_by_the_rest_schema():
    """Every key the tool can send must be a GraphFactRequest field."""
    from axon.api_schemas import GraphFactRequest

    sent = {"subject", "relation", "object", "description", "confidence", "replace", "project"}
    assert sent <= set(GraphFactRequest.model_fields)


# ---------------------------------------------------------------------------
# Sharing
# ---------------------------------------------------------------------------


def test_share_project_proxies_post():
    from unittest.mock import patch

    from axon.mcp_server import share_project

    rv = {"share_string": "axon-share-v1:..."}
    with patch("httpx.AsyncClient.post", _mock_post(rv)):
        assert _run(share_project(project="default", grantee="alice")) == rv


def test_redeem_share_and_list_shares_proxy():
    from unittest.mock import patch

    from axon.mcp_server import list_shares, redeem_share

    mock = _mock_post({"mount_name": "alice_p"})
    with patch("httpx.AsyncClient.post", mock):
        _run(redeem_share("abc"))
    assert mock.call_args.kwargs["json"] == {"share_string": "abc"}
    get = _mock_get({"sharing": [], "shared": []})
    with patch("httpx.AsyncClient.get", get):
        _run(list_shares())
    assert get.call_args.args[0].endswith("/share/list")


def test_project_params_are_documented_as_assertions():
    """`project` never switches; no docstring may call it a "Target project"."""
    import inspect

    import axon.mcp_server as mod

    for name in EXPECTED_MCP_TOOL_NAMES:
        doc = inspect.getdoc(getattr(mod, name)) or ""
        assert "Target project" not in doc, name


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


# ---------------------------------------------------------------------------
# Share-mount tools (#51–#54): proxy behavior
# ---------------------------------------------------------------------------


def test_share_project_passes_ttl_days_when_set():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import share_project

    async def _run():
        rv = {"key_id": "sk_x", "share_string": "abc", "expires_at": "2099-01-01T00:00:00+00:00"}
        mock = _mock_post(rv)
        with patch("httpx.AsyncClient.post", mock):
            await share_project(project="p", grantee="bob", ttl_days=14)
        args, kwargs = mock.call_args
        assert args[0].endswith("/share/generate")
        body = kwargs["json"]
        assert body == {"project": "p", "grantee": "bob", "ttl_days": 14}

    asyncio.run(_run())


def test_share_project_omits_ttl_days_when_none():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import share_project

    async def _run():
        mock = _mock_post({"key_id": "sk_x"})
        with patch("httpx.AsyncClient.post", mock):
            await share_project(project="p", grantee="bob")
        args, kwargs = mock.call_args
        body = kwargs["json"]
        # ttl_days omitted entirely when None — preserves the legacy wire format.
        assert "ttl_days" not in body
        assert body == {"project": "p", "grantee": "bob"}

    asyncio.run(_run())


def test_extend_share_proxies_with_key_id_and_ttl():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import extend_share

    async def _run():
        rv = {"key_id": "sk_a", "expires_at": "2099-01-01T00:00:00+00:00"}
        mock = _mock_post(rv)
        with patch("httpx.AsyncClient.post", mock):
            result = await extend_share(key_id="sk_a", ttl_days=30)
        assert result == rv
        args, kwargs = mock.call_args
        assert args[0].endswith("/share/extend")
        assert kwargs["json"] == {"key_id": "sk_a", "ttl_days": 30}

    asyncio.run(_run())


def test_extend_share_passes_null_ttl_to_clear_expiry():
    import asyncio
    from unittest.mock import patch

    from axon.mcp_server import extend_share

    async def _run():
        mock = _mock_post({"expires_at": None})
        with patch("httpx.AsyncClient.post", mock):
            await extend_share(key_id="sk_a", ttl_days=None)
        args, kwargs = mock.call_args
        # null ttl_days IS forwarded — REST semantics treat it as "clear expiry".
        assert kwargs["json"] == {"key_id": "sk_a", "ttl_days": None}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Errors carry the server's detail, not just httpx's status line
# ---------------------------------------------------------------------------


def _error_response(status: int, payload=None, text: str | None = None):
    import httpx

    req = httpx.Request("POST", "http://localhost:8420/x")
    if payload is not None:
        return httpx.Response(status, json=payload, request=req)
    return httpx.Response(status, text=text or "", request=req)


def test_400_detail_reaches_the_agent():
    from unittest.mock import AsyncMock, patch

    import pytest

    from axon.mcp_server import AxonAPIError, set_config

    detail = "Unknown config key(s) ['nope']; nothing was applied."
    resp = _error_response(400, {"detail": detail})
    with patch("httpx.AsyncClient.post", AsyncMock(return_value=resp)):
        with pytest.raises(AxonAPIError) as exc:
            _run(set_config({"nope": 1}))
    assert exc.value.status == 400
    assert "nope" in str(exc.value) and "400" in str(exc.value)
    assert "/config/set" in str(exc.value)


def test_409_detail_names_the_active_project():
    from unittest.mock import AsyncMock, patch

    import pytest

    from axon.mcp_server import AxonAPIError, query_knowledge

    detail = "Brain is serving project 'alpha', not 'beta'. Use POST /project/switch to change."
    resp = _error_response(409, {"detail": detail})
    with patch("httpx.AsyncClient.post", AsyncMock(return_value=resp)):
        with pytest.raises(AxonAPIError, match="serving project 'alpha'"):
            _run(query_knowledge("q", project="beta"))


def test_get_errors_carry_detail_and_non_json_falls_back_to_text():
    from unittest.mock import AsyncMock, patch

    import pytest

    from axon.mcp_server import AxonAPIError, list_knowledge

    resp = _error_response(503, text="Brain not initialized")
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=resp)):
        with pytest.raises(AxonAPIError, match="503.*Brain not initialized"):
            _run(list_knowledge())


def test_structured_detail_is_serialised():
    from unittest.mock import AsyncMock, patch

    import pytest

    from axon.mcp_server import AxonAPIError, update_fact

    resp = _error_response(422, {"detail": [{"loc": ["body", "relation"], "msg": "bad"}]})
    with patch("httpx.AsyncClient.post", AsyncMock(return_value=resp)):
        with pytest.raises(AxonAPIError, match="relation"):
            _run(update_fact("A", "!!", "B"))


def test_graph_retrieve_forwards_project_assertion():
    from unittest.mock import patch

    from axon.mcp_server import graph_retrieve

    mock = _mock_post({"contexts": []})
    with patch("httpx.AsyncClient.post", mock):
        _run(graph_retrieve("who", project="alpha"))
    assert mock.call_args.kwargs["json"] == {"query": "who", "project": "alpha"}
