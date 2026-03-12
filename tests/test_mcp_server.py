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
    """All 10 expected MCP tools must be registered on the ``mcp`` instance."""
    from axon.mcp_server import mcp

    registered = set(mcp._tool_manager._tools.keys())
    assert registered == EXPECTED_MCP_TOOL_NAMES, (
        f"Tool mismatch.\n  Missing: {EXPECTED_MCP_TOOL_NAMES - registered}\n"
        f"  Extra:   {registered - EXPECTED_MCP_TOOL_NAMES}"
    )


def test_mcp_server_tool_count():
    """Exactly 10 tools must be registered — not more, not fewer."""
    from axon.mcp_server import mcp

    assert len(mcp._tool_manager._tools) == len(EXPECTED_MCP_TOOL_NAMES)


def test_main_is_callable():
    """The ``main`` entry point must be importable and callable."""
    from axon.mcp_server import main

    assert callable(main)
