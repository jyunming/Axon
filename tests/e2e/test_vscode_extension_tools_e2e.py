"""Deterministic qualification of all LM tools (Lane A+B)."""


import json


import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.extension]


# ---------------------------------------------------------------------------


# Lane A: Manifest contract


# ---------------------------------------------------------------------------


def test_extension_manifest_contract(extension_root_path):
    manifest = json.loads((extension_root_path / "package.json").read_text())

    tools = {t["name"] for t in manifest["contributes"]["languageModelTools"]}

    actual_tools = {
        "search_knowledge",
        "query_knowledge",
        "ingest_text",
        "ingest_url",
        "ingest_path",
        "get_job_status",
        "list_projects",
        "switch_project",
        "create_project",
        "delete_project",
        "delete_documents",
        "list_knowledge",
        "clear_knowledge",
        "update_settings",
        "share_project",
        "redeem_share",
        "revoke_share",
        "list_shares",
        "init_store",
        "get_store_status",
        "refresh_ingest",
        "get_stale_docs",
        "graph_status",
        "graph_finalize",
        "get_current_settings",
    }

    # Verify manifest has them

    for t in actual_tools:
        assert t in tools, f"Tool {t} missing from manifest"


# ---------------------------------------------------------------------------


# Lane B: Tool direct invocation


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name, tool_input, expected_path",
    [
        ("search_knowledge", {"query": "test"}, "/search"),
        ("query_knowledge", {"query": "why"}, "/query"),
        ("ingest_text", {"text": "val"}, "/add_text"),
        ("ingest_url", {"url": "http://a"}, "/ingest_url"),
        ("ingest_path", {"path": "/tmp"}, "/ingest"),
        ("get_job_status", {"job_id": "j1"}, "/ingest/status/j1"),
        ("list_projects", {}, "/projects"),
        ("switch_project", {"name": "p1"}, "/project/switch"),
        ("create_project", {"name": "p2"}, "/project/new"),
        ("delete_project", {"name": "p3"}, "/project/delete/p3"),
        ("delete_documents", {"docIds": ["d1"]}, "/delete"),
        ("list_knowledge", {}, "/collection"),
        ("clear_knowledge", {}, "/clear"),
        ("update_settings", {"top_k": 10}, "/config/update"),
        ("share_project", {"project": "p", "grantee": "g"}, "/share/generate"),
        ("redeem_share", {"share_string": "s"}, "/share/redeem"),
        ("revoke_share", {"key_id": "k"}, "/share/revoke"),
        ("list_shares", {}, "/share/list"),
        ("init_store", {"base_path": "/b"}, "/store/init"),
        ("refresh_ingest", {}, "/ingest/refresh"),
        ("get_stale_docs", {}, "/collection/stale"),
        ("graph_status", {}, "/graph/status"),
        ("graph_finalize", {}, "/graph/finalize"),
        ("get_current_settings", {}, "/config"),
    ],
)
def test_tool_invocations(run_tool, live_recorder_server, tool_name, tool_input, expected_path):
    base_url, recorded = live_recorder_server

    extra = {"_inputResponse": "7"} if tool_name == "get_stale_docs" else None

    res = run_tool(base_url, tool_name, tool_input, extra)

    assert not res.get("toolError"), f"Tool {tool_name} failed: {res.get('toolError')}"

    # Verify hit correct endpoint

    paths = [r["path"] for r in recorded]

    assert any(
        p == expected_path or p == expected_path.rstrip("/") for p in paths
    ), f"{tool_name} did not hit {expected_path}, hit: {paths}"


def test_show_graph_tool(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, "show_graph", {"query": "test graph"})

    assert not res.get("toolError")

    assert res["panelCount"] == 1


def test_ingest_image_tool(run_tool, live_recorder_server, tmp_path):
    base_url, recorded = live_recorder_server

    img = tmp_path / "test.png"

    img.write_bytes(b"PNG")

    extra = {
        "_copilotModels": [{"id": "gpt-4o", "capabilities": {"supportsImageToText": True}}],
        "_copilotResponseText": "Diagram showing Axon data flow.",
    }

    res = run_tool(base_url, "ingest_image", {"imagePath": str(img)}, extra)

    assert not res.get("toolError")

    assert any(r["path"] == "/add_text" for r in recorded)
