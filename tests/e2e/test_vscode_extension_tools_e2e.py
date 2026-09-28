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

    # 0.5.0: the 18 MCP tools + the two client-side tools (show_graph, ingest_image).
    expected_tools = {
        "search_knowledge",
        "query_knowledge",
        "ingest_knowledge",
        "get_job_status",
        "list_knowledge",
        "delete_documents",
        "list_projects",
        "switch_project",
        "create_project",
        "get_config",
        "set_config",
        "graph_retrieve",
        "update_fact",
        "share_project",
        "redeem_share",
        "list_shares",
        "revoke_share",
        "extend_share",
        "show_graph",
        "ingest_image",
    }

    assert (
        tools == expected_tools
    ), f"missing: {sorted(expected_tools - tools)}  extra: {sorted(tools - expected_tools)}"


# ---------------------------------------------------------------------------


# Lane B: Tool direct invocation


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name, tool_input, expected_path",
    [
        ("search_knowledge", {"query": "test"}, "/search"),
        ("query_knowledge", {"query": "why"}, "/query"),
        ("ingest_knowledge", {"text": "val"}, "/add_text"),
        ("ingest_knowledge", {"docs": [{"text": "a"}]}, "/add_texts"),
        ("ingest_knowledge", {"url": "http://a"}, "/ingest_url"),
        ("ingest_knowledge", {"path": "/tmp"}, "/ingest"),
        ("ingest_knowledge", {"refresh": True}, "/ingest/refresh"),
        ("get_job_status", {"job_id": "j1"}, "/ingest/status/j1"),
        ("list_projects", {}, "/projects"),
        ("switch_project", {"project_name": "p1"}, "/project/switch"),
        ("create_project", {"name": "p2"}, "/project/new"),
        ("delete_documents", {"doc_ids": ["d1"]}, "/delete"),
        ("list_knowledge", {}, "/collection"),
        ("get_config", {}, "/config"),
        ("get_config", {"validate": True}, "/config/validate"),
        ("set_config", {"settings": {"top_k": 10}}, "/config/set"),
        ("graph_retrieve", {"query": "who"}, "/graph/retrieve"),
        ("update_fact", {"subject": "A", "relation": "WORKS_FOR", "object": "B"}, "/graph/facts"),
        ("share_project", {"project": "p", "grantee": "g"}, "/share/generate"),
        ("redeem_share", {"share_string": "s"}, "/share/redeem"),
        ("revoke_share", {"key_id": "k"}, "/share/revoke"),
        ("list_shares", {}, "/share/list"),
        ("extend_share", {"key_id": "k", "ttl_days": 7}, "/share/extend"),
    ],
)
def test_tool_invocations(run_tool, live_recorder_server, tool_name, tool_input, expected_path):
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, tool_name, tool_input)

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


def test_ingest_knowledge_rejects_zero_or_many_sources(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, "ingest_knowledge", {"text": "a", "url": "http://a"})

    assert not res.get("toolError")
    assert "exactly one" in json.dumps(res)
    assert not recorded, f"no request should be sent, got {[r['path'] for r in recorded]}"


def test_set_config_sends_one_batched_request_without_persist(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, "set_config", {"settings": {"top_k": 4, "rerank": True}})

    assert not res.get("toolError")
    sets = [r for r in recorded if r["path"] == "/config/set"]
    assert len(sets) == 1
    assert sets[0]["body"] == {"settings": {"top_k": 4, "rerank": True}, "persist": False}


def test_revoke_share_never_sends_rotate(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, "revoke_share", {"key_id": "ssk_x", "project": "p", "rotate": True})

    assert not res.get("toolError")
    (req,) = [r for r in recorded if r["path"] == "/share/revoke"]
    assert req["body"] == {"key_id": "ssk_x", "project": "p"}
