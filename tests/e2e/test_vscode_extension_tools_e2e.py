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
        "axon_searchKnowledge",
        "axon_queryKnowledge",
        "axon_ingestText",
        "axon_ingestUrl",
        "axon_ingestPath",
        "axon_getIngestStatus",
        "axon_listProjects",
        "axon_switchProject",
        "axon_createProject",
        "axon_deleteProject",
        "axon_deleteDocuments",
        "axon_getCollection",
        "axon_clearCollection",
        "axon_updateSettings",
        "axon_shareProject",
        "axon_redeemShare",
        "axon_revokeShare",
        "axon_listShares",
        "axon_initStore",
        "axon_ingestImage",
        "axon_refreshIngest",
        "axon_listStaleDocs",
        "axon_clearKnowledgeBase",
        "axon_showGraphStatus",
        "axon_showGraph",
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
        ("axon_searchKnowledge", {"query": "test"}, "/search"),
        ("axon_queryKnowledge", {"query": "why"}, "/query"),
        ("axon_ingestText", {"text": "val"}, "/add_text"),
        ("axon_ingestUrl", {"url": "http://a"}, "/ingest_url"),
        ("axon_ingestPath", {"path": "/tmp"}, "/ingest"),
        ("axon_getIngestStatus", {"job_id": "j1"}, "/ingest/status/j1"),
        ("axon_listProjects", {}, "/projects"),
        ("axon_switchProject", {"name": "p1"}, "/project/switch"),
        ("axon_createProject", {"name": "p2"}, "/project/new"),
        ("axon_deleteProject", {"name": "p3"}, "/project/delete/p3"),
        ("axon_deleteDocuments", {"docIds": ["d1"]}, "/delete"),
        ("axon_getCollection", {}, "/collection"),
        ("axon_clearCollection", {}, "/clear"),
        ("axon_updateSettings", {"top_k": 10}, "/config/update"),
        ("axon_shareProject", {"project": "p", "grantee": "g"}, "/share/generate"),
        ("axon_redeemShare", {"share_string": "s"}, "/share/redeem"),
        ("axon_revokeShare", {"key_id": "k"}, "/share/revoke"),
        ("axon_listShares", {}, "/share/list"),
        ("axon_initStore", {"base_path": "/b"}, "/store/init"),
        ("axon_refreshIngest", {}, "/ingest/refresh"),
        ("axon_listStaleDocs", {}, "/collection/stale"),
        ("axon_clearKnowledgeBase", {}, "/clear"),
        ("axon_showGraphStatus", {}, "/graph/status"),
    ],
)
def test_tool_invocations(run_tool, live_recorder_server, tool_name, tool_input, expected_path):
    base_url, recorded = live_recorder_server
    extra = {"_inputResponse": "7"} if tool_name == "axon_listStaleDocs" else None
    res = run_tool(base_url, tool_name, tool_input, extra)

    assert not res.get("toolError"), f"Tool {tool_name} failed: {res.get('toolError')}"
    # Verify hit correct endpoint
    paths = [r["path"] for r in recorded]
    assert any(
        p == expected_path or p == expected_path.rstrip("/") for p in paths
    ), f"{tool_name} did not hit {expected_path}, hit: {paths}"


def test_axon_showGraph_tool(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server
    res = run_tool(base_url, "axon_showGraph", {"query": "test graph"})

    assert not res.get("toolError")
    assert res["panelCount"] == 1


def test_axon_ingestImage_tool(run_tool, live_recorder_server, tmp_path):
    base_url, recorded = live_recorder_server
    img = tmp_path / "test.png"
    img.write_bytes(b"PNG")

    extra = {
        "_copilotModels": [{"id": "gpt-4o", "capabilities": {"supportsImageToText": True}}],
        "_copilotResponseText": "Diagram showing Axon data flow.",
    }
    res = run_tool(base_url, "axon_ingestImage", {"imagePath": str(img)}, extra)
    assert not res.get("toolError")
    assert any(r["path"] == "/add_text" for r in recorded)
