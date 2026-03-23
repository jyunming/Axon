"""Deterministic qualification of public commands and UI flows (Lane C)."""
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.extension]


@pytest.mark.parametrize(
    "command, extra, expected_path",
    [
        ("axon.switchProject", {"_quickPickResponse": "engineering"}, "/project/switch"),
        ("axon.createProject", {"_inputResponses": ["new-p", ""]}, "/project/new"),
        (
            "axon.ingestFile",
            {"_activeFile": "/tmp/f.txt", "_activeFileText": "file body"},
            "/add_texts",
        ),
        ("axon.ingestWorkspace", {"_workspaceFolders": ["/w"]}, "/ingest"),
        ("axon.ingestFolder", {"_openDialogResult": "/f"}, "/ingest"),
        ("axon.initStore", {"_inputResponse": "/store"}, "/store/init"),
        ("axon.shareProject", {"_inputResponses": ["p1", "bob"]}, "/share/generate"),
        ("axon.redeemShare", {"_inputResponse": "share-str"}, "/share/redeem"),
        ("axon.revokeShare", {}, "/share/revoke"),
        ("axon.listShares", {}, "/share/list"),
        ("axon.refreshIngest", {}, "/ingest/refresh"),
        ("axon.listStaleDocs", {"_inputResponse": "7"}, "/collection/stale"),
        ("axon.clearKnowledgeBase", {"_confirmResponse": "Clear Knowledge Base"}, "/clear"),
        ("axon.showGraphStatus", {}, "/graph/status"),
    ],
)
def test_command_invocations(run_tool, live_recorder_server, command, extra, expected_path):
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, f"cmd:{command}", {}, extra)

    assert not res.get("toolError"), f"Command {command} failed: {res.get('toolError')}"
    paths = [r["path"] for r in recorded]
    assert any(
        p == expected_path or p == expected_path.rstrip("/") for p in paths
    ), f"Command {command} did not hit {expected_path}, hit: {paths}"


def test_show_graph_commands(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    res1 = run_tool(base_url, "cmd:axon.showGraphForQuery", {}, {"_inputResponse": "how it works"})
    assert res1["panelCount"] == 1

    res2 = run_tool(
        base_url, "cmd:axon.showGraphForSelection", {}, {"_selectedText": "selected topic"}
    )
    assert res2["panelCount"] == 1
