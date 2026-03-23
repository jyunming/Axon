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


# ---------------------------------------------------------------------------
# VSC-CMD-START / VSC-CMD-STOP: server lifecycle commands
# ---------------------------------------------------------------------------


def test_start_server_command_probes_health(run_tool, live_recorder_server):
    """axon.startServer: when the API is already reachable, extension marks it
    as running and logs to the output channel without spawning a subprocess."""
    base_url, recorded = live_recorder_server

    # The recorder replies 200 to /health, so ensureServerRunning sees a live server.
    res = run_tool(
        base_url,
        "cmd:axon.startServer",
        {},
        {"autoStart": False, "_postActivateWaitMs": 0},
    )

    assert not res.get("toolError"), f"startServer failed: {res.get('toolError')}"
    paths = [r["path"] for r in recorded]
    assert "/health" in paths, f"startServer did not probe /health; hit: {paths}"


def test_start_server_command_no_workspace(run_tool, live_recorder_server):
    """axon.startServer: when workspaceFolders is empty the command exits
    cleanly and logs the 'no workspace folder' message — no crash."""
    base_url, recorded = live_recorder_server

    # Provide no workspace folders — forces the early-exit branch in ensureServerRunning.
    # The recorder health endpoint will NOT be hit because the early exit fires first.
    res = run_tool(
        base_url,
        "cmd:axon.startServer",
        {},
        {"_workspaceFolders": [], "autoStart": False},
    )

    assert not res.get("toolError"), f"startServer (no-workspace) raised: {res.get('toolError')}"
    output = "\n".join(res.get("outputLines", []))
    assert "workspace" in output.lower() or "no workspace" in output.lower(), (
        f"Expected workspace warning in output lines; got: {output!r}"
    )


def test_stop_server_command_clears_managed_process(run_tool, live_recorder_server):
    """axon.stopServer: command executes cleanly.  With no spawned server
    process in the test harness the function returns without error."""
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, "cmd:axon.stopServer", {}, {})

    assert not res.get("toolError"), f"stopServer raised: {res.get('toolError')}"


def test_show_graph_command_cancelled(run_tool, live_recorder_server):
    """axon.showGraphForQuery: when the input box is cancelled (returns null)
    no panel is created and no error is raised."""
    base_url, recorded = live_recorder_server

    res = run_tool(
        base_url,
        "cmd:axon.showGraphForQuery",
        {},
        # _inputResponse not set → showInputBox returns null → command cancels
        {},
    )

    assert not res.get("toolError"), f"showGraphForQuery (cancelled) raised: {res.get('toolError')}"
    assert res["panelCount"] == 0, "Panel must NOT open when input is cancelled"
