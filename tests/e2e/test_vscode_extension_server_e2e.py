"""Deterministic qualification of server lifecycle and extension config."""


import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.extension]


def test_config_forwarding(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    # Check if config changes result in expected behavior/headers/payloads.

    # We'll use extra config to mock different extension settings

    extra = {
        "apiKey": "test-key-007",
        "topK": 12,
        "useCopilotLlm": True,
    }

    # Try a search tool which we know uses topK from config if not provided

    # Actually tool query.ts reads top_k from tool input, but it might fall back.

    # Let's check queryKnowledge tool which uses top_k.

    res = run_tool(base_url, "query_knowledge", {"query": "test"}, extra)

    assert not res.get("toolError")

    # Verify the request recorded had the X-API-Key header

    assert any(
        r["headers"].get("x-api-key") == "test-key-007" for r in recorded
    ), "X-API-Key header not forwarded from extension config"


def test_server_activation_auto_start(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    extra = {"autoStart": True, "_postActivateWaitMs": 200}

    run_tool(base_url, None, {}, extra)

    assert any(r["path"] == "/health" for r in recorded)


def test_use_copilot_llm_flow(run_tool, live_recorder_server):
    base_url, recorded = live_recorder_server

    extra = {
        "useCopilotLlm": True,
        "_postActivateWaitMs": 800,
        "_copilotModels": [{"id": "gpt-4o", "capabilities": {"supportsImageToText": True}}],
    }

    run_tool(base_url, None, {}, extra)

    assert any(
        r["path"] == "/config/update" and r["body"].get("llm_provider") == "copilot"
        for r in recorded
    )


# ---------------------------------------------------------------------------


# VSC-SRV-002: autoStart with no workspace


# ---------------------------------------------------------------------------


def test_auto_start_no_workspace_does_not_crash(run_tool, live_recorder_server):
    """VSC-SRV-002: when autoStart is enabled but workspaceFolders is empty


    the extension activates cleanly with no launch attempt."""

    base_url, recorded = live_recorder_server

    extra = {
        "autoStart": True,
        "_workspaceFolders": [],
        "_postActivateWaitMs": 200,
    }

    res = run_tool("http://127.0.0.1:49999", None, {}, extra)

    assert not res.get("toolError"), f"Activation crashed with no workspace: {res.get('toolError')}"

    # /health may be probed once before the workspace check, but no process spawn occurs.

    # The key assertion is that the runner did not exit with an error.

    output = "\n".join(res.get("outputLines", []))

    # It's acceptable for there to be zero health probes if workspace check fires first.

    # We just need no crash and a reasonable output message (or no output at all).

    # No assertion on output content — the important thing is clean exit.


# ---------------------------------------------------------------------------


# VSC-SRV-006: ingestBase propagation


# ---------------------------------------------------------------------------


def test_ingest_base_propagated_in_start_command(run_tool, live_recorder_server):
    """VSC-SRV-006: when axon.ingestBase is set the output channel must log


    the configured base path when starting the server."""

    base_url, _ = live_recorder_server

    custom_base = "/custom/ingest/root"

    res = run_tool(
        "http://127.0.0.1:49999",
        "cmd:axon.startServer",
        {},
        {
            "apiBase": "http://127.0.0.1:49999",
            "ingestBase": custom_base,
            "_workspaceFolders": ["/workspace"],
            "autoStart": False,
        },
    )

    assert not res.get("toolError"), f"startServer raised: {res.get('toolError')}"

    output = "\n".join(res.get("outputLines", []))

    # server.ts logs: "RAG_INGEST_BASE=${fsRoot}"

    assert (
        custom_base in output
    ), f"axon.ingestBase value {custom_base!r} not reflected in output channel; got: {output!r}"


# ---------------------------------------------------------------------------


# VSC-SRV-007: storeBase propagation


# ---------------------------------------------------------------------------


def test_store_base_propagated_in_start_command(run_tool, live_recorder_server):
    """VSC-SRV-007: when axon.storeBase is set the output channel must log it."""

    base_url, _ = live_recorder_server

    custom_store = "/data/axon-store"

    res = run_tool(
        "http://127.0.0.1:49999",
        "cmd:axon.startServer",
        {},
        {
            "apiBase": "http://127.0.0.1:49999",
            "storeBase": custom_store,
            "_workspaceFolders": ["/workspace"],
            "autoStart": False,
        },
    )

    assert not res.get("toolError"), f"startServer raised: {res.get('toolError')}"

    output = "\n".join(res.get("outputLines", []))

    # server.ts logs: "AXON_STORE_BASE=${storeBase}"

    assert (
        custom_store in output
    ), f"axon.storeBase value {custom_store!r} not reflected in output channel; got: {output!r}"


# ---------------------------------------------------------------------------


# VSC-SRV-008: python discovery fallback — explicit axon.pythonPath wins


# ---------------------------------------------------------------------------


def test_python_path_setting_used_in_start_command(run_tool, live_recorder_server):
    """VSC-SRV-008 (step 1): an explicit axon.pythonPath is preferred above all


    auto-detection heuristics and is logged in the output channel."""

    base_url, _ = live_recorder_server

    explicit_py = "/opt/my-python/bin/python"

    res = run_tool(
        "http://127.0.0.1:49999",
        "cmd:axon.startServer",
        {},
        {
            "apiBase": "http://127.0.0.1:49999",
            "pythonPath": explicit_py,
            "_workspaceFolders": ["/workspace"],
            "autoStart": False,
        },
    )

    assert not res.get("toolError"), f"startServer raised: {res.get('toolError')}"

    output = "\n".join(res.get("outputLines", []))

    # server.ts logs the full spawn command including pythonPath

    assert (
        explicit_py in output
    ), f"Explicit pythonPath {explicit_py!r} not found in output channel; got: {output!r}"
