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
    res = run_tool(base_url, "axon_queryKnowledge", {"query": "test"}, extra)

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
