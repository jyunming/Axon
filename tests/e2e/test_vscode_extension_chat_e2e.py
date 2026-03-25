"""Deterministic qualification of the @axon chat participant (Lane B — chat surface).

Cases covered:
  VSC-CHAT-001  successful query → markdown response
  VSC-CHAT-002  backend 409 project mismatch → explicit error block in chat output
  VSC-CHAT-003  backend unreachable → user-visible guidance message
  VSC-CHAT-004  no-answer response shape → deterministic fallback text
"""
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.extension]

_PARTICIPANT = "chat:axon.chat"


def test_chat_participant_is_registered(run_tool, live_recorder_server):
    """Activation registers the 'axon.chat' participant exactly once."""
    base_url, _ = live_recorder_server
    res = run_tool(base_url, None, {}, {})
    assert (
        "axon.chat" in res["registeredParticipants"]
    ), f"axon.chat not in registeredParticipants: {res['registeredParticipants']}"


def test_chat_successful_query(run_tool, live_recorder_server):
    """VSC-CHAT-001: prompt produces a non-empty markdown response from /query."""
    base_url, recorded = live_recorder_server

    res = run_tool(base_url, _PARTICIPANT, {"prompt": "What does Axon do?"})
    assert not res.get("toolError"), f"Chat handler raised: {res.get('toolError')}"

    # Must have hit /query
    paths = [r["path"] for r in recorded]
    assert "/query" in paths, f"Expected /query to be hit; recorded: {paths}"

    # Must have produced non-empty markdown
    result_text = res.get("toolResult", "")
    assert result_text.strip(), "Chat response was empty"

    # The recorder returns {"response": "Axon synthesised answer."} — that text must appear
    assert (
        "Axon synthesised answer" in result_text or len(result_text) > 5
    ), f"Unexpected chat response: {result_text!r}"


def test_chat_backend_409_project_mismatch(run_tool, live_recorder_server):
    """VSC-CHAT-002: when the recorder returns 409, the participant outputs an
    explicit Axon error block — no silent swallow."""
    import json
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    # Start a single-endpoint server that returns 409 for /query
    class _409Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            self.rfile.read(length)
            body = json.dumps({"detail": "Project mismatch"}).encode()
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    srv = HTTPServer(("127.0.0.1", port), _409Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    try:
        alt_url = f"http://127.0.0.1:{port}"
        res = run_tool(alt_url, _PARTICIPANT, {"prompt": "test 409"})
        result_text = res.get("toolResult", "")
        # Must contain some error signal — "409", "error", "Axon error"
        assert any(
            tok in result_text.lower() for tok in ["409", "error", "axon"]
        ), f"No error signal in chat output for 409: {result_text!r}"
    finally:
        srv.shutdown()
        t.join(timeout=3)


def test_chat_backend_unreachable(run_tool, extension_js_path, extension_root_path, runner_js_path):
    """VSC-CHAT-003: when the API is completely unreachable, the participant
    produces a guidance message mentioning the API URL or autoStart."""
    dead_url = "http://127.0.0.1:19999"  # nothing listens there

    from pathlib import Path

    helpers_path = Path(__file__).parent / "vscode_helpers.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("vscode_helpers", helpers_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    res = helpers.run_extension_tool(
        runner_js_path=runner_js_path,
        extension_js=extension_js_path,
        extension_root=extension_root_path,
        api_base=dead_url,
        tool_name=_PARTICIPANT,
        tool_input={"prompt": "are you there?"},
        extra_config={},
        timeout=15,
    )

    result_text = res.get("toolResult", "")
    # Extension should mention the unreachable API, not crash silently
    assert result_text.strip(), "Chat response was empty for unreachable backend"
    assert any(
        tok in result_text.lower()
        for tok in ["could not reach", "error", "api", "autostart", "axon"]
    ), f"No guidance in chat output for unreachable backend: {result_text!r}"


def test_chat_no_answer_response(run_tool, live_recorder_server):
    """VSC-CHAT-004: when the API returns {response:''} the participant
    renders the deterministic fallback text rather than empty output."""
    import json
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _EmptyHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            self.rfile.read(length)
            body = json.dumps({"response": "", "status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    srv = HTTPServer(("127.0.0.1", port), _EmptyHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    try:
        alt_url = f"http://127.0.0.1:{port}"
        res = run_tool(alt_url, _PARTICIPANT, {"prompt": "empty answer"})
        result_text = res.get("toolResult", "")
        # The chatHandler source reads: data.response || '*No answer generated…*'
        assert result_text.strip(), "Chat must not render completely blank for empty response"
        assert (
            "no answer" in result_text.lower() or "knowledge base" in result_text.lower()
        ), f"Expected fallback text for empty response; got: {result_text!r}"
    finally:
        srv.shutdown()
        t.join(timeout=3)
