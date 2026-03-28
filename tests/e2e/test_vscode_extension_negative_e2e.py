"""Deterministic negative-path qualification (Lane F).


Every test here verifies that a bad input, rejected request, or broken


backend produces a deterministic, user-readable error message — not a


silent success, an uncaught exception, or an empty string.


Coverage matrix


---------------


NEG-001  API unreachable — search_knowledge


NEG-002  API unreachable — query_knowledge


NEG-003  400 bad request — create_project (invalid name)


NEG-004  403 read-only rejection — ingest_text


NEG-005  403 read-only rejection — clear_knowledge


NEG-006  403 read-only rejection — graph_finalize


NEG-007  404 missing job — get_job_status


NEG-008  404 missing project — switch_project


NEG-009  409 project mismatch — query_knowledge with project param


NEG-010  unsupported image extension — ingest_image (.gif)


NEG-011  no Copilot model available — ingest_image


NEG-012  empty Copilot image description — ingest_image


NEG-013  graph endpoint failure — graph_status 500


NEG-014  store not active — share_project 404


NEG-015  command cancellation — axon.switchProject cancelled quick-pick


"""


from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.extension]


# ---------------------------------------------------------------------------


# Helpers


# ---------------------------------------------------------------------------


def _error_server(status: int, detail: str):
    """Context manager that yields a base_url where every POST/GET returns


    ``status`` with the given detail payload, plus 200 for /health."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, body_dict):
            payload = json.dumps(body_dict).encode()

            self.send_response(code)

            self.send_header("Content-Type", "application/json")

            self.send_header("Content-Length", str(len(payload)))

            self.end_headers()

            self.wfile.write(payload)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                self._reply(200, {"status": "ok"})

            else:
                self._reply(status, {"detail": detail})

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))

            self.rfile.read(length)

            self._reply(status, {"detail": detail})

    import contextlib

    @contextlib.contextmanager
    def _ctx():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.bind(("127.0.0.1", 0))

        _, port = sock.getsockname()

        sock.close()

        srv = HTTPServer(("127.0.0.1", port), _Handler)

        t = threading.Thread(target=srv.serve_forever, daemon=True)

        t.start()

        try:
            yield f"http://127.0.0.1:{port}"

        finally:
            srv.shutdown()

            t.join(timeout=3)

    return _ctx()


def _assert_error_text(res: dict, *fragments: str):
    """Assert that toolResult contains at least one of the given fragments and


    toolError is not set (the tool must surface the error gracefully)."""

    assert not res.get(
        "toolError"
    ), f"Tool raised an uncaught exception instead of returning error text: {res.get('toolError')}"

    text = res.get("toolResult", "")

    assert text.strip(), "toolResult was empty — error was silently swallowed"

    assert any(
        f.lower() in text.lower() for f in fragments
    ), f"None of {fragments!r} found in tool result: {text!r}"


# ---------------------------------------------------------------------------


# NEG-001 / NEG-002: API unreachable


# ---------------------------------------------------------------------------


def test_search_api_unreachable(run_tool, extension_js_path, extension_root_path, runner_js_path):
    """NEG-001: search_knowledge surfaces a connection-error message."""

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "vscode_helpers", Path(__file__).parent / "vscode_helpers.py"
    )

    helpers = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(helpers)

    res = helpers.run_extension_tool(
        runner_js_path=runner_js_path,
        extension_js=extension_js_path,
        extension_root=extension_root_path,
        api_base="http://127.0.0.1:19998",
        tool_name="search_knowledge",
        tool_input={"query": "unreachable"},
        timeout=15,
    )

    _assert_error_text(res, "error", "axon", "connect", "econnrefused", "fetch")


def test_query_api_unreachable(run_tool, extension_js_path, extension_root_path, runner_js_path):
    """NEG-002: query_knowledge surfaces a connection-error message."""

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "vscode_helpers", Path(__file__).parent / "vscode_helpers.py"
    )

    helpers = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(helpers)

    res = helpers.run_extension_tool(
        runner_js_path=runner_js_path,
        extension_js=extension_js_path,
        extension_root=extension_root_path,
        api_base="http://127.0.0.1:19997",
        tool_name="query_knowledge",
        tool_input={"query": "unreachable"},
        timeout=15,
    )

    _assert_error_text(res, "error", "axon", "connect", "econnrefused", "fetch")


# ---------------------------------------------------------------------------


# NEG-003: 400 bad request


# ---------------------------------------------------------------------------


def test_create_project_bad_name_400(run_tool, live_recorder_server):
    """NEG-003: create_project propagates the 400 API error text to the user."""

    base_url, _ = live_recorder_server

    with _error_server(400, "Invalid project name: must not contain spaces") as err_url:
        res = run_tool(err_url, "create_project", {"name": "bad name"})

    _assert_error_text(res, "400", "invalid", "error")


# ---------------------------------------------------------------------------


# NEG-004 / NEG-005 / NEG-006: 403 read-only / mounted rejections


# ---------------------------------------------------------------------------


def test_ingest_text_403_mounted(run_tool, live_recorder_server):
    """NEG-004: ingest_text on a mounted/read-only project shows 403 error."""

    with _error_server(403, "Write to mounted project not allowed") as err_url:
        res = run_tool(err_url, "ingest_text", {"text": "hello"})

    _assert_error_text(res, "403", "error", "not allowed", "mounted")


def test_clear_collection_403_mounted(run_tool, live_recorder_server):
    """NEG-005: clear_knowledge on a read-only project shows 403 error."""

    with _error_server(403, "Write to mounted project not allowed") as err_url:
        res = run_tool(err_url, "clear_knowledge", {})

    _assert_error_text(res, "403", "error", "not allowed", "mounted")


def test_finalize_graph_403_mounted(run_tool, live_recorder_server):
    """NEG-006: graph_finalize on a read-only project shows 403 error."""

    with _error_server(403, "Write to mounted project not allowed") as err_url:
        res = run_tool(err_url, "graph_finalize", {})

    _assert_error_text(res, "403", "error", "finalize", "graph")


# ---------------------------------------------------------------------------


# NEG-007: 404 missing job


# ---------------------------------------------------------------------------


def test_get_ingest_status_404_missing_job(run_tool, live_recorder_server):
    """NEG-007: get_job_status with an unknown job_id surfaces 'not found'."""

    with _error_server(404, "Job not found") as err_url:
        res = run_tool(err_url, "get_job_status", {"job_id": "ghost-job"})

    _assert_error_text(res, "404", "not found", "job", "error")


# ---------------------------------------------------------------------------


# NEG-008: 404 missing project


# ---------------------------------------------------------------------------


def test_switch_project_404_missing(run_tool, live_recorder_server):
    """NEG-008: switch_project to a non-existent project surfaces a 404 error."""

    with _error_server(404, "Project 'ghost' not found") as err_url:
        res = run_tool(err_url, "switch_project", {"name": "ghost"})

    _assert_error_text(res, "404", "not found", "error")


# ---------------------------------------------------------------------------


# NEG-009: 409 project mismatch


# ---------------------------------------------------------------------------


def test_query_409_project_mismatch(run_tool, live_recorder_server):
    """NEG-009: query_knowledge with an active different project returns 409 error text."""

    with _error_server(409, "Active project is 'default', requested 'research'") as err_url:
        res = run_tool(err_url, "query_knowledge", {"query": "test", "project": "research"})

    _assert_error_text(res, "409", "error", "project")


# ---------------------------------------------------------------------------


# NEG-010 / NEG-011 / NEG-012: ingest_image failure modes


# ---------------------------------------------------------------------------


def test_ingest_image_unsupported_extension(run_tool, live_recorder_server, tmp_path):
    """NEG-010: .gif files are rejected with an explicit unsupported-format message."""

    base_url, _ = live_recorder_server

    gif = tmp_path / "anim.gif"

    gif.write_bytes(b"GIF89a")

    res = run_tool(base_url, "ingest_image", {"imagePath": str(gif)})

    _assert_error_text(res, "unsupported", "gif", "format", "extension", "error")


def test_ingest_image_no_copilot_model(run_tool, live_recorder_server, tmp_path):
    """NEG-011: when no Copilot model is available the tool returns a clear message."""

    base_url, _ = live_recorder_server

    img = tmp_path / "photo.png"

    img.write_bytes(b"PNG")

    # Empty model list → no suitable model

    res = run_tool(
        base_url,
        "ingest_image",
        {"imagePath": str(img)},
        {"_copilotModels": []},
    )

    _assert_error_text(res, "copilot", "model", "available", "error", "vision")


def test_ingest_image_empty_description(run_tool, live_recorder_server, tmp_path):
    """NEG-012: when Copilot returns an empty description the tool errors explicitly."""

    base_url, _ = live_recorder_server

    img = tmp_path / "blank.png"

    img.write_bytes(b"PNG")

    res = run_tool(
        base_url,
        "ingest_image",
        {"imagePath": str(img)},
        {
            "_copilotModels": [{"id": "gpt-4o", "capabilities": {"supportsImageToText": True}}],
            "_copilotResponseText": "",
        },
    )

    _assert_error_text(res, "empty", "description", "error", "no description")


# ---------------------------------------------------------------------------


# NEG-013: graph status 500


# ---------------------------------------------------------------------------


def test_show_graph_status_500(run_tool, live_recorder_server):
    """NEG-013: graph_status with a 500 backend error surfaces the error text."""

    with _error_server(500, "Internal server error") as err_url:
        res = run_tool(err_url, "graph_status", {})

    _assert_error_text(res, "error", "500", "graph", "status")


# ---------------------------------------------------------------------------


# NEG-014: share / store not active


# ---------------------------------------------------------------------------


def test_share_project_store_not_active_404(run_tool, live_recorder_server):
    """NEG-014: share_project when AxonStore is not initialised returns 404 error."""

    with _error_server(404, "AxonStore not initialised") as err_url:
        res = run_tool(err_url, "share_project", {"project": "default", "grantee": "bob"})

    _assert_error_text(res, "404", "store", "error", "not")


# ---------------------------------------------------------------------------


# NEG-015: command cancellation


# ---------------------------------------------------------------------------


def test_switch_project_command_cancelled(run_tool, live_recorder_server):
    """NEG-015: axon.switchProject with cancelled quick-pick produces no error


    and makes no HTTP request."""

    base_url, recorded = live_recorder_server

    # Returning None from showQuickPick simulates user pressing Escape.

    res = run_tool(
        base_url,
        "cmd:axon.switchProject",
        {},
        {"_quickPickResponse": None},
    )

    assert not res.get("toolError"), f"Command raised: {res.get('toolError')}"

    paths = [r["path"] for r in recorded]

    assert (
        "/project/switch" not in paths
    ), "Expected no /project/switch call when quick-pick is cancelled"
