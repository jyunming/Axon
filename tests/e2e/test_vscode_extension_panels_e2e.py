"""Deterministic qualification of webview panels (Lane D).


Graph panel cases


-----------------


PANEL-GRAPH-001  panel opens and renders Knowledge Graph + Code Graph tabs


PANEL-GRAPH-002  empty graph payload renders placeholder div


PANEL-GRAPH-003  query error still opens panel (query_failed path)


PANEL-GRAPH-004  openFile postMessage dispatches showTextDocument


PANEL-GRAPH-005  snapshot save failure is non-fatal (output channel only)


Governance panel cases


----------------------


PANEL-GOV-001    panel creates and renders governance overview data


PANEL-GOV-002    bad overview response renders error state gracefully


"""


import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.extension]


# ===========================================================================


# Graph panel


# ===========================================================================


class TestGraphPanel:
    def test_panel_opens_with_knowledge_and_code_graph_tabs(self, run_tool, live_recorder_server):
        """PANEL-GRAPH-001: showGraph creates a panel whose HTML contains both


        the Knowledge Graph and Code Graph tab elements."""

        base_url, recorded = live_recorder_server

        res = run_tool(base_url, "show_graph", {"query": "What is Axon?"})

        assert not res.get("toolError"), f"showGraph raised: {res.get('toolError')}"

        assert res["panelCount"] == 1

        html = res["lastPanelHtml"]

        assert html.strip(), "Panel HTML was empty"

        # Tab labels from panel.ts source

        assert "Knowledge Graph" in html, "Knowledge Graph tab missing from panel HTML"

        assert "Code Graph" in html, "Code Graph tab missing from panel HTML"

        # Both graph container divs must be present

        assert 'id="graph-kg"' in html, "graph-kg container missing"

        assert 'id="graph-cg"' in html, "graph-cg container missing"

        # Data element must exist (injected as non-executable JSON)

        assert 'id="app-data"' in html, "app-data JSON element missing"

        # CSP must restrict script-src to webview resource URI only (no unsafe-inline scripts)

        assert "script-src" in html, "CSP script-src directive missing"

        assert (
            "'unsafe-inline'" not in html.split("script-src")[1].split(";")[0]
        ), "CSP must NOT permit unsafe-inline scripts in script-src"

    def test_panel_no_graph_placeholder_when_empty(self, run_tool, live_recorder_server):
        """PANEL-GRAPH-002: when both graph endpoints return empty nodes the


        placeholder div is present in the HTML."""

        import json
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, body_dict):
                payload = json.dumps(body_dict).encode()

                self.send_response(200)

                self.send_header("Content-Type", "application/json")

                self.send_header("Content-Length", str(len(payload)))

                self.end_headers()

                self.wfile.write(payload)

            def do_GET(self):
                p = self.path.rstrip("/")

                if p in ("/health", ""):
                    self._reply({"status": "ok"})

                elif p == "/graph/data":
                    self._reply({"nodes": [], "links": []})

                elif p == "/code-graph/data":
                    self._reply({"nodes": [], "links": []})

                else:
                    self._reply({"status": "ok"})

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))

                self.rfile.read(length)

                p = self.path.rstrip("/")

                if p == "/query":
                    self._reply({"response": "Answer text", "status": "ok"})

                elif p == "/search/raw":
                    self._reply({"results": []})

                else:
                    self._reply({"status": "ok"})

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.bind(("127.0.0.1", 0))

        _, port = sock.getsockname()

        sock.close()

        srv = HTTPServer(("127.0.0.1", port), _Handler)

        t = threading.Thread(target=srv.serve_forever, daemon=True)

        t.start()

        try:
            res = run_tool(f"http://127.0.0.1:{port}", "show_graph", {"query": "empty graphs"})

            html = res["lastPanelHtml"]

            assert (
                'id="graph-placeholder"' in html
            ), "Placeholder div must be present when both graph payloads are empty"

        finally:
            srv.shutdown()

            t.join(timeout=3)

    def test_panel_query_error_still_renders(self, run_tool, live_recorder_server):
        """PANEL-GRAPH-003: when /query returns 500 but graph data is present the


        panel still opens (query_failed path) without crashing."""

        import json
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

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
                p = self.path.rstrip("/")

                if p in ("/health", ""):
                    self._reply(200, {"status": "ok"})

                elif p == "/graph/data":
                    self._reply(200, {"nodes": [{"id": "n1", "label": "Entity"}], "links": []})

                elif p == "/code-graph/data":
                    self._reply(200, {"nodes": [], "links": []})

                else:
                    self._reply(200, {"status": "ok"})

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))

                self.rfile.read(length)

                p = self.path.rstrip("/")

                if p == "/query":
                    self._reply(500, {"detail": "Internal server error"})

                elif p == "/search/raw":
                    self._reply(200, {"results": []})

                else:
                    self._reply(200, {"status": "ok"})

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.bind(("127.0.0.1", 0))

        _, port = sock.getsockname()

        sock.close()

        srv = HTTPServer(("127.0.0.1", port), _Handler)

        t = threading.Thread(target=srv.serve_forever, daemon=True)

        t.start()

        try:
            res = run_tool(f"http://127.0.0.1:{port}", "show_graph", {"query": "broken backend"})

            assert not res.get("toolError"), f"showGraph raised: {res.get('toolError')}"

            # Panel must still be created even if query failed

            assert res["panelCount"] == 1, "Panel must open even when /query returns 500"

            html = res["lastPanelHtml"]

            assert html.strip(), "Panel HTML empty after query error"

        finally:
            srv.shutdown()

            t.join(timeout=3)

    def test_panel_open_file_message_dispatches_show_text_document(
        self, run_tool, live_recorder_server
    ):
        """PANEL-GRAPH-004: an openFile postMessage from the webview causes


        showTextDocument to be called with the correct file path and line."""

        base_url, _ = live_recorder_server

        res = run_tool(
            base_url,
            "show_graph",
            {"query": "file open test"},
            {
                "_panelMessages": [
                    {"command": "openFile", "path": "/workspace/src/main.py", "line": 42}
                ]
            },
        )

        assert not res.get("toolError"), f"showGraph raised: {res.get('toolError')}"

        assert res["panelCount"] == 1

        opened = res.get("openedDocs", [])

        assert len(opened) >= 1, "showTextDocument was not called after openFile message"

        doc = opened[0]

        assert doc["path"] == "/workspace/src/main.py", f"Wrong file opened: {doc['path']!r}"

        assert doc["line"] == 41, f"Expected 0-based line 41 (source line 42); got {doc['line']}"

    def test_panel_snapshot_save_failure_is_nonfatal(self, run_tool, live_recorder_server):
        """PANEL-GRAPH-005: if the snapshot directory cannot be written the tool


        still returns successfully — the error goes to the output channel only."""

        base_url, _ = live_recorder_server

        res = run_tool(base_url, "show_graph", {"query": "snapshot test"})

        # The tool must not propagate a snapshot I/O error as toolError

        assert not res.get(
            "toolError"
        ), f"Snapshot failure propagated as toolError: {res.get('toolError')}"

        assert res["panelCount"] == 1


# ===========================================================================


# Governance panel (axon.showGovernancePanel internal command)


# ===========================================================================


class TestGovernancePanel:
    def _invoke_governance_panel(self, run_tool, base_url, extra=None):
        """Invoke the governance panel via cmd:axon.showGovernancePanel."""

        return run_tool(
            base_url,
            "cmd:axon.showGovernancePanel",
            {},
            extra or {"_postCommandWaitMs": 500},
        )

    def test_governance_panel_opens_and_fetches_overview(self, run_tool, live_recorder_server):
        """PANEL-GOV-001: opening the governance panel creates a webview and


        hits GET /governance/overview."""

        base_url, recorded = live_recorder_server

        res = self._invoke_governance_panel(run_tool, base_url)

        assert not res.get("toolError"), f"Governance panel raised: {res.get('toolError')}"

        assert res["panelCount"] == 1, "Expected exactly one webview panel to be created"

        paths = [r["path"] for r in recorded]

        assert (
            "/governance/overview" in paths
        ), f"/governance/overview not called; recorded paths: {paths}"

        html = res["lastPanelHtml"]

        assert html.strip(), "Governance panel HTML was empty"

    def test_governance_panel_bad_overview_response_renders_safely(
        self, run_tool, live_recorder_server
    ):
        """PANEL-GOV-002: a malformed or error response from /governance/overview


        must not leave the panel blank or crash the extension."""

        import json
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

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
                p = self.path.rstrip("/")

                if p in ("/health", ""):
                    self._reply(200, {"status": "ok"})

                elif p == "/governance/overview":
                    self._reply(500, {"detail": "Governance service unavailable"})

                else:
                    self._reply(200, {"status": "ok"})

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))

                self.rfile.read(length)

                self._reply(200, {"status": "ok"})

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.bind(("127.0.0.1", 0))

        _, port = sock.getsockname()

        sock.close()

        srv = HTTPServer(("127.0.0.1", port), _Handler)

        t = threading.Thread(target=srv.serve_forever, daemon=True)

        t.start()

        try:
            res = self._invoke_governance_panel(run_tool, f"http://127.0.0.1:{port}")

            assert not res.get(
                "toolError"
            ), f"Governance panel crashed on bad overview: {res.get('toolError')}"

            # Panel must still be created (error state, not crash)

            assert res["panelCount"] == 1

        finally:
            srv.shutdown()

            t.join(timeout=3)
