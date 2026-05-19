"""Tests for the branded `/docs`, `/redoc`, `/favicon.ico`, and `/brand/*`
endpoints added when the Axon icon was wired into the FastAPI app.

These guard against three regressions:

1. `/docs` and `/redoc` start rendering FastAPI's default favicon again
   (e.g. someone removes the custom Swagger UI / ReDoc overrides).
2. `/favicon.ico` stops redirecting (browsers will then 404-spam logs).
3. The `/brand/` static mount disappears or stops shipping the SVGs
   from inside the package — the SVGs live at `src/axon/brand/*.svg`
   so the wheel-installed copy of axon-rag serves them too.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


def test_docs_references_axon_favicon():
    """Swagger UI HTML must point at the Axon favicon, not FastAPI's default."""
    r = client.get("/docs")
    assert r.status_code == 200
    assert "/brand/axon-favicon.svg" in r.text
    # Sanity: the default FastAPI favicon URL should not appear.
    assert "fastapi.tiangolo.com/img/favicon" not in r.text


def test_redoc_references_axon_favicon():
    """ReDoc HTML must point at the Axon favicon too."""
    r = client.get("/redoc")
    assert r.status_code == 200
    assert "/brand/axon-favicon.svg" in r.text
    assert "fastapi.tiangolo.com/img/favicon" not in r.text


def test_docs_title_includes_axon():
    """Branded title makes the tab readable in a sea of API docs."""
    r = client.get("/docs")
    assert r.status_code == 200
    m = re.search(r"<title>([^<]+)</title>", r.text)
    assert m is not None
    assert "Axon" in m.group(1)


def test_favicon_ico_redirects_to_canonical_svg():
    """Browsers hammer /favicon.ico unprompted — must 302 to the SVG."""
    r = client.get("/favicon.ico", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/brand/axon-favicon.svg"


def test_brand_mount_serves_favicon_svg():
    """The /brand/ static mount must serve the favicon SVG itself."""
    r = client.get("/brand/axon-favicon.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg")
    # SVG should contain the recognisable hex polygon coordinates.
    assert b"polygon" in r.content
    assert b"#00d4b4" in r.content  # Axon teal


def test_brand_mount_serves_full_icon_svg():
    """The /brand/ mount must also serve the full icon SVG (not just favicon)."""
    r = client.get("/brand/axon-icon.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg")


def test_brand_mount_serves_wordmark_svg():
    """The /brand/ mount must serve the wordmark SVG too — used by docs sites."""
    r = client.get("/brand/axon-wordmark.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg")


def test_brand_mount_blocks_path_traversal():
    """The static mount must not serve files outside src/axon/brand/."""
    # Starlette's StaticFiles normalises and rejects traversal — confirm.
    r = client.get("/brand/../api.py")
    # Either 404 (not found) or 400 (bad request) — both are safe; what we
    # care about is that we don't get a 200 with python source.
    assert r.status_code != 200 or b"def " not in r.content
