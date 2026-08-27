"""Unit tests for single-instance detection (``axon.server_client``).

These are pure unit tests — ``urllib.request.urlopen`` is monkeypatched so no
real network or server is required.
"""

from __future__ import annotations

import json
import types
import urllib.error
from unittest import mock

from axon import server_client as sc


class _Resp:
    """Minimal context-manager stand-in for an ``http.client.HTTPResponse``."""

    def __init__(self, status: int = 200, body: bytes = b""):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _cfg(**kw):
    c = types.SimpleNamespace(
        api_host="127.0.0.1", api_port=8420, api_key="", projects_root=r"C:\store"
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _fake_urlopen(routes: dict[str, _Resp]):
    """Return a urlopen replacement that maps URL suffixes to responses."""

    def _open(req, timeout=None):
        url = req.full_url
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                return resp
        raise AssertionError(f"unexpected URL: {url}")

    return _open


# --- resolve_api_base ----------------------------------------------------


def test_resolve_api_base_default():
    assert sc.resolve_api_base(_cfg()) == "http://127.0.0.1:8420"


def test_resolve_api_base_env_override(monkeypatch):
    monkeypatch.setenv("AXON_API_BASE", "http://elsewhere:9000/")
    assert sc.resolve_api_base(_cfg()) == "http://elsewhere:9000"


def test_headers_includes_api_key():
    h = sc._headers(_cfg(api_key="secret"))
    assert h["X-API-Key"] == "secret" and h["X-Axon-Surface"] == "cli"
    assert "X-API-Key" not in sc._headers(_cfg(api_key=""))


# --- detect_server -------------------------------------------------------


def test_detect_server_none_when_down(monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(sc.urllib.request, "urlopen", _boom)
    assert sc.detect_server(_cfg()) is None


def test_detect_server_routes_when_store_matches(monkeypatch):
    routes = {
        "/health/ready": _Resp(200, json.dumps({"status": "ok", "project": "p"}).encode()),
        "/config": _Resp(200, json.dumps({"projects_root": r"C:\store"}).encode()),
    }
    monkeypatch.setattr(sc.urllib.request, "urlopen", _fake_urlopen(routes))
    got = sc.detect_server(_cfg(projects_root=r"C:\store"))
    assert got is not None
    assert got["project"] == "p"
    assert got["_api_base"].endswith(":8420")


def test_detect_server_skips_on_store_mismatch(monkeypatch):
    routes = {
        "/health/ready": _Resp(200, json.dumps({"status": "ok"}).encode()),
        "/config": _Resp(200, json.dumps({"projects_root": r"C:\OTHER"}).encode()),
    }
    monkeypatch.setattr(sc.urllib.request, "urlopen", _fake_urlopen(routes))
    assert sc.detect_server(_cfg(projects_root=r"C:\store")) is None


def test_detect_server_skips_on_non_200(monkeypatch):
    routes = {"/health/ready": _Resp(503, b'{"status":"initializing"}')}
    monkeypatch.setattr(sc.urllib.request, "urlopen", _fake_urlopen(routes))
    assert sc.detect_server(_cfg()) is None


def test_detect_server_never_raises_on_mock_config():
    # A MagicMock config (as used by CLI unit tests) has non-int api_port; the
    # probe must fail safe to None, never raise (regression: InvalidURL leaked).
    assert sc.detect_server(mock.MagicMock()) is None


# --- store singleton lock ------------------------------------------------


def test_store_lock_roundtrip(tmp_path, monkeypatch):
    cfg = _cfg(projects_root=str(tmp_path))
    monkeypatch.setattr(
        sc.urllib.request,
        "urlopen",
        _fake_urlopen({"/health/ready": _Resp(200, b'{"status":"ok"}')}),
    )
    assert sc.find_live_server_for_store(cfg) is None  # no lock yet
    sc.write_store_lock(cfg, "0.0.0.0", 8420)
    found = sc.find_live_server_for_store(cfg)
    assert found is not None and found["port"] == 8420 and found["pid"] == __import__("os").getpid()
    sc.release_store_lock(cfg)
    assert sc.find_live_server_for_store(cfg) is None  # released


def test_store_lock_stale_is_ignored(tmp_path, monkeypatch):
    cfg = _cfg(projects_root=str(tmp_path))
    sc.write_store_lock(cfg, "127.0.0.1", 8420)

    def _dead(*a, **k):
        raise OSError("connection refused")  # recorded server is gone

    monkeypatch.setattr(sc.urllib.request, "urlopen", _dead)
    assert sc.find_live_server_for_store(cfg) is None  # stale → treated as absent


def test_release_store_lock_leaves_other_pid(tmp_path):
    import json as _json

    cfg = _cfg(projects_root=str(tmp_path))
    lock = tmp_path / sc._LOCK_NAME
    lock.write_text(_json.dumps({"host": "127.0.0.1", "port": 8420, "pid": 999999}))
    sc.release_store_lock(cfg)  # not our pid → must not delete
    assert lock.exists()


def test_find_live_server_treats_503_as_alive(tmp_path, monkeypatch):
    # A server whose brain is still initializing answers /health/ready with 503.
    # It is ALIVE, so the lock must NOT be treated as stale (else a 2nd server
    # could start during the 1st's startup and race on the store).
    cfg = _cfg(projects_root=str(tmp_path))
    sc.write_store_lock(cfg, "127.0.0.1", 8420)

    def _busy(*a, **k):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:8420/health/ready", 503, "initializing", {}, None
        )

    monkeypatch.setattr(sc.urllib.request, "urlopen", _busy)
    found = sc.find_live_server_for_store(cfg)
    assert found is not None and found["port"] == 8420


def test_detect_server_sends_api_key_on_config_probe(monkeypatch):
    # /config is not on the auth-bypass list; the store-match probe must send
    # X-API-Key or a secured server 401s and routing never engages.
    seen = {}

    def _open(req, timeout=None):
        seen[req.full_url] = {k.lower(): v for k, v in req.header_items()}
        if req.full_url.endswith("/health/ready"):
            return _Resp(200, json.dumps({"status": "ok", "project": "p"}).encode())
        if req.full_url.endswith("/config"):
            return _Resp(200, json.dumps({"projects_root": r"C:\store"}).encode())
        raise AssertionError(req.full_url)

    monkeypatch.setattr(sc.urllib.request, "urlopen", _open)
    got = sc.detect_server(_cfg(api_key="secret", projects_root=r"C:\store"))
    assert got is not None
    cfg_url = next(u for u in seen if u.endswith("/config"))
    assert seen[cfg_url].get("x-api-key") == "secret"
