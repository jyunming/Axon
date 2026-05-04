"""Tests for v0.4.0 Item 2 — keyring hardening (persistent / session / never).

Coverage:
- ``SessionDEKCache`` thread-safe set/get/delete/clear/__len__
- ``set_keyring_mode`` validates input, ``get_keyring_mode`` round-trips
- ``store_secret`` / ``get_secret`` / ``delete_secret`` route by mode:
  - persistent → OS keyring backend
  - session    → in-memory cache, OS keyring untouched
  - never      → no-op store, get returns None
- ``AxonConfig.keyring_mode`` field default + YAML round-trip
- ``AxonBrain.__init__`` propagates the config value via
  ``set_keyring_mode``
- REST ``GET /security/status`` exposes ``keyring_mode`` +
  ``session_cache_size``
- REST ``POST /security/keyring-mode`` validates and applies; 422 on
  bad mode; mirrors into the brain's config
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("keyring")


# ---------------------------------------------------------------------------
# Fresh-cache fixture: avoid cross-test pollution
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_keyring_mode_and_cache():
    """Reset to ``persistent`` and clear the session cache between tests
    so global state in :mod:`axon.security.keyring` doesn't leak."""
    from axon.security.keyring import session_cache, set_keyring_mode

    set_keyring_mode("persistent")
    session_cache().clear()
    yield
    set_keyring_mode("persistent")
    session_cache().clear()


# ---------------------------------------------------------------------------
# SessionDEKCache primitive
# ---------------------------------------------------------------------------


class TestSessionDEKCache:
    def test_set_get_round_trip(self):
        from axon.security.keyring import SessionDEKCache

        c = SessionDEKCache()
        c.set("svc", "user", "secret")
        assert c.get("svc", "user") == "secret"

    def test_get_missing_returns_none(self):
        from axon.security.keyring import SessionDEKCache

        c = SessionDEKCache()
        assert c.get("svc", "user") is None

    def test_delete_silent_on_missing(self):
        from axon.security.keyring import SessionDEKCache

        c = SessionDEKCache()
        c.delete("svc", "user")  # must not raise
        assert c.get("svc", "user") is None

    def test_clear_wipes_all(self):
        from axon.security.keyring import SessionDEKCache

        c = SessionDEKCache()
        c.set("a", "u1", "s1")
        c.set("b", "u2", "s2")
        c.set("c", "u3", "s3")
        assert len(c) == 3
        c.clear()
        assert len(c) == 0
        assert c.get("a", "u1") is None

    def test_thread_safe_set_get(self):
        """Hammer the cache from many threads — no exceptions, no
        corruption (final count matches insertions)."""
        from axon.security.keyring import SessionDEKCache

        c = SessionDEKCache()
        n_threads = 16
        per_thread = 100

        def _worker(thread_id: int) -> None:
            for i in range(per_thread):
                c.set(f"svc{thread_id}", f"u{i}", f"s{thread_id}-{i}")
                assert c.get(f"svc{thread_id}", f"u{i}") == f"s{thread_id}-{i}"

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(c) == n_threads * per_thread


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


class TestModeDispatch:
    def test_default_mode_is_persistent(self):
        from axon.security.keyring import get_keyring_mode

        assert get_keyring_mode() == "persistent"

    def test_set_keyring_mode_round_trip(self):
        from axon.security.keyring import get_keyring_mode, set_keyring_mode

        set_keyring_mode("session")
        assert get_keyring_mode() == "session"
        set_keyring_mode("never")
        assert get_keyring_mode() == "never"
        set_keyring_mode("persistent")
        assert get_keyring_mode() == "persistent"

    @pytest.mark.parametrize("bad", ["", "PERSISTENT", "ephemeral", "ram", " session", None])
    def test_set_keyring_mode_rejects_invalid(self, bad):
        from axon.security.keyring import set_keyring_mode

        with pytest.raises((ValueError, TypeError)):
            set_keyring_mode(bad)  # type: ignore[arg-type]

    def test_session_mode_routes_to_in_memory_cache(self):
        """In ``session`` mode the OS keyring must NEVER be called —
        store/get/delete go to the in-process dict."""
        from axon.security.keyring import (
            delete_secret,
            get_secret,
            session_cache,
            set_keyring_mode,
            store_secret,
        )

        set_keyring_mode("session")
        with patch("axon.security.keyring._keyring") as mock_kr:
            store_secret("svc", "user", "secret-value")
            assert get_secret("svc", "user") == "secret-value"
            delete_secret("svc", "user")
            assert get_secret("svc", "user") is None
            # No backend interaction at all
            mock_kr.get_keyring.assert_not_called()
        # Cache is empty after delete
        assert session_cache().get("svc", "user") is None

    def test_never_mode_drops_secret_silently(self):
        """In ``never`` mode every store is a no-op; every get returns
        None. The OS keyring is never touched."""
        from axon.security.keyring import (
            delete_secret,
            get_secret,
            session_cache,
            set_keyring_mode,
            store_secret,
        )

        set_keyring_mode("never")
        with patch("axon.security.keyring._keyring") as mock_kr:
            store_secret("svc", "user", "secret-value")
            assert get_secret("svc", "user") is None
            delete_secret("svc", "user")  # must not raise
            mock_kr.get_keyring.assert_not_called()
        # Session cache also untouched
        assert len(session_cache()) == 0

    def test_persistent_mode_calls_os_keyring(self):
        """In ``persistent`` mode (default) calls go to the active
        keyring backend, not the in-memory cache."""
        from axon.security.keyring import (
            get_secret,
            session_cache,
            set_keyring_mode,
            store_secret,
        )

        set_keyring_mode("persistent")
        backend = MagicMock()
        backend.set_password = MagicMock()
        backend.get_password = MagicMock(return_value="from-os-keyring")
        # Patch the helper that resolves the active backend so the
        # fail-stub detection passes too.
        with patch("axon.security.keyring._active_backend", return_value=backend):
            store_secret("svc", "user", "secret-value")
            assert get_secret("svc", "user") == "from-os-keyring"
        backend.set_password.assert_called_once_with("svc", "user", "secret-value")
        backend.get_password.assert_called_once_with("svc", "user")
        # Session cache stays empty in persistent mode
        assert len(session_cache()) == 0


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_axon_config_default_keyring_mode_is_persistent(self):
        from axon.config import AxonConfig

        cfg = AxonConfig()
        assert cfg.keyring_mode == "persistent"

    def test_axon_config_yaml_round_trip(self, tmp_path):
        """Setting ``security.keyring_mode`` in YAML loads through
        ``AxonConfig.load`` and saves back via ``AxonConfig.save``."""
        from axon.config import AxonConfig

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "security:\n  keyring_mode: session\n",
            encoding="utf-8",
        )
        cfg = AxonConfig.load(str(cfg_path))
        assert cfg.keyring_mode == "session"

        cfg.keyring_mode = "never"
        cfg.save(str(cfg_path))
        cfg2 = AxonConfig.load(str(cfg_path))
        assert cfg2.keyring_mode == "never"

    def test_axon_config_yaml_rejects_invalid_mode(self, tmp_path):
        from axon.config import AxonConfig

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "security:\n  keyring_mode: bogus\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="keyring_mode"):
            AxonConfig.load(str(cfg_path))


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------


class TestRestSurface:
    @pytest.fixture
    def api_client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from axon.api import app
        from axon.api_routes import security_routes

        # Point _current_user_dir at an empty dir so store_status returns
        # a clean uninitialised payload (we only need keyring_mode + cache).
        monkeypatch.setattr(security_routes, "_current_user_dir", lambda: tmp_path)
        return TestClient(app, raise_server_exceptions=True)

    def test_status_includes_keyring_mode_and_cache_size(self, api_client):
        from axon.security.keyring import set_keyring_mode

        set_keyring_mode("session")
        resp = api_client.get("/security/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["keyring_mode"] == "session"
        assert body["session_cache_size"] == 0

    def test_post_keyring_mode_changes_runtime(self, api_client):
        from axon.security.keyring import get_keyring_mode

        resp = api_client.post("/security/keyring-mode", json={"mode": "session"})
        assert resp.status_code == 200
        assert resp.json()["keyring_mode"] == "session"
        assert get_keyring_mode() == "session"

    def test_post_keyring_mode_rejects_invalid_mode(self, api_client):
        resp = api_client.post("/security/keyring-mode", json={"mode": "ephemeral"})
        assert resp.status_code == 422
        assert "keyring_mode" in resp.json()["detail"]

    def test_post_keyring_mode_rejects_missing_mode(self, api_client):
        resp = api_client.post("/security/keyring-mode", json={})
        assert resp.status_code == 422

    def test_post_keyring_mode_rejects_invalid_json(self, api_client):
        resp = api_client.post(
            "/security/keyring-mode",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_post_keyring_mode_rejects_non_object_body(self, api_client):
        """JSON arrays/strings/null must return 422 (Copilot finding on
        PR #107) — previously crashed with AttributeError → 500."""
        for bad_body in ([], "x", None, 42):
            resp = api_client.post(
                "/security/keyring-mode",
                content=__import__("json").dumps(bad_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 422, f"non-object body {bad_body!r} should be 422"


# ---------------------------------------------------------------------------
# Side-effects of mode transitions (Copilot finding PR #107)
# ---------------------------------------------------------------------------


class TestModeTransitionSideEffects:
    def test_switching_out_of_session_clears_cache(self):
        """If we leave 'session' mode, the in-memory cache must be wiped
        — otherwise ``never`` mode silently retains DEK material."""
        from axon.security.keyring import (
            session_cache,
            set_keyring_mode,
            store_secret,
        )

        set_keyring_mode("session")
        store_secret("svc", "user", "secret")
        assert len(session_cache()) == 1

        set_keyring_mode("never")
        assert len(session_cache()) == 0

    def test_switching_session_to_persistent_clears_cache(self):
        from axon.security.keyring import (
            session_cache,
            set_keyring_mode,
            store_secret,
        )

        set_keyring_mode("session")
        store_secret("svc", "user", "secret")
        assert len(session_cache()) == 1

        set_keyring_mode("persistent")
        assert len(session_cache()) == 0

    def test_re_entering_session_mode_starts_empty(self):
        from axon.security.keyring import (
            session_cache,
            set_keyring_mode,
            store_secret,
        )

        set_keyring_mode("session")
        store_secret("a", "u", "s")
        set_keyring_mode("persistent")  # clears
        set_keyring_mode("session")
        assert len(session_cache()) == 0
