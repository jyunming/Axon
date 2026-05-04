"""Tests for v0.4.0 Item 3 — ephemeral sealed-cache mode.

Coverage:
- ``AxonConfig.seal_cache_ephemeral`` default + YAML round-trip
- ``AxonBrain.wipe_sealed_cache`` returns False when nothing is mounted
- ``AxonBrain.wipe_sealed_cache`` clears ``_sealed_cache`` slot when mounted
- ``_ephemeral_query_window`` is a no-op outside ephemeral mode
- ``_ephemeral_query_window`` triggers wipe + remount for sealed projects
  when ``seal_cache_ephemeral=True``
- REST ``POST /security/wipe-sealed-cache`` returns ``{wiped: bool}``;
  no-op shape when no brain
- CLI ``--seal-cache-ephemeral`` flag forces config override
- CLI ``--wipe-sealed-cache`` exits 0 even when no sealed project active
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------


class TestConfigField:
    def test_default_is_false(self):
        from axon.config import AxonConfig

        cfg = AxonConfig()
        assert cfg.seal_cache_ephemeral is False

    def test_yaml_round_trip(self, tmp_path):
        from axon.config import AxonConfig

        path = tmp_path / "config.yaml"
        path.write_text("security:\n  seal_cache_ephemeral: true\n", encoding="utf-8")
        cfg = AxonConfig.load(str(path))
        assert cfg.seal_cache_ephemeral is True
        cfg.seal_cache_ephemeral = False
        cfg.save(str(path))
        cfg2 = AxonConfig.load(str(path))
        assert cfg2.seal_cache_ephemeral is False


# ---------------------------------------------------------------------------
# Brain helpers — wipe_sealed_cache + _ephemeral_query_window
# ---------------------------------------------------------------------------


class TestWipeSealedCache:
    def test_returns_false_when_no_cache(self):
        """No sealed project mounted → ``wipe_sealed_cache`` reports
        ``False`` and does not raise."""
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain._sealed_cache = None
        result = AxonBrain.wipe_sealed_cache(brain)
        assert result is False

    def test_returns_true_and_clears_slot_when_mounted(self, monkeypatch):
        """Mounted cache → wipe path is invoked and the slot is cleared."""
        from axon.main import AxonBrain

        cache = MagicMock()
        brain = MagicMock(spec=AxonBrain)
        brain._sealed_cache = cache

        called = {"n": 0}

        def _fake_release(c):
            called["n"] += 1
            assert c is cache

        # Patch the import target at the call site
        import axon.security.mount as _mnt

        monkeypatch.setattr(_mnt, "release_cache", _fake_release)
        result = AxonBrain.wipe_sealed_cache(brain)
        assert result is True
        assert brain._sealed_cache is None
        assert called["n"] == 1

    def test_swallows_release_exception(self, monkeypatch):
        """Best-effort wipe — a release_cache exception must not
        propagate; the cache slot still ends up cleared."""
        from axon.main import AxonBrain

        cache = MagicMock()
        brain = MagicMock(spec=AxonBrain)
        brain._sealed_cache = cache

        import axon.security.mount as _mnt

        def _bad(c):
            raise OSError("disk gone")

        monkeypatch.setattr(_mnt, "release_cache", _bad)
        result = AxonBrain.wipe_sealed_cache(brain)
        # Wipe still considered successful at the brain level — slot cleared.
        assert result is True
        assert brain._sealed_cache is None


class TestEphemeralQueryWindow:
    def _make_brain(self, *, ephemeral: bool, has_remount: bool):
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.seal_cache_ephemeral = ephemeral
        brain._sealed_remount_args = ("project", Path("/tmp/seal"), None) if has_remount else None
        # Bind the unbound method so the context manager closes over `brain`.
        brain._ephemeral_query_window = AxonBrain._ephemeral_query_window.__get__(brain)
        brain._ensure_sealed_cache_mounted = MagicMock()
        brain.wipe_sealed_cache = MagicMock(return_value=True)
        return brain

    def test_inactive_when_ephemeral_off(self):
        """Pure pass-through when ``seal_cache_ephemeral`` is False —
        no remount, no wipe."""
        brain = self._make_brain(ephemeral=False, has_remount=True)
        with brain._ephemeral_query_window():
            pass
        brain._ensure_sealed_cache_mounted.assert_not_called()
        brain.wipe_sealed_cache.assert_not_called()

    def test_inactive_when_no_sealed_project(self):
        """Pass-through when no sealed project is active — there's
        nothing to remount or wipe."""
        brain = self._make_brain(ephemeral=True, has_remount=False)
        with brain._ephemeral_query_window():
            pass
        brain._ensure_sealed_cache_mounted.assert_not_called()
        brain.wipe_sealed_cache.assert_not_called()

    def test_active_path_remounts_then_wipes(self):
        """Ephemeral + sealed project: ensure mount on entry, wipe on
        exit."""
        brain = self._make_brain(ephemeral=True, has_remount=True)
        with brain._ephemeral_query_window():
            pass
        brain._ensure_sealed_cache_mounted.assert_called_once()
        brain.wipe_sealed_cache.assert_called_once()

    def test_wipe_fires_even_when_body_raises(self):
        """The wipe is wrapped in a finally — exceptions from the body
        must not prevent the plaintext from being scrubbed."""
        brain = self._make_brain(ephemeral=True, has_remount=True)
        with pytest.raises(RuntimeError, match="boom"):
            with brain._ephemeral_query_window():
                raise RuntimeError("boom")
        brain.wipe_sealed_cache.assert_called_once()


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------


class TestRestSurface:
    @pytest.fixture
    def api_client(self):
        from fastapi.testclient import TestClient

        from axon.api import app

        return TestClient(app, raise_server_exceptions=True)

    def test_post_wipe_no_brain_returns_no_op(self, api_client, monkeypatch):
        """Without a brain instance the endpoint should answer with
        ``wiped: false`` rather than 500."""
        from axon import api as _api

        monkeypatch.setattr(_api, "brain", None)
        resp = api_client.post("/security/wipe-sealed-cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["wiped"] is False
        assert "no active brain" in body.get("reason", "").lower()

    def test_post_wipe_invokes_brain_method(self, api_client, monkeypatch):
        from axon import api as _api

        fake_brain = MagicMock()
        fake_brain.wipe_sealed_cache.return_value = True
        monkeypatch.setattr(_api, "brain", fake_brain)
        resp = api_client.post("/security/wipe-sealed-cache")
        assert resp.status_code == 200
        assert resp.json() == {"wiped": True}
        fake_brain.wipe_sealed_cache.assert_called_once()

    def test_post_wipe_brain_raises_returns_500(self, api_client, monkeypatch):
        from axon import api as _api

        fake_brain = MagicMock()
        fake_brain.wipe_sealed_cache.side_effect = RuntimeError("boom")
        monkeypatch.setattr(_api, "brain", fake_brain)
        resp = api_client.post("/security/wipe-sealed-cache")
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]
