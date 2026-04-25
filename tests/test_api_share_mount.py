"""REST API tests for the share-mount endpoints added in #51–#54.

Covers the surfaces left untested by the unit tests on the original
branch: /mount/refresh, /share/extend, /share/generate(ttl_days), and
the /query → 503 + X-Axon-Mount-Sync-Pending wiring.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app
from axon.version_marker import MountSyncPendingError

client = TestClient(app, raise_server_exceptions=False)


def _make_brain():
    """Minimal mock brain with the attributes touched by these endpoints."""
    brain = MagicMock()
    brain.config.top_k = 5
    brain.config.hybrid_search = True
    brain.config.rerank = False
    brain.config.hyde = False
    brain.config.multi_query = False
    brain.config.discussion_fallback = True
    brain.config.similarity_threshold = 0.3
    brain.config.step_back = False
    brain.config.query_decompose = False
    brain.config.compress_context = False
    brain._apply_overrides.return_value = brain.config
    return brain


# ---------------------------------------------------------------------------
# POST /mount/refresh
# ---------------------------------------------------------------------------


class TestMountRefreshEndpoint:
    def test_503_when_brain_not_initialized(self):
        api_module.brain = None
        resp = client.post("/mount/refresh")
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"].lower()

    def test_success_no_refresh_needed(self):
        brain = _make_brain()
        brain.refresh_mount.return_value = False
        brain._mount_version_marker = {"seq": 7, "generated_at": "2026-04-25T00:00:00+00:00"}
        api_module.brain = brain

        resp = client.post("/mount/refresh")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["refreshed"] is False
        assert body["seq"] == 7
        brain.refresh_mount.assert_called_once()

    def test_success_with_refresh(self):
        brain = _make_brain()
        brain.refresh_mount.return_value = True
        brain._mount_version_marker = {"seq": 8, "generated_at": "2026-04-25T01:00:00+00:00"}
        api_module.brain = brain

        resp = client.post("/mount/refresh")
        assert resp.status_code == 200
        body = resp.json()
        assert body["refreshed"] is True
        assert body["seq"] == 8

    def test_503_with_sync_pending_header(self):
        brain = _make_brain()
        brain.refresh_mount.side_effect = MountSyncPendingError("indices replicating")
        api_module.brain = brain

        resp = client.post("/mount/refresh")
        assert resp.status_code == 503
        # Dedicated header so REST clients can detect-and-retry without parsing the body.
        assert resp.headers.get("X-Axon-Mount-Sync-Pending") == "true"
        body = resp.json()
        assert body["status"] == "sync_pending"
        assert "replicating" in body["detail"]

    def test_500_for_unrelated_exception(self):
        brain = _make_brain()
        brain.refresh_mount.side_effect = RuntimeError("unrelated boom")
        api_module.brain = brain

        resp = client.post("/mount/refresh")
        assert resp.status_code == 500
        assert "unrelated boom" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /query — MountSyncPendingError → 503 + header
# ---------------------------------------------------------------------------


class TestQuerySyncPending:
    def test_query_returns_503_when_per_query_refresh_raises(self):
        brain = _make_brain()
        brain.query.side_effect = MountSyncPendingError("indices in flight")
        brain._last_provenance = {}
        api_module.brain = brain

        resp = client.post("/query", json={"query": "hello"})
        assert resp.status_code == 503
        assert resp.headers.get("X-Axon-Mount-Sync-Pending") == "true"
        body = resp.json()
        assert body["status"] == "sync_pending"
        assert "in flight" in body["detail"]


# ---------------------------------------------------------------------------
# POST /share/generate — ttl_days passthrough
# ---------------------------------------------------------------------------


class TestShareGenerateTtlDays:
    def _wire_brain_with_project(self, tmp_path):
        brain = _make_brain()
        brain.config.axon_store_mode = True
        brain.config.axon_store_dir = str(tmp_path)
        brain.config.projects_root = str(tmp_path / "Workspace")
        api_module.brain = brain
        proj = tmp_path / "Workspace" / "myproject"
        proj.mkdir(parents=True)
        (proj / "meta.json").write_text('{"name": "myproject"}', encoding="utf-8")
        return brain

    def test_ttl_days_passthrough_to_shares_module(self, tmp_path):
        self._wire_brain_with_project(tmp_path)
        fake_result = {
            "key_id": "sk_x",
            "share_string": "abc",
            "project": "myproject",
            "grantee": "bob",
            "owner": "alice",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        with patch("axon.shares.generate_share_key", return_value=fake_result) as gen_mock:
            resp = client.post(
                "/share/generate",
                json={"project": "myproject", "grantee": "bob", "ttl_days": 7},
            )
        assert resp.status_code == 200
        # Endpoint must forward ttl_days into the underlying shares.generate_share_key call.
        _, kwargs = gen_mock.call_args
        assert kwargs.get("ttl_days") == 7
        assert resp.json()["expires_at"] == "2099-01-01T00:00:00+00:00"

    def test_ttl_days_omitted_defaults_to_none(self, tmp_path):
        self._wire_brain_with_project(tmp_path)
        fake_result = {
            "key_id": "sk_x",
            "share_string": "abc",
            "project": "myproject",
            "grantee": "bob",
            "owner": "alice",
            "expires_at": None,
        }
        with patch("axon.shares.generate_share_key", return_value=fake_result) as gen_mock:
            resp = client.post(
                "/share/generate",
                json={"project": "myproject", "grantee": "bob"},
            )
        assert resp.status_code == 200
        _, kwargs = gen_mock.call_args
        assert kwargs.get("ttl_days") is None

    def test_invalid_ttl_days_returns_422(self, tmp_path):
        self._wire_brain_with_project(tmp_path)
        with patch(
            "axon.shares.generate_share_key",
            side_effect=ValueError("ttl_days must be a positive integer"),
        ):
            resp = client.post(
                "/share/generate",
                json={"project": "myproject", "grantee": "bob", "ttl_days": 0},
            )
        assert resp.status_code == 422
        assert "positive integer" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /share/extend
# ---------------------------------------------------------------------------


class TestShareExtendEndpoint:
    def test_success_returns_new_expires_at(self, tmp_path):
        brain = _make_brain()
        brain.config.projects_root = str(tmp_path)
        api_module.brain = brain
        fake_result = {
            "key_id": "sk_a",
            "project": "myproject",
            "grantee": "bob",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        with patch("axon.shares.extend_share_key", return_value=fake_result) as ext_mock:
            resp = client.post("/share/extend", json={"key_id": "sk_a", "ttl_days": 30})
        assert resp.status_code == 200
        _, kwargs = ext_mock.call_args
        assert kwargs.get("ttl_days") == 30
        assert kwargs.get("key_id") == "sk_a"
        assert resp.json()["expires_at"] == "2099-01-01T00:00:00+00:00"

    def test_clear_expiry_with_null_ttl(self, tmp_path):
        brain = _make_brain()
        brain.config.projects_root = str(tmp_path)
        api_module.brain = brain
        with patch(
            "axon.shares.extend_share_key",
            return_value={"key_id": "sk_a", "project": "p", "grantee": "g", "expires_at": None},
        ) as ext_mock:
            resp = client.post("/share/extend", json={"key_id": "sk_a", "ttl_days": None})
        assert resp.status_code == 200
        _, kwargs = ext_mock.call_args
        assert kwargs.get("ttl_days") is None
        assert resp.json()["expires_at"] is None

    def test_unknown_key_returns_404(self, tmp_path):
        brain = _make_brain()
        brain.config.projects_root = str(tmp_path)
        api_module.brain = brain
        with patch(
            "axon.shares.extend_share_key",
            side_effect=ValueError("Key 'sk_missing' not found."),
        ):
            resp = client.post("/share/extend", json={"key_id": "sk_missing", "ttl_days": 7})
        assert resp.status_code == 404

    def test_revoked_key_returns_409(self, tmp_path):
        brain = _make_brain()
        brain.config.projects_root = str(tmp_path)
        api_module.brain = brain
        with patch(
            "axon.shares.extend_share_key",
            side_effect=ValueError("Key 'sk_a' is revoked; revoked keys cannot be extended."),
        ):
            resp = client.post("/share/extend", json={"key_id": "sk_a", "ttl_days": 7})
        assert resp.status_code == 409

    def test_invalid_ttl_returns_422(self, tmp_path):
        brain = _make_brain()
        brain.config.projects_root = str(tmp_path)
        api_module.brain = brain
        with patch(
            "axon.shares.extend_share_key",
            side_effect=ValueError("ttl_days must be a positive integer"),
        ):
            resp = client.post("/share/extend", json={"key_id": "sk_a", "ttl_days": -1})
        assert resp.status_code == 422
