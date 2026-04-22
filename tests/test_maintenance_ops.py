"""tests/test_maintenance_ops.py — Tests for Axon maintenance state machine.

Covers:
- apply_maintenance_state() for all valid states and error cases
- get_maintenance_status() snapshot shape and content
- /collection/stale API endpoint (stale-doc filtering)
- Runtime lease behaviour under maintenance states
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def isolated_projects(tmp_path, monkeypatch):
    """Redirect PROJECTS_ROOT to tmp_path so tests never touch ~/.axon."""
    import axon.projects as _p

    monkeypatch.setattr(_p, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(_p, "_ACTIVE_FILE", tmp_path / ".active_project")
    return tmp_path


@pytest.fixture()
def fresh_reg():
    """Return a brand-new LeaseRegistry, isolated from the process singleton."""
    from axon.runtime import LeaseRegistry

    return LeaseRegistry()


@pytest.fixture()
def proj(isolated_projects):
    """Create a 'testproj' project and return its name."""
    from axon.projects import ensure_project

    ensure_project("testproj")
    return "testproj"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _apply(proj_name: str, state: str, reg):
    """Call apply_maintenance_state() with the given registry injected."""
    with patch("axon.runtime.get_registry", return_value=reg):
        from axon.maintenance import apply_maintenance_state

        return apply_maintenance_state(proj_name, state)


def _get_status(proj_name: str, reg):
    """Call get_maintenance_status() with the given registry injected."""
    with patch("axon.runtime.get_registry", return_value=reg):
        from axon.maintenance import get_maintenance_status

        return get_maintenance_status(proj_name)


# ──────────────────────────────────────────────────────────────────────────────
# apply_maintenance_state() — state machine transitions
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyMaintenanceState:
    _EXPECTED_KEYS = {"status", "project", "maintenance_state", "active_leases", "epoch"}

    def test_draining_returns_correct_keys(self, proj, fresh_reg):
        result = _apply(proj, "draining", fresh_reg)
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_draining_status_and_fields(self, proj, fresh_reg):
        result = _apply(proj, "draining", fresh_reg)
        assert result["status"] == "ok"
        assert result["project"] == proj
        assert result["maintenance_state"] == "draining"

    def test_draining_starts_drain_in_registry(self, proj, fresh_reg):
        _apply(proj, "draining", fresh_reg)
        assert fresh_reg.snapshot(proj)["draining"] is True

    def test_normal_stops_drain(self, proj, fresh_reg):
        _apply(proj, "draining", fresh_reg)
        assert fresh_reg.snapshot(proj)["draining"] is True

        result = _apply(proj, "normal", fresh_reg)
        assert result["maintenance_state"] == "normal"
        assert fresh_reg.snapshot(proj)["draining"] is False

    def test_readonly_starts_drain(self, proj, fresh_reg):
        result = _apply(proj, "readonly", fresh_reg)
        assert result["maintenance_state"] == "readonly"
        assert fresh_reg.snapshot(proj)["draining"] is True

    def test_offline_starts_drain(self, proj, fresh_reg):
        result = _apply(proj, "offline", fresh_reg)
        assert result["maintenance_state"] == "offline"
        assert fresh_reg.snapshot(proj)["draining"] is True

    def test_invalid_state_raises_value_error(self, proj, fresh_reg):
        with pytest.raises(ValueError, match="Invalid maintenance state"):
            _apply(proj, "banana", fresh_reg)

    def test_nonexistent_project_raises_value_error(self, isolated_projects, fresh_reg):
        with pytest.raises(ValueError, match="does not exist"):
            _apply("ghost-project", "draining", fresh_reg)

    def test_active_leases_reflected_in_result(self, proj, fresh_reg):
        # Acquire a lease before starting the drain so active count is 1
        lease = fresh_reg.acquire(proj)
        result = _apply(proj, "draining", fresh_reg)
        assert result["active_leases"] == 1
        lease.close()

    def test_epoch_included_in_result(self, proj, fresh_reg):
        fresh_reg.bump_epoch(proj)
        result = _apply(proj, "draining", fresh_reg)
        assert result["epoch"] == 1

    def test_state_persisted_to_meta_json(self, proj, isolated_projects, fresh_reg):
        """apply_maintenance_state must durably persist the state in meta.json."""
        _apply(proj, "readonly", fresh_reg)

        from axon.projects import get_maintenance_state

        assert get_maintenance_state(proj) == "readonly"

    def test_roundtrip_normal_after_offline(self, proj, fresh_reg):
        _apply(proj, "offline", fresh_reg)
        assert fresh_reg.snapshot(proj)["draining"] is True

        _apply(proj, "normal", fresh_reg)
        assert fresh_reg.snapshot(proj)["draining"] is False

        from axon.projects import get_maintenance_state

        assert get_maintenance_state(proj) == "normal"


# ──────────────────────────────────────────────────────────────────────────────
# get_maintenance_status() — snapshot content
# ──────────────────────────────────────────────────────────────────────────────


class TestGetMaintenanceStatus:
    _EXPECTED_KEYS = {"project", "maintenance_state", "active_leases", "epoch", "draining"}

    def test_returns_correct_keys(self, proj, fresh_reg):
        status = _get_status(proj, fresh_reg)
        assert set(status.keys()) == self._EXPECTED_KEYS

    def test_default_state_is_normal(self, proj, fresh_reg):
        status = _get_status(proj, fresh_reg)
        assert status["maintenance_state"] == "normal"
        assert status["draining"] is False

    def test_reflects_draining_after_apply(self, proj, fresh_reg):
        _apply(proj, "draining", fresh_reg)
        status = _get_status(proj, fresh_reg)
        assert status["maintenance_state"] == "draining"
        assert status["draining"] is True

    def test_reflects_readonly_state(self, proj, fresh_reg):
        _apply(proj, "readonly", fresh_reg)
        status = _get_status(proj, fresh_reg)
        assert status["maintenance_state"] == "readonly"
        assert status["draining"] is True

    def test_project_name_in_result(self, proj, fresh_reg):
        status = _get_status(proj, fresh_reg)
        assert status["project"] == proj

    def test_active_leases_reflected(self, proj, fresh_reg):
        lease = fresh_reg.acquire(proj)
        status = _get_status(proj, fresh_reg)
        assert status["active_leases"] == 1
        lease.close()

    def test_epoch_reflected(self, proj, fresh_reg):
        fresh_reg.bump_epoch(proj)
        fresh_reg.bump_epoch(proj)
        status = _get_status(proj, fresh_reg)
        assert status["epoch"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# /collection/stale endpoint — stale-doc filtering
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def stale_client():
    """TestClient with ingest router mounted; _source_hashes cleared before/after."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from axon import api as _api
    from axon.api_routes.ingest import router as ingest_router

    app = FastAPI()
    app.include_router(ingest_router)

    _api._source_hashes.clear()
    yield TestClient(app)
    _api._source_hashes.clear()


def _seed_hash(project: str, doc_id: str, days_ago: float) -> None:
    """Insert a fake ingestion record into api._source_hashes."""
    from axon import api as _api

    ts = datetime.now(timezone.utc).timestamp() - days_ago * 86_400
    dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    _api._source_hashes.setdefault(project, {})[f"hash_{doc_id}"] = {
        "doc_id": doc_id,
        "last_ingested_at": dt_str,
    }


class TestGetStaleDocs:
    def test_empty_store_returns_zero(self, stale_client):
        resp = stale_client.get("/collection/stale?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["stale_docs"] == []

    def test_old_doc_returned(self, stale_client):
        _seed_hash("proj1", "doc-old", days_ago=10)
        resp = stale_client.get("/collection/stale?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["stale_docs"][0]["doc_id"] == "doc-old"

    def test_recent_doc_not_returned(self, stale_client):
        _seed_hash("proj1", "doc-new", days_ago=2)
        resp = stale_client.get("/collection/stale?days=7")
        body = resp.json()
        assert body["total"] == 0

    def test_days_zero_returns_all_docs(self, stale_client):
        """days=0 means cutoff=now; everything older than the instant is stale."""
        _seed_hash("proj1", "doc-a", days_ago=0.01)  # ~15 minutes ago
        resp = stale_client.get("/collection/stale?days=0")
        body = resp.json()
        assert body["total"] >= 1

    def test_mixed_ages_across_projects(self, stale_client):
        _seed_hash("alpha", "doc-stale", days_ago=15)
        _seed_hash("beta", "doc-fresh", days_ago=1)
        resp = stale_client.get("/collection/stale?days=7")
        body = resp.json()
        ids = [d["doc_id"] for d in body["stale_docs"]]
        assert "doc-stale" in ids
        assert "doc-fresh" not in ids

    def test_threshold_days_in_response(self, stale_client):
        resp = stale_client.get("/collection/stale?days=14")
        body = resp.json()
        assert body["threshold_days"] == 14

    def test_negative_days_returns_400(self, stale_client):
        resp = stale_client.get("/collection/stale?days=-1")
        assert resp.status_code == 400

    def test_stale_doc_has_expected_keys(self, stale_client):
        _seed_hash("myproj", "doc-xyz", days_ago=30)
        resp = stale_client.get("/collection/stale?days=7")
        body = resp.json()
        assert body["total"] == 1
        doc = body["stale_docs"][0]
        assert set(doc.keys()) >= {"doc_id", "project", "last_ingested_at", "age_days"}

    def test_age_days_approximate(self, stale_client):
        _seed_hash("myproj", "doc-ten", days_ago=10)
        resp = stale_client.get("/collection/stale?days=7")
        body = resp.json()
        doc = body["stale_docs"][0]
        # age_days should be close to 10 (within 0.5 day tolerance)
        assert abs(doc["age_days"] - 10) < 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Runtime lease behaviour under maintenance states
# ──────────────────────────────────────────────────────────────────────────────


class TestRuntimeLeaseUnderMaintenance:
    def test_write_blocked_when_draining(self, proj, fresh_reg):
        _apply(proj, "draining", fresh_reg)
        with pytest.raises(PermissionError, match="draining"):
            fresh_reg.acquire(proj)

    def test_write_blocked_when_readonly(self, proj, fresh_reg):
        """readonly triggers drain coordination — new writes must be blocked."""
        _apply(proj, "readonly", fresh_reg)
        with pytest.raises(PermissionError, match="draining"):
            fresh_reg.acquire(proj)

    def test_write_blocked_when_offline(self, proj, fresh_reg):
        _apply(proj, "offline", fresh_reg)
        with pytest.raises(PermissionError, match="draining"):
            fresh_reg.acquire(proj)

    def test_write_restored_after_normal(self, proj, fresh_reg):
        _apply(proj, "draining", fresh_reg)
        _apply(proj, "normal", fresh_reg)
        lease = fresh_reg.acquire(proj)
        assert fresh_reg.active_lease_count(proj) == 1
        lease.close()

    def test_read_snapshot_available_in_readonly_state(self, proj, fresh_reg):
        """Registry snapshot (read-path) must stay accessible during drain."""
        _apply(proj, "readonly", fresh_reg)
        snap = fresh_reg.snapshot(proj)
        assert snap["draining"] is True
        assert snap["project"] == proj

    def test_inflight_write_completes_after_drain_started(self, proj, fresh_reg):
        """A lease acquired *before* drain must close cleanly."""
        lease = fresh_reg.acquire(proj)
        _apply(proj, "draining", fresh_reg)
        # In-flight write closes without error
        lease.close()
        assert fresh_reg.active_lease_count(proj) == 0

    def test_drain_completes_when_inflight_lease_released(self, proj, fresh_reg):
        lease = fresh_reg.acquire(proj)
        _apply(proj, "draining", fresh_reg)

        # Drain not yet done while lease is held
        assert fresh_reg.wait_for_drain(proj, timeout=0.05) is False
        lease.close()
        assert fresh_reg.wait_for_drain(proj, timeout=1.0) is True

    def test_default_project_write_always_allowed_regardless_of_drain(self, fresh_reg):
        """The 'default' project is exempt from drain mechanics."""
        fresh_reg.start_drain("default")
        # Must not raise
        lease = fresh_reg.acquire("default")
        assert isinstance(lease, object)
        lease.close()
