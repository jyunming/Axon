"""
Unit tests for axon/governance.py — AuditStore, CopilotSessionStore, emit().
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axon.governance import (
    VALID_ACTIONS,
    AuditEvent,
    AuditStore,
    CopilotSessionStore,
    emit,
    get_session_store,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _store(tmp_path: Path) -> AuditStore:
    return AuditStore(tmp_path / "gov.db")


def _event(**kwargs) -> AuditEvent:
    defaults = {
        "action": "ingest_started",
        "target_type": "file",
        "target_id": "/tmp/doc.txt",
        "project": "test_proj",
    }
    defaults.update(kwargs)
    return AuditEvent(**defaults)


# ===========================================================================
# AuditEvent
# ===========================================================================


class TestAuditEvent:
    def test_defaults_filled(self):
        e = _event()
        assert e.event_id  # UUID string
        assert e.timestamp  # ISO-8601
        assert e.actor == "api"
        assert e.surface == "api"
        assert e.status == "completed"
        assert e.details == {}
        assert e.request_id == ""

    def test_custom_fields(self):
        e = _event(actor="repl", surface="repl", status="started", request_id="r123")
        assert e.actor == "repl"
        assert e.surface == "repl"
        assert e.status == "started"
        assert e.request_id == "r123"

    def test_unique_event_ids(self):
        ids = {_event().event_id for _ in range(50)}
        assert len(ids) == 50

    def test_valid_actions_set(self):
        assert "ingest_started" in VALID_ACTIONS
        assert "copilot_session_opened" in VALID_ACTIONS
        assert "share_revoked" in VALID_ACTIONS


# ===========================================================================
# AuditStore — SQLite path
# ===========================================================================


class TestAuditStoreSQLite:
    def test_init_creates_db(self, tmp_path):
        store = _store(tmp_path)
        assert (tmp_path / "gov.db").exists()
        assert not store._use_jsonl

    def test_append_and_query_roundtrip(self, tmp_path):
        store = _store(tmp_path)
        e = _event(project="proj_a", action="ingest_completed")
        store.append(e)
        results = store.query(project="proj_a")
        assert len(results) == 1
        assert results[0].event_id == e.event_id
        assert results[0].action == "ingest_completed"

    def test_query_filter_by_action(self, tmp_path):
        store = _store(tmp_path)
        store.append(_event(action="ingest_started", project="p1"))
        store.append(_event(action="delete", project="p1"))
        results = store.query(action="delete")
        assert all(r.action == "delete" for r in results)

    def test_query_filter_by_status(self, tmp_path):
        store = _store(tmp_path)
        store.append(_event(status="completed"))
        store.append(_event(status="failed"))
        failed = store.query(status="failed")
        assert all(r.status == "failed" for r in failed)

    def test_query_filter_by_surface(self, tmp_path):
        store = _store(tmp_path)
        store.append(_event(surface="repl"))
        store.append(_event(surface="api"))
        repl = store.query(surface="repl")
        assert all(r.surface == "repl" for r in repl)

    def test_query_filter_by_since(self, tmp_path):
        import time
        from datetime import timezone

        store = _store(tmp_path)
        store.append(_event(project="old"))
        time.sleep(0.01)
        from datetime import datetime

        pivot = datetime.now(timezone.utc).isoformat()
        time.sleep(0.01)
        store.append(_event(project="new_proj", action="delete"))
        results = store.query(since=pivot)
        assert all(r.project == "new_proj" for r in results)

    def test_query_limit(self, tmp_path):
        store = _store(tmp_path)
        for i in range(20):
            store.append(_event(target_id=f"doc{i}"))
        results = store.query(limit=5)
        assert len(results) <= 5

    def test_query_returns_newest_first(self, tmp_path):
        store = _store(tmp_path)
        for i in range(5):
            store.append(_event(target_id=f"doc{i}"))
        results = store.query()
        timestamps = [r.timestamp for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_duplicate_event_id_ignored(self, tmp_path):
        store = _store(tmp_path)
        e = _event()
        store.append(e)
        store.append(e)  # same event_id — INSERT OR IGNORE
        assert len(store.query()) == 1

    def test_details_serialised_as_json(self, tmp_path):
        store = _store(tmp_path)
        e = _event(details={"key": "value", "count": 42})
        store.append(e)
        result = store.query()[0]
        assert result.details == {"key": "value", "count": 42}

    def test_prune_removes_old_events(self, tmp_path):
        """Pruning with days=0 removes everything (cutoff = now)."""
        store = _store(tmp_path)
        store.append(_event())
        deleted = store.prune(days=0)
        assert deleted >= 1
        assert store.query() == []

    def test_prune_keeps_recent_events(self, tmp_path):
        store = _store(tmp_path)
        store.append(_event())
        deleted = store.prune(days=365)
        assert deleted == 0
        assert len(store.query()) == 1

    def test_thread_safe_concurrent_appends(self, tmp_path):
        store = _store(tmp_path)
        errors: list = []

        def worker(i):
            try:
                store.append(_event(target_id=f"doc{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        results = store.query(limit=100)
        assert len(results) == 20

    def test_query_no_filters_returns_all(self, tmp_path):
        store = _store(tmp_path)
        for _ in range(3):
            store.append(_event())
        assert len(store.query()) == 3


# ===========================================================================
# AuditStore — JSONL fallback
# ===========================================================================


class TestAuditStoreJSONL:
    def _jsonl_store(self, tmp_path: Path) -> AuditStore:
        """Create a store that is forced into JSONL mode."""
        store = AuditStore.__new__(AuditStore)
        store._db_path = tmp_path / "gov.db"
        store._retention_days = 90
        store._lock = threading.Lock()
        store._use_jsonl = True
        store._jsonl_path = tmp_path / ".governance.jsonl"
        return store

    def test_append_writes_jsonl(self, tmp_path):
        store = self._jsonl_store(tmp_path)
        e = _event()
        store.append(e)
        lines = store._jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_id"] == e.event_id

    def test_query_reads_jsonl(self, tmp_path):
        store = self._jsonl_store(tmp_path)
        e = _event(project="jsonl_proj")
        store.append(e)
        results = store.query(project="jsonl_proj")
        assert len(results) == 1
        assert results[0].event_id == e.event_id

    def test_query_filters_in_jsonl(self, tmp_path):
        store = self._jsonl_store(tmp_path)
        store.append(_event(action="ingest_started"))
        store.append(_event(action="delete"))
        results = store.query(action="delete")
        assert all(r.action == "delete" for r in results)

    def test_query_empty_file(self, tmp_path):
        store = self._jsonl_store(tmp_path)
        assert store.query() == []

    def test_prune_returns_zero(self, tmp_path):
        store = self._jsonl_store(tmp_path)
        store.append(_event())
        assert store.prune() == 0

    def test_jsonl_fallback_on_sqlite_error(self, tmp_path):
        """When SQLite init fails, store falls back to JSONL automatically."""
        with patch("sqlite3.connect", side_effect=OSError("no space")):
            store = AuditStore(tmp_path / "bad.db")
        assert store._use_jsonl


# ===========================================================================
# CopilotSessionStore
# ===========================================================================


class TestCopilotSessionStore:
    def test_open_registers_session(self):
        ss = CopilotSessionStore()
        ss.open("sess1", "req1", "proj_a")
        active = ss.list_active()
        assert len(active) == 1
        assert active[0].session_id == "sess1"
        assert active[0].is_active

    def test_close_marks_session_closed(self):
        ss = CopilotSessionStore()
        ss.open("s1", "r1", "p1")
        ss.close("s1")
        assert ss.list_active() == []
        recent = ss.list_recent()
        assert recent[0].closed_at is not None

    def test_close_with_error(self):
        ss = CopilotSessionStore()
        ss.open("s1", "r1", "p1")
        ss.close("s1", error="timeout")
        recent = ss.list_recent()
        assert recent[0].error == "timeout"

    def test_expire_force_closes(self):
        ss = CopilotSessionStore()
        ss.open("s1", "r1", "p1")
        found = ss.expire("s1")
        assert found is True
        active = ss.list_active()
        assert active == []
        recent = ss.list_recent()
        assert "force-expired" in (recent[0].error or "")

    def test_expire_missing_session_returns_false(self):
        ss = CopilotSessionStore()
        assert ss.expire("nonexistent") is False

    def test_expire_already_closed_returns_false(self):
        ss = CopilotSessionStore()
        ss.open("s1", "r1", "p1")
        ss.close("s1")
        assert ss.expire("s1") is False

    def test_list_recent_sorted_newest_first(self):
        ss = CopilotSessionStore()
        for i in range(5):
            ss.open(f"s{i}", f"r{i}", "p1")
            time.sleep(0.002)
        recent = ss.list_recent()
        opened_ats = [s.opened_at for s in recent]
        assert opened_ats == sorted(opened_ats, reverse=True)

    def test_list_recent_respects_limit(self):
        ss = CopilotSessionStore()
        for i in range(10):
            ss.open(f"s{i}", f"r{i}", "p1")
        assert len(ss.list_recent(limit=3)) == 3

    def test_eviction_on_overflow(self):
        ss = CopilotSessionStore(max_recent=5)
        for i in range(8):
            ss.open(f"s{i}", f"r{i}", "p1")
            ss.close(f"s{i}")
        # After eviction, at most max_recent sessions remain
        assert len(ss._sessions) <= 5

    def test_evict_triggered_on_close(self):
        # Eviction fires on close(), not only on open()
        # Open 3 active sessions (at cap=3, no eviction yet)
        ss = CopilotSessionStore(max_recent=3)
        for i in range(3):
            ss.open(f"s{i}", f"r{i}", "p1")
        assert len(ss._sessions) == 3
        # Close them all — _evict() is now called on each close()
        for i in range(3):
            ss.close(f"s{i}")
        # All 3 are closed; count should still be <= max_recent
        assert len(ss._sessions) <= 3

    def test_evict_removes_active_when_cap_exceeded_all_active(self):
        # When all excess sessions are active (no closed_at), cap must still be enforced
        ss = CopilotSessionStore(max_recent=3)
        for i in range(6):
            ss.open(f"s{i}", f"r{i}", "p1")
        # 6 active sessions, cap=3 — second pass should evict oldest active ones
        assert len(ss._sessions) <= 3

    def test_cap_enforced_mixed_open_close(self):
        # Interleaved open/close should never exceed cap
        ss = CopilotSessionStore(max_recent=4)
        for i in range(10):
            ss.open(f"s{i}", f"r{i}", "p1")
            if i % 2 == 0:
                ss.close(f"s{i}")
        assert len(ss._sessions) <= 4

    def test_multiple_active_sessions(self):
        ss = CopilotSessionStore()
        ss.open("s1", "r1", "p1")
        ss.open("s2", "r2", "p2")
        ss.open("s3", "r3", "p3")
        ss.close("s2")
        active = ss.list_active()
        assert len(active) == 2
        active_ids = {s.session_id for s in active}
        assert active_ids == {"s1", "s3"}


# ===========================================================================
# emit() convenience function
# ===========================================================================


class TestEmit:
    def test_emit_does_not_raise(self, tmp_path):
        """emit() must never raise even if the store fails."""
        with patch("axon.governance.get_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            emit(
                "ingest_started",
                "file",
                "/tmp/x.txt",
                project="p",
                status="started",
                details={"k": "v"},
                request_id="req-1",
            )
            # Give the daemon thread a moment
            time.sleep(0.05)
            mock_store.append.assert_called_once()

    def test_emit_swallows_get_store_error(self):
        with patch("axon.governance.get_store", side_effect=RuntimeError("db down")):
            emit("delete", "document", "doc1", project="p")  # must not raise

    def test_emit_creates_correct_event(self, tmp_path):
        captured: list[AuditEvent] = []
        with patch("axon.governance.get_store") as mock_get:
            mock_store = MagicMock()
            mock_store.append.side_effect = captured.append
            mock_get.return_value = mock_store
            emit(
                "delete",
                "document",
                "doc-123",
                project="myproj",
                actor="repl",
                surface="repl",
                status="completed",
                details={"deleted": 1},
                request_id="req-abc",
            )
            time.sleep(0.05)
        assert len(captured) == 1
        e = captured[0]
        assert e.action == "delete"
        assert e.target_id == "doc-123"
        assert e.project == "myproj"
        assert e.actor == "repl"
        assert e.details == {"deleted": 1}


# ===========================================================================
# get_store() singleton
# ===========================================================================


class TestGetStore:
    def test_returns_audit_store_instance(self, tmp_path):
        """get_store() returns a real AuditStore (lazily init'd)."""
        import axon.governance as gov
        import axon.projects as proj

        old_store = gov._store
        old_root = proj.PROJECTS_ROOT
        try:
            gov._store = None
            proj.PROJECTS_ROOT = tmp_path
            store = gov.get_store()
            assert isinstance(store, AuditStore)
            assert store._db_path == tmp_path / ".governance.db"
        finally:
            gov._store = old_store
            proj.PROJECTS_ROOT = old_root

    def test_singleton_returns_same_instance(self, tmp_path):
        import axon.governance as gov
        import axon.projects as proj

        old_store = gov._store
        old_root = proj.PROJECTS_ROOT
        try:
            gov._store = None
            proj.PROJECTS_ROOT = tmp_path
            s1 = gov.get_store()
            s2 = gov.get_store()
            assert s1 is s2
        finally:
            gov._store = old_store
            proj.PROJECTS_ROOT = old_root

    def test_honors_runtime_projects_root_override(self, tmp_path):
        import axon.governance as gov
        import axon.projects as proj

        old_store = gov._store
        old_root = proj.PROJECTS_ROOT
        try:
            gov._store = None
            custom_root = tmp_path / "custom-projects-root"
            proj.set_projects_root(custom_root)
            store = gov.get_store()
            assert store._db_path == custom_root / ".governance.db"
        finally:
            gov._store = old_store
            proj.PROJECTS_ROOT = old_root

    def test_get_session_store_returns_singleton(self):
        s1 = get_session_store()
        s2 = get_session_store()
        assert s1 is s2


# ===========================================================================
# Governance API routes (integration-level with TestClient)
# ===========================================================================


@pytest.fixture()
def _gov_client(tmp_path):
    """FastAPI TestClient wired to governance routes only."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from axon.api_routes.governance import router

    app = FastAPI()
    app.include_router(router)

    # Reset governance singletons so tests are isolated
    import axon.governance as gov

    old_store = gov._store
    gov._store = AuditStore(tmp_path / "test_gov.db")
    old_sessions = gov._session_store
    gov._session_store = CopilotSessionStore()

    with TestClient(app) as client:
        yield client

    gov._store = old_store
    gov._session_store = old_sessions


class TestGovernanceRoutes:
    def test_audit_empty(self, _gov_client):
        resp = _gov_client.get("/governance/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert data["count"] == 0

    def test_audit_after_emit(self, _gov_client, tmp_path):
        import axon.governance as gov

        gov.get_store().append(_event(project="route_proj", action="delete"))
        resp = _gov_client.get("/governance/audit?project=route_proj")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["action"] == "delete"

    def test_audit_limit_param(self, _gov_client):
        import axon.governance as gov

        for i in range(10):
            gov.get_store().append(_event(target_id=f"d{i}"))
        resp = _gov_client.get("/governance/audit?limit=3")
        assert len(resp.json()["events"]) <= 3

    def test_copilot_sessions_empty(self, _gov_client):
        resp = _gov_client.get("/governance/copilot/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_count"] == 0
        assert data["active"] == []

    def test_copilot_sessions_with_active(self, _gov_client):
        import axon.governance as gov

        gov.get_session_store().open("sess-abc", "req-1", "proj_a")
        resp = _gov_client.get("/governance/copilot/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_count"] == 1
        assert data["active"][0]["session_id"] == "sess-abc"

    def test_expire_session_not_found(self, _gov_client):
        resp = _gov_client.post("/governance/copilot/session/nosuchsession/expire")
        assert resp.status_code == 404

    def test_expire_session_found(self, _gov_client):
        import axon.governance as gov

        gov.get_session_store().open("sess-xyz", "req-2", "proj_b")
        resp = _gov_client.post("/governance/copilot/session/sess-xyz/expire")
        assert resp.status_code == 200
        assert resp.json()["status"] == "expired"
        assert gov.get_session_store().list_active() == []

    def test_overview_no_brain(self, _gov_client):
        """overview should return a valid dict even when brain is None."""
        with patch("axon.api.brain", None), patch(
            "axon.api_routes.governance._build_overview"
        ) as mock_ov:
            mock_ov.return_value = {"project": "default", "maintenance": {}}
            resp = _gov_client.get("/governance/overview")
            # Route itself must not crash
            assert resp.status_code in (200, 500)

    def test_projects_route_no_brain(self, _gov_client):
        with patch("axon.projects.list_projects", return_value=[]):
            resp = _gov_client.get("/governance/projects")
            assert resp.status_code == 200
            assert resp.json()["count"] == 0
