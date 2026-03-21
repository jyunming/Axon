"""
tests/test_governance_routes.py — HTTP-layer tests for api_routes/governance.py

Covers:
- GET  /governance/overview        (with and without brain, subsystem errors)
- GET  /governance/audit           (filters, edge cases)
- GET  /governance/copilot/sessions
- GET  /governance/projects        (active project graph state, maint fallback)
- POST /governance/graph/rebuild   (ok, brain missing, finalize error, permission error)
- POST /governance/project/maintenance (ok, invalid name, ValueError, 500)
- POST /governance/copilot/session/{id}/expire (ok, not found)
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain(active_project="default"):
    brain = MagicMock()
    brain._active_project = active_project
    brain._entity_graph = {"ent1": {}, "ent2": {}}
    brain._relation_graph = {"r1": {}}
    brain._community_summaries = {"c1": "summary"}
    brain.get_stale_docs.return_value = [{"id": "d1"}]
    brain.finalize_graph.return_value = None
    return brain


def _make_session(session_id="s1", closed=False):
    from axon.governance import CopilotSession

    return CopilotSession(
        session_id=session_id,
        request_id="r1",
        project="proj",
        opened_at=datetime.now(timezone.utc).isoformat(),
        closed_at=datetime.now(timezone.utc).isoformat() if closed else None,
    )


@pytest.fixture(autouse=True)
def reset_state():
    orig_brain = api_module.brain
    api_module.brain = None
    api_module._jobs.clear()
    yield
    api_module.brain = orig_brain
    api_module._jobs.clear()


# ---------------------------------------------------------------------------
# Patch aliases (functions imported lazily inside route bodies)
# ---------------------------------------------------------------------------
_REGISTRY = "axon.runtime.get_registry"
_MAINT_STATUS = "axon.maintenance.get_maintenance_status"
_LIST_PROJ = "axon.projects.list_projects"
_APPLY_MAINT = "axon.maintenance.apply_maintenance_state"
_GOV_STORE = "axon.governance.get_store"
_GOV_SS = "axon.governance.get_session_store"
_GOV_EMIT = "axon.governance.emit"


# ---------------------------------------------------------------------------
# GET /governance/overview
# ---------------------------------------------------------------------------


class TestGovernanceOverview:
    def _base_patches(self, leases=None, maint=None, projects=None, sessions=None):
        """Context managers for the common 'happy path' subsystems."""
        leases = leases or []
        maint = maint or {"maintenance_state": "active", "active_leases": 0}
        projects = projects if projects is not None else []
        sessions = sessions or []
        return (
            patch(_REGISTRY),
            patch(_MAINT_STATUS, return_value=maint),
            patch(_LIST_PROJ, return_value=projects),
            patch(_GOV_SS),
        )

    def test_overview_no_brain_returns_200(self):
        with patch(_REGISTRY) as mr, patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ), patch(_LIST_PROJ, return_value=[]), patch(_GOV_SS) as mss:
            mr.return_value.snapshot_all.return_value = []
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.status_code == 200
        data = r.json()
        assert "project" in data
        assert "graph" in data
        assert data["active_ingest_jobs"] == 0

    def test_overview_with_brain(self):
        brain = _make_brain()
        api_module.brain = brain
        with patch(_REGISTRY) as mr, patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ), patch(_LIST_PROJ, return_value=[{"name": "default"}]), patch(_GOV_SS) as mss:
            mr.return_value.snapshot_all.return_value = []
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.status_code == 200
        data = r.json()
        assert data["graph"]["entity_count"] == 2
        assert data["graph"]["relation_count"] == 1
        assert data["graph"]["community_count"] == 1
        assert data["stale_doc_count"] == 1
        assert data["project_count"] == 1

    def test_overview_registry_raises(self):
        with patch(_REGISTRY, side_effect=RuntimeError("no registry")), patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ), patch(_LIST_PROJ, return_value=[]), patch(_GOV_SS) as mss:
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.status_code == 200
        assert r.json()["active_leases"] == []

    def test_overview_maintenance_raises(self):
        with patch(_REGISTRY) as mr, patch(_MAINT_STATUS, side_effect=Exception("fail")), patch(
            _LIST_PROJ, return_value=[]
        ), patch(_GOV_SS) as mss:
            mr.return_value.snapshot_all.return_value = []
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.status_code == 200
        assert r.json()["maintenance"]["maintenance_state"] == "unknown"

    def test_overview_list_projects_raises(self):
        with patch(_REGISTRY) as mr, patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ), patch(_LIST_PROJ, side_effect=Exception("fail")), patch(_GOV_SS) as mss:
            mr.return_value.snapshot_all.return_value = []
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.status_code == 200
        assert r.json()["project_count"] == 0

    def test_overview_stale_docs_raises(self):
        brain = _make_brain()
        brain.get_stale_docs.side_effect = Exception("stale fail")
        api_module.brain = brain
        with patch(_REGISTRY) as mr, patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ), patch(_LIST_PROJ, return_value=[]), patch(_GOV_SS) as mss:
            mr.return_value.snapshot_all.return_value = []
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.status_code == 200
        assert r.json()["stale_doc_count"] == 0

    def test_overview_active_jobs_counted(self):
        api_module._jobs["j1"] = {"status": "processing"}
        api_module._jobs["j2"] = {"status": "done"}
        with patch(_REGISTRY) as mr, patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ), patch(_LIST_PROJ, return_value=[]), patch(_GOV_SS) as mss:
            mr.return_value.snapshot_all.return_value = []
            mss.return_value.list_active.return_value = []
            r = client.get("/governance/overview")
        assert r.json()["active_ingest_jobs"] == 1


# ---------------------------------------------------------------------------
# GET /governance/audit
# ---------------------------------------------------------------------------


class TestGovernanceAudit:
    def _mock_event(self):
        from axon.governance import AuditEvent

        return AuditEvent(
            action="ingest_started",
            target_type="file",
            target_id="/tmp/doc.txt",
            project="proj",
        )

    def test_audit_returns_events(self):
        ev = self._mock_event()
        with patch(_GOV_STORE) as ms:
            ms.return_value.query.return_value = [ev]
            r = client.get("/governance/audit")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["events"][0]["action"] == "ingest_started"

    def test_audit_filter_params_passed(self):
        with patch(_GOV_STORE) as ms:
            ms.return_value.query.return_value = []
            r = client.get(
                "/governance/audit"
                "?project=proj&action=delete&surface=api"
                "&status=completed&since=2026-01-01T00:00:00Z&limit=10"
            )
        assert r.status_code == 200
        ms.return_value.query.assert_called_once_with(
            project="proj",
            action="delete",
            surface="api",
            status="completed",
            since="2026-01-01T00:00:00Z",
            limit=10,
        )

    def test_audit_empty_result(self):
        with patch(_GOV_STORE) as ms:
            ms.return_value.query.return_value = []
            r = client.get("/governance/audit")
        assert r.json()["count"] == 0
        assert r.json()["events"] == []

    def test_audit_limit_over_max_rejected(self):
        with patch(_GOV_STORE) as ms:
            ms.return_value.query.return_value = []
            r = client.get("/governance/audit?limit=9999")
        assert r.status_code == 422

    def test_audit_limit_zero_rejected(self):
        with patch(_GOV_STORE) as ms:
            ms.return_value.query.return_value = []
            r = client.get("/governance/audit?limit=0")
        assert r.status_code == 422

    def test_audit_event_fields_serialized(self):
        ev = self._mock_event()
        with patch(_GOV_STORE) as ms:
            ms.return_value.query.return_value = [ev]
            r = client.get("/governance/audit")
        e = r.json()["events"][0]
        assert "event_id" in e
        assert "timestamp" in e
        assert "actor" in e
        assert "details" in e


# ---------------------------------------------------------------------------
# GET /governance/copilot/sessions
# ---------------------------------------------------------------------------


class TestGovernanceCopilotSessions:
    def test_returns_active_and_recent(self):
        active = _make_session("s1", closed=False)
        recent = _make_session("s2", closed=True)
        with patch(_GOV_SS) as mss:
            mss.return_value.list_active.return_value = [active]
            mss.return_value.list_recent.return_value = [active, recent]
            r = client.get("/governance/copilot/sessions")
        assert r.status_code == 200
        data = r.json()
        assert data["active_count"] == 1
        assert len(data["active"]) == 1
        assert len(data["recent"]) == 2

    def test_empty_sessions(self):
        with patch(_GOV_SS) as mss:
            mss.return_value.list_active.return_value = []
            mss.return_value.list_recent.return_value = []
            r = client.get("/governance/copilot/sessions")
        assert r.status_code == 200
        assert r.json()["active_count"] == 0

    def test_session_fields_serialized(self):
        s = _make_session("sid-xyz")
        with patch(_GOV_SS) as mss:
            mss.return_value.list_active.return_value = [s]
            mss.return_value.list_recent.return_value = [s]
            r = client.get("/governance/copilot/sessions")
        data = r.json()["active"][0]
        assert data["session_id"] == "sid-xyz"
        assert "opened_at" in data
        assert "is_active" in data

    def test_limit_param_forwarded(self):
        with patch(_GOV_SS) as mss:
            mss.return_value.list_active.return_value = []
            mss.return_value.list_recent.return_value = []
            client.get("/governance/copilot/sessions?limit=5")
        mss.return_value.list_recent.assert_called_with(limit=5)


# ---------------------------------------------------------------------------
# GET /governance/projects
# ---------------------------------------------------------------------------


class TestGovernanceProjects:
    def test_returns_projects(self):
        with patch(_LIST_PROJ, return_value=[{"name": "p1"}, {"name": "p2"}]), patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ):
            r = client.get("/governance/projects")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert data["projects"][0]["name"] == "p1"

    def test_projects_includes_maintenance(self):
        maint = {"maintenance_state": "drain", "active_leases": 3}
        with patch(_LIST_PROJ, return_value=[{"name": "myproj"}]), patch(
            _MAINT_STATUS, return_value=maint
        ):
            r = client.get("/governance/projects")
        assert r.json()["projects"][0]["maintenance"]["maintenance_state"] == "drain"

    def test_active_project_gets_graph_state(self):
        brain = _make_brain(active_project="myproj")
        api_module.brain = brain
        with patch(_LIST_PROJ, return_value=[{"name": "myproj"}]), patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ):
            r = client.get("/governance/projects")
        proj = r.json()["projects"][0]
        assert proj["graph"]["entity_count"] == 2
        assert proj["graph"]["community_count"] == 1

    def test_inactive_project_no_graph_state(self):
        brain = _make_brain(active_project="other")
        api_module.brain = brain
        with patch(_LIST_PROJ, return_value=[{"name": "myproj"}]), patch(
            _MAINT_STATUS, return_value={"maintenance_state": "active", "active_leases": 0}
        ):
            r = client.get("/governance/projects")
        assert r.json()["projects"][0]["graph"] == {}

    def test_list_projects_raises_500(self):
        with patch(_LIST_PROJ, side_effect=Exception("db error")):
            r = client.get("/governance/projects")
        assert r.status_code == 500

    def test_maintenance_raises_falls_back(self):
        with patch(_LIST_PROJ, return_value=[{"name": "p1"}]), patch(
            _MAINT_STATUS, side_effect=Exception("maint fail")
        ):
            r = client.get("/governance/projects")
        assert r.status_code == 200
        assert r.json()["projects"][0]["maintenance"]["maintenance_state"] == "unknown"


# ---------------------------------------------------------------------------
# POST /governance/graph/rebuild
# ---------------------------------------------------------------------------


class TestGovernanceGraphRebuild:
    def test_rebuild_ok(self):
        brain = _make_brain()
        brain._community_summaries = {"c1": "s", "c2": "s"}
        api_module.brain = brain
        with patch(_GOV_EMIT):
            r = client.post("/governance/graph/rebuild")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["community_summary_count"] == 2

    def test_rebuild_no_brain_503(self):
        r = client.post("/governance/graph/rebuild")
        assert r.status_code == 503

    def test_rebuild_permission_error_403(self):
        brain = _make_brain()
        brain.finalize_graph.side_effect = PermissionError("read-only")
        api_module.brain = brain
        with patch(_GOV_EMIT):
            r = client.post("/governance/graph/rebuild")
        assert r.status_code == 403

    def test_rebuild_generic_error_500(self):
        brain = _make_brain()
        brain.finalize_graph.side_effect = RuntimeError("internal error")
        api_module.brain = brain
        with patch(_GOV_EMIT):
            r = client.post("/governance/graph/rebuild")
        assert r.status_code == 500

    def test_rebuild_emits_started_and_completed(self):
        brain = _make_brain()
        api_module.brain = brain
        emitted_statuses = []
        with patch(
            _GOV_EMIT,
            side_effect=lambda *a, **kw: emitted_statuses.append(kw.get("status", "completed")),
        ):
            client.post("/governance/graph/rebuild")
        assert "started" in emitted_statuses
        assert "completed" in emitted_statuses

    def test_rebuild_emits_failed_on_error(self):
        brain = _make_brain()
        brain.finalize_graph.side_effect = RuntimeError("crash")
        api_module.brain = brain
        emitted_statuses = []
        with patch(
            _GOV_EMIT,
            side_effect=lambda *a, **kw: emitted_statuses.append(kw.get("status", "completed")),
        ):
            client.post("/governance/graph/rebuild")
        assert "failed" in emitted_statuses


# ---------------------------------------------------------------------------
# POST /governance/project/maintenance
# ---------------------------------------------------------------------------


class TestGovernanceSetMaintenance:
    def test_set_maintenance_ok(self):
        with patch(_APPLY_MAINT, return_value={"status": "ok", "state": "drain"}), patch(_GOV_EMIT):
            r = client.post("/governance/project/maintenance?name=myproj&state=drain")
        assert r.status_code == 200
        assert r.json()["state"] == "drain"

    def test_invalid_project_name_422(self):
        r = client.post("/governance/project/maintenance?name=bad/name/too/deep/x/y/z&state=drain")
        assert r.status_code == 422

    def test_value_error_project_missing_404(self):
        with patch(_APPLY_MAINT, side_effect=ValueError("project does not exist")), patch(
            _GOV_EMIT
        ):
            r = client.post("/governance/project/maintenance?name=myproj&state=drain")
        assert r.status_code == 404

    def test_value_error_other_422(self):
        with patch(_APPLY_MAINT, side_effect=ValueError("invalid state value")), patch(_GOV_EMIT):
            r = client.post("/governance/project/maintenance?name=myproj&state=bad")
        assert r.status_code == 422

    def test_generic_error_500(self):
        with patch(_APPLY_MAINT, side_effect=RuntimeError("db crash")), patch(_GOV_EMIT):
            r = client.post("/governance/project/maintenance?name=myproj&state=drain")
        assert r.status_code == 500

    def test_emits_maintenance_changed(self):
        emitted = []
        with patch(_APPLY_MAINT, return_value={"status": "ok"}), patch(
            _GOV_EMIT, side_effect=lambda *a, **kw: emitted.append(a[0])
        ):
            client.post("/governance/project/maintenance?name=myproj&state=drain")
        assert emitted.count("maintenance_changed") >= 1


# ---------------------------------------------------------------------------
# POST /governance/copilot/session/{session_id}/expire
# ---------------------------------------------------------------------------


class TestGovernanceExpireSession:
    def test_expire_ok(self):
        with patch(_GOV_SS) as mss, patch(_GOV_EMIT):
            mss.return_value.expire.return_value = True
            r = client.post("/governance/copilot/session/sess-123/expire")
        assert r.status_code == 200
        assert r.json()["status"] == "expired"
        assert r.json()["session_id"] == "sess-123"

    def test_expire_not_found_404(self):
        with patch(_GOV_SS) as mss:
            mss.return_value.expire.return_value = False
            r = client.post("/governance/copilot/session/unknown/expire")
        assert r.status_code == 404

    def test_expire_emits_audit(self):
        emitted = []
        with patch(_GOV_SS) as mss, patch(
            _GOV_EMIT, side_effect=lambda *a, **kw: emitted.append(a[0])
        ):
            mss.return_value.expire.return_value = True
            client.post("/governance/copilot/session/sess-abc/expire")
        assert "copilot_session_closed" in emitted
