"""Governance read + control routes for the Operator Console."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger("AxonAPI")
router = APIRouter(prefix="/governance", tags=["governance"])


# ---------------------------------------------------------------------------
# Helper: safe overview aggregation
# ---------------------------------------------------------------------------


def _build_overview(brain, jobs: dict) -> dict:
    """Aggregate operator status from all live subsystems."""
    from axon import governance as gov
    from axon.maintenance import get_maintenance_status
    from axon.projects import list_projects
    from axon.runtime import get_registry

    project = getattr(brain, "_active_project", "default") if brain else "default"

    # Lease / drain state
    try:
        registry = get_registry()
        leases = registry.snapshot_all()
    except Exception:
        leases = {}

    # Maintenance state
    try:
        maint = get_maintenance_status(project)
    except Exception:
        maint = {"maintenance_state": "unknown", "active_leases": 0}

    # Project list
    try:
        projects = list_projects()
    except Exception:
        projects = []

    # Graph state
    entity_count = 0
    relation_count = 0
    community_count = 0
    if brain:
        entity_count = len(getattr(brain, "_entity_graph", {}) or {})
        relation_count = len(getattr(brain, "_relation_graph", {}) or {})
        community_count = len(getattr(brain, "_community_summaries", {}) or {})

    # Stale docs
    stale_count = 0
    if brain:
        try:
            stale = brain.get_stale_docs(days=30)
            stale_count = len(stale) if stale else 0
        except Exception:
            pass

    # Active jobs
    active_jobs = sum(1 for j in jobs.values() if j.get("status") == "processing")

    # Copilot sessions
    session_store = gov.get_session_store()
    active_sessions = len(session_store.list_active())

    return {
        "project": project,
        "maintenance": maint,
        "graph": {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "community_count": community_count,
        },
        "stale_doc_count": stale_count,
        "active_ingest_jobs": active_jobs,
        "active_leases": leases,
        "copilot_sessions_active": active_sessions,
        "project_count": len(projects),
    }


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------


@router.get("/overview")
async def governance_overview():
    """Return aggregated operator status: project, graph, leases, shares, Copilot."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        return _build_overview(None, _api._jobs)
    return _build_overview(brain, _api._jobs)


@router.get("/audit")
async def governance_audit(
    project: str | None = Query(None, description="Filter by project name"),
    action: str | None = Query(None, description="Filter by action"),
    surface: str | None = Query(None, description="Filter by surface"),
    status: str | None = Query(None, description="Filter by status"),
    since: str | None = Query(None, description="ISO-8601 lower bound for timestamp"),
    limit: int = Query(50, ge=1, le=1000, description="Max events to return"),
):
    """Return audit log entries matching the given filters, newest first."""
    from axon import governance as gov

    events = gov.get_store().query(
        project=project,
        action=action,
        surface=surface,
        status=status,
        since=since,
        limit=limit,
    )
    return {
        "events": [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "actor": e.actor,
                "surface": e.surface,
                "project": e.project,
                "action": e.action,
                "target_type": e.target_type,
                "target_id": e.target_id,
                "status": e.status,
                "details": e.details,
                "request_id": e.request_id,
            }
            for e in events
        ],
        "count": len(events),
    }


@router.get("/copilot/sessions")
async def governance_copilot_sessions(
    limit: int = Query(20, ge=1, le=100),
):
    """Return active and recent Copilot bridge sessions."""
    from axon import governance as gov

    store = gov.get_session_store()
    active = store.list_active()
    recent = store.list_recent(limit=limit)

    def _ser(s):
        return {
            "session_id": s.session_id,
            "request_id": s.request_id,
            "project": s.project,
            "opened_at": s.opened_at,
            "closed_at": s.closed_at,
            "is_active": s.is_active,
            "error": s.error,
        }

    return {
        "active": [_ser(s) for s in active],
        "recent": [_ser(s) for s in recent],
        "active_count": len(active),
    }


@router.get("/projects")
async def governance_projects():
    """Return all projects with their maintenance and graph state."""
    from axon import api as _api
    from axon.maintenance import get_maintenance_status
    from axon.projects import list_projects

    brain = _api.brain
    try:
        projects = list_projects()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result = []
    for p in projects:
        name = p.get("name", "")
        try:
            maint = get_maintenance_status(name)
        except Exception:
            maint = {"maintenance_state": "unknown", "active_leases": 0}
        graph_state = {}
        if brain and getattr(brain, "_active_project", None) == name:
            graph_state = {
                "entity_count": len(getattr(brain, "_entity_graph", {}) or {}),
                "community_count": len(getattr(brain, "_community_summaries", {}) or {}),
            }
        result.append({**p, "maintenance": maint, "graph": graph_state})

    return {"projects": result, "count": len(result)}


# ---------------------------------------------------------------------------
# Control APIs
# ---------------------------------------------------------------------------


@router.post("/graph/rebuild")
async def governance_graph_rebuild(req: Request):
    """Audited operator wrapper: trigger graph community rebuild."""
    from axon import api as _api
    from axon import governance as gov

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    project = getattr(brain, "_active_project", "default")
    gov.emit(
        "graph_finalize",
        "graph",
        project,
        project=project,
        status="started",
        surface=surface,
        request_id=rid,
    )
    try:
        await asyncio.to_thread(brain.finalize_graph, True)
        summary_count = len(brain._community_summaries)
        gov.emit(
            "graph_finalize",
            "graph",
            project,
            project=project,
            details={"community_summary_count": summary_count},
            surface=surface,
            request_id=rid,
        )
        return {"status": "ok", "community_summary_count": summary_count}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        logger.error("governance graph rebuild failed: %s", exc)
        gov.emit(
            "graph_finalize",
            "graph",
            project,
            project=project,
            status="failed",
            details={"error": "rebuild error"},
            surface=surface,
            request_id=rid,
        )
        raise HTTPException(status_code=500, detail="Graph rebuild failed")


@router.post("/project/maintenance")
async def governance_set_maintenance(
    name: str,
    state: str,
    req: Request,
):
    """Audited operator wrapper: set project maintenance state."""
    from axon import governance as gov
    from axon.api_schemas import _VALID_PROJECT_NAME_RE
    from axon.maintenance import apply_maintenance_state

    if not _VALID_PROJECT_NAME_RE.match(name):
        raise HTTPException(status_code=422, detail=f"Invalid project name: '{name}'")
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    gov.emit(
        "maintenance_changed",
        "project",
        name,
        project=name,
        status="started",
        details={"state": state},
        surface=surface,
        request_id=rid,
    )
    try:
        result = apply_maintenance_state(name, state)
        gov.emit(
            "maintenance_changed",
            "project",
            name,
            project=name,
            details={"state": state, "result": result},
            surface=surface,
            request_id=rid,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=404 if "does not exist" in str(exc) else 422, detail=str(exc)
        )
    except Exception as exc:
        logger.error("governance set_maintenance failed: %s", exc)
        raise HTTPException(status_code=500, detail="Maintenance state update failed")


@router.post("/copilot/session/{session_id}/expire")
async def governance_expire_session(session_id: str, req: Request):
    """Force-close a stuck Copilot bridge session."""
    from axon import governance as gov

    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    store = gov.get_session_store()
    found = store.expire(session_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already closed.",
        )
    gov.emit(
        "copilot_session_closed",
        "copilot_session",
        session_id,
        project="",
        details={"reason": "operator-expired"},
        surface=surface,
        request_id=rid,
    )
    return {"status": "expired", "session_id": session_id}
