"""Graph status, finalize, visualization, and data routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger("AxonAPI")
router = APIRouter()


@router.get("/graph/status")
async def get_graph_status():
    """Return current GraphRAG community build status."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    in_progress = getattr(brain, "_community_build_in_progress", False)
    summary_count = len(getattr(brain, "_community_summaries", {}) or {})
    return {
        "community_build_in_progress": in_progress,
        "community_summary_count": summary_count,
    }


@router.post("/graph/finalize")
async def finalize_graph(request: Request):
    """Trigger an explicit community rebuild."""
    import asyncio

    from axon import api as _api
    from axon import governance as gov

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    rid = getattr(request.state, "request_id", "")
    surface = getattr(request.state, "surface", "api")
    project = getattr(brain, "_active_project", "default")
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
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"finalize_graph failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/visualize", tags=["graph"])
async def get_graph_visualization():
    """Return the entity–relation graph as an interactive HTML page."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        html = brain.export_graph_html(open_browser=False)
        return HTMLResponse(content=html)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/data", tags=["graph"])
async def graph_data():
    """Return the entity/relation knowledge-graph payload for VS Code webview consumption."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    payload = brain.build_graph_payload()
    if isinstance(payload, dict):
        return payload
    return {"nodes": [], "links": []}


@router.get("/code-graph/data", tags=["graph"])
async def code_graph_data():
    """Return the code structure graph as JSON for VS Code webview."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain.build_code_graph_payload()
