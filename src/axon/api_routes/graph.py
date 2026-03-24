"""Graph status, finalize, visualization, and data routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from axon.api import get_brain
from axon.api_routes.ingest import _enforce_write_access
from axon.main import AxonBrain

logger = logging.getLogger("AxonAPI")
router = APIRouter()


@router.get("/graph/status")
async def get_graph_status(brain: AxonBrain = Depends(get_brain)):
    """Return current GraphRAG community build status."""
    in_progress = getattr(brain, "_community_build_in_progress", False)
    summary_count = len(getattr(brain, "_community_summaries", {}) or {})
    entity_count = len(getattr(brain, "_entity_graph", {}) or {})
    code_node_count = len((getattr(brain, "_code_graph", {}) or {}).get("nodes", {}))
    graph_ready = entity_count > 0 or code_node_count > 0
    return {
        "community_build_in_progress": in_progress,
        "community_summary_count": summary_count,
        "entity_count": entity_count,
        "code_node_count": code_node_count,
        "graph_ready": graph_ready,
    }


@router.post("/graph/finalize")
async def finalize_graph(request: Request, brain: AxonBrain = Depends(get_brain)):
    """Trigger an explicit community rebuild."""
    import asyncio

    from axon import governance as gov

    _enforce_write_access(brain, "finalize_graph")
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
async def get_graph_visualization(brain: AxonBrain = Depends(get_brain)):
    """Return the entity–relation graph as an interactive HTML page."""
    try:
        html = brain.export_graph_html(open_browser=False)
        return HTMLResponse(content=html)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/data", tags=["graph"])
async def graph_data(brain: AxonBrain = Depends(get_brain)):
    """Return the entity/relation knowledge-graph payload for VS Code webview consumption."""
    payload = brain.build_graph_payload()
    if isinstance(payload, dict):
        return payload
    return {"nodes": [], "links": []}


@router.get("/code-graph/data", tags=["graph"])
async def code_graph_data(brain: AxonBrain = Depends(get_brain)):
    """Return the code structure graph as JSON for VS Code webview."""
    return brain.build_code_graph_payload()
