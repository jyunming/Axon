"""Graph status, finalize, visualization, and data routes."""


from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from axon.api_routes import _enforce_write_access
from axon.api_routes import enforce_project as _enforce_project
from axon.api_schemas import QueryVisualizeRequest, SearchVisualizeRequest

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
async def finalize_graph(request: Request):
    """Trigger an explicit community rebuild.

    When the active backend has no community step (e.g. ``dynamic_graph``),
    the response carries ``status: "not_applicable"`` and an explanatory
    ``detail`` string instead of silently doing nothing.
    """
    import asyncio

    from axon import api as _api
    from axon import governance as gov

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_write_access(brain, "finalize_graph")
    rid = getattr(request.state, "request_id", "")
    surface = getattr(request.state, "surface", "api")
    project = getattr(brain, "_active_project", "default")
    try:
        backend = getattr(brain, "_graph_backend", None)
        finalize_status = "ok"
        finalize_detail = ""
        backend_id = ""
        if backend is not None and callable(getattr(backend, "finalize", None)):
            result = await asyncio.to_thread(backend.finalize, True)
            finalize_status = getattr(result, "status", "ok")
            finalize_detail = getattr(result, "detail", "")
            backend_id = getattr(result, "backend_id", "")
        else:
            await asyncio.to_thread(brain.finalize_graph, True)
        summary_count = len(getattr(brain, "_community_summaries", {}) or {})
        gov.emit(
            "graph_finalize",
            "graph",
            project,
            project=project,
            details={
                "community_summary_count": summary_count,
                "backend_status": finalize_status,
            },
            surface=surface,
            request_id=rid,
        )
        return {
            "status": finalize_status,
            "community_summary_count": summary_count,
            "backend_id": backend_id,
            "detail": finalize_detail,
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"finalize_graph failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/conflicts", tags=["graph"])
async def graph_conflicts(
    project: str | None = Query(None, description="Expected active project"),
    limit: int = Query(100, ge=1, le=1000, description="Max conflicts to return"),
):
    """Return facts whose status is ``conflicted``.

    Conflicts arise when two exclusive-relation facts (``MARRIED_TO``,
    ``IS_CEO_OF``, …) with the same ``scope_key`` and overlapping ``valid_at``
    are ingested. Backends that don't track conflicts return an empty list
    with ``supported: false``.
    """
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(project, brain)
    backend = getattr(brain, "_graph_backend", None)
    if backend is None:
        return {"backend": "none", "supported": False, "conflicts": []}
    fn = getattr(backend, "list_conflicts", None)
    if not callable(fn):
        return {
            "backend": getattr(backend, "BACKEND_ID", "unknown"),
            "supported": False,
            "conflicts": [],
        }
    try:
        rows = fn(limit=int(limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "backend": getattr(backend, "BACKEND_ID", "unknown"),
        "supported": True,
        "conflicts": rows,
    }


@router.post("/graph/retrieve", tags=["graph"])
async def graph_retrieve(request: Request):
    """Run the active graph backend's ``retrieve`` directly with a
    :class:`RetrievalConfig`. Returns graph contexts only — no LLM call.

    Surfaces ``point_in_time`` historical queries and per-query
    ``federation_weights`` overrides without going through the full
    ``/query`` pipeline. The main ``/query`` endpoint still routes through
    the legacy GraphRAG mixin for backward compatibility.
    """
    import asyncio
    from datetime import datetime as _dt

    from axon import api as _api
    from axon.api_schemas import GraphRetrieveRequest
    from axon.graph_backends.base import RetrievalConfig

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    body = await request.json()
    try:
        payload = GraphRetrieveRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _enforce_project(payload.project, brain)
    backend = getattr(brain, "_graph_backend", None)
    if backend is None or not callable(getattr(backend, "retrieve", None)):
        return {"backend": "none", "contexts": []}
    pit_dt: _dt | None = None
    if payload.point_in_time:
        try:
            pit_dt = _dt.fromisoformat(payload.point_in_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"point_in_time must be ISO-8601: {exc}",
            ) from exc
    cfg = RetrievalConfig(
        top_k=payload.top_k or 10,
        point_in_time=pit_dt,
        federation_weights=payload.federation_weights,
    )
    try:
        contexts = await asyncio.to_thread(backend.retrieve, payload.query, cfg, None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "backend": getattr(backend, "BACKEND_ID", "unknown"),
        "contexts": [_serialise_context(c) for c in contexts],
    }


def _serialise_context(ctx) -> dict:
    """Convert a GraphContext dataclass to a JSON-friendly dict."""
    return {
        "context_id": ctx.context_id,
        "context_type": ctx.context_type,
        "text": ctx.text,
        "score": float(ctx.score),
        "rank": int(ctx.rank),
        "backend_id": ctx.backend_id,
        "source_id": ctx.source_id,
        "source_doc_id": ctx.source_doc_id,
        "source_chunk_id": ctx.source_chunk_id,
        "metadata": ctx.metadata or {},
        "valid_at": ctx.valid_at.isoformat() if ctx.valid_at else None,
        "invalid_at": ctx.invalid_at.isoformat() if ctx.invalid_at else None,
        "evidence_ids": list(ctx.evidence_ids or []),
        "matched_entity_names": list(ctx.matched_entity_names or []),
        "hop_count": int(ctx.hop_count),
    }


@router.post("/query/visualize", response_class=HTMLResponse, tags=["graph"])
async def query_visualize(request: QueryVisualizeRequest):
    """Run a query and return a self-contained HTML page with answer, sources, and highlighted graph.
    Equivalent to opening the VS Code graph panel after a query — but as a standalone
    browser page.  Hit nodes (matched by chunk ID or file path) are highlighted in gold;
    first-degree neighbours in orange.  Click a node to see its description and evidence.
    Unlike GET /graph/visualize, this endpoint:
    - Shows the query and LLM answer in the left panel.
    - Highlights graph nodes that are directly relevant to the search results.
    - Does not claim file navigation (no VS Code editor available outside the extension).
    """
    import asyncio

    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        # Build query overrides — disable RAPTOR/GraphRAG by default to avoid
        # LLM calls during the query phase; caller can re-enable explicitly.
        overrides: dict = {"discuss": True}
        if request.top_k is not None:
            overrides["top_k"] = request.top_k
        if request.raptor is not None:
            overrides["raptor"] = request.raptor
        else:
            overrides["raptor"] = False
        if request.graph_rag is not None:
            overrides["graph_rag"] = request.graph_rag
        else:
            overrides["graph_rag"] = False
        result = await asyncio.to_thread(brain.query, query, overrides=overrides)
        answer: str = result.get("response", "") if isinstance(result, dict) else str(result)
        sources: list = result.get("sources", []) if isinstance(result, dict) else []
        kg = brain.build_graph_payload()
        cg = brain.build_code_graph_payload()
        html = brain.render_query_graph_html(query, answer, sources, kg, cg)
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"query_visualize failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/visualize", response_class=HTMLResponse, tags=["graph"])
async def search_visualize(request: SearchVisualizeRequest):
    """Run a search and return a self-contained HTML page with chunks and highlighted graph.
    Like POST /query/visualize but without LLM generation — shows raw retrieved chunks
    in the left panel instead of an answer.  Useful for inspecting retrieval quality
    without spending LLM tokens.  Hit nodes (matched by chunk ID or file path) are
    highlighted in gold; first-degree neighbours in orange.
    """
    import asyncio

    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    try:
        overrides: dict = {}
        if request.top_k is not None:
            overrides["top_k"] = request.top_k
        if request.threshold is not None:
            overrides["similarity_threshold"] = float(request.threshold)
        results, _diag, _trace = await asyncio.to_thread(
            brain.search_raw,
            query,
            filters=request.filters,
            overrides=overrides or None,
        )
        # Build a plain-text "answer" summarising the top chunks for the left panel.
        if results:
            lines = [f"**{len(results)} chunks retrieved** (no LLM answer — search only)\n"]
            for i, r in enumerate(results[:5]):
                src = (r.get("metadata") or {}).get("source", "") or r.get("source", "")
                score = r.get("score")
                score_str = f"  score {score:.3f}" if isinstance(score, float) else ""
                lines.append(f"**[{i+1}]** `{src}`{score_str}")
            answer = "\n".join(lines)
        else:
            answer = "*No chunks matched the query.*"
        kg = brain.build_graph_payload()
        cg = brain.build_code_graph_payload()
        html = brain.render_query_graph_html(query, answer, results, kg, cg)
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"search_visualize failed: {e}")
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


@router.get("/graph/backend/status", tags=["graph"])
async def graph_backend_status():
    """Return the active graph backend's status dict."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    backend = getattr(brain, "_graph_backend", None)
    if backend is None:
        return {"backend": "none", "ready": False}
    try:
        status = backend.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return status


@router.get("/graph/data", tags=["graph"])
async def graph_data(project: str | None = Query(None, description="Expected active project")):
    """Return the entity/relation knowledge-graph payload for VS Code webview consumption."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(project, brain)
    backend = getattr(brain, "_graph_backend", None)
    if backend is not None and callable(getattr(backend, "graph_data", None)):
        try:
            payload = backend.graph_data()
        except Exception:
            return {"nodes": [], "links": []}
        if hasattr(payload, "to_dict"):
            result = payload.to_dict()
            return result if isinstance(result, dict) else {"nodes": [], "links": []}
        if isinstance(payload, dict):
            return payload
        return {"nodes": [], "links": []}
    # Fall back to the legacy build_graph_payload() on AxonBrain
    if callable(getattr(brain, "build_graph_payload", None)):
        payload = brain.build_graph_payload()
        if isinstance(payload, dict):
            return payload
    return {"nodes": [], "links": []}


@router.get("/code-graph/data", tags=["graph"])
async def code_graph_data(project: str | None = Query(None, description="Expected active project")):
    """Return the code structure graph as JSON for VS Code webview."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(project, brain)
    return brain.build_code_graph_payload()
