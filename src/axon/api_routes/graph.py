"""Graph status, finalize, visualization, and data routes."""


from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from axon.api_routes.ingest import _enforce_write_access
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
    """Trigger an explicit community rebuild."""

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
