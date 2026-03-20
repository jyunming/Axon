"""Query, search, and clear routes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from axon.api_schemas import QueryRequest, SearchRequest, SearchResult

logger = logging.getLogger("AxonAPI")
router = APIRouter()


@router.post("/query")
async def query_brain(request: QueryRequest):
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        overrides = {
            "top_k": request.top_k,
            "similarity_threshold": request.threshold,
            "hybrid_search": request.hybrid,
            "rerank": request.rerank,
            "hyde": request.hyde,
            "multi_query": request.multi_query,
            "step_back": request.step_back,
            "query_decompose": request.decompose,
            "compress_context": request.compress,
            "discussion_fallback": request.discuss,
            "llm_temperature": request.temperature,
        }

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        timeout = request.timeout or float(os.getenv("AXON_QUERY_TIMEOUT", "120"))

        if request.dry_run:
            try:
                results, diag, _trace = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: brain.search_raw(
                            request.query, filters=request.filters, overrides=overrides
                        ),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Dry-run retrieval timed out")
            return {
                "query": request.query,
                "dry_run": True,
                "results": results,
                "diagnostics": diag.to_dict(),
            }

        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: brain.query(
                        request.query, filters=request.filters, overrides=overrides
                    ),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Query timed out after {timeout}s. Try disabling HyDE/Rerank or simplifying your query.",
            )

        cfg = brain._apply_overrides(overrides)
        settings = {
            "top_k": cfg.top_k,
            "hybrid": cfg.hybrid_search,
            "rerank": cfg.rerank,
            "hyde": cfg.hyde,
            "multi_query": cfg.multi_query,
            "step_back": cfg.step_back,
            "decompose": cfg.query_decompose,
            "compress": cfg.compress_context,
            "discuss": cfg.discussion_fallback,
        }
        out: dict = {"query": request.query, "response": response, "settings": settings}
        if request.include_diagnostics:
            out["diagnostics"] = brain._last_diagnostics.to_dict()
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_brain_stream(request: QueryRequest):
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    overrides = {
        "top_k": request.top_k,
        "similarity_threshold": request.threshold,
        "hybrid_search": request.hybrid,
        "rerank": request.rerank,
        "hyde": request.hyde,
        "multi_query": request.multi_query,
        "step_back": request.step_back,
        "query_decompose": request.decompose,
        "compress_context": request.compress,
        "discussion_fallback": request.discuss,
        "llm_temperature": request.temperature,
    }

    def generate():
        try:
            for chunk in brain.query_stream(
                request.query, filters=request.filters, overrides=overrides
            ):
                if isinstance(chunk, dict):
                    yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/search", response_model=list[SearchResult])
async def search_brain(request: SearchRequest):
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        loop = asyncio.get_running_loop()
        overrides: dict[str, Any] = {}
        if request.top_k is not None:
            overrides["top_k"] = int(request.top_k)
        if request.threshold is not None:
            overrides["similarity_threshold"] = float(request.threshold)
        cfg = brain._apply_overrides(overrides) if overrides else None
        retrieval_data = await loop.run_in_executor(
            None,
            lambda: brain._execute_retrieval(request.query, filters=request.filters, cfg=cfg),
        )
        results = retrieval_data["results"]
        top_k = request.top_k or brain.config.top_k
        return results[:top_k]
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/raw")
async def search_raw_endpoint(request: SearchRequest, include_trace: bool = False):
    """Run retrieval without calling the LLM."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        loop = asyncio.get_running_loop()
        overrides: dict = {}
        if request.top_k is not None:
            overrides["top_k"] = int(request.top_k)
        if request.threshold is not None:
            overrides["similarity_threshold"] = float(request.threshold)
        results, diag, trace = await loop.run_in_executor(
            None,
            lambda: brain.search_raw(
                request.query,
                filters=request.filters,
                overrides=overrides or None,
            ),
        )
        out: dict = {
            "query": request.query,
            "results": results,
            "diagnostics": diag.to_dict(),
        }
        if include_trace:
            out["trace"] = trace.to_dict()
        return out
    except Exception as e:
        logger.error(f"Error during /search/raw: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_brain():
    """Clear the active project's vector store, BM25 index, hash store, and entity graph."""
    import pathlib

    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        vs = brain.vector_store
        provider = vs.provider
        if provider == "chroma" and vs.client is not None:
            vs.client.delete_collection("axon")
            vs.collection = vs.client.create_collection(
                name="axon", metadata={"hnsw:space": "cosine"}
            )
        elif provider == "qdrant" and vs.client is not None:
            try:
                vs.client.delete_collection("axon")
            except Exception:
                pass
            vs._init_store()
        elif provider == "lancedb" and vs.client is not None:
            try:
                vs.client.drop_table("axon")
            except Exception:
                pass
            vs.collection = None

        if brain.bm25 is not None:
            brain.bm25.corpus.clear()
            brain.bm25.bm25 = None
            brain.bm25.save()

        brain._ingested_hashes = set()
        brain._save_hash_store()
        brain._entity_graph = {}
        brain._save_entity_graph()

        _meta_path = pathlib.Path(brain._embedding_meta_path)
        if _meta_path.exists():
            try:
                _meta_path.unlink()
            except OSError:
                pass

        return {"status": "success", "message": "Collection cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
