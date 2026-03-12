import hashlib
import logging
import os
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from axon.main import AxonBrain, AxonConfig
from axon.projects import list_projects as _list_projects

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AxonAPI")


def _validate_ingest_path(path: str) -> str:
    """Validate that path is within the allowed base directory."""
    allowed_base = pathlib.Path(os.getenv("RAG_INGEST_BASE", ".")).resolve()
    abs_path = pathlib.Path(path).resolve()
    try:
        abs_path.relative_to(allowed_base)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Path '{path}' is outside the allowed ingest directory. Set RAG_INGEST_BASE to permit additional paths.",
        )
    return str(abs_path)


app = FastAPI(
    title="Axon API",
    description="REST API for agent orchestration and document retrieval",
    version="2.0.0",
)

# Optional API key authentication — enabled when RAG_API_KEY env var is set
_RAG_API_KEY: str | None = os.getenv("RAG_API_KEY")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce X-API-Key header when RAG_API_KEY is configured."""
    if _RAG_API_KEY:
        # Allow health-check without auth so load-balancers can probe freely
        if request.url.path != "/health":
            provided = request.headers.get("X-API-Key")
            if provided != _RAG_API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-API-Key header."},
                )
    return await call_next(request)


# Global Brain Instance
brain: AxonBrain | None = None

# ---------------------------------------------------------------------------
# Source-level dedup store
# Keyed by project name → content_hash → {doc_id, last_ingested_at}.
# This dict is in-memory only; it is lost on server restart.
# ---------------------------------------------------------------------------
_source_hashes: dict[str, dict[str, dict]] = {}

# ---------------------------------------------------------------------------
# Async ingest job status store  (P1-C)
# Keyed by job_id.  In-memory only; single-worker deployments only.
# Max 1000 jobs retained; entries older than 60 min are evicted on each write.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_MAX_JOBS = 1000
_JOB_TTL_SECONDS = 3600  # 60 minutes


def _evict_old_jobs() -> None:
    """Remove completed jobs older than _JOB_TTL_SECONDS and cap at _MAX_JOBS."""
    cutoff = datetime.now(timezone.utc).timestamp() - _JOB_TTL_SECONDS
    expired = [jid for jid, job in list(_jobs.items()) if job.get("started_at_ts", 0) < cutoff]
    for jid in expired:
        _jobs.pop(jid, None)
    # Hard cap: keep only the most recent _MAX_JOBS by start time
    if len(_jobs) > _MAX_JOBS:
        oldest = sorted(_jobs, key=lambda j: _jobs[j].get("started_at_ts", 0))
        for jid in oldest[: len(_jobs) - _MAX_JOBS]:
            _jobs.pop(jid, None)


def _compute_content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the normalised text content."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _check_dedup(text: str, project: str = "_global") -> dict | None:
    """Check whether *text* was already ingested in *project*.

    Returns a ``{status, reason, doc_id}`` dict if the content is a duplicate
    (caller should short-circuit and return that response without ingesting).
    Returns ``None`` if the content is new — does NOT record the hash; call
    :func:`_record_dedup` after a successful ingest to do that.
    """
    content_hash = _compute_content_hash(text)
    bucket = _source_hashes.get(project, {})
    if content_hash in bucket:
        existing = bucket[content_hash]
        return {
            "status": "skipped",
            "reason": "already_ingested",
            "doc_id": existing["doc_id"],
        }
    return None


def _record_dedup(text: str, doc_id: str, project: str = "_global") -> None:
    """Record the content hash *after* a successful ingest.

    Must only be called once ``brain.ingest()`` (or equivalent) has succeeded
    so that a failed ingest cannot poison the dedup store.
    """
    content_hash = _compute_content_hash(text)
    _source_hashes.setdefault(project, {})[content_hash] = {
        "doc_id": doc_id,
        "last_ingested_at": datetime.now(timezone.utc).isoformat(),
    }


@app.on_event("startup")
async def startup_event():
    global brain
    try:
        config_path = os.getenv("AXON_CONFIG_PATH")
        config = AxonConfig.load(config_path)
        brain = AxonBrain(config)
        logger.info("Axon initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Axon: {e}")


# Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to ask the brain")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters for retrieval")
    stream: bool = Field(
        False, description="Whether to stream the response (not fully implemented in REST yet)"
    )
    # Per-request RAG overrides (match CLI flags exactly)
    top_k: int | None = Field(None, ge=1, description="Override number of chunks to retrieve")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override similarity threshold (0.0–1.0)"
    )
    hybrid: bool | None = Field(None, description="Override hybrid BM25+vector search toggle")
    rerank: bool | None = Field(None, description="Override cross-encoder re-ranking toggle")
    hyde: bool | None = Field(None, description="Override HyDE query transformation toggle")
    multi_query: bool | None = Field(None, description="Override multi-query retrieval toggle")
    step_back: bool | None = Field(None, description="Override step-back prompting toggle")
    decompose: bool | None = Field(None, description="Override query decomposition toggle")
    compress: bool | None = Field(None, description="Override LLM context compression toggle")
    discuss: bool | None = Field(None, description="Override discussion fallback toggle")


class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string for semantic search")
    top_k: int | None = Field(None, description="Number of documents to return")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")


class IngestRequest(BaseModel):
    path: str = Field(
        ...,
        description="Path to a file or directory to ingest. Must be within RAG_INGEST_BASE (default: current working directory).",
    )


class TextIngestRequest(BaseModel):
    text: str = Field(..., description="The content to ingest")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Metadata for the document"
    )
    doc_id: str | None = Field(None, description="Optional unique ID for the document")
    project: str | None = Field(
        None,
        description="Target project namespace. Defaults to the active project when omitted.",
    )


class BatchDocItem(BaseModel):
    """A single item within a batch ingest request."""

    text: str = Field(..., description="The content to ingest")
    doc_id: str | None = Field(
        None, description="Optional unique ID; a UUID4 prefix is assigned if omitted"
    )
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Optional metadata")


class BatchTextIngestRequest(BaseModel):
    docs: list[BatchDocItem] = Field(
        ..., description="List of documents to ingest in one batch (one embedding call)"
    )
    project: str | None = Field(
        None,
        description="Target project namespace applied to all docs. Defaults to the active project.",
    )


class URLIngestRequest(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch and ingest")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional extra metadata merged with the loader's source metadata",
    )
    project: str | None = Field(
        None,
        description="Target project namespace. Defaults to the active project when omitted.",
    )


class DeleteRequest(BaseModel):
    doc_ids: list[str] = Field(..., description="List of document IDs to delete")


class ProjectSwitchRequest(BaseModel):
    project_name: str = Field(
        ..., description="Project name to switch to, or 'default' for the global knowledge base"
    )


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "axon_ready": brain is not None}


@app.post("/query")
async def query_brain(request: QueryRequest):
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
        }
        response = brain.query(request.query, filters=request.filters, overrides=overrides)
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
        return {"query": request.query, "response": response, "settings": settings}
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_brain_stream(request: QueryRequest):
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
    }

    def generate():
        try:
            import json

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


@app.post("/search", response_model=list[SearchResult])
async def search_brain(request: SearchRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        # We need to expose the search method from AxonBrain more directly
        # or use the vector_store directly.
        # For agentic use, we'll implement a search flow here.
        query_embedding = brain.embedding.embed_query(request.query)
        top_k = request.top_k or brain.config.top_k

        results = brain.vector_store.search(
            query_embedding, top_k=top_k, filter_dict=request.filters
        )
        return results
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    validated_path = _validate_ingest_path(request.path)

    try:
        requested_path = pathlib.Path(os.path.realpath(validated_path))
        if not requested_path.exists():
            raise HTTPException(status_code=404, detail="Path does not exist")
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")

    # P1-C: create a tracked job
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    _evict_old_jobs()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "path": str(requested_path),
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "completed_at": None,
        "documents_ingested": None,
        "error": None,
    }

    def process_ingestion():
        import asyncio

        try:
            if requested_path.is_dir():
                asyncio.run(brain.load_directory(str(requested_path)))
            else:
                from axon.loaders import DirectoryLoader

                ext = requested_path.suffix.lower()
                loader_mgr = DirectoryLoader()
                if ext in loader_mgr.loaders:
                    docs = loader_mgr.loaders[ext].load(str(requested_path))
                    brain.ingest(docs)
                else:
                    logger.warning(f"Unsupported file type: {ext}")
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    background_tasks.add_task(process_ingestion)
    return {
        "message": f"Ingestion started for {validated_path}",
        "status": "processing",
        "job_id": job_id,
    }


@app.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str):
    """Poll the status of an async ingest job started by POST /ingest.

    Returns 404 if the job_id is unknown or has been evicted (TTL: 60 min).
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have already been evicted (TTL: 60 min) or the job_id is invalid.",
        )
    # Return a clean view without the internal timestamp field
    return {k: v for k, v in job.items() if k != "started_at_ts"}


@app.post("/add_text")
async def add_text(request: TextIngestRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    doc_id = request.doc_id or f"agent_doc_{uuid.uuid4().hex[:8]}"
    project_key = request.project or "_global"

    skip = _check_dedup(request.text, project_key)
    if skip:
        return {**skip, "doc_id": skip["doc_id"]}

    doc = {
        "id": doc_id,
        "text": request.text,
        "metadata": request.metadata or {"source": "api_agent", "type": "agent_input"},
    }

    try:
        brain.ingest([doc])
        _record_dedup(request.text, doc_id, project_key)
        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error adding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_texts")
async def add_texts(request: BatchTextIngestRequest):
    """Ingest a list of documents in a single embedding batch.

    Each item is checked for duplicate content before ingestion. Duplicate
    items receive ``status: skipped`` in the response without calling the
    embedding model. All non-duplicate items are ingested in one
    ``brain.ingest()`` call (one batched embedding round-trip).
    """
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    results: list[dict] = []
    docs_to_ingest: list[dict] = []
    project_key = request.project or "_global"

    # pending_records maps doc_id → item.text for dedup recording after ingest
    pending_records: list[tuple[str, str]] = []
    # within-batch dedup: track content_hash → doc_id for items in this request
    batch_hashes: dict[str, str] = {}

    for item in request.docs:
        doc_id = item.doc_id or f"agent_doc_{uuid.uuid4().hex[:8]}"
        skip = _check_dedup(item.text, project_key)
        if skip:
            results.append({"id": skip["doc_id"], "status": "skipped", "error": None})
            continue
        content_hash = _compute_content_hash(item.text)
        if content_hash in batch_hashes:
            results.append({"id": batch_hashes[content_hash], "status": "skipped", "error": None})
            continue
        batch_hashes[content_hash] = doc_id
        doc = {
            "id": doc_id,
            "text": item.text,
            "metadata": item.metadata or {"source": "api_agent", "type": "agent_input"},
        }
        docs_to_ingest.append(doc)
        pending_records.append((doc_id, item.text))
        results.append({"id": doc_id, "status": "created", "error": None})

    if docs_to_ingest:
        try:
            brain.ingest(docs_to_ingest)
            for doc_id, text in pending_records:
                _record_dedup(text, doc_id, project_key)
        except Exception as e:
            logger.error(f"Error batch-ingesting texts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return results


@app.post("/ingest_url")
async def ingest_url(request: URLIngestRequest):
    """Fetch an HTTP/HTTPS URL and ingest its text content.

    The URL is validated for scheme and SSRF-safe IP ranges before any network
    request is made. Binary or non-text responses are rejected. Duplicate
    content (same SHA-256 as a previously ingested document) is skipped.
    """
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    from axon.loaders import URLLoader

    loader = URLLoader()
    project_key = request.project or "_global"
    try:
        docs = loader.load(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error fetching URL '{request.url}': {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if not docs:
        raise HTTPException(status_code=422, detail="No content could be extracted from the URL.")

    doc = docs[0]
    if request.metadata:
        doc["metadata"].update(request.metadata)

    skip = _check_dedup(doc["text"], project_key)
    if skip:
        return {**skip, "doc_id": skip["doc_id"], "url": request.url}

    try:
        brain.ingest([doc])
        _record_dedup(doc["text"], doc["id"], project_key)
    except Exception as exc:
        logger.error(f"Error ingesting URL content: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ingested", "doc_id": doc["id"], "url": request.url}


@app.get("/collection")
async def get_collection():
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        docs = brain.list_documents()
        return {
            "total_files": len(docs),
            "total_chunks": sum(d["chunks"] for d in docs),
            "files": [{"source": d["source"], "chunks": d["chunks"]} for d in docs],
        }
    except Exception as e:
        logger.error(f"Error listing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/stale")
async def get_stale_docs(days: int = 7):
    """Return documents that have not been re-ingested within *days* calendar days.

    Staleness is measured against the ``last_ingested_at`` timestamp recorded in
    the in-memory dedup store (``_source_hashes``).  Documents ingested before
    the current server process started will not appear here — restart tracking
    only begins once the server has seen a document in its current lifetime.

    Query param:
    - ``days`` (int, default 7): flag docs not re-ingested in this many days.
    """
    if days < 0:
        raise HTTPException(status_code=400, detail="'days' must be >= 0.")

    cutoff = datetime.now(timezone.utc).timestamp() - days * 86_400
    stale: list[dict] = []
    for project_name, hashes in _source_hashes.items():
        for _content_hash, meta in hashes.items():
            try:
                ingested_ts = datetime.fromisoformat(meta["last_ingested_at"]).timestamp()
            except (KeyError, ValueError):
                continue
            if ingested_ts < cutoff:
                stale.append(
                    {
                        "doc_id": meta["doc_id"],
                        "project": project_name,
                        "last_ingested_at": meta["last_ingested_at"],
                        "age_days": round(
                            (datetime.now(timezone.utc).timestamp() - ingested_ts) / 86_400, 1
                        ),
                    }
                )
    return {"stale_docs": stale, "total": len(stale), "threshold_days": days}


@app.get("/projects")
async def get_projects():
    """List all projects known to the Axon system.

    Returns the on-disk project list from the projects store merged with
    in-memory session tracking from the dedup store.
    """
    try:
        on_disk = _list_projects()
    except Exception as exc:  # projects root may not exist yet in fresh installs
        logger.warning(f"Could not enumerate on-disk projects: {exc}")
        on_disk = []

    in_memory = list(_source_hashes.keys())
    on_disk_names = {p["name"] for p in on_disk}
    # Surface projects only seen in memory (e.g. via /add_text with project=)
    memory_only = [
        {"name": n, "source": "memory_only"}
        for n in in_memory
        if n not in on_disk_names and n != "_global"
    ]
    return {
        "projects": on_disk,
        "memory_only": memory_only,
        "total": len(on_disk) + len(memory_only),
    }


@app.post("/delete")
async def delete_documents(request: DeleteRequest):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        brain.vector_store.delete_by_ids(request.doc_ids)
        # Delete from BM25
        if brain.bm25 is not None:
            brain.bm25.delete_documents(request.doc_ids)
        return {"status": "success", "deleted": len(request.doc_ids), "doc_ids": request.doc_ids}
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project/switch")
async def switch_project(request: ProjectSwitchRequest):
    """Switch the active project, reinitializing vector store and BM25."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        brain.switch_project(request.project_name)
        return {"status": "success", "active_project": request.project_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Project switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for axon-api command."""
    host = os.getenv("AXON_HOST", "0.0.0.0")
    port = int(os.getenv("AXON_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
