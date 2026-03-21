"""Axon REST API — lifespan, app creation, middleware, and router registration."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from axon import shares as _shares  # noqa: F401 — tests patch axon.api._shares.*
from axon.api_schemas import _compute_content_hash  # noqa: F401
from axon.main import AxonBrain, AxonConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AxonAPI")

# ---------------------------------------------------------------------------
# Global mutable state — tests access these directly as api_module.brain etc.
# ---------------------------------------------------------------------------

# Global Brain Instance
brain: AxonBrain | None = None

# Source-level dedup store: project → content_hash → {doc_id, last_ingested_at}
_source_hashes: dict[str, dict[str, dict]] = {}

# Async ingest job status store (in-memory, single-worker deployments only)
_jobs: dict[str, dict] = {}
_MAX_JOBS = 1000
_JOB_TTL_SECONDS = 3600  # 60 minutes

# ---------------------------------------------------------------------------
# State-using helpers (depend on module-level brain/_jobs/_source_hashes)
# ---------------------------------------------------------------------------


def _evict_old_jobs() -> None:
    """Remove completed jobs older than _JOB_TTL_SECONDS and cap at _MAX_JOBS."""
    cutoff = datetime.now(timezone.utc).timestamp() - _JOB_TTL_SECONDS
    expired = [jid for jid, job in list(_jobs.items()) if job.get("started_at_ts", 0) < cutoff]
    for jid in expired:
        _jobs.pop(jid, None)
    if len(_jobs) > _MAX_JOBS:
        oldest = sorted(_jobs, key=lambda j: _jobs[j].get("started_at_ts", 0))
        for jid in oldest[: len(_jobs) - _MAX_JOBS]:
            _jobs.pop(jid, None)


def _check_dedup(text: str, project: str = "_global") -> dict | None:
    """Check whether *text* was already ingested in *project*.

    Returns a ``{status, reason, doc_id}`` dict if the content is a duplicate.
    Returns ``None`` if the content is new.
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
    """Record the content hash after a successful ingest."""
    content_hash = _compute_content_hash(text)
    _source_hashes.setdefault(project, {})[content_hash] = {
        "doc_id": doc_id,
        "last_ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_user_dir() -> Path:
    """Return the current user's AxonStore directory, or raise 503 if not in store mode."""
    if not brain or not brain.config.axon_store_mode:
        raise HTTPException(
            status_code=503,
            detail="AxonStore mode is not active. Run POST /store/init first.",
        )
    return Path(brain.config.projects_root)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain
    try:
        config_path = os.getenv("AXON_CONFIG_PATH")
        config = AxonConfig.load(config_path)
        brain = AxonBrain(config)
        logger.info("Axon initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Axon: {e}")
    yield
    if brain:
        brain.close()
        logger.info("Axon shut down cleanly")


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Axon API",
    description="REST API for agent orchestration and document retrieval",
    version="0.9.0",
    lifespan=lifespan,
)

# Optional API key authentication
_RAG_API_KEY: str | None = os.getenv("RAG_API_KEY")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach X-Request-ID to every request/response for audit traceability."""
    from uuid import uuid4

    rid = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce X-API-Key header when RAG_API_KEY is configured."""
    if _RAG_API_KEY:
        if request.url.path != "/health":
            provided = request.headers.get("X-API-Key")
            if provided != _RAG_API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-API-Key header."},
                )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Router registration — each sub-module registers its own APIRouter
# ---------------------------------------------------------------------------

from axon.api_routes.governance import router as _governance_router  # noqa: E402
from axon.api_routes.graph import router as _graph_router  # noqa: E402
from axon.api_routes.ingest import router as _ingest_router  # noqa: E402
from axon.api_routes.maintenance import router as _maintenance_router  # noqa: E402
from axon.api_routes.projects import router as _projects_router  # noqa: E402
from axon.api_routes.query import router as _query_router  # noqa: E402
from axon.api_routes.registry import router as _registry_router  # noqa: E402
from axon.api_routes.shares import router as _shares_router  # noqa: E402

app.include_router(_query_router)
app.include_router(_ingest_router)
app.include_router(_projects_router)
app.include_router(_graph_router)
app.include_router(_shares_router)
app.include_router(_maintenance_router)
app.include_router(_registry_router)
app.include_router(_governance_router)

# ---------------------------------------------------------------------------
# Backward-compat re-exports so existing importers of axon.api still work
# ---------------------------------------------------------------------------

from axon.api_schemas import (  # noqa: E402,F401
    _BLOCKED_PATH_PREFIXES,
    _VALID_PROJECT_NAME_RE,
    BatchDocItem,
    BatchTextIngestRequest,
    ConfigUpdateRequest,
    CopilotAgentRequest,
    CopilotMessage,
    CopilotTaskResult,
    DeleteRequest,
    IngestRequest,
    MaintenanceStateRequest,
    ProjectCreateRequest,
    ProjectSwitchRequest,
    QueryRequest,
    SearchRequest,
    SearchResult,
    ShareGenerateRequest,
    ShareRedeemRequest,
    ShareRevokeRequest,
    StoreInitRequest,
    TextIngestRequest,
    URLIngestRequest,
    _validate_ingest_path,
)


def main():
    """Main entry point for axon-api command."""
    host = os.getenv("AXON_HOST", "0.0.0.0")
    port = int(os.getenv("AXON_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
