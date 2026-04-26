"""Axon REST API — lifespan, app creation, middleware, and router registration."""


from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from axon import __version__
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


def get_brain() -> AxonBrain:
    """FastAPI dependency that ensures the brain is initialized."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain


def get_brain_optional() -> AxonBrain | None:
    """FastAPI dependency that returns the brain if available, else None."""
    return brain


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
    """Remove completed/failed jobs older than _JOB_TTL_SECONDS and cap at _MAX_JOBS.
    Processing jobs are never TTL-evicted so long-running ingests remain queryable.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - _JOB_TTL_SECONDS
    expired = [
        jid
        for jid, job in list(_jobs.items())
        if job.get("started_at_ts", 0) < cutoff and job.get("status") != "processing"
    ]
    for jid in expired:
        _jobs.pop(jid, None)
    if len(_jobs) > _MAX_JOBS:
        # Prefer evicting finished jobs; only remove active ones as last resort.
        finished = sorted(
            (j for j in _jobs if _jobs[j].get("status") != "processing"),
            key=lambda j: _jobs[j].get("started_at_ts", 0),
        )
        active = sorted(
            (j for j in _jobs if _jobs[j].get("status") == "processing"),
            key=lambda j: _jobs[j].get("started_at_ts", 0),
        )
        to_remove = finished + active
        for jid in to_remove[: len(_jobs) - _MAX_JOBS]:
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


def _purge_dedup(doc_ids: list[str], project: str | None = None) -> None:
    """Remove dedup entries that reference *doc_ids*."""
    targets = []
    if project:
        targets.append(project)
    if "_global" not in targets:
        targets.append("_global")
    for target in targets:
        bucket = _source_hashes.get(target)
        if not bucket:
            continue
        for content_hash, meta in list(bucket.items()):
            if meta.get("doc_id") in doc_ids:
                bucket.pop(content_hash, None)
        if not bucket:
            _source_hashes.pop(target, None)


def _get_user_dir() -> Path:
    """Return the current user's AxonStore directory, or raise 503 if brain is unavailable."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized.")
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
        # Option A: auto-init store on first run so the user never hits a
        # "store not found" failure on a fresh install.
        _auto_init_store(config)
        brain = AxonBrain(config)
        logger.info("Axon initialized successfully")
    except Exception as e:
        # Re-raise so the server fails fast rather than serving every
        # request with brain=None and 503-ing the user. Previously this
        # was a silent log-and-continue, which masked config errors and
        # surfaced as confusing "Brain not initialized" everywhere.
        logger.error(f"Failed to initialize Axon: {e}")
        raise
    yield
    if brain:
        brain.close()
        logger.info("Axon shut down cleanly")


def _auto_init_store(config: AxonConfig) -> None:
    """Create the default AxonStore layout when it does not yet exist.
    This is a silent first-run helper — it runs ``ensure_user_project()`` only
    when the user directory is missing so that brand-new installs work without
    requiring a manual ``init_store`` call first.
    """
    from axon.projects import ensure_user_project

    user_dir = Path(config.projects_root)
    store_meta = user_dir / "store_meta.json"
    if not store_meta.exists():
        logger.info("AxonStore not found at %s — creating default layout.", user_dir)
        ensure_user_project(user_dir)
        logger.info("AxonStore initialised at %s.", user_dir)


# ---------------------------------------------------------------------------

# App creation

# ---------------------------------------------------------------------------

app = FastAPI(
    title="Axon API",
    description="REST API for agent orchestration and document retrieval",
    version=__version__,
    lifespan=lifespan,
)

# Apply CORS based on AxonConfig.api_allow_origins. Without this, the
# ``api.allow_origins`` config knob was silently dead — the YAML loaded,
# the dataclass field accepted the value, but no middleware enforced it,
# so cross-origin browser callers (e.g. the VS Code panel webview) saw
# blocked responses regardless of configuration.
#
# Read directly from the YAML file (not via AxonConfig.load(), which
# materialises a default config on disk if none exists). This avoids
# import-time filesystem side effects and lets late-set
# ``AXON_CONFIG_PATH`` env vars in tests still be honoured by the
# lifespan handler — CORS just defaults to "no origins" when the file
# isn't present, which matches the dataclass default.
try:
    from fastapi.middleware.cors import CORSMiddleware as _CORSMiddleware

    def _load_cors_origins_from_disk() -> list[str]:
        from axon.config import _USER_CONFIG_PATH

        cfg_path = os.getenv("AXON_CONFIG_PATH") or str(_USER_CONFIG_PATH)
        if not os.path.isfile(cfg_path):
            return []
        try:
            import yaml as _yaml  # type: ignore[import-untyped]

            with open(cfg_path, encoding="utf-8") as fh:
                raw = _yaml.safe_load(fh) or {}
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("Could not parse %s for CORS: %s", cfg_path, exc)
            return []
        api_section = raw.get("api") if isinstance(raw, dict) else None
        if not isinstance(api_section, dict):
            return []
        origins = api_section.get("allow_origins") or []
        return [str(o) for o in origins if o]

    _cors_origins = _load_cors_origins_from_disk()
    if _cors_origins:
        app.add_middleware(
            _CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Axon-Mount-Sync-Pending"],
        )
        logger.info("CORS enabled for origins: %s", _cors_origins)
except ImportError:  # pragma: no cover — fastapi installed implies starlette
    pass

# Optional API key authentication

_RAG_API_KEY: str | None = os.getenv("RAG_API_KEY")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach X-Request-ID and X-Axon-Surface to every request for audit traceability."""
    from uuid import uuid4

    rid = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = rid
    request.state.surface = request.headers.get("X-Axon-Surface", "api")
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request count + latency in the Prometheus exporter.

    No-op when prometheus-client is not installed (the helper functions
    short-circuit). Lives next to add_request_id so both observability
    middlewares run on every request.
    """
    import time as _time

    from axon.api_routes import metrics as _metrics

    started = _time.perf_counter()
    response = await call_next(request)
    duration = _time.perf_counter() - started
    # Use route template when available so high-cardinality dynamic
    # path segments (e.g. /projects/{name}) collapse into one series.
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    _metrics.record_request(
        path=path,
        method=request.method,
        status=response.status_code,
        duration_seconds=duration,
    )
    return response


# Paths that bypass the X-API-Key check. /metrics is public so Prometheus
# scrapers do not need to ship the secret; /health{,/live,/ready} keep
# liveness probes from spuriously failing when the API key rotates.
_AUTH_BYPASS_EXACT = frozenset({"/health", "/health/live", "/health/ready", "/metrics", "/gui"})
_AUTH_BYPASS_PREFIX = ("/v1/health", "/v1/metrics", "/gui/")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce X-API-Key header when RAG_API_KEY is configured."""
    if _RAG_API_KEY:
        path = request.url.path
        if path in _AUTH_BYPASS_EXACT or path.startswith(_AUTH_BYPASS_PREFIX):
            return await call_next(request)
        provided = request.headers.get("X-API-Key")
        if not hmac.compare_digest(provided or "", _RAG_API_KEY):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header."},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Static files for WebGUI
# ---------------------------------------------------------------------------

from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

gui_dir = Path(__file__).parent / "gui"
if gui_dir.exists():

    @app.get("/gui", include_in_schema=False)
    async def gui_redirect():
        return RedirectResponse(url="/gui/")

    app.mount("/gui/", StaticFiles(directory=str(gui_dir), html=True), name="gui")


# ---------------------------------------------------------------------------
# Router registration — each sub-module registers its own APIRouter
# ---------------------------------------------------------------------------

from axon.api_routes.config_routes import router as _config_router  # noqa: E402
from axon.api_routes.governance import router as _governance_router  # noqa: E402
from axon.api_routes.graph import router as _graph_router  # noqa: E402
from axon.api_routes.health import router as _health_router  # noqa: E402
from axon.api_routes.ingest import router as _ingest_router  # noqa: E402
from axon.api_routes.maintenance import router as _maintenance_router  # noqa: E402
from axon.api_routes.metrics import router as _metrics_router  # noqa: E402
from axon.api_routes.projects import router as _projects_router  # noqa: E402
from axon.api_routes.query import router as _query_router  # noqa: E402
from axon.api_routes.registry import router as _registry_router  # noqa: E402
from axon.api_routes.security_routes import router as _security_router  # noqa: E402
from axon.api_routes.shares import router as _shares_router  # noqa: E402

_ROUTERS = (
    _health_router,
    _metrics_router,
    _config_router,
    _query_router,
    _ingest_router,
    _projects_router,
    _graph_router,
    _shares_router,
    _maintenance_router,
    _registry_router,
    _governance_router,
    _security_router,
)

for _router in _ROUTERS:
    app.include_router(_router)
    app.include_router(_router, prefix="/v1")

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
    uvicorn.run("axon.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
