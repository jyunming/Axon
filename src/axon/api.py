import asyncio
import getpass
import hashlib
import json
import logging
import os
import pathlib
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from axon import shares as _shares
from axon.main import AxonBrain, AxonConfig
from axon.projects import list_projects as _list_projects

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AxonAPI")


_BLOCKED_PATH_PREFIXES: tuple[pathlib.Path, ...] = tuple(
    pathlib.Path(p)
    for p in [
        # Windows system roots
        "C:/Windows",
        "C:/Windows/System32",
        "C:/Windows/SysWOW64",
        "C:/Program Files",
        "C:/Program Files (x86)",
        # Unix system roots
        "/etc",
        "/proc",
        "/sys",
        "/boot",
        "/root",
        "/usr/bin",
        "/usr/sbin",
        "/bin",
        "/sbin",
    ]
)


def _validate_ingest_path(path: str) -> str:
    """Validate that path is within the allowed base directory and not a blocked system path."""
    allowed_base = pathlib.Path(os.getenv("RAG_INGEST_BASE", ".")).resolve()
    abs_path = pathlib.Path(path).resolve()

    # Reject blocked system directories regardless of RAG_INGEST_BASE
    for blocked in _BLOCKED_PATH_PREFIXES:
        try:
            abs_path.relative_to(blocked)
            raise HTTPException(
                status_code=403,
                detail=f"Path '{path}' resolves to a blocked system directory.",
            )
        except ValueError:
            pass

    try:
        abs_path.relative_to(allowed_base)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Path '{path}' is outside the allowed ingest directory. Set RAG_INGEST_BASE to permit additional paths.",
        )
    return str(abs_path)


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


app = FastAPI(
    title="Axon API",
    description="REST API for agent orchestration and document retrieval",
    version="1.0.0",
    lifespan=lifespan,
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


# Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to ask the brain")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters for retrieval")
    stream: bool = Field(
        False, description="Whether to stream the response (use POST /query/stream instead)"
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
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Override LLM temperature for this request (0.0–2.0)"
    )
    timeout: float | None = Field(None, gt=0, description="Query timeout in seconds (default 120)")


class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string for semantic search")
    top_k: int | None = Field(None, description="Number of documents to return")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override similarity threshold for this request"
    )


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
    project_name: str | None = Field(
        None, description="Project name to switch to, or 'default' for the global knowledge base"
    )
    name: str | None = Field(None, description="Alias for project_name")

    @property
    def final_name(self) -> str:
        val = self.project_name or self.name
        if not val:
            raise ValueError("Either 'project_name' or 'name' must be provided")
        return val


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., description="Name of the new project to create")
    description: str = Field("", description="Optional description for the project")


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class StoreInitRequest(BaseModel):
    base_path: str = Field(
        ..., description="Base directory under which AxonStore/ will be created."
    )


class ShareGenerateRequest(BaseModel):
    project: str = Field(..., description="Project name to share.")
    grantee: str = Field(..., description="OS username of the recipient.")
    write_access: bool = Field(
        False, description="Grant write (ingest) access. Default: read-only."
    )


class ShareRedeemRequest(BaseModel):
    share_string: str = Field(..., description="The base64 share string from the owner.")


class ShareRevokeRequest(BaseModel):
    key_id: str = Field(..., description="The key_id to revoke (e.g. 'sk_a1b2c3d4').")


def _get_user_dir() -> Path:
    """Return the current user's AxonStore directory, or raise 503 if not in store mode."""
    if not brain or not brain.config.axon_store_mode:
        raise HTTPException(
            status_code=503,
            detail="AxonStore mode is not active. Run POST /store/init first.",
        )
    return Path(brain.config.projects_root)


# Endpoints
class CopilotMessage(BaseModel):
    role: str
    content: str


class CopilotAgentRequest(BaseModel):
    """Payload sent by GitHub Copilot to the agent endpoint."""

    messages: list[CopilotMessage]
    copilot_references: list[dict] = Field(default_factory=list)
    agent_request_id: str | None = None


@app.post("/copilot/agent")
async def copilot_agent_handler(request: Request, body: CopilotAgentRequest):
    """
    Handle chat requests from GitHub Copilot.
    Supports slash commands: /search, /ingest, /projects.
    """
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    user_query = body.messages[-1].content if body.messages else ""
    if not user_query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # 1. Parse Slash Commands
    parts = user_query.strip().split(maxsplit=1)
    command = parts[0].lower() if parts[0].startswith("/") else None
    args = parts[1] if len(parts) > 1 else ""

    async def response_stream():
        yield f"data: {json.dumps({'type': 'created', 'id': body.agent_request_id})}\n\n"

        try:
            if command == "/search":
                yield f"data: {json.dumps({'type': 'text', 'content': f'🔍 Searching Axon for: {args}...'})}\n\n"
                retrieval_data = brain._execute_retrieval(args)
                results = retrieval_data["results"]
                if not results:
                    content = "No relevant documents found."
                else:
                    content = "### Search Results\n\n"
                    for i, r in enumerate(results[:5]):
                        content += f"**{i+1}. {os.path.basename(r['metadata'].get('source', 'unknown'))}** (Score: {r['score']:.2f})\n"
                        content += f"> {r['text'][:200]}...\n\n"
                yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

            elif command == "/ingest":
                yield f"data: {json.dumps({'type': 'text', 'content': f'📥 Ingesting URL: {args}...'})}\n\n"
                # Use existing ingest_url logic
                from axon.loaders import URLLoader

                loader = URLLoader()
                docs = loader.load(args)
                if docs:
                    brain.ingest(docs)
                    yield f"data: {json.dumps({'type': 'text', 'content': f'✅ Successfully ingested: {args}'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'text', 'content': f'❌ Failed to ingest: {args}'})}\n\n"

            elif command == "/projects":
                projects = _list_projects()
                content = "### Available Axon Projects\n\n"
                for p in projects:
                    content += f"- **{p['name']}**: {p.get('description', 'No description')}\n"
                yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

            else:
                # Default: RAG query
                answer = brain.query(
                    user_query,
                    chat_history=[
                        {"role": m.role, "content": m.content} for m in body.messages[:-1]
                    ],
                )

                # Yield result with GitHub-compatible formatting
                yield f"data: {json.dumps({'type': 'text', 'content': answer})}\n\n"

        except Exception as e:
            logger.error(f"Copilot Agent error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")


class ConfigUpdateRequest(BaseModel):
    # Model
    llm_provider: str | None = None
    llm_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    # RAG
    top_k: int | None = Field(None, ge=1, le=50)
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    hybrid_search: bool | None = None
    hybrid_weight: float | None = Field(None, ge=0.0, le=1.0)
    rerank: bool | None = None
    reranker_model: str | None = None
    hyde: bool | None = None
    multi_query: bool | None = None
    step_back: bool | None = None
    query_decompose: bool | None = None
    compress_context: bool | None = None
    truth_grounding: bool | None = None
    discussion_fallback: bool | None = None
    raptor: bool | None = None
    graph_rag: bool | None = None
    # Persistence
    persist: bool = Field(False, description="Whether to save these changes to config.yaml")


@app.get("/config")
async def get_config():
    """Return the current active configuration."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain.config


@app.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """Update global configuration settings."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    update_data = request.dict(exclude_unset=True)
    persist = update_data.pop("persist", False)

    reinit_llm = "llm_provider" in update_data or "llm_model" in update_data
    reinit_embed = "embedding_provider" in update_data or "embedding_model" in update_data
    reinit_rerank = "reranker_model" in update_data

    for k, v in update_data.items():
        if hasattr(brain.config, k):
            setattr(brain.config, k, v)

    if reinit_llm:
        from axon.main import OpenLLM

        brain.llm = OpenLLM(brain.config)
    if reinit_embed:
        from axon.main import OpenEmbedding

        brain.embedding = OpenEmbedding(brain.config)
    if reinit_rerank and brain.config.rerank:
        from axon.main import OpenReranker

        brain.reranker = OpenReranker(brain.config)

    if persist:
        brain.config.save()

    return {"status": "success", "config": brain.config, "persisted": persist}


@app.post("/project/delete/{name}")
async def delete_project_endpoint(name: str):
    """Delete a project and all its data."""
    from axon.projects import ProjectHasChildrenError, delete_project

    # Reject attempts to delete ShareMount symlinks (those are not owned data)
    if name.startswith("ShareMount/") or name == "ShareMount":
        raise HTTPException(
            status_code=400,
            detail="ShareMount entries are symlinks to shared projects and cannot be deleted via this endpoint.",
        )

    # In store mode: check if project has active (non-revoked) issued share keys
    if brain and brain.config.axon_store_mode:
        user_dir = Path(brain.config.projects_root)
        shares_info = _shares.list_shares(user_dir)
        active_grantees = [
            s["grantee"]
            for s in shares_info.get("sharing", [])
            if s["project"] == name and not s.get("revoked", False)
        ]
        if active_grantees:
            raise HTTPException(
                status_code=409,
                detail=f"Project '{name}' has active shares with: {', '.join(active_grantees)}. Revoke shares before deleting.",
            )

    if brain and brain._active_project == name:
        brain.switch_project("default")

    try:
        delete_project(name)
        return {"status": "success", "message": f"Project '{name}' deleted."}
    except ProjectHasChildrenError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/store/init")
async def store_init(request: StoreInitRequest):
    """Initialise AxonStore at the given base path and update config.yaml."""
    global brain
    base = Path(request.base_path).expanduser().resolve()
    username = getpass.getuser()
    store_root = base / "AxonStore"
    user_dir = store_root / username

    # Create directory structure
    from axon.projects import ensure_user_namespace

    ensure_user_namespace(user_dir)

    # Update config: set axon_store_base, repoint paths
    config = brain.config if brain else AxonConfig()
    config.axon_store_base = str(base)
    config.axon_store_mode = True
    config.projects_root = str(user_dir)
    config.vector_store_path = str(user_dir / "_default" / "chroma_data")
    config.bm25_path = str(user_dir / "_default" / "bm25_index")

    # Save config — migrate: add store block, remove projects_root if present
    try:
        config.save()
    except Exception as e:
        logger.warning(f"Could not save config after store init: {e}")

    # Reinitialise brain with new config
    if brain:
        brain.close()
    brain = AxonBrain(config)

    return {
        "status": "ok",
        "store_path": str(store_root),
        "user_dir": str(user_dir),
        "username": username,
    }


@app.get("/store/whoami")
async def store_whoami():
    """Return current user identity and AxonStore status."""
    username = getpass.getuser()
    if brain and brain.config.axon_store_mode:
        return {
            "username": username,
            "store_path": str(Path(brain.config.projects_root).parent.parent),
            "user_dir": brain.config.projects_root,
            "active_project": brain.config.project
            if hasattr(brain.config, "project")
            else "_default",
            "store_mode": True,
        }
    return {"username": username, "store_mode": False}


@app.post("/share/generate")
async def share_generate(request: ShareGenerateRequest):
    """Generate a share key allowing another user to access one of your projects."""
    user_dir = _get_user_dir()
    # Validate project exists
    project_dir = user_dir / request.project
    if not project_dir.exists() or not (project_dir / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project}' not found.")
    result = _shares.generate_share_key(
        owner_user_dir=user_dir,
        project=request.project,
        grantee=request.grantee,
        write_access=request.write_access,
    )
    return result


@app.post("/share/redeem")
async def share_redeem(request: ShareRedeemRequest):
    """Redeem a share string, mounting the owner's project in your ShareMount/."""
    user_dir = _get_user_dir()
    try:
        result = _shares.redeem_share_key(
            grantee_user_dir=user_dir,
            share_string=request.share_string,
        )
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/share/revoke")
async def share_revoke(request: ShareRevokeRequest):
    """Revoke a share key. The grantee's symlink is removed on their next access."""
    user_dir = _get_user_dir()
    try:
        result = _shares.revoke_share_key(owner_user_dir=user_dir, key_id=request.key_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@app.get("/share/list")
async def share_list():
    """List shares for the current user: both issued (sharing) and received (shared)."""
    user_dir = _get_user_dir()
    # Lazily clean up revoked received shares
    removed = _shares.validate_received_shares(user_dir)
    result = _shares.list_shares(user_dir)
    if removed:
        result["removed_stale"] = removed
    return result


@app.get("/sessions")
async def list_sessions():
    """List all saved chat sessions for the active project."""
    from axon.main import _list_sessions

    return {"sessions": _list_sessions()}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve a specific session by ID."""
    from axon.main import _load_session

    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/clear")
async def clear_brain():
    """Clear the active project's vector store, BM25 index, hash store, and entity graph."""
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

        # Delete embedding metadata so the project can be re-ingested with a
        # different embedding model without hitting a stale mismatch error.
        _meta_path = pathlib.Path(brain._embedding_meta_path)
        if _meta_path.exists():
            try:
                _meta_path.unlink()
            except OSError:
                pass

        return {"status": "success", "message": "Collection cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CopilotTaskResult(BaseModel):
    result: str | None = None
    error: str | None = None


@app.get("/llm/copilot/tasks")
async def get_copilot_tasks():
    """Poll for pending LLM tasks intended for the VS Code Copilot bridge."""
    from axon.main import _copilot_bridge_lock, _copilot_task_queue

    with _copilot_bridge_lock:
        tasks = list(_copilot_task_queue)
        _copilot_task_queue.clear()
    return {"tasks": tasks}


@app.post("/llm/copilot/result/{task_id}")
async def submit_copilot_result(task_id: str, body: CopilotTaskResult):
    """Submit the result of a Copilot LLM task back to the backend."""
    from axon.main import _copilot_bridge_lock, _copilot_responses

    with _copilot_bridge_lock:
        if task_id in _copilot_responses:
            _copilot_responses[task_id]["result"] = body.result
            _copilot_responses[task_id]["error"] = body.error
            _copilot_responses[task_id]["event"].set()
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Task not found or expired")


@app.get("/health")
async def health_check():
    """Return 200 with status 'ok' when the brain is ready; 503 with status 'initializing' when not yet available."""
    if brain is None:
        return JSONResponse({"status": "initializing"}, status_code=503)
    return {"status": "ok", "project": getattr(brain, "_active_project", "default")}


@app.get("/tracked-docs")
async def list_tracked_docs():
    """List all tracked document sources with metadata."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return {"docs": brain.get_doc_versions()}


@app.post("/ingest/refresh")
async def refresh_docs():
    """Re-check all tracked files and re-ingest changed ones."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    import hashlib as _hashlib

    versions = brain.get_doc_versions()
    results: dict[str, list[str]] = {"skipped": [], "reingest_needed": [], "missing": []}
    for source_id, record in versions.items():
        if not os.path.exists(source_id):
            results["missing"].append(source_id)
            continue
        try:
            with open(source_id, "rb") as f:
                content_hash = _hashlib.md5(f.read()).hexdigest()
            if content_hash != record.get("content_hash"):
                results["reingest_needed"].append(source_id)
            else:
                results["skipped"].append(source_id)
        except Exception:
            results["missing"].append(source_id)
    return results


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
            "llm_temperature": request.temperature,
        }

        # Offload sync brain.query to a threadpool to keep the event loop free
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        timeout = request.timeout or float(os.getenv("AXON_QUERY_TIMEOUT", "120"))
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
        return {"query": request.query, "response": response, "settings": settings}
    except HTTPException:
        raise
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
        "llm_temperature": request.temperature,
    }

    def generate():
        try:
            # Use query_stream (generator)
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
        # Offload sync retrieval execution to a threadpool
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
        from axon.loaders import DirectoryLoader

        try:
            loader_mgr = DirectoryLoader()
            if requested_path.is_dir():
                # Use the sync loader to avoid asyncio.run() inside a background
                # thread — on Windows, ProactorEventLoop created by asyncio.run()
                # in a non-main thread can hang indefinitely.
                docs = loader_mgr.load(str(requested_path))
            else:
                ext = requested_path.suffix.lower()
                if ext in loader_mgr.loaders:
                    docs = loader_mgr.loaders[ext].load(str(requested_path))
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    docs = []
            if docs:
                brain.ingest(docs)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["documents_ingested"] = len(docs)
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

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    doc_id = request.doc_id or f"agent_doc_{uuid.uuid4().hex[:8]}"
    project_key = request.project or brain._active_project

    skip = _check_dedup(request.text, project_key)
    if skip:
        return {**skip, "doc_id": skip["doc_id"]}

    # Use SmartTextLoader to detect and handle tabular content in raw text
    from axon.loaders import SmartTextLoader

    loader = SmartTextLoader()
    documents = loader.load_text(request.text, source=doc_id)

    # Apply metadata to all resulting chunks
    if request.metadata:
        for doc in documents:
            doc["metadata"].update(request.metadata)

    try:
        brain.ingest(documents)
        _record_dedup(request.text, doc_id, project_key)
        return {"status": "success", "doc_id": doc_id, "chunks": len(documents)}
    except Exception as e:
        logger.error(f"Error adding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_texts")
async def add_texts(request: BatchTextIngestRequest):
    """Ingest a list of documents in a single embedding batch."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    results: list[dict] = []
    docs_to_ingest: list[dict] = []
    project_key = request.project or brain._active_project

    # pending_records maps doc_id → item.text for dedup recording after ingest
    pending_records: list[tuple[str, str]] = []
    # within-batch dedup: track content_hash → doc_id for items in this request
    batch_hashes: dict[str, str] = {}

    from axon.loaders import SmartTextLoader

    loader = SmartTextLoader()

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

        item_docs = loader.load_text(item.text, source=doc_id)
        if item.metadata:
            for d in item_docs:
                d["metadata"].update(item.metadata)

        docs_to_ingest.extend(item_docs)
        pending_records.append((doc_id, item.text))
        results.append({"id": doc_id, "status": "created", "chunks": len(item_docs)})

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
    """Fetch an HTTP/HTTPS URL and ingest its text content."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    from axon.loaders import URLLoader

    loader = URLLoader()
    project_key = request.project or brain._active_project
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
    """Return documents that have not been re-ingested within *days* calendar days."""
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
    """List all projects known to the Axon system."""
    # In store mode: lazily clean up revoked received shares
    if brain and brain.config.axon_store_mode:
        try:
            user_dir = Path(brain.config.projects_root)
            _shares.validate_received_shares(user_dir)
        except Exception as exc:
            logger.warning(f"Could not validate received shares: {exc}")

    try:
        on_disk = _list_projects()
    except Exception as exc:
        logger.warning(f"Could not enumerate on-disk projects: {exc}")
        on_disk = []

    in_memory = list(_source_hashes.keys())
    on_disk_names = {p["name"] for p in on_disk}
    memory_only = [
        {"name": n, "source": "memory_only"}
        for n in in_memory
        if n not in on_disk_names and n != "_global"
    ]

    # In store mode: include share mount entries
    shared_mounts = []
    if brain and brain.config.axon_store_mode:
        try:
            from axon.projects import list_share_mounts

            user_dir = Path(brain.config.projects_root)
            mounts = list_share_mounts(user_dir)
            shared_mounts = [
                {
                    "name": f"ShareMount/{m['name']}",
                    "owner": m["owner"],
                    "project": m["project"],
                    "is_broken": m["is_broken"],
                    "is_shared": True,
                }
                for m in mounts
                if not m["is_broken"]
            ]
        except Exception as exc:
            logger.warning(f"Could not enumerate share mounts: {exc}")

    return {
        "projects": on_disk,
        "memory_only": memory_only,
        "shared_mounts": shared_mounts,
        "total": len(on_disk) + len(memory_only) + len(shared_mounts),
    }


@app.post("/delete")
async def delete_documents(request: DeleteRequest):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        # Resolve which IDs actually exist before deleting so the count is accurate
        existing = brain.vector_store.get_by_ids(request.doc_ids)
        existing_ids_set = {doc["id"] for doc in existing}
        # Preserve original request order for the response
        existing_ids = [i for i in request.doc_ids if i in existing_ids_set]
        not_found = [i for i in request.doc_ids if i not in existing_ids_set]
        if existing_ids:
            brain.vector_store.delete_by_ids(existing_ids)
            if brain.bm25 is not None:
                brain.bm25.delete_documents(existing_ids)
        return {
            "status": "success",
            "deleted": len(existing_ids),
            "doc_ids": existing_ids,
            "not_found": not_found,
        }
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


_VALID_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}(?:/[A-Za-z0-9_\-]{1,64}){0,4}$")


@app.post("/project/new")
async def create_project(request: ProjectCreateRequest):
    """Create a new project directory and metadata."""
    from axon.projects import ensure_project

    if not request.name or not _VALID_PROJECT_NAME_RE.match(request.name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid project name. Use 1-5 slash-separated segments of "
                "1-64 alphanumeric characters, hyphens, or underscores "
                "(e.g. 'research/papers/2024')."
            ),
        )
    try:
        ensure_project(request.name, description=request.description)
        return {"status": "success", "project": request.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project/switch")
async def switch_project(request: ProjectSwitchRequest):
    """Switch the active project, reinitializing vector store and BM25."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        project_name = request.final_name
        brain.switch_project(project_name)
        return {"status": "success", "active_project": project_name}
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
