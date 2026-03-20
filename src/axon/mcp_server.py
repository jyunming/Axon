"""
src/axon/mcp_server.py

MCP stdio server for Axon — exposes the Axon REST API as MCP tools so
Copilot (or any other agent) can call them from agent mode.

Tool names here are deliberately shorter than the OpenAI-format names in
tools.py; do not conflate the two sets.

Environment variables
---------------------
RAG_API_BASE  : Base URL of the running Axon API  (default: http://localhost:8000)
RAG_API_KEY   : API key for X-API-Key header      (default: empty — auth disabled)

Usage
-----
Run as a stdio process (used by .vscode/mcp.json):

    python -m axon.mcp_server
    # or after pip install -e .:
    axon-mcp
"""

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE: str = os.getenv("RAG_API_BASE", "http://localhost:8000").rstrip("/")
API_KEY: str | None = os.getenv("RAG_API_KEY") or None

mcp = FastMCP("axon")


def _headers() -> dict[str, str]:
    """Return request headers, including X-API-Key when configured."""
    h: dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


async def _get(path: str) -> Any:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(f"{API_BASE}{path}", headers=_headers())
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, body: dict) -> Any:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{API_BASE}{path}", json=body, headers=_headers())
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def ingest_text(text: str, metadata: dict | None = None, project: str | None = None) -> Any:
    """Ingest a single text document into the Axon knowledge base.

    Prefer ingest_texts for multiple documents — it uses one embedding call.
    Always set metadata.source so the collection can be audited.
    Duplicate content (same SHA-256) is silently skipped; status will be 'skipped'.

    Args:
        text: The text content to store.
        metadata: Optional dict, e.g. {"source": "https://...", "topic": "react"}.
        project: Target project namespace. Omit to use the active project.
    """
    body: dict = {"text": text}
    if metadata:
        body["metadata"] = metadata
    if project:
        body["project"] = project
    return await _post("/add_text", body)


@mcp.tool()
async def ingest_texts(docs: list[dict], project: str | None = None) -> Any:
    """Ingest multiple documents in a single batched embedding call.

    Each item must have at least a "text" key. Optional keys: "doc_id", "metadata".
    This is the preferred ingest tool — never call ingest_text in a loop.

    Args:
        docs: List of dicts, each with "text" and optional "doc_id"/"metadata".
        project: Target project namespace applied to all docs.
    """
    body: dict = {"docs": docs}
    if project:
        body["project"] = project
    return await _post("/add_texts", body)


@mcp.tool()
async def ingest_url(url: str, metadata: dict | None = None, project: str | None = None) -> Any:
    """Fetch an HTTP/HTTPS URL and ingest its text content.

    HTML is stripped automatically. Private/internal URLs (127.x, 10.x,
    192.168.x, 169.254.x, 172.16-31.x) are blocked server-side.

    Args:
        url: The HTTP or HTTPS URL to fetch.
        metadata: Optional extra metadata merged with the page's source metadata.
        project: Target project namespace.
    """
    body: dict = {"url": url}
    if metadata:
        body["metadata"] = metadata
    if project:
        body["project"] = project
    return await _post("/ingest_url", body)


@mcp.tool()
async def ingest_path(path: str) -> Any:
    """Ingest a local file or directory into the knowledge base (async).

    Returns immediately with a job_id. Poll get_job_status(job_id) until
    status is 'completed' or 'failed'. Path must be within RAG_INGEST_BASE.

    Args:
        path: Absolute or relative path to a file or directory.
    """
    return await _post("/ingest", {"path": path})


@mcp.tool()
async def get_job_status(job_id: str) -> Any:
    """Poll the status of an async ingest job started by ingest_path.

    Returns a dict with: job_id, status (processing|completed|failed),
    started_at, completed_at, path, error.

    Args:
        job_id: The job_id returned by ingest_path.
    """
    return await _get(f"/ingest/status/{job_id}")


@mcp.tool()
async def search_knowledge(query: str, top_k: int = 5, filters: dict | None = None) -> Any:
    """Retrieve raw document chunks from the knowledge base.

    Best for multi-step reasoning where you want to inspect individual chunks
    before synthesising an answer. Use query_knowledge for direct answers.

    Args:
        query: The search query string.
        top_k: Number of chunks to return (default 5).
        filters: Optional metadata filters, e.g. {"source": "https://..."}.
    """
    body: dict = {"query": query, "top_k": top_k}
    if filters:
        body["filters"] = filters
    return await _post("/search", body)


@mcp.tool()
async def query_knowledge(query: str, top_k: int | None = None, filters: dict | None = None) -> Any:
    """Ask a question and get a synthesised answer from the knowledge base.

    Performs retrieval + generation in one call. Use search_knowledge instead
    if you need to inspect raw chunks before answering.

    Args:
        query: The question to ask.
        top_k: Number of chunks to retrieve for context (overrides global setting).
        filters: Optional metadata filters for retrieval.
    """
    body: dict = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    if filters:
        body["filters"] = filters
    return await _post("/query", body)


@mcp.tool()
async def list_knowledge() -> Any:
    """List all indexed sources in the active project with chunk counts.

    Call this before a large ingest to check what's already indexed and avoid
    re-ingesting duplicate content.
    """
    return await _get("/collection")


@mcp.tool()
async def switch_project(project_name: str) -> Any:
    """Switch the knowledge base to a different project namespace.

    WARNING: This mutates global server state. Do not call from concurrent
    request handlers. Prefer passing 'project' directly to ingest tools instead.

    Args:
        project_name: The project name to activate, e.g. "react-docs".
    """
    return await _post("/project/switch", {"project_name": project_name})


@mcp.tool()
async def delete_documents(doc_ids: list[str]) -> Any:
    """Remove documents from the knowledge base by their IDs.

    Deletes from both the vector store and the BM25 index.

    Args:
        doc_ids: List of document IDs to delete.
    """
    return await _post("/delete", {"doc_ids": doc_ids})


@mcp.tool()
async def list_projects() -> Any:
    """List all knowledge base projects.

    Returns on-disk projects (with metadata) plus any project seen only in the
    current server session.  Call this to discover available namespaces before
    switching or querying a project.
    """
    return await _get("/projects")


@mcp.tool()
async def get_stale_docs(days: int = 7) -> Any:
    """Return documents that have not been re-ingested within *days* calendar days.

    Use this to identify outdated knowledge that should be refreshed.  Only
    documents ingested during the current server process lifetime are tracked —
    restart tracking begins fresh after each server restart.

    Args:
        days: Flag documents not re-ingested within this many days (default 7).
    """
    return await _get(f"/collection/stale?days={days}")


@mcp.tool()
async def create_project(name: str, description: str = "") -> Any:
    """Create a new knowledge base project namespace.

    Args:
        name: Name of the project to create.
        description: Optional description of the project contents.
    """
    return await _post("/project/new", {"name": name, "description": description})


@mcp.tool()
async def delete_project(name: str) -> Any:
    """Delete a knowledge base project and all its data.

    DANGER: This action is irreversible. It deletes all vectors and local files
    associated with the project.

    Args:
        name: Name of the project to delete.
    """
    return await _post(f"/project/delete/{name}", {})


@mcp.tool()
async def clear_knowledge() -> Any:
    """Wipe all data from the active project's vector store and index.

    Use this to reset a project without deleting the namespace itself.
    """
    return await _post("/clear", {})


@mcp.tool()
async def get_current_settings() -> Any:
    """Return the active Axon RAG and model configuration.

    Call this to check current top_k, threshold, and strategy settings.
    """
    return await _get("/config")


@mcp.tool()
async def update_settings(
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    hybrid_search: bool | None = None,
    rerank: bool | None = None,
    hyde: bool | None = None,
    multi_query: bool | None = None,
    step_back: bool | None = None,
    query_decompose: bool | None = None,
    compress_context: bool | None = None,
    graph_rag: bool | None = None,
    raptor: bool | None = None,
) -> Any:
    """Update global Axon RAG and retrieval settings for the current session.

    Args:
        top_k: Number of chunks to retrieve (1-50).
        similarity_threshold: Minimum match score (0.0-1.0).
        hybrid_search: Toggle hybrid BM25 + Vector search.
        rerank: Toggle cross-encoder reranking.
        hyde: Toggle Hypothetical Document Embeddings.
        multi_query: Toggle multi-query retrieval (3 rephrased queries merged).
        step_back: Toggle step-back prompting (abstract query before retrieval).
        query_decompose: Toggle query decomposition into atomic sub-questions.
        compress_context: Toggle LLM context compression before generation.
        raptor: Toggle RAPTOR hierarchical summaries.
        graph_rag: Toggle GraphRAG entity expansion.
    """
    body = {k: v for k, v in locals().items() if v is not None and k != "body"}
    return await _post("/config/update", body)


@mcp.tool()
async def list_sessions() -> Any:
    """List all saved chat sessions for the active project."""
    return await _get("/sessions")


@mcp.tool()
async def get_session(session_id: str) -> Any:
    """Retrieve a specific chat session by its ID.

    Args:
        session_id: The ID of the session to load.
    """
    return await _get(f"/session/{session_id}")


@mcp.tool()
async def share_project(
    project: str,
    grantee: str,
) -> Any:
    """Generate a share key allowing another user to access one of your projects.

    Requires AxonStore mode to be active. The returned share_string should be
    transmitted to the grantee out-of-band (e.g. Slack, email). The grantee
    then calls redeem_share to mount the project in their ShareMount/.
    All shares are read-only; write access is not supported.

    Args:
        project: Name of the project to share (must exist).
        grantee: OS username of the recipient.
    """
    return await _post(
        "/share/generate",
        {"project": project, "grantee": grantee, "write_access": False},
    )


@mcp.tool()
async def redeem_share(share_string: str) -> Any:
    """Redeem a share string, mounting the owner's project in your ShareMount/.

    Requires AxonStore mode to be active. After redemption, the shared project
    appears as ShareMount/{owner}_{project} and can be queried normally.

    Args:
        share_string: The base64 share string generated by share_project() on the owner's machine.
    """
    return await _post("/share/redeem", {"share_string": share_string})


@mcp.tool()
async def list_shares() -> Any:
    """List all active shares for the current AxonStore user.

    Returns 'sharing' (projects this user has shared with others, with revocation
    status) and 'shared' (projects others have shared with this user, with mount
    names). Use to audit access or troubleshoot missing shared projects.

    Requires AxonStore mode to be active. Call init_store() first if not yet
    initialised.
    """
    return await _get("/share/list")


@mcp.tool()
async def init_store(base_path: str) -> Any:
    """Initialise AxonStore multi-user mode at the given base directory.

    Must be called once before any share-related tools (list_shares,
    share_project, redeem_share, revoke_share) will work. Safe to call
    repeatedly — subsequent calls update the base path and reinitialise
    the brain.

    Args:
        base_path: Absolute path to the directory where the AxonStore/
                   folder will be created (e.g. '/data' creates
                   '/data/AxonStore/<username>/').
    """
    return await _post("/store/init", {"base_path": base_path})


@mcp.tool()
async def get_active_leases() -> Any:
    """Return active write-lease counts for all projects currently tracked by the server.

    Operator tool — shows which projects have in-flight write operations,
    whether they are draining, and their epoch counter.  Use this to check
    whether it is safe to put a project into 'readonly' or 'offline' maintenance
    state (wait for active_leases to reach 0 first).
    """
    return await _get("/registry/leases")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the axon-mcp console script."""
    mcp.run()


if __name__ == "__main__":
    main()
