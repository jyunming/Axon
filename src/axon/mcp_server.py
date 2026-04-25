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
    """Return request headers, including X-API-Key and surface attribution."""

    h: dict[str, str] = {"Content-Type": "application/json", "X-Axon-Surface": "mcp"}

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

        project: Target project. Omit to use the active project.

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

        project: Target project applied to all docs.

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

        project: Target project.

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
async def refresh_ingest(project: str | None = None) -> Any:
    """Re-ingest all sources that have changed on disk since they were last indexed.

    Compares SHA-256 hashes of previously indexed files against their current

    on-disk content. Changed files are re-chunked and re-embedded; unchanged

    files are skipped.  Returns a job_id for async polling via get_job_status.

    Args:

        project: Target project. Omit to use the active project.

    """

    if project:
        await _post("/project/switch", {"project_name": project})

    return await _post("/ingest/refresh", {})


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
async def search_knowledge(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    project: str | None = None,
) -> Any:
    """Retrieve raw document chunks from the knowledge base.

    Best for multi-step reasoning where you want to inspect individual chunks

    before synthesising an answer. Use query_knowledge for direct answers.

    Args:

        query: The search query string.

        top_k: Number of chunks to return (default 5).

        filters: Optional metadata filters, e.g. {"source": "https://..."}.

        project: Expected active project. Returns 409 if it does not match

            the brain's current active project. Use switch_project to change

            the active project before calling this tool.

    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    body: dict = {"query": query, "top_k": top_k}

    if filters:
        body["filters"] = filters

    if project:
        body["project"] = project

    return await _post("/search", body)


@mcp.tool()
async def query_knowledge(
    query: str,
    top_k: int | None = None,
    filters: dict | None = None,
    project: str | None = None,
) -> Any:
    """Ask a question and get a synthesised answer from the knowledge base.

    Performs retrieval + generation in one call. Use search_knowledge instead

    if you need to inspect raw chunks before answering.

    Args:

        query: The question to ask.

        top_k: Number of chunks to retrieve for context (overrides global setting).

        filters: Optional metadata filters for retrieval.

        project: Expected active project. Returns 409 if it does not match

            the brain's current active project. Use switch_project to change

            the active project before calling this tool.

    """

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    body: dict = {"query": query}

    if top_k is not None:
        body["top_k"] = top_k

    if filters:
        body["filters"] = filters

    if project:
        body["project"] = project

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
    """Switch the knowledge base to a different project.

    WARNING: This mutates global server state. Do not call from concurrent

    request handlers. Prefer passing 'project' directly to ingest tools instead.

    Args:

        project_name: The project name to activate, e.g. "react-docs".

    """

    return await _post("/project/switch", {"project_name": project_name})


@mcp.tool()
async def refresh_mount() -> Any:
    """Re-read the owner's version marker for an active mounted share and
    reopen project handles if the owner has re-ingested.

    No-op when the active project is not a mount (mounts/<name>). On a
    cloud-sync mid-replication race the API may respond with HTTP 503
    + ``X-Axon-Mount-Sync-Pending: true``; retry after a few seconds.

    Returns:
        ``{"status": "success", "refreshed": bool, "seq": int|null}``
        when refresh completed, or ``{"status": "sync_pending", ...}``
        when the owner advanced but index files are still in flight.
    """

    return await _post("/mount/refresh", {})


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
    """Create a new knowledge base project.

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
    truth_grounding: bool | None = None,
    discussion_fallback: bool | None = None,
    sentence_window: bool | None = None,
    sentence_window_size: int | None = None,
    crag_lite: bool | None = None,
    code_graph: bool | None = None,
    graph_rag_mode: str | None = None,
    cite: bool | None = None,
    persist: bool = False,
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

        truth_grounding: Toggle truth-grounding enforcement on retrieved chunks.

        discussion_fallback: Allow general-knowledge fallback when no chunks found.

        sentence_window: Toggle sentence-window retrieval (expands chunks with context sentences).

        sentence_window_size: Number of surrounding sentences per side (1-10, default 2).

        crag_lite: Toggle CRAG-lite corrective retrieval on low-confidence chunks.

        code_graph: Toggle code-graph retrieval for code-related queries.

        graph_rag_mode: GraphRAG query mode — "local", "global", or "hybrid".

        cite: Include inline source citations in generated answers.

        persist: Save these settings to config.yaml so they survive restarts.
            Defaults to False (session-only changes).

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
    ttl_days: int | None = None,
) -> Any:
    """Generate a share key allowing another user to access one of your projects.

    The returned share_string should be transmitted to the grantee out-of-band

    (e.g. Slack, email). The grantee then calls redeem_share to mount the project.

    All shares are read-only; write access is not supported.

    Args:

        project: Name of the project to share (must exist).

        grantee: OS username of the recipient.

        ttl_days: Optional time-to-live in days. When set, the share
            automatically expires after this many days; owners can renew
            with extend_share. None (default) means no expiry.

    """

    body: dict[str, Any] = {"project": project, "grantee": grantee}
    if ttl_days is not None:
        body["ttl_days"] = ttl_days
    return await _post("/share/generate", body)


@mcp.tool()
async def redeem_share(share_string: str) -> Any:
    """Redeem a share string, creating a mount descriptor in your mounts/ directory.

    After redemption, the shared project appears as mounts/{owner}_{project}

    and can be queried normally.

    Args:

        share_string: The base64 share string generated by share_project() on the owner's machine.

    """

    return await _post("/share/redeem", {"share_string": share_string})


@mcp.tool()
async def list_shares() -> Any:
    """List all active shares for the current user.

    Returns 'sharing' (projects this user has shared with others, with revocation

    status) and 'shared' (projects others have shared with this user, with mount

    names). Use to audit access or troubleshoot missing shared projects.

    """

    return await _get("/share/list")


@mcp.tool()
async def revoke_share(key_id: str) -> Any:
    """Revoke a previously generated share key, cutting off the grantee's access.

    The grantee's mount becomes broken immediately; they will receive a 404 on

    their next project-list or switch attempt.  Use list_shares to find the

    key_id of the share you want to revoke.

    Args:

        key_id: The key ID of the share to revoke (from list_shares output).

    """

    return await _post("/share/revoke", {"key_id": key_id})


@mcp.tool()
async def extend_share(key_id: str, ttl_days: int | None = None) -> Any:
    """Renew a share key's expiry, or clear it (``ttl_days=null``).

    Pairs with ``share_project(ttl_days=...)`` to give owners a hard
    cutoff for forgotten shares while still letting them keep an
    in-use share alive on demand.

    Args:

        key_id: The key ID of the share to extend (from list_shares).

        ttl_days: New time-to-live in days, measured from now.
            None clears the expiry entirely.

    """

    return await _post("/share/extend", {"key_id": key_id, "ttl_days": ttl_days})


@mcp.tool()
async def get_store_status() -> Any:
    """Check whether the AxonStore has been initialised.

    Returns store metadata (path, version, creation date) when the store

    exists, or ``{"initialized": false}`` on a fresh install.  Clients should

    call this on startup before any other tool to decide whether to prompt the

    user to run ``init_store``.

    """

    return await _get("/store/status")


@mcp.tool()
async def init_store(base_path: str, persist: bool = False) -> Any:
    """Initialise AxonStore multi-user mode at the given base directory.

    Must be called once before any share-related tools (list_shares,

    share_project, redeem_share, revoke_share) will work. Safe to call

    repeatedly — subsequent calls update the base path and reinitialise

    the brain.

    Args:

        base_path: Absolute path to the directory where the AxonStore/

                   folder will be created (e.g. '/data' creates

                   '/data/AxonStore/<username>/').

        persist: Write the new store path to config.yaml so it survives

                 server restarts. Defaults to False so that test or

                 temporary calls do not permanently alter the user config.

                 Pass True only when you intend to switch to AxonStore

                 mode permanently.

    """

    return await _post("/store/init", {"base_path": base_path, "persist": persist})


# ---------------------------------------------------------------------------
# Sealed-store tools (Phase 2 of #SEALED).
# Mirror the /security/* REST endpoints so MCP clients can manage the
# encryption-at-rest store without poking the HTTP API directly.
# ---------------------------------------------------------------------------


@mcp.tool()
async def security_status() -> Any:
    """Return current sealed-store status (initialized + unlocked flags).

    Use this on startup to decide whether to prompt for ``security_bootstrap``
    (first-time setup) or ``security_unlock`` (existing user). Returns
    ``initialized: false`` on a fresh install, ``initialized: true,
    unlocked: false`` after bootstrap until the next unlock.
    """

    return await _get("/security/status")


@mcp.tool()
async def security_bootstrap(passphrase: str) -> Any:
    """Initialise the sealed-store with a passphrase (one-time setup).

    Generates a fresh master key, wraps it under a passphrase-derived
    KEK, and stores the wrapped record in the OS keyring. After this
    succeeds the store is automatically unlocked for the rest of this
    process; the passphrase is required for every subsequent unlock.

    Args:
        passphrase: The user's chosen passphrase. Cannot be empty.
            There is NO recovery — losing this passphrase means losing
            access to every project sealed under this master.
    """

    return await _post("/security/bootstrap", {"passphrase": passphrase})


@mcp.tool()
async def security_unlock(passphrase: str) -> Any:
    """Unlock the sealed-store so sealed projects can be queried.

    Required after every process restart before ``project_seal`` or any
    sealed-project switch will work. Rate-limited: 5 wrong attempts
    inside 5 minutes triggers a 429 lockout per client IP.

    Args:
        passphrase: The passphrase supplied at ``security_bootstrap``
            time (or the current passphrase after a rotation).
    """

    return await _post("/security/unlock", {"passphrase": passphrase})


@mcp.tool()
async def security_lock() -> Any:
    """Clear the in-process master key cache.

    Subsequent sealed-project queries will fail with a "store is locked"
    error until ``security_unlock`` is called again. Use before walking
    away from the machine; the orphan-cleanup hook will wipe any
    plaintext mount caches on next process boot.
    """

    return await _post("/security/lock", {})


@mcp.tool()
async def security_change_passphrase(old_passphrase: str, new_passphrase: str) -> Any:
    """Re-wrap the master key under a new passphrase.

    Project DEKs are not touched (they're wrapped under the master, not
    the passphrase), so this is O(1) regardless of how many sealed
    projects you have. The new passphrase is required for every future
    ``security_unlock`` call.

    Args:
        old_passphrase: The current passphrase. Required to unwrap the
            existing master before re-wrapping.
        new_passphrase: The new passphrase. Cannot be empty.
    """

    return await _post(
        "/security/change-passphrase",
        {"old_passphrase": old_passphrase, "new_passphrase": new_passphrase},
    )


@mcp.tool()
async def seal_project(project_name: str, migration_mode: str = "in_place") -> Any:
    """Encrypt every content file in a project in place (one-shot).

    Walks the project directory and rewrites ``meta.json`` plus every
    file under ``bm25_index/`` and ``vector_store_data/`` as AXSL-sealed
    AES-256-GCM ciphertext. Each file is replaced atomically (tempfile
    + os.replace), so a crash mid-seal leaves the original or the new
    sealed version on disk but never a partial write.

    Idempotent — re-sealing an already-sealed project is a no-op
    (returns ``status="already_sealed"``).

    Requires the sealed-store to be unlocked (see ``security_unlock``).
    The active project switches to "default" during the operation and
    switches back when done.

    Args:
        project_name: Name of the open project to seal. Must exist.
        migration_mode: Reserved for future variants; only ``"in_place"``
            is implemented in v1.
    """

    return await _post(
        "/project/seal",
        {"project_name": project_name, "migration_mode": migration_mode},
    )


@mcp.tool()
async def graph_status() -> Any:
    """Return current GraphRAG knowledge-graph status.

    Reports entity count, edge count, community summary count, whether a

    community rebuild is in progress, and whether the graph is ready for

    graph-augmented retrieval.  Use before running graph_finalize() to check

    whether a rebuild is actually needed.

    """

    return await _get("/graph/status")


@mcp.tool()
async def graph_finalize() -> Any:
    """Trigger an explicit GraphRAG community detection rebuild.

    Rebuilds community summaries from the current entity graph.  Call this

    after a large ingest batch when you want graph-augmented answers to

    reflect the latest knowledge without waiting for the automatic rebuild.

    Returns the number of community summaries produced.

    """

    return await _post("/graph/finalize", {})


@mcp.tool()
async def graph_data() -> Any:
    """Return the full entity/relation knowledge-graph payload as JSON.

    Returns a dict with 'nodes' and 'links' arrays describing every entity

    and relation currently in the graph.  Useful for inspection, export, or

    building custom visualisations.  Returns empty arrays when no graph has

    been built yet.

    """

    return await _get("/graph/data")


@mcp.tool()
async def graph_backend_status() -> Any:
    """Return the active graph backend's status dict.

    Reports which backend is active (graphrag or dynamic), whether it is

    ready, and backend-specific health metrics (entity count, edge count,

    node counts, etc.).  Use this to distinguish between the GraphRAG

    community-graph backend and the dynamic SQLite-WAL graph backend.

    """

    return await _get("/graph/backend/status")


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
