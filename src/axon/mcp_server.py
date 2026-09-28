"""
src/axon/mcp_server.py

MCP stdio server for Axon — exposes a deliberately small set of the Axon REST
API as MCP tools so Copilot, Claude or any other agent can call them from
agent mode.

The tool set (18 tools, 0.5.0) covers what an agent needs to *use* a knowledge
base: ask, search, ingest, inspect, delete individual documents, pick or create
a project, read and tune config, read and write the graph, and share projects.
Destructive, credential and administrative operations (clear, delete project,
store / sealed-store / pack / unpack, sessions, graph admin, hard revoke with
key rotation) are human-only — REST, CLI, REPL — see
``axon.surface_contract`` for the human route of each.

``project`` parameters are *assertions*: the server answers 409 when the brain
is serving a different project. Tools never switch projects implicitly — call
switch_project first.

Tool names here are deliberately shorter than the OpenAI-format names in
tools.py; do not conflate the two sets.

Environment variables
---------------------
RAG_API_BASE  : Base URL of the running Axon API  (default: http://localhost:8420)
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


API_BASE: str = os.getenv("RAG_API_BASE", "http://localhost:8420").rstrip("/")


API_KEY: str | None = os.getenv("RAG_API_KEY") or None


mcp = FastMCP("axon")


def _headers() -> dict[str, str]:
    """Return request headers, including X-API-Key and surface attribution."""
    h: dict[str, str] = {"Content-Type": "application/json", "X-Axon-Surface": "mcp"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


class AxonAPIError(RuntimeError):
    """An Axon REST call failed. The message carries the HTTP status and the
    server's ``detail`` (e.g. the unknown config keys, the active project on a
    409, "project is required for sealed shares") so the agent can act on it —
    ``httpx``'s own message only names the status and URL."""

    def __init__(self, status: int, detail: str, method: str, path: str):
        self.status = status
        self.detail = detail
        super().__init__(f"Axon API {method} {path} failed ({status}): {detail}")


def _raise_for_status(resp: httpx.Response, method: str, path: str) -> None:
    if resp.is_success:
        return
    detail: Any = None
    try:
        data = resp.json()
        detail = data.get("detail", data) if isinstance(data, dict) else data
    except Exception:
        detail = None
    if detail is None or detail == "":
        detail = resp.text or resp.reason_phrase
    if not isinstance(detail, str):
        import json

        detail = json.dumps(detail)
    raise AxonAPIError(resp.status_code, detail, method, path)


async def _get(path: str, params: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(f"{API_BASE}{path}", headers=_headers(), params=params)
        _raise_for_status(resp, "GET", path)
        return resp.json()


async def _post(path: str, body: dict) -> Any:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{API_BASE}{path}", json=body, headers=_headers())
        _raise_for_status(resp, "POST", path)
        return resp.json()


def _with_project(body: dict, project: str | None) -> dict:
    """Add the ``project`` assertion to *body* when the caller gave one."""
    if project:
        body["project"] = project
    return body


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


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
        project: Expected active project (an assertion, not a switch). Returns
            409 if it does not match the brain's active project — call
            switch_project first.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    body: dict = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    if filters:
        body["filters"] = filters
    return await _post("/query", _with_project(body, project))


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
        project: Expected active project (an assertion, not a switch). Returns
            409 if it does not match the brain's active project — call
            switch_project first.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    body: dict = {"query": query, "top_k": top_k}
    if filters:
        body["filters"] = filters
    return await _post("/search", _with_project(body, project))


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@mcp.tool()
async def ingest_knowledge(
    text: str | None = None,
    docs: list[dict] | None = None,
    url: str | None = None,
    path: str | None = None,
    refresh: bool = False,
    metadata: dict | None = None,
    doc_id: str | None = None,
    project: str | None = None,
) -> Any:
    """Add knowledge to the active project. Give exactly ONE source:
    - text: one text document (set metadata.source so it can be audited).
    - docs: many documents in one batched embedding call — a list of
      {"text": ..., "doc_id"?: ..., "metadata"?: {...}}. Prefer this over
      calling ingest_knowledge(text=...) in a loop.
    - url: an HTTP/HTTPS page; HTML is stripped. Private/internal addresses
      are blocked server-side.
    - path: a local file or directory (must be within RAG_INGEST_BASE and on
      the machine running axon-api). Asynchronous — returns a job_id; poll
      get_job_status(job_id) until 'completed' or 'failed'.
    - refresh=True: re-ingest previously indexed files whose content changed
      on disk. Asynchronous — returns a job_id.
    Duplicate content (same SHA-256) is skipped with status 'skipped'.
    Args:
        text: Text content to store.
        docs: Batch of documents (see above).
        url: URL to fetch and ingest.
        path: File or directory path to ingest.
        refresh: Re-ingest changed files instead of adding new content.
        metadata: Metadata for text/url ingest, e.g. {"source": "...", "topic": "react"}.
        doc_id: Optional stable ID for text ingest (delete_documents accepts it).
        project: Expected active project (an assertion, not a switch). Returns
            409 if it does not match the active project — call switch_project
            first.
    """
    sources = [
        name
        for name, given in (
            ("text", text is not None),
            ("docs", docs is not None),
            ("url", url is not None),
            ("path", path is not None),
            ("refresh", bool(refresh)),
        )
        if given
    ]
    if len(sources) != 1:
        raise ValueError(
            "ingest_knowledge needs exactly one of text, docs, url, path or refresh=True "
            f"(got {', '.join(sources) if sources else 'none'})"
        )
    source = sources[0]
    if source == "text":
        body: dict = {"text": text}
        if metadata:
            body["metadata"] = metadata
        if doc_id:
            body["doc_id"] = doc_id
        return await _post("/add_text", _with_project(body, project))
    if source == "docs":
        return await _post("/add_texts", _with_project({"docs": docs}, project))
    if source == "url":
        body = {"url": url}
        if metadata:
            body["metadata"] = metadata
        return await _post("/ingest_url", _with_project(body, project))
    if source == "path":
        return await _post("/ingest", _with_project({"path": path}, project))
    return await _post("/ingest/refresh", _with_project({}, project))


@mcp.tool()
async def get_job_status(job_id: str) -> Any:
    """Poll an async ingest job started by ingest_knowledge(path=...) or
    ingest_knowledge(refresh=True).
    Returns a dict with: job_id, status (processing|completed|failed),
    started_at, completed_at, error and job-specific counters.
    Args:
        job_id: The job_id returned by ingest_knowledge.
    """
    return await _get(f"/ingest/status/{job_id}")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_knowledge() -> Any:
    """List all indexed sources in the active project with chunk counts.
    Call this before a large ingest to check what's already indexed and avoid
    re-ingesting duplicate content.
    """
    return await _get("/collection")


@mcp.tool()
async def delete_documents(doc_ids: list[str]) -> Any:
    """Remove documents or chunks from the active project by their IDs.
    Deletes from the vector store, the BM25 index and the graph, and clears the
    dedup records, so the same text can be ingested again afterwards.
    (Wiping a whole project is human-only: REPL /clear, `axon --clear --yes`.)
    Args:
        doc_ids: Chunk IDs, or document IDs (the doc_id given to
            ingest_knowledge); a document ID deletes all its chunks.
    """
    return await _post("/delete", {"doc_ids": doc_ids})


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_projects() -> Any:
    """List all knowledge base projects.
    Returns on-disk projects (with metadata) plus any project seen only in the
    current server session.  Call this to discover available namespaces before
    switching or querying a project.
    """
    return await _get("/projects")


@mcp.tool()
async def switch_project(project_name: str) -> Any:
    """Switch the knowledge base to a different project.
    WARNING: This mutates global server state — every later call (from any
    client) runs against the new project.
    Args:
        project_name: The project name to activate, e.g. "react-docs" or
            "mounts/alice_research" for a redeemed share.
    """
    return await _post("/project/switch", {"project_name": project_name})


@mcp.tool()
async def create_project(name: str, description: str = "", graph_backend: str | None = None) -> Any:
    """Create a new knowledge base project.
    Args:
        name: Name of the project to create.
        description: Optional description of the project contents.
        graph_backend: Graph backend for this project — "graphrag" (default),
            "dynamic_graph", or "none". Immutable once set. Not settable here:
            "federated" (config.yaml-only override).
    """
    body: dict = {"name": name, "description": description}
    if graph_backend is not None:
        body["graph_backend"] = graph_backend
    return await _post("/project/new", body)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_config(validate: bool = False) -> Any:
    """Return the active Axon configuration.

    Secrets (API keys, tokens) are masked as ``***`` by the server — this tool
    can never read a credential back out. Use it to discover the exact field
    names accepted by set_config().

    Args:
        validate: Also check the on-disk config for errors, unknown keys and
            risky combinations (e.g. GraphRAG + a slow local LLM). Returns
            {"config": ..., "validation": ...} instead of the bare config.
    """
    config = await _get("/config")
    if not validate:
        return config
    return {"config": config, "validation": await _get("/config/validate")}


@mcp.tool()
async def set_config(settings: dict, persist: bool = False) -> Any:
    """Set one or more Axon configuration fields in a single call.

    ``settings`` maps keys to values. A key is a dot-notation alias
    (``chunk.strategy``, ``llm.model``, ``rag.top_k``) or any ``AxonConfig``
    field name (``graph_rag_depth``, ``chunk_size``, ``hybrid_search``). Call
    get_config() to see every field. All keys are checked first: one unknown
    key rejects the whole batch (400) and nothing is applied.

    Changing ``llm_provider`` / ``llm_model`` / ``embedding_*`` / ``rerank``
    reinitialises the affected component once, so the next query uses it
    immediately. If that reinitialisation fails (e.g. a provider whose extra is
    not installed) every key in the batch is rolled back, nothing is saved,
    and the call fails with the reason. Switching the embedding model
    invalidates existing vectors — re-ingest after.

    Args:
        settings: {key: value} map, e.g. {"top_k": 8, "rerank": true}.
        persist: Also write the changes to config.yaml so they survive a
            restart. Default False (running server only).
    """
    return await _post("/config/set", {"settings": settings, "persist": persist})


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@mcp.tool()
async def graph_retrieve(
    query: str,
    top_k: int | None = None,
    point_in_time: str | None = None,
    federation_weights: dict[str, float] | None = None,
    project: str | None = None,
) -> Any:
    """Run the active graph backend's retrieve() and return graph contexts only.
    Surfaces point-in-time historical queries and per-query federation weight
    overrides without going through the full /query LLM pipeline.
    Args:
        query: The query string.
        top_k: Maximum graph contexts to return (default 10, max 200).
        point_in_time: ISO-8601 timestamp; return facts valid at that instant.
            Honoured only by bi-temporal backends (``dynamic_graph``); ignored
            elsewhere.
        federation_weights: Per-query RRF weights for the federated backend.
            Keys: ``graphrag``, ``dynamic_graph``. Ignored by other backends.
        project: Expected active project (an assertion, not a switch). Returns
            409 if it does not match the active project.
    """
    body: dict[str, Any] = {"query": query}
    if top_k is not None:
        body["top_k"] = int(top_k)
    if point_in_time is not None:
        body["point_in_time"] = point_in_time
    if federation_weights is not None:
        body["federation_weights"] = federation_weights
    return await _post("/graph/retrieve", _with_project(body, project))


@mcp.tool()
async def update_fact(
    subject: str,
    relation: str,
    object: str,  # noqa: A002 — mirrors the REST field name
    description: str = "",
    confidence: float = 1.0,
    replace: bool | None = None,
    project: str | None = None,
) -> Any:
    """Assert or correct one fact (subject, relation, object) in the active
    project's knowledge graph. Stored with bi-temporal history: a replaced fact
    is superseded, not deleted, so graph_retrieve(point_in_time=...) still sees it.
    Only ``dynamic_graph`` and ``federated`` projects store facts; ``graphrag``
    and ``none`` answer status ``not_applicable``.
    Args:
        subject: Subject entity, e.g. "Alice".
        relation: Relation, e.g. "WORKS_FOR" or "works for" (normalised to
            upper snake case).
        object: Object entity, e.g. "Acme Corp".
        description: Optional free-text note stored with the fact.
        confidence: 0.0-1.0 (default 1.0).
        replace: True = make this the only current fact for (subject,
            relation), superseding the others; False = add alongside them;
            None (default) = replace for exclusive relations (IS_CEO_OF,
            MARRIED_TO, ...), add otherwise.
        project: Expected active project (an assertion, not a switch).
    Returns ``{status: created|superseded|unchanged|conflicted|not_applicable,
    fact_id, superseded_ids, conflicted_ids, backend_id, detail}``.
    """
    body: dict[str, Any] = {"subject": subject, "relation": relation, "object": object}
    if description:
        body["description"] = description
    if confidence != 1.0:
        body["confidence"] = confidence
    if replace is not None:
        body["replace"] = replace
    return await _post("/graph/facts", _with_project(body, project))


# ---------------------------------------------------------------------------
# Sharing
# ---------------------------------------------------------------------------


@mcp.tool()
async def share_project(
    project: str,
    grantee: str,
    ttl_days: int | None = None,
) -> Any:
    """Generate a share key allowing another user to read one of your projects.
    The returned share_string should be transmitted to the grantee out-of-band
    (e.g. Slack, email). The grantee then calls redeem_share to mount the project.
    All shares are read-only; write access is not supported.
    Sealed projects (encrypted at rest) get a SEALED envelope automatically;
    the sealed store must be unlocked by the user first (409 otherwise).
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
    and can be queried after switch_project.
    Sealed shares are detected automatically; their key is stored in the OS
    keyring.
    Args:
        share_string: The share string generated by share_project() on the owner's machine.
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
async def revoke_share(key_id: str, project: str | None = None) -> Any:
    """Revoke a previously generated share key (soft revoke).
    - Plaintext shares (key_id ``sk_...``): the grantee loses access on their
      next project-list or switch.
    - Sealed shares (key_id ``ssk_...``): deletes the share's key wrap so it
      can no longer be redeemed; ``project`` is required. A grantee who
      already redeemed keeps the key they cached. Hard revoke — rotating the
      project key and re-encrypting it — is human-only: `axon --share-rotate`,
      REPL ``/share revoke <ssk_id> --project <name> --rotate``.
    Args:
        key_id: The key ID of the share to revoke (from list_shares output).
        project: Project name — required for sealed (``ssk_``) shares,
            ignored for plaintext ones.
    """
    body: dict[str, Any] = {"key_id": key_id}
    if project is not None:
        body["project"] = project
    return await _post("/share/revoke", body)


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the axon-mcp console script."""
    from axon.logging_setup import configure_logging

    configure_logging()
    mcp.run()


if __name__ == "__main__":
    main()
