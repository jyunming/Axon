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


def _get(path: str) -> Any:
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(f"{API_BASE}{path}", headers=_headers())
        resp.raise_for_status()
        return resp.json()


def _post(path: str, body: dict) -> Any:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{API_BASE}{path}", json=body, headers=_headers())
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def ingest_text(text: str, metadata: dict | None = None, project: str | None = None) -> dict:
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
    return _post("/add_text", body)


@mcp.tool()
def ingest_texts(docs: list[dict], project: str | None = None) -> list[dict]:
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
    return _post("/add_texts", body)


@mcp.tool()
def ingest_url(url: str, metadata: dict | None = None, project: str | None = None) -> dict:
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
    return _post("/ingest_url", body)


@mcp.tool()
def ingest_path(path: str) -> dict:
    """Ingest a local file or directory into the knowledge base (async).

    Returns immediately with a job_id. Poll get_job_status(job_id) until
    status is 'completed' or 'failed'. Path must be within RAG_INGEST_BASE.

    Args:
        path: Absolute or relative path to a file or directory.
    """
    return _post("/ingest", {"path": path})


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Poll the status of an async ingest job started by ingest_path.

    Returns a dict with: job_id, status (processing|completed|failed),
    started_at, completed_at, path, error.

    Args:
        job_id: The job_id returned by ingest_path.
    """
    return _get(f"/ingest/status/{job_id}")


@mcp.tool()
def search_knowledge(query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
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
    return _post("/search", body)


@mcp.tool()
def query_knowledge(query: str, filters: dict | None = None) -> dict:
    """Ask a question and get a synthesised answer from the knowledge base.

    Performs retrieval + generation in one call. Use search_knowledge instead
    if you need to inspect raw chunks before answering.

    Args:
        query: The question to ask.
        filters: Optional metadata filters for retrieval.
    """
    body: dict = {"query": query}
    if filters:
        body["filters"] = filters
    return _post("/query", body)


@mcp.tool()
def list_knowledge() -> dict:
    """List all indexed sources in the active project with chunk counts.

    Call this before a large ingest to check what's already indexed and avoid
    re-ingesting duplicate content.
    """
    return _get("/collection")


@mcp.tool()
def switch_project(project_name: str) -> dict:
    """Switch the knowledge base to a different project namespace.

    WARNING: This mutates global server state. Do not call from concurrent
    request handlers. Prefer passing 'project' directly to ingest tools instead.

    Args:
        project_name: The project name to activate, e.g. "react-docs".
    """
    return _post("/project/switch", {"project_name": project_name})


@mcp.tool()
def delete_documents(doc_ids: list[str]) -> dict:
    """Remove documents from the knowledge base by their IDs.

    Deletes from both the vector store and the BM25 index.

    Args:
        doc_ids: List of document IDs to delete.
    """
    return _post("/delete", {"doc_ids": doc_ids})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the axon-mcp console script."""
    mcp.run()


if __name__ == "__main__":
    main()
