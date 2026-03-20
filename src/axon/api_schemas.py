"""Pydantic request/response models and pure utility functions for the Axon API."""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

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

_VALID_PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,49}(?:/[a-z0-9][a-z0-9_-]{0,49}){0,4}$")


def _validate_ingest_path(path: str) -> str:
    """Validate that path is within the allowed base directory and not a blocked system path."""
    allowed_base = pathlib.Path(os.getenv("RAG_INGEST_BASE", ".")).resolve()
    abs_path = pathlib.Path(path).resolve()

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
            detail=(
                f"Path '{path}' is outside the allowed ingest directory. "
                "Set RAG_INGEST_BASE to permit additional paths."
            ),
        )
    return str(abs_path)


def _compute_content_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the normalised text content."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to ask the brain")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters for retrieval")
    stream: bool = Field(
        False, description="Whether to stream the response (use POST /query/stream instead)"
    )
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
    include_diagnostics: bool = Field(
        False,
        description="When True, include retrieval diagnostics in response",
    )
    dry_run: bool = Field(
        False,
        description="Skip LLM; return ranked chunks + diagnostics without calling generation model",
    )


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
        description=(
            "Path to a file or directory to ingest. "
            "Must be within RAG_INGEST_BASE (default: current working directory)."
        ),
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
        None,
        description="Project name to switch to, or 'default' for the global knowledge base",
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


class ShareRedeemRequest(BaseModel):
    share_string: str = Field(..., description="The base64 share string from the owner.")


class ShareRevokeRequest(BaseModel):
    key_id: str = Field(..., description="The key_id to revoke (e.g. 'sk_a1b2c3d4').")


class CopilotMessage(BaseModel):
    role: str
    content: str


class CopilotAgentRequest(BaseModel):
    """Payload sent by GitHub Copilot to the agent endpoint."""

    messages: list[CopilotMessage]
    copilot_references: list[dict] = Field(default_factory=list)
    agent_request_id: str | None = None


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


class CopilotTaskResult(BaseModel):
    result: str | None = None
    error: str | None = None


class MaintenanceStateRequest(BaseModel):
    name: str = Field(..., description="Project name (slash-separated for sub-projects).")
    state: str = Field(
        ...,
        description="Maintenance state: 'normal', 'draining', 'readonly', or 'offline'.",
    )
