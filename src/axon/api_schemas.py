"""Pydantic request/response models and pure utility functions for the Axon API."""


from __future__ import annotations

import hashlib
import os
import pathlib
import re
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field, SecretStr

# Default upper bounds applied to free-form text fields exposed by the REST API.
# These guard against accidental or malicious oversized payloads before any
# embedding / loader work is performed.  Bulk uploads should use /ingest/upload
# (which has its own byte cap) or the file-path-based /ingest endpoint.
MAX_QUERY_FIELD_CHARS = 8192  # characters — generous for any realistic question
MAX_TEXT_FIELD_CHARS = (
    10_000_000  # characters — ~10 MB of ASCII text; /ingest/upload enforces a separate byte cap
)
MAX_URL_FIELD_CHARS = 2048  # characters — RFC 7230 practical maximum

# ---------------------------------------------------------------------------


# Path security


# ---------------------------------------------------------------------------


_BLOCKED_PATH_PREFIXES: tuple[pathlib.Path, ...] = tuple(
    pathlib.Path(p).resolve()
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
    from axon.rust_bridge import get_rust_bridge

    bridge = get_rust_bridge()
    if bridge.can_sha256():
        result = bridge.compute_sha256(text)
        if result is not None:
            return result
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------


# Pydantic models


# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        max_length=MAX_QUERY_FIELD_CHARS,
        description="The question or prompt to ask the brain",
    )
    project: str | None = Field(
        None,
        description=(
            "Target project. Must match the brain's active project; use POST /project/switch "
            "to change the active project before querying. Returns 409 on mismatch."
        ),
    )
    filters: dict[str, Any] | None = Field(None, description="Metadata filters for retrieval")
    stream: bool = Field(
        False, description="Whether to stream the response (use POST /query/stream instead)"
    )
    top_k: int | None = Field(None, ge=1, description="Override number of chunks to retrieve")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override similarity threshold (0.0-1.0)"
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
        None, ge=0.0, le=2.0, description="Override LLM temperature for this request (0.0-2.0)"
    )
    timeout: float | None = Field(None, gt=0, description="Query timeout in seconds (default 120)")
    include_diagnostics: bool = Field(
        False,
        description="When True, include retrieval diagnostics in response",
    )
    include_citations: bool = Field(
        True,
        description=(
            "When True (default), include ``sources`` and ``citations`` arrays "
            "in the response — sources lists every retrieved chunk made "
            "available to the LLM (slim form, text truncated to 500 chars); "
            "citations is a list of structured spans extracted from the "
            "response, one per ``[N]`` / ``[Document N]`` marker. Set to "
            "False to skip the extra payload (e.g. high-throughput agents "
            "that only need the answer string)."
        ),
    )
    dry_run: bool = Field(
        False,
        description="Skip LLM; return ranked chunks + diagnostics without calling generation model",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string for semantic search")
    project: str | None = Field(
        None,
        description=(
            "Target project. Must match the brain's active project; use POST /project/switch "
            "to change the active project before searching. Returns 409 on mismatch."
        ),
    )
    top_k: int | None = Field(
        None, ge=1, description="Number of documents to return (must be at least 1)"
    )
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override similarity threshold for this request"
    )


class QueryVisualizeRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to visualize")
    top_k: int | None = Field(None, ge=1, description="Override number of chunks to retrieve")
    project: str | None = Field(None, description="Target project (must match active project)")
    raptor: bool | None = Field(None, description="Override RAPTOR retrieval for this request")
    graph_rag: bool | None = Field(None, description="Override GraphRAG retrieval for this request")


class SearchVisualizeRequest(BaseModel):
    query: str = Field(..., description="The search query to visualize")
    top_k: int | None = Field(None, ge=1, description="Override number of chunks to retrieve")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override similarity threshold"
    )
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")
    project: str | None = Field(None, description="Target project (must match active project)")


_VALID_FEDERATION_KEYS = frozenset({"graphrag", "dynamic_graph"})


class GraphRetrieveRequest(BaseModel):
    """Body for ``POST /graph/retrieve``.

    ``point_in_time`` enables historical queries against backends that store
    bi-temporal facts (currently only ``dynamic_graph``). On other backends
    it is ignored. ``federation_weights`` overrides the project-level RRF
    weights for a single retrieve and is consumed only by the federated
    backend.
    """

    query: str = Field(
        ...,
        max_length=MAX_QUERY_FIELD_CHARS,
        description="The query string to retrieve graph contexts for",
    )
    project: str | None = Field(None, description="Target project (must match active project)")
    top_k: int | None = Field(
        None, ge=1, le=200, description="Maximum graph contexts to return (default 10)"
    )
    point_in_time: str | None = Field(
        None,
        description=(
            "ISO-8601 timestamp; return facts valid at that instant. "
            "Only honoured by backends with bi-temporal storage."
        ),
    )
    federation_weights: dict[str, float] | None = Field(
        None,
        description=(
            "Per-query RRF weights for the federated backend. "
            "Keys: ``graphrag``, ``dynamic_graph``. Values must be >= 0. "
            "Ignored by other backends."
        ),
    )

    def __init__(self, **data):
        super().__init__(**data)
        fw = self.federation_weights
        if fw is not None:
            unknown = set(fw.keys()) - _VALID_FEDERATION_KEYS
            if unknown:
                raise ValueError(
                    f"federation_weights contains unknown key(s): {sorted(unknown)}. "
                    f"Allowed keys: {sorted(_VALID_FEDERATION_KEYS)}"
                )
            for k, v in fw.items():
                if v < 0:
                    raise ValueError(f"federation_weights['{k}'] must be >= 0 (got {v})")


class IngestRequest(BaseModel):
    path: str = Field(
        ...,
        description=(
            "Path to a file or directory to ingest. "
            "Must be within RAG_INGEST_BASE (default: current working directory)."
        ),
    )


class TextIngestRequest(BaseModel):
    text: str = Field(
        ...,
        max_length=MAX_TEXT_FIELD_CHARS,
        description="The content to ingest",
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Metadata for the document"
    )
    doc_id: str | None = Field(None, description="Optional unique ID for the document")
    project: str | None = Field(
        None,
        description="Optional project assertion. When provided, it must match the active project.",
    )


class BatchDocItem(BaseModel):
    """A single item within a batch ingest request."""

    text: str = Field(
        ...,
        max_length=MAX_TEXT_FIELD_CHARS,
        description="The content to ingest",
    )
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
        description="Optional project assertion. When provided, it must match the active project.",
    )


class URLIngestRequest(BaseModel):
    url: str = Field(
        ...,
        max_length=MAX_URL_FIELD_CHARS,
        description="HTTP or HTTPS URL to fetch and ingest",
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional extra metadata merged with the loader's source metadata",
    )
    project: str | None = Field(
        None,
        description="Optional project assertion. When provided, it must match the active project.",
    )


class DeleteRequest(BaseModel):
    doc_ids: list[str] = Field(..., description="List of stored IDs to delete from the collection.")


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
    security_mode: str | None = Field(
        None,
        description="Optional security mode for the project (e.g. 'sealed_v1')",
    )


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class StoreInitRequest(BaseModel):
    base_path: str = Field(
        ..., description="Base directory under which AxonStore/ will be created."
    )
    persist: bool = Field(
        False,
        description=(
            "Write the new store.base to config.yaml so the setting survives server restarts. "
            "Defaults to False so that test/scan calls do not permanently alter the user config. "
            "Pass True to persist the change across restarts."
        ),
    )


class ShareGenerateRequest(BaseModel):
    project: str = Field(..., description="Project name to share.")
    grantee: str = Field(..., description="OS username of the recipient.")
    ttl_days: int | None = Field(
        default=None,
        description=(
            "Optional time-to-live in days. When set, the share automatically "
            "expires this many days after creation. Owners can renew via "
            "POST /share/extend. None (default) means no expiry."
        ),
    )


class ShareRedeemRequest(BaseModel):
    share_string: str = Field(..., description="The base64 share string from the owner.")


class ShareRevokeRequest(BaseModel):
    key_id: str = Field(..., description="The key_id to revoke (e.g. 'sk_a1b2c3d4').")
    project: str | None = Field(
        default=None,
        description=(
            "Project name — required for sealed shares (key_id starting with "
            "'ssk_') so the revoke can locate the wrap file. Ignored for "
            "legacy plaintext shares."
        ),
    )
    rotate: bool = Field(
        default=False,
        description=(
            "Hard revoke for sealed shares: rotate the project DEK and "
            "re-encrypt every content file, invalidating ALL existing share "
            "wraps. Surviving grantees must re-issue + re-redeem. Ignored "
            "for legacy shares."
        ),
    )


class ShareExtendRequest(BaseModel):
    key_id: str = Field(..., description="The key_id to extend (e.g. 'sk_a1b2c3d4').")
    ttl_days: int | None = Field(
        default=None,
        description=(
            "New time-to-live in days, measured from now. None clears the "
            "expiry entirely (key never expires until revoked)."
        ),
    )


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
    sentence_window: bool | None = None
    sentence_window_size: int | None = Field(None, ge=1, le=10)
    crag_lite: bool | None = None
    code_graph: bool | None = None
    code_graph_bridge: bool | None = None
    graph_rag_mode: str | None = Field(None, pattern="^(local|global|hybrid)$")
    graph_rag_community: bool | None = None
    graph_rag_relations: bool | None = None
    graph_rag_ner_backend: str | None = Field(None, pattern="^(llm|gliner)$")
    graph_rag_depth: str | None = Field(None, pattern="^(light|standard|deep)$")
    graph_rag_budget: int | None = Field(None, ge=1)
    graph_rag_relation_budget: int | None = Field(None, ge=1)
    cite: bool | None = None
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


class SecurityBootstrapRequest(BaseModel):
    # SecretStr hides the passphrase in __repr__ and logging output.
    # Route handlers must call ``.get_secret_value()`` to retrieve the
    # plain string before passing it to the security backend.
    passphrase: SecretStr = Field(
        ..., description="Passphrase to bootstrap the security store with."
    )


class SecurityUnlockRequest(BaseModel):
    passphrase: SecretStr = Field(..., description="Passphrase to unlock the security store.")


class SecurityChangePassphraseRequest(BaseModel):
    old_passphrase: SecretStr = Field(..., description="Current passphrase.")
    new_passphrase: SecretStr = Field(..., description="New passphrase to set.")


class ProjectRotateKeysRequest(BaseModel):
    project_name: str = Field(..., description="Name of the sealed project to rotate keys for.")


class ProjectSealRequest(BaseModel):
    project_name: str = Field(..., description="Name of the open project to seal.")
    migration_mode: str = Field("snapshot", description="Migration mode: 'snapshot' or 'live'.")
