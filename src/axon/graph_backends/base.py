"""GraphBackend Protocol and shared data types.

This module defines the stable interface all graph backends must satisfy.
Concrete implementations live in sibling modules:
  - graphrag_backend.py  (existing GraphRAG, shipped in v0.1)
  - dynamic_graph_backend.py  (SQLite-WAL temporal graph, planned for v0.3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class GraphContext:
    """A single context item returned by GraphBackend.retrieve()."""

    context_id: str
    context_type: str  # "entity" | "relation" | "community" | "fact"
    text: str
    score: float
    rank: int
    backend_id: str  # "graphrag" | "dynamic"
    source_id: str = ""
    source_doc_id: str = ""
    source_chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    evidence_ids: list[str] = field(default_factory=list)
    # Entity names that matched the query and led to this context being retrieved.
    # Used by query_router to build GraphRAG local-search context headers.
    matched_entity_names: list[str] = field(default_factory=list)
    # Multi-hop diagnostics (Epic 1/4)
    hop_count: int = 0
    path: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class IngestResult:
    """Summary returned by GraphBackend.ingest()."""

    entities_added: int = 0
    relations_added: int = 0
    chunks_processed: int = 0
    backend_id: str = ""


@dataclass
class FinalizationResult:
    """Summary returned by GraphBackend.finalize()."""

    communities_built: int = 0
    time_elapsed: float = 0.0
    backend_id: str = ""


@dataclass
class GraphDataFilters:
    """Optional filters accepted by GraphBackend.graph_data()."""

    entity_types: list[str] | None = None
    min_degree: int = 0
    limit: int | None = None


@dataclass
class RetrievalConfig:
    """Retrieval parameters passed to GraphBackend.retrieve()."""

    top_k: int = 10


@dataclass
class GraphPayload:
    """Renderer-neutral graph payload.
    Shape is identical to what :meth:`AxonBrain.build_graph_payload` returns::

        {
            "nodes": [{"id", "name", "label", "type", ...}, ...],
            "links": [{"source", "target", "label", "relation", ...}, ...],
        }
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "links": self.links}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

_REQUIRED_METHODS = frozenset(
    {"ingest", "retrieve", "finalize", "clear", "delete_documents", "status", "graph_data"}
)


@runtime_checkable
class GraphBackend(Protocol):
    """Pluggable graph strategy interface.
    All seven methods are required.  Implementations must NOT raise
    ``NotImplementedError`` for ``status()`` — callers use it to probe
    readiness without triggering side effects.
    """

    def ingest(self, chunks: list[dict]) -> IngestResult:
        """Process a batch of document chunks and update the graph."""
        ...

    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        existing_results: list[dict] | None = None,
    ) -> list[GraphContext]:
        """Return graph-enriched contexts relevant to *query*.
        Args:
            query: The search query string.
            cfg: Optional retrieval parameters (top_k, etc.).
            existing_results: Chunks already present in the retrieval result set.
                Backends may use this to skip fetching already-present chunks
                (deduplication hint — not required for correctness).
        Returns:
            A list of new :class:`GraphContext` objects not already in
            *existing_results*.  Each context carries ``matched_entity_names``
            listing the entity names that caused this context to be retrieved.
        """
        ...

    def finalize(self, force: bool = False) -> FinalizationResult:
        """Trigger post-ingest finalisation (e.g. community detection).
        Implementations may defer this to a background thread unless
        *force* is True.
        """
        ...

    def clear(self) -> None:
        """Remove all graph state (entities, relations, communities, facts)."""
        ...

    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Remove all graph state derived from the given chunk IDs."""
        ...

    def status(self) -> dict:
        """Return a lightweight status/health dict (no side effects)."""
        ...

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        """Return the current graph as a renderer-neutral payload."""
        ...
