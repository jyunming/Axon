"""NoneGraphBackend — the explicit "no graph state is tracked" backend.

Selected when a project's ``graph_backend`` is ``"none"``. Every method is a
no-op or returns an empty/inert result; nothing is persisted. This matches
the locked v1 spec in ``docs/architecture/DYNAMIC_GRAPH_ROADMAP.md`` (§ `none`
backend semantics).
"""
from __future__ import annotations

from typing import Any

from axon.graph_backends.base import (
    FinalizationResult,
    GraphContext,
    GraphDataFilters,
    GraphPayload,
    IngestResult,
    RetrievalConfig,
)

BACKEND_ID = "none"


class NoneGraphBackend:
    """Inert ``GraphBackend`` — tracks no graph state, does nothing on ingest."""

    BACKEND_ID = BACKEND_ID

    def __init__(self, brain: Any) -> None:
        self._brain = brain  # unused; kept for constructor symmetry with other backends

    def ingest(self, chunks: list[dict]) -> IngestResult:
        return IngestResult(chunks_processed=len(chunks), backend_id=BACKEND_ID)

    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        existing_results: list[dict] | None = None,
    ) -> list[GraphContext]:
        return []

    def finalize(self, force: bool = False) -> FinalizationResult:
        return FinalizationResult(
            backend_id=BACKEND_ID,
            status="not_applicable",
            detail="graph_backend is 'none' — no graph state is tracked for this project.",
        )

    def clear(self) -> None:
        pass

    def delete_documents(self, chunk_ids: list[str]) -> None:
        pass

    def status(self) -> dict:
        return {"backend": BACKEND_ID, "enabled": False}

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        return GraphPayload()
