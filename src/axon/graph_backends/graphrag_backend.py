"""GraphRagBackend — thin adapter over AxonBrain's existing GraphRagMixin.

This backend delegates all graph operations to the methods already present on
``AxonBrain``.  It exists to satisfy the ``GraphBackend`` Protocol so that
the S3 refactor can wire ``AxonBrain`` to use ``self._graph_backend`` instead
of calling ``GraphRagMixin`` methods directly.

The ``ingest()`` method is intentionally a pass-through: entity/relation
extraction already happens inside ``AxonBrain.ingest()``.  This backend does
not re-run extraction; it only exposes the post-ingest graph state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axon.graph_backends.base import (
    FinalizationResult,
    GraphContext,
    GraphDataFilters,
    GraphPayload,
    IngestResult,
    RetrievalConfig,
)

if TYPE_CHECKING:
    pass

BACKEND_ID = "graphrag"


class GraphRagBackend:
    """Adapts ``AxonBrain``'s GraphRAG graph state to the ``GraphBackend`` Protocol."""

    def __init__(self, brain: Any) -> None:
        self._brain = brain

    # ------------------------------------------------------------------
    # GraphBackend protocol
    # ------------------------------------------------------------------
    def ingest(self, chunks: list[dict]) -> IngestResult:
        """No-op adapter — extraction is performed by ``AxonBrain.ingest()``.
        Entity/relation extraction is deeply embedded in the main ingest
        pipeline.  This method records the chunk count for bookkeeping only;
        the actual graph update has already occurred.
        """
        return IngestResult(
            chunks_processed=len(chunks),
            backend_id=BACKEND_ID,
        )

    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        existing_results: list[dict] | None = None,
    ) -> list[GraphContext]:
        """Delegate to ``_expand_with_entity_graph`` and convert to GraphContext.
        Passes *existing_results* to ``_expand_with_entity_graph`` so that
        already-retrieved chunks are not fetched again from the vector store.
        Only the newly-added chunks are returned as :class:`GraphContext` objects.
        Each context carries ``matched_entity_names`` — the entity names that
        matched the query and triggered graph expansion.  ``query_router`` uses
        these names to build the GraphRAG local-search context header.
        """
        _existing = existing_results or []
        _existing_ids = {r.get("id") for r in _existing if r.get("id")}
        # _expand_with_entity_graph returns (all_results, matched_entity_names)
        # where all_results = existing + newly fetched.
        expanded, matched_entities = self._brain._expand_with_entity_graph(query, _existing, cfg)
        # Store for the caller to read even when no new contexts are returned
        # (e.g. all entity-linked chunks already present in existing_results).
        self._last_matched_entity_names: list[str] = matched_entities
        # Return only the newly added contexts (not already in existing_results).
        return [
            GraphContext(
                context_id=r.get("id", ""),
                context_type="entity",
                text=r.get("text", r.get("page_content", "")),
                score=float(r.get("score", 0.5)),
                rank=i,
                backend_id=BACKEND_ID,
                source_chunk_id=r.get("id", ""),
                metadata=r.get("metadata", {}),
                matched_entity_names=matched_entities,
            )
            for i, r in enumerate(expanded)
            if r.get("id") not in _existing_ids
        ]

    def finalize(self, force: bool = False) -> FinalizationResult:
        """Trigger community detection on the attached brain.
        Delegates to ``AxonBrain.finalize_graph()`` which rebuilds community
        summaries when the graph is dirty or *force* is True.

        Failures are logged + propagated so a broken Leiden detector
        cannot return ``FinalizationResult`` claiming success (audit
        P1: previously swallowed all exceptions silently).
        """
        try:
            self._brain.finalize_graph(force=force)
        except Exception:
            import logging

            logging.getLogger("Axon").exception(
                "graphrag finalize failed (force=%s); state may be partially rebuilt", force
            )
            raise
        n_communities = len(getattr(self._brain, "_community_summaries", {}))
        return FinalizationResult(communities_built=n_communities, backend_id=BACKEND_ID)

    def clear(self) -> None:
        """Clear all GraphRAG state from the attached brain.

        Holds ``_graph_lock`` to prevent concurrent ``/graph/data``
        readers from iterating mid-clear (audit P1: previously
        unlocked, would crash readers with ``RuntimeError: dictionary
        changed size during iteration``).
        """
        with self._brain._graph_lock:
            self._brain._entity_graph.clear()
            self._brain._relation_graph.clear()
            self._brain._community_levels.clear()
            self._brain._community_summaries.clear()

    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Remove chunk IDs from entity graph; drop entities that become empty.

        Holds ``_graph_lock`` for the same reason as :meth:`clear`.
        """
        chunk_id_set = set(chunk_ids)
        with self._brain._graph_lock:
            for entity in list(self._brain._entity_graph):
                node = self._brain._entity_graph[entity]
                if not isinstance(node, dict):
                    continue
                node["chunk_ids"] = [c for c in node.get("chunk_ids", []) if c not in chunk_id_set]
                if not node["chunk_ids"]:
                    del self._brain._entity_graph[entity]

    def status(self) -> dict:
        """Return lightweight graph statistics (no side effects)."""
        brain = self._brain
        return {
            "backend": BACKEND_ID,
            "entities": len(brain._entity_graph),
            "relations": sum(len(v) for v in brain._relation_graph.values()),
            "communities": len(brain._community_levels.get(0, {})),
            "community_summaries": len(brain._community_summaries),
        }

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        """Return the current graph payload, optionally filtered."""
        raw = self._brain.build_graph_payload()
        nodes: list[dict] = raw["nodes"]
        links: list[dict] = raw["links"]
        if filters is not None:
            if filters.entity_types:
                allowed = set(filters.entity_types)
                nodes = [n for n in nodes if n.get("type") in allowed]
            if filters.min_degree > 0:
                nodes = [n for n in nodes if n.get("degree", 0) >= filters.min_degree]
            if filters.limit is not None:
                nodes = nodes[: filters.limit]
        return GraphPayload(nodes=nodes, links=links)
