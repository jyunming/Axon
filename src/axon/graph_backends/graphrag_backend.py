"""GraphRagBackend — adapts a composed GraphRagEngine to the GraphBackend Protocol.

Owns a dedicated ``GraphRagEngine`` instance (``self._engine``) that holds all
GraphRAG entity/relation/community state and logic — see
``graphrag_engine.py`` for why this composition exists instead of
``AxonBrain`` inheriting ``GraphRagMixin`` directly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from axon.graph_backends.base import (
    FinalizationResult,
    GraphContext,
    GraphDataFilters,
    GraphPayload,
    IngestResult,
    RetrievalConfig,
)
from axon.graph_backends.graphrag_engine import GraphRagEngine

if TYPE_CHECKING:
    pass

logger = logging.getLogger("Axon")

BACKEND_ID = "graphrag"


class GraphRagBackend:
    """Adapts a composed ``GraphRagEngine``'s GraphRAG state to the ``GraphBackend`` Protocol."""

    BACKEND_ID = BACKEND_ID

    def __init__(self, brain: Any) -> None:
        self._brain = brain
        self._engine = GraphRagEngine(brain)
        self._engine.load()

    # ------------------------------------------------------------------
    # GraphRAG-specific bridge — hasattr-guarded by callers (main.py,
    # query_router.py), mirroring the existing list_conflicts precedent.
    # Not on the shared GraphBackend Protocol: these are GraphRAG-specific,
    # not something every backend needs to implement.
    # ------------------------------------------------------------------
    def load(self) -> None:
        """(Re)load all GraphRAG state from disk. Called by switch_project()."""
        self._engine.load()

    def merge_descendants(self, descendants: list[str]) -> None:
        """Merge descendant projects' graph state into this project's in-memory state."""
        self._engine.merge_descendants(descendants)

    def expand_with_entity_graph(
        self, query: str, results: list[dict], cfg: Any = None
    ) -> tuple[list[dict], list[str]]:
        """Expand retrieval results using GraphRAG entity linkage. See
        GraphRagEngine.expand_with_entity_graph() for the full algorithm.
        """
        return self._engine.expand_with_entity_graph(query, results, cfg)

    def local_search_context(self, query: str, matched_entities: list[str], cfg: Any) -> str:
        """Build the GraphRAG local-search context header for *query*."""
        return self._engine._local_search_context(query, matched_entities, cfg)

    def global_search_map_reduce(self, query: str, cfg: Any) -> str:
        """Run GraphRAG global-search map-reduce over community summaries for *query*."""
        return self._engine._global_search_map_reduce(query, cfg)

    def classify_query_needs_graphrag(self, query: str, mode: str) -> bool:
        """Return True if *query* should trigger GraphRAG expansion (auto-routing).

        *mode* is the classifier strategy ("heuristic" / "llm"), not a bool.
        """
        return self._engine._classify_query_needs_graphrag(query, mode)

    def ensure_community_summaries(
        self, query_hint: str, index_community_reports: bool = True
    ) -> None:
        """Lazily build community summaries if the graph has communities but no
        summaries yet (community_levels populated, community_summaries empty).

        Double-checked locking under the brain's ``_community_rebuild_lock``
        (proxied onto the engine) — safe to call unconditionally, including
        concurrently; this is a cheap no-op once summaries exist. Subsumes
        the ``_community_levels``-and-not-``_community_summaries`` lazy guard
        that previously lived duplicated in query_router.py's query() and
        query_stream(), since ``_community_levels`` is engine-owned state
        query_router.py can no longer read directly.
        """
        if not self._engine._community_levels or self.has_community_summaries():
            return
        with self._engine._community_rebuild_lock:
            if not self.has_community_summaries():
                self._engine._generate_community_summaries(query_hint=query_hint)
                if index_community_reports:
                    self._engine._index_community_reports_in_vector_store()

    def flush_ingest_saves(self) -> None:
        """Persist entity/relation/claims graph state and the extraction cache.

        Called by AxonBrain.finalize_ingest() after a batch-mode ingest,
        since entity/relation/claims graphs are normally saved incrementally
        during ingest but batch mode (``ingest_batch_mode=True``) defers
        those writes until finalize_ingest() is explicitly called.

        The extraction-cache flush below is unconditional (not gated on
        ingest_batch_mode) — it mirrors the pre-M2 AxonBrain.finalize_ingest(),
        which flushed a dirty extraction cache regardless of batch mode.
        Without this, GraphRagEngine.ingest_chunks()'s deferred cache writes
        (skipped mid-batch via _defer_saves) would stay unpersisted until
        some later close()/flush() happened to run.
        """
        engine = self._engine
        if getattr(engine.config, "ingest_batch_mode", False):
            engine._save_entity_graph()
            engine._save_relation_graph()
            if getattr(engine, "_claims_graph", None):
                engine._save_claims_graph()
            logger.info("finalize_ingest: entity/relation/claims graphs saved.")
        if getattr(engine, "_graph_rag_cache_dirty", False):
            engine._save_graph_rag_extraction_cache()

    def flush(self) -> None:
        """Flush any dirty in-memory GraphRAG state and pending background persists."""
        engine = self._engine
        if getattr(engine, "_graph_rag_cache_dirty", False):
            try:
                engine._save_graph_rag_extraction_cache()
            except Exception as e:
                logger.debug("Could not flush graph_rag extraction cache: %s", e)
        try:
            engine._flush_pending_saves()
        except Exception as e:
            logger.debug("Could not flush pending graph saves: %s", e)

    def close(self) -> None:
        """Flush dirty graph state and shut down the engine's background persist executor."""
        self.flush()
        persist_exec = getattr(self._engine, "_persist_executor_internal", None)
        if persist_exec is not None:
            try:
                persist_exec.shutdown(wait=True, cancel_futures=False)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Persist executor shutdown raised: %s", exc)
            self._engine._persist_executor_internal = None

    # ------------------------------------------------------------------
    # GraphBackend protocol
    # ------------------------------------------------------------------
    def ingest(self, chunks: list[dict]) -> IngestResult:
        """Extract entities/relations/claims from *chunks* and merge into the graph."""
        return self._engine.ingest_chunks(chunks)

    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        existing_results: list[dict] | None = None,
    ) -> list[GraphContext]:
        """Delegate to ``expand_with_entity_graph`` and convert to GraphContext.
        Passes *existing_results* to ``expand_with_entity_graph`` so that
        already-retrieved chunks are not fetched again from the vector store.
        Only the newly-added chunks are returned as :class:`GraphContext` objects.
        Each context carries ``matched_entity_names`` — the entity names that
        matched the query and triggered graph expansion.  ``query_router`` uses
        these names to build the GraphRAG local-search context header.
        """
        _existing = existing_results or []
        _existing_ids = {r.get("id") for r in _existing if r.get("id")}
        # expand_with_entity_graph returns (all_results, matched_entity_names)
        # where all_results = existing + newly fetched.
        expanded, matched_entities = self._engine.expand_with_entity_graph(query, _existing, cfg)
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
        """Trigger community detection on the composed engine.
        Delegates to ``GraphRagEngine.finalize_graph()`` which rebuilds community
        summaries when the graph is dirty or *force* is True.

        Failures are logged + propagated so a broken Leiden detector
        cannot return ``FinalizationResult`` claiming success (audit
        P1: previously swallowed all exceptions silently).
        """
        try:
            self._engine.finalize_graph(force=force)
        except Exception:
            logger.exception(
                "graphrag finalize failed (force=%s); state may be partially rebuilt", force
            )
            raise
        n_communities = len(getattr(self._engine, "_community_summaries", {}))
        return FinalizationResult(communities_built=n_communities, backend_id=BACKEND_ID)

    _PERSISTABLE_SAVE_METHODS = (
        "_save_entity_graph",
        "_save_relation_graph",
        "_save_community_levels",
        "_save_community_summaries",
        "_save_community_hierarchy",
        "_save_claims_graph",
        "_save_entity_embeddings",
    )

    def clear(self, *, persist: bool = False) -> None:
        """Clear all GraphRAG state from the composed engine.

        Delegates to ``GraphRagEngine._reset_graph_state()``, the single
        source of truth for "wipe graph state" — it resets every graph-
        related field (not just the four core dicts, also the traversal
        cache and the community-build-in-progress flag) and holds
        ``_graph_lock`` internally (audit P1: previously unlocked, would
        crash readers with ``RuntimeError: dictionary changed size during
        iteration``).

        ``_reset_graph_state()`` is memory-only by design (also used by
        read-only scope switching, which must never write project data to
        disk) — when *persist* is True, additionally call the 7 ``_save_*``
        methods so the now-empty state is actually written to disk.
        """
        self._engine._reset_graph_state()
        if persist:
            for method_name in self._PERSISTABLE_SAVE_METHODS:
                method = getattr(self._engine, method_name, None)
                if callable(method):
                    method()

    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Remove chunk IDs from entity/relation/claims graph state.

        Delegates to ``GraphRagEngine._prune_entity_graph()``, the existing
        ``GraphRagMixin`` method that also prunes the relation graph and
        claims graph, updates the entity token index, recomputes entity
        frequency, and persists the changes to disk — a straight
        entity-graph-only reimplementation here previously dropped all of
        that.
        """
        self._engine._prune_entity_graph(set(chunk_ids))

    def status(self) -> dict:
        """Return lightweight graph statistics (no side effects)."""
        engine = self._engine
        return {
            "backend": BACKEND_ID,
            "entities": len(engine._entity_graph),
            "relations": sum(len(v) for v in engine._relation_graph.values()),
            "communities": len(engine._community_levels.get(0, {})),
            "community_summaries": len(engine._community_summaries),
            "community_build_in_progress": bool(
                getattr(engine, "_community_build_in_progress", False)
            ),
            "community_graph_dirty": bool(getattr(engine, "_community_graph_dirty", False)),
        }

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        """Return the current graph payload, optionally filtered."""
        raw = self._engine.build_graph_payload()
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

    def has_entities(self) -> bool:
        return bool(self._engine._entity_graph)

    def has_community_summaries(self) -> bool:
        return bool(self._engine._community_summaries)
