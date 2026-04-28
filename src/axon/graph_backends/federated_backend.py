"""FederatedGraphBackend — runs GraphRagBackend + DynamicGraphBackend concurrently.

Retrieval results from both backends are fused using per-backend weighted
Reciprocal Rank Fusion (RRF).  All other protocol methods (ingest, finalize,
clear, delete_documents) delegate to each backend sequentially.

Select this backend with ``graph_backend: "federated"`` in config.yaml.
Per-backend RRF weights are tunable via ``graph_federation_weights``::

    graph_federation_weights:
      graphrag: 1.0       # default
      dynamic_graph: 1.0  # default

Wall-clock retrieval latency ≈ max(t_graphrag, t_dynamic) because both
backends are queried concurrently via ThreadPoolExecutor(max_workers=2).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from axon.graph_backends.base import (
    FinalizationResult,
    GraphContext,
    GraphDataFilters,
    GraphPayload,
    IngestResult,
    RetrievalConfig,
)

BACKEND_ID = "federated"


def _weighted_rrf(
    per_backend: dict[str, list[GraphContext]],
    weights: dict[str, float],
    k: int = 60,
) -> list[GraphContext]:
    """Fuse per-backend GraphContext lists with weighted Reciprocal Rank Fusion.

    For each backend b, result at rank i scores ``weights[b] / (k + i)``.
    Results with the same ``context_id`` are merged by summing their RRF scores.
    When two backends surface contexts with the same ``source_chunk_id``, the
    one with the higher RRF score is kept (dedup by source chunk).
    """
    rrf_scores: dict[str, float] = {}
    best_ctx: dict[str, GraphContext] = {}

    for backend_id, contexts in per_backend.items():
        w = weights.get(backend_id, 1.0)
        sorted_ctxs = sorted(contexts, key=lambda c: c.score, reverse=True)
        for i, ctx in enumerate(sorted_ctxs):
            rrf = w / (k + i + 1)
            cid = ctx.context_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf
            if cid not in best_ctx or rrf > (rrf_scores.get(cid, 0.0) - rrf):
                best_ctx[cid] = ctx

    # Dedup by source_chunk_id — keep highest RRF score when same source chunk
    # appears in both backends.
    chunk_best: dict[str, str] = {}  # source_chunk_id → context_id with best rrf
    for cid, ctx in best_ctx.items():
        scid = ctx.source_chunk_id
        if scid and scid in chunk_best:
            existing_cid = chunk_best[scid]
            if rrf_scores.get(cid, 0.0) > rrf_scores.get(existing_cid, 0.0):
                chunk_best[scid] = cid
        elif scid:
            chunk_best[scid] = cid

    # Remove the lower-scored duplicate when a source_chunk_id clash exists
    deduped_ids = set(best_ctx.keys())
    for scid, winner_cid in chunk_best.items():
        for cid in list(deduped_ids):
            if best_ctx.get(cid) and best_ctx[cid].source_chunk_id == scid and cid != winner_cid:
                deduped_ids.discard(cid)

    fused = sorted(
        (best_ctx[cid] for cid in deduped_ids if cid in best_ctx),
        key=lambda c: rrf_scores.get(c.context_id, 0.0),
        reverse=True,
    )
    for rank, ctx in enumerate(fused):
        ctx.score = rrf_scores.get(ctx.context_id, 0.0)
        ctx.rank = rank
    return fused


class FederatedGraphBackend:
    """Concurrent federation of GraphRagBackend + DynamicGraphBackend.

    Both backends are instantiated eagerly at construction time and share the
    same ``brain`` reference.  If either backend fails to initialise (e.g.
    ``[sealed]`` extra missing for Dynamic Graph), the error is logged and that
    backend is silently dropped so the other continues to function.
    """

    BACKEND_ID = BACKEND_ID

    def __init__(self, brain: Any) -> None:
        import logging

        _log = logging.getLogger("Axon")
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend
        from axon.graph_backends.graphrag_backend import GraphRagBackend

        self._backends: list[Any] = []
        for cls in (GraphRagBackend, DynamicGraphBackend):
            try:
                self._backends.append(cls(brain))
            except Exception as exc:
                _log.warning("FederatedGraphBackend: skipping %s — %s", cls.__name__, exc)

        raw: dict = getattr(getattr(brain, "config", None), "graph_federation_weights", {}) or {}
        self._weights: dict[str, float] = {
            "graphrag": float(raw.get("graphrag", 1.0)),
            "dynamic_graph": float(raw.get("dynamic_graph", 1.0)),
        }

    # ------------------------------------------------------------------
    # Retrieve (concurrent)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        existing_results: list[dict] | None = None,
    ) -> list[GraphContext]:
        if not self._backends:
            return []
        per_backend: dict[str, list[GraphContext]] = {}
        with ThreadPoolExecutor(max_workers=len(self._backends)) as pool:
            future_to_id = {
                pool.submit(b.retrieve, query, cfg, existing_results): b.BACKEND_ID
                for b in self._backends
            }
            for future in as_completed(future_to_id):
                bid = future_to_id[future]
                try:
                    per_backend[bid] = future.result()
                except Exception:
                    per_backend[bid] = []
        return _weighted_rrf(per_backend, self._weights)

    # ------------------------------------------------------------------
    # Ingest / finalize / clear / delete_documents (sequential delegation)
    # ------------------------------------------------------------------
    def ingest(self, chunks: list[dict]) -> IngestResult:
        total = IngestResult(backend_id=BACKEND_ID)
        for b in self._backends:
            try:
                r = b.ingest(chunks)
                total.entities_added += r.entities_added
                total.relations_added += r.relations_added
                total.chunks_processed = max(total.chunks_processed, r.chunks_processed)
            except Exception:
                pass
        return total

    def finalize(self, force: bool = False) -> FinalizationResult:
        total = FinalizationResult(backend_id=BACKEND_ID)
        for b in self._backends:
            try:
                r = b.finalize(force=force)
                total.communities_built += r.communities_built
                total.time_elapsed += r.time_elapsed
            except Exception:
                pass
        return total

    def clear(self) -> None:
        for b in self._backends:
            try:
                b.clear()
            except Exception:
                pass

    def delete_documents(self, chunk_ids: list[str]) -> None:
        for b in self._backends:
            try:
                b.delete_documents(chunk_ids)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Status / graph_data
    # ------------------------------------------------------------------
    def status(self) -> dict:
        merged: dict[str, Any] = {"backend": BACKEND_ID, "sub_backends": []}
        for b in self._backends:
            try:
                merged["sub_backends"].append(b.status())
            except Exception:
                pass
        return merged

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        nodes: dict[str, dict] = {}
        links: list[dict] = []
        for b in self._backends:
            try:
                payload = b.graph_data(filters)
                for node in payload.nodes:
                    nid = node.get("id", "")
                    if nid and nid not in nodes:
                        nodes[nid] = node
                links.extend(payload.links)
            except Exception:
                pass
        return GraphPayload(nodes=list(nodes.values()), links=links)
