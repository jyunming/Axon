"""
axon/sparse_retrieval.py — Sparse retrieval interface design (Epic 4 Story 4.3).

Design intent
-------------
This module defines the *interface* for learned sparse retrieval so that a future
implementation can be plugged into Axon's existing hybrid path without destabilising
the current BM25 + dense fusion.

The interface is intentionally minimal:

- :class:`SparseVector` — wire format for a learned sparse document/query vector
- :class:`SparseRetriever` — Protocol for future implementations (SPLADE, BGE-M3
  sparse head, etc.)
- :func:`fuse_sparse` — fusion helper that combines dense-fused results with a sparse
  retriever's output, mirroring the existing :func:`~axon.retrievers.weighted_score_fusion`
  contract

Integration point
-----------------
The hook lives in :meth:`~axon.query_router.QueryRouterMixin._execute_retrieval`,
immediately after the BM25 / dense fusion block (``query_router.py`` ≈ line 767).
The integration is gated on ``brain._sparse_retriever is not None`` so existing
behaviour is unaffected until a real implementation is registered:

.. code-block:: python

    # query_router.py — inside _execute_retrieval(), after BM25 fusion:
    if getattr(self, "_sparse_retriever", None) is not None:
        from axon.sparse_retrieval import fuse_sparse
        results = fuse_sparse(self._sparse_retriever, query, results, top_k=cfg.top_k)

``AxonBrain.__init__`` should initialise the slot::

    self._sparse_retriever: SparseRetriever | None = None

Deferred items (out of scope for Story 4.3)
--------------------------------------------
- Actual SPLADE or BGE-M3 sparse-head implementation
- Sparse index storage (inverted index or Qdrant sparse vectors)
- Per-document sparse vector generation during ingest

These are tracked in ``SPARSE_RETRIEVAL_MILESTONE.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SparseVector wire format
# ---------------------------------------------------------------------------


@dataclass
class SparseVector:
    """A learned sparse embedding in coordinate-list format.
    Attributes
    ----------
    indices:
        Token / feature indices with non-zero weights.
    values:
        Corresponding weight values (must be the same length as *indices*).
    dim:
        Vocabulary / feature space size.  ``0`` means unknown / unspecified.
    model:
        Model identifier that produced this vector (for provenance tracking).
    """

    indices: list[int]
    values: list[float]
    dim: int = 0
    model: str = ""

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector: indices length ({len(self.indices)}) "
                f"must equal values length ({len(self.values)})"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable representation for storage / debug output."""
        return {
            "indices": self.indices,
            "values": self.values,
            "dim": self.dim,
            "model": self.model,
            "nnz": len(self.indices),
        }


def empty_sparse_vector(dim: int = 0, model: str = "") -> SparseVector:
    """Return an empty sparse vector (no non-zero entries)."""
    return SparseVector(indices=[], values=[], dim=dim, model=model)


# ---------------------------------------------------------------------------
# SparseRetriever Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SparseRetriever(Protocol):
    """Protocol for learned sparse retrieval backends.
    Any class that implements :meth:`search` satisfies this protocol and can
    be assigned to ``brain._sparse_retriever`` without inheriting from a base
    class.  This keeps the interface open for third-party implementations.
    Example future implementations:
    - ``BgeSparseRetriever`` — uses the sparse head of BGE-M3 via FastEmbed
    - ``SpladeRetriever`` — uses a standalone SPLADE checkpoint
    - ``QdrantSparseRetriever`` — delegates to Qdrant's native sparse-vector support
    """

    def encode_query(self, query: str) -> SparseVector:
        """Encode a query string into a sparse vector.
        Parameters
        ----------
        query:
            Raw query text.
        Returns
        -------
        SparseVector
            Learned sparse representation of the query.
        """
        ...

    def search(
        self,
        query_vector: SparseVector,
        top_k: int = 10,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Retrieve the top-k documents most relevant to *query_vector*.
        The return format mirrors :meth:`~axon.vector_store.OpenVectorStore.search`::

            [{"id": str, "text": str, "score": float, "metadata": dict}, ...]
        Parameters
        ----------
        query_vector:
            Sparse query representation produced by :meth:`encode_query`.
        top_k:
            Maximum number of results to return.
        filter_dict:
            Optional metadata filters (same semantics as dense search).
        Returns
        -------
        list[dict]
            Ranked results, highest score first.
        """
        ...


# ---------------------------------------------------------------------------
# Fusion helper
# ---------------------------------------------------------------------------


def fuse_sparse(
    retriever: SparseRetriever,
    query: str,
    dense_results: list[dict],
    *,
    top_k: int = 10,
    sparse_weight: float = 0.3,
    filter_dict: dict | None = None,
) -> list[dict]:
    """Fuse dense-retrieval results with a sparse retriever via weighted score fusion.
    Designed to be called from :meth:`~axon.query_router.QueryRouterMixin._execute_retrieval`
    as a drop-in extension after the existing BM25 fusion step.
    Parameters
    ----------
    retriever:
        Any object satisfying :class:`SparseRetriever`.
    query:
        Raw query string (passed to :meth:`SparseRetriever.encode_query`).
    dense_results:
        Already-fused dense + BM25 results (input to sparse fusion).
    top_k:
        Maximum results to return after fusion.
    sparse_weight:
        Weight applied to sparse scores during normalised combination.
        Dense scores receive weight ``1 - sparse_weight``.
    filter_dict:
        Forwarded to the sparse retriever's :meth:`~SparseRetriever.search` call.
    Returns
    -------
    list[dict]
        Re-ranked and merged results, highest score first.
    """
    try:
        q_vec = retriever.encode_query(query)
        sparse_hits = retriever.search(q_vec, top_k=top_k, filter_dict=filter_dict)
    except Exception as exc:
        # Sparse retrieval failure must never degrade the main path.
        logger.warning("sparse retrieval failed, falling back to dense-only: %s", exc)
        return dense_results
    # Build unified score map from dense results
    merged: dict[str, dict] = {r["id"]: dict(r) for r in dense_results}
    dense_max = max((r["score"] for r in dense_results), default=1.0) or 1.0
    sparse_max = max((r["score"] for r in sparse_hits), default=1.0) or 1.0
    for hit in sparse_hits:
        doc_id = hit["id"]
        sparse_norm = hit["score"] / sparse_max
        if doc_id in merged:
            # Combine: preserve dense score, add sparse contribution
            dense_norm = merged[doc_id]["score"] / dense_max
            merged[doc_id]["score"] = (
                1.0 - sparse_weight
            ) * dense_norm + sparse_weight * sparse_norm
            merged[doc_id]["metadata"] = merged[doc_id].get("metadata", {})
            merged[doc_id]["metadata"]["sparse_score"] = round(sparse_norm, 4)
        else:
            # New document from sparse path — add with sparse score
            entry = dict(hit)
            entry["score"] = sparse_weight * sparse_norm
            entry.setdefault("metadata", {})
            entry["metadata"]["sparse_score"] = round(sparse_norm, 4)
            merged[doc_id] = entry
    return sorted(merged.values(), key=lambda d: d["score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# No-op stub for testing / dependency injection
# ---------------------------------------------------------------------------


class _NoOpSparseRetriever:
    """Pass-through implementation used in unit tests.
    Always returns an empty result list so the fusion path can be exercised
    without a real sparse index.
    """

    def __init__(self, model: str = "noop") -> None:
        self._model = model

    def encode_query(self, query: str) -> SparseVector:
        return empty_sparse_vector(model=self._model)

    def search(
        self,
        query_vector: SparseVector,
        top_k: int = 10,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        return []


NoOpSparseRetriever = _NoOpSparseRetriever
