"""
axon/sparse_retrieval.py — Sparse retrieval for Axon.

This module provides:

- :class:`SparseVector` — wire format for a learned sparse document/query vector.
- :class:`SparseRetriever` — Protocol satisfied by any sparse retrieval backend.
- :class:`SpladeSparseRetriever` — concrete SPLADE-based backend (Phase 1).
- :func:`fuse_sparse` — fusion helper that merges sparse retrieval results into
  the dense / hybrid pipeline using normalised score blending.
- :class:`NoOpSparseRetriever` — pass-through for tests / dependency injection.

Phase 1 design (audit batch E1)
-------------------------------
The :class:`SpladeSparseRetriever` implements the same coarse interface as
:class:`axon.retrievers.BM25Retriever` (``add``, ``search``, ``save``, ``load``).
It is purposely simple — vectors are stored as ``{doc_id: {token_id: weight}}``
and search is a brute-force dot-product. This is fine for tens of thousands of
documents and keeps the dependency surface minimal; an inverted-index fast path
can be added in Phase 2 without changing the public API.

The SPLADE encoder uses ``transformers`` and ``torch``, both lazy-imported.
Install via the optional extra::

    pip install axon-rag[sparse]

If ``transformers``/``torch`` are unavailable, attempting to construct
:class:`SpladeSparseRetriever` raises :class:`ImportError`.  Brain-level
integration handles this gracefully (logs a warning and leaves
``self._sparse_retriever`` as ``None`` so the pipeline is unchanged).

Integration point
-----------------
:meth:`AxonBrain.__init__` initialises ``self._sparse_retriever`` only when
``cfg.sparse_retrieval`` is true.  In :meth:`QueryRouterMixin._execute_retrieval`
the retriever is consulted after the BM25 fusion block, with results merged via
:func:`fuse_sparse` (normalised weighted-sum fusion).  Failures are logged and
swallowed — the dense / hybrid pipeline is never destabilised.
"""

from __future__ import annotations

import json
import logging
import os
import threading
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

    Any class that implements :meth:`encode_query` and :meth:`search` satisfies
    this protocol and can be assigned to ``brain._sparse_retriever`` without
    inheriting from a base class.

    Concrete implementations shipped with Axon:
    - :class:`SpladeSparseRetriever` — SPLADE via ``transformers``
    - :class:`NoOpSparseRetriever` — test stub
    """

    def encode_query(self, query: str) -> SparseVector:
        """Encode a query string into a sparse vector."""
        ...

    def search(
        self,
        query_vector: SparseVector,
        top_k: int = 10,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Retrieve the top-k documents most relevant to *query_vector*.

        Returns rows of ``{"id", "text", "score", "metadata"}``.
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

    Designed to be called from :meth:`QueryRouterMixin._execute_retrieval` as a
    drop-in extension after the existing BM25 fusion step.  Sparse retrieval
    failure must never degrade the main path — exceptions are logged and the
    dense results are returned unchanged.
    """
    try:
        q_vec = retriever.encode_query(query)
        sparse_hits = retriever.search(q_vec, top_k=top_k, filter_dict=filter_dict)
    except Exception as exc:
        logger.warning("sparse retrieval failed, falling back to dense-only: %s", exc)
        return dense_results
    merged: dict[str, dict] = {r["id"]: dict(r) for r in dense_results}
    dense_max = max((r["score"] for r in dense_results), default=1.0) or 1.0
    sparse_max = max((r["score"] for r in sparse_hits), default=1.0) or 1.0
    for hit in sparse_hits:
        doc_id = hit["id"]
        sparse_norm = hit["score"] / sparse_max
        if doc_id in merged:
            dense_norm = merged[doc_id]["score"] / dense_max
            merged[doc_id]["score"] = (
                1.0 - sparse_weight
            ) * dense_norm + sparse_weight * sparse_norm
            merged[doc_id]["metadata"] = merged[doc_id].get("metadata", {})
            merged[doc_id]["metadata"]["sparse_score"] = round(sparse_norm, 4)
        else:
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


# ---------------------------------------------------------------------------
# SPLADE backend (Phase 1)
# ---------------------------------------------------------------------------


_DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


class SpladeSparseRetriever:
    """SPLADE-based sparse retriever — Phase 1 backend.

    Mirrors the coarse interface of :class:`axon.retrievers.BM25Retriever`:

    - :meth:`add` — encode + persist sparse vectors for a batch of documents.
    - :meth:`search` — encode the query and rank by dot product.
    - :meth:`save` / :meth:`load` — JSON persistence.
    - :meth:`encode_query` — produce a :class:`SparseVector` for the query.

    The encoder lazily imports ``transformers`` and ``torch`` so the dependency
    is optional.  Constructing the retriever without those installed raises
    :class:`ImportError` — callers (e.g. :class:`AxonBrain`) handle this.

    Storage format
    --------------
    Documents are persisted as JSON to ``{storage_path}/sparse_index.json``::

        {
          "model": "naver/splade-cocondenser-ensembledistil",
          "dim": 30522,
          "docs": {
            "<doc_id>": {
              "text": "...",
              "metadata": {...},
              "vec": {"<token_id>": <weight>, ...}
            },
            ...
          }
        }

    Token-id keys are stringified for JSON compatibility and converted back to
    ints on load.
    """

    INDEX_FILENAME = "sparse_index.json"

    def __init__(
        self,
        storage_path: str = "./sparse_index",
        model: str = _DEFAULT_SPLADE_MODEL,
        device: str | None = None,
        max_length: int = 256,
        eager_load_model: bool = True,
    ) -> None:
        self.storage_path = storage_path
        self.model_name = model
        self.max_length = int(max_length)
        self.device = device  # None → auto (CPU)
        self.dim: int = 0
        # docs: {doc_id: {"text": str, "metadata": dict, "vec": {token_id: weight}}}
        self.docs: dict[str, dict[str, Any]] = {}
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._lock = threading.Lock()
        os.makedirs(self.storage_path, exist_ok=True)
        self.load()
        if eager_load_model and self._model is None:
            # Surface ImportError now (caller catches and disables sparse path).
            self._ensure_model_loaded()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-import transformers + torch and load the SPLADE checkpoint."""
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover — environment-dependent
            raise ImportError(
                "SpladeSparseRetriever requires the 'sparse' extra. "
                "Install with: pip install axon-rag[sparse]"
            ) from exc
        with self._lock:
            if self._model is not None:
                return
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
            if self.device:
                self._model.to(self.device)
            try:
                self.dim = int(self._model.config.vocab_size)
            except Exception:
                self.dim = 0

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> SparseVector:
        """Run SPLADE on *text* and return a :class:`SparseVector`."""
        self._ensure_model_loaded()
        torch = self._torch
        assert torch is not None and self._tokenizer is not None and self._model is not None
        with torch.no_grad():
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            logits = outputs.logits  # (1, seq, vocab)
            attention_mask = inputs.get("attention_mask")
            # Standard SPLADE pooling: log(1 + ReLU(logits)) max-pooled over tokens.
            relu_log = torch.log1p(torch.relu(logits))
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(relu_log).float()
                relu_log = relu_log * mask
            sparse_vec, _ = torch.max(relu_log, dim=1)  # (1, vocab)
            sparse_vec = sparse_vec.squeeze(0)
            nz_mask = sparse_vec > 0
            indices = nz_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            values = sparse_vec[nz_mask].tolist()
        return SparseVector(
            indices=[int(i) for i in indices],
            values=[float(v) for v in values],
            dim=self.dim,
            model=self.model_name,
        )

    def encode_query(self, query: str) -> SparseVector:
        """Encode *query* into a SPLADE sparse vector.

        On any encoder failure returns an empty vector so the retrieval path
        can degrade gracefully (caller will fall back to dense / BM25).
        """
        try:
            return self._encode(query)
        except Exception as exc:  # pragma: no cover — runtime / model failure
            logger.warning("SPLADE encode_query failed: %s", exc)
            return empty_sparse_vector(dim=self.dim, model=self.model_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, documents: list[dict[str, Any]], save: bool = True) -> int:
        """Encode and persist sparse vectors for *documents*.

        Each document must have ``id`` and ``text``; ``metadata`` is optional.
        Returns the number of documents added (encoder failures are skipped).
        """
        if not documents:
            return 0
        added = 0
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            if not doc_id or not text:
                continue
            try:
                vec = self._encode(text)
            except Exception as exc:
                logger.warning("SPLADE encode failed for doc %s: %s", doc_id, exc)
                continue
            self.docs[str(doc_id)] = {
                "text": text,
                "metadata": doc.get("metadata", {}) or {},
                "vec": {int(i): float(v) for i, v in zip(vec.indices, vec.values)},
            }
            added += 1
        if save and added:
            self.save()
        return added

    # Mirror the BM25Retriever `add_documents` name for callsite parity.
    add_documents = add

    # ------------------------------------------------------------------
    # Search (brute-force dot product)
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: SparseVector | str,
        top_k: int = 10,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Rank stored documents by dot product against *query_vector*.

        Accepts either a pre-encoded :class:`SparseVector` or a raw query
        string (encoded on the fly) for caller convenience.
        """
        if isinstance(query_vector, str):
            query_vector = self.encode_query(query_vector)
        if not query_vector.indices or not self.docs:
            return []
        q_map = {int(i): float(v) for i, v in zip(query_vector.indices, query_vector.values)}
        scored: list[tuple[float, str]] = []
        for doc_id, payload in self.docs.items():
            if filter_dict and not _matches_filter(payload.get("metadata", {}), filter_dict):
                continue
            doc_vec = payload["vec"]
            # Iterate the smaller side for speed.
            if len(q_map) <= len(doc_vec):
                score = sum(w * doc_vec.get(t, 0.0) for t, w in q_map.items())
            else:
                score = sum(w * q_map.get(t, 0.0) for t, w in doc_vec.items())
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for score, doc_id in scored[:top_k]:
            payload = self.docs[doc_id]
            out.append(
                {
                    "id": doc_id,
                    "text": payload.get("text", ""),
                    "score": float(score),
                    "metadata": dict(payload.get("metadata", {}) or {}),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> str:
        return os.path.join(self.storage_path, self.INDEX_FILENAME)

    def save(self) -> None:
        """Persist the sparse index to disk as JSON."""
        os.makedirs(self.storage_path, exist_ok=True)
        # Stringify token-id keys for JSON compatibility.
        serial_docs: dict[str, Any] = {}
        for doc_id, payload in self.docs.items():
            serial_docs[doc_id] = {
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}) or {},
                "vec": {str(int(t)): float(w) for t, w in payload.get("vec", {}).items()},
            }
        blob = {
            "model": self.model_name,
            "dim": int(self.dim),
            "docs": serial_docs,
        }
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        os.replace(tmp, self.index_path)

    def load(self) -> None:
        """Load the sparse index from disk if present."""
        path = self.index_path
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load sparse index at %s: %s", path, exc)
            return
        self.model_name = blob.get("model", self.model_name)
        self.dim = int(blob.get("dim", 0) or 0)
        raw_docs = blob.get("docs", {}) or {}
        self.docs = {}
        for doc_id, payload in raw_docs.items():
            vec_raw = payload.get("vec", {}) or {}
            self.docs[str(doc_id)] = {
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}) or {},
                "vec": {int(t): float(w) for t, w in vec_raw.items()},
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_filter(metadata: dict, filter_dict: dict) -> bool:
    """Lightweight metadata filter — exact-match on each key."""
    if not filter_dict:
        return True
    for key, want in filter_dict.items():
        if metadata.get(key) != want:
            return False
    return True
