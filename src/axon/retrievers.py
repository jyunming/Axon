import heapq
import json
import logging
import os
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger("Axon.Retrievers")


class BM25Retriever:
    """Keyword-based retriever using BM25 algorithm.

    Complements vector search for specific term matching.

    Sync hot-spot audit (Epic 6 Story 6.1)
    ----------------------------------------
    The following paths were audited for event-loop risk:

    * ``add_documents()`` — previously rebuilt the full ``BM25Okapi`` index on
      every call (O(N) per batch, O(N²) over a whole ingest run).  **Mitigated
      in Story 6.2** via a lazy-rebuild flag: the index is now rebuilt exactly
      once on the first ``search()`` call after the last write, not on every
      add.  Disk writes remain synchronous but are already bounded by
      ``save_deferred`` batching in the ingest path.

    * ``search()`` — ``BM25Okapi.get_scores()`` is CPU-bound.  **Already
      mitigated**: ``QueryRouterMixin._execute_retrieval()`` offloads BM25
      searches to ``self._executor`` (a ``ThreadPoolExecutor``), preventing
      event-loop blocking on the async API path.

    * ``save()`` — synchronous atomic file write.  No change needed; writes are
      short relative to the ingest batch, and the ``save_deferred`` flag already
      lets callers defer them.

    * ``delete_documents()`` — rebuilds index + saves synchronously.  Index
      rebuild deferred to next ``search()`` via the lazy flag; save is still
      immediate (delete is an explicit operator action, not a hot path).

    * ``load()`` — called once at startup only.  No event-loop risk.

    Residual risk: ``save()`` blocks the calling thread for large corpora
    (10 k+ docs / several MB JSON).  Acceptable for the local-first deployment
    model; address with async file I/O if a high-throughput multi-tenant API
    becomes a requirement.
    """

    def __init__(self, storage_path: str = "./bm25_index"):
        self.storage_path = storage_path
        self.corpus_file = os.path.join(storage_path, "bm25_corpus.json")
        self.bm25 = None
        self._dirty: bool = False  # True when corpus was modified but index not yet rebuilt
        self.corpus: list[
            dict[str, Any]
        ] = []  # List of dicts: {'id': id, 'text': text, 'metadata': meta}

        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        self.load()

    def close(self):
        """Release index and corpus references."""
        self.bm25 = None
        self.corpus = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer."""
        return text.lower().split()

    def add_documents(self, documents: list[dict[str, Any]], save_deferred: bool = False) -> None:
        """Add documents to the BM25 index.  No-op if *documents* is empty.

        The BM25Okapi index is **not rebuilt immediately**; it is rebuilt lazily
        on the next :meth:`search` call.  This eliminates the O(N²) rebuild cost
        that occurred during a full ingest run when this method was called for
        each chunk batch (Epic 6, Story 6.2).

        Args:
            documents: List of document dicts with keys ``id``, ``text``, ``metadata``.
            save_deferred: When ``True``, skip the disk write — corpus is updated in
                memory only.  Call :meth:`flush` (or :meth:`save`) when the batch
                is complete.
        """
        if not documents:
            return
        self.corpus.extend(documents)
        self._dirty = True  # index will be rebuilt lazily on next search()
        if not save_deferred:
            self.save()

    def _rebuild_index(self) -> None:
        """Rebuild BM25Okapi from the current corpus and clear the dirty flag."""
        if self.corpus:
            tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
        self._dirty = False

    def flush(self) -> None:
        """Explicitly save corpus — call after deferred batch ingest session."""
        self.save()

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search the BM25 index, rebuilding it first if the corpus was modified."""
        if self._dirty:
            self._rebuild_index()
        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices efficiently using a heap
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

        results = []
        for i in top_indices:
            if scores[i] > 0:
                doc = self.corpus[i].copy()
                doc["score"] = float(scores[i])
                results.append(doc)

        return results

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Remove documents by ID.

        The index rebuild is deferred to the next :meth:`search` call via the
        lazy-rebuild flag (Epic 6, Story 6.2).  The corpus is saved immediately
        since delete is an explicit operator action, not a hot-path operation.
        """
        original_count = len(self.corpus)
        self.corpus = [doc for doc in self.corpus if doc["id"] not in doc_ids]
        if len(self.corpus) < original_count:
            self._dirty = True  # rebuild deferred to next search()
            self.save()

    def save(self):
        """Save the corpus to disk as JSON (atomic write via temp file)."""
        tmp_file = self.corpus_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False)

        # os.replace is atomic on POSIX and uses MoveFileEx(REPLACE_EXISTING) on
        # Windows — safe even when the destination already exists. Fall back to a
        # direct copy if os.replace fails (PermissionError from an exclusive lock,
        # or OSError/WinError 87 EINVAL on some Windows file systems).
        try:
            os.replace(tmp_file, self.corpus_file)
        except OSError:
            import shutil

            try:
                shutil.copy2(tmp_file, self.corpus_file)
            finally:
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass
        logger.info(f"💾 BM25 corpus saved to {self.corpus_file}")

    def load(self):
        """Load corpus from JSON."""
        if os.path.exists(self.corpus_file):
            try:
                with open(self.corpus_file, encoding="utf-8") as f:
                    self.corpus = json.load(f)
                if self.corpus:
                    tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"📂 Loaded BM25 corpus with {len(self.corpus)} documents")
            except Exception as e:
                logger.error(f"Failed to load BM25 corpus: {e}")
                self.corpus = []
                self.bm25 = None


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize a list of scores to [0.0, 1.0]."""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if max_val == min_val:
        return [1.0] * len(scores) if max_val > 0 else [0.0] * len(scores)
    return [(s - min_val) / (max_val - min_val) for s in scores]


def weighted_score_fusion(
    vector_results: list[dict], bm25_results: list[dict], weight: float = 0.7
) -> list[dict]:
    """Merge results using a normalized convex combination of scores.

    weight: 1.0 = Pure Semantic, 0.0 = Pure Lexical.
    """
    all_docs = {}

    # 1. Extract and normalize Vector scores
    v_scores = [doc.get("score", 0.0) for doc in vector_results]
    v_norm = _min_max_normalize(v_scores)

    for i, doc in enumerate(vector_results):
        doc_id = doc["id"]
        all_docs[doc_id] = doc.copy()
        # Keep original vector score for thresholding
        all_docs[doc_id]["vector_score"] = doc.get("score", 0.0)
        # Initialize fused score with weighted vector component
        all_docs[doc_id]["score"] = v_norm[i] * weight

    # 2. Extract and normalize BM25 scores
    b_scores = [doc.get("score", 0.0) for doc in bm25_results]
    b_norm = _min_max_normalize(b_scores)

    for i, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id not in all_docs:
            all_docs[doc_id] = doc.copy()
            # If not found in vector search, assume 0.0 semantic similarity
            all_docs[doc_id]["vector_score"] = 0.0
            all_docs[doc_id]["score"] = 0.0
            all_docs[doc_id]["fused_only"] = True

        # Add weighted BM25 component
        all_docs[doc_id]["score"] += b_norm[i] * (1.0 - weight)

    # 3. Sort by final fused score
    final_results = list(all_docs.values())
    final_results.sort(key=lambda x: x["score"], reverse=True)

    return final_results


def reciprocal_rank_fusion(
    vector_results: list[dict], bm25_results: list[dict], k: int = 60
) -> list[dict]:
    """Merge results from multiple retrievers using Reciprocal Rank Fusion.

    The original cosine similarity score from vector search is preserved in the
    ``vector_score`` field so the UI can display a meaningful relevance value.
    The RRF-fused score (used only for ranking) is stored in ``score``.
    """
    fused_scores: dict[str, float] = {}

    # Preserve original cosine scores keyed by doc_id
    vector_scores = {doc["id"]: doc.get("score", 0.0) for doc in vector_results}

    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 1 / (rank + k)
            doc["fused_only"] = True
        else:
            fused_scores[doc_id] += 1 / (rank + k)

    all_docs = {doc["id"]: doc for doc in vector_results}
    for doc in bm25_results:
        if doc["id"] not in all_docs:
            all_docs[doc["id"]] = doc

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    final_results = []
    for doc_id in sorted_ids:
        doc = all_docs[doc_id]
        doc["score"] = fused_scores[doc_id]
        # Expose the original cosine similarity for display purposes
        if doc_id in vector_scores:
            doc["vector_score"] = vector_scores[doc_id]
        final_results.append(doc)

    return final_results
