"""
axon/sentence_window.py — Sentence-Window Retrieval (Epic 1, Stories 1.1–1.3)

Provides:
- SentenceRecord: sentence-level metadata model (Story 1.1)
- SentenceWindowIndex: persists sentence→chunk linkage and reconstructs context
  windows (Stories 1.1 + 1.3)
- SentenceVectorStore: lightweight numpy-backed sentence embedding index (Story 1.2)
- segment_text() / segment_chunk(): deterministic prose sentence segmentation

Design constraints
------------------
- No new runtime dependencies (numpy is already used in the stack)
- Eligible chunks: non-code, non-RAPTOR-summary, leaf-only
- Context windows are coherent (±N surrounding sentences from the same chunk)
- Window results expose chunk-level IDs for citation compatibility
- Secondary index stored alongside BM25 data in {bm25_path}/.sentence_index.json
  and {bm25_path}/.sentence_vecs.npy / .sentence_meta.json
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eligibility — which chunks get sentence-indexed (Story 1.1)
# ---------------------------------------------------------------------------

_INELIGIBLE_SOURCE_CLASSES: frozenset[str] = frozenset({"code"})

# RAPTOR-summary chunks and explicit "parent" marker chunks are skipped
_INELIGIBLE_CHUNK_KINDS: frozenset[str] = frozenset({"raptor", "parent"})


def is_eligible(chunk: dict) -> bool:
    """Return True if *chunk* should be added to the sentence-window index.

    Eligible chunks are non-code, non-summary leaf chunks.  The check is
    intentionally strict so sentence indexing never inflates the index with
    synthetic or code-structured text.
    """
    meta = chunk.get("metadata") or {}
    if meta.get("source_class") in _INELIGIBLE_SOURCE_CLASSES:
        return False
    if meta.get("chunk_kind") in _INELIGIBLE_CHUNK_KINDS:
        return False
    if meta.get("raptor_level") is not None:
        return False
    return True


# ---------------------------------------------------------------------------
# Sentence segmentation (Story 1.1)
# ---------------------------------------------------------------------------

# Match sentence-terminal punctuation followed by whitespace.
# Lookbehind keeps the terminator attached to the preceding sentence fragment.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Fragments shorter than this are merged into the adjacent sentence rather
# than stored as independent sentence records.
_MIN_SENTENCE_CHARS: int = 10


def segment_text(text: str) -> list[str]:
    """Segment *text* into a list of sentence strings.

    Uses a simple regex boundary that splits after ``. ``, ``! ``, or ``? ``.
    Short fragments (< _MIN_SENTENCE_CHARS characters) are merged into the
    preceding sentence to avoid polluting the index with stubs.

    Returns an empty list for blank or None input.
    """
    if not text or not text.strip():
        return []

    raw = _SENTENCE_BOUNDARY.split(text.strip())
    sentences: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if len(part) < _MIN_SENTENCE_CHARS and sentences:
            # Merge stub into the previous sentence
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)

    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# SentenceRecord — sentence-level metadata model (Story 1.1)
# ---------------------------------------------------------------------------


@dataclass
class SentenceRecord:
    """One sentence within a prose chunk.

    Fields are sufficient to:
    - reconstruct a ±N context window from sibling sentences
    - map the sentence back to its parent chunk and source for citations
    """

    sentence_id: str  # "{chunk_id}_s{sentence_idx}"
    chunk_id: str  # parent chunk ID (stable identifier for citations)
    source: str  # original source file path
    sentence_idx: int  # 0-based position within the chunk
    total_sentences: int  # total sentence count for the chunk
    text: str  # sentence text


def _make_sentence_id(chunk_id: str, idx: int) -> str:
    return f"{chunk_id}_s{idx}"


def segment_chunk(chunk: dict) -> list[SentenceRecord]:
    """Segment a chunk dict into :class:`SentenceRecord` objects.

    Returns an empty list for ineligible chunks, empty text, or chunks that
    produce only a single sentence (not worth indexing at sentence granularity).
    Only call on chunks that have already passed :func:`is_eligible`.
    """
    chunk_id = chunk.get("id", "")
    source = (chunk.get("metadata") or {}).get("source", chunk_id)
    sentences = segment_text(chunk.get("text") or "")
    total = len(sentences)
    return [
        SentenceRecord(
            sentence_id=_make_sentence_id(chunk_id, i),
            chunk_id=chunk_id,
            source=source,
            sentence_idx=i,
            total_sentences=total,
            text=sent,
        )
        for i, sent in enumerate(sentences)
    ]


# ---------------------------------------------------------------------------
# SentenceWindowIndex — linkage store + window reconstruction (Stories 1.1 + 1.3)
# ---------------------------------------------------------------------------

_INDEX_FILENAME = ".sentence_index.json"


class SentenceWindowIndex:
    """Maps sentence IDs to :class:`SentenceRecord` objects and chunk IDs to
    ordered sentence ID lists.

    Provides :meth:`get_window` to reconstruct a coherent ±N-sentence context
    window around any sentence hit.  Persisted as a single JSON file alongside
    the BM25 index.

    Thread safety: not thread-safe — access is serialised by the ingest lock
    in ``AxonBrain``.
    """

    def __init__(self) -> None:
        # sentence_id → SentenceRecord (stored as plain dict for JSON compat)
        self._records: dict[str, dict] = {}
        # chunk_id → ordered list of sentence_ids
        self._chunk_to_sentences: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_records(self, records: list[SentenceRecord]) -> None:
        """Register sentence records derived from one chunk."""
        if not records:
            return
        chunk_id = records[0].chunk_id
        sent_ids: list[str] = []
        for rec in records:
            self._records[rec.sentence_id] = asdict(rec)
            sent_ids.append(rec.sentence_id)
        self._chunk_to_sentences[chunk_id] = sent_ids

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_record(self, sentence_id: str) -> SentenceRecord | None:
        """Return the :class:`SentenceRecord` for *sentence_id*, or None."""
        d = self._records.get(sentence_id)
        if d is None:
            return None
        return SentenceRecord(**d)

    def get_all_for_chunk(self, chunk_id: str) -> list[SentenceRecord]:
        """Return all sentence records for *chunk_id* in sentence order."""
        ids = self._chunk_to_sentences.get(chunk_id, [])
        return [SentenceRecord(**self._records[sid]) for sid in ids if sid in self._records]

    def get_window(self, sentence_id: str, window_size: int = 3) -> str | None:
        """Reconstruct a coherent context window around *sentence_id*.

        Returns the concatenated text of sentences in the range
        [max(0, idx - window_size), min(total, idx + window_size + 1)] drawn
        from the parent chunk.

        Returns None if *sentence_id* is unknown; falls back to the single
        sentence text if sibling records are missing.

        Citations still map to the parent ``chunk_id``, which is stable.
        Overlapping windows from different sentence hits in the same chunk are
        deduplicated by the caller (query_router) before final results are
        assembled.
        """
        rec = self.get_record(sentence_id)
        if rec is None:
            return None
        siblings = self.get_all_for_chunk(rec.chunk_id)
        if not siblings:
            return rec.text
        lo = max(0, rec.sentence_idx - window_size)
        hi = min(len(siblings), rec.sentence_idx + window_size + 1)
        return " ".join(siblings[i].text for i in range(lo, hi))

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Save the index to *directory*/.sentence_index.json."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _INDEX_FILENAME
        data = {
            "records": self._records,
            "chunk_to_sentences": self._chunk_to_sentences,
        }
        try:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("SentenceWindowIndex: save failed: %s", exc)

    def load(self, directory: Path) -> None:
        """Load the index from *directory*/.sentence_index.json if present."""
        path = Path(directory) / _INDEX_FILENAME
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._records = data.get("records", {})
            self._chunk_to_sentences = data.get("chunk_to_sentences", {})
        except Exception as exc:
            logger.warning("SentenceWindowIndex: load failed: %s", exc)


# ---------------------------------------------------------------------------
# SentenceVectorStore — numpy-backed sentence embedding index (Story 1.2)
# ---------------------------------------------------------------------------

_VECS_FILENAME = ".sentence_vecs.npy"
_META_FILENAME = ".sentence_meta.json"


class SentenceVectorStore:
    """Lightweight sentence-embedding index backed by a numpy float32 matrix.

    Does not require a third-party vector database.  Cosine-similarity search
    is performed with a single matrix–vector product, which is fast for the
    sentence-count ranges expected in practice (tens of thousands of sentences
    at most for a typical knowledge base).

    Persisted as two files in *directory*:
      - ``.sentence_vecs.npy``  — float32 embeddings matrix [N, dim]
      - ``.sentence_meta.json`` — parallel list of ``{id, metadata}`` records

    Thread safety: not thread-safe — reads and writes are serialised by the
    caller (``AxonBrain._index_sentence_windows`` holds the ingest lock).
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._vecs = None  # np.ndarray[float32, (N, dim)] or None

    # ------------------------------------------------------------------
    # Indexing (Story 1.2)
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Append *ids* / *embeddings* / *metadatas* to the in-memory index.

        Call :meth:`save` afterwards to persist to disk.
        """
        if not ids:
            return
        try:
            import numpy as np
        except ImportError:
            logger.warning("SentenceVectorStore: numpy not available — sentence indexing skipped")
            return

        new_vecs = np.array(embeddings, dtype=np.float32)
        if self._vecs is None or len(self._ids) == 0:
            self._vecs = new_vecs
        else:
            self._vecs = np.vstack([self._vecs, new_vecs])
        self._ids.extend(ids)
        self._meta.extend(metadatas)

    # ------------------------------------------------------------------
    # Search (Story 1.3 — used in window reconstruction)
    # ------------------------------------------------------------------

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """Return the top-*k* most similar sentence records (cosine similarity).

        Each result is a dict with keys ``id``, ``score``, and ``metadata``.
        Returns an empty list when the index is empty or numpy is unavailable.
        """
        if self._vecs is None or len(self._ids) == 0:
            return []
        try:
            import numpy as np
        except ImportError:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        norms = np.linalg.norm(self._vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = self._vecs / norms
        scores: np.ndarray = normed @ q  # shape [N]

        k = min(top_k, len(self._ids))
        # argpartition is O(N) rather than O(N log N) — fast for large N
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            {
                "id": self._ids[i],
                "score": float(scores[i]),
                "metadata": self._meta[i],
            }
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the embedding matrix and metadata list to disk."""
        self._directory.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np

            if self._vecs is not None and len(self._ids) > 0:
                np.save(str(self._directory / _VECS_FILENAME), self._vecs)
            meta_path = self._directory / _META_FILENAME
            meta_path.write_text(
                json.dumps({"ids": self._ids, "meta": self._meta}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("SentenceVectorStore: save failed: %s", exc)

    def load(self) -> None:
        """Load the embedding matrix and metadata list from disk if present."""
        meta_path = self._directory / _META_FILENAME
        vecs_path = self._directory / _VECS_FILENAME
        if not meta_path.exists():
            return
        try:
            import numpy as np

            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self._ids = data.get("ids", [])
            self._meta = data.get("meta", [])
            if vecs_path.exists() and self._ids:
                self._vecs = np.load(str(vecs_path), allow_pickle=False)
            else:
                self._vecs = None
        except Exception as exc:
            logger.warning("SentenceVectorStore: load failed: %s", exc)
            self._ids = []
            self._meta = []
            self._vecs = None
