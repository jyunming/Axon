import heapq
import json
import logging
import os
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore[assignment,misc]

from axon.rust_bridge import get_rust_bridge

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

    _JSONL_COMPACTION_THRESHOLD: int = 1000

    def __init__(
        self,
        storage_path: str = "./bm25_index",
        engine: str = "python",
        rust_fallback_enabled: bool = True,
    ):
        self.storage_path = storage_path
        self.corpus_file = os.path.join(storage_path, "bm25_corpus.json")
        self.corpus_file_zst = os.path.join(storage_path, "bm25_corpus.json.zst")
        self.corpus_file_msgpack = os.path.join(storage_path, "bm25_corpus.msgpack")
        self.corpus_file_msgpack_zst = os.path.join(storage_path, "bm25_corpus.msgpack.zst")
        self.corpus_file_log = os.path.join(storage_path, ".bm25_log.jsonl")
        self._compress_enabled = os.getenv("AXON_BM25_COMPRESS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            self._compress_min_bytes = int(os.getenv("AXON_BM25_COMPRESS_MIN_BYTES", "131072"))
        except ValueError:
            self._compress_min_bytes = 131072
        try:
            self._compress_level = int(os.getenv("AXON_BM25_COMPRESS_LEVEL", "6"))
        except ValueError:
            self._compress_level = 6
        self._numpy_topk_enabled = os.getenv("AXON_BM25_NUMPY_TOPK", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        _dedup_mode = os.getenv("AXON_BM25_CORPUS_DEDUP", "0").strip().lower()
        if _dedup_mode in {"1", "true", "yes", "on"}:
            self._corpus_dedup_enabled = True
        elif _dedup_mode in {"0", "false", "no", "off"}:
            self._corpus_dedup_enabled = False
        else:
            # auto: preserve fast zstd load behavior; apply dedup for JSON storage.
            self._corpus_dedup_enabled = not self._compress_enabled
        self._corpus_dedup_fast_load_enabled = os.getenv(
            "AXON_BM25_CORPUS_DEDUP_FAST_LOAD", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._corpus_dedup_lazy_load_enabled = os.getenv(
            "AXON_BM25_CORPUS_DEDUP_LAZY_LOAD", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._text_intern_mode = os.getenv("AXON_BM25_TEXT_INTERN", "auto").strip().lower()
        if self._text_intern_mode in {"1", "true", "yes", "on"}:
            self._text_intern_mode = "on"
        elif self._text_intern_mode in {"0", "false", "no", "off"}:
            self._text_intern_mode = "off"
        elif self._text_intern_mode != "auto":
            self._text_intern_mode = "auto"
        try:
            self._text_intern_auto_min_docs = int(
                os.getenv("AXON_BM25_TEXT_INTERN_AUTO_MIN_DOCS", "2000")
            )
        except ValueError:
            self._text_intern_auto_min_docs = 2000
        try:
            self._text_intern_auto_min_dup_ratio = float(
                os.getenv("AXON_BM25_TEXT_INTERN_AUTO_MIN_DUP_RATIO", "0.1")
            )
        except ValueError:
            self._text_intern_auto_min_dup_ratio = 0.1
        self.bm25 = None
        self._rust_index = None
        self._rust = get_rust_bridge()
        self._rust_fallback_enabled = rust_fallback_enabled
        self._bm25_backend = "python"
        self._orjson = None
        try:
            import orjson  # type: ignore

            self._orjson = orjson
        except Exception:
            self._orjson = None
        self._dirty: bool = False  # True when corpus was modified but index not yet rebuilt
        self._dedup_payload: dict[str, Any] | None = None
        self._dedup_doc_count: int = 0
        self.corpus: list[
            dict[str, Any]
        ] = []  # List of dicts: {'id': id, 'text': text, 'metadata': meta}
        self._pending_log_entries: list[dict] = []
        if engine == "rust":
            if self._rust.can_bm25():
                self._bm25_backend = "rust"
            elif not rust_fallback_enabled:
                raise RuntimeError(
                    "BM25 engine is set to 'rust' but Rust BM25 capability is unavailable."
                )
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        self.load()

    def close(self):
        """Release index and corpus references."""
        self.bm25 = None
        self._rust_index = None
        self._dedup_payload = None
        self._dedup_doc_count = 0
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
        self._ensure_corpus_materialized()
        if self._text_intern_mode != "off":
            self._intern_document_texts(documents)
        self.corpus.extend(documents)
        self._pending_log_entries.extend(documents)
        self._dirty = True  # index will be rebuilt lazily on next search()
        if not save_deferred:
            self.save()

    def _ensure_corpus_materialized(self) -> None:
        if self._dedup_payload is None:
            return
        payload = self._dedup_payload
        if self._corpus_dedup_fast_load_enabled:
            self.corpus = self._decode_loaded_corpus_fast(payload)
        else:
            self.corpus = self._decode_loaded_corpus(payload)
        if self._text_intern_mode != "off":
            self._intern_document_texts(self.corpus)
        self._dedup_payload = None
        self._dedup_doc_count = 0

    def _intern_document_texts(self, docs: list[dict[str, Any]]) -> None:
        """Canonicalize repeated text values to a shared string object."""
        if not docs:
            return
        if self._text_intern_mode == "off":
            return
        if self._text_intern_mode == "auto":
            min_docs = max(1, int(self._text_intern_auto_min_docs))
            if len(docs) < min_docs:
                return
            sample_size = min(len(docs), 1024)
            sample = docs[:sample_size]
            texts = [d.get("text") for d in sample if isinstance(d, dict)]
            text_count = len(texts)
            if text_count < 2:
                return
            unique_count = len({t for t in texts if isinstance(t, str)})
            dup_ratio = 1.0 - (unique_count / max(text_count, 1))
            if dup_ratio < float(self._text_intern_auto_min_dup_ratio):
                return
        pool: dict[str, str] = {}
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            text = doc.get("text")
            if not isinstance(text, str):
                continue
            shared = pool.get(text)
            if shared is None:
                pool[text] = text
                shared = text
            doc["text"] = shared

    def _rebuild_index(self) -> None:
        """Rebuild BM25Okapi from the current corpus and clear the dirty flag."""
        self._ensure_corpus_materialized()
        if not self.corpus:
            self.bm25 = None
            self._rust_index = None
            self._dirty = False
            return
        if self._bm25_backend == "rust":
            rust_index = self._rust.build_bm25_index(self.corpus)
            if rust_index is not None:
                self._rust_index = rust_index
                self.bm25 = None
                self._dirty = False
                return
            if not self._rust_fallback_enabled:
                raise RuntimeError("Rust BM25 index build failed and fallback is disabled.")
            logger.warning("Rust BM25 build failed; falling back to Python BM25.")
            self._bm25_backend = "python"
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._rust_index = None
        self._dirty = False

    def flush(self) -> None:
        """Explicitly save corpus — call after deferred batch ingest session.
        If pending log entries are below the compaction threshold, appends them
        to the JSONL delta log instead of rewriting the full corpus file.
        Falls back to a full ``save()`` when the threshold is reached or when
        no incremental entries are pending.
        """
        use_log = (
            bool(self._pending_log_entries)
            and len(self._pending_log_entries) < self._JSONL_COMPACTION_THRESHOLD
        )
        if use_log:
            try:
                with open(self.corpus_file_log, "a", encoding="utf-8") as lf:
                    for entry in self._pending_log_entries:
                        lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logger.info(
                    "💾 BM25 log appended %d docs to %s",
                    len(self._pending_log_entries),
                    self.corpus_file_log,
                )
                self._pending_log_entries.clear()
                return
            except Exception as e:
                logger.warning("BM25 log append failed; falling back to full save: %s", e)
        # Full save: no pending entries, threshold exceeded, or log write failed
        self.save()
        try:
            if os.path.exists(self.corpus_file_log):
                os.remove(self.corpus_file_log)
        except OSError:
            pass
        self._pending_log_entries.clear()

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search the BM25 index, rebuilding it first if the corpus was modified."""
        if self._dedup_payload is not None and self._dirty:
            # Ensure query path sees full docs before scoring.
            self._ensure_corpus_materialized()
        if self._dirty:
            self._rebuild_index()
        if not self.corpus:
            return []
        if self._bm25_backend == "rust":
            if self._rust_index is None:
                return []
            rust_scores = self._rust.search_bm25(self._rust_index, query=query, top_k=top_k)
            if rust_scores is None:
                if not self._rust_fallback_enabled:
                    raise RuntimeError("Rust BM25 search failed and fallback is disabled.")
                logger.warning("Rust BM25 search failed; falling back to Python BM25.")
                self._bm25_backend = "python"
                self._rebuild_index()
            else:
                results: list[dict[str, Any]] = []
                for item in rust_scores:
                    idx = item.get("index")
                    if not isinstance(idx, int) or idx < 0 or idx >= len(self.corpus):
                        continue
                    score = float(item.get("score", 0.0))
                    if score <= 0:
                        continue
                    doc = self.corpus[idx].copy()
                    doc["score"] = score
                    results.append(doc)
                return results[:top_k]
        if self.bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = None
        if self._numpy_topk_enabled:
            try:
                import numpy as np

                arr = np.asarray(scores)
                n = int(arr.size)
                k = min(int(top_k), n)
                if k > 0:
                    # Use argpartition for O(n) selection then sort only top-k.
                    idx = np.argpartition(arr, -k)[-k:]
                    idx = idx[np.argsort(arr[idx])[::-1]]
                    top_indices = idx.tolist()
            except Exception:
                top_indices = None
        if top_indices is None:
            # Fallback: Get top-k indices efficiently using a heap
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
        self._ensure_corpus_materialized()
        original_count = len(self.corpus)
        self.corpus = [doc for doc in self.corpus if doc["id"] not in doc_ids]
        if len(self.corpus) < original_count:
            self._dirty = True  # rebuild deferred to next search()
            # Log is now stale after deletion — force a full rewrite on next save
            try:
                if os.path.exists(self.corpus_file_log):
                    os.remove(self.corpus_file_log)
            except OSError:
                pass
            self._pending_log_entries.clear()
            self.save()

    def save(self):
        """Save corpus to disk (msgpack.zst preferred, then JSON or JSON.zst)."""
        self._ensure_corpus_materialized()
        # Fast-path: msgpack encode + optional zstd compress
        if self._corpus_dedup_enabled and self._rust.can_corpus_msgpack():
            payload_obj = self._build_dedup_corpus_payload()
            raw_mp = self._rust.encode_corpus_msgpack(payload_obj["texts"], payload_obj["docs"])
            if raw_mp is not None:
                try:
                    import zstandard as zstd

                    cctx = zstd.ZstdCompressor(level=self._compress_level)
                    tmp = self.corpus_file_msgpack_zst + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(cctx.compress(raw_mp))
                    os.replace(tmp, self.corpus_file_msgpack_zst)
                    # Clean up other corpus files
                    for old in (
                        self.corpus_file_msgpack,
                        self.corpus_file_zst,
                        self.corpus_file,
                    ):
                        try:
                            if os.path.exists(old):
                                os.remove(old)
                        except OSError:
                            pass
                    logger.info(
                        "💾 BM25 corpus saved to %s (msgpack+zst)", self.corpus_file_msgpack_zst
                    )
                    return
                except ImportError:
                    # zstd not available — save uncompressed msgpack
                    tmp = self.corpus_file_msgpack + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(raw_mp)
                    os.replace(tmp, self.corpus_file_msgpack)
                    for old in (self.corpus_file_zst, self.corpus_file):
                        try:
                            if os.path.exists(old):
                                os.remove(old)
                        except OSError:
                            pass
                    logger.info("💾 BM25 corpus saved to %s (msgpack)", self.corpus_file_msgpack)
                    return
                except Exception as e:
                    logger.warning("BM25 msgpack save failed; falling back to JSON: %s", e)
        payload_obj = (
            self._build_dedup_corpus_payload() if self._corpus_dedup_enabled else self.corpus
        )
        payload_bytes = self._json_dumps_bytes(payload_obj)
        use_zst = self._compress_enabled and len(payload_bytes) >= max(0, self._compress_min_bytes)
        if use_zst:
            try:
                import zstandard as zstd

                cctx = zstd.ZstdCompressor(level=self._compress_level)
                tmp_zst = self.corpus_file_zst + ".tmp"
                with open(tmp_zst, "wb") as f:
                    f.write(cctx.compress(payload_bytes))
                os.replace(tmp_zst, self.corpus_file_zst)
                # Best-effort cleanup of uncompressed corpus to avoid duplicate storage.
                try:
                    if os.path.exists(self.corpus_file):
                        os.remove(self.corpus_file)
                except OSError:
                    pass
                logger.info(f"💾 BM25 corpus saved to {self.corpus_file_zst} (compressed)")
                return
            except Exception as e:
                logger.warning("BM25 compression unavailable; falling back to JSON save: %s", e)
        tmp_file = self.corpus_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(payload_bytes.decode("utf-8"))
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

    def _json_dumps_bytes(self, payload: Any) -> bytes:
        if self._orjson is not None:
            return self._orjson.dumps(payload, option=self._orjson.OPT_NON_STR_KEYS)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _json_loads(self, data: bytes | str) -> Any:
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data
        if self._orjson is not None:
            return self._orjson.loads(data_bytes)
        return json.loads(data_bytes.decode("utf-8"))

    def _build_dedup_corpus_payload(self) -> dict[str, Any]:
        """Build a compact JSON payload that deduplicates repeated text bodies."""
        # Try Rust fast-path first
        if self._rust.can_dedup_corpus_payload():
            result = self._rust.build_dedup_corpus_payload(self.corpus)
            if result is not None:
                texts, docs = result
                return {"format": "dedup_v1", "texts": texts, "docs": docs}
        texts: list[str] = []
        text_to_idx: dict[str, int] = {}
        docs: list[dict[str, Any]] = []
        for doc in self.corpus:
            text = str(doc.get("text", ""))
            idx = text_to_idx.get(text)
            if idx is None:
                idx = len(texts)
                text_to_idx[text] = idx
                texts.append(text)
            docs.append(
                {
                    "id": str(doc.get("id", "")),
                    "t": idx,
                    "metadata": doc.get("metadata", {}),
                }
            )
        return {"format": "dedup_v1", "texts": texts, "docs": docs}

    @staticmethod
    def _decode_loaded_corpus(payload: Any) -> list[dict[str, Any]]:
        """Decode either legacy corpus list or dedup_v1 corpus payload."""
        if isinstance(payload, list):
            # Legacy format: list[{"id","text","metadata"}]
            out: list[dict[str, Any]] = []
            for doc in payload:
                if not isinstance(doc, dict):
                    continue
                out.append(
                    {
                        "id": str(doc.get("id", "")),
                        "text": str(doc.get("text", "")),
                        "metadata": doc.get("metadata", {}),
                    }
                )
            return out
        if not isinstance(payload, dict):
            return []
        if payload.get("format") != "dedup_v1":
            return []
        texts_raw = payload.get("texts")
        docs_raw = payload.get("docs")
        if not isinstance(texts_raw, list) or not isinstance(docs_raw, list):
            return []
        texts = [str(t) for t in texts_raw]
        out: list[dict[str, Any]] = []
        for d in docs_raw:
            if not isinstance(d, dict):
                continue
            text_idx = d.get("t")
            if not isinstance(text_idx, int) or text_idx < 0 or text_idx >= len(texts):
                continue
            out.append(
                {
                    "id": str(d.get("id", "")),
                    "text": texts[text_idx],
                    "metadata": d.get("metadata", {}),
                }
            )
        return out

    def _decode_loaded_corpus_fast(self, payload: Any) -> list[dict[str, Any]]:
        """Fast-path decode for dedup_v1 payload with minimal validation."""
        if isinstance(payload, list):
            return payload  # legacy path already materialized as required shape
        if not isinstance(payload, dict) or payload.get("format") != "dedup_v1":
            return self._decode_loaded_corpus(payload)
        texts = payload.get("texts")
        docs = payload.get("docs")
        if not isinstance(texts, list) or not isinstance(docs, list):
            return self._decode_loaded_corpus(payload)
        try:
            # Trust writer schema for speed; fallback to strict path on first error.
            return [
                {"id": d["id"], "text": texts[d["t"]], "metadata": d.get("metadata", {})}
                for d in docs
            ]
        except Exception:
            return self._decode_loaded_corpus(payload)

    def _load_msgpack_corpus(self, raw_mp: bytes) -> list[dict] | None:
        """Decode msgpack bytes to corpus list via Rust bridge."""
        result = self._rust.decode_corpus_msgpack(raw_mp)
        return result

    def _replay_jsonl_log(self) -> None:
        """Replay the JSONL delta log into self.corpus if it exists.
        Called at the end of every successful :meth:`load` branch so that
        incremental entries written by :meth:`flush` are merged back into the
        corpus.  Compacts automatically when the log exceeds
        ``_JSONL_COMPACTION_THRESHOLD``.
        """
        log_path = self.corpus_file_log
        if not os.path.exists(log_path):
            return
        try:
            # Ensure lazy-loaded dedup payload is fully materialised first so
            # appended log docs are not overwritten by a later materialisation.
            self._ensure_corpus_materialized()
            replayed = 0
            with open(log_path, encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        if isinstance(doc, dict) and doc.get("id"):
                            self.corpus.append(doc)
                            replayed += 1
                    except Exception:
                        pass  # skip corrupted lines
            if replayed > 0:
                logger.info("BM25 log: replayed %d incremental docs from %s", replayed, log_path)
                self._dirty = True
                if replayed >= self._JSONL_COMPACTION_THRESHOLD:
                    logger.info("BM25 log: compacting (replayed >= threshold)")
                    self.save()
                    try:
                        os.remove(log_path)
                    except OSError:
                        pass
        except Exception as e:
            logger.warning("BM25 log replay failed: %s", e)

    def load(self):
        """Load corpus from msgpack.zst (preferred), then json.zst, msgpack, or json."""
        try:
            # Priority 1: msgpack.zst (fastest)
            if os.path.exists(self.corpus_file_msgpack_zst):
                try:
                    import zstandard as zstd

                    with open(self.corpus_file_msgpack_zst, "rb") as f:
                        raw = f.read()
                    dctx = zstd.ZstdDecompressor()
                    raw_mp = dctx.decompress(raw)
                    corpus = self._load_msgpack_corpus(raw_mp)
                    if corpus is not None:
                        self.corpus = corpus
                        self.bm25 = None
                        self._rust_index = None
                        self._dirty = bool(self.corpus)
                        self._dedup_payload = None
                        self._dedup_doc_count = 0
                        if self._text_intern_mode != "off" and self.corpus:
                            self._intern_document_texts(self.corpus)
                        logger.info(
                            "📂 Loaded BM25 corpus with %d documents (msgpack+zst)",
                            len(self.corpus),
                        )
                        self._replay_jsonl_log()
                        return
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning("Failed to load msgpack.zst BM25 corpus: %s", e)
            # Priority 2: json.zst
            if os.path.exists(self.corpus_file_zst):
                try:
                    import zstandard as zstd

                    with open(self.corpus_file_zst, "rb") as f:
                        raw = f.read()
                    dctx = zstd.ZstdDecompressor()
                    raw_json = dctx.decompress(raw)
                    # Try Rust JSON decode first
                    if self._rust.can_decode_corpus_json():
                        rust_corpus = self._rust.decode_corpus_json(raw_json)
                        if rust_corpus is not None:
                            self.corpus = rust_corpus
                            self.bm25 = None
                            self._rust_index = None
                            self._dirty = bool(self.corpus)
                            self._dedup_payload = None
                            self._dedup_doc_count = 0
                            if self._text_intern_mode != "off" and self.corpus:
                                self._intern_document_texts(self.corpus)
                            logger.info(
                                "📂 Loaded BM25 corpus with %d documents (rust json+zst)",
                                len(self.corpus),
                            )
                            self._replay_jsonl_log()
                            return
                    payload = self._json_loads(raw_json)
                    if (
                        self._corpus_dedup_lazy_load_enabled
                        and isinstance(payload, dict)
                        and payload.get("format") == "dedup_v1"
                    ):
                        self._dedup_payload = payload
                        docs = payload.get("docs")
                        self._dedup_doc_count = len(docs) if isinstance(docs, list) else 0
                        self.corpus = []
                    elif self._corpus_dedup_fast_load_enabled:
                        self.corpus = self._decode_loaded_corpus_fast(payload)
                    else:
                        self.corpus = self._decode_loaded_corpus(payload)
                    self.bm25 = None
                    self._rust_index = None
                    self._dirty = bool(self.corpus) or bool(self._dedup_doc_count)
                    if self._text_intern_mode != "off" and self.corpus:
                        self._intern_document_texts(self.corpus)
                    logger.info(
                        "📂 Loaded BM25 corpus with %d documents (compressed)",
                        len(self.corpus) if self._dedup_doc_count == 0 else self._dedup_doc_count,
                    )
                    self._replay_jsonl_log()
                    return
                except Exception as e:
                    logger.warning("Failed to load compressed BM25 corpus: %s", e)
            # Priority 3: uncompressed msgpack
            if os.path.exists(self.corpus_file_msgpack):
                try:
                    with open(self.corpus_file_msgpack, "rb") as f:
                        raw_mp = f.read()
                    corpus = self._load_msgpack_corpus(raw_mp)
                    if corpus is not None:
                        self.corpus = corpus
                        self.bm25 = None
                        self._rust_index = None
                        self._dirty = bool(self.corpus)
                        self._dedup_payload = None
                        self._dedup_doc_count = 0
                        if self._text_intern_mode != "off" and self.corpus:
                            self._intern_document_texts(self.corpus)
                        logger.info(
                            "📂 Loaded BM25 corpus with %d documents (msgpack)",
                            len(self.corpus),
                        )
                        self._replay_jsonl_log()
                        return
                except Exception as e:
                    logger.warning("Failed to load msgpack BM25 corpus: %s", e)
            # Priority 4: plain JSON (with Rust fast-path)
            if os.path.exists(self.corpus_file):
                with open(self.corpus_file, "rb") as f:
                    raw_bytes = f.read()
                # Try Rust JSON decode first (5–10× faster than Python for large corpora)
                if self._rust.can_decode_corpus_json():
                    rust_corpus = self._rust.decode_corpus_json(raw_bytes)
                    if rust_corpus is not None:
                        self.corpus = rust_corpus
                        self.bm25 = None
                        self._rust_index = None
                        self._dirty = bool(self.corpus)
                        self._dedup_payload = None
                        self._dedup_doc_count = 0
                        if self._text_intern_mode != "off" and self.corpus:
                            self._intern_document_texts(self.corpus)
                        logger.info(
                            "📂 Loaded BM25 corpus with %d documents (rust json)", len(self.corpus)
                        )
                        self._replay_jsonl_log()
                        return
                payload = self._json_loads(raw_bytes)
                if (
                    self._corpus_dedup_lazy_load_enabled
                    and isinstance(payload, dict)
                    and payload.get("format") == "dedup_v1"
                ):
                    self._dedup_payload = payload
                    docs = payload.get("docs")
                    self._dedup_doc_count = len(docs) if isinstance(docs, list) else 0
                    self.corpus = []
                elif self._corpus_dedup_fast_load_enabled:
                    self.corpus = self._decode_loaded_corpus_fast(payload)
                else:
                    self.corpus = self._decode_loaded_corpus(payload)
                self.bm25 = None
                self._rust_index = None
                self._dirty = bool(self.corpus) or bool(self._dedup_doc_count)
                if self._text_intern_mode != "off" and self.corpus:
                    self._intern_document_texts(self.corpus)
                n_docs = len(self.corpus) if self._dedup_doc_count == 0 else self._dedup_doc_count
                logger.info(f"📂 Loaded BM25 corpus with {n_docs} documents")
                self._replay_jsonl_log()
        except Exception as e:
            logger.error(f"Failed to load BM25 corpus: {e}")
            self.corpus = []
            self.bm25 = None
            self._rust_index = None
            self._dedup_payload = None
            self._dedup_doc_count = 0


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
    # Try Rust fast-path
    _rust = get_rust_bridge()
    if _rust.can_score_fusion():
        result = _rust.score_fusion_weighted(vector_results, bm25_results, weight)
        if result is not None:
            return result
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
    # Try Rust fast-path
    _rust = get_rust_bridge()
    if _rust.can_score_fusion():
        result = _rust.score_fusion_rrf(vector_results, bm25_results, k)
        if result is not None:
            return result
    fused_scores: dict[str, float] = {}
    fused_only_ids: set[str] = set()
    # Preserve original cosine scores keyed by doc_id
    vector_scores = {doc["id"]: doc.get("score", 0.0) for doc in vector_results}
    # RRF formula requires 1-indexed rank: score = 1 / (k + rank)
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 1 / (k + rank)
            fused_only_ids.add(doc_id)
        else:
            fused_scores[doc_id] += 1 / (k + rank)
    # Use shallow copies to avoid mutating the caller's input lists
    all_docs = {doc["id"]: dict(doc) for doc in vector_results}
    for doc in bm25_results:
        if doc["id"] not in all_docs:
            all_docs[doc["id"]] = dict(doc)
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    final_results = []
    for doc_id in sorted_ids:
        doc = all_docs[doc_id]
        doc["score"] = fused_scores[doc_id]
        if doc_id in fused_only_ids:
            doc["fused_only"] = True
        # Expose the original cosine similarity for display purposes
        if doc_id in vector_scores:
            doc["vector_score"] = vector_scores[doc_id]
        final_results.append(doc)
    return final_results
