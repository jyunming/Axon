"""
src/axon/vector_store.py

OpenVectorStore, MultiVectorStore, and MultiBM25Retriever extracted from main.py
for Phase 2 of the Axon refactor.
"""

import logging
from typing import Any

from axon.config import AxonConfig

logger = logging.getLogger("Axon")

_MERGED_VIEW_WRITE_ERROR = (
    "Cannot write to a merged parent project view. " "Switch to a specific sub-project first."
)


class OpenVectorStore:
    """Unified interface over ChromaDB, Qdrant, and LanceDB vector stores.

    Initialized via :class:`AxonConfig`.  Supports ``add``, ``search``,
    ``get_by_ids``, ``delete_by_ids``, and ``list_documents`` operations
    across all three backends.  Qdrant can operate in local file mode or
    remote mode (set ``qdrant_url`` in config).
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = config.vector_store
        self.client: Any = None
        self.collection: Any = None
        self._init_store()

    def _init_store(self):
        if self.provider == "chroma":
            import chromadb

            logger.info(f"Initializing ChromaDB: {self.config.vector_store_path}")
            try:
                self.client = chromadb.PersistentClient(path=self.config.vector_store_path)
                self.collection = self.client.get_or_create_collection(
                    name="axon", metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                # Catch the specific WSL/SQLite readonly error (code 8)
                if "(code: 8)" in str(e) and "readonly database" in str(e).lower():
                    msg = (
                        f"\n\n[bold red]ERROR:[/bold red] ChromaDB failed to initialize at: {self.config.vector_store_path}\n"
                        "This typically happens in WSL when using a Windows-mounted drive (/mnt/c/...). \n"
                        "SQLite does not support locking on these mounts.\n\n"
                        "FIX: Store your data in the Linux filesystem instead:\n"
                        "  1. Set environment variable: [bold]CHROMA_DATA_PATH=~/axon_data[/bold]\n"
                        "  2. Or edit config.yaml to use a path like: [bold]~/axon_data[/bold]\n"
                    )
                    from rich.console import Console

                    Console().print(msg)
                    raise RuntimeError(
                        f"ChromaDB failed to initialize at: {self.config.vector_store_path}"
                    )
                raise e
        elif self.provider == "qdrant":
            from qdrant_client import QdrantClient

            if getattr(self.config, "qdrant_url", ""):
                logger.info(f"Initializing Qdrant (remote): {self.config.qdrant_url}")
                self.client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key or None,
                )
            else:
                logger.info(f"Initializing Qdrant (local): {self.config.vector_store_path}")
                self.client = QdrantClient(path=self.config.vector_store_path)
        elif self.provider == "lancedb":
            import lancedb

            logger.info(f"Initializing LanceDB: {self.config.vector_store_path}")
            self.client = lancedb.connect(self.config.vector_store_path)
            try:
                self.collection = self.client.open_table("axon")
            except Exception:
                self.collection = None  # created lazily on first add()

    def close(self):
        """Release any open file handles or database connections."""
        if self.provider == "chroma" and self.client:
            # ChromaDB 0.4.x+ uses a persistent client that should be closed if supported
            if hasattr(self.client, "close"):
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None
            self.collection = None
        elif self.provider == "lancedb" and self.client:
            self.client = None
            self.collection = None
        elif self.provider == "qdrant" and self.client:
            self.client = None

    # Chroma hard limit per collection.add() call.
    _CHROMA_MAX_BATCH = 5000

    @staticmethod
    def _sanitize_chroma_meta(metadatas: list[dict] | None) -> list[dict] | None:
        """Coerce metadata dicts to Chroma-safe scalar types.

        Chroma only accepts ``str | int | float | bool`` per field.
        - list  → pipe-joined string (e.g. imports, calls, env_vars)
        - None  → omitted
        - other → str(v)
        """
        if metadatas is None:
            return None
        result: list[dict] = []
        for meta in metadatas:
            sanitized: dict = {}
            for k, v in meta.items():
                if isinstance(v, str | int | float | bool):
                    sanitized[k] = v
                elif isinstance(v, list):
                    if v:
                        sanitized[k] = "|".join(str(x) for x in v)
                    # empty list → omit
                elif v is not None:
                    sanitized[k] = str(v)
                # None → omit
            result.append(sanitized)
        return result

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] = None,
    ):
        if self.provider == "chroma":
            # Chroma enforces a hard per-call limit (~5461 rows). Slice into safe batches
            # so that large post-split payloads (e.g. long-contract corpora) do not crash.
            _bs = OpenVectorStore._CHROMA_MAX_BATCH
            _safe_meta = OpenVectorStore._sanitize_chroma_meta(metadatas)
            for _start in range(0, max(len(ids), 1), _bs):
                _end = _start + _bs
                _ids_b = ids[_start:_end]
                _texts_b = texts[_start:_end]
                _emb_b = embeddings[_start:_end]
                _meta_b = _safe_meta[_start:_end] if _safe_meta is not None else None
                if not _ids_b:
                    break
                try:
                    self.collection.add(
                        ids=_ids_b,
                        documents=_texts_b,
                        embeddings=_emb_b,
                        metadatas=_meta_b,
                    )
                except Exception as e:
                    if "dimension" in str(e).lower():
                        logger.error(
                            f"Embedding dimension mismatch in ChromaDB! Expected: {self.config.embedding_model}. "
                            "Try clearing the project data or switch to a different project."
                        )
                    raise
        elif self.provider == "qdrant":
            from qdrant_client.models import PointStruct

            # Build and upsert PointStructs incrementally per batch so peak
            # memory stays bounded to _CHROMA_MAX_BATCH rows, not O(N).
            _bs = OpenVectorStore._CHROMA_MAX_BATCH
            _n = max(len(ids), 1)
            for _start in range(0, _n, _bs):
                _end = _start + _bs
                _batch = [
                    PointStruct(
                        id=ids[i],
                        vector=embeddings[i],
                        payload={"text": texts[i], **(metadatas[i] if metadatas else {})},
                    )
                    for i in range(_start, min(_end, len(ids)))
                ]
                if not _batch:
                    break
                self.client.upsert(collection_name="axon", points=_batch)
        elif self.provider == "lancedb":
            import json

            rows = []
            for i, (doc_id, emb, text) in enumerate(zip(ids, embeddings, texts)):
                meta = metadatas[i] if metadatas else {}
                rows.append(
                    {
                        "id": doc_id,
                        "vector": emb,
                        "text": text,
                        "source": meta.get("source", ""),
                        "metadata_json": json.dumps(meta),
                    }
                )
            if self.collection is None:
                self.collection = self.client.create_table(
                    "axon", data=rows, mode="overwrite", metric="cosine"
                )
            else:
                self.collection.add(rows)

    def list_documents(self) -> list[dict[str, Any]]:
        """Return all unique source files stored in the knowledge base with chunk counts.

        Returns:
            List of dicts sorted by source name, each with keys:
                - source (str): The metadata 'source' value, or 'unknown' if not set.
                - chunks (int): Number of chunks stored for that source.
                - doc_ids (List[str]): All chunk IDs belonging to this source.
        """
        if self.provider == "chroma":
            result = self.collection.get(include=["metadatas"])
            sources: dict[str, dict[str, Any]] = {}
            for doc_id, meta in zip(
                result["ids"], result["metadatas"] or [{}] * len(result["ids"])
            ):
                source = (meta or {}).get("source", "unknown")
                if source not in sources:
                    sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
                sources[source]["chunks"] += 1
                sources[source]["doc_ids"].append(doc_id)
            return sorted(sources.values(), key=lambda x: x["source"])
        elif self.provider == "qdrant":
            results, _ = self.client.scroll(collection_name="axon", limit=10000, with_payload=True)
            sources: dict[str, dict[str, Any]] = {}
            for point in results:
                source = point.payload.get("source", "unknown")
                if source not in sources:
                    sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
                sources[source]["chunks"] += 1
                sources[source]["doc_ids"].append(str(point.id))
            return sorted(sources.values(), key=lambda x: x["source"])
        elif self.provider == "lancedb":
            if self.collection is None:
                return []
            rows = self.collection.to_arrow().to_pydict()
            sources: dict[str, dict[str, Any]] = {}
            for doc_id, source in zip(rows.get("id", []), rows.get("source", [])):
                src = source or "unknown"
                if src not in sources:
                    sources[src] = {"source": src, "chunks": 0, "doc_ids": []}
                sources[src]["chunks"] += 1
                sources[src]["doc_ids"].append(doc_id)
            return sorted(sources.values(), key=lambda x: x["source"])
        return []

    def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict = None
    ) -> list[dict]:
        if self.provider == "chroma":
            where = None
            if filter_dict:
                if len(filter_dict) == 1:
                    key, val = next(iter(filter_dict.items()))
                    where = {key: {"$eq": val}}
                else:
                    where = {"$and": [{k: {"$eq": v}} for k, v in filter_dict.items()]}
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where
            )
            return [
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                for i in range(len(results["ids"][0]))
            ]
        elif self.provider == "qdrant":
            results = self.client.search(
                collection_name="axon", query_vector=query_embedding, limit=top_k
            )
            return [
                {
                    "id": str(r.id),
                    "text": r.payload.get("text", ""),
                    "score": r.score,
                    "metadata": {k: v for k, v in r.payload.items() if k != "text"},
                }
                for r in results
            ]
        elif self.provider == "lancedb":
            import json

            if self.collection is None:
                return []
            results = self.collection.search(query_embedding).limit(top_k).to_list()
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "score": max(0.0, 1.0 - r.get("_distance", 1.0)),
                    "metadata": json.loads(r.get("metadata_json", "{}")),
                }
                for r in results
            ]

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetch stored documents by their IDs (used by GraphRAG expansion).

        Returns a list of result dicts in the same format as search(), with score=1.0
        since these docs are fetched by exact ID (not scored).
        """
        if not ids:
            return []
        if self.provider == "chroma":
            result = self.collection.get(ids=ids, include=["documents", "metadatas"])
            result_ids = result.get("ids") or []
            result_docs = result.get("documents") or []
            result_metas = result.get("metadatas") or []
            num_ids = len(result_ids)
            if len(result_docs) < num_ids:
                result_docs = list(result_docs) + [""] * (num_ids - len(result_docs))
            if len(result_metas) < num_ids:
                result_metas = list(result_metas) + [{}] * (num_ids - len(result_metas))
            docs = []
            for i in range(num_ids):
                docs.append(
                    {
                        "id": result_ids[i],
                        "text": result_docs[i] or "",
                        "score": 1.0,
                        "metadata": result_metas[i] or {},
                    }
                )
            return docs
        if self.provider == "qdrant":
            try:
                points = self.client.retrieve(
                    collection_name="axon",
                    ids=ids,
                    with_payload=True,
                )
                return [
                    {
                        "id": str(p.id),
                        "text": p.payload.get("text", ""),
                        "score": 1.0,
                        "metadata": {k: v for k, v in p.payload.items() if k != "text"},
                    }
                    for p in points
                ]
            except Exception as e:
                logger.debug(f"GraphRAG get_by_ids (Qdrant) failed: {e}")
                return []
        if self.provider == "lancedb":
            import json

            if self.collection is None:
                return []
            try:
                id_str = ", ".join(f"'{i}'" for i in ids)
                rows = self.collection.search().where(f"id IN ({id_str})", prefilter=True).to_list()
                return [
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "score": 1.0,
                        "metadata": json.loads(r.get("metadata_json", "{}")),
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.debug(f"GraphRAG get_by_ids (LanceDB) failed: {e}")
                return []
        return []

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete documents by ID from the vector store."""
        if not ids:
            return
        if self.provider == "chroma":
            self.collection.delete(ids=ids)
        elif self.provider == "qdrant":
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name="axon",
                points_selector=PointIdsList(points=ids),
            )
        elif self.provider == "lancedb":
            if self.collection is None:
                return
            id_str = ", ".join(f"'{i}'" for i in ids)
            self.collection.delete(f"id IN ({id_str})")


class MultiVectorStore:
    """Read-only fan-out wrapper over multiple OpenVectorStore instances.

    Used when a parent project is active: queries are dispatched to all
    descendant stores and the top-k results (by score) are returned merged.
    Writes are NOT supported — use the project's own OpenVectorStore for that.
    """

    def __init__(self, stores: list[OpenVectorStore]):
        self._stores = stores
        # Expose provider/collection from the first store so callers that
        # inspect brain.vector_store.provider / .collection still work.
        self.provider = stores[0].provider if stores else "chroma"
        self.collection = stores[0].collection if stores else None

    def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict = None
    ) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        seen: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(self._stores))) as ex:
            futures = [
                ex.submit(store.search, query_embedding, top_k, filter_dict)
                for store in self._stores
            ]
            for fut in futures:
                for doc in fut.result():
                    doc_id = doc["id"]
                    if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
                        seen[doc_id] = doc
        return sorted(seen.values(), key=lambda d: d["score"], reverse=True)[:top_k]

    def add(self, *args, **kwargs):
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for store in self._stores:
            for doc in store.list_documents():
                src = doc["source"]
                if src not in seen:
                    seen[src] = doc.copy()
                else:
                    seen[src]["chunks"] += doc["chunks"]
                    seen[src]["doc_ids"].extend(doc["doc_ids"])
        return sorted(seen.values(), key=lambda x: x["source"])

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        seen: dict[str, dict] = {}
        for store in self._stores:
            for doc in store.get_by_ids(ids):
                seen[doc["id"]] = doc
        return list(seen.values())

    def delete_by_ids(self, ids: list[str]) -> None:
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def delete_documents(self, ids: list[str]) -> None:
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)


class MultiBM25Retriever:
    """Read-only fan-out wrapper over multiple BM25Retriever instances.

    Merges BM25 results from all descendant stores and returns the top-k
    by score. Writes are NOT supported.
    """

    def __init__(self, retrievers: list) -> None:
        self._retrievers = retrievers

    def close(self):
        for r in self._retrievers:
            if hasattr(r, "close"):
                r.close()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        seen: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(self._retrievers))) as ex:
            futures = [ex.submit(r.search, query, top_k) for r in self._retrievers]
            for fut in futures:
                for doc in fut.result():
                    doc_id = doc["id"]
                    if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
                        seen[doc_id] = doc
        return sorted(seen.values(), key=lambda d: d["score"], reverse=True)[:top_k]

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Disallow deletes in merged read-only views."""
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def add_documents(self, *args, **kwargs):
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)
