"""Collection-level mutation helpers shared across API and REPL surfaces."""
from __future__ import annotations

import os
import pathlib
from typing import Any


def _call_optional(obj: Any, method_name: str) -> None:
    method = getattr(obj, method_name, None)
    if callable(method):
        method()


def clear_active_project(brain: Any) -> None:
    """Clear vector, retrieval, and graph state for the active project."""
    vs = brain.vector_store
    provider = getattr(vs, "provider", None)
    client = getattr(vs, "client", None)
    if provider == "chroma" and client is not None:
        client.delete_collection("axon")
        vs.collection = client.create_collection(name="axon", metadata={"hnsw:space": "cosine"})
    elif provider == "qdrant" and client is not None:
        try:
            client.delete_collection("axon")
        except Exception:
            pass
        vs._init_store()
    elif provider == "lancedb" and client is not None:
        try:
            client.drop_table("axon")
        except Exception:
            pass
        vs.collection = None
    bm25 = getattr(brain, "bm25", None)
    if bm25 is not None:
        corpus = getattr(bm25, "corpus", None)
        if corpus is not None:
            corpus.clear()
        bm25.bm25 = None
        _call_optional(bm25, "save")
    brain._ingested_hashes = set()
    _call_optional(brain, "_save_hash_store")
    brain._doc_versions = {}
    _call_optional(brain, "_save_doc_versions")
    # Clear + persist go through the graph backend (single source of truth
    # for "wipe graph state"; also correctly clears non-GraphRAG backends
    # like dynamic_graph, which the old field-by-field _save_* calls here
    # never touched — they always wrote GraphRagMixin's own state
    # regardless of which backend was actually active). persist=True asks
    # the backend to also write its now-empty state to disk, not just
    # reset in memory.
    graph_backend = getattr(brain, "_graph_backend", None)
    if graph_backend is not None and callable(getattr(graph_backend, "clear", None)):
        graph_backend.clear(persist=True)
    brain._code_graph = {"nodes": {}, "edges": []}
    _call_optional(brain, "_save_code_graph")
    # In-memory only, brain-owned (not GraphRAG state — see
    # GraphRagEngine._reset_graph_state()'s docstring for why it no longer
    # resets this). Clearing here keeps "wipe this project's data" complete.
    # Guarded by the same lock RAPTOR's own reads/writes/evictions use
    # (main.py's _summarise_window) — a lock-free reset here could race a
    # background summarization thread and silently drop or reintroduce an
    # entry right after a user-initiated wipe.
    if hasattr(brain, "_raptor_summary_cache"):
        raptor_lock = getattr(brain, "_raptor_cache_lock", None)
        if raptor_lock is not None:
            with raptor_lock:
                brain._raptor_summary_cache = {}
        else:
            brain._raptor_summary_cache = {}
    meta_path = getattr(brain, "_embedding_meta_path", None)
    if isinstance(meta_path, str | os.PathLike):
        path = pathlib.Path(meta_path)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
