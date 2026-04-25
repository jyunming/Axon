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
    brain._entity_graph = {}
    _call_optional(brain, "_save_entity_graph")
    brain._relation_graph = {}
    _call_optional(brain, "_save_relation_graph")
    brain._community_levels = {}
    _call_optional(brain, "_save_community_levels")
    brain._community_summaries = {}
    _call_optional(brain, "_save_community_summaries")
    brain._community_hierarchy = {}
    _call_optional(brain, "_save_community_hierarchy")
    brain._community_children = {}
    brain._community_graph_dirty = False
    brain._community_build_in_progress = False
    brain._claims_graph = {}
    _call_optional(brain, "_save_claims_graph")
    brain._entity_embeddings = {}
    _call_optional(brain, "_save_entity_embeddings")
    brain._entity_description_buffer = {}
    brain._relation_description_buffer = {}
    brain._text_unit_entity_map = {}
    brain._text_unit_relation_map = {}
    brain._raptor_summary_cache = {}
    brain._code_graph = {"nodes": {}, "edges": []}
    _call_optional(brain, "_save_code_graph")
    meta_path = getattr(brain, "_embedding_meta_path", None)
    if isinstance(meta_path, str | os.PathLike):
        path = pathlib.Path(meta_path)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
