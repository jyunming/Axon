"""GraphRagEngine — owns all GraphRAG entity/relation/community state and logic.

This is the ownership-inversion half of the M2 backend-boundary refactor:
``AxonBrain`` no longer inherits :class:`~axon.graph_rag.GraphRagMixin`
directly. Instead, :class:`GraphRagBackend` composes a ``GraphRagEngine``
instance (``self._engine``) that *does* inherit ``GraphRagMixin`` — every one
of its ~90 methods keeps its original body unchanged, since ``self`` inside
those methods is now this engine rather than the brain.

Two kinds of state a ``GraphRagMixin`` method reads are handled differently:

- Graph state (``_entity_graph``, locks, caches, ...) is owned directly on
  this instance via ``GraphRagMixin``'s own lazy-init ``@property`` pattern —
  no proxying needed, it just works once the eager-init attributes below
  exist on ``self``.
- A handful of resources are genuinely brain-owned (the LLM, embedding
  model, vector store, shared executor, ...) — those are proxied via
  ``@property`` back onto the attached brain.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import TYPE_CHECKING, Any

from axon._lru_ttl_cache import lru_ttl_get, lru_ttl_put
from axon.graph_backends.base import IngestResult
from axon.graph_rag import GraphRagMixin

if TYPE_CHECKING:
    from axon.main import AxonBrain

logger = logging.getLogger("Axon")


class GraphRagEngine(GraphRagMixin):
    """Owns GraphRAG entity/relation/community graph state and logic.

    Composed into :class:`~axon.graph_backends.graphrag_backend.GraphRagBackend`
    as ``self._engine``.
    """

    def __init__(self, brain: AxonBrain) -> None:
        self._brain = brain
        # Eagerly initialise state that GraphRagMixin's @property getters would
        # otherwise create lazily on first access. Lazy init has a TOCTOU race
        # under concurrent first-access (two threads both pass the hasattr
        # check, both create a new lock/dict, one overwrites the other —
        # leaving threads holding "different" objects for the same critical
        # section). Doing it once here, single-threaded in __init__, removes
        # the race — mirrors the identical rationale previously documented
        # inline in AxonBrain.__init__ before this state moved here.
        self._graph_lock_internal: threading.RLock = threading.RLock()
        # Dedicated leaf lock for the GraphRAG LLM/extraction cache
        # (_gr_cache_get/_gr_cache_put). Deliberately NOT _graph_lock: those
        # cache helpers are called from worker threads dispatched by
        # self._executor.map/.submit (community summarization, claim
        # extraction, description canonicalization) while the calling thread
        # may already hold _graph_lock — an RLock only reenters for the
        # *same* thread, so a worker thread blocks forever waiting on a lock
        # the dispatching thread still holds. Keeping the cache lock separate
        # and strictly leaf-level (never acquired while holding _graph_lock
        # or _traversal_cache_lock) avoids that deadlock class entirely.
        self._gr_cache_lock_internal: threading.Lock = threading.Lock()
        self._traversal_cache_lock_internal: threading.Lock = threading.Lock()
        self._entity_token_index_internal: dict[str, set[str]] = {}
        self._pending_persist_futures_internal: list[concurrent.futures.Future] = []
        self._persist_executor_internal: concurrent.futures.ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Proxies onto brain-owned resources GraphRagMixin methods read directly.
    # ------------------------------------------------------------------
    @property
    def config(self):
        return self._brain.config

    @property
    def llm(self):
        return self._brain.llm

    @property
    def embedding(self):
        return self._brain.embedding

    @property
    def vector_store(self):
        return self._brain.vector_store

    @property
    def _own_vector_store(self):
        return self._brain._own_vector_store

    @property
    def _executor(self):
        return self._brain._executor

    @property
    def _community_rebuild_lock(self):
        return self._brain._community_rebuild_lock

    # Source-policy tables are QueryRouterMixin class attributes; AxonBrain
    # resolves them via MRO, but this engine doesn't inherit QueryRouterMixin,
    # so ingest_chunks() (which reads them for GraphRAG-eligibility filtering)
    # needs an explicit proxy.
    @property
    def _SOURCE_POLICY(self):
        return self._brain._SOURCE_POLICY

    @property
    def _SOURCE_POLICY_DEFAULT(self):
        return self._brain._SOURCE_POLICY_DEFAULT

    def _assert_write_allowed(self, operation: str = "write") -> None:
        self._brain._assert_write_allowed(operation)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """(Re)load all GraphRAG state from disk.

        Called once right after construction (mirroring AxonBrain.__init__'s
        old inline load block) and again on every switch_project() (mirroring
        switch_project's old inline reload block) — same _load_* calls, same
        order, in both cases. Also owns the transient/derived state that
        needs resetting alongside a (re)load: the traversal cache,
        _community_build_in_progress, and _last_matched_entities.
        """
        self._graph_rag_cache: dict = self._load_graph_rag_extraction_cache()
        self._graph_rag_cache_dirty: bool = False
        self._entity_graph: dict = self._load_entity_graph()
        self._rebuild_entity_token_index()
        self._relation_graph: dict = self._load_relation_graph()
        self._community_levels: dict = self._load_community_levels()
        self._community_summaries: dict = self._load_community_summaries()
        self._community_graph_dirty: bool = False
        self._community_build_in_progress: bool = False
        self._last_matched_entities: list = []
        self._entity_embeddings: dict = self._load_entity_embeddings()
        self._entity_description_buffer: dict = {}
        self._claims_graph: dict = self._load_claims_graph()
        self._community_hierarchy: dict[int, int] = self._load_community_hierarchy()
        self._community_children: dict[int, list[int]] = {}
        for child_cid, parent_cid in self._community_hierarchy.items():
            if parent_cid is not None:
                if parent_cid not in self._community_children:
                    self._community_children[parent_cid] = []
                if child_cid not in self._community_children[parent_cid]:
                    self._community_children[parent_cid].append(child_cid)
        self._relation_description_buffer: dict = {}
        self._text_unit_entity_map: dict[str, list[str]] = {}
        self._text_unit_relation_map: dict[str, list[tuple[str, str]]] = {}
        with self._traversal_cache_lock:
            self._traversal_cache.clear()

    def merge_descendants(self, descendants: list[str]) -> None:
        """Merge entity/relation/embedding/claims/community-summary state from
        each descendant project's own graph files into this (parent)
        project's in-memory graph state, so GraphRAG expansion from a parent
        project sees data ingested into its sub-projects too.

        Absorbed verbatim from switch_project()'s old inline descendant-merge
        block. Callers must only invoke this when *descendants* is non-empty.
        """
        import json as _json
        import pathlib

        from axon.projects import project_bm25_path
        from axon.rust_bridge import get_rust_bridge

        bridge = get_rust_bridge()
        with self._graph_lock:
            for desc in descendants:
                desc_bm25_path = project_bm25_path(desc)
                desc_base = pathlib.Path(desc_bm25_path)
                # --- entity graph ---
                desc_graph_path = desc_base / ".entity_graph.json"
                desc_mp_path = desc_base / ".entity_graph.msgpack"
                raw = None
                if desc_mp_path.exists() and bridge.can_entity_graph_codec():
                    try:
                        raw = bridge.decode_entity_graph(desc_mp_path.read_bytes())
                    except Exception:
                        raw = None
                if raw is None and desc_graph_path.exists():
                    try:
                        raw = _json.loads(desc_graph_path.read_text(encoding="utf-8"))
                    except Exception:
                        raw = None
                if isinstance(raw, dict):
                    for entity, node in raw.items():
                        if not isinstance(entity, str):
                            continue
                        if not isinstance(node, dict):
                            continue
                        doc_ids = node.get("chunk_ids", [])
                        if not doc_ids:
                            continue
                        existing = self._entity_graph.get(entity)
                        if existing is None:
                            self._entity_graph[entity] = {
                                "description": node.get("description", ""),
                                "type": node.get("type", "UNKNOWN"),
                                "chunk_ids": [d for d in doc_ids if isinstance(d, str)],
                                "frequency": len([d for d in doc_ids if isinstance(d, str)]),
                                "degree": node.get("degree", 0),
                            }
                            self._token_index_add(entity)
                        elif isinstance(existing, dict):
                            existing_ids = set(existing.get("chunk_ids", []))
                            new_ids = [
                                d for d in doc_ids if isinstance(d, str) and d not in existing_ids
                            ]
                            if new_ids:
                                existing.setdefault("chunk_ids", []).extend(new_ids)
                                existing["frequency"] = len(existing["chunk_ids"])
                # --- relation graph ---
                desc_rel_path = desc_base / ".relation_graph.json"
                desc_rel_mp_path = desc_base / ".relation_graph.msgpack"
                raw_rel = None
                if desc_rel_mp_path.exists() and bridge.can_relation_graph_codec():
                    try:
                        raw_rel = bridge.decode_relation_graph(desc_rel_mp_path.read_bytes())
                    except Exception:
                        raw_rel = None
                if raw_rel is None and desc_rel_path.exists():
                    try:
                        raw_rel = _json.loads(desc_rel_path.read_text(encoding="utf-8"))
                    except Exception:
                        raw_rel = None
                if isinstance(raw_rel, dict):
                    for src, entries in raw_rel.items():
                        if isinstance(src, str) and isinstance(entries, list):
                            if src not in self._relation_graph:
                                self._relation_graph[src] = []
                            existing_keys = {
                                (e.get("target"), e.get("relation"), e.get("chunk_id"))
                                for e in self._relation_graph[src]
                            }
                            for entry in entries:
                                if isinstance(entry, dict):
                                    key = (
                                        entry.get("target"),
                                        entry.get("relation"),
                                        entry.get("chunk_id"),
                                    )
                                    if key not in existing_keys:
                                        self._relation_graph[src].append(entry)
                                        existing_keys.add(key)
                # --- entity embeddings ---
                desc_emb_path = desc_base / ".entity_embeddings.json"
                if desc_emb_path.exists():
                    try:
                        raw = _json.loads(desc_emb_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for entity_key, embedding in raw.items():
                                if (
                                    isinstance(entity_key, str)
                                    and entity_key not in self._entity_embeddings
                                ):
                                    self._entity_embeddings[entity_key] = embedding
                    except Exception as e:
                        logger.warning(f"Could not merge entity embeddings for '{desc}': {e}")
                # --- claims ---
                desc_claims_path = desc_base / ".claims_graph.json"
                if desc_claims_path.exists():
                    try:
                        raw = _json.loads(desc_claims_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for chunk_id, claims in raw.items():
                                if isinstance(chunk_id, str) and isinstance(claims, list):
                                    if chunk_id not in self._claims_graph:
                                        self._claims_graph[chunk_id] = []
                                    existing_claim_keys = {
                                        (c.get("subject"), c.get("object"), c.get("type"))
                                        for c in self._claims_graph[chunk_id]
                                    }
                                    for claim in claims:
                                        if isinstance(claim, dict):
                                            key = (
                                                claim.get("subject"),
                                                claim.get("object"),
                                                claim.get("type"),
                                            )
                                            if key not in existing_claim_keys:
                                                self._claims_graph[chunk_id].append(claim)
                                                existing_claim_keys.add(key)
                    except Exception as e:
                        logger.warning(f"Could not merge claims for '{desc}': {e}")
                # --- community summaries (namespaced to avoid community-ID collision) ---
                desc_summ_path = desc_base / ".community_summaries.json"
                if desc_summ_path.exists():
                    try:
                        raw = _json.loads(desc_summ_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for summ_key, summary in raw.items():
                                if isinstance(summ_key, str) and isinstance(summary, dict):
                                    namespaced_key = f"desc_{desc}_{summ_key}"
                                    if namespaced_key not in self._community_summaries:
                                        self._community_summaries[namespaced_key] = dict(summary)
                    except Exception as e:
                        logger.warning(f"Could not merge community summaries for '{desc}': {e}")

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest_chunks(self, documents: list[dict]) -> IngestResult:
        """Extract entities/relations/claims from *documents* and merge into
        the graph; trigger (or defer) community rebuild.

        Absorbed verbatim from the body of AxonBrain.ingest()'s old
        ``if self.config.graph_rag:`` block. *documents* is the full
        post-chunking/dedup/RAPTOR document list for this ingest call —
        eligibility filtering (which chunks are candidates for entity
        extraction) happens inside this method, same as before.
        """
        _defer_saves = getattr(self.config, "ingest_batch_mode", False)
        _policy_on = getattr(self.config, "source_policy_enabled", False)
        updated = False
        # Only extract entities from actual document chunks (optionally include RAPTOR level-1)
        _include_raptor = getattr(self.config, "graph_rag_include_raptor_summaries", False)
        # Skip GraphRAG entity extraction for large sources when raptor=True.
        # Sources with >= raptor_graphrag_leaf_skip_threshold leaf chunks bypass
        # extraction; their RAPTOR summaries still enter GraphRAG if the include flag is set.
        _skip_threshold = getattr(self.config, "raptor_graphrag_leaf_skip_threshold", 20)
        _leaf_count_by_source: dict = {}
        for _doc in documents:
            if not _doc.get("metadata", {}).get("raptor_level"):
                _src = _doc.get("metadata", {}).get("source", _doc["id"])
                _leaf_count_by_source[_src] = _leaf_count_by_source.get(_src, 0) + 1
        _large_sources: set = set()
        if self.config.raptor and _skip_threshold > 0:
            _large_sources = {
                src for src, cnt in _leaf_count_by_source.items() if cnt >= _skip_threshold
            }
            if _large_sources:
                logger.info(
                    f"   GraphRAG: skipping leaf-chunk entity extraction for "
                    f"{len(_large_sources)} large source(s) (>= {_skip_threshold} leaf chunks)"
                )
        chunks_to_process = []
        for _doc in documents:
            _lvl = _doc.get("metadata", {}).get("raptor_level")
            _src = _doc.get("metadata", {}).get("source", _doc["id"])
            if not _lvl:  # leaf chunk
                if _src not in _large_sources:
                    chunks_to_process.append(_doc)
                # else: leaf from large source → skip; RAPTOR summary will cover it
            elif _lvl == 1:  # RAPTOR level-1 summary
                # Auto-include for large sources when RAPTOR is on (regardless of include flag)
                _auto_raptor = self.config.raptor and _src in _large_sources
                if _include_raptor or _auto_raptor:
                    chunks_to_process.append(_doc)
        if _policy_on:
            _grag_ok_list: list = []
            _grag_pol_skipped: set = set()
            for _d in chunks_to_process:
                _dtype = _d.get("metadata", {}).get("dataset_type", "doc")
                _, _g_ok = self._SOURCE_POLICY.get(_dtype, self._SOURCE_POLICY_DEFAULT)
                if _g_ok:
                    _grag_ok_list.append(_d)
                else:
                    _grag_pol_skipped.add(_d.get("metadata", {}).get("source", _d["id"]))
            if _grag_pol_skipped:
                logger.info(
                    "   GraphRAG: source_policy skipped %d source(s)",
                    len(_grag_pol_skipped),
                )
            chunks_to_process = _grag_ok_list
        # Skip chunks already present in the entity graph (cross-restart dedup)
        _already_extracted = self._build_extracted_chunk_ids()
        if _already_extracted:
            _before = len(chunks_to_process)
            chunks_to_process = [c for c in chunks_to_process if c["id"] not in _already_extracted]
            _skipped = _before - len(chunks_to_process)
            if _skipped:
                logger.info("   GraphRAG: skipping %d already-extracted chunk(s).", _skipped)
        if not chunks_to_process:
            logger.info("   GraphRAG: all chunks already extracted — nothing to do.")
        else:
            logger.info(f"   GraphRAG: Extracting entities from {len(chunks_to_process)} chunks...")
        _relations_enabled = bool(self.config.graph_rag_relations)
        _min_ent = getattr(self.config, "graph_rag_min_entities_for_relations", 3)
        _rel_budget = getattr(self.config, "graph_rag_relation_budget", 0)
        (
            results,
            rel_results,
            _rel_chunks,
            _relations_pipelined,
        ) = self._extract_graph_llm_batches(
            chunks_to_process,
            relations_enabled=_relations_enabled,
            min_entities_for_relations=_min_ent,
            relation_budget=_rel_budget,
        )
        # Track entity keys extracted this run for embedding (Item 5)
        from axon.rust_bridge import get_rust_bridge

        _rust_bridge = get_rust_bridge()
        with self._graph_lock:
            entities_extracted_this_run: list = []
            total_entities = 0
            _touched_entity_keys: set[str] = set()
            # Build a lookup from doc_id to doc for metadata writing (Item 7)
            doc_by_id = {doc["id"]: doc for doc in chunks_to_process}
            _entity_graph_changed = False
            _use_rust_entity_merge = (
                bool(results)
                and bool(getattr(self.config, "graph_rag_rust_merge_entities", False))
                and _rust_bridge.can_merge_entities_into_graph()
            )
            for doc_id, entities in results:
                total_entities += len(entities)
                for ent in entities:
                    if not isinstance(ent, dict) or not ent.get("name"):
                        continue
                    entities_extracted_this_run.append(ent)
                    key = ent["name"].lower().strip()
                    if not key:
                        continue
                    _touched_entity_keys.add(key)
                    existing = self._entity_graph.get(key)
                    if existing is None:
                        _entity_graph_changed = True
                    elif isinstance(existing, dict):
                        chunk_ids = existing.get("chunk_ids", [])
                        if doc_id not in chunk_ids:
                            _entity_graph_changed = True
                    else:
                        _use_rust_entity_merge = False
                        if doc_id not in existing:
                            _entity_graph_changed = True
            _merged_entities_in_rust = False
            if _use_rust_entity_merge:
                _merged_entities_in_rust = (
                    _rust_bridge.merge_entities_into_graph(self._entity_graph, results) is not None
                )
            if _merged_entities_in_rust and _entity_graph_changed:
                updated = True
                self._community_graph_dirty = True
            for doc_id, entities in results:
                for ent in entities:  # ent is now {"name": ..., "type": ..., "description": ...}
                    key = ent["name"].lower().strip() if isinstance(ent, dict) else ent.lower()
                    if not key:
                        continue
                    if not _merged_entities_in_rust:
                        if key not in self._entity_graph:
                            desc = ent.get("description", "") if isinstance(ent, dict) else ""
                            ent_type = (
                                ent.get("type", "UNKNOWN") if isinstance(ent, dict) else "UNKNOWN"
                            )
                            self._entity_graph[key] = {
                                "description": desc,
                                "type": ent_type,
                                "chunk_ids": [],
                                "frequency": 0,
                                "degree": 0,
                            }
                            self._token_index_add(key)
                        elif isinstance(self._entity_graph[key], dict):
                            # Update type if not yet set
                            if (
                                not self._entity_graph[key].get("type")
                                or self._entity_graph[key].get("type") == "UNKNOWN"
                            ):
                                new_type = (
                                    ent.get("type", "UNKNOWN")
                                    if isinstance(ent, dict)
                                    else "UNKNOWN"
                                )
                                if new_type and new_type != "UNKNOWN":
                                    self._entity_graph[key]["type"] = new_type
                        if isinstance(self._entity_graph[key], dict):
                            self._entity_graph[key].setdefault("chunk_ids", [])
                            if doc_id not in self._entity_graph[key]["chunk_ids"]:
                                self._entity_graph[key]["chunk_ids"].append(doc_id)
                                updated = True
                                self._community_graph_dirty = True
                        else:
                            # Legacy list format — migrate on the fly
                            if doc_id not in self._entity_graph[key]:
                                self._entity_graph[key].append(doc_id)
                                updated = True
                                self._community_graph_dirty = True
                    if isinstance(self._entity_graph.get(key), dict):
                        # Item 10: collect descriptions for canonicalization
                        if isinstance(ent, dict) and ent.get("description"):
                            desc_buf = self._entity_description_buffer.setdefault(key, [])
                            desc_buf.append(ent["description"])
                        if (
                            not self._entity_graph[key].get("description")
                            and isinstance(ent, dict)
                            and ent.get("description")
                        ):
                            self._entity_graph[key]["description"] = ent["description"]
                # Item 7: Write entity IDs back into chunk metadata for text-unit linkage
                doc = doc_by_id.get(doc_id)
                if doc is not None and entities and doc.get("metadata") is not None:
                    doc["metadata"]["entity_ids"] = [
                        e["name"].lower() for e in entities if isinstance(e, dict) and e.get("name")
                    ]
                # GAP 9: Update text_unit_entity_map
                self._text_unit_entity_map[doc_id] = [
                    e["name"] for e in entities if isinstance(e, dict) and e.get("name")
                ]
            # Item 2: Update frequency only for entities touched in this ingest run
            # (avoids O(|V|) scan of the full entity graph on every ingest batch)
            for entity_key in _touched_entity_keys:
                node = self._entity_graph.get(entity_key)
                if isinstance(node, dict):
                    node["frequency"] = len(node.get("chunk_ids", []))
            if updated and not _defer_saves:
                self._save_entity_graph()
            if total_entities == 0:
                logger.warning(
                    "GraphRAG: entity extraction returned 0 entities across all chunks. "
                    "This may be caused by an LLM that is too small or refused to extract entities. "
                    "GraphRAG relationship expansion will have no effect for this ingestion."
                )
            # Relation extraction: build SUBJECT | RELATION | OBJECT triples
            if _relations_enabled:
                _entity_count_by_doc = {doc_id: len(ents) for doc_id, ents in results}
                _rel_candidate_count = sum(
                    1
                    for doc in chunks_to_process
                    if _entity_count_by_doc.get(doc["id"], 0) >= _min_ent
                )
                if _relations_pipelined:
                    logger.info(
                        f"   GraphRAG: Pipelined relation extraction for {len(_rel_chunks)} chunks "
                        f"(skipped {len(chunks_to_process) - len(_rel_chunks)} below "
                        f"{_min_ent}-entity threshold)..."
                    )
                elif _rel_budget > 0 and _rel_candidate_count > _rel_budget:
                    logger.info(
                        f"   GraphRAG: Extracting relations from {len(_rel_chunks)} chunks "
                        f"(budget cap; {len(chunks_to_process) - len(_rel_chunks)} skipped)..."
                    )
                else:
                    logger.info(
                        f"   GraphRAG: Extracting relations from {len(_rel_chunks)} chunks "
                        f"(skipped {len(chunks_to_process) - len(_rel_chunks)} below "
                        f"{_min_ent}-entity threshold)..."
                    )
                rg_updated = False
                # Rust fast-path for relation graph merge
                if rel_results and _rust_bridge.can_relation_merge():
                    _added = _rust_bridge.merge_relations_into_graph(
                        self._relation_graph, rel_results
                    )
                    if _added > 0:
                        rg_updated = True
                        self._community_graph_dirty = True
                    # Still run Python loop for _relation_description_buffer (side-effect only)
                    for _doc_id, triples in rel_results:
                        for triple in triples:
                            if not isinstance(triple, dict):
                                continue
                            description = triple.get("description", "")
                            if not description:
                                continue
                            src_lower = triple.get("subject", "").lower().strip()
                            tgt_lower = triple.get("object", "").lower().strip()
                            if src_lower:
                                pair = (src_lower, tgt_lower)
                                if pair not in self._relation_description_buffer:
                                    self._relation_description_buffer[pair] = []
                                self._relation_description_buffer[pair].append(description)
                else:
                    for doc_id, triples in rel_results:
                        for triple in triples:
                            # triple is now a dict: {subject, relation, object, description}
                            if isinstance(triple, dict):
                                subject = triple.get("subject", "")
                                relation = triple.get("relation", "")
                                obj = triple.get("object", "")
                                description = triple.get("description", "")
                            else:
                                # Legacy tuple format fallback
                                subject, relation, obj = triple
                                description = ""
                            src_lower = subject.lower().strip()
                            if not src_lower:
                                continue
                            entry = {
                                "target": obj.lower().strip(),
                                "relation": relation.strip(),
                                "chunk_id": doc_id,
                                "description": description,
                                "strength": triple.get("strength", 5)
                                if isinstance(triple, dict)
                                else 5,
                                "support_count": 1,
                            }
                            if src_lower not in self._relation_graph:
                                self._relation_graph[src_lower] = []
                            # Item 8: weight tracking — increment weight for same (target, relation) pair
                            rel_tgt = entry["target"]
                            rel_relation = entry["relation"]
                            existing_entry = next(
                                (
                                    e
                                    for e in self._relation_graph[src_lower]
                                    if e.get("target") == rel_tgt
                                    and e.get("relation") == rel_relation
                                ),
                                None,
                            )
                            if existing_entry:
                                # Accumulate strength-based weight (sum of LM-derived strengths)
                                existing_entry["weight"] = existing_entry.get(
                                    "weight", 1
                                ) + entry.get("strength", 1)
                                existing_entry["support_count"] = (
                                    existing_entry.get("support_count", 1) + 1
                                )
                                # GAP 7: accumulate text_unit_ids
                                if "text_unit_ids" not in existing_entry:
                                    existing_entry["text_unit_ids"] = [
                                        existing_entry.get("chunk_id", "")
                                    ]
                                if doc_id not in existing_entry["text_unit_ids"]:
                                    existing_entry["text_unit_ids"].append(doc_id)
                                rg_updated = True
                            else:
                                entry["weight"] = entry.get("strength", 1)
                                entry["text_unit_ids"] = [doc_id]
                                self._relation_graph[src_lower].append(entry)
                                rg_updated = True
                            # GAP 3b: update relation description buffer
                            if description:
                                pair = (src_lower, rel_tgt)
                                if pair not in self._relation_description_buffer:
                                    self._relation_description_buffer[pair] = []
                                self._relation_description_buffer[pair].append(description)
                if rg_updated and not _defer_saves:
                    self._save_relation_graph()
                if getattr(self.config, "graph_rag_relation_backend", "llm") == "rebel":
                    _rg_edge_count = sum(len(v) for v in self._relation_graph.values())
                    if _rg_edge_count == 0 and len(_rel_chunks) > 0:
                        logger.warning(
                            "GraphRAG REBEL: processed %d chunks but produced 0 relation edges. "
                            "If using a local model path, verify the checkpoint contains pretrained weights "
                            "(a 'newly initialized weights' warning from transformers indicates an invalid checkpoint).",
                            len(_rel_chunks),
                        )
                    else:
                        logger.info(
                            "GraphRAG REBEL: %d relation edges from %d chunks.",
                            _rg_edge_count,
                            len(_rel_chunks),
                        )
                # Normalize relation targets into entity graph so traversal never KeyErrors
                if rg_updated or updated:
                    _stub_added = False
                    for _src, _entries in self._relation_graph.items():
                        for _entry in _entries:
                            _tgt = _entry.get("target", "").lower().strip()
                            if not _tgt:
                                continue
                            if _tgt not in self._entity_graph:
                                self._entity_graph[_tgt] = {
                                    "description": "",
                                    "type": "UNKNOWN",
                                    "chunk_ids": [],
                                    "frequency": 0,
                                    "degree": 0,
                                }
                                self._token_index_add(_tgt)
                                _stub_added = True
                            # Ensure the relation's source chunk is in the target's chunk_ids
                            _cid = _entry.get("chunk_id", "")
                            if _cid:
                                _tgt_node = self._entity_graph[_tgt]
                                if isinstance(_tgt_node, dict):
                                    _tgt_node.setdefault("chunk_ids", [])
                                    if _cid not in _tgt_node["chunk_ids"]:
                                        _tgt_node["chunk_ids"].append(_cid)
                                        _tgt_node["frequency"] = len(_tgt_node["chunk_ids"])
                                        _stub_added = True
                    if _stub_added and not _defer_saves:
                        self._save_entity_graph()
                # GAP 9: Update text_unit_relation_map
                for doc_id, triples in rel_results:
                    self._text_unit_relation_map[doc_id] = [
                        (t.get("subject", ""), t.get("object", ""))
                        if isinstance(t, dict)
                        else (t[0], t[2])
                        for t in triples
                    ]
            if getattr(self, "_graph_rag_cache_dirty", False) and not _defer_saves:
                self._save_graph_rag_extraction_cache()
            # Item 2: Recompute degree for entities touched by this ingest's relations only
            # (avoids O(|V|) scan of the full entity graph on every ingest batch)
            for entity_key in _touched_entity_keys:
                if isinstance(self._entity_graph.get(entity_key), dict):
                    self._entity_graph[entity_key]["degree"] = len(
                        self._relation_graph.get(entity_key, [])
                    )
            # Item 5: Embed entity descriptions for query-time matching
            if getattr(self.config, "graph_rag_entity_embedding_match", True):
                entity_keys_this_batch = list(
                    {ent["name"].lower() for ent in entities_extracted_this_run if ent.get("name")}
                )
                self._embed_entities(entity_keys_this_batch)
            # Item 10: Canonicalize entity descriptions
            # A3: also run for "deep" tier
            _depth = getattr(self.config, "graph_rag_depth", "standard")
            if self.config.graph_rag and (
                getattr(self.config, "graph_rag_canonicalize", False) or _depth == "deep"
            ):
                self._canonicalize_entity_descriptions()
            if self.config.graph_rag and getattr(
                self.config, "graph_rag_canonicalize_relations", False
            ):
                self._canonicalize_relation_descriptions()
            # Item 11: Extract claims
            # A3: also run for "deep" tier
            claims_changed = False
            if getattr(self.config, "graph_rag_claims", False) or _depth == "deep":
                logger.info(
                    f"   GraphRAG: Extracting claims from {len(chunks_to_process)} chunks..."
                )

                def _proc_claims(doc):
                    return doc["id"], self._extract_claims(doc["text"])

                claim_results = list(self._executor.map(_proc_claims, chunks_to_process))
                with self._graph_lock:
                    for doc_id, claims in claim_results:
                        if claims:
                            # GAP 5: set text_unit_id on each claim
                            for claim in claims:
                                if isinstance(claim, dict):
                                    claim["text_unit_id"] = doc_id
                            self._claims_graph[doc_id] = claims
                            claims_changed = True
                    if claims_changed and not _defer_saves:
                        self._save_claims_graph()
            if self.config.graph_rag_community and self._community_graph_dirty:
                if getattr(self.config, "graph_rag_community_defer", True):
                    pass  # leave dirty; caller must invoke finalize_graph()
                else:
                    self._community_graph_dirty = False
                    if self.config.graph_rag_community_async:

                        def _debounced_rebuild():
                            import time as _time

                            self._community_build_in_progress = True
                            try:
                                _time.sleep(self.config.graph_rag_community_rebuild_debounce_s)
                                self._rebuild_communities()
                            finally:
                                self._community_build_in_progress = False

                        self._executor.submit(_debounced_rebuild)
                    else:
                        self._rebuild_communities()
        relations_added = sum(len(triples) for _, triples in rel_results) if rel_results else 0
        return IngestResult(
            entities_added=total_entities,
            relations_added=relations_added,
            chunks_processed=len(documents),
            backend_id="graphrag",
        )

    # ------------------------------------------------------------------
    # Query-time entity-graph expansion
    # ------------------------------------------------------------------
    def expand_with_entity_graph(
        self, query: str, results: list[dict], cfg: Any = None
    ) -> tuple[list[dict], list[str]]:
        """Expand retrieval results using GraphRAG entity linkage.
        1. Extract entities from the query.
        2. Match query entities against the entity graph using Jaccard similarity.
        3. Perform 1-hop traversal via the relation graph (when enabled).
        4. Fetch any chunks not already in results, tag with _graph_expanded.
        5. Return the expanded list (top_k slicing is deferred to the caller).

        Moved verbatim from QueryRouterMixin._expand_with_entity_graph() —
        this is a query-orchestration concern layered on top of graph
        storage, but it depends so heavily on GraphRAG-owned state
        (_entity_graph, _relation_graph, _entity_token_index, _graph_lock,
        the traversal cache) that it lives here now rather than dragging
        that state back onto QueryRouterMixin.
        """
        # Item 5: Union LLM-extracted entities with embedding-based matches.
        # LLM extraction captures exact textual mentions; embedding matching adds semantic neighbors.
        query_entities = self._extract_entities(query)
        if (
            getattr(self.config, "graph_rag_entity_embedding_match", True)
            and self._entity_embeddings
        ):
            matched_keys = self._match_entities_by_embedding(query)
            seen_names = {e.get("name", "").lower() for e in query_entities}
            for k in matched_keys:
                if k.lower() not in seen_names:
                    query_entities.append({"name": k, "type": "UNKNOWN", "description": ""})
        if not query_entities:
            return results, []
        active_top_k = (
            cfg.top_k if (cfg is not None and cfg.top_k is not None) else self.config.top_k
        )
        active_cfg = cfg if cfg is not None else self.config
        existing_ids = {r["id"] for r in results}
        # {doc_id: best_score} so we don't lower a score if the same ID matches again
        extra_id_scores: dict[str, float] = {}
        matched_entities: set[str] = set()
        with self._graph_lock:
            for query_entity in query_entities:
                # Support both new dict-node format and legacy list format
                q_name = (
                    query_entity if isinstance(query_entity, str) else query_entity.get("name", "")
                )
                if not q_name:
                    continue
                # Token-index candidate lookup: gather entities that share at least
                # one token with the query entity, then score only that candidate set
                # (typically 10-200 nodes) instead of all |V| entities.
                # Falls back to full scan when index is empty (e.g. direct attribute
                # assignment in tests without calling _rebuild_entity_token_index).
                q_lower = q_name.lower().strip()
                q_tokens = q_lower.split()
                token_idx = self._entity_token_index
                if token_idx:
                    candidates: set[str] = set()
                    for tok in q_tokens:
                        bucket = token_idx.get(tok)
                        if bucket:
                            candidates.update(bucket)
                    candidate_iter = [(eid, self._entity_graph.get(eid)) for eid in candidates]
                else:
                    # Fallback: index not populated yet
                    candidate_iter = list(self._entity_graph.items())
                for eid, node in candidate_iter:
                    if node is None:
                        continue
                    score = self._entity_matches(q_name, eid)
                    if score <= 0.0:
                        continue
                    matched_entities.add(eid)
                    # Scale matched score into [0.5, 0.8) range so it is clearly below
                    # a direct vector-match score but still meaningfully ranked.
                    doc_score = 0.5 + score * 0.3
                    doc_ids = node.get("chunk_ids", [])
                    for did in doc_ids:
                        if did not in existing_ids:
                            if extra_id_scores.get(did, 0.0) < doc_score:
                                extra_id_scores[did] = doc_score

            # Multi-hop traversal via relation graph
            def _cfg_get(name, default):
                return getattr(active_cfg, name, default)

            max_hops = _cfg_get("graph_rag_max_hops", 1)
            hop_decay = _cfg_get("graph_rag_hop_decay", 0.7)
            # Performance guard for large graphs (Epic 1/4)
            large_threshold = _cfg_get("graph_rag_large_graph_threshold", 50000)
            if len(self._entity_graph) > large_threshold and max_hops > 1:
                logger.info(
                    f"   GraphRAG: large graph detected ({len(self._entity_graph)} nodes); "
                    f"capping max_hops at 1 for performance."
                )
                max_hops = 1
            use_relations = (
                getattr(active_cfg, "graph_rag_relations", True)
                and self._relation_graph
                and max_hops > 0
            )
            if use_relations and matched_entities:
                # Check traversal cache before running BFS.
                # Cache key covers entity set + hop params so different configs
                # don't collide.
                _cache_key = (frozenset(matched_entities), max_hops, hop_decay)
                _bfs_scores: dict[str, float] | None = None
                _cached = lru_ttl_get(
                    self._traversal_cache,
                    self._traversal_cache_lock,
                    _cache_key,
                    self._traversal_cache_ttl,
                )
                if _cached is not None:
                    _bfs_scores = _cached[1]
                if _bfs_scores is None:
                    # BFS for multi-hop traversal
                    _bfs_scores = {}
                    current_hop_entities = set(matched_entities)
                    visited_entities = set(matched_entities)
                    for hop in range(1, max_hops + 1):
                        next_hop_entities = set()
                        # Score for this hop decays from base 0.8
                        # Hop 1: 0.8 * 0.7 = 0.56
                        # Hop 2: 0.56 * 0.7 = 0.392
                        hop_score = 0.8 * (hop_decay**hop)
                        for src_entity in current_hop_entities:
                            for entry in self._relation_graph.get(src_entity, []):
                                target = entry.get("target", "").lower()
                                if not target or target in visited_entities:
                                    continue
                                visited_entities.add(target)
                                next_hop_entities.add(target)
                                target_node = self._entity_graph.get(target, {})
                                target_chunk_ids = target_node.get("chunk_ids", [])
                                for did in target_chunk_ids:
                                    if _bfs_scores.get(did, 0.0) < hop_score:
                                        _bfs_scores[did] = hop_score
                        if not next_hop_entities:
                            break
                        current_hop_entities = next_hop_entities
                    # Store BFS result in traversal cache
                    lru_ttl_put(
                        self._traversal_cache,
                        self._traversal_cache_lock,
                        _cache_key,
                        dict(_bfs_scores),
                        maxsize=self._traversal_cache_maxsize,
                    )
                # Merge BFS scores into extra_id_scores (skip IDs already in results)
                for did, hop_score in _bfs_scores.items():
                    if did not in existing_ids:
                        if extra_id_scores.get(did, 0.0) < hop_score:
                            extra_id_scores[did] = hop_score
        if not extra_id_scores:
            return results, list(matched_entities)
        # Fetch the extra chunks from the vector store (capped to avoid huge fetches)
        extra_ids = list(extra_id_scores.keys())[:active_top_k]
        try:
            extra_results = self.vector_store.get_by_ids(extra_ids)
            if extra_results:
                logger.info(
                    f"   GraphRAG: expanded results by {len(extra_results)} entity-linked doc(s)"
                )
                for r in extra_results:
                    # Use a very low fallback if somehow missing from map
                    r["score"] = extra_id_scores.get(r["id"], 0.01)
                    r["_graph_expanded"] = True
                results = list(results) + extra_results
        except Exception as e:
            logger.debug(f"GraphRAG expansion failed: {e}")
        return results, list(matched_entities)
