"""GraphRAG entity/community graph management and retrieval (GraphRagMixin)."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("Axon")

_GRAPHRAG_REDUCE_SYSTEM_PROMPT = (
    "You are a helpful assistant responding to questions about a dataset by synthesizing the "
    "perspectives from multiple data analysts.\n\n"
    "Generate a response of the target length and format that responds to the user's question, "
    "summarize all the reports from multiple analysts who focused on different parts of the "
    "dataset.\n\n"
    "Note that the analysts' reports provided below are ranked in the importance of addressing "
    "the user's question.\n\n"
    "If you don't know the answer or if the input reports do not contain sufficient information "
    "to provide an answer, just say so. Do not make anything up.\n\n"
    "The response should preserve the original meaning and use of modal verbs such as 'shall', "
    "'may' or 'will'.\n\n"
    "Points supported by data should list their data references as follows:\n"
    '"This is an example sentence supported by multiple data references [Data: Reports (report1), (report2)]."\n\n'
    "**Do not list more than 5 data references in a single reference**. Instead, list the top 5 "
    'most relevant references and add "+more" to indicate that there are more.\n\n'
    "Everything needs to be explained at length and in detail, with specific quotes and passages "
    "from the data providing evidence to back up each and every claim."
)

_GRAPHRAG_NO_DATA_ANSWER = (
    "I am sorry but I am unable to answer this question given the provided data."
)


class GraphRagMixin:
    def _load_entity_graph(self) -> dict:
        """Load persisted entity→doc_id graph from disk.

        Shape: {entity_lower: {"description": str, "chunk_ids": list[str]}}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return {}
                cleaned: dict = {}
                for key, value in raw.items():
                    if not isinstance(key, str):
                        continue
                    if isinstance(value, dict) and "chunk_ids" in value:
                        node = value
                        node.setdefault("type", "UNKNOWN")
                        node.setdefault("frequency", len(node.get("chunk_ids", [])))
                        node.setdefault("degree", 0)
                        cleaned[key] = node
                    # Otherwise skip malformed entries
                return cleaned
            except Exception:
                pass
        return {}

    def _save_entity_graph(self) -> None:
        """Persist entity graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._entity_graph), encoding="utf-8")

    def _load_code_graph(self) -> dict:
        """Load code graph from disk. Returns empty graph if not found."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "nodes" in data and "edges" in data:
                    return data
            except Exception:
                pass
        return {"nodes": {}, "edges": []}

    def _save_code_graph(self) -> None:
        """Persist code graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._code_graph), encoding="utf-8")

    def _load_relation_graph(self) -> dict:
        """Load persisted relation graph from disk.

        Shape: {source_entity_lower: [{target: str, relation: str, chunk_id: str}]}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".relation_graph.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return {}
                cleaned: dict = {}
                for key, value in raw.items():
                    if not isinstance(key, str) or not isinstance(value, list):
                        continue
                    cleaned[key] = [
                        entry
                        for entry in value
                        if isinstance(entry, dict)
                        and "target" in entry
                        and "relation" in entry
                        and "chunk_id" in entry
                    ]
                return cleaned
            except Exception:
                pass
        return {}

    def _save_relation_graph(self) -> None:
        """Persist relation graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".relation_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._relation_graph), encoding="utf-8")

    def _load_community_levels(self) -> dict:
        """Load persisted hierarchical community levels from disk.

        Shape: {level_int: {entity_lower: community_id}}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_levels.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            pass
        return {}

    def _save_community_levels(self) -> None:
        """Persist hierarchical community levels to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_levels.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({str(k): v for k, v in self._community_levels.items()}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Could not save community levels: {e}")

    def _load_community_hierarchy(self) -> dict:
        """Load persisted community hierarchy from disk.

        Shape: {cluster_id_int: parent_cluster_id_int_or_None}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_hierarchy.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # JSON keys are strings; support both legacy int keys and
                    # level-qualified string keys like "1_3".
                    result = {}
                    for k, v in raw.items():
                        key = k if ("_" in str(k)) else (int(k) if str(k).isdigit() else k)
                        val = (
                            None
                            if v is None
                            else (
                                v
                                if (isinstance(v, str) and "_" in v)
                                else (int(str(v)) if str(v).isdigit() else v)
                            )
                        )
                        result[key] = val
                    return result
        except Exception:
            pass
        return {}

    def _save_community_hierarchy(self) -> None:
        """Persist community hierarchy to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_hierarchy.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({str(k): v for k, v in self._community_hierarchy.items()}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Could not save community hierarchy: {e}")

    def _load_community_summaries(self) -> dict:
        """Load persisted community summaries from disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_summaries.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_community_summaries(self) -> None:
        """Persist community summaries to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_summaries.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._community_summaries), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save community summaries: {e}")

    def _load_entity_embeddings(self) -> dict:
        """Load persisted entity embeddings from disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_embeddings.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_entity_embeddings(self) -> None:
        """Persist entity embeddings to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_embeddings.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._entity_embeddings), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save entity embeddings: {e}")

    def _load_claims_graph(self) -> dict:
        """Load persisted claims graph from disk.

        Shape: {chunk_id: [claim_dict, ...]}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".claims_graph.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_claims_graph(self) -> None:
        """Persist claims graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".claims_graph.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._claims_graph), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save claims graph: {e}")

    def _build_networkx_graph(self):
        """Build a NetworkX undirected graph from entity and relation data."""
        import networkx as nx

        _min_freq_raw = getattr(self.config, "graph_rag_entity_min_frequency", 1)
        _min_freq = int(_min_freq_raw) if isinstance(_min_freq_raw, int | float) else 1
        G = nx.Graph()
        for entity, node in self._entity_graph.items():
            if isinstance(node, dict) and node.get("frequency", 1) < _min_freq:
                continue
            desc = node.get("description", "") if isinstance(node, dict) else ""
            G.add_node(entity, description=desc)
        for src, entries in self._relation_graph.items():
            for entry in entries:
                tgt = entry.get("target", "")
                if src and tgt:
                    edge_weight = entry.get("weight", 1)
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] += edge_weight
                    else:
                        G.add_edge(
                            src,
                            tgt,
                            weight=edge_weight,
                            relation=entry.get("relation", ""),
                            description=entry.get("description", ""),
                        )
        return G

    def _run_community_detection(self) -> dict:
        """Run Louvain community detection. Returns {entity_lower: community_id}."""
        try:
            import networkx as nx  # noqa: F401 — ensures ImportError fires when networkx is absent
            import networkx.algorithms.community as nx_comm
        except ImportError:
            logger.warning(
                "GraphRAG community detection requires networkx. "
                "Install with: pip install networkx"
            )
            return {}
        G = self._build_networkx_graph()
        if len(G.nodes) < 2:
            return dict.fromkeys(G.nodes, 0)
        try:
            communities = nx_comm.louvain_communities(G, seed=42)
            mapping = {}
            for cid, members in enumerate(communities):
                for entity in members:
                    mapping[entity] = cid
            return mapping
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}

    def _run_hierarchical_community_detection(self) -> tuple:
        """Run hierarchical community detection.

        Returns:
            (community_levels, community_hierarchy, community_children)
            - community_levels: {level_int: {entity_lower: cluster_id}}
            - community_hierarchy: {cluster_id: parent_cluster_id}  (None for root)
            - community_children: {cluster_id: [child_cluster_ids]}
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "GraphRAG community detection requires networkx. "
                "Install with: pip install networkx"
            )
            return {}, {}, {}

        G = self._build_networkx_graph()
        if G.number_of_nodes() < 2:
            nodes = list(G.nodes())
            return {0: dict.fromkeys(nodes, 0)}, {0: None}, {0: []}

        n_levels = max(1, getattr(self.config, "graph_rag_community_levels", 2))
        max_cluster_size = getattr(self.config, "graph_rag_community_max_cluster_size", 10)
        seed = getattr(self.config, "graph_rag_leiden_seed", 42)
        use_lcc = getattr(self.config, "graph_rag_community_use_lcc", True)

        components = list(nx.connected_components(G))
        if use_lcc and components:
            try:
                lcc_nodes = max(components, key=len)
                dropped = G.number_of_nodes() - len(lcc_nodes)
                if dropped > 0:
                    logger.info(
                        f"GraphRAG: use_lcc=True — clustering {len(lcc_nodes)} nodes, "
                        f"dropping {dropped} nodes in {len(components) - 1} smaller components"
                    )
                G = G.subgraph(lcc_nodes).copy()
            except Exception:
                pass
        elif not use_lcc and len(components) > 1:
            logger.debug(
                f"GraphRAG: clustering all {len(components)} connected components "
                f"({G.number_of_nodes()} total nodes)"
            )

        _backend = getattr(self.config, "graph_rag_community_backend", "auto")

        try:
            # Try graspologic hierarchical Leiden — skipped when backend != "auto"
            if _backend != "auto":
                raise ImportError("backend override — skipping graspologic")
            from graspologic.partition import hierarchical_leiden

            partitions = hierarchical_leiden(G, max_cluster_size=max_cluster_size, random_seed=seed)
            community_levels: dict = {}
            community_hierarchy: dict = {}
            community_children: dict = {}

            for triple in partitions:
                level = triple.level
                entity = triple.node
                cluster = triple.cluster
                parent = getattr(triple, "parent_cluster", None)

                if level not in community_levels:
                    community_levels[level] = {}
                community_levels[level][entity] = cluster

                if cluster not in community_hierarchy:
                    community_hierarchy[cluster] = parent
                if parent is not None:
                    if parent not in community_children:
                        community_children[parent] = []
                    if cluster not in community_children[parent]:
                        community_children[parent].append(cluster)

            # Normalize Leiden hierarchy to level-qualified string keys to match Louvain format
            # and avoid cross-level integer collisions.
            normalized_hierarchy: dict = {}
            normalized_children: dict = {}
            # Build a level-lookup for each (level, cluster) → level-qualified key
            cluster_to_level: dict = {}
            for lvl, cmap in community_levels.items():
                for _ent, cid in cmap.items():
                    if cid not in cluster_to_level:
                        cluster_to_level[cid] = lvl
            for raw_cid, raw_parent in community_hierarchy.items():
                lvl = cluster_to_level.get(raw_cid, 0)
                norm_key = f"{lvl}_{raw_cid}"
                if raw_parent is None:
                    norm_parent = None
                else:
                    parent_lvl = cluster_to_level.get(raw_parent, max(0, lvl - 1))
                    norm_parent = f"{parent_lvl}_{raw_parent}"
                normalized_hierarchy[norm_key] = norm_parent
                if norm_parent is not None:
                    if norm_parent not in normalized_children:
                        normalized_children[norm_parent] = []
                    if norm_key not in normalized_children[norm_parent]:
                        normalized_children[norm_parent].append(norm_key)

            return community_levels, normalized_hierarchy, normalized_children

        except ImportError:
            logger.warning(
                "GraphRAG: graspologic not installed — falling back to leidenalg/Louvain. "
                "pip install axon[graphrag]"
            )

        # Tier-2: multi-resolution Leiden via leidenalg — skipped when backend="louvain"
        try:
            if _backend == "louvain":
                raise ImportError("backend override — skipping leidenalg")
            import igraph as _ig
            import leidenalg as _la

            try:
                import numpy as np

                resolutions = list(np.linspace(0.5, 1.5, n_levels)) if n_levels > 1 else [1.0]
            except ImportError:
                step = 1.0 / max(n_levels - 1, 1) if n_levels > 1 else 0
                resolutions = [0.5 + i * step for i in range(n_levels)]

            _G_ig = _ig.Graph.from_networkx(G)
            community_levels: dict = {}
            for level_idx, resolution in enumerate(resolutions):
                try:
                    partition = _la.find_partition(
                        _G_ig,
                        _la.ModularityVertexPartition,
                        seed=seed,
                    )
                    node_names = (
                        _G_ig.vs["_nx_name"]
                        if "_nx_name" in _G_ig.vs.attributes()
                        else list(G.nodes())
                    )
                    cmap = {}
                    for cid, members in enumerate(partition):
                        for idx in members:
                            cmap[node_names[idx]] = cid
                    community_levels[level_idx] = cmap
                except Exception as _e:
                    logger.debug("leidenalg at resolution %s failed: %s", resolution, _e)

            if community_levels:
                # Build synthetic hierarchy (same approach as Louvain fallback below)
                community_hierarchy: dict = {}
                community_children: dict = {}
                if len(community_levels) > 1:
                    _levels_sorted = sorted(community_levels.keys())
                    for _i in range(1, len(_levels_sorted)):
                        _fine = _levels_sorted[_i]
                        _coarse = _levels_sorted[_i - 1]
                        for _fine_cid in set(community_levels[_fine].values()):
                            _fine_key = f"{_fine}_{_fine_cid}"
                            _members = [
                                n for n, c in community_levels[_fine].items() if c == _fine_cid
                            ]
                            _votes: dict = {}
                            for _m in _members:
                                _p = community_levels[_coarse].get(_m)
                                if _p is not None:
                                    _votes[_p] = _votes.get(_p, 0) + 1
                            _parent_cid = max(_votes, key=_votes.get) if _votes else None
                            _parent_key = (
                                f"{_coarse}_{_parent_cid}" if _parent_cid is not None else None
                            )
                            community_hierarchy[_fine_key] = _parent_key
                            if _parent_key:
                                community_children.setdefault(_parent_key, [])
                                if _fine_key not in community_children[_parent_key]:
                                    community_children[_parent_key].append(_fine_key)
                    for _cid in set(community_levels[_levels_sorted[0]].values()):
                        _root_key = f"{_levels_sorted[0]}_{_cid}"
                        community_hierarchy.setdefault(_root_key, None)
                else:
                    for _cid in set(community_levels[0].values()):
                        community_hierarchy[f"0_{_cid}"] = None
                logger.debug(
                    "GraphRAG: leidenalg produced %d community levels.", len(community_levels)
                )
                return community_levels, community_hierarchy, community_children
        except ImportError:
            pass  # fall through to networkx Louvain

        # Synthetic parent-child mapping using multiple-resolution Louvain
        try:
            import networkx.algorithms.community as nx_comm
        except ImportError:
            return {0: dict.fromkeys(G.nodes(), 0)}, {0: None}, {0: []}

        try:
            import numpy as np

            resolutions = list(np.linspace(0.5, 1.5, n_levels)) if n_levels > 1 else [1.0]
        except ImportError:
            step = 1.0 / max(n_levels - 1, 1) if n_levels > 1 else 0
            resolutions = [0.5 + i * step for i in range(n_levels)]

        community_levels = {}

        for level_idx, resolution in enumerate(resolutions):
            try:
                partition = nx_comm.louvain_communities(G, seed=seed, resolution=resolution)
                cmap = {}
                for cid, nodes in enumerate(partition):
                    for node in nodes:
                        cmap[node] = cid
                community_levels[level_idx] = cmap
            except Exception as e:
                logger.debug(f"Louvain at resolution {resolution} failed: {e}")

        if not community_levels:
            return {0: dict.fromkeys(G.nodes(), 0)}, {0: None}, {0: []}

        # Synthetic parent mapping — use level-qualified keys to avoid cross-level collisions
        community_hierarchy = {}
        community_children = {}

        if len(community_levels) > 1:
            levels_sorted = sorted(community_levels.keys())
            for i in range(1, len(levels_sorted)):
                fine_level = levels_sorted[i]
                coarse_level = levels_sorted[i - 1]
                fine_map = community_levels[fine_level]
                coarse_map = community_levels[coarse_level]

                fine_clusters = set(fine_map.values())
                for fine_cid in fine_clusters:
                    fine_key = f"{fine_level}_{fine_cid}"
                    fine_members = [n for n, c in fine_map.items() if c == fine_cid]
                    coarse_votes: dict = {}
                    for m in fine_members:
                        parent_cid = coarse_map.get(m)
                        if parent_cid is not None:
                            coarse_votes[parent_cid] = coarse_votes.get(parent_cid, 0) + 1
                    parent_cid = max(coarse_votes, key=coarse_votes.get) if coarse_votes else None
                    parent_key = f"{coarse_level}_{parent_cid}" if parent_cid is not None else None
                    community_hierarchy[fine_key] = parent_key
                    if parent_key is not None:
                        if parent_key not in community_children:
                            community_children[parent_key] = []
                        if fine_key not in community_children[parent_key]:
                            community_children[parent_key].append(fine_key)

            for cid in set(community_levels[levels_sorted[0]].values()):
                root_key = f"{levels_sorted[0]}_{cid}"
                if root_key not in community_hierarchy:
                    community_hierarchy[root_key] = None
        else:
            for cid in set(community_levels[0].values()):
                community_hierarchy[f"0_{cid}"] = None

        return community_levels, community_hierarchy, community_children

    def _rebuild_communities(self) -> None:
        """Run community detection and generate summaries."""
        with self._community_rebuild_lock:
            # P1: Entity alias resolution — merge near-duplicates before community detection
            if getattr(self.config, "graph_rag_entity_resolve", False):
                self._resolve_entity_aliases()
            logger.info("GraphRAG: Running community detection...")
            result = self._run_hierarchical_community_detection()
            if isinstance(result, tuple) and len(result) == 3:
                community_levels, hierarchy, children = result
            else:
                # Legacy: _run_hierarchical_community_detection returned a plain dict
                community_levels = result if isinstance(result, dict) else {}
                hierarchy = {}
                children = {}
            self._community_levels = community_levels
            self._community_hierarchy = hierarchy
            self._community_children = children
            self._save_community_hierarchy()
            if self._community_levels:
                self._save_community_levels()
                total = sum(len(set(m.values())) for m in self._community_levels.values())
                logger.info(
                    f"GraphRAG: {len(self._community_levels)} community levels, "
                    f"{total} total communities detected."
                )
                if self.config.graph_rag_community:
                    _lazy = getattr(self.config, "graph_rag_community_lazy", False)
                    if _lazy:
                        logger.info(
                            "GraphRAG: community_lazy=True — skipping summarization; "
                            "will generate on first global query."
                        )
                    else:
                        self._generate_community_summaries()
                        if getattr(self.config, "graph_rag_index_community_reports", True):
                            self._index_community_reports_in_vector_store()

    def finalize_graph(self, force: bool = False) -> None:
        """Explicitly trigger community rebuild.

        Use after batch ingest with ``graph_rag_community_defer=True`` to run
        community detection once when all documents have been ingested.
        Set ``force=True`` to rebuild even when the in-memory dirty flag is not
        set (for example, after a server restart or project switch).
        """
        self._assert_write_allowed("finalize_graph")
        if force or self._community_graph_dirty:
            self._community_graph_dirty = False
            self._rebuild_communities()

    # ── Graph visualization ──────────────────────────────────────────────────

    _VIZ_TYPE_COLORS: dict[str, str] = {
        "PERSON": "#4e79a7",
        "ORGANIZATION": "#f28e2b",
        "GEO": "#59a14f",
        "EVENT": "#e15759",
        "CONCEPT": "#76b7b2",
        "PRODUCT": "#edc948",
        "UNKNOWN": "#bab0ab",
    }

    def _generate_community_summaries(self, query_hint: str = "") -> None:
        """Generate LLM summaries for each detected community cluster across all levels.

        When *query_hint* is provided (lazy mode), communities are ranked by relevance to
        the query first so the most useful ones get LLM treatment before the budget cap.
        The LLM cap is tightened to ``graph_rag_global_top_communities`` in lazy mode,
        replacing the full ``graph_rag_community_llm_max_total`` limit.
        """
        if not self._community_levels:
            return
        import hashlib as _hashlib
        import json as _json
        import re as _re
        from collections import defaultdict

        def _member_hash(level_idx: int, members: list) -> str:
            members_sorted = sorted(members)
            raw = f"{level_idx}|{'|'.join(members_sorted)}"
            return _hashlib.md5(raw.encode()).hexdigest()

        summaries = {}
        total_communities = sum(len(set(m.values())) for m in self._community_levels.values())

        _min_size = getattr(self.config, "graph_rag_community_min_size", 3)
        _top_n_per_level = getattr(self.config, "graph_rag_community_llm_top_n_per_level", 15)
        _max_total = getattr(self.config, "graph_rag_community_llm_max_total", 30)

        # Lazy mode: tighten cap to graph_rag_global_top_communities so only the most
        # query-relevant communities receive LLM treatment on the first global query.
        _lazy_cap = getattr(self.config, "graph_rag_global_top_communities", 0)
        if query_hint and _lazy_cap > 0:
            _max_total = min(_max_total, _lazy_cap)
            logger.info(
                "GraphRAG: lazy mode — generating community summaries for query "
                "(LLM cap: %d of %d communities).",
                _max_total,
                total_communities,
            )
        else:
            logger.info("GraphRAG: Generating summaries for %d communities...", total_communities)

        # Pre-compute query relevance scores when a hint is provided.
        _query_words: set = set(query_hint.lower().split()) if query_hint else set()

        def _query_relevance(members: list) -> float:
            """Fraction of query words present in community entity names."""
            if not _query_words:
                return 0.0
            member_words = {w for m in members for w in m.lower().split()}
            return len(_query_words & member_words) / len(_query_words)

        _llm_calls_issued = 0

        def _community_score(members: list) -> float:
            """Composite score = density × size. Used to rank which communities get LLM."""
            member_set = set(members)
            internal_edges = sum(
                1
                for src in members
                for entry in self._relation_graph.get(src, [])
                if entry.get("target", "") in member_set
            )
            return (internal_edges / max(len(members), 1)) * len(members)

        def _template_summary(level_idx: int, cid, members: list, new_hash: str) -> tuple:
            """Deterministic zero-LLM summary for small/capped communities."""
            size = len(members)
            sample = ", ".join(sorted(members)[:5])
            suffix = f" (+{size - 5} more)" if size > 5 else ""
            title = f"Community {cid}"
            summary = f"A cluster of {size} entities: {sample}{suffix}."
            return f"{level_idx}_{cid}", {
                "title": title,
                "summary": summary,
                "findings": [],
                "rank": float(size),
                "full_content": f"# {title}\n\n{summary}",
                "entities": members,
                "size": size,
                "level": level_idx,
                "member_hash": new_hash,
                "template": True,
            }

        def _summarise(args):
            level_idx, cid, members = args
            summary_key = f"{level_idx}_{cid}"
            new_hash = _member_hash(level_idx, members)
            existing = self._community_summaries.get(summary_key, {})
            if existing.get("member_hash") == new_hash and existing.get("full_content"):
                # Membership unchanged — reuse cached summary
                cached = dict(existing)
                cached["member_hash"] = new_hash
                return summary_key, cached
            ent_parts = []
            for e in members[:30]:
                node = self._entity_graph.get(e, {})
                desc = node.get("description", "") if isinstance(node, dict) else ""
                ent_type = node.get("type", "") if isinstance(node, dict) else ""
                if desc:
                    ent_parts.append(
                        f"- {e} [{ent_type}]: {desc}" if ent_type else f"- {e}: {desc}"
                    )
                else:
                    ent_parts.append(f"- {e}")
            member_set = set(members)
            rel_parts = []
            for src in members[:20]:
                for entry in self._relation_graph.get(src, [])[:5]:
                    tgt = entry.get("target", "")
                    if tgt in member_set:
                        rel_desc = entry.get("description") or (
                            f"{src} {entry.get('relation', '')} {tgt}"
                        )
                        rel_parts.append(f"- {rel_desc}")
            # GAP 3a: token-budget check — substitute sub-community reports when too large
            max_ctx_tokens = getattr(self.config, "graph_rag_community_max_context_tokens", 4000)
            entity_rel_text = "\n".join(ent_parts + rel_parts)
            if len(entity_rel_text) // 4 > max_ctx_tokens:
                # Children keys are level-qualified strings (e.g. "1_3") — look up directly
                parent_key = f"{level_idx}_{cid}"
                sub_community_keys = self._community_children.get(
                    parent_key, self._community_children.get(cid, [])
                )
                sub_reports = []
                ranked_sub = sorted(
                    [k for k in sub_community_keys if k in self._community_summaries],
                    key=lambda k: self._community_summaries[k].get("rank", 0.0),
                    reverse=True,
                )
                for sub_key in ranked_sub[:5]:
                    sub_reports.append(self._community_summaries[sub_key].get("full_content", ""))
                if sub_reports:
                    entity_rel_text = "\n\n".join(
                        f"Sub-community Report:\n{r}" for r in sub_reports
                    )
            else:
                entity_rel_text = "\n".join(ent_parts)
                if rel_parts:
                    entity_rel_text += "\n\nKey relationships:\n" + "\n".join(rel_parts[:20])
            # GAP 3c: Include claims in community context — auto-enabled if claims extraction is on
            include_claims = getattr(self.config, "graph_rag_community_include_claims", False)
            if not include_claims and getattr(self.config, "graph_rag_claims", False):
                include_claims = True
            claim_parts = []
            if include_claims and self._claims_graph:
                for entity in members[:10]:
                    for chunk_claims in self._claims_graph.values():
                        for claim in chunk_claims:
                            if claim.get("subject", "").lower() == entity.lower():
                                status = claim.get("status", "")
                                desc = claim.get("description", "")
                                claim_parts.append(f"  [{status}] {entity}: {desc}")
                if claim_parts:
                    claim_parts = claim_parts[:10]
            context = entity_rel_text
            if claim_parts:
                context += "\n\nClaims:\n" + "\n".join(claim_parts)
            prompt = (
                "You are analyzing a knowledge graph community.\n\n"
                "Community members and descriptions:\n"
                f"{context}\n\n"
                "Generate a community report as JSON with this exact structure:\n"
                "{\n"
                '  "title": "<2-5 word theme name>",\n'
                '  "summary": "<2-3 sentence overview>",\n'
                '  "findings": [\n'
                '    {"summary": "<headline>", "explanation": "<1-2 sentence detail>"},\n'
                "    ... up to 8 findings\n"
                "  ],\n"
                '  "rank": <float 0-10 importance>\n'
                "}\n"
                "Output only valid JSON. No markdown code blocks."
            )
            try:
                raw_text = self.llm.complete(
                    prompt,
                    system_prompt="You are an expert knowledge graph analyst.",
                )
                try:
                    raw_stripped = _re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
                    parsed = _json.loads(raw_stripped)
                    title = str(parsed.get("title", f"Community {cid}"))
                    summary = str(parsed.get("summary", ""))
                    findings = parsed.get("findings", [])
                    if not isinstance(findings, list):
                        findings = []
                    rank = float(parsed.get("rank", 5.0))
                except Exception:
                    title = f"Community {cid}"
                    summary = raw_text.strip() if raw_text else ""
                    findings = []
                    rank = 5.0
                findings_text = "\n".join(
                    f"- {f.get('summary', '')}: {f.get('explanation', '')}"
                    for f in findings
                    if isinstance(f, dict)
                )
                full_content = f"# {title}\n\n{summary}"
                if findings_text:
                    full_content += f"\n\nFindings:\n{findings_text}"
                return summary_key, {
                    "title": title,
                    "summary": summary,
                    "findings": findings,
                    "rank": rank,
                    "full_content": full_content,
                    "entities": members,
                    "size": len(members),
                    "level": level_idx,
                    "member_hash": new_hash,
                }
            except Exception as e:
                logger.debug(f"Community summary failed for community {summary_key}: {e}")
                return summary_key, {
                    "title": f"Community {cid}",
                    "summary": "",
                    "findings": [],
                    "rank": 5.0,
                    "full_content": "",
                    "entities": members,
                    "size": len(members),
                    "level": level_idx,
                    "member_hash": new_hash,
                }

        # Process levels from finest (highest idx) to coarsest — SEQUENTIALLY so sub-community
        # reports are available when summarizing parent communities (true bottom-up composition).
        for level_idx in sorted(self._community_levels.keys(), reverse=True):
            level_map = self._community_levels[level_idx]
            community_entities: dict = defaultdict(list)
            for entity, cid in level_map.items():
                community_entities[cid].append(entity)

            llm_items, template_items = [], []
            if query_hint:
                # Prioritise query-relevant communities so they consume LLM budget first.
                ranked = sorted(
                    community_entities.items(),
                    key=lambda kv: (_query_relevance(kv[1]), _community_score(kv[1])),
                    reverse=True,
                )
            else:
                ranked = sorted(
                    community_entities.items(),
                    key=lambda kv: _community_score(kv[1]),
                    reverse=True,
                )

            for rank_pos, (cid, members) in enumerate(ranked):
                summary_key = f"{level_idx}_{cid}"
                new_hash = _member_hash(level_idx, members)
                # Cache hit: always reuse regardless of triage
                existing = self._community_summaries.get(summary_key, {})
                if existing.get("member_hash") == new_hash and existing.get("full_content"):
                    summaries[summary_key] = dict(existing)
                    continue
                # Triage gates (applied in order):
                if _min_size > 0 and len(members) < _min_size:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                if _top_n_per_level > 0 and rank_pos >= _top_n_per_level:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                if _max_total > 0 and _llm_calls_issued >= _max_total:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                _llm_calls_issued += 1
                llm_items.append((level_idx, cid, members))

            for args in template_items:
                key, val = _template_summary(*args)
                summaries[key] = val
            for summary_key, summary_dict in self._executor.map(_summarise, llm_items):
                summaries[summary_key] = summary_dict
            # Make this level's summaries available before summarizing coarser levels
            self._community_summaries = dict(summaries)

        self._save_community_summaries()
        logger.info(f"GraphRAG: Community summaries generated for {len(summaries)} communities.")

    def _index_community_reports_in_vector_store(self) -> None:
        """Index community reports as synthetic documents in the vector store."""
        if not self._community_summaries:
            return
        docs_to_add = []
        for summary_key, cs in self._community_summaries.items():
            full_content = cs.get("full_content") or cs.get("summary", "")
            if not full_content:
                continue
            doc_id = f"__community__{summary_key}"
            content_hash = cs.get("member_hash", "")
            # Skip re-indexing if content is unchanged since last index
            if cs.get("indexed_hash") == content_hash and content_hash:
                continue
            cs["indexed_hash"] = content_hash
            docs_to_add.append(
                {
                    "id": doc_id,
                    "text": full_content,
                    "metadata": {
                        "graph_rag_type": "community_report",
                        "community_key": summary_key,
                        "level": cs.get("level", 0),
                        "rank": cs.get("rank", 5.0),
                        "title": cs.get("title", ""),
                        "source": "__community_report__",
                    },
                }
            )
        if docs_to_add:
            try:
                ids = [d["id"] for d in docs_to_add]
                texts = [d["text"] for d in docs_to_add]
                metadatas = [d["metadata"] for d in docs_to_add]
                embeddings = self.embedding.embed(texts)
                self._own_vector_store.add(ids, texts, embeddings, metadatas)
                logger.info(
                    f"GraphRAG: Indexed {len(docs_to_add)} community reports in vector store."
                )
            except Exception as e:
                logger.warning(f"Could not index community reports: {e}")

    def _global_search_map_reduce(self, query: str, cfg) -> str:
        """Map-reduce global search over community reports (Item 4)."""
        import json as _json
        import re as _re

        if not self._community_summaries:
            return ""

        min_score = getattr(cfg, "graph_rag_global_min_score", 20)
        top_points = getattr(cfg, "graph_rag_global_top_points", 50)

        # Filter summaries to the target level
        target_level = getattr(cfg, "graph_rag_community_level", 0)
        target_level_prefix = f"{target_level}_"
        level_summaries = {
            k: v for k, v in self._community_summaries.items() if k.startswith(target_level_prefix)
        }
        # Fall back to all summaries if nothing matches the target level
        if not level_summaries:
            level_summaries = self._community_summaries

        # TASK 12: Dynamic community pre-filter — cheap token-overlap relevance score
        _top_n_communities = getattr(cfg, "graph_rag_global_top_communities", 0)
        if _top_n_communities > 0 and len(level_summaries) > _top_n_communities:
            _query_words = set(query.lower().split())

            def _community_relevance(cs_item: tuple) -> float:
                _, cs = cs_item
                text = ((cs.get("title") or "") + " " + (cs.get("summary") or "")).lower()
                return len(_query_words & set(text.split())) / max(len(_query_words), 1)

            sorted_summaries = sorted(
                level_summaries.items(), key=_community_relevance, reverse=True
            )
            level_summaries = dict(sorted_summaries[:_top_n_communities])
            logger.debug("GraphRAG global: pre-filtered to top %d communities.", _top_n_communities)

        # Chunk reports so large reports don't get hard-truncated and later sections aren't lost
        _MAP_CHUNK_CHARS = int(getattr(cfg, "graph_rag_global_map_max_length", 500) or 500) * 4

        def _chunk_report(cid: str, cs: dict) -> list[tuple[str, str]]:
            """Split a community report into bounded chunks. Returns [(cid_chunk_id, text)]."""
            report = cs.get("full_content") or cs.get("summary", "")
            if not report:
                return []
            chunks = []
            for i in range(0, len(report), _MAP_CHUNK_CHARS):
                chunk_text = report[i : i + _MAP_CHUNK_CHARS]
                chunks.append((f"{cid}#{i}", chunk_text))
            return chunks

        # Build shuffled chunk list for unbiased map phase
        import random as _random

        all_chunks: list[tuple[str, str]] = []
        for cid, cs in level_summaries.items():
            all_chunks.extend(_chunk_report(cid, cs))
        rng = _random.Random(42)
        rng.shuffle(all_chunks)

        # TASK 14: Token-level compression of community report chunks before LLM map phase
        if getattr(cfg, "graph_rag_report_compress", False) is True and all_chunks:
            _ratio = getattr(cfg, "graph_rag_report_compress_ratio", 0.5)
            try:
                _lingua = self._ensure_llmlingua()
                _compressed_chunks = []
                for _cid, _text in all_chunks:
                    if not _text:
                        _compressed_chunks.append((_cid, _text))
                        continue
                    try:
                        _out = _lingua.compress_prompt(_text, rate=_ratio, force_tokens=["\n"])
                        _compressed_chunks.append((_cid, _out["compressed_prompt"]))
                    except Exception:
                        _compressed_chunks.append((_cid, _text))
                all_chunks = _compressed_chunks
                logger.info(
                    "GraphRAG map-reduce: compressed %d report chunks (ratio=%.2f)",
                    len(all_chunks),
                    _ratio,
                )
            except ImportError:
                logger.warning(
                    "graph_rag_report_compress=True but llmlingua not installed. "
                    "pip install axon[llmlingua]"
                )

        def _map_community(args):
            chunk_id, report_chunk = args
            if not report_chunk:
                return []
            prompt = (
                "---Role---\n"
                "You are a helpful assistant responding to questions about the dataset.\n\n"
                "---Goal---\n"
                "Generate a list of key points relevant to answering the question, "
                "based solely on the community report below.\n"
                "If the report is not relevant, return an empty JSON array.\n\n"
                f"---Question---\n{query}\n\n"
                f"---Community Report---\n{report_chunk}\n\n"
                "---Response Format---\n"
                'Return a JSON array: [{"point": "...", "score": 1-100}, ...]\n'
                "Higher score = more relevant. Output only valid JSON."
            )
            try:
                raw = self.llm.complete(
                    prompt,
                    system_prompt="You are a knowledge graph analysis assistant.",
                )
                raw_clean = _re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                points = _json.loads(raw_clean)
                if not isinstance(points, list):
                    return []
                return [
                    (float(p.get("score", 0)), str(p.get("point", "")))
                    for p in points
                    if isinstance(p, dict) and p.get("point")
                ]
            except Exception:
                return []

        # Map phase — parallel over shuffled chunks
        # TASK 14: Use a dedicated pool when graph_rag_map_workers is set, so map-reduce
        # does not starve the shared executor during concurrent ingest.
        all_points = []
        _map_workers_cfg = getattr(cfg, "graph_rag_map_workers", 0)
        try:
            if isinstance(_map_workers_cfg, int) and _map_workers_cfg > 0:
                from concurrent.futures import ThreadPoolExecutor as _TPE

                with _TPE(max_workers=_map_workers_cfg) as _map_pool:
                    results = list(_map_pool.map(_map_community, all_chunks))
            else:
                results = list(self._executor.map(_map_community, all_chunks))
            for point_list in results:
                all_points.extend(point_list)
        except Exception:
            return ""

        # Filter and sort
        filtered = [(score, point) for score, point in all_points if score >= min_score]
        filtered.sort(reverse=True)
        top = filtered[:top_points]

        if not top:
            return _GRAPHRAG_NO_DATA_ANSWER

        # --- REDUCE PHASE: token-budget assembly ---
        reduce_max_tokens = getattr(cfg, "graph_rag_global_reduce_max_tokens", 8000)
        analyst_lines = []
        token_estimate = 0
        for idx, (score, point) in enumerate(top):
            line = f"----Analyst {idx + 1}----\nImportance Score: {score:.0f}\n{point}"
            # Rough token estimate: 1 token ≈ 4 characters
            token_estimate += len(line) // 4
            if token_estimate > reduce_max_tokens:
                break
            analyst_lines.append(line)

        if not analyst_lines:
            return _GRAPHRAG_NO_DATA_ANSWER

        reduce_context = "\n\n".join(analyst_lines)
        reduce_max_length = getattr(cfg, "graph_rag_global_reduce_max_length", 500)
        reduce_prompt = (
            f"The following analytic reports have been generated for the query:\n\n"
            f"Query: {query}\n\n"
            f"Reports:\n\n{reduce_context}\n\n"
            f"Using the reports above, generate a comprehensive response to the query."
            f"\nRespond in at most {reduce_max_length} tokens."
        )
        reduce_system_prompt = _GRAPHRAG_REDUCE_SYSTEM_PROMPT
        if getattr(cfg, "graph_rag_global_allow_general_knowledge", False):
            reduce_system_prompt = (
                reduce_system_prompt
                + " You may supplement the provided reports with your own general knowledge"
                " where relevant."
            )
        try:
            response = self.llm.complete(
                reduce_prompt,
                system_prompt=reduce_system_prompt,
            )
            return response.strip() if response else _GRAPHRAG_NO_DATA_ANSWER
        except Exception as e:
            logger.debug(f"GraphRAG global reduce failed: {e}")
            # Fallback: return raw analyst summaries
            return "**Knowledge Graph Findings:**\n\n" + reduce_context

    def _get_incoming_relations(self, entity: str) -> list[dict]:
        """Return all relation entries where entity is the target (GAP 4: incoming edges)."""
        results = []
        ent_lower = entity.lower()
        for src, entries in self._relation_graph.items():
            for entry in entries:
                if entry.get("target", "").lower() == ent_lower:
                    results.append({**entry, "source": src, "direction": "incoming"})
        return results

    def _local_search_context(self, query: str, matched_entities: list, cfg) -> str:
        """Build structured GraphRAG local context using unified candidate ranking.

        TASK 11: Replaces fixed 25/50/25 budget split with joint ranking across all
        artifact types, then greedy-fill up to total_budget tokens.
        """
        if not matched_entities:
            return ""

        import re as _re

        # --- Phase 1: Setup ---
        total_budget = getattr(cfg, "graph_rag_local_max_context_tokens", 8000)
        top_k_entities = getattr(cfg, "graph_rag_local_top_k_entities", 10)
        top_k_relationships = getattr(cfg, "graph_rag_local_top_k_relationships", 10)
        include_weight = getattr(cfg, "graph_rag_local_include_relationship_weight", False)

        entity_weight = getattr(cfg, "graph_rag_local_entity_weight", 3.0)
        relation_weight = getattr(cfg, "graph_rag_local_relation_weight", 2.0)
        community_weight = getattr(cfg, "graph_rag_local_community_weight", 1.5)
        text_unit_weight = getattr(cfg, "graph_rag_local_text_unit_weight", 1.0)

        _query_tokens = set(query.lower().split()) | set(_re.split(r"[\s\W_]+", query.lower()))
        _boost = getattr(self.config, "graph_rag_exact_entity_boost", 3.0)

        def _entity_degree(ent: str) -> int:
            outgoing = len(self._relation_graph.get(ent.lower(), []))
            incoming = len(self._get_incoming_relations(ent))
            return outgoing + incoming

        def _entity_raw(ent: str) -> float:
            return _entity_degree(ent) * (_boost if ent.lower() in _query_tokens else 1.0)

        ranked_entities = sorted(matched_entities, key=_entity_raw, reverse=True)
        ranked_entities = ranked_entities[:top_k_entities]

        # --- Phase 2: Collect all candidates into a flat list ---
        candidates: list[dict] = []

        # Entities
        raw_scores = [_entity_raw(e) for e in ranked_entities]
        max_raw = max(raw_scores, default=1.0) or 1.0
        for ent, raw in zip(ranked_entities, raw_scores):
            node = self._entity_graph.get(ent.lower(), {})
            desc = node.get("description", "") if isinstance(node, dict) else ""
            ent_type = node.get("type", "") if isinstance(node, dict) else ""
            if not desc:
                continue
            line = f"  - {ent} [{ent_type}]: {desc}" if ent_type else f"  - {ent}: {desc}"
            candidates.append(
                {
                    "type": "entity",
                    "score": entity_weight * (raw / max_raw),
                    "text": line,
                    "tokens": len(line) // 4 + 1,
                    "key": ent.lower(),
                }
            )

        # Relations — collect outgoing (top 3/entity) + incoming (top 2/entity)
        def _mutual_count(e_entry, ranked):
            tgt = e_entry.get("target", "")
            return sum(
                1
                for ee in ranked
                if tgt in {x.get("target", "") for x in self._relation_graph.get(ee.lower(), [])}
            )

        rel_candidates_raw: list[tuple[float, dict, str]] = []
        seen_rel_keys: set = set()
        for ent in ranked_entities:
            ent_lower = ent.lower()
            outgoing = self._relation_graph.get(ent_lower, [])
            sorted_outgoing = sorted(
                outgoing, key=lambda e: _mutual_count(e, ranked_entities), reverse=True
            )
            for entry in sorted_outgoing[:3]:
                mc = _mutual_count(entry, ranked_entities)
                rel_candidates_raw.append((mc, entry, ent))
            for entry in self._get_incoming_relations(ent)[:2]:
                mc = _mutual_count(entry, ranked_entities)
                rel_candidates_raw.append((mc, entry, ent))

        # Sort by mutual count, cap at top_k_relationships, normalize
        rel_candidates_raw.sort(key=lambda x: x[0], reverse=True)
        rel_candidates_raw = rel_candidates_raw[:top_k_relationships]
        max_mutual = max((x[0] for x in rel_candidates_raw), default=0)
        for mc, entry, ent in rel_candidates_raw:
            src = entry.get("source", ent)
            desc = entry.get("description") or (
                f"{src} {entry.get('relation', '')} {entry.get('target', '')}"
            )
            weight_str = ""
            if include_weight and entry.get("weight"):
                weight_str = f" (weight: {entry['weight']})"
            line = f"  - {desc}{weight_str}"
            rel_key = line[:80]
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)
            within = (mc + 1) / (max_mutual + 1)
            candidates.append(
                {
                    "type": "relation",
                    "score": relation_weight * within,
                    "text": line,
                    "tokens": len(line) // 4 + 1,
                }
            )

        # Communities — finest level, deduplicated by summary_key
        if self._community_levels and self._community_summaries:
            finest = max(self._community_levels.keys()) if self._community_levels else None
            seen_community_keys: set = set()
            community_raws: list[tuple[float, str, str]] = []
            for ent in ranked_entities:
                cid = (
                    self._community_levels.get(finest, {}).get(ent.lower())
                    if finest is not None
                    else None
                )
                if cid is None:
                    continue
                summary_key = f"{finest}_{cid}"
                if summary_key in seen_community_keys:
                    continue
                seen_community_keys.add(summary_key)
                cs = self._community_summaries.get(summary_key, {})
                if not cs:
                    cs = self._community_summaries.get(str(cid), {})
                summary = cs.get("summary", "")
                if not summary:
                    continue
                sentences = summary.split(". ")
                snippet = ". ".join(sentences[:3]).strip()
                if snippet and not snippet.endswith("."):
                    snippet += "."
                title = cs.get("title", f"Community {cid}")
                community_text = f"**Community Context ({title}):**\n  {snippet}"
                rank = cs.get("rank", 0.5)
                community_raws.append((rank, community_text, summary_key))
            max_rank = max((r[0] for r in community_raws), default=1.0) or 1.0
            for rank, ctext, _ in community_raws:
                within = rank / max_rank
                candidates.append(
                    {
                        "type": "community",
                        "score": community_weight * within,
                        "text": ctext,
                        "tokens": len(ctext) // 4 + 1,
                    }
                )

        # Text units — from entity chunk_ids, cap at 20, rank by relation count
        seen_chunks: set = set()
        text_unit_ids_all: list[str] = []
        for ent in ranked_entities:
            node = self._entity_graph.get(ent.lower(), {})
            chunk_ids = node.get("chunk_ids", []) if isinstance(node, dict) else []
            for cid in chunk_ids:
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    text_unit_ids_all.append(cid)
                if len(text_unit_ids_all) >= 20:
                    break
            if len(text_unit_ids_all) >= 20:
                break

        def _tu_rel_count(cid: str) -> int:
            return len(self._text_unit_relation_map.get(cid, []))

        max_rel = max((_tu_rel_count(c) for c in text_unit_ids_all), default=0)
        for cid in text_unit_ids_all:
            try:
                docs = self.vector_store.get_by_ids([cid])
                if not docs:
                    continue
                text_snippet = docs[0].get("text", "")[:200].strip()
                if not text_snippet:
                    continue
                line = f"  [{cid}] {text_snippet}..."
                rc = _tu_rel_count(cid)
                within = (rc + 1) / (max_rel + 1)
                candidates.append(
                    {
                        "type": "text_unit",
                        "score": text_unit_weight * within,
                        "text": line,
                        "tokens": len(line) // 4 + 1,
                    }
                )
            except Exception:
                pass

        # Claims — high-signal factual assertions, fixed score
        if self._claims_graph:
            claim_lines: list[str] = []
            for ent in ranked_entities[:3]:
                for chunk_claims in self._claims_graph.values():
                    for claim in chunk_claims:
                        if claim.get("subject", "").lower() == ent.lower():
                            status = claim.get("status", "SUSPECTED")
                            desc = claim.get("description", "")
                            if desc:
                                claim_lines.append(f"  - [{status}] {desc}")
            for line in claim_lines[:10]:
                candidates.append(
                    {
                        "type": "claim",
                        "score": entity_weight * 1.0,
                        "text": line,
                        "tokens": len(line) // 4 + 1,
                    }
                )

        # --- Phase 3: Sort and greedy-fill ---
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected: list[dict] = []
        used = 0
        for c in candidates:
            if used + c["tokens"] <= total_budget:
                selected.append(c)
                used += c["tokens"]
            # continue (not break) — smaller items later may still fit

        # --- Phase 4: Reassemble into section-formatted output ---
        by_type: dict[str, list[str]] = {
            "entity": [],
            "relation": [],
            "community": [],
            "text_unit": [],
            "claim": [],
        }
        for c in selected:
            by_type[c["type"]].append(c["text"])

        parts: list[str] = []
        if by_type["entity"]:
            parts.append("**Matched Entities:**\n" + "\n".join(by_type["entity"]))
        if by_type["relation"]:
            parts.append("**Relevant Relationships:**\n" + "\n".join(by_type["relation"]))
        for ctext in by_type["community"]:
            parts.append(ctext)
        if by_type["text_unit"]:
            parts.append("**Source Text Units:**\n" + "\n".join(by_type["text_unit"]))
        if by_type["claim"]:
            parts.append("**Claims / Facts:**\n" + "\n".join(by_type["claim"]))

        return "\n\n".join(parts)

    def _entity_matches(self, q_entity: str, g_entity: str) -> float:
        """Return Jaccard-based match score between 0.0 and 1.0."""
        q = q_entity.lower().strip()
        g = g_entity.lower().strip()
        if q == g:
            return 1.0
        q_tokens = set(q.split())
        g_tokens = set(g.split())
        if len(q_tokens) == 1 and len(g_tokens) == 1:
            return 0.0  # single tokens must match exactly
        intersection = q_tokens & g_tokens
        union = q_tokens | g_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard if jaccard >= 0.4 else 0.0

    _VALID_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "GEO", "EVENT", "CONCEPT", "PRODUCT"}

    _HOLISTIC_KEYWORDS = frozenset(
        [
            "summarize",
            "summary",
            "overview",
            "all",
            "every",
            "compare",
            "list all",
            "across",
            "throughout",
            "entire",
            "whole",
            "global",
            "what are all",
            "how many",
            "count",
            "trend",
            "pattern",
            "themes",
            "main topics",
        ]
    )

    def _classify_query_needs_graphrag(self, query: str, mode: str) -> bool:
        """Return True if GraphRAG global search is warranted for this query (TASK 14)."""
        if mode == "heuristic":
            q_lower = query.lower()
            if any(kw in q_lower for kw in self._HOLISTIC_KEYWORDS):
                return True
            if len(query.split()) > 20:
                return True
            return False
        elif mode == "llm":
            prompt = (
                "Does the following question require a holistic, corpus-wide analysis "
                "(e.g. summarization, comparison, listing all topics) rather than a "
                "targeted factual lookup? Answer only YES or NO.\n\nQuestion: " + query
            )
            try:
                ans = self.llm.complete(prompt, max_tokens=5).strip().upper()
                return ans.startswith("Y")
            except Exception:
                return False
        return False

    # ── keyword sets for multi-class router ───────────────────────────────────
    _SYNTHESIS_KEYWORDS: frozenset = frozenset(
        {
            "summarize",
            "overview",
            "compare",
            "contrast",
            "explain",
            "discuss",
            "survey",
            "themes",
            "analysis",
        }
    )
    _TABLE_KEYWORDS: frozenset = frozenset(
        {
            "table",
            "row",
            "column",
            "value",
            "count",
            "average",
            "maximum",
            "minimum",
            "statistic",
            "how many",
            "list all",
        }
    )
    _ENTITY_KEYWORDS: frozenset = frozenset(
        {
            "relationship",
            "related to",
            "who",
            "works with",
            "connected",
            "linked",
            "colleague",
            "dependency",
        }
    )
    _CORPUS_KEYWORDS: frozenset = frozenset(
        {
            "all documents",
            "entire corpus",
            "everything",
            "main topics",
            "key themes",
            "across all",
        }
    )

    def _ensure_llmlingua(self):
        """Lazy-initialise the LLMLingua-2 prompt compressor (TASK 14)."""
        if not hasattr(self, "_llmlingua") or self._llmlingua is None:
            from llmlingua import PromptCompressor

            _model = getattr(
                self.config,
                "graph_rag_llmlingua_model",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            )
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG LLMLingua: loading model '%s'%s…", _model, " (local)" if _local else ""
            )
            self._llmlingua = PromptCompressor(
                model_name=_model,
                use_llmlingua2=True,
                device_map="cpu",
                **({"local_files_only": True} if _local else {}),
            )
        return self._llmlingua

    def _ensure_gliner(self):
        """Lazy-initialise the GLiNER NER model (TASK 14)."""
        if not hasattr(self, "_gliner_model") or self._gliner_model is None:
            from gliner import GLiNER

            _model = getattr(self.config, "graph_rag_gliner_model", "urchade/gliner_medium-v2.1")
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG GLiNER: loading model '%s'%s…", _model, " (local)" if _local else ""
            )
            self._gliner_model = GLiNER.from_pretrained(_model, local_files_only=_local)
        return self._gliner_model

    _GLINER_LABELS = ["person", "organization", "location", "event", "concept", "product"]
    _GLINER_TYPE_MAP = {
        "person": "PERSON",
        "organization": "ORGANIZATION",
        "location": "GEO",
        "event": "EVENT",
        "concept": "CONCEPT",
        "product": "PRODUCT",
    }

    def _ensure_rebel(self):
        """Lazy-initialise the REBEL relation extraction pipeline (P2)."""
        if not hasattr(self, "_rebel_pipeline") or self._rebel_pipeline is None:
            try:
                from transformers import pipeline as _hf_pipeline
            except ImportError:
                raise ImportError("REBEL backend requires transformers. pip install axon[rebel]")
            _model = getattr(self.config, "graph_rag_rebel_model", "Babelscape/rebel-large")
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG REBEL: loading model '%s'%s…",
                _model,
                " (local)" if _local else " (first-run download may take time)",
            )
            self._rebel_pipeline = _hf_pipeline(
                "text2text-generation",
                model=_model,
                tokenizer=_model,
                device=-1,  # CPU — avoids CUDA dependency
                **({"local_files_only": True} if _local else {}),
            )
        return self._rebel_pipeline

    @staticmethod
    def _parse_rebel_output(text: str) -> list[dict]:
        """Parse REBEL's ``<triplet> SUBJ <subj> OBJ <obj> REL`` output format.

        REBEL encodes triplets as:
            <triplet> head_entity <subj> tail_entity <obj> relation_type

        Returns a list of relation dicts compatible with ``_extract_relations``.
        """
        triplets: list[dict] = []
        subject = object_ = relation = ""
        current = "x"
        tokens = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
        for token in tokens:
            if token == "<triplet>":
                if subject and relation and object_:
                    triplets.append(
                        {
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "object": object_.strip(),
                            "description": "",
                            "strength": 5,
                        }
                    )
                subject = object_ = relation = ""
                current = "t"
            elif token == "<subj>":
                current = "s"
            elif token == "<obj>":
                current = "o"
            elif current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
        # Flush final triplet
        if subject and relation and object_:
            triplets.append(
                {
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip(),
                    "description": "",
                    "strength": 5,
                }
            )
        return triplets

    def _extract_relations_rebel(self, text: str) -> list[dict]:
        """Extract relations using REBEL — no LLM call, structured triplet output (P2)."""
        try:
            pipe = self._ensure_rebel()
            # REBEL's tokeniser caps at 512 tokens; truncate input to ~1 000 chars
            outputs = pipe(text[:1000], max_length=512, return_tensors=False)
            raw = outputs[0]["generated_text"] if outputs else ""
            return self._parse_rebel_output(raw)[:15]
        except ImportError:
            logger.warning(
                "graph_rag_relation_backend='rebel' but transformers is not installed. "
                "pip install axon[rebel]"
            )
            return []
        except Exception:
            return []

    def _extract_entities_gliner(self, text: str) -> list[dict]:
        """Extract entities using GLiNER (TASK 14 — NER-only; no LLM call)."""
        model = self._ensure_gliner()
        try:
            preds = model.predict_entities(text[:3000], self._GLINER_LABELS, threshold=0.5)
            seen: set = set()
            entities = []
            for p in preds:
                name = p["text"].strip()
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                entities.append(
                    {
                        "name": name,
                        "type": self._GLINER_TYPE_MAP.get(p["label"].lower(), "CONCEPT"),
                        "description": "",
                    }
                )
            return entities[:20]
        except Exception:
            return []

    def _extract_entities_light(self, text: str) -> list[dict]:
        """Lightweight entity extraction using regex noun-phrase heuristics (no LLM).

        Picks capitalized multi-word phrases (2–4 tokens) from the text.
        Returns up to 20 entities with type=CONCEPT and empty description.
        """
        import re as _re

        pattern = _re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b")
        seen: set = set()
        entities: list[dict] = []
        for match in pattern.finditer(text):
            phrase = match.group(1)
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                entities.append({"name": phrase, "type": "CONCEPT", "description": ""})
            if len(entities) >= 20:
                break
        return entities

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using the LLM.

        Returns a list of dicts with shape:
          {"name": str, "type": str, "description": str}
        Returns an empty list on failure or when the LLM produces no output.
        """
        # A3: light tier — skip LLM entirely
        if getattr(self.config, "graph_rag_depth", "standard") == "light":
            return self._extract_entities_light(text)

        # TASK 14: GLiNER fast-path — skip LLM for NER when backend is "gliner"
        if getattr(self.config, "graph_rag_ner_backend", "llm") == "gliner":
            return self._extract_entities_gliner(text)

        prompt = (
            "Extract the key named entities from the following text.\n"
            "For each entity output one line:\n"
            "  ENTITY_NAME | ENTITY_TYPE | one-sentence description\n"
            "ENTITY_TYPE must be one of: PERSON, ORGANIZATION, GEO, EVENT, CONCEPT, PRODUCT\n"
            "No bullets, numbering, or extra text. If no entities, output nothing.\n\n"
            + text[:3000]
        )
        try:
            raw = self.llm.complete(
                prompt, system_prompt="You are a named entity extraction specialist."
            )
            entities = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    ent_type = parts[1].upper()
                    if ent_type not in self._VALID_ENTITY_TYPES:
                        ent_type = "CONCEPT"
                    entities.append(
                        {
                            "name": parts[0],
                            "type": ent_type,
                            "description": parts[2],
                        }
                    )
                elif len(parts) == 2:
                    # Old 2-column format — no type
                    entities.append(
                        {
                            "name": parts[0],
                            "type": "UNKNOWN",
                            "description": parts[1],
                        }
                    )
                else:
                    entities.append(
                        {
                            "name": parts[0],
                            "type": "UNKNOWN",
                            "description": "",
                        }
                    )
            return entities[:20]
        except Exception:
            return []

    def _extract_relations(self, text: str) -> list[dict]:
        """Extract SUBJECT | RELATION | OBJECT | description quads from text via the LLM.

        Returns up to 15 relation dicts with shape
        {"subject": str, "relation": str, "object": str, "description": str}.
        Returns an empty list on failure or when the LLM produces no output.
        """
        # A3: light tier skips all relation extraction
        if getattr(self.config, "graph_rag_depth", "standard") == "light":
            return []

        # P2: REBEL fast-path — skip LLM when backend is "rebel"
        if getattr(self.config, "graph_rag_relation_backend", "llm") == "rebel":
            return self._extract_relations_rebel(text)

        prompt = (
            "Extract key relationships from the following text.\n"
            "For each relationship output one line:\n"
            "  SUBJECT | RELATION | OBJECT | one-sentence description | strength (1-10)\n"
            "Strength: 1=weak/incidental, 10=core/defining. "
            "No bullets or extra text. If no clear relationships, output nothing.\n\n" + text[:3000]
        )
        try:
            raw = self.llm.complete(
                prompt, system_prompt="You are a knowledge graph extraction specialist."
            )
            triples = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    try:
                        strength = max(1, min(10, int(parts[4])))
                    except (ValueError, IndexError):
                        strength = 5
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": parts[3],
                            "strength": strength,
                        }
                    )
                elif len(parts) >= 4:
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": parts[3],
                            "strength": 5,
                        }
                    )
                elif len(parts) == 3 and all(parts):
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": "",
                            "strength": 5,
                        }
                    )
            return triples[:15]
        except Exception:
            return []

    def _embed_entities(self, entity_keys: list) -> None:
        """Embed entity descriptions; store in _entity_embeddings (Item 5)."""
        to_embed = []
        for key in entity_keys:
            node = self._entity_graph.get(key, {})
            if isinstance(node, dict):
                desc = node.get("description", "")
                if desc and key not in self._entity_embeddings:
                    to_embed.append((key, f"{key}: {desc}"))
        if not to_embed:
            return
        keys, texts = zip(*to_embed)
        try:
            vectors = self.embedding.embed(list(texts))
            for k, v in zip(keys, vectors):
                self._entity_embeddings[k] = v if isinstance(v, list) else list(v)
            self._save_entity_embeddings()
            logger.info(f"GraphRAG: Embedded {len(to_embed)} entity descriptions.")
        except Exception as e:
            logger.debug(f"Entity embedding failed: {e}")

    def _match_entities_by_embedding(self, query: str, top_k: int = 5) -> list:
        """Return entity keys matching the query by embedding cosine similarity (Item 5)."""
        if not self._entity_embeddings:
            return []
        try:
            import numpy as np

            q_vec = np.array(self.embedding.embed_query(query))
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return []
            scored = []
            for entity_key, e_vec in self._entity_embeddings.items():
                ev = np.array(e_vec)
                ev_norm = np.linalg.norm(ev)
                if ev_norm == 0:
                    continue
                sim = float(np.dot(q_vec, ev) / (q_norm * ev_norm))
                scored.append((sim, entity_key))
            threshold = getattr(self.config, "graph_rag_entity_match_threshold", 0.5)
            scored = [(s, k) for s, k in scored if s >= threshold]
            scored.sort(reverse=True)
            return [k for _, k in scored[:top_k]]
        except Exception as e:
            logger.debug(f"Entity embedding match failed: {e}")
            return []

    def _extract_claims(self, text: str) -> list:
        """Extract factual claims from text (Item 11). Returns list of claim dicts."""
        prompt = (
            "Extract factual claims from the following text.\n"
            "For each claim output one line:\n"
            "  SUBJECT | OBJECT | CLAIM_TYPE | STATUS | DESCRIPTION | START_DATE | END_DATE | SOURCE_QUOTE\n"
            "  STATUS must be: TRUE, FALSE, or SUSPECTED\n"
            "  CLAIM_TYPE examples: acquisition, partnership, product_launch, regulatory_action, financial\n"
            "  START_DATE and END_DATE: ISO8601 date (YYYY-MM-DD) or 'unknown' if not mentioned\n"
            "  SOURCE_QUOTE: verbatim short quote from the text (max 100 chars) or 'none'\n"
            "No bullets or extra text. If no clear claims, output nothing.\n\n" + text[:3000]
        )
        try:
            raw = self.llm.complete(
                prompt,
                system_prompt="You are a fact extraction specialist.",
            )
            claims = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 8:
                    status = parts[3].upper()
                    if status not in ("TRUE", "FALSE", "SUSPECTED"):
                        status = "SUSPECTED"
                    claims.append(
                        {
                            "subject": parts[0],
                            "object": parts[1],
                            "type": parts[2],
                            "status": status,
                            "description": parts[4],
                            "start_date": parts[5] if parts[5] != "unknown" else None,
                            "end_date": parts[6] if parts[6] != "unknown" else None,
                            "source_text": parts[7] if parts[7] != "none" else None,
                            "text_unit_id": None,  # filled in by caller
                        }
                    )
                elif len(parts) >= 5:
                    status = parts[3].upper()
                    if status not in ("TRUE", "FALSE", "SUSPECTED"):
                        status = "SUSPECTED"
                    claims.append(
                        {
                            "subject": parts[0],
                            "object": parts[1],
                            "type": parts[2],
                            "status": status,
                            "description": parts[4],
                            "start_date": None,
                            "end_date": None,
                            "source_text": None,
                            "text_unit_id": None,
                        }
                    )
            return claims[:10]
        except Exception:
            return []

    def _resolve_entity_aliases(self) -> int:
        """Merge semantically equivalent entity nodes into a single canonical node (P1).

        Uses cosine similarity on entity-name embeddings from the active embedding model.
        Entities whose name-embeddings exceed ``graph_rag_entity_resolve_threshold`` are
        grouped; within each group the node with the most chunk_ids becomes canonical and
        all alias nodes are merged into it (chunk_ids, description, relations).

        Returns the number of alias nodes merged (0 if nothing to merge).
        """
        import numpy as np

        threshold = getattr(self.config, "graph_rag_entity_resolve_threshold", 0.92)
        max_entities = getattr(self.config, "graph_rag_entity_resolve_max", 5000)

        keys = [k for k, v in self._entity_graph.items() if isinstance(v, dict)]
        n = len(keys)
        if n < 2:
            return 0
        if n > max_entities:
            logger.warning(
                "GraphRAG entity resolution: entity graph has %d nodes (limit=%d). "
                "Skipping alias resolution — increase graph_rag_entity_resolve_max to enable.",
                n,
                max_entities,
            )
            return 0

        # Embed entity names (not descriptions) — aliases share similar surface forms
        try:
            raw_emb = self.embedding.embed(keys)
        except Exception as exc:
            logger.warning("GraphRAG entity resolution: embedding failed (%s). Skipping.", exc)
            return 0

        emb = np.array(raw_emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms  # unit-normalise for cosine via dot product

        # Union-Find for grouping similar entities
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        # O(n²) pairwise similarity — acceptable for n ≤ max_entities (default 5 000)
        sim = emb @ emb.T  # shape (n, n)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    _union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(_find(i), []).append(i)

        merged = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            # Canonical = entity with most chunk_ids (highest coverage)
            canon_idx = max(
                members,
                key=lambda i: len(self._entity_graph[keys[i]].get("chunk_ids", [])),
            )
            canon_key = keys[canon_idx]
            canon_node = self._entity_graph[canon_key]

            for idx in members:
                if idx == canon_idx:
                    continue
                alias_key = keys[idx]
                alias_node = self._entity_graph[alias_key]

                # Merge chunk_ids
                for cid in alias_node.get("chunk_ids", []):
                    if cid not in canon_node["chunk_ids"]:
                        canon_node["chunk_ids"].append(cid)

                # Inherit description if canonical has none
                if not canon_node.get("description") and alias_node.get("description"):
                    canon_node["description"] = alias_node["description"]

                # Migrate relation_graph entries keyed by the alias
                if alias_key in self._relation_graph:
                    canon_rels = self._relation_graph.setdefault(canon_key, [])
                    canon_rels.extend(self._relation_graph.pop(alias_key))

                # Rewrite subject/object references inside all relation lists
                for rel_list in self._relation_graph.values():
                    for rel in rel_list:
                        if isinstance(rel, dict):
                            if rel.get("subject", "").lower() == alias_key:
                                rel["subject"] = canon_key
                            if rel.get("object", "").lower() == alias_key:
                                rel["object"] = canon_key

                del self._entity_graph[alias_key]
                merged += 1

            # Refresh frequency for canonical
            canon_node["frequency"] = len(canon_node["chunk_ids"])

        if merged > 0:
            logger.info(
                "GraphRAG entity resolution: merged %d alias nodes into canonical entities "
                "(threshold=%.2f, %d nodes remaining).",
                merged,
                threshold,
                len(self._entity_graph),
            )
            self._community_graph_dirty = True

        return merged

    def _canonicalize_entity_descriptions(self) -> None:
        """Synthesize canonical descriptions for entities with multiple descriptions (Item 10)."""
        min_occ = getattr(self.config, "graph_rag_canonicalize_min_occurrences", 3)
        to_canonicalize = {
            k: descs
            for k, descs in self._entity_description_buffer.items()
            if len(set(descs)) > 1 and len(descs) >= min_occ
        }
        if not to_canonicalize:
            return

        logger.info(f"GraphRAG: Canonicalizing descriptions for {len(to_canonicalize)} entities...")

        def _synthesize(args):
            entity_key, descs = args
            unique_descs = list(dict.fromkeys(descs))[:10]  # deduplicate, cap at 10
            node = self._entity_graph.get(entity_key, {})
            entity_type = node.get("type", "UNKNOWN") if isinstance(node, dict) else "UNKNOWN"
            numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(unique_descs))
            prompt = (
                f"Entity: {entity_key}\nType: {entity_type}\n\n"
                "The following descriptions were extracted from different documents:\n"
                f"{numbered}\n\n"
                "Write a single comprehensive description that synthesizes all of the above. "
                "Output only the description, no preamble."
            )
            try:
                result = self.llm.complete(
                    prompt,
                    system_prompt="You are a knowledge graph curation specialist.",
                )
                return entity_key, result.strip() if result else unique_descs[0]
            except Exception:
                return entity_key, unique_descs[0]

        results = list(self._executor.map(_synthesize, to_canonicalize.items()))
        changed = False
        for entity_key, canonical_desc in results:
            node = self._entity_graph.get(entity_key)
            if isinstance(node, dict) and canonical_desc:
                node["description"] = canonical_desc
                changed = True
        if changed:
            self._save_entity_graph()
        self._entity_description_buffer.clear()

    def _canonicalize_relation_descriptions(self) -> None:
        """Synthesize canonical descriptions for repeated (subject, object) pairs (GAP 3b)."""
        if not getattr(self.config, "graph_rag_canonicalize_relations", False):
            return
        min_occ = getattr(self.config, "graph_rag_canonicalize_relations_min_occurrences", 2)
        to_canonicalize = {
            pair: descs
            for pair, descs in self._relation_description_buffer.items()
            if len(descs) >= min_occ
        }
        if not to_canonicalize:
            return

        def _synthesize(args):
            (src, tgt), descs = args
            unique_descs = list(dict.fromkeys(descs))[:10]
            numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(unique_descs))
            prompt = (
                f"Relationship: {src} → {tgt}\n\n"
                f"The following descriptions were extracted from different documents:\n{numbered}\n\n"
                "Write a single comprehensive description that synthesizes all of the above. "
                "Output only the description, no preamble."
            )
            try:
                result = self.llm.complete(
                    prompt,
                    system_prompt="You are a knowledge graph curation specialist.",
                )
                return (src, tgt), result.strip() if result else unique_descs[0]
            except Exception:
                return (src, tgt), unique_descs[0]

        results = list(self._executor.map(_synthesize, to_canonicalize.items()))
        changed = False
        for (src, tgt), canonical_desc in results:
            if src in self._relation_graph:
                for entry in self._relation_graph[src]:
                    if entry.get("target", "").lower() == tgt and canonical_desc:
                        entry["description"] = canonical_desc
                        changed = True
        if changed:
            self._save_relation_graph()
        self._relation_description_buffer.clear()

    def _prune_entity_graph(self, deleted_ids: set) -> None:
        """Remove deleted chunk IDs from entity graph and relation graph.

        Entries that become empty are deleted entirely.  Persists changes to disk.
        """
        eg_changed = False
        for entity in list(self._entity_graph):
            node = self._entity_graph[entity]
            chunk_ids = node.get("chunk_ids", [])
            after = [d for d in chunk_ids if d not in deleted_ids]
            if len(after) != len(chunk_ids):
                eg_changed = True
                if after:
                    self._entity_graph[entity]["chunk_ids"] = after
                    # Recompute frequency after pruning
                    self._entity_graph[entity]["frequency"] = len(after)
                else:
                    del self._entity_graph[entity]
        if eg_changed:
            self._save_entity_graph()

        rg_changed = False
        for src in list(self._relation_graph):
            before = self._relation_graph[src]
            after = [entry for entry in before if entry.get("chunk_id") not in deleted_ids]
            if len(after) != len(before):
                rg_changed = True
                if after:
                    self._relation_graph[src] = after
                else:
                    del self._relation_graph[src]
        if rg_changed:
            self._save_relation_graph()

        # Prune claims graph (Item 11)
        if hasattr(self, "_claims_graph"):
            claims_changed = False
            for chunk_id in list(self._claims_graph):
                if chunk_id in deleted_ids:
                    del self._claims_graph[chunk_id]
                    claims_changed = True
            if claims_changed:
                self._save_claims_graph()
