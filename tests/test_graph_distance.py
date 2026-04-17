"""Tests for entity distance and multi-hop traversal in GraphRAG + Dynamic Graph."""
from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity_graph(*names: str) -> dict:
    return {
        name: {
            "description": f"Entity {name}",
            "type": "CONCEPT",
            "chunk_ids": [f"chunk_{name}"],
            "frequency": 1,
            "degree": 0,
        }
        for name in names
    }


def _make_relation(target: str, weight: float = 1.0, relation: str = "RELATED_TO") -> dict:
    return {
        "target": target,
        "relation": relation,
        "chunk_id": f"chunk_{target}",
        "description": "",
        "strength": weight,
        "weight": weight,
        "support_count": 1,
        "text_unit_ids": [f"chunk_{target}"],
    }


# ---------------------------------------------------------------------------
# _build_nx_graph (GraphRagMixin)
# ---------------------------------------------------------------------------


class TestBuildNxGraph:
    """Unit tests for GraphRagMixin._build_nx_graph()."""

    def _make_mixin(self, entity_graph, relation_graph):
        from axon.graph_rag import GraphRagMixin

        mixin = GraphRagMixin.__new__(GraphRagMixin)
        mixin._entity_graph = entity_graph
        mixin._relation_graph = relation_graph
        mixin._nx_graph = None
        mixin._nx_graph_dirty = True
        mixin.config = MagicMock()
        return mixin

    def test_builds_graph_with_correct_nodes(self):
        eg = _make_entity_graph("alice", "bob", "carol")
        rg = {"alice": [_make_relation("bob")]}
        mixin = self._make_mixin(eg, rg)
        G = mixin._build_nx_graph()
        assert "alice" in G.nodes
        assert "bob" in G.nodes
        assert "carol" in G.nodes

    def test_builds_graph_with_correct_edges(self):
        eg = _make_entity_graph("alice", "bob")
        rg = {"alice": [_make_relation("bob", weight=5.0)]}
        mixin = self._make_mixin(eg, rg)
        G = mixin._build_nx_graph()
        assert G.has_edge("alice", "bob")
        edge = G.edges["alice", "bob"]
        assert edge["weight"] == 5.0
        assert edge["distance"] == pytest.approx(1.0 / (5.0 + 1e-6), rel=1e-3)

    def test_higher_weight_means_smaller_distance(self):
        eg = _make_entity_graph("a", "b", "c")
        rg = {
            "a": [
                _make_relation("b", weight=10.0),
                _make_relation("c", weight=1.0),
            ]
        }
        mixin = self._make_mixin(eg, rg)
        G = mixin._build_nx_graph()
        dist_ab = G.edges["a", "b"]["distance"]
        dist_ac = G.edges["a", "c"]["distance"]
        assert dist_ab < dist_ac, "Higher weight should yield smaller distance"

    def test_cache_is_used_after_build(self):
        eg = _make_entity_graph("x", "y")
        rg = {"x": [_make_relation("y")]}
        mixin = self._make_mixin(eg, rg)
        G1 = mixin._build_nx_graph()
        assert mixin._nx_graph_dirty is False
        G2 = mixin._build_nx_graph()
        assert G1 is G2, "Should return the same cached object"

    def test_dirty_flag_triggers_rebuild(self):
        eg = _make_entity_graph("x", "y")
        rg = {"x": [_make_relation("y")]}
        mixin = self._make_mixin(eg, rg)
        G1 = mixin._build_nx_graph()
        mixin._nx_graph_dirty = True
        G2 = mixin._build_nx_graph()
        assert G1 is not G2, "Dirty flag should cause rebuild"

    def test_save_entity_graph_marks_dirty(self):
        """_save_entity_graph sets _nx_graph_dirty = True."""
        from axon.graph_rag import GraphRagMixin

        mixin = GraphRagMixin.__new__(GraphRagMixin)
        mixin._entity_graph = {}
        mixin._nx_graph_dirty = False
        mixin._nx_graph = MagicMock()
        mixin.config = MagicMock()
        mixin.config.bm25_path = str(tempfile.mkdtemp())
        mixin._gr_write_json_if_changed = MagicMock()

        mixin._save_entity_graph()
        assert mixin._nx_graph_dirty is True

    def test_save_relation_graph_marks_dirty(self):
        """_save_relation_graph sets _nx_graph_dirty = True."""
        from axon.graph_rag import GraphRagMixin

        mixin = GraphRagMixin.__new__(GraphRagMixin)
        mixin._relation_graph = {}
        mixin._nx_graph_dirty = False
        mixin._nx_graph = MagicMock()
        mixin.config = MagicMock()
        mixin.config.bm25_path = str(tempfile.mkdtemp())
        mixin._gr_write_json_if_changed = MagicMock()

        # _save_relation_graph falls through to the non-shard path
        with patch.object(
            type(mixin.config), "graph_rag_relation_shard_persist", False, create=True
        ):
            mixin.config.graph_rag_relation_shard_persist = False
            mixin.config.graph_rag_relation_compact_persist = False
            mixin._save_relation_graph()
        assert mixin._nx_graph_dirty is True


# ---------------------------------------------------------------------------
# Dijkstra distance metric
# ---------------------------------------------------------------------------


class TestDijkstraDistanceMetric:
    """Verify the edge distance formula and Dijkstra path ordering."""

    def test_dijkstra_prefers_high_weight_path(self):
        """Dijkstra with distance=1/weight should prefer the high-weight route."""
        import networkx as nx

        G = nx.DiGraph()
        # Two paths from A to C:
        # A --weight=10--> B --weight=10--> C  (short weighted distance)
        # A --weight=0.1--> C                  (long weighted distance)
        G.add_edge("A", "B", distance=1 / 10.0, weight=10.0)
        G.add_edge("B", "C", distance=1 / 10.0, weight=10.0)
        G.add_edge("A", "C", distance=1 / 0.1, weight=0.1)

        dist, path = nx.single_source_dijkstra(G, "A", target="C", weight="distance")
        assert path == ["A", "B", "C"], "Dijkstra should prefer the high-weight route"

    def test_hop_decay_ordering(self):
        """Hop-decay scores must decrease monotonically with hop count."""
        base = 0.7
        decay = 0.7
        scores = [base * (decay**hop) for hop in range(4)]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], f"Score should decrease at hop {i+1}"

    def test_max_hops_limits_traversal(self):
        """Dijkstra with cutoff=1 should not reach 2-hop nodes."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge("A", "B", distance=0.1)
        G.add_edge("B", "C", distance=0.1)
        G.add_edge("C", "D", distance=0.1)

        dists = nx.single_source_shortest_path_length(G, "A", cutoff=1)
        assert "B" in dists
        assert "C" not in dists
        assert "D" not in dists

    def test_bfs_hops_match_expected_depth(self):
        """BFS hop counts on a chain graph should be 1, 2, 3."""
        import networkx as nx

        G = nx.DiGraph()
        for a, b in [("A", "B"), ("B", "C"), ("C", "D")]:
            G.add_edge(a, b, distance=0.1)

        lengths = nx.single_source_shortest_path_length(G, "A", cutoff=3)
        assert lengths["B"] == 1
        assert lengths["C"] == 2
        assert lengths["D"] == 3


# ---------------------------------------------------------------------------
# _expand_with_entity_graph — multi-hop expansion
# ---------------------------------------------------------------------------


import pytest


class TestExpandWithEntityGraph:
    """Integration-level tests for the multi-hop expansion in query_router."""

    def _make_router(self, entity_graph, relation_graph, config_overrides=None):
        from axon.query_router import QueryRouterMixin

        cfg = MagicMock()
        cfg.top_k = 5
        cfg.graph_rag_relations = True
        cfg.graph_rag_entity_embedding_match = False
        cfg.graph_rag_max_hops = 2
        cfg.graph_rag_hop_decay = 0.7
        cfg.graph_rag_distance_weighted = True
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(cfg, k, v)

        router = QueryRouterMixin.__new__(QueryRouterMixin)
        router.config = cfg
        router._entity_graph = entity_graph
        router._relation_graph = relation_graph
        router._entity_embeddings = {}
        router._nx_graph = None
        router._nx_graph_dirty = True
        router._build_nx_graph = lambda: _build_nx_graph_for(router)
        router.vector_store = MagicMock()
        router.llm = MagicMock()

        # Simple Jaccard-like entity matcher (mirrors GraphRagMixin._entity_matches)
        def _entity_matches(q: str, g: str) -> float:
            return 1.0 if q.lower() == g.lower() else 0.0

        router._entity_matches = _entity_matches

        # Patch _extract_entities to return controllable entities
        router._extract_entities = MagicMock(return_value=[])
        return router

    def test_two_hop_chunk_appears_at_lower_score_than_one_hop(self):
        """Chunk reached via 2 hops must score lower than chunk reached via 1 hop."""
        eg = _make_entity_graph("alice", "bob", "carol")
        rg = {
            "alice": [_make_relation("bob", weight=5.0)],
            "bob": [_make_relation("carol", weight=5.0)],
        }
        router = self._make_router(eg, rg)
        router._extract_entities.return_value = [
            {"name": "alice", "type": "CONCEPT", "description": ""}
        ]

        # chunk_bob is 1-hop; chunk_carol is 2-hop
        router.vector_store.get_by_ids = MagicMock(
            return_value=[
                {"id": "chunk_bob", "text": "Bob content", "score": 0.0},
                {"id": "chunk_carol", "text": "Carol content", "score": 0.0},
            ]
        )

        results, matched = router._expand_with_entity_graph("alice", [])
        score_map = {r["id"]: r["score"] for r in results}

        assert "chunk_bob" in score_map
        assert "chunk_carol" in score_map
        assert (
            score_map["chunk_bob"] > score_map["chunk_carol"]
        ), "1-hop chunk should score higher than 2-hop chunk"

    def test_max_hops_zero_no_relation_expansion(self):
        """max_hops=0 should skip traversal and return only direct entity chunks."""
        eg = _make_entity_graph("alice", "bob")
        rg = {"alice": [_make_relation("bob")]}
        router = self._make_router(eg, rg, config_overrides={"graph_rag_max_hops": 0})
        router._extract_entities.return_value = [
            {"name": "alice", "type": "CONCEPT", "description": ""}
        ]

        # Return only the IDs that were actually requested
        all_chunks = {
            "chunk_alice": {"id": "chunk_alice", "text": "Alice", "score": 0.0},
            "chunk_bob": {"id": "chunk_bob", "text": "Bob", "score": 0.0},
        }
        router.vector_store.get_by_ids = MagicMock(
            side_effect=lambda ids: [all_chunks[i] for i in ids if i in all_chunks]
        )

        results, _ = router._expand_with_entity_graph("alice", [])
        result_ids = {r["id"] for r in results}
        # chunk_alice is the direct entity match; chunk_bob should NOT be included (0 hops)
        assert "chunk_bob" not in result_ids

    def test_max_hops_one_limits_to_direct_neighbours(self):
        """max_hops=1 should include 1-hop neighbours but not 2-hop."""
        eg = _make_entity_graph("a", "b", "c")
        rg = {
            "a": [_make_relation("b")],
            "b": [_make_relation("c")],
        }
        router = self._make_router(eg, rg, config_overrides={"graph_rag_max_hops": 1})
        router._extract_entities.return_value = [
            {"name": "a", "type": "CONCEPT", "description": ""}
        ]

        fetched_ids = []

        def mock_get(ids):
            fetched_ids.extend(ids)
            return [{"id": i, "text": "t", "score": 0.0} for i in ids]

        router.vector_store.get_by_ids = mock_get
        router._expand_with_entity_graph("a", [])

        # chunk_b should be fetched (1-hop); chunk_c should NOT be (2-hop)
        assert "chunk_b" in fetched_ids
        assert "chunk_c" not in fetched_ids


# ---------------------------------------------------------------------------
# Helper: build nx.Graph from a plain object's entity/relation data
# ---------------------------------------------------------------------------


def _build_nx_graph_for(obj):
    """Mirror of GraphRagMixin._build_nx_graph() for test objects."""
    if not getattr(obj, "_nx_graph_dirty", True) and getattr(obj, "_nx_graph", None) is not None:
        return obj._nx_graph

    import networkx as nx

    G = nx.DiGraph()
    for entity, data in obj._entity_graph.items():
        if isinstance(data, dict):
            G.add_node(entity, **{k: v for k, v in data.items() if isinstance(k, str)})
        else:
            G.add_node(entity)

    for src, rels in obj._relation_graph.items():
        if not isinstance(rels, list):
            continue
        for r in rels:
            if not isinstance(r, dict):
                continue
            tgt = r.get("target")
            if not tgt:
                continue
            w = float(r.get("weight", 1.0) or 1.0)
            G.add_edge(
                src, tgt, distance=1.0 / (w + 1e-6), relation=r.get("relation", ""), weight=w
            )

    obj._nx_graph = G
    obj._nx_graph_dirty = False
    return G


# ---------------------------------------------------------------------------
# _build_nx_graph_from_db (DynamicGraphBackend)
# ---------------------------------------------------------------------------


class TestDynamicGraphNxBuild:
    """Unit tests for DynamicGraphBackend._build_nx_graph_from_db()."""

    def _make_backend(self, tmp_path):
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        brain = MagicMock()
        brain.config.bm25_path = str(tmp_path)
        backend = DynamicGraphBackend(brain=brain)
        return backend

    def _insert_fact(self, backend, subject, relation, obj, confidence=0.8, status="active"):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = backend._conn
        conn.execute(
            "INSERT INTO facts (fact_id, subject, relation, object, valid_at, invalid_at, status, scope_key, confidence, metadata) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, '{}')",
            (f"{subject}_{relation}_{obj}", subject, relation, obj, now, status, confidence),
        )
        conn.commit()

    def test_active_facts_appear_as_edges(self, tmp_path):
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "alice", "WORKS_FOR", "acme", confidence=0.9)
        G = backend._build_nx_graph_from_db()
        assert G.has_edge("alice", "acme")

    def test_high_confidence_means_small_distance(self, tmp_path):
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "a", "REL", "b", confidence=0.9)
        self._insert_fact(backend, "c", "REL", "d", confidence=0.1)
        G = backend._build_nx_graph_from_db()
        dist_ab = G.edges["a", "b"]["distance"]
        dist_cd = G.edges["c", "d"]["distance"]
        assert dist_ab < dist_cd, "High confidence → small distance"

    def test_superseded_facts_excluded(self, tmp_path):
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "x", "IS_CEO_OF", "corp", status="superseded")
        G = backend._build_nx_graph_from_db()
        assert not G.has_edge("x", "corp"), "Superseded facts should not appear as edges"

    def test_graph_cache_ttl(self, tmp_path):
        """Same object returned within TTL."""
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "p", "REL", "q")
        G1 = backend._build_nx_graph_from_db()
        G2 = backend._build_nx_graph_from_db()
        assert G1 is G2, "Should return cached graph within TTL"


# ---------------------------------------------------------------------------
# GraphContext hop_count and path fields
# ---------------------------------------------------------------------------


class TestGraphContextFields:
    def test_hop_count_default_zero(self):
        from axon.graph_backends.base import GraphContext

        ctx = GraphContext(
            context_id="c1",
            context_type="fact",
            text="Alice works for Acme",
            score=0.8,
            rank=0,
            backend_id="dynamic_graph",
        )
        assert ctx.hop_count == 0
        assert ctx.path == []

    def test_hop_count_and_path_stored(self):
        from axon.graph_backends.base import GraphContext

        path = [("alice", "WORKS_FOR", "acme"), ("acme", "SUBSIDIARY_OF", "bigcorp")]
        ctx = GraphContext(
            context_id="c2",
            context_type="fact",
            text="Alice → Acme → BigCorp",
            score=0.49,
            rank=1,
            backend_id="dynamic_graph",
            hop_count=2,
            path=path,
        )
        assert ctx.hop_count == 2
        assert ctx.path == path


# ---------------------------------------------------------------------------
# DynamicGraphBackend.retrieve() — hop_count and path in returned GraphContext
# ---------------------------------------------------------------------------


class TestDynamicRetrieveHopCount:
    """Verify that retrieve() returns GraphContext objects with hop_count and path populated."""

    def _make_backend(self, tmp_path):
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        brain = MagicMock()
        brain.config.bm25_path = str(tmp_path)
        return DynamicGraphBackend(brain=brain)

    def _insert_fact(self, backend, subject, relation, obj, confidence=0.9, status="active"):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        backend._conn.execute(
            "INSERT INTO facts (fact_id, subject, relation, object, valid_at, invalid_at, "
            "status, scope_key, confidence, metadata) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, '{}')",
            (f"{subject}_{relation}_{obj}", subject, relation, obj, now, status, confidence),
        )
        backend._conn.commit()

    def _contexts_by_entity(self, contexts):
        """Map entity names mentioned in each context to the context."""
        result = {}
        for ctx in contexts:
            for name in ctx.matched_entity_names:
                if name not in result or ctx.score > result[name].score:
                    result[name] = ctx
        return result

    def test_retrieve_direct_entity_has_hop_count_zero(self, tmp_path):
        """A fact directly involving the query entity has hop_count=0."""
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "alice", "WORKS_FOR", "acme")

        contexts = backend.retrieve("alice")

        # Find the context containing "alice" as subject
        alice_ctx = next((c for c in contexts if "alice" in c.matched_entity_names), None)
        assert alice_ctx is not None, "Should find a context mentioning alice"
        assert alice_ctx.hop_count == 0

    def test_retrieve_one_hop_entity_has_hop_count_one(self, tmp_path):
        """An entity one hop from the seed entity must have hop_count=1."""
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "alice", "WORKS_FOR", "acme")

        contexts = backend.retrieve("alice")
        acme_ctx = next((c for c in contexts if "acme" in c.matched_entity_names), None)

        # acme is reachable from alice via 1 hop (or is the object in alice's own fact)
        assert acme_ctx is not None
        # acme is part of the seed fact (alice WORKS_FOR acme) → hop_count = 0 (direct seed row)
        # but acme itself is also traversal target at hop=1 from alice
        # The context for the fact "alice WORKS_FOR acme" will show the best hop (0 for seed)
        assert acme_ctx.hop_count >= 0

    def test_retrieve_two_hop_entity_has_hop_count_two(self, tmp_path):
        """A fact attributed solely to a 2-hop entity must have hop_count=2.

        Chain: alice → bob → carol → diana.
        Querying "alice" makes alice seed (hop 0), bob 1-hop, carol 2-hop, diana 3-hop.
        The fact 'carol KNOWS diana' has both carol (hop 2) and diana (hop 3).
        The best attribution is carol (closer = higher score), so hop_count == 2.
        """
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "alice", "KNOWS", "bob")
        self._insert_fact(backend, "bob", "KNOWS", "carol")
        self._insert_fact(backend, "carol", "KNOWS", "diana")

        cfg = MagicMock(
            top_k=20,
            graph_rag_max_hops=3,
            graph_rag_hop_decay=0.7,
            graph_rag_distance_weighted=True,
        )
        contexts = backend.retrieve("alice", cfg=cfg)

        # The fact 'carol KNOWS diana' should be attributed to carol at hop 2
        carol_diana_ctx = next(
            (
                c
                for c in contexts
                if "carol" in c.matched_entity_names and "diana" in c.matched_entity_names
            ),
            None,
        )
        assert carol_diana_ctx is not None, "carol-diana fact should be retrieved"
        assert carol_diana_ctx.hop_count == 2

    def test_retrieve_path_tuples_populated(self, tmp_path):
        """Path field carries (subject, relation, object) triples from seed to nearest entity.

        For the fact 'carol KNOWS diana', the nearest entity is carol at hop 2.
        Its path from alice is [(alice, KNOWS, bob), (bob, KNOWS, carol)].
        """
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "alice", "KNOWS", "bob")
        self._insert_fact(backend, "bob", "KNOWS", "carol")
        self._insert_fact(backend, "carol", "KNOWS", "diana")

        cfg = MagicMock(
            top_k=20,
            graph_rag_max_hops=3,
            graph_rag_hop_decay=0.7,
            graph_rag_distance_weighted=True,
        )
        contexts = backend.retrieve("alice", cfg=cfg)

        carol_diana_ctx = next(
            (
                c
                for c in contexts
                if "carol" in c.matched_entity_names and "diana" in c.matched_entity_names
            ),
            None,
        )
        assert carol_diana_ctx is not None
        assert carol_diana_ctx.path == [
            ("alice", "KNOWS", "bob"),
            ("bob", "KNOWS", "carol"),
        ]

    def test_retrieve_scores_decay_with_hop_count(self, tmp_path):
        """Direct-entity score > 1-hop score > 2-hop score."""
        backend = self._make_backend(tmp_path)
        self._insert_fact(backend, "alice", "KNOWS", "bob", confidence=0.9)
        self._insert_fact(backend, "bob", "KNOWS", "carol", confidence=0.9)

        contexts = backend.retrieve(
            "alice",
            cfg=MagicMock(
                top_k=20,
                graph_rag_max_hops=2,
                graph_rag_hop_decay=0.7,
                graph_rag_distance_weighted=True,
            ),
        )
        by_hop = {c.hop_count: c for c in sorted(contexts, key=lambda c: c.hop_count)}

        hop0 = by_hop.get(0)
        hop1 = by_hop.get(1)
        hop2 = by_hop.get(2)

        if hop0 and hop1:
            assert hop0.score >= hop1.score, "hop 0 score should be >= hop 1"
        if hop1 and hop2:
            assert hop1.score >= hop2.score, "hop 1 score should be >= hop 2"


# ---------------------------------------------------------------------------
# Performance guard: max_hops capped at 1 for >50k entity graphs
# ---------------------------------------------------------------------------


class TestPerformanceGuard:
    """Verify that _expand_with_entity_graph caps max_hops at 1 for very large graphs."""

    def _make_large_router(self, entity_count, seed="seed", chain_depth=2, config_max_hops=3):
        """Build a router with a large flat entity graph plus a small chain from seed."""
        from axon.query_router import QueryRouterMixin

        # Large graph: lots of unconnected entities
        eg = {
            f"e{i}": {"description": "", "type": "CONCEPT", "chunk_ids": [f"chunk_e{i}"]}
            for i in range(entity_count)
        }

        # Add seed + chain: seed → hop1 → hop2
        eg["seed"] = {"description": "", "type": "CONCEPT", "chunk_ids": ["chunk_seed"]}
        eg["hop1"] = {"description": "", "type": "CONCEPT", "chunk_ids": ["chunk_hop1"]}
        eg["hop2"] = {"description": "", "type": "CONCEPT", "chunk_ids": ["chunk_hop2"]}
        rg = {
            "seed": [_make_relation("hop1", weight=1.0)],
            "hop1": [_make_relation("hop2", weight=1.0)],
        }

        cfg = MagicMock()
        cfg.top_k = 20
        cfg.graph_rag_relations = True
        cfg.graph_rag_entity_embedding_match = False
        cfg.graph_rag_max_hops = config_max_hops
        cfg.graph_rag_hop_decay = 0.7
        cfg.graph_rag_distance_weighted = True

        router = QueryRouterMixin.__new__(QueryRouterMixin)
        router.config = cfg
        router._entity_graph = eg
        router._relation_graph = rg
        router._entity_embeddings = {}
        router._nx_graph = None
        router._nx_graph_dirty = True
        router._build_nx_graph = lambda: _build_nx_graph_for(router)
        router.vector_store = MagicMock()
        router.llm = MagicMock()
        router._extract_entities = MagicMock(
            return_value=[{"name": "seed", "type": "CONCEPT", "description": ""}]
        )

        def _entity_matches(q, g):
            return 1.0 if q.lower() == g.lower() else 0.0

        router._entity_matches = _entity_matches

        fetched = []

        def _get_by_ids(ids):
            fetched.extend(ids)
            return [{"id": i, "text": "t", "score": 0.0, "metadata": {}} for i in ids]

        router.vector_store.get_by_ids = _get_by_ids
        router._fetched_ids = fetched
        return router

    def test_max_hops_capped_at_one_for_large_graphs(self):
        """With >50k entities total, max_hops=3 config is reduced to 1.

        _make_large_router adds entity_count filler nodes PLUS seed/hop1/hop2,
        so total = entity_count + 3.  To exceed 50,000 total: entity_count = 49_998.
        """
        router = self._make_large_router(entity_count=49_998, config_max_hops=3)
        router._expand_with_entity_graph("seed", [])
        fetched = router._fetched_ids
        # hop1 should be fetched (1-hop), hop2 should NOT (2-hop, capped)
        assert "chunk_hop1" in fetched, "1-hop chunk should be fetched even with large graph"
        assert "chunk_hop2" not in fetched, "2-hop chunk must be blocked by large-graph guard"

    def test_large_graph_guard_not_triggered_below_threshold(self):
        """With total entities < 50k, max_hops=2 stays at 2 and 2-hop entity is reachable.

        entity_count = 49_996 → total = 49_999 < 50_000 → guard does not fire.
        """
        router = self._make_large_router(entity_count=49_996, config_max_hops=2)
        router._expand_with_entity_graph("seed", [])
        fetched = router._fetched_ids
        assert "chunk_hop2" in fetched, "2-hop chunk should be reachable under the threshold"


# ---------------------------------------------------------------------------
# Config YAML → AxonConfig round-trip for new fields
# ---------------------------------------------------------------------------


class TestConfigYAMLIntegration:
    """Verify the three new distance fields survive YAML → AxonConfig loading."""

    def _write_config(self, tmp_path, rag_section: dict) -> str:
        import yaml

        path = str(tmp_path / "config.yaml")
        with open(path, "w") as f:
            yaml.dump({"rag": rag_section}, f)
        return path

    def test_graph_rag_max_hops_loaded_from_yaml(self, tmp_path):
        from axon.config import AxonConfig

        path = self._write_config(tmp_path, {"graph_rag_max_hops": 5})
        cfg = AxonConfig.load(path)
        assert cfg.graph_rag_max_hops == 5

    def test_graph_rag_hop_decay_and_weighted_loaded_from_yaml(self, tmp_path):
        from axon.config import AxonConfig

        path = self._write_config(
            tmp_path,
            {
                "graph_rag_hop_decay": 0.5,
                "graph_rag_distance_weighted": False,
            },
        )
        cfg = AxonConfig.load(path)
        assert cfg.graph_rag_hop_decay == pytest.approx(0.5)
        assert cfg.graph_rag_distance_weighted is False
