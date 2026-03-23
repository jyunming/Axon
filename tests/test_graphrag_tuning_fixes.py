"""
tests/test_graphrag_tuning_fixes.py

Regression tests for GraphRAG tuning improvements from
Qualification/CodexQual/GRAPH_RAG_DEVELOPER_IMPROVEMENT_ITEMS_2026_03_22.md

Items covered
-------------
P0-3: Unsafe GraphRAG defaults
  - graph_rag_relation_budget: 0 → 30 (unbounded relation extraction is dangerous)
  - graph_rag_entity_min_frequency: 1 → 2 (singleton entities pollute graph)

P0-2: Query graph chunks dropped before synthesis
  - _merge_graph_slots interleaves base + graph-expanded by score
  - graph-expanded chunks always survive up to budget, even when budget < top_k

Item 13 (leidenalg): resolution parameter was iterated but not passed to
  find_partition — fixed by switching to CPMVertexPartition with
  resolution_parameter=resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# P0-3: Safe defaults
# ---------------------------------------------------------------------------


class TestSafeGraphDefaults:
    """P0-3 — default config values must be safe for production corpora."""

    def test_relation_budget_default_is_30(self):
        """graph_rag_relation_budget must default to 30, not 0.

        Default of 0 = unlimited relation extraction, which makes ingest cost
        proportional to corpus size without bound.
        """
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.graph_rag_relation_budget == 30, (
            f"Expected graph_rag_relation_budget=30 (bounded), got {cfg.graph_rag_relation_budget}. "
            "Default of 0 (unlimited) causes unbounded LLM calls on large corpora."
        )

    def test_entity_min_frequency_default_is_2(self):
        """graph_rag_entity_min_frequency must default to 2, not 1.

        Default of 1 = no pruning; one-off entities pollute the community graph.
        """
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.graph_rag_entity_min_frequency == 2, (
            f"Expected graph_rag_entity_min_frequency=2 (prune singletons), "
            f"got {cfg.graph_rag_entity_min_frequency}."
        )

    def test_budget_zero_can_be_set_explicitly(self):
        """Users can still set budget=0 for unbounded experiments."""
        from axon.main import AxonConfig

        cfg = AxonConfig(graph_rag_relation_budget=0)
        assert cfg.graph_rag_relation_budget == 0

    def test_min_frequency_one_can_be_set_explicitly(self):
        """Users can still set min_frequency=1 for small corpora."""
        from axon.main import AxonConfig

        cfg = AxonConfig(graph_rag_entity_min_frequency=1)
        assert cfg.graph_rag_entity_min_frequency == 1


# ---------------------------------------------------------------------------
# P0-2: _merge_graph_slots
# ---------------------------------------------------------------------------


class TestMergeGraphSlots:
    """P0-2 — graph-expanded chunks must survive into synthesis context."""

    def _r(self, id_: str, score: float, graph_expanded: bool = False) -> dict:
        r = {"id": id_, "score": score, "text": f"text for {id_}"}
        if graph_expanded:
            r["_graph_expanded"] = True
        return r

    def test_graph_slots_always_survive_when_budget_positive(self):
        """Graph-expanded chunks always appear in results when budget > 0."""
        from axon.query_router import QueryRouterMixin

        base = [self._r(f"b{i}", 0.9 - i * 0.05) for i in range(5)]
        expanded = [self._r("g1", 0.60, graph_expanded=True)]
        results = base + expanded

        merged = QueryRouterMixin._merge_graph_slots(results, top_k=5, budget=3)

        assert any(
            r["id"] == "g1" for r in merged
        ), "Graph-expanded chunk must survive even though it ranked below top_k by score."

    def test_high_scoring_graph_chunk_lands_in_top_k(self):
        """A graph-expanded chunk that scores better than base chunks enters the top-k."""
        from axon.query_router import QueryRouterMixin

        base = [self._r(f"b{i}", 0.5 - i * 0.05) for i in range(5)]
        # This graph chunk scores highest of all
        expanded = [self._r("g_hot", 0.99, graph_expanded=True)]
        results = base + expanded

        merged = QueryRouterMixin._merge_graph_slots(results, top_k=5, budget=1)

        ids = [r["id"] for r in merged]
        assert "g_hot" in ids, "High-scoring graph chunk must be in merged results."
        assert ids[0] == "g_hot", "Highest-scoring item should be first."

    def test_budget_caps_graph_slot_count(self):
        """No more than budget graph-expanded chunks appear in results."""
        from axon.query_router import QueryRouterMixin

        base = [self._r(f"b{i}", 0.9 - i * 0.01) for i in range(5)]
        expanded = [self._r(f"g{i}", 0.3, graph_expanded=True) for i in range(10)]
        results = base + expanded

        merged = QueryRouterMixin._merge_graph_slots(results, top_k=5, budget=3)

        graph_count = sum(1 for r in merged if r.get("_graph_expanded"))
        assert graph_count <= 3, f"Expected at most 3 graph slots (budget=3), got {graph_count}."

    def test_no_duplicate_ids_in_merged(self):
        """If a graph-expanded chunk already ranked in top_k, it is not duplicated."""
        from axon.query_router import QueryRouterMixin

        # Graph chunk has a very high score — will enter top_k naturally
        g_chunk = self._r("g_dup", 0.99, graph_expanded=True)
        base = [self._r(f"b{i}", 0.8 - i * 0.1) for i in range(4)]
        results = base + [g_chunk]

        merged = QueryRouterMixin._merge_graph_slots(results, top_k=5, budget=2)

        ids = [r["id"] for r in merged]
        assert (
            ids.count("g_dup") == 1
        ), "A graph chunk that ranked naturally must not be added again as a budget slot."

    def test_no_graph_expansion_without_expanded_chunks(self):
        """When no chunks are graph-expanded, result is just top_k base chunks."""
        from axon.query_router import QueryRouterMixin

        base = [self._r(f"b{i}", 0.9 - i * 0.05) for i in range(8)]

        merged = QueryRouterMixin._merge_graph_slots(base, top_k=5, budget=3)

        assert len(merged) == 5
        assert all(not r.get("_graph_expanded") for r in merged)

    def test_result_size_bounded_by_top_k_plus_budget(self):
        """Result size never exceeds top_k + budget."""
        from axon.query_router import QueryRouterMixin

        base = [self._r(f"b{i}", 0.9 - i * 0.01) for i in range(20)]
        expanded = [self._r(f"g{i}", 0.3, graph_expanded=True) for i in range(20)]

        merged = QueryRouterMixin._merge_graph_slots(base + expanded, top_k=10, budget=5)

        assert (
            len(merged) <= 15
        ), f"Result must be bounded by top_k(10) + budget(5) = 15, got {len(merged)}."

    def test_zero_budget_returns_empty_graph_slots(self):
        """When budget=0, no graph-expanded chunks are added beyond natural top_k."""
        from axon.query_router import QueryRouterMixin

        base = [self._r(f"b{i}", 0.9 - i * 0.05) for i in range(5)]
        expanded = [self._r("g1", 0.3, graph_expanded=True)]

        merged = QueryRouterMixin._merge_graph_slots(base + expanded, top_k=5, budget=0)

        # With budget=0, the low-scoring graph chunk should not survive
        graph_count = sum(1 for r in merged if r.get("_graph_expanded"))
        assert graph_count == 0


# ---------------------------------------------------------------------------
# Item 13: leidenalg resolution passed to find_partition
# ---------------------------------------------------------------------------


class TestLeidenalgResolutionParameter:
    """Item 13 — leidenalg must pass resolution to CPMVertexPartition, not ignore it."""

    def test_find_partition_called_with_resolution_parameter(self):
        """_run_hierarchical_community_detection passes resolution_parameter to find_partition."""
        from unittest.mock import MagicMock

        # Build minimal mock chain for leidenalg
        mock_la = MagicMock()
        mock_ig = MagicMock()

        # Mock graph with nodes that have _nx_name attribute
        mock_graph = MagicMock()
        mock_graph.vs.attributes.return_value = ["_nx_name"]
        mock_graph.vs.__getitem__ = MagicMock(return_value=["entity_a", "entity_b", "entity_c"])
        mock_ig.Graph.from_networkx.return_value = mock_graph

        # partition membership: 3 nodes, 2 communities
        mock_partition = [[0, 1], [2]]
        mock_la.find_partition.return_value = mock_partition

        with patch.dict(
            "sys.modules",
            {
                "igraph": mock_ig,
                "leidenalg": mock_la,
                "graspologic": None,
                "graspologic.partition": None,
            },
        ):
            entity_graph = {
                "entity_a": {"description": "A", "frequency": 3},
                "entity_b": {"description": "B", "frequency": 3},
                "entity_c": {"description": "C", "frequency": 3},
            }
            relation_graph = {
                "entity_a": [{"target": "entity_b", "relation": "r", "weight": 1}],
                "entity_b": [{"target": "entity_c", "relation": "r", "weight": 1}],
            }

            from axon.main import AxonBrain, AxonConfig

            cfg = AxonConfig(
                graph_rag=True,
                graph_rag_community_backend="leidenalg",
                graph_rag_community_levels=2,  # triggers multi-resolution loop
            )
            brain = AxonBrain.__new__(AxonBrain)
            brain.config = cfg
            brain._entity_graph = entity_graph
            brain._relation_graph = relation_graph

            brain._run_hierarchical_community_detection()

        # Each find_partition call must have resolution_parameter= keyword
        call_kwargs = [call.kwargs for call in mock_la.find_partition.call_args_list]
        assert call_kwargs, "find_partition was never called."
        for kwargs in call_kwargs:
            assert "resolution_parameter" in kwargs, (
                "find_partition must be called with resolution_parameter= so that "
                "multi-level community detection uses different resolutions. "
                f"Got kwargs: {kwargs}"
            )

    def test_different_levels_use_different_resolutions(self):
        """When graph_rag_community_levels > 1, each level uses a different resolution value."""

        mock_la = MagicMock()
        mock_ig = MagicMock()
        mock_graph = MagicMock()
        mock_graph.vs.attributes.return_value = ["_nx_name"]
        mock_graph.vs.__getitem__ = MagicMock(return_value=["entity_a", "entity_b", "entity_c"])
        mock_ig.Graph.from_networkx.return_value = mock_graph
        mock_la.find_partition.return_value = [[0, 1], [2]]

        with patch.dict(
            "sys.modules",
            {
                "igraph": mock_ig,
                "leidenalg": mock_la,
                "graspologic": None,
                "graspologic.partition": None,
            },
        ):
            entity_graph = {
                "entity_a": {"description": "A", "frequency": 3},
                "entity_b": {"description": "B", "frequency": 3},
                "entity_c": {"description": "C", "frequency": 3},
            }
            relation_graph = {"entity_a": [{"target": "entity_b", "relation": "r", "weight": 1}]}

            from axon.main import AxonBrain, AxonConfig

            cfg = AxonConfig(
                graph_rag=True,
                graph_rag_community_backend="leidenalg",
                graph_rag_community_levels=3,
            )
            brain = AxonBrain.__new__(AxonBrain)
            brain.config = cfg
            brain._entity_graph = entity_graph
            brain._relation_graph = relation_graph

            brain._run_hierarchical_community_detection()

        resolutions = [
            call.kwargs.get("resolution_parameter")
            for call in mock_la.find_partition.call_args_list
            if "resolution_parameter" in call.kwargs
        ]
        assert (
            len(resolutions) == 3
        ), f"Expected 3 find_partition calls for 3 levels, got {len(resolutions)}."
        # All resolutions must be distinct when levels > 1
        assert len(set(resolutions)) > 1, (
            "Multi-level community detection must use different resolution values per level. "
            f"Got: {resolutions}"
        )
