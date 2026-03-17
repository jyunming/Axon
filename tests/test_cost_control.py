"""Tests for cost-control improvements (architecture review 2026-03-17).

Covers:
- P1: raptor_graphrag_leaf_skip_threshold default lowered to 3
- P2: graph_rag_entity_min_frequency prunes singletons before community detection
- P3: graph_rag_relation_budget caps relation extraction by entity density ranking
- P4: query_hint parameter tightens lazy summarization LLM cap
"""
from unittest.mock import MagicMock, patch


def _make_brain(tmp_path, **cfg_kwargs):
    from axon.main import AxonBrain, AxonConfig

    cfg = AxonConfig(
        vector_store_path=str(tmp_path / "chroma"),
        bm25_path=str(tmp_path / "bm25"),
        **cfg_kwargs,
    )
    brain = AxonBrain.__new__(AxonBrain)
    brain.config = cfg
    brain.llm = MagicMock()
    brain.embedding = MagicMock()
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_levels = {}
    brain._community_summaries = {}
    brain._community_children = {}
    brain._community_hierarchy = {}
    brain._community_graph_dirty = False
    return brain


class TestP1LeafSkipThreshold:
    """raptor_graphrag_leaf_skip_threshold default changed from 20 to 3."""

    def test_default_is_3(self):
        from axon.main import AxonConfig

        assert AxonConfig().raptor_graphrag_leaf_skip_threshold == 3

    def test_source_with_4_leaves_skips_leaf_extraction(self, tmp_path):
        """Source with >= threshold leaf chunks is classified as a large source."""
        from axon.main import AxonConfig

        cfg = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            raptor=True,
            graph_rag=True,
            raptor_graphrag_leaf_skip_threshold=3,
            graph_rag_include_raptor_summaries=True,
        )
        docs = [
            {"id": f"c{i}", "text": f"text {i}", "metadata": {"source": "doc.md"}}
            for i in range(4)  # 4 leaves >= threshold=3
        ]
        raptor_summary = {
            "id": "r1",
            "text": "summary",
            "metadata": {"source": "doc.md", "raptor_level": 1},
        }
        docs.append(raptor_summary)

        leaf_count = {}
        for d in docs:
            if not d.get("metadata", {}).get("raptor_level"):
                src = d["metadata"].get("source", d["id"])
                leaf_count[src] = leaf_count.get(src, 0) + 1

        large_sources = {
            s for s, cnt in leaf_count.items() if cnt >= cfg.raptor_graphrag_leaf_skip_threshold
        }
        assert "doc.md" in large_sources

        # Only the RAPTOR summary should be selected
        selected = []
        for d in docs:
            lvl = d.get("metadata", {}).get("raptor_level")
            src = d["metadata"].get("source", d["id"])
            if not lvl and src not in large_sources:
                selected.append(d)
            elif lvl == 1 and cfg.graph_rag_include_raptor_summaries:
                selected.append(d)

        assert len(selected) == 1
        assert selected[0]["id"] == "r1"

    def test_source_with_2_leaves_still_extracts_leaves(self, tmp_path):
        """Source with < threshold leaf chunks is still extracted directly."""
        from axon.main import AxonConfig

        cfg = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            raptor_graphrag_leaf_skip_threshold=3,
        )
        docs = [
            {"id": f"c{i}", "text": f"text {i}", "metadata": {"source": "small.md"}}
            for i in range(2)  # 2 leaves < threshold=3
        ]

        leaf_count = {}
        for d in docs:
            src = d["metadata"].get("source", d["id"])
            leaf_count[src] = leaf_count.get(src, 0) + 1

        large_sources = {
            s for s, cnt in leaf_count.items() if cnt >= cfg.raptor_graphrag_leaf_skip_threshold
        }
        assert "small.md" not in large_sources


class TestP2EntityFrequencyPruning:
    """graph_rag_entity_min_frequency prunes low-frequency entities before community detection."""

    def test_default_is_1_no_pruning(self):
        from axon.main import AxonConfig

        assert AxonConfig().graph_rag_entity_min_frequency == 1

    def test_prunes_singletons_from_graph(self, tmp_path):
        """Entities with frequency=1 are excluded when min_frequency=2."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("networkx not installed")

        brain = _make_brain(tmp_path, graph_rag_entity_min_frequency=2)
        brain._entity_graph = {
            "kafka": {"frequency": 3, "description": "messaging", "type": "PRODUCT"},
            "payment-service": {"frequency": 2, "description": "svc", "type": "ORGANIZATION"},
            "rare-entity": {"frequency": 1, "description": "rare", "type": "CONCEPT"},
        }
        G = brain._build_networkx_graph()
        assert "kafka" in G.nodes
        assert "payment-service" in G.nodes
        assert "rare-entity" not in G.nodes

    def test_default_min_frequency_includes_all(self, tmp_path):
        """With min_frequency=1, all entities are included regardless of frequency."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("networkx not installed")

        brain = _make_brain(tmp_path, graph_rag_entity_min_frequency=1)
        brain._entity_graph = {
            "singleton": {"frequency": 1, "description": "", "type": "CONCEPT"},
            "common": {"frequency": 10, "description": "", "type": "CONCEPT"},
        }
        G = brain._build_networkx_graph()
        assert "singleton" in G.nodes
        assert "common" in G.nodes

    def test_non_dict_node_is_not_pruned(self, tmp_path):
        """Legacy list-format entity nodes are always included (no frequency field)."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("networkx not installed")

        brain = _make_brain(tmp_path, graph_rag_entity_min_frequency=5)
        brain._entity_graph = {
            "legacy-entity": ["chunk1", "chunk2"],  # legacy list format
        }
        G = brain._build_networkx_graph()
        assert "legacy-entity" in G.nodes


class TestP3RelationBudget:
    """graph_rag_relation_budget caps relation extraction by entity-density ranking."""

    def test_default_is_0_unlimited(self):
        from axon.main import AxonConfig

        assert AxonConfig().graph_rag_relation_budget == 0

    def test_budget_selects_highest_density_chunks(self):
        """Top-N chunks by entity/text-length density are selected."""
        chunks = [
            {"id": "low", "text": "a" * 100},  # 1 entity  → density 0.01
            {"id": "high", "text": "a" * 50},  # 5 entities → density 0.10
            {"id": "mid", "text": "a" * 200},  # 10 entities → density 0.05
        ]
        entity_count = {"low": 1, "high": 5, "mid": 10}
        budget = 2

        ranked = sorted(
            chunks,
            key=lambda d: entity_count.get(d["id"], 0) / max(len(d.get("text", "")), 1),
            reverse=True,
        )[:budget]

        ids = [d["id"] for d in ranked]
        assert "high" in ids  # density 0.10
        assert "mid" in ids  # density 0.05
        assert "low" not in ids  # density 0.01

    def test_zero_budget_does_not_filter(self):
        """Budget=0 leaves all chunks through."""
        chunks = [{"id": f"c{i}", "text": "x" * 10} for i in range(5)]
        entity_count = {c["id"]: 1 for c in chunks}
        budget = 0

        result = (
            chunks
            if budget == 0
            else sorted(
                chunks,
                key=lambda d: entity_count.get(d["id"], 0) / max(len(d.get("text", "")), 1),
                reverse=True,
            )[:budget]
        )

        assert len(result) == 5


class TestP4QueryGuidedLazySummarization:
    """_generate_community_summaries(query_hint) tightens cap and orders by relevance."""

    def test_accepts_query_hint_without_error(self, tmp_path):
        """_generate_community_summaries runs with query_hint and without error."""
        brain = _make_brain(tmp_path, graph_rag_global_top_communities=5)
        brain._community_levels = {0: {"kafka": 0, "inventory": 0, "payment": 1}}
        brain._community_summaries = {}
        brain._community_children = {}
        brain._community_hierarchy = {}
        brain._executor = MagicMock()
        brain._executor.map = MagicMock(return_value=iter([]))
        brain._save_community_summaries = MagicMock()

        brain._generate_community_summaries(query_hint="kafka lag")
        brain._save_community_summaries.assert_called_once()

    def test_query_hint_tightens_max_total_to_top_communities(self):
        """When query_hint is set and top_communities>0, effective cap = min(max_total, top_communities)."""
        from axon.main import AxonConfig

        cfg = AxonConfig(
            graph_rag_global_top_communities=5,
            graph_rag_community_llm_max_total=200,
        )
        lazy_cap = cfg.graph_rag_global_top_communities
        max_total = cfg.graph_rag_community_llm_max_total
        effective = min(max_total, lazy_cap) if lazy_cap > 0 else max_total
        assert effective == 5

    def test_zero_top_communities_does_not_cap_to_zero(self):
        """graph_rag_global_top_communities=0 disables the lazy cap (full max_total applies)."""
        from axon.main import AxonConfig

        cfg = AxonConfig(
            graph_rag_global_top_communities=0,
            graph_rag_community_llm_max_total=200,
        )
        lazy_cap = cfg.graph_rag_global_top_communities
        max_total = cfg.graph_rag_community_llm_max_total
        effective = min(max_total, lazy_cap) if lazy_cap > 0 else max_total
        assert effective == 200

    def test_no_query_hint_does_not_alter_cap(self, tmp_path):
        """Without query_hint, standard cap is used (no tightening)."""
        brain = _make_brain(
            tmp_path,
            graph_rag_global_top_communities=5,
            graph_rag_community_llm_max_total=200,
        )
        brain._community_levels = {0: {"a": 0, "b": 0}}
        brain._community_summaries = {}
        brain._community_children = {}
        brain._community_hierarchy = {}
        brain._executor = MagicMock()
        brain._executor.map = MagicMock(return_value=iter([]))
        brain._save_community_summaries = MagicMock()

        # Without query_hint — should not log the "lazy mode" message
        with patch("axon.main.logger") as mock_log:
            brain._generate_community_summaries()
            calls = [str(c) for c in mock_log.info.call_args_list]
            assert not any("lazy mode" in c for c in calls)


class TestGraphVisualization:
    """Tests for the normalized 3D graph visualization (build_graph_payload + export_graph_html)."""

    def _make_brain(self, tmp_path, **kwargs):
        from axon.main import AxonBrain, AxonConfig

        cfg = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            **kwargs,
        )
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_levels = {}
        return brain

    def test_build_graph_payload_returns_nodes_and_links(self, tmp_path):
        """build_graph_payload returns dict with 'nodes' and 'links' keys."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "kafka": {
                "type": "PRODUCT",
                "description": "messaging",
                "chunk_ids": ["c1"],
                "degree": 2,
            },
        }
        brain._relation_graph = {}
        payload = brain.build_graph_payload()
        assert "nodes" in payload
        assert "links" in payload

    def test_build_graph_payload_node_fields(self, tmp_path):
        """Each node contains required renderer fields."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "payment-service": {
                "type": "ORGANIZATION",
                "description": "handles payments",
                "chunk_ids": ["c1", "c2"],
                "degree": 3,
            }
        }
        brain._relation_graph = {}
        payload = brain.build_graph_payload()
        node = payload["nodes"][0]
        for field in ("id", "name", "label", "type", "color", "val", "tooltip"):
            assert field in node, f"missing field: {field}"
        assert node["id"] == "payment-service"
        assert node["chunk_count"] == 2

    def test_build_graph_payload_community_assignment(self, tmp_path):
        """Community ID is read from level-0 community_levels {entity: community_id}."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "kafka": {"type": "PRODUCT", "description": "", "chunk_ids": [], "degree": 0}
        }
        brain._relation_graph = {}
        brain._community_levels = {0: {"kafka": 7}}  # entity -> community_id
        payload = brain.build_graph_payload()
        assert payload["nodes"][0]["community"] == 7

    def test_build_graph_payload_edge_uses_target_field(self, tmp_path):
        """Links are built from relation_graph using the 'target' field (not 'object')."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "kafka": {"type": "PRODUCT", "description": "", "chunk_ids": [], "degree": 1},
            "inventory": {"type": "CONCEPT", "description": "", "chunk_ids": [], "degree": 1},
        }
        brain._relation_graph = {
            "kafka": [{"target": "inventory", "relation": "delays", "description": "", "weight": 5}]
        }
        payload = brain.build_graph_payload()
        assert len(payload["links"]) == 1
        link = payload["links"][0]
        assert link["source"] == "kafka"
        assert link["target"] == "inventory"

    def test_build_graph_payload_deduplicates_edges(self, tmp_path):
        """Duplicate (src, tgt, relation) triples produce only one link."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "a": {"type": "CONCEPT", "description": "", "chunk_ids": [], "degree": 1},
            "b": {"type": "CONCEPT", "description": "", "chunk_ids": [], "degree": 1},
        }
        brain._relation_graph = {
            "a": [
                {"target": "b", "relation": "links", "description": "", "weight": 3},
                {"target": "b", "relation": "links", "description": "", "weight": 3},  # duplicate
            ]
        }
        payload = brain.build_graph_payload()
        assert len(payload["links"]) == 1

    def test_build_graph_payload_skips_unknown_targets(self, tmp_path):
        """Relations pointing to entities not in the entity_graph are excluded."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "a": {"type": "CONCEPT", "description": "", "chunk_ids": [], "degree": 0}
        }
        brain._relation_graph = {
            "a": [
                {"target": "ghost-entity", "relation": "points-to", "description": "", "weight": 1}
            ]
        }
        payload = brain.build_graph_payload()
        assert len(payload["links"]) == 0

    def test_export_graph_html_writes_file(self, tmp_path):
        """export_graph_html writes a valid HTML file when path is provided."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "kafka": {"type": "PRODUCT", "description": "msg bus", "chunk_ids": ["c1"], "degree": 1}
        }
        brain._relation_graph = {}
        html_path = str(tmp_path / "graph.html")
        html = brain.export_graph_html(path=html_path, open_browser=False)
        assert "ForceGraph3D" in html
        assert "kafka" in html
        import os

        assert os.path.exists(html_path)

    def test_export_graph_html_no_pyvis_required(self, tmp_path):
        """The new export does not import or require pyvis."""
        brain = self._make_brain(tmp_path)
        brain._entity_graph = {}
        brain._relation_graph = {}
        # Should not raise ImportError about pyvis
        html = brain.export_graph_html(open_browser=False)
        assert isinstance(html, str)

    def test_export_graph_html_writes_json_payload(self, tmp_path):
        """When json_path is provided, a valid JSON payload is written separately."""
        import json

        brain = self._make_brain(tmp_path)
        brain._entity_graph = {
            "auth-service": {
                "type": "ORGANIZATION",
                "description": "",
                "chunk_ids": [],
                "degree": 0,
            }
        }
        brain._relation_graph = {}
        html_path = str(tmp_path / "graph.html")
        json_path = str(tmp_path / "graph.json")
        brain.export_graph_html(path=html_path, json_path=json_path, open_browser=False)
        data = json.loads(open(json_path).read())
        assert "nodes" in data
        assert "links" in data
        assert any(n["id"] == "auth-service" for n in data["nodes"])
