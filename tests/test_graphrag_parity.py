"""Parity tests: GraphRagBackend output matches direct AxonBrain calls.

These tests ensure that wrapping AxonBrain's existing GraphRAG methods inside
GraphRagBackend does not change the observable output — the adapter is a pure
pass-through for graph state and retrieval results.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from axon.graph_backends.base import GraphDataFilters, IngestResult
from axon.graph_backends.graphrag_backend import GraphRagBackend

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_brain(
    entity_graph: dict | None = None,
    relation_graph: dict | None = None,
    community_levels: dict | None = None,
    community_summaries: dict | None = None,
    graph_payload: dict | None = None,
    expand_return: tuple | None = None,
) -> MagicMock:
    brain = MagicMock()
    brain._entity_graph = entity_graph or {}
    brain._relation_graph = relation_graph or {}
    brain._community_levels = community_levels or {}
    brain._community_summaries = community_summaries or {}

    _payload = graph_payload or {"nodes": [], "links": []}
    brain.build_graph_payload.return_value = _payload
    brain._expand_with_entity_graph.return_value = expand_return or ([], [])
    return brain


# ---------------------------------------------------------------------------
# graph_data parity
# ---------------------------------------------------------------------------


class TestGraphDataParity:
    def test_empty_graph_matches_brain(self):
        brain = _make_brain()
        backend = GraphRagBackend(brain)

        raw = brain.build_graph_payload()
        payload = backend.graph_data()

        assert payload.nodes == raw["nodes"]
        assert payload.links == raw["links"]

    def test_graph_payload_matches_brain_nodes_and_links(self):
        brain = _make_brain(
            graph_payload={
                "nodes": [
                    {"id": "alice", "name": "Alice", "type": "PERSON", "degree": 3},
                    {"id": "bob", "name": "Bob", "type": "PERSON", "degree": 1},
                ],
                "links": [
                    {"source": "alice", "target": "bob", "relation": "KNOWS"},
                ],
            }
        )
        backend = GraphRagBackend(brain)
        payload = backend.graph_data()

        assert len(payload.nodes) == 2
        assert len(payload.links) == 1
        assert payload.nodes[0]["id"] == "alice"
        assert payload.links[0]["relation"] == "KNOWS"

    def test_graph_data_to_dict_accepted_by_graph_render(self):
        """graph_data().to_dict() has the same shape as build_graph_payload()."""
        raw = {
            "nodes": [{"id": "x", "name": "X", "type": "CONCEPT", "degree": 0}],
            "links": [],
        }
        brain = _make_brain(graph_payload=raw)
        backend = GraphRagBackend(brain)
        d = backend.graph_data().to_dict()

        assert d == raw

    def test_graph_data_entity_type_filter(self):
        brain = _make_brain(
            graph_payload={
                "nodes": [
                    {"id": "alice", "type": "PERSON", "degree": 2},
                    {"id": "acme", "type": "ORGANIZATION", "degree": 1},
                ],
                "links": [],
            }
        )
        backend = GraphRagBackend(brain)
        payload = backend.graph_data(GraphDataFilters(entity_types=["PERSON"]))
        assert len(payload.nodes) == 1
        assert payload.nodes[0]["id"] == "alice"

    def test_graph_data_min_degree_filter(self):
        brain = _make_brain(
            graph_payload={
                "nodes": [
                    {"id": "hub", "type": "CONCEPT", "degree": 5},
                    {"id": "leaf", "type": "CONCEPT", "degree": 0},
                ],
                "links": [],
            }
        )
        backend = GraphRagBackend(brain)
        payload = backend.graph_data(GraphDataFilters(min_degree=1))
        assert len(payload.nodes) == 1
        assert payload.nodes[0]["id"] == "hub"

    def test_graph_data_limit_filter(self):
        brain = _make_brain(
            graph_payload={
                "nodes": [{"id": str(i), "type": "X", "degree": 0} for i in range(10)],
                "links": [],
            }
        )
        backend = GraphRagBackend(brain)
        payload = backend.graph_data(GraphDataFilters(limit=3))
        assert len(payload.nodes) == 3


# ---------------------------------------------------------------------------
# status parity
# ---------------------------------------------------------------------------


class TestStatusParity:
    def test_status_entity_count_matches_entity_graph(self):
        brain = _make_brain(
            entity_graph={"alice": {"chunk_ids": ["c1"]}, "bob": {"chunk_ids": ["c2"]}}
        )
        backend = GraphRagBackend(brain)
        s = backend.status()
        assert s["entities"] == 2

    def test_status_relation_count_matches_relation_graph(self):
        brain = _make_brain(relation_graph={"alice": [{"target": "bob", "relation": "KNOWS"}]})
        backend = GraphRagBackend(brain)
        s = backend.status()
        assert s["relations"] == 1

    def test_status_community_count_matches_community_levels(self):
        brain = _make_brain(community_levels={0: {"alice": 0, "bob": 0, "carol": 1}})
        backend = GraphRagBackend(brain)
        s = backend.status()
        assert s["communities"] == 3

    def test_status_backend_id(self):
        backend = GraphRagBackend(_make_brain())
        assert backend.status()["backend"] == "graphrag"


# ---------------------------------------------------------------------------
# retrieve parity
# ---------------------------------------------------------------------------


class TestRetrieveParity:
    def test_retrieve_delegates_to_expand_with_entity_graph(self):
        brain = _make_brain()
        brain._expand_with_entity_graph.return_value = (
            [{"id": "c1", "text": "Alice works at ACME", "score": 0.75, "metadata": {}}],
            ["alice"],
        )
        backend = GraphRagBackend(brain)
        results = backend.retrieve("Who works at ACME?")

        brain._expand_with_entity_graph.assert_called_once_with("Who works at ACME?", [], None)
        assert len(results) == 1
        assert results[0].context_id == "c1"
        assert results[0].score == 0.75
        assert results[0].backend_id == "graphrag"
        assert results[0].rank == 0

    def test_retrieve_populates_matched_entity_names(self):
        brain = _make_brain()
        brain._expand_with_entity_graph.return_value = (
            [{"id": "c1", "text": "text", "score": 0.7, "metadata": {}}],
            ["alice", "bob"],
        )
        backend = GraphRagBackend(brain)
        results = backend.retrieve("query")
        assert results[0].matched_entity_names == ["alice", "bob"]

    def test_retrieve_deduplicates_existing_results(self):
        existing = [{"id": "c1", "text": "existing", "score": 0.9, "metadata": {}}]
        brain = _make_brain()
        # _expand_with_entity_graph returns existing + new chunk
        brain._expand_with_entity_graph.return_value = (
            [
                {"id": "c1", "text": "existing", "score": 0.9, "metadata": {}},
                {"id": "c2", "text": "new", "score": 0.7, "metadata": {}},
            ],
            ["alice"],
        )
        backend = GraphRagBackend(brain)
        results = backend.retrieve("query", existing_results=existing)
        # Only the new chunk should be returned
        assert len(results) == 1
        assert results[0].context_id == "c2"
        # existing_results are passed to the underlying expand call
        brain._expand_with_entity_graph.assert_called_once_with("query", existing, None)

    def test_retrieve_empty_when_no_entities_match(self):
        brain = _make_brain()
        brain._expand_with_entity_graph.return_value = ([], [])
        backend = GraphRagBackend(brain)
        results = backend.retrieve("unknown query")
        assert results == []

    def test_retrieve_ranks_are_sequential(self):
        chunks = [
            {"id": f"c{i}", "text": f"text {i}", "score": 1.0 - i * 0.1, "metadata": {}}
            for i in range(5)
        ]
        brain = _make_brain()
        brain._expand_with_entity_graph.return_value = (chunks, [])
        backend = GraphRagBackend(brain)
        results = backend.retrieve("query")
        ranks = [r.rank for r in results]
        assert ranks == list(range(5))


# ---------------------------------------------------------------------------
# clear / delete_documents parity
# ---------------------------------------------------------------------------


class TestMutationParity:
    def test_clear_empties_entity_graph(self):
        brain = _make_brain(
            entity_graph={"alice": {"chunk_ids": ["c1"]}},
            relation_graph={"alice": [{"target": "bob"}]},
            community_levels={0: {"alice": 0}},
            community_summaries={0: "summary"},
        )
        backend = GraphRagBackend(brain)
        backend.clear()
        assert brain._entity_graph == {}
        assert brain._relation_graph == {}
        assert brain._community_levels == {}
        assert brain._community_summaries == {}

    def test_delete_documents_removes_chunk_from_entity(self):
        brain = _make_brain(entity_graph={"alice": {"chunk_ids": ["c1", "c2"]}})
        backend = GraphRagBackend(brain)
        backend.delete_documents(["c1"])
        assert brain._entity_graph["alice"]["chunk_ids"] == ["c2"]

    def test_delete_documents_removes_entity_when_all_chunks_gone(self):
        brain = _make_brain(entity_graph={"alice": {"chunk_ids": ["c1"]}})
        backend = GraphRagBackend(brain)
        backend.delete_documents(["c1"])
        assert "alice" not in brain._entity_graph

    def test_delete_documents_leaves_unrelated_entities(self):
        brain = _make_brain(
            entity_graph={
                "alice": {"chunk_ids": ["c1"]},
                "bob": {"chunk_ids": ["c2"]},
            }
        )
        backend = GraphRagBackend(brain)
        backend.delete_documents(["c1"])
        assert "alice" not in brain._entity_graph
        assert "bob" in brain._entity_graph


# ---------------------------------------------------------------------------
# ingest no-op
# ---------------------------------------------------------------------------


class TestIngestNoOp:
    def test_ingest_does_not_call_brain_extraction(self):
        brain = _make_brain()
        backend = GraphRagBackend(brain)
        result = backend.ingest([{"id": "c1", "text": "hello"}])
        assert isinstance(result, IngestResult)
        assert result.chunks_processed == 1
        # Extraction happens inside AxonBrain.ingest(), not here
        brain._extract_entities.assert_not_called()
        brain._extract_relations.assert_not_called()

    def test_ingest_empty_chunks(self):
        backend = GraphRagBackend(_make_brain())
        result = backend.ingest([])
        assert result.chunks_processed == 0
