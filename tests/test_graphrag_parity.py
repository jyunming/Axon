"""Parity tests: GraphRagBackend output matches direct AxonBrain calls.

These tests ensure that wrapping AxonBrain's existing GraphRAG methods inside
GraphRagBackend does not change the observable output — the adapter is a pure
pass-through for graph state and retrieval results.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axon.graph_backends.base import GraphDataFilters, IngestResult
from axon.graph_backends.graphrag_backend import GraphRagBackend
from axon.main import AxonBrain, AxonConfig

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
    """clear()/delete_documents() now delegate to the real GraphRagMixin
    methods (_reset_graph_state()/_prune_entity_graph()) instead of
    reimplementing a partial version of their behavior — see
    GRAPH_BACKEND_NEXT_STEPS.md Phase 1. Full-behavior coverage (relation/
    claims pruning, persistence, token-index cleanup) lives with those
    methods' own tests in test_main.py / test_graph_rag.py; these tests only
    verify the adapter delegates with the right arguments.
    """

    def test_clear_delegates_to_reset_graph_state(self):
        brain = _make_brain()
        backend = GraphRagBackend(brain)
        backend.clear()
        brain._reset_graph_state.assert_called_once_with()

    def test_delete_documents_delegates_to_prune_entity_graph(self):
        brain = _make_brain()
        backend = GraphRagBackend(brain)
        backend.delete_documents(["c1", "c2"])
        brain._prune_entity_graph.assert_called_once_with({"c1", "c2"})


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


# ---------------------------------------------------------------------------
# End-to-end fixture-driven parity: real AxonBrain, canned LLM, real
# ingest -> extraction -> community-build -> render. Complements the
# MagicMock adapter-contract tests above (which only verify delegation) with
# genuine behavior coverage — this is the drift detector Phase 3/4 rely on.
# See docs/architecture/GRAPH_BACKEND_NEXT_STEPS.md Phase 2.
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "graphrag_parity"
_SCENARIOS = [
    "codebase",
    "issue_thread",
    "paper_abstract",
    "project_doc",
    "software_guide",
    "stdlib_docs",
]


def _load_fixture(name: str) -> dict:
    d = _FIXTURES_DIR / name
    return {
        "input_text": (d / "input.txt").read_text(encoding="utf-8"),
        "canned": json.loads((d / "canned_extraction.json").read_text(encoding="utf-8")),
        "expected": json.loads((d / "expected_graph.json").read_text(encoding="utf-8")),
    }


def _canned_llm(canned: dict):
    """Match on prompt BODY content, not system_prompt substrings — the
    README's documented matcher (system_prompt substrings "named entities"/
    "relationships") never matches the real prompts and would silently
    no-op. This mirrors the proven pattern in test_mixin_integration.py.
    Anything else (community-summary JSON prompts, etc.) falls through to a
    plain-text default: _generate_community_summaries gracefully treats
    non-JSON output as raw summary text rather than raising.
    """

    def _complete(prompt, system_prompt=None, **kwargs):
        prompt_l = prompt.lower()
        if "extract the key named entities" in prompt_l:
            return canned["entities"]
        if "extract key relationships" in prompt_l:
            return canned["relations"]
        return "no-op summary"

    return _complete


class TestGraphRagParityFixtures:
    @pytest.fixture
    def make_brain(self, tmp_path):
        @contextmanager
        def _make(canned: dict):
            config = AxonConfig(
                axon_store_base=str(tmp_path),
                bm25_path=str(tmp_path),
                vector_store_path=str(tmp_path),
                graph_rag=True,
                graph_rag_relations=True,
                graph_rag_min_entities_for_relations=0,
                graph_rag_llm_fused_extraction=False,
                graph_rag_community=True,
                graph_rag_community_lazy=False,
                raptor=False,
                similarity_threshold=0.0,
            )
            # Patches stay live for the whole test body (not just brain
            # construction) — matches test_mixin_integration.py's proven
            # pattern, so anything lazily re-touching these classes during
            # ingest/finalize stays mocked too.
            with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
                "axon.main.OpenEmbedding"
            ) as mock_embed, patch("axon.main.OpenLLM") as mock_llm, patch(
                "axon.main.OpenReranker"
            ), patch(
                "axon.projects.ensure_project"
            ), patch(
                "axon.projects.ensure_user_project"
            ):
                mock_embed.return_value.embed.return_value = [[0.1] * 8]
                mock_embed.return_value.embed_query.return_value = [0.1] * 8
                mock_llm.return_value.complete.side_effect = _canned_llm(canned)
                brain = AxonBrain(config)
                # Store-isolation guard (GRAPH_BACKEND_NEXT_STEPS.md "Safety
                # notes") — never let a fixture-driven test resolve onto the
                # real global store.
                assert str(tmp_path) in str(brain.config.projects_root)
                try:
                    yield brain
                finally:
                    brain.close()

        return _make

    @pytest.mark.parametrize("scenario", _SCENARIOS)
    def test_ingest_extraction_community_render_parity(self, make_brain, scenario):
        fixture = _load_fixture(scenario)
        with make_brain(fixture["canned"]) as brain:
            brain.ingest([{"id": f"{scenario}_doc", "text": fixture["input_text"]}])

            expected = fixture["expected"]
            entity_keys = {k.lower() for k in expected["expected_entity_keys"]}
            got_keys = set(brain._entity_graph.keys())
            missing = entity_keys - got_keys
            assert not missing, f"{scenario}: missing entities {missing} in {sorted(got_keys)}"

            for name, expected_type in expected.get("expected_entity_types", {}).items():
                node = brain._entity_graph.get(name.lower())
                assert node is not None, f"{scenario}: {name} missing from entity graph"
                assert node["type"] == expected_type, (
                    f"{scenario}: {name} type {node['type']!r} != " f"expected {expected_type!r}"
                )

            for subject, obj in expected.get("expected_relation_pairs", []):
                subj_l, obj_l = subject.lower(), obj.lower()
                rels = brain._relation_graph.get(subj_l, [])
                targets = {r.get("target") for r in rels}
                assert obj_l in targets, (
                    f"{scenario}: relation {subject}->{obj} not found; "
                    f"{subj_l} has targets {targets}"
                )

            # Close the loop: finalize (community build) + render, both
            # asserted through the backend surface — not the mixin methods
            # directly — so this stays stable when Phase 3/4 relocate them.
            result = brain._graph_backend.finalize()
            # Pinned > 0, not just >= 0: every fixture scenario produces real
            # communities (verified 4-9 across the 6 scenarios) — >= 0 would
            # pass even if community detection silently no-op'd.
            assert result.communities_built > 0, (
                f"{scenario}: expected at least one community, got " f"{result.communities_built}"
            )

            payload = brain._graph_backend.graph_data()
            assert payload.nodes, f"{scenario}: render produced no nodes"
            rendered_ids = {n["id"] for n in payload.nodes}
            assert (
                entity_keys <= rendered_ids
            ), f"{scenario}: render missing {entity_keys - rendered_ids}"


class TestCommunitySummarizationDoesNotDeadlock:
    """Regression test for a deadlock discovered while wiring the fixtures
    above: _rebuild_communities holds _graph_lock (an RLock) for its whole
    body, then _generate_community_summaries dispatches _summarise onto a
    real ThreadPoolExecutor; worker threads call _gr_cache_get, which used
    to also acquire _graph_lock — a different OS thread than the one
    holding it, so RLock's same-thread reentry never applies and every
    worker blocks forever. Fixed by giving the GraphRAG cache its own
    dedicated leaf lock (_gr_cache_lock) that's never held while dispatching
    executor work. Runs finalize() in a daemon thread with a bounded join so
    a regression fails in 30s instead of hanging the test run / CI.
    """

    def test_finalize_does_not_deadlock_on_graph_lock(self, tmp_path):
        import threading

        fixture = _load_fixture("codebase")
        config = AxonConfig(
            axon_store_base=str(tmp_path),
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag=True,
            graph_rag_relations=True,
            graph_rag_min_entities_for_relations=0,
            graph_rag_llm_fused_extraction=False,
            graph_rag_community=True,
            graph_rag_community_lazy=False,
            raptor=False,
            similarity_threshold=0.0,
        )
        with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.main.OpenEmbedding"
        ) as mock_embed, patch("axon.main.OpenLLM") as mock_llm, patch(
            "axon.main.OpenReranker"
        ), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_project"
        ):
            mock_embed.return_value.embed.return_value = [[0.1] * 8]
            mock_embed.return_value.embed_query.return_value = [0.1] * 8
            mock_llm.return_value.complete.side_effect = _canned_llm(fixture["canned"])
            brain = AxonBrain(config)
            assert str(tmp_path) in str(brain.config.projects_root)
            try:
                brain.ingest([{"id": "codebase_doc", "text": fixture["input_text"]}])

                result_holder: dict = {}

                def _run():
                    result_holder["result"] = brain._graph_backend.finalize()

                thread = threading.Thread(target=_run, daemon=True)
                thread.start()
                thread.join(timeout=30)
                assert not thread.is_alive(), (
                    "community summarization deadlocked under _graph_lock "
                    "(finalize() did not return within 30s)"
                )
                assert result_holder["result"].communities_built > 0
            finally:
                brain.close()
