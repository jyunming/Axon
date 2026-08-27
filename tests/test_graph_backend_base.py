"""Architecture tests for the GraphBackend Protocol.

Verifies:
  1. The Protocol exposes exactly the 9 required methods.
  2. GraphRagBackend satisfies the Protocol (runtime isinstance check).
  3. DynamicGraphBackend satisfies the Protocol (full SQLite implementation).
  4. A minimal hand-rolled object satisfies the Protocol.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from axon.graph_backends.base import (
    _REQUIRED_METHODS,
    FinalizationResult,
    GraphBackend,
    GraphContext,
    GraphDataFilters,
    GraphPayload,
    IngestResult,
    RetrievalConfig,
)
from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend
from axon.graph_backends.federated_backend import FederatedGraphBackend
from axon.graph_backends.graphrag_backend import GraphRagBackend
from axon.graph_backends.none_backend import NoneGraphBackend
from axon.graph_rag import GraphRagMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_backend(tmp_path) -> DynamicGraphBackend:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    cfg = SimpleNamespace(bm25_path=str(tmp_path), graph_backend="dynamic_graph")
    llm = MagicMock()
    llm.complete.return_value = ""
    brain = SimpleNamespace(config=cfg, llm=llm)
    return DynamicGraphBackend(brain)


def _make_graphrag_backend() -> GraphRagBackend:
    brain = MagicMock()
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_levels = {}
    brain._community_summaries = {}
    brain.build_graph_payload.return_value = {"nodes": [], "links": []}
    brain._expand_with_entity_graph.return_value = ([], [])
    return GraphRagBackend(brain)


def _make_none_backend() -> NoneGraphBackend:
    return NoneGraphBackend(MagicMock())


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


class TestProtocolShape:
    def test_required_methods_count(self):
        assert len(_REQUIRED_METHODS) == 9

    def test_required_method_names(self):
        expected = {
            "ingest",
            "retrieve",
            "finalize",
            "clear",
            "delete_documents",
            "status",
            "graph_data",
            "has_entities",
            "has_community_summaries",
        }
        assert _REQUIRED_METHODS == expected

    def test_protocol_is_runtime_checkable(self):
        # runtime_checkable allows isinstance checks
        assert hasattr(GraphBackend, "__protocol_attrs__") or hasattr(GraphBackend, "_is_protocol")


# ---------------------------------------------------------------------------
# GraphRagBackend satisfies Protocol
# ---------------------------------------------------------------------------


class TestGraphRagBackendProtocol:
    def test_isinstance_graphbackend(self):
        backend = _make_graphrag_backend()
        assert isinstance(backend, GraphBackend)

    def test_has_all_required_methods(self):
        backend = _make_graphrag_backend()
        for method in _REQUIRED_METHODS:
            assert hasattr(backend, method), f"Missing method: {method}"
            assert callable(getattr(backend, method)), f"Not callable: {method}"

    def test_ingest_returns_ingest_result(self):
        backend = _make_graphrag_backend()
        result = backend.ingest([{"id": "c1", "text": "hello"}])
        assert isinstance(result, IngestResult)
        assert result.chunks_processed == 1
        assert result.backend_id == "graphrag"

    def test_retrieve_returns_list(self):
        backend = _make_graphrag_backend()
        result = backend.retrieve("query")
        assert isinstance(result, list)

    def test_finalize_returns_finalization_result(self):
        backend = _make_graphrag_backend()
        result = backend.finalize()
        assert isinstance(result, FinalizationResult)
        assert result.backend_id == "graphrag"

    def test_status_returns_dict(self):
        backend = _make_graphrag_backend()
        s = backend.status()
        assert isinstance(s, dict)
        assert s["backend"] == "graphrag"
        assert "entities" in s
        assert "relations" in s

    def test_graph_data_returns_graph_payload(self):
        backend = _make_graphrag_backend()
        payload = backend.graph_data()
        assert isinstance(payload, GraphPayload)
        assert isinstance(payload.nodes, list)
        assert isinstance(payload.links, list)

    def test_graph_data_to_dict_shape(self):
        backend = _make_graphrag_backend()
        d = backend.graph_data().to_dict()
        assert set(d.keys()) == {"nodes", "links"}

    def test_has_entities_false_when_empty(self):
        backend = _make_graphrag_backend()
        assert backend.has_entities() is False

    def test_has_entities_true_when_populated(self):
        backend = _make_graphrag_backend()
        backend._brain._entity_graph = {"alice": {"type": "PERSON"}}
        assert backend.has_entities() is True

    def test_has_community_summaries_false_when_empty(self):
        backend = _make_graphrag_backend()
        assert backend.has_community_summaries() is False

    def test_has_community_summaries_true_when_populated(self):
        backend = _make_graphrag_backend()
        backend._brain._community_summaries = {"0_0": {"full_content": "summary"}}
        assert backend.has_community_summaries() is True


# ---------------------------------------------------------------------------
# GraphRagBackend.clear(persist=True) actually rewrites on-disk state
# ---------------------------------------------------------------------------


class TestGraphRagBackendClearPersist:
    """Regression test: clear(persist=True) must write the now-empty state
    to disk, not just reset it in memory. Before this, collection_ops.py
    called the 7 brain._save_* methods directly and unconditionally — which
    meant non-GraphRAG backends (dynamic_graph, none) never got their
    on-disk state cleared at all, since those direct calls always hit
    GraphRagMixin's own _save_* methods regardless of the active backend.
    Now the backend owns persistence of its own clear.
    """

    @staticmethod
    def _make_real_graphrag_brain(tmp_path):
        from types import SimpleNamespace

        class _RealFakeBrain(GraphRagMixin):
            pass

        brain = _RealFakeBrain()
        brain.config = SimpleNamespace(bm25_path=str(tmp_path))
        brain._entity_graph = {"alice": {"type": "PERSON", "chunk_ids": ["c1"], "degree": 1}}
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}]
        }
        brain._community_levels = {0: {"alice": 0}}
        brain._community_hierarchy = {}
        brain._community_summaries = {"0_0": {"full_content": "a summary"}}
        brain._claims_graph = {"c1": [{"subject": "a", "object": "b", "type": "t"}]}
        brain._entity_embeddings = {"alice": [0.1, 0.2]}
        brain._own_vector_store = MagicMock()
        return brain

    def test_clear_persist_true_rewrites_files_empty(self, tmp_path):
        """Reads back through brain._load_entity_graph()/_load_relation_graph()
        rather than hand-parsing a specific on-disk file — persistence may
        take the msgpack fast path (writing .entity_graph.msgpack and
        deleting any stale .entity_graph.json) or the JSON fallback
        depending on rust-bridge availability; the load path handles both
        transparently, which is what actually matters here.
        """
        brain = self._make_real_graphrag_brain(tmp_path)
        backend = GraphRagBackend(brain)

        # Write real, non-empty state to disk first — proves clear()
        # actually rewrites persisted state rather than it just never
        # having existed.
        for method_name in GraphRagBackend._PERSISTABLE_SAVE_METHODS:
            getattr(brain, method_name)()
        brain._flush_pending_saves()

        assert brain._load_entity_graph()
        assert brain._load_relation_graph()

        backend.clear(persist=True)
        brain._flush_pending_saves()

        assert brain._load_entity_graph() == {}
        assert brain._load_relation_graph() == {}

    def test_clear_persist_false_leaves_disk_state_untouched(self, tmp_path):
        """persist=False (the default) resets in-memory state only — matches
        the read-only-scope-switching requirement that switching scope must
        never write project data to disk.
        """
        brain = self._make_real_graphrag_brain(tmp_path)
        backend = GraphRagBackend(brain)
        for method_name in GraphRagBackend._PERSISTABLE_SAVE_METHODS:
            getattr(brain, method_name)()
        brain._flush_pending_saves()

        assert brain._load_entity_graph()  # non-empty on disk before clear

        backend.clear()  # persist defaults to False
        brain._flush_pending_saves()

        assert brain._entity_graph == {}
        # On-disk state must be untouched — still the pre-clear content.
        assert brain._load_entity_graph()


# ---------------------------------------------------------------------------
# DynamicGraphBackend stub satisfies Protocol
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendProtocol:
    def test_isinstance_graphbackend(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        assert isinstance(backend, GraphBackend)

    def test_has_all_required_methods(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        for method in _REQUIRED_METHODS:
            assert hasattr(backend, method), f"Missing method: {method}"
            assert callable(getattr(backend, method)), f"Not callable: {method}"

    def test_ingest_returns_ingest_result(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        result = backend.ingest([])
        assert isinstance(result, IngestResult)
        assert result.backend_id == "dynamic_graph"

    def test_retrieve_returns_list(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        result = backend.retrieve("q")
        assert isinstance(result, list)

    def test_finalize_returns_finalization_result(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        result = backend.finalize()
        assert isinstance(result, FinalizationResult)
        assert result.backend_id == "dynamic_graph"

    def test_clear_does_not_raise(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        backend.clear()  # should not raise

    def test_delete_documents_does_not_raise(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        backend.delete_documents(["c1"])  # should not raise (empty db)

    def test_status_returns_dict_without_raising(self, tmp_path):
        # status() must never raise — callers use it to probe readiness
        backend = _make_dynamic_backend(tmp_path)
        s = backend.status()
        assert isinstance(s, dict)
        assert s["backend"] == "dynamic_graph"

    def test_graph_data_returns_graph_payload(self, tmp_path):
        backend = _make_dynamic_backend(tmp_path)
        payload = backend.graph_data()
        assert isinstance(payload, GraphPayload)

    def test_has_entities_always_false(self, tmp_path):
        # dynamic_graph tracks its own SQLite tables, not brain._entity_graph —
        # this predicate gates GraphRAG-mixin local/global search, which
        # dynamic_graph never drives.
        assert _make_dynamic_backend(tmp_path).has_entities() is False

    def test_has_community_summaries_always_false(self, tmp_path):
        assert _make_dynamic_backend(tmp_path).has_community_summaries() is False


# ---------------------------------------------------------------------------
# NoneGraphBackend satisfies Protocol
# ---------------------------------------------------------------------------


class TestNoneGraphBackendProtocol:
    def test_isinstance_graphbackend(self):
        backend = _make_none_backend()
        assert isinstance(backend, GraphBackend)

    def test_has_all_required_methods(self):
        backend = _make_none_backend()
        for method in _REQUIRED_METHODS:
            assert hasattr(backend, method), f"Missing method: {method}"
            assert callable(getattr(backend, method)), f"Not callable: {method}"

    def test_ingest_returns_ingest_result(self):
        backend = _make_none_backend()
        result = backend.ingest([{"id": "c1", "text": "hello"}, {"id": "c2", "text": "world"}])
        assert isinstance(result, IngestResult)
        assert result.chunks_processed == 2
        assert result.backend_id == "none"

    def test_retrieve_returns_empty_list(self):
        backend = _make_none_backend()
        result = backend.retrieve("query")
        assert result == []

    def test_finalize_returns_not_applicable(self):
        backend = _make_none_backend()
        result = backend.finalize()
        assert isinstance(result, FinalizationResult)
        assert result.backend_id == "none"
        assert result.status == "not_applicable"
        assert result.detail

    def test_clear_does_not_raise(self):
        _make_none_backend().clear()

    def test_delete_documents_does_not_raise(self):
        _make_none_backend().delete_documents(["c1"])

    def test_status_returns_dict(self):
        s = _make_none_backend().status()
        assert isinstance(s, dict)
        assert s["backend"] == "none"
        assert s["enabled"] is False

    def test_graph_data_returns_empty_graph_payload(self):
        payload = _make_none_backend().graph_data()
        assert isinstance(payload, GraphPayload)
        assert payload.nodes == []
        assert payload.links == []

    def test_no_list_conflicts(self):
        # Matches GraphRagBackend — no conflict tracking, capability probes
        # (hasattr checks in api_routes/graph.py, repl.py) must treat this
        # as "unsupported" rather than crashing.
        assert not hasattr(_make_none_backend(), "list_conflicts")

    def test_has_entities_always_false(self):
        assert _make_none_backend().has_entities() is False

    def test_has_community_summaries_always_false(self):
        assert _make_none_backend().has_community_summaries() is False


# ---------------------------------------------------------------------------
# FederatedGraphBackend delegates has_entities/has_community_summaries
# ---------------------------------------------------------------------------


class TestFederatedGraphBackendProtocol:
    """Unlike NoneGraphBackend/DynamicGraphBackend, Federated must NOT return
    a constant False: it wraps a real GraphRagBackend sharing the same
    brain, and GraphRAG entity extraction runs during ingest independent of
    which graph_backend is selected — so a federated-configured project can
    carry a genuinely populated brain._entity_graph.
    """

    @staticmethod
    def _make_federated(entities_present: bool, summaries_present: bool) -> FederatedGraphBackend:
        backend = FederatedGraphBackend.__new__(FederatedGraphBackend)
        graphrag = MagicMock()
        graphrag.BACKEND_ID = "graphrag"
        graphrag.has_entities.return_value = entities_present
        graphrag.has_community_summaries.return_value = summaries_present
        dynamic = MagicMock()
        dynamic.BACKEND_ID = "dynamic_graph"
        dynamic.has_entities.return_value = False
        dynamic.has_community_summaries.return_value = False
        backend._backends = [graphrag, dynamic]
        backend._weights = {"graphrag": 1.0, "dynamic_graph": 1.0}
        return backend

    def test_has_entities_true_when_sub_backend_has_entities(self):
        backend = self._make_federated(entities_present=True, summaries_present=False)
        assert backend.has_entities() is True

    def test_has_entities_false_when_no_sub_backend_has_entities(self):
        backend = self._make_federated(entities_present=False, summaries_present=False)
        assert backend.has_entities() is False

    def test_has_community_summaries_true_when_sub_backend_has_summaries(self):
        backend = self._make_federated(entities_present=False, summaries_present=True)
        assert backend.has_community_summaries() is True

    def test_has_community_summaries_false_when_no_sub_backend_has_summaries(self):
        backend = self._make_federated(entities_present=False, summaries_present=False)
        assert backend.has_community_summaries() is False

    def test_has_entities_survives_sub_backend_exception(self):
        backend = self._make_federated(entities_present=True, summaries_present=False)
        backend._backends[0].has_entities.side_effect = RuntimeError("boom")
        # First sub-backend raises; second (dynamic) reports False — overall False,
        # and the exception must not propagate.
        assert backend.has_entities() is False


# ---------------------------------------------------------------------------
# Minimal hand-rolled object satisfies Protocol
# ---------------------------------------------------------------------------


class TestMinimalProtocolConformance:
    def test_minimal_object_passes_isinstance(self):
        class _Minimal:
            def ingest(self, chunks):
                return IngestResult()

            def retrieve(self, query, cfg=None, existing_results=None):
                return []

            def finalize(self, force=False):
                return FinalizationResult()

            def clear(self):
                pass

            def delete_documents(self, chunk_ids):
                pass

            def status(self):
                return {}

            def graph_data(self, filters=None):
                return GraphPayload()

            def has_entities(self):
                return False

            def has_community_summaries(self):
                return False

        assert isinstance(_Minimal(), GraphBackend)

    def test_missing_one_method_fails_isinstance(self):
        class _Incomplete:
            def ingest(self, chunks):
                return IngestResult()

            def retrieve(self, query, cfg=None, existing_results=None):
                return []

            def finalize(self, force=False):
                return FinalizationResult()

            def clear(self):
                pass

            def delete_documents(self, chunk_ids):
                pass

            def status(self):
                return {}

            # graph_data is intentionally missing

        assert not isinstance(_Incomplete(), GraphBackend)


# ---------------------------------------------------------------------------
# Data type smoke tests
# ---------------------------------------------------------------------------


class TestDataTypes:
    def test_graph_context_defaults(self):
        ctx = GraphContext(
            context_id="x",
            context_type="entity",
            text="hello",
            score=0.9,
            rank=0,
            backend_id="graphrag",
        )
        assert ctx.valid_at is None
        assert ctx.invalid_at is None
        assert ctx.evidence_ids == []
        assert ctx.matched_entity_names == []

    def test_graph_data_filters_defaults(self):
        f = GraphDataFilters()
        assert f.entity_types is None
        assert f.min_degree == 0
        assert f.limit is None

    def test_retrieval_config_default_top_k(self):
        cfg = RetrievalConfig()
        assert cfg.top_k == 10

    def test_graph_payload_to_dict(self):
        p = GraphPayload(nodes=[{"id": "a"}], links=[])
        d = p.to_dict()
        assert d == {"nodes": [{"id": "a"}], "links": []}


# ---------------------------------------------------------------------------
# Phase 2 enforcement target — tracks shim removal goal
# ---------------------------------------------------------------------------


class TestPhase2ShimRemoval:
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Phase 2 target: AxonBrain should not inherit from GraphRagMixin once "
            "all graph ops are fully routed through self._graph_backend.*. "
            "Flip to xfail(strict=True) or remove when Phase 2 is complete."
        ),
    )
    def test_axon_brain_does_not_inherit_graphragmixin(self):
        """Verify via AST (not grep) that AxonBrain no longer inherits GraphRagMixin.

        This test is currently XFAIL — it marks the Phase 2 architectural goal.
        When AxonBrain's GraphRagMixin inheritance is removed, this test will pass
        and should be promoted to a strict (non-xfail) assertion.
        """
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent / "src" / "axon" / "main.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "AxonBrain":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "GraphRagMixin" not in base_names, (
                    "AxonBrain still inherits from GraphRagMixin. "
                    "Phase 2 requires removing this inheritance and routing all "
                    "graph operations through self._graph_backend.*"
                )
                return
        pytest.fail("Could not find AxonBrain class in main.py")
