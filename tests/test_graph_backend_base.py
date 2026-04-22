"""Architecture tests for the GraphBackend Protocol.

Verifies:
  1. The Protocol exposes exactly the 7 required methods.
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
from axon.graph_backends.graphrag_backend import GraphRagBackend

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


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


class TestProtocolShape:
    def test_required_methods_count(self):
        assert len(_REQUIRED_METHODS) == 7

    def test_required_method_names(self):
        expected = {
            "ingest",
            "retrieve",
            "finalize",
            "clear",
            "delete_documents",
            "status",
            "graph_data",
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
