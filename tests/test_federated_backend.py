"""Tests for FederatedGraphBackend (src/axon/graph_backends/federated_backend.py).

The federated backend is the weighted-RRF fusion of GraphRagBackend (the durable
entity graph) and DynamicGraphBackend (bi-temporal, time-bounded facts) — i.e.
the mechanism by which one retrieve() returns both long-lived and fast-changing
knowledge. It shipped with zero dedicated tests; this file covers the fusion
maths and the delegation/fault-isolation behaviour of every other method.

Sub-backends are faked rather than constructed for real: the point is the
federation logic, and building a real GraphRAG + SQLite pair per test would be
slow and would couple these tests to two unrelated subsystems.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axon.graph_backends.base import FinalizationResult, GraphContext, IngestResult
from axon.graph_backends.federated_backend import (
    BACKEND_ID,
    FederatedGraphBackend,
    _weighted_rrf,
)


def _ctx(cid: str, score: float, backend_id: str, *, chunk: str = "", rank: int = 0):
    return GraphContext(
        context_id=cid,
        context_type="fact",
        text=f"text-{cid}",
        score=score,
        rank=rank,
        backend_id=backend_id,
        source_chunk_id=chunk,
    )


class _FakeBackend:
    """Minimal stand-in for a GraphBackend sub-backend."""

    def __init__(self, backend_id, contexts=None, **behaviour):
        self.BACKEND_ID = backend_id
        self._contexts = contexts or []
        self._b = behaviour
        self.calls: list[str] = []

    def retrieve(self, query, cfg=None, existing_results=None):
        self.calls.append("retrieve")
        if self._b.get("retrieve_raises"):
            raise RuntimeError("retrieve boom")
        return list(self._contexts)

    def ingest(self, chunks):
        self.calls.append("ingest")
        if self._b.get("ingest_raises"):
            raise RuntimeError("ingest boom")
        return self._b.get(
            "ingest_result", IngestResult(backend_id=self.BACKEND_ID, chunks_processed=len(chunks))
        )

    def finalize(self, force=False):
        self.calls.append("finalize")
        if self._b.get("finalize_raises"):
            raise RuntimeError("finalize boom")
        return self._b.get(
            "finalize_result", FinalizationResult(backend_id=self.BACKEND_ID, status="ok")
        )

    def clear(self, *, persist=False):
        self.calls.append("clear")
        if self._b.get("clear_raises"):
            raise RuntimeError("clear boom")

    def delete_documents(self, chunk_ids):
        self.calls.append("delete_documents")
        if self._b.get("delete_raises"):
            raise RuntimeError("delete boom")

    def close(self):
        self.calls.append("close")


def _federated(*backends, weights=None):
    """Build a FederatedGraphBackend with the given fake sub-backends."""
    fed = FederatedGraphBackend.__new__(FederatedGraphBackend)
    fed._backends = list(backends)
    fed._weights = weights or {"graphrag": 1.0, "dynamic_graph": 1.0}
    return fed


# ---------------------------------------------------------------------------
# _weighted_rrf — the fusion maths
# ---------------------------------------------------------------------------


class TestWeightedRRF:
    def test_empty_input_returns_empty(self):
        assert _weighted_rrf({}, {}) == []

    def test_single_backend_preserves_score_order(self):
        got = _weighted_rrf(
            {"graphrag": [_ctx("a", 0.1, "graphrag"), _ctx("b", 0.9, "graphrag")]},
            {"graphrag": 1.0},
        )
        assert [c.context_id for c in got] == ["b", "a"]

    def test_rank_is_reassigned_sequentially(self):
        got = _weighted_rrf(
            {"graphrag": [_ctx("a", 0.9, "graphrag"), _ctx("b", 0.5, "graphrag")]},
            {"graphrag": 1.0},
        )
        assert [c.rank for c in got] == [0, 1]

    def test_higher_weight_backend_outranks_lower(self):
        """The weight is the whole point: a backend the caller trusts more for
        this query should surface above an equally-ranked result from the other."""
        got = _weighted_rrf(
            {
                "graphrag": [_ctx("g1", 0.5, "graphrag")],
                "dynamic_graph": [_ctx("d1", 0.5, "dynamic_graph")],
            },
            {"graphrag": 0.1, "dynamic_graph": 10.0},
        )
        assert [c.context_id for c in got] == ["d1", "g1"]

    def test_weights_are_symmetric(self):
        """Same setup with the weights swapped must flip the order — guards
        against a hardcoded backend preference."""
        per = {
            "graphrag": [_ctx("g1", 0.5, "graphrag")],
            "dynamic_graph": [_ctx("d1", 0.5, "dynamic_graph")],
        }
        first = _weighted_rrf(per, {"graphrag": 10.0, "dynamic_graph": 0.1})
        second = _weighted_rrf(per, {"graphrag": 0.1, "dynamic_graph": 10.0})
        assert [c.context_id for c in first] == ["g1", "d1"]
        assert [c.context_id for c in second] == ["d1", "g1"]

    def test_same_context_id_from_both_backends_sums_scores(self):
        """Agreement across backends should reinforce, not overwrite."""
        both = _weighted_rrf(
            {
                "graphrag": [_ctx("shared", 0.5, "graphrag")],
                "dynamic_graph": [_ctx("shared", 0.5, "dynamic_graph")],
            },
            {"graphrag": 1.0, "dynamic_graph": 1.0},
        )
        one = _weighted_rrf({"graphrag": [_ctx("shared", 0.5, "graphrag")]}, {"graphrag": 1.0})
        assert len(both) == 1
        assert both[0].score > one[0].score

    def test_same_source_chunk_is_deduped_keeping_higher_score(self):
        """Both backends can derive different contexts from one chunk; the
        caller should not see the same underlying evidence twice."""
        got = _weighted_rrf(
            {
                "graphrag": [_ctx("g1", 0.9, "graphrag", chunk="chunk-7")],
                "dynamic_graph": [_ctx("d1", 0.9, "dynamic_graph", chunk="chunk-7")],
            },
            {"graphrag": 10.0, "dynamic_graph": 0.1},
        )
        assert len(got) == 1
        assert got[0].context_id == "g1"

    def test_distinct_source_chunks_are_not_deduped(self):
        got = _weighted_rrf(
            {
                "graphrag": [_ctx("g1", 0.9, "graphrag", chunk="chunk-1")],
                "dynamic_graph": [_ctx("d1", 0.9, "dynamic_graph", chunk="chunk-2")],
            },
            {"graphrag": 1.0, "dynamic_graph": 1.0},
        )
        assert {c.context_id for c in got} == {"g1", "d1"}

    def test_empty_source_chunk_id_never_dedupes(self):
        """A blank source_chunk_id is 'unknown', not 'the same chunk'."""
        got = _weighted_rrf(
            {
                "graphrag": [_ctx("g1", 0.9, "graphrag", chunk="")],
                "dynamic_graph": [_ctx("d1", 0.8, "dynamic_graph", chunk="")],
            },
            {"graphrag": 1.0, "dynamic_graph": 1.0},
        )
        assert len(got) == 2

    def test_missing_weight_defaults_to_one(self):
        got = _weighted_rrf({"unknown_backend": [_ctx("x", 0.5, "unknown_backend")]}, {})
        assert len(got) == 1 and got[0].score > 0


# ---------------------------------------------------------------------------
# retrieve — concurrency, weight override, fault isolation
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_no_backends_returns_empty(self):
        assert _federated().retrieve("q") == []

    def test_fuses_results_from_both_backends(self):
        fed = _federated(
            _FakeBackend("graphrag", [_ctx("g1", 0.9, "graphrag")]),
            _FakeBackend("dynamic_graph", [_ctx("d1", 0.8, "dynamic_graph")]),
        )
        assert {c.context_id for c in fed.retrieve("q")} == {"g1", "d1"}

    def test_per_query_weights_override_project_defaults(self):
        """cfg.federation_weights is the documented per-call override."""
        fed = _federated(
            _FakeBackend("graphrag", [_ctx("g1", 0.5, "graphrag")]),
            _FakeBackend("dynamic_graph", [_ctx("d1", 0.5, "dynamic_graph")]),
            weights={"graphrag": 10.0, "dynamic_graph": 0.1},
        )
        cfg = SimpleNamespace(federation_weights={"graphrag": 0.1, "dynamic_graph": 10.0})
        assert [c.context_id for c in fed.retrieve("q", cfg)][0] == "d1"

    def test_partial_per_query_override_keeps_the_other_default(self):
        fed = _federated(
            _FakeBackend("graphrag", [_ctx("g1", 0.5, "graphrag")]),
            _FakeBackend("dynamic_graph", [_ctx("d1", 0.5, "dynamic_graph")]),
            weights={"graphrag": 1.0, "dynamic_graph": 5.0},
        )
        cfg = SimpleNamespace(federation_weights={"graphrag": 50.0})
        assert [c.context_id for c in fed.retrieve("q", cfg)][0] == "g1"

    def test_empty_override_dict_falls_back_to_project_weights(self):
        fed = _federated(
            _FakeBackend("graphrag", [_ctx("g1", 0.5, "graphrag")]),
            _FakeBackend("dynamic_graph", [_ctx("d1", 0.5, "dynamic_graph")]),
            weights={"graphrag": 0.1, "dynamic_graph": 10.0},
        )
        assert [c.context_id for c in fed.retrieve("q", SimpleNamespace(federation_weights={}))][
            0
        ] == "d1"

    def test_one_backend_failing_does_not_lose_the_other(self):
        """Fault isolation is the reason retrieve() catches per-future: a
        dynamic-graph SQLite error must not blank out GraphRAG's answer."""
        fed = _federated(
            _FakeBackend("graphrag", [_ctx("g1", 0.9, "graphrag")]),
            _FakeBackend("dynamic_graph", retrieve_raises=True),
        )
        got = fed.retrieve("q")
        assert [c.context_id for c in got] == ["g1"]

    def test_both_backends_failing_returns_empty_not_raise(self):
        fed = _federated(
            _FakeBackend("graphrag", retrieve_raises=True),
            _FakeBackend("dynamic_graph", retrieve_raises=True),
        )
        assert fed.retrieve("q") == []

    def test_every_backend_is_queried(self):
        a = _FakeBackend("graphrag", [])
        b = _FakeBackend("dynamic_graph", [])
        _federated(a, b).retrieve("q")
        assert "retrieve" in a.calls and "retrieve" in b.calls


# ---------------------------------------------------------------------------
# ingest / finalize / list_conflicts / clear / delete_documents
# ---------------------------------------------------------------------------


class TestIngest:
    def test_sums_entities_and_relations_across_backends(self):
        fed = _federated(
            _FakeBackend(
                "graphrag",
                ingest_result=IngestResult(
                    backend_id="graphrag", entities_added=3, relations_added=2, chunks_processed=5
                ),
            ),
            _FakeBackend(
                "dynamic_graph",
                ingest_result=IngestResult(
                    backend_id="dynamic_graph",
                    entities_added=4,
                    relations_added=1,
                    chunks_processed=5,
                ),
            ),
        )
        r = fed.ingest([{"id": "c1"}])
        assert (r.entities_added, r.relations_added) == (7, 3)

    def test_chunks_processed_is_max_not_sum(self):
        """Both backends see the same chunks; summing would double-count."""
        fed = _federated(
            _FakeBackend(
                "graphrag", ingest_result=IngestResult(backend_id="graphrag", chunks_processed=5)
            ),
            _FakeBackend(
                "dynamic_graph",
                ingest_result=IngestResult(backend_id="dynamic_graph", chunks_processed=5),
            ),
        )
        assert fed.ingest([{"id": "c1"}]).chunks_processed == 5

    def test_one_backend_failing_still_records_the_other(self):
        fed = _federated(
            _FakeBackend(
                "graphrag", ingest_result=IngestResult(backend_id="graphrag", entities_added=3)
            ),
            _FakeBackend("dynamic_graph", ingest_raises=True),
        )
        assert fed.ingest([{"id": "c1"}]).entities_added == 3

    def test_result_is_tagged_with_the_federation_id(self):
        """The caller should see the federation as the source, not whichever
        sub-backend happened to answer last."""
        result = _federated(_FakeBackend("graphrag")).ingest([])
        assert result.backend_id == BACKEND_ID


class TestFinalize:
    def test_error_beats_ok(self):
        """Status priority is error > ok > not_applicable — masking a real
        failure as 'ok' would tell the caller the graph is consistent when it
        is not."""
        fed = _federated(
            _FakeBackend(
                "graphrag", finalize_result=FinalizationResult(backend_id="graphrag", status="ok")
            ),
            _FakeBackend(
                "dynamic_graph",
                finalize_result=FinalizationResult(backend_id="dynamic_graph", status="error"),
            ),
        )
        assert fed.finalize().status == "error"

    def test_ok_beats_not_applicable(self):
        fed = _federated(
            _FakeBackend(
                "graphrag", finalize_result=FinalizationResult(backend_id="graphrag", status="ok")
            ),
            _FakeBackend(
                "dynamic_graph",
                finalize_result=FinalizationResult(
                    backend_id="dynamic_graph", status="not_applicable"
                ),
            ),
        )
        assert fed.finalize().status == "ok"

    def test_all_not_applicable_stays_not_applicable(self):
        fed = _federated(
            _FakeBackend(
                "graphrag",
                finalize_result=FinalizationResult(backend_id="graphrag", status="not_applicable"),
            ),
            _FakeBackend(
                "dynamic_graph",
                finalize_result=FinalizationResult(
                    backend_id="dynamic_graph", status="not_applicable"
                ),
            ),
        )
        assert fed.finalize().status == "not_applicable"

    def test_raising_backend_becomes_error_status(self):
        fed = _federated(
            _FakeBackend(
                "graphrag", finalize_result=FinalizationResult(backend_id="graphrag", status="ok")
            ),
            _FakeBackend("dynamic_graph", finalize_raises=True),
        )
        r = fed.finalize()
        assert r.status == "error"
        assert "dynamic_graph" in r.detail

    def test_communities_built_is_summed(self):
        fed = _federated(
            _FakeBackend(
                "graphrag",
                finalize_result=FinalizationResult(
                    backend_id="graphrag", status="ok", communities_built=4
                ),
            ),
            _FakeBackend(
                "dynamic_graph",
                finalize_result=FinalizationResult(
                    backend_id="dynamic_graph", status="ok", communities_built=1
                ),
            ),
        )
        assert fed.finalize().communities_built == 5


class TestListConflicts:
    def test_rows_are_tagged_with_their_backend(self):
        a = _FakeBackend("graphrag")
        a.list_conflicts = lambda limit: [{"fact_id": "f1"}]
        b = _FakeBackend("dynamic_graph")
        b.list_conflicts = lambda limit: [{"fact_id": "f2"}]
        rows = _federated(a, b).list_conflicts()
        assert {r["backend"] for r in rows} == {"graphrag", "dynamic_graph"}

    def test_limit_budget_is_shared_across_backends(self):
        a = _FakeBackend("graphrag")
        a.list_conflicts = lambda limit: [{"fact_id": f"f{i}"} for i in range(limit)]
        b = _FakeBackend("dynamic_graph")
        b.list_conflicts = lambda limit: [{"fact_id": "should-not-appear"}]
        rows = _federated(a, b).list_conflicts(limit=3)
        assert len(rows) == 3

    def test_backend_without_list_conflicts_is_skipped(self):
        a = _FakeBackend("graphrag")  # no list_conflicts attribute
        b = _FakeBackend("dynamic_graph")
        b.list_conflicts = lambda limit: [{"fact_id": "f1"}]
        assert len(_federated(a, b).list_conflicts()) == 1

    def test_raising_backend_is_skipped_not_fatal(self):
        def _boom(limit):
            raise RuntimeError("conflicts boom")

        a = _FakeBackend("graphrag")
        a.list_conflicts = _boom
        b = _FakeBackend("dynamic_graph")
        b.list_conflicts = lambda limit: [{"fact_id": "f1"}]
        assert len(_federated(a, b).list_conflicts()) == 1


class TestClearAndDelete:
    def test_clear_reaches_every_backend_despite_a_failure(self):
        """A silent clear() failure would leave stale state in one half of the
        federation while the caller believes it was wiped."""
        a = _FakeBackend("graphrag", clear_raises=True)
        b = _FakeBackend("dynamic_graph")
        _federated(a, b).clear()
        assert "clear" in b.calls

    def test_delete_documents_reaches_every_backend_despite_a_failure(self):
        a = _FakeBackend("graphrag", delete_raises=True)
        b = _FakeBackend("dynamic_graph")
        _federated(a, b).delete_documents(["c1"])
        assert "delete_documents" in b.calls


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """__init__ builds both sub-backends eagerly. patch() targets are real
    callables, not MagicMocks: the production warning path formats
    ``cls.__name__``, which an auto-specced Mock does not provide."""

    def _brain(self, weights=None):
        return SimpleNamespace(config=SimpleNamespace(graph_federation_weights=weights or {}))

    @staticmethod
    def _ok(backend_id):
        class _Ctor:
            def __new__(cls, _brain):
                return _FakeBackend(backend_id)

        _Ctor.__name__ = backend_id
        return _Ctor

    @staticmethod
    def _boom(backend_id):
        class _Ctor:
            def __new__(cls, _brain):
                raise RuntimeError("cannot init")

        _Ctor.__name__ = backend_id
        return _Ctor

    def _patched(self, graphrag_ctor, dynamic_ctor):
        return (
            patch("axon.graph_backends.graphrag_backend.GraphRagBackend", graphrag_ctor),
            patch("axon.graph_backends.dynamic_graph_backend.DynamicGraphBackend", dynamic_ctor),
        )

    def test_weights_are_read_from_config(self):
        a, b = self._patched(self._ok("graphrag"), self._ok("dynamic_graph"))
        with a, b:
            fed = FederatedGraphBackend(self._brain({"graphrag": 2.0, "dynamic_graph": 0.5}))
        assert fed._weights == {"graphrag": 2.0, "dynamic_graph": 0.5}

    def test_weights_default_to_one_when_unconfigured(self):
        a, b = self._patched(self._ok("graphrag"), self._ok("dynamic_graph"))
        with a, b:
            fed = FederatedGraphBackend(self._brain())
        assert fed._weights == {"graphrag": 1.0, "dynamic_graph": 1.0}

    def test_a_backend_that_fails_to_construct_is_dropped_not_fatal(self):
        """Documented behaviour: e.g. the [sealed] extra missing for dynamic
        graph must not take the whole federation down."""
        a, b = self._patched(self._ok("graphrag"), self._boom("dynamic_graph"))
        with a, b:
            fed = FederatedGraphBackend(self._brain())
        assert [sub.BACKEND_ID for sub in fed._backends] == ["graphrag"]

    def test_all_backends_failing_leaves_an_inert_federation(self):
        """An inert federation must degrade to empty results, not raise on use."""
        a, b = self._patched(self._boom("graphrag"), self._boom("dynamic_graph"))
        with a, b:
            fed = FederatedGraphBackend(self._brain())
        assert fed._backends == []
        assert fed.retrieve("q") == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
