"""Tests for v0.3.2 graph surface additions:

- Item (1): capability flags on FinalizationResult (status="not_applicable")
- Item (5): federated weight per-query override via RetrievalConfig
- Item (8): point-in-time pass-through (RetrievalConfig.point_in_time)
- Item (9): list_conflicts() on DynamicGraphBackend + federated aggregation

Each test runs offline — LLM calls are mocked so the suite has no
network or model dependency.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from axon.graph_backends.base import (
    FinalizationResult,
    GraphContext,
    RetrievalConfig,
)
from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend
from axon.graph_backends.federated_backend import FederatedGraphBackend

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_brain(tmp_path, federation_weights: dict | None = None):
    cfg = SimpleNamespace(
        bm25_path=str(tmp_path),
        graph_backend="dynamic_graph",
        graph_federation_weights=federation_weights or {},
    )
    llm = MagicMock()
    llm.complete.return_value = ""
    return SimpleNamespace(config=cfg, llm=llm)


def _insert_fact(
    backend: DynamicGraphBackend,
    *,
    fact_id: str,
    subject: str,
    relation: str,
    object_: str,
    valid_at: datetime,
    status: str = "active",
    scope_key: str | None = None,
    confidence: float = 1.0,
) -> None:
    """Direct SQL insert so we can test list_conflicts without going through
    the full ingest path (which requires LLM responses)."""
    with backend._write_lock:
        backend._conn.execute(
            "INSERT INTO facts (fact_id, subject, relation, object, valid_at, "
            "status, scope_key, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')",
            (
                fact_id,
                subject,
                relation,
                object_,
                valid_at.isoformat(),
                status,
                scope_key,
                confidence,
            ),
        )
        backend._conn.commit()


# ---------------------------------------------------------------------------
# Item (1): Capability flags on FinalizationResult
# ---------------------------------------------------------------------------


class TestCapabilityFlags:
    def test_finalization_result_has_status_field(self):
        r = FinalizationResult()
        assert r.status == "ok"
        assert r.detail == ""

    def test_dynamic_graph_finalize_returns_not_applicable(self, tmp_path):
        brain = _make_brain(tmp_path)
        backend = DynamicGraphBackend(brain)
        result = backend.finalize(force=True)
        assert result.status == "not_applicable"
        assert result.backend_id == "dynamic_graph"
        assert "community" in result.detail.lower()

    def test_federated_finalize_aggregates_status(self, tmp_path):
        brain = _make_brain(tmp_path, federation_weights={})

        # Patch GraphRagBackend.finalize so the federation includes one "ok"
        # backend and one "not_applicable" backend; expect aggregate "ok".
        def _patched_init(self, _brain):
            self._backends = []
            self._weights = {"graphrag": 1.0, "dynamic_graph": 1.0}
            ok_backend = MagicMock()
            ok_backend.BACKEND_ID = "graphrag"
            ok_backend.finalize = MagicMock(
                return_value=FinalizationResult(
                    backend_id="graphrag",
                    communities_built=2,
                    status="ok",
                )
            )
            self._backends.append(ok_backend)
            self._backends.append(DynamicGraphBackend(_brain))

        original = FederatedGraphBackend.__init__
        try:
            FederatedGraphBackend.__init__ = _patched_init  # type: ignore[assignment]
            fed = FederatedGraphBackend(brain)
            r = fed.finalize(force=True)
            assert r.status == "ok"
            assert r.backend_id == "federated"
            assert r.communities_built == 2
        finally:
            FederatedGraphBackend.__init__ = original  # type: ignore[assignment]

    def test_federated_finalize_all_not_applicable(self, tmp_path):
        brain = _make_brain(tmp_path, federation_weights={})

        def _patched_init(self, _brain):
            self._backends = [DynamicGraphBackend(_brain), DynamicGraphBackend(_brain)]
            self._weights = {"graphrag": 1.0, "dynamic_graph": 1.0}

        original = FederatedGraphBackend.__init__
        try:
            FederatedGraphBackend.__init__ = _patched_init  # type: ignore[assignment]
            fed = FederatedGraphBackend(brain)
            r = fed.finalize(force=True)
            assert r.status == "not_applicable"
        finally:
            FederatedGraphBackend.__init__ = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Item (5): Federated weight per-query override
# ---------------------------------------------------------------------------


class TestFederationWeightOverride:
    def test_retrieval_config_has_federation_weights(self):
        cfg = RetrievalConfig(federation_weights={"graphrag": 0.3, "dynamic_graph": 0.7})
        assert cfg.federation_weights == {"graphrag": 0.3, "dynamic_graph": 0.7}
        assert RetrievalConfig().federation_weights is None

    def test_per_query_weights_override_project_default(self, tmp_path):
        """Per-query cfg.federation_weights wins over self._weights."""
        brain = _make_brain(tmp_path, federation_weights={"graphrag": 1.0, "dynamic_graph": 1.0})

        # Build two fake backends that return one context each.
        gr_ctx = GraphContext(
            context_id="gr-1",
            context_type="entity",
            text="g",
            score=1.0,
            rank=0,
            backend_id="graphrag",
            source_chunk_id="chunk-A",
        )
        dy_ctx = GraphContext(
            context_id="dy-1",
            context_type="fact",
            text="d",
            score=1.0,
            rank=0,
            backend_id="dynamic_graph",
            source_chunk_id="chunk-B",
        )

        captured_weights: dict = {}

        def _patched_init(self, _brain):
            self._weights = {"graphrag": 1.0, "dynamic_graph": 1.0}
            gr = MagicMock()
            gr.BACKEND_ID = "graphrag"
            gr.retrieve = MagicMock(return_value=[gr_ctx])
            dy = MagicMock()
            dy.BACKEND_ID = "dynamic_graph"
            dy.retrieve = MagicMock(return_value=[dy_ctx])
            self._backends = [gr, dy]

        original_init = FederatedGraphBackend.__init__
        # Hook the fusion call so we can capture weights actually applied.
        from axon.graph_backends import federated_backend as fed_mod

        original_rrf = fed_mod._weighted_rrf

        def _capturing_rrf(per_backend, weights, k=60):
            captured_weights.update(weights)
            return original_rrf(per_backend, weights, k=k)

        try:
            FederatedGraphBackend.__init__ = _patched_init  # type: ignore[assignment]
            fed_mod._weighted_rrf = _capturing_rrf  # type: ignore[assignment]
            fed = FederatedGraphBackend(brain)
            cfg = RetrievalConfig(
                top_k=5,
                federation_weights={"graphrag": 0.1, "dynamic_graph": 9.0},
            )
            fed.retrieve("test", cfg)
            assert captured_weights["graphrag"] == pytest.approx(0.1)
            assert captured_weights["dynamic_graph"] == pytest.approx(9.0)
        finally:
            FederatedGraphBackend.__init__ = original_init  # type: ignore[assignment]
            fed_mod._weighted_rrf = original_rrf  # type: ignore[assignment]

    def test_request_schema_rejects_unknown_keys(self):
        """GraphRetrieveRequest rejects federation_weights with non-canonical keys."""
        from axon.api_schemas import GraphRetrieveRequest

        with pytest.raises(Exception, match="unknown key"):
            GraphRetrieveRequest(
                query="hello",
                federation_weights={"graphrag": 1.0, "made_up_backend": 0.5},
            )

    def test_request_schema_rejects_negative_weights(self):
        """GraphRetrieveRequest rejects negative federation_weights values."""
        from axon.api_schemas import GraphRetrieveRequest

        with pytest.raises(Exception, match=">= 0"):
            GraphRetrieveRequest(
                query="hello",
                federation_weights={"graphrag": -1.0, "dynamic_graph": 1.0},
            )

    def test_request_schema_accepts_valid_weights(self):
        """Valid federation_weights pass through unchanged."""
        from axon.api_schemas import GraphRetrieveRequest

        req = GraphRetrieveRequest(
            query="hello",
            federation_weights={"graphrag": 0.3, "dynamic_graph": 1.7},
        )
        assert req.federation_weights == {"graphrag": 0.3, "dynamic_graph": 1.7}

    def test_no_override_uses_project_default(self, tmp_path):
        brain = _make_brain(tmp_path, federation_weights={"graphrag": 2.0, "dynamic_graph": 0.5})

        captured_weights: dict = {}

        def _patched_init(self, _brain):
            self._weights = {"graphrag": 2.0, "dynamic_graph": 0.5}
            self._backends = []

        from axon.graph_backends import federated_backend as fed_mod

        original_init = FederatedGraphBackend.__init__
        original_rrf = fed_mod._weighted_rrf

        def _capturing_rrf(per_backend, weights, k=60):
            captured_weights.update(weights)
            return []

        try:
            FederatedGraphBackend.__init__ = _patched_init  # type: ignore[assignment]
            fed_mod._weighted_rrf = _capturing_rrf  # type: ignore[assignment]
            fed = FederatedGraphBackend(brain)
            # Add a fake backend so retrieve() doesn't short-circuit
            fake = MagicMock()
            fake.BACKEND_ID = "graphrag"
            fake.retrieve = MagicMock(return_value=[])
            fed._backends = [fake]
            fed.retrieve("test", RetrievalConfig(top_k=5))  # No override
            assert captured_weights["graphrag"] == pytest.approx(2.0)
            assert captured_weights["dynamic_graph"] == pytest.approx(0.5)
        finally:
            FederatedGraphBackend.__init__ = original_init  # type: ignore[assignment]
            fed_mod._weighted_rrf = original_rrf  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Item (9): list_conflicts() on DynamicGraphBackend + federated aggregation
# ---------------------------------------------------------------------------


class TestListConflicts:
    def test_empty_db_returns_empty_list(self, tmp_path):
        brain = _make_brain(tmp_path)
        backend = DynamicGraphBackend(brain)
        assert backend.list_conflicts() == []

    def test_returns_only_conflicted_facts(self, tmp_path):
        brain = _make_brain(tmp_path)
        backend = DynamicGraphBackend(brain)
        now = datetime.now(timezone.utc)
        _insert_fact(
            backend,
            fact_id="active-1",
            subject="alice",
            relation="MARRIED_TO",
            object_="bob",
            valid_at=now,
            status="active",
        )
        _insert_fact(
            backend,
            fact_id="conflict-1",
            subject="alice",
            relation="MARRIED_TO",
            object_="carol",
            valid_at=now,
            status="conflicted",
            scope_key="alice:MARRIED_TO",
        )
        _insert_fact(
            backend,
            fact_id="conflict-2",
            subject="alice",
            relation="MARRIED_TO",
            object_="dave",
            valid_at=now - timedelta(days=1),
            status="conflicted",
            scope_key="alice:MARRIED_TO",
        )
        rows = backend.list_conflicts()
        assert len(rows) == 2
        ids = {r["fact_id"] for r in rows}
        assert ids == {"conflict-1", "conflict-2"}
        # Newest first
        assert rows[0]["fact_id"] == "conflict-1"

    def test_limit_is_honoured(self, tmp_path):
        brain = _make_brain(tmp_path)
        backend = DynamicGraphBackend(brain)
        now = datetime.now(timezone.utc)
        for i in range(5):
            _insert_fact(
                backend,
                fact_id=f"conf-{i}",
                subject="x",
                relation="LEADS",
                object_=f"team-{i}",
                valid_at=now - timedelta(seconds=i),
                status="conflicted",
                scope_key="x:LEADS",
            )
        rows = backend.list_conflicts(limit=2)
        assert len(rows) == 2

    def test_federated_list_conflicts_aggregates(self, tmp_path):
        brain = _make_brain(tmp_path, federation_weights={})

        def _patched_init(self, _brain):
            dyn = DynamicGraphBackend(_brain)
            now = datetime.now(timezone.utc)
            _insert_fact(
                dyn,
                fact_id="fed-conf-1",
                subject="x",
                relation="MARRIED_TO",
                object_="y",
                valid_at=now,
                status="conflicted",
                scope_key="x:MARRIED_TO",
            )
            self._backends = [dyn]  # only one sub-backend
            self._weights = {"graphrag": 1.0, "dynamic_graph": 1.0}

        original_init = FederatedGraphBackend.__init__
        try:
            FederatedGraphBackend.__init__ = _patched_init  # type: ignore[assignment]
            fed = FederatedGraphBackend(brain)
            rows = fed.list_conflicts(limit=10)
            assert len(rows) == 1
            assert rows[0]["fact_id"] == "fed-conf-1"
            # Federation tags each row with its source backend
            assert rows[0]["backend"] == "dynamic_graph"
        finally:
            FederatedGraphBackend.__init__ = original_init  # type: ignore[assignment]

    def test_federated_list_conflicts_when_no_subbackend_supports_it(self, tmp_path):
        brain = _make_brain(tmp_path, federation_weights={})

        def _patched_init(self, _brain):
            stub = MagicMock(spec=[])  # no list_conflicts attribute
            stub.BACKEND_ID = "graphrag"
            self._backends = [stub]
            self._weights = {"graphrag": 1.0, "dynamic_graph": 1.0}

        original_init = FederatedGraphBackend.__init__
        try:
            FederatedGraphBackend.__init__ = _patched_init  # type: ignore[assignment]
            fed = FederatedGraphBackend(brain)
            assert fed.list_conflicts() == []
        finally:
            FederatedGraphBackend.__init__ = original_init  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Item (8): point-in-time pass-through (already implemented; cover the
# config plumbing so a regression in RetrievalConfig is caught early).
# ---------------------------------------------------------------------------


class TestPointInTimeRetrieval:
    def test_retrieval_config_default_is_none(self):
        assert RetrievalConfig().point_in_time is None

    def test_dynamic_graph_filters_by_point_in_time(self, tmp_path):
        """At ``point_in_time=T``, only facts whose [valid_at, invalid_at)
        window covers T must be returned."""
        brain = _make_brain(tmp_path)
        backend = DynamicGraphBackend(brain)
        t_early = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t_mid = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t_late = datetime(2025, 12, 1, tzinfo=timezone.utc)
        # Fact A: alice WORKS_AT acme, valid Jan-Jun (closed window).
        # Fact B: zara WORKS_AT globex, valid Jul-onwards (open window).
        # Use disjoint entities so multi-hop BFS doesn't bring B into A's
        # neighborhood (multi-hop doesn't re-apply point_in_time today; that
        # gap is tracked separately).
        with backend._write_lock:
            backend._conn.execute(
                "INSERT INTO facts (fact_id, subject, relation, object, valid_at, "
                "invalid_at, status, scope_key, confidence, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')",
                (
                    "fA",
                    "alice",
                    "WORKS_AT",
                    "acme",
                    t_early.isoformat(),
                    t_mid.isoformat(),
                    "active",
                    None,
                    1.0,
                ),
            )
            backend._conn.execute(
                "INSERT INTO facts (fact_id, subject, relation, object, valid_at, "
                "invalid_at, status, scope_key, confidence, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')",
                (
                    "fB",
                    "zara",
                    "WORKS_AT",
                    "globex",
                    datetime(2025, 7, 1, tzinfo=timezone.utc).isoformat(),
                    None,
                    "active",
                    None,
                    1.0,
                ),
            )
            backend._conn.commit()
        # Query at March: only fact A should be found
        t_march = datetime(2025, 3, 1, tzinfo=timezone.utc)
        # Query only A's entities; B's entities (zara/globex) must not be
        # in the multi-hop fringe, otherwise B leaks in via BFS.
        rows_march = backend.retrieve(
            "alice acme", RetrievalConfig(point_in_time=t_march, top_k=10)
        )
        # Only fact-derived contexts; ignore any entity-type contexts that may
        # also be returned by the multi-hop expansion.
        fact_ids = {c.context_id for c in rows_march if c.context_type == "fact"}
        assert "fA" in fact_ids
        assert "fB" not in fact_ids
        # Query at December: only fact B
        # Same reasoning: query only B's entities for the December check.
        rows_dec = backend.retrieve("zara globex", RetrievalConfig(point_in_time=t_late, top_k=10))
        fact_ids_dec = {c.context_id for c in rows_dec if c.context_type == "fact"}
        assert "fB" in fact_ids_dec
        assert "fA" not in fact_ids_dec
