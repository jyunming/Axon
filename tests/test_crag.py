"""
tests/test_crag.py — Unit tests for axon.crag (Epic 2, Stories 2.1–2.3)

Coverage targets:
- assess_confidence(): all five signals, each verdict (high / medium / low)
- evaluate_correction_policy(): all four decision branches
- RetrievalConfidence.to_dict() / CorrectionDecision.to_dict()
- Edge cases: zero candidates, single result, empty result set
"""
from __future__ import annotations

from axon.crag import (
    _VERDICT_HIGH,
    CorrectionDecision,
    RetrievalConfidence,
    assess_confidence,
    evaluate_correction_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(score: float, source: str = "doc.txt", chunk_id: str = "c1") -> dict:
    return {
        "id": chunk_id,
        "score": score,
        "metadata": {"source": source},
    }


# ---------------------------------------------------------------------------
# Story 2.1 — assess_confidence
# ---------------------------------------------------------------------------


class TestAssessConfidenceSignals:
    def test_empty_results_gives_low_verdict(self):
        conf = assess_confidence([], total_candidates=0, similarity_threshold=0.5)
        assert conf.verdict == "low"
        assert conf.fallback_recommended is True
        assert conf.score == 0.0

    def test_single_result_no_spread(self):
        results = [_make_result(0.9)]
        conf = assess_confidence(results, total_candidates=1, similarity_threshold=0.5)
        # spread signal = 0 (need ≥ 2 results); result_count partial; diversity partial
        assert conf.factors["score_spread"] == 0.0
        assert conf.factors["result_count"] < 1.0

    def test_high_confidence_three_diverse_results(self):
        results = [
            _make_result(0.95, source="a.txt", chunk_id="c1"),
            _make_result(0.80, source="b.txt", chunk_id="c2"),
            _make_result(0.60, source="c.txt", chunk_id="c3"),
        ]
        conf = assess_confidence(results, total_candidates=4, similarity_threshold=0.5)
        assert conf.verdict == "high"
        assert conf.score >= _VERDICT_HIGH
        assert conf.fallback_recommended is False

    def test_low_confidence_one_result_low_score(self):
        results = [_make_result(0.35)]
        conf = assess_confidence(results, total_candidates=10, similarity_threshold=0.5)
        assert conf.verdict == "low"
        assert conf.fallback_recommended is True

    def test_medium_confidence_range(self):
        # Two moderate-score results, same source — diversity penalty
        results = [
            _make_result(0.60, source="same.txt", chunk_id="c1"),
            _make_result(0.55, source="same.txt", chunk_id="c2"),
        ]
        conf = assess_confidence(results, total_candidates=8, similarity_threshold=0.5)
        # Diversity is capped at 1 src → signal = 0.5; should land somewhere medium or low
        assert conf.verdict in ("medium", "low")

    def test_result_count_signal_capped_at_one(self):
        results = [_make_result(0.9, chunk_id=f"c{i}") for i in range(10)]
        conf = assess_confidence(results, total_candidates=10, similarity_threshold=0.5)
        assert conf.factors["result_count"] == 1.0

    def test_pass_rate_signal_capped_at_one(self):
        results = [_make_result(0.9)]
        # All candidates pass threshold → pass_rate = 1.0/1 → signal = 1/(0.25)=4 → capped at 1
        conf = assess_confidence(results, total_candidates=1, similarity_threshold=0.5)
        assert conf.factors["threshold_pass_rate"] == 1.0

    def test_zero_candidates_no_results_pass_rate_zero(self):
        conf = assess_confidence([], total_candidates=0, similarity_threshold=0.5)
        assert conf.factors["threshold_pass_rate"] == 0.0

    def test_zero_candidates_with_results_pass_rate_one(self):
        # Edge case: if caller passes total_candidates=0 but there are filtered results
        results = [_make_result(0.9)]
        conf = assess_confidence(results, total_candidates=0, similarity_threshold=0.5)
        assert conf.factors["threshold_pass_rate"] == 1.0

    def test_score_uses_vector_score_fallback(self):
        result = {"id": "c1", "vector_score": 0.8, "metadata": {"source": "f.txt"}}
        conf = assess_confidence([result], total_candidates=1, similarity_threshold=0.5)
        # top_score_signal should be > 0 (uses vector_score when score absent)
        assert conf.factors["top_score"] > 0

    def test_score_field_preferred_over_vector_score(self):
        result = {
            "id": "c1",
            "score": 0.95,
            "vector_score": 0.1,
            "metadata": {"source": "f.txt"},
        }
        conf = assess_confidence([result], total_candidates=1, similarity_threshold=0.5)
        # Should use 0.95, not 0.1
        assert conf.factors["top_score"] > 0.5

    def test_weights_sum_to_one(self):
        from axon.crag import _WEIGHTS

        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9

    def test_factors_keys_match_weights(self):
        from axon.crag import _WEIGHTS

        results = [_make_result(0.8)]
        conf = assess_confidence(results, total_candidates=1, similarity_threshold=0.5)
        assert set(conf.factors.keys()) == set(_WEIGHTS.keys())

    def test_to_dict_round_trip(self):
        results = [_make_result(0.9, source="a.txt")]
        conf = assess_confidence(results, total_candidates=2, similarity_threshold=0.5)
        d = conf.to_dict()
        assert isinstance(d["score"], float)
        assert d["verdict"] in ("high", "medium", "low")
        assert isinstance(d["factors"], dict)
        assert isinstance(d["fallback_recommended"], bool)

    def test_source_diversity_uses_id_fallback(self):
        # Results with no 'metadata.source' — should fall back to id
        results = [
            {"id": "c1", "score": 0.9, "metadata": {}},
            {"id": "c2", "score": 0.8, "metadata": {}},
        ]
        conf = assess_confidence(results, total_candidates=2, similarity_threshold=0.5)
        # Two unique ids → diversity_signal = 1.0
        assert conf.factors["source_diversity"] == 1.0


# ---------------------------------------------------------------------------
# Story 2.2 — evaluate_correction_policy
# ---------------------------------------------------------------------------


class TestEvaluateCorrectionPolicy:
    def _low_conf(self) -> RetrievalConfidence:
        return RetrievalConfidence(score=0.20, verdict="low", factors={}, fallback_recommended=True)

    def _medium_conf(self) -> RetrievalConfidence:
        return RetrievalConfidence(
            score=0.55, verdict="medium", factors={}, fallback_recommended=False
        )

    def _high_conf(self) -> RetrievalConfidence:
        return RetrievalConfidence(
            score=0.80, verdict="high", factors={}, fallback_recommended=False
        )

    # Case 1 — no local results
    def test_no_local_results_truth_grounding_on(self):
        decision = evaluate_correction_policy(
            self._high_conf(),
            has_local_results=False,
            truth_grounding_enabled=True,
        )
        assert decision.trust_local is False
        assert decision.trigger_web_fallback is True
        assert decision.reason == "no_local_results"

    def test_no_local_results_truth_grounding_off(self):
        decision = evaluate_correction_policy(
            self._high_conf(),
            has_local_results=False,
            truth_grounding_enabled=False,
        )
        assert decision.trust_local is False
        assert decision.trigger_web_fallback is False
        assert decision.reason == "no_local_results_no_web"

    # Case 2 — low confidence
    def test_low_confidence_truth_grounding_on_triggers_fallback(self):
        decision = evaluate_correction_policy(
            self._low_conf(),
            has_local_results=True,
            truth_grounding_enabled=True,
        )
        assert decision.trust_local is False
        assert decision.trigger_web_fallback is True
        assert "low_confidence(" in decision.reason

    def test_low_confidence_truth_grounding_off_trusts_local(self):
        decision = evaluate_correction_policy(
            self._low_conf(),
            has_local_results=True,
            truth_grounding_enabled=False,
        )
        assert decision.trust_local is True
        assert decision.trigger_web_fallback is False
        assert "low_confidence_no_web" in decision.reason

    def test_low_confidence_reason_contains_score(self):
        conf = RetrievalConfidence(score=0.15, verdict="low", factors={}, fallback_recommended=True)
        decision = evaluate_correction_policy(
            conf, has_local_results=True, truth_grounding_enabled=True
        )
        assert "0.15" in decision.reason

    # Case 3 — medium confidence
    def test_medium_confidence_trusts_local(self):
        decision = evaluate_correction_policy(
            self._medium_conf(),
            has_local_results=True,
            truth_grounding_enabled=True,
        )
        assert decision.trust_local is True
        assert decision.trigger_web_fallback is False
        assert decision.reason == "medium_confidence_local_trusted"

    def test_medium_confidence_trusts_local_web_off(self):
        decision = evaluate_correction_policy(
            self._medium_conf(),
            has_local_results=True,
            truth_grounding_enabled=False,
        )
        assert decision.trust_local is True
        assert decision.trigger_web_fallback is False

    # Case 4 — high confidence
    def test_high_confidence_trusts_local_silently(self):
        decision = evaluate_correction_policy(
            self._high_conf(),
            has_local_results=True,
            truth_grounding_enabled=True,
        )
        assert decision.trust_local is True
        assert decision.trigger_web_fallback is False
        assert decision.reason == "high_confidence"

    def test_custom_threshold_raises_boundary(self):
        # Score 0.35 with threshold 0.30 → above threshold → medium case
        conf = RetrievalConfidence(
            score=0.35, verdict="medium", factors={}, fallback_recommended=False
        )
        decision = evaluate_correction_policy(
            conf,
            has_local_results=True,
            truth_grounding_enabled=True,
            crag_lite_threshold=0.30,
        )
        assert decision.trust_local is True

    def test_custom_threshold_below_triggers_low_path(self):
        # Score 0.25, threshold 0.30 → below threshold → low-confidence fallback
        conf = RetrievalConfidence(
            score=0.25, verdict="medium", factors={}, fallback_recommended=False
        )
        decision = evaluate_correction_policy(
            conf,
            has_local_results=True,
            truth_grounding_enabled=True,
            crag_lite_threshold=0.30,
        )
        assert decision.trigger_web_fallback is True

    # to_dict
    def test_correction_decision_to_dict(self):
        dec = CorrectionDecision(
            trust_local=True, trigger_web_fallback=False, reason="high_confidence"
        )
        d = dec.to_dict()
        assert d == {
            "trust_local": True,
            "trigger_web_fallback": False,
            "reason": "high_confidence",
        }


# ---------------------------------------------------------------------------
# Story 2.3 — diagnostic field coverage via assess_confidence integration
# ---------------------------------------------------------------------------


class TestDiagnosticFields:
    """Verify the fields that are plumbed into CodeRetrievalDiagnostics."""

    def test_retrieval_confidence_exposes_all_diagnostic_fields(self):
        results = [
            _make_result(0.9, source="a.txt", chunk_id="c1"),
            _make_result(0.7, source="b.txt", chunk_id="c2"),
        ]
        conf = assess_confidence(results, total_candidates=5, similarity_threshold=0.5)
        d = conf.to_dict()
        # All fields that query_router plumbs into diagnostics
        assert "score" in d
        assert "verdict" in d
        assert "factors" in d
        assert "fallback_recommended" in d

    def test_correction_decision_exposes_fallback_reason(self):
        conf = RetrievalConfidence(
            score=0.18, verdict="low", factors={"result_count": 0.0}, fallback_recommended=True
        )
        decision = evaluate_correction_policy(
            conf, has_local_results=True, truth_grounding_enabled=True
        )
        assert decision.reason  # non-empty — surfaced in structured log
        assert decision.trigger_web_fallback is True
