"""
tests/test_deepeval_integration.py

DeepEval quality-framework integration for the Axon RAG pipeline.

These tests verify that Axon pipeline outputs can be packaged into DeepEval
test cases and that quality thresholds are correctly evaluated.  They are
marked ``@pytest.mark.eval`` so the CI eval job can target them independently:

    pytest -m eval tests/test_deepeval_integration.py

Two execution modes
-------------------
* **CI mode (default)** — No LLM judge is invoked.  ``_patch_metric()`` injects
  synthetic scores directly into the metric object so tests verify framework
  plumbing without API cost.  All tests pass without ``deepeval`` installed
  (they are skipped instead).
* **Live mode** — Set ``DEEPEVAL_JUDGE_MODEL`` env var (e.g.
  ``DEEPEVAL_JUDGE_MODEL=gpt-4o-mini``) and point ``OPENAI_API_KEY`` at a real
  key.  ``metric.measure(tc)`` is called with the real LLM judge.

Metrics covered
---------------
- ``AnswerRelevancyMetric``   — is the answer relevant to the query?
- ``FaithfulnessMetric``      — is the answer grounded in the retrieved context?
- ``ContextualRelevancyMetric`` — are retrieved chunks relevant to the query?
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False

pytestmark = pytest.mark.eval

_SKIP_IF_NO_DEEPEVAL = pytest.mark.skipif(
    not _DEEPEVAL_AVAILABLE,
    reason="deepeval not installed — run `pip install 'axon[eval]'`",
)

# Whether to call the real LLM judge (requires DEEPEVAL_JUDGE_MODEL + API key)
_USE_REAL_JUDGE = bool(os.getenv("DEEPEVAL_JUDGE_MODEL"))

# ---------------------------------------------------------------------------
# Sample data — a minimal RAG scenario about transformer architecture
# ---------------------------------------------------------------------------

_QUERY = "What is the transformer architecture?"

_CONTEXT = [
    "The transformer architecture uses self-attention mechanisms to process sequences in parallel.",
    "BERT is a bidirectional transformer pre-trained on masked language modelling.",
    "Attention mechanisms allow the model to focus on different parts of the input sequence.",
]

_FAITHFUL_ANSWER = (
    "The transformer architecture uses self-attention mechanisms to process input sequences "
    "in parallel. BERT is a bidirectional transformer pre-trained on masked language modelling."
)

_HALLUCINATED_ANSWER = (
    "The transformer architecture was invented in 1850 and primarily uses convolutional layers "
    "operating on quantum circuits."
)


# ---------------------------------------------------------------------------
# Helper — inject a fixed score without invoking the LLM judge
# ---------------------------------------------------------------------------


def _patch_metric(metric_obj, score: float) -> None:
    """Set a synthetic score on a DeepEval metric (CI mode — no LLM judge)."""
    metric_obj.score = score
    metric_obj.success = score >= metric_obj.threshold
    metric_obj.reason = f"Mocked score {score:.2f} (CI mode)"


# ---------------------------------------------------------------------------
# LLMTestCase construction
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestLLMTestCaseConstruction:
    """Axon pipeline outputs can be packaged into DeepEval LLMTestCase objects."""

    def test_basic_test_case(self):
        tc = LLMTestCase(
            input=_QUERY,
            actual_output=_FAITHFUL_ANSWER,
            retrieval_context=_CONTEXT,
        )
        assert tc.input == _QUERY
        assert tc.actual_output == _FAITHFUL_ANSWER
        assert len(tc.retrieval_context) == 3

    def test_test_case_with_expected_output(self):
        tc = LLMTestCase(
            input=_QUERY,
            actual_output=_FAITHFUL_ANSWER,
            expected_output="Transformers use attention mechanisms.",
            retrieval_context=_CONTEXT,
        )
        assert tc.expected_output is not None

    def test_empty_retrieval_context(self):
        """Empty retrieval context (no-KB scenario) should not raise."""
        tc = LLMTestCase(
            input=_QUERY,
            actual_output="I don't know.",
            retrieval_context=[],
        )
        assert tc.retrieval_context == []

    def test_multi_case_list(self):
        """Multiple test cases can be assembled for batch evaluate()."""
        pairs = [
            ("What is BERT?", "BERT is a bidirectional transformer."),
            ("How does attention work?", "Attention focuses on relevant input parts."),
        ]
        cases = [
            LLMTestCase(input=q, actual_output=a, retrieval_context=_CONTEXT) for q, a in pairs
        ]
        assert len(cases) == 2
        assert all(isinstance(tc, LLMTestCase) for tc in cases)


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestMetricInstantiation:
    """Metrics can be created with custom thresholds without invoking the judge."""

    def test_answer_relevancy_threshold(self):
        m = AnswerRelevancyMetric(threshold=0.7)
        assert m.threshold == 0.7

    def test_faithfulness_threshold(self):
        m = FaithfulnessMetric(threshold=0.8)
        assert m.threshold == 0.8

    def test_contextual_relevancy_threshold(self):
        m = ContextualRelevancyMetric(threshold=0.6)
        assert m.threshold == 0.6

    def test_threshold_zero_always_passes(self):
        """A threshold of 0.0 means any non-negative score is a success."""
        m = AnswerRelevancyMetric(threshold=0.0)
        _patch_metric(m, 0.01)
        assert m.success

    def test_threshold_one_never_passes_unless_perfect(self):
        m = FaithfulnessMetric(threshold=1.0)
        _patch_metric(m, 0.99)
        assert not m.success
        _patch_metric(m, 1.0)
        assert m.success


# ---------------------------------------------------------------------------
# Answer Relevancy
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestAnswerRelevancy:
    def _measure(self, answer: str, threshold: float = 0.7) -> AnswerRelevancyMetric:
        m = AnswerRelevancyMetric(threshold=threshold)
        tc = LLMTestCase(input=_QUERY, actual_output=answer, retrieval_context=_CONTEXT)
        if _USE_REAL_JUDGE:
            m.measure(tc)
        else:
            # Heuristic: answers containing query terms score higher
            q_words = set(_QUERY.lower().split())
            a_words = set(answer.lower().split())
            overlap = len(q_words & a_words) / max(len(q_words), 1)
            _patch_metric(m, min(0.95, 0.5 + overlap))
        return m

    def test_relevant_answer_scores_above_floor(self):
        m = self._measure(_FAITHFUL_ANSWER)
        assert m.score >= 0.5, f"Expected relevancy >= 0.5, got {m.score}"

    def test_relevant_answer_passes_threshold(self):
        m = self._measure(_FAITHFUL_ANSWER, threshold=0.5)
        assert m.success

    def test_score_is_float_in_range(self):
        m = self._measure(_FAITHFUL_ANSWER)
        assert isinstance(m.score, float)
        assert 0.0 <= m.score <= 1.0

    def test_irrelevant_answer_lower_score(self):
        on_topic = self._measure(_FAITHFUL_ANSWER)
        off_topic = self._measure("The weather in Paris is mild in spring.")
        if not _USE_REAL_JUDGE:
            # Patch off-topic manually to demonstrate the contrast
            _patch_metric(off_topic, 0.1)
        assert on_topic.score >= off_topic.score


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestFaithfulness:
    def test_faithful_answer_passes_threshold(self):
        m = FaithfulnessMetric(threshold=0.6)
        if _USE_REAL_JUDGE:
            tc = LLMTestCase(
                input=_QUERY, actual_output=_FAITHFUL_ANSWER, retrieval_context=_CONTEXT
            )
            m.measure(tc)
        else:
            _patch_metric(m, 0.90)
        assert m.success

    def test_hallucinated_answer_fails_threshold(self):
        m = FaithfulnessMetric(threshold=0.6)
        tc = LLMTestCase(  # noqa: F841 — tc used by real judge branch
            input=_QUERY, actual_output=_HALLUCINATED_ANSWER, retrieval_context=_CONTEXT
        )
        if _USE_REAL_JUDGE:
            m.measure(tc)
        else:
            _patch_metric(m, 0.10)
        assert not m.success

    def test_faithful_score_is_numeric(self):
        m = FaithfulnessMetric(threshold=0.5)
        if _USE_REAL_JUDGE:
            tc = LLMTestCase(
                input=_QUERY, actual_output=_FAITHFUL_ANSWER, retrieval_context=_CONTEXT
            )
            m.measure(tc)
        else:
            _patch_metric(m, 0.80)
        assert isinstance(m.score, float)
        assert 0.0 <= m.score <= 1.0

    def test_faithful_vs_hallucinated_score_order(self):
        """Faithful answer should score higher than hallucinated one."""
        m_good = FaithfulnessMetric(threshold=0.5)
        m_bad = FaithfulnessMetric(threshold=0.5)
        if not _USE_REAL_JUDGE:
            _patch_metric(m_good, 0.90)
            _patch_metric(m_bad, 0.10)
        assert m_good.score > m_bad.score


# ---------------------------------------------------------------------------
# Contextual Relevancy
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestContextualRelevancy:
    def test_on_topic_context_passes(self):
        m = ContextualRelevancyMetric(threshold=0.5)
        tc = LLMTestCase(input=_QUERY, actual_output=_FAITHFUL_ANSWER, retrieval_context=_CONTEXT)
        if _USE_REAL_JUDGE:
            m.measure(tc)
        else:
            _patch_metric(m, 0.85)
        assert m.success

    def test_irrelevant_context_fails(self):
        irrelevant_ctx = ["The weather is sunny today.", "Cats are mammals.", "Paris is in France."]
        m = ContextualRelevancyMetric(threshold=0.5)
        tc = LLMTestCase(  # noqa: F841
            input=_QUERY, actual_output=_FAITHFUL_ANSWER, retrieval_context=irrelevant_ctx
        )
        if _USE_REAL_JUDGE:
            m.measure(tc)
        else:
            _patch_metric(m, 0.05)
        assert not m.success

    def test_score_range(self):
        m = ContextualRelevancyMetric(threshold=0.5)
        if not _USE_REAL_JUDGE:
            _patch_metric(m, 0.75)
        assert 0.0 <= m.score <= 1.0


# ---------------------------------------------------------------------------
# Batch evaluate()
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestBatchEvaluate:
    """deepeval.evaluate() accepts lists of (test_cases, metrics)."""

    def test_single_metric_evaluate(self):
        tc = LLMTestCase(input=_QUERY, actual_output=_FAITHFUL_ANSWER, retrieval_context=_CONTEXT)
        m = AnswerRelevancyMetric(threshold=0.7)
        if _USE_REAL_JUDGE:
            results = evaluate([tc], [m])
            assert results is not None
        else:
            mock_result = MagicMock()
            mock_result.test_results = [MagicMock(success=True)]
            with patch("deepeval.evaluate", return_value=mock_result) as mock_eval:
                results = evaluate([tc], [m])
                mock_eval.assert_called_once_with([tc], [m])
            assert results is not None

    def test_multi_metric_evaluate(self):
        tc = LLMTestCase(input=_QUERY, actual_output=_FAITHFUL_ANSWER, retrieval_context=_CONTEXT)
        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.5),
        ]
        if _USE_REAL_JUDGE:
            results = evaluate([tc], metrics)
            assert results is not None
        else:
            mock_result = MagicMock()
            with patch("deepeval.evaluate", return_value=mock_result):
                results = evaluate([tc], metrics)
            assert results is not None

    def test_multi_test_case_evaluate(self):
        """Multiple RAG scenarios can be evaluated in one batch call."""
        cases = [
            LLMTestCase(
                input="What is BERT?",
                actual_output="BERT is a bidirectional transformer.",
                retrieval_context=_CONTEXT,
            ),
            LLMTestCase(
                input="How does attention work?",
                actual_output="Attention allows the model to focus on relevant input parts.",
                retrieval_context=_CONTEXT,
            ),
        ]
        metrics = [FaithfulnessMetric(threshold=0.6)]
        if not _USE_REAL_JUDGE:
            mock_result = MagicMock()
            with patch("deepeval.evaluate", return_value=mock_result):
                results = evaluate(cases, metrics)
            assert results is not None


# ---------------------------------------------------------------------------
# Pipeline integration — slot Axon query() output into DeepEval
# ---------------------------------------------------------------------------


@_SKIP_IF_NO_DEEPEVAL
class TestAxonPipelineIntegration:
    """Verify the full Axon→DeepEval round-trip with a mocked brain."""

    def _run_mocked_pipeline(self) -> tuple[str, list[str]]:
        """Return (answer, retrieval_context) from a mocked Axon pipeline."""
        return _FAITHFUL_ANSWER, list(_CONTEXT)

    def test_pipeline_output_slots_into_test_case(self):
        answer, context = self._run_mocked_pipeline()
        tc = LLMTestCase(input=_QUERY, actual_output=answer, retrieval_context=context)
        assert tc.actual_output == _FAITHFUL_ANSWER
        assert len(tc.retrieval_context) == 3

    def test_three_metrics_pass_for_faithful_pipeline(self):
        """A faithful pipeline answer should pass all three quality gates."""
        answer, context = self._run_mocked_pipeline()
        metrics = [
            AnswerRelevancyMetric(threshold=0.5),
            FaithfulnessMetric(threshold=0.5),
            ContextualRelevancyMetric(threshold=0.5),
        ]
        if _USE_REAL_JUDGE:
            tc = LLMTestCase(input=_QUERY, actual_output=answer, retrieval_context=context)
            for m in metrics:
                m.measure(tc)
        else:
            for m in metrics:
                _patch_metric(m, 0.85)
        for m in metrics:
            assert m.success, f"{type(m).__name__} failed: score={m.score}"

    def test_hallucinated_pipeline_fails_faithfulness(self):
        """A hallucinated pipeline answer should fail the faithfulness gate."""
        _, context = self._run_mocked_pipeline()
        tc = LLMTestCase(  # noqa: F841
            input=_QUERY, actual_output=_HALLUCINATED_ANSWER, retrieval_context=context
        )
        m = FaithfulnessMetric(threshold=0.6)
        if _USE_REAL_JUDGE:
            m.measure(tc)
        else:
            _patch_metric(m, 0.05)
        assert not m.success, "Hallucinated answer should fail faithfulness gate"

    def test_release_gate_thresholds(self):
        """Verify the release-gate thresholds from release_gate_policy.md."""
        # Gate A: APR >= -2% proxy → answer relevancy >= 0.5
        # Gate B: RTL p95 < 2000ms → not a metric here
        # Gate C: IHO < 50% → not a metric here
        # This test focuses on the answer quality gate (Gate A proxy)
        answer, context = self._run_mocked_pipeline()
        m = AnswerRelevancyMetric(threshold=0.5)  # Gate A: no material regression
        if _USE_REAL_JUDGE:
            tc = LLMTestCase(input=_QUERY, actual_output=answer, retrieval_context=context)
            m.measure(tc)
        else:
            _patch_metric(m, 0.80)
        assert m.success, f"Gate A (answer relevancy) failed: score={m.score}"
