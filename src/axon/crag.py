"""
axon/crag.py — CRAG-Lite Retrieval Correction (Epic 2, Stories 2.1–2.3)

Provides a lightweight corrective decision layer that evaluates retrieval
quality before deciding whether to trust local results or escalate to a web
fallback.

Design
------
- Story 2.1 (confidence contract): ``RetrievalConfidence`` dataclass aggregates
  five heuristic signals into a single [0, 1] score with a deterministic verdict.
- Story 2.2 (correction policy): ``evaluate_correction_policy()`` decides
  whether to trust local retrieval, and whether to trigger web fallback — all
  without calling the LLM.
- Story 2.3 (diagnostics): ``CorrectionDecision`` carries the full audit trail
  (confidence, verdict, factors, fallback reason) for surfacing in API responses
  and structured logs.

Integration point
-----------------
``query_router._execute_retrieval()`` calls ``assess_confidence()`` and
``evaluate_correction_policy()`` after threshold filtering, replacing the
existing hard-wired "fallback only when filtered_results is empty" guard.
The original guard remains active when ``crag_lite=False`` (default).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Story 2.1 — Retrieval confidence contract
# ---------------------------------------------------------------------------

# Minimum result count considered "strong" — fewer results lower confidence.
_MIN_STRONG_RESULT_COUNT: int = 3
# Score spread threshold — a result set whose best-minus-worst spread is at
# least this wide has good differentiation between relevant/irrelevant docs.
_GOOD_SPREAD: float = 0.25
# Minimum unique-source count for diversity bonus.
_MIN_SOURCE_DIVERSITY: int = 2
# Fraction of raw candidates that should pass the threshold before the pass
# rate signal contributes meaningfully.
_GOOD_PASS_RATE: float = 0.25

# Weighted signal contributions — must sum to 1.0.
_WEIGHTS = {
    "result_count": 0.30,
    "top_score": 0.30,
    "score_spread": 0.15,
    "source_diversity": 0.15,
    "threshold_pass_rate": 0.10,
}

_VERDICT_HIGH: float = 0.70
_VERDICT_MEDIUM: float = 0.40


@dataclass
class RetrievalConfidence:
    """Confidence assessment for a single retrieval result set.

    ``score`` is a weighted average of five normalised signals in [0, 1].
    ``verdict`` is deterministic: ``"high"`` ≥ 0.70, ``"medium"`` ≥ 0.40,
    ``"low"`` < 0.40.
    ``factors`` exposes the per-signal contributions for diagnostics.
    ``fallback_recommended`` is True when the verdict is ``"low"``.
    """

    score: float
    verdict: Literal["high", "medium", "low"]
    factors: dict[str, float] = field(default_factory=dict)
    fallback_recommended: bool = False

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "fallback_recommended": self.fallback_recommended,
        }


def assess_confidence(
    filtered_results: list[dict],
    total_candidates: int,
    similarity_threshold: float,
) -> RetrievalConfidence:
    """Compute a retrieval-confidence assessment from the filtered result set.

    This function is *deterministic in heuristic mode* — the same inputs always
    produce the same output.  It never calls the LLM.

    Args:
        filtered_results:      Results that survived the similarity threshold.
        total_candidates:      Total dense+BM25 candidates before threshold
                               filtering (used to compute the pass-rate signal).
        similarity_threshold:  The configured threshold (used to normalise the
                               top-score signal relative to the threshold).

    Returns:
        A :class:`RetrievalConfidence` with score, verdict, and per-signal
        factor contributions.
    """
    n = len(filtered_results)

    # --- Signal 1: result count ---
    count_signal = min(1.0, n / _MIN_STRONG_RESULT_COUNT)

    # --- Signal 2: top score ---
    if n > 0:
        scores = [r.get("score", r.get("vector_score", 0.0)) for r in filtered_results]
        top_score = max(scores)
        # Normalise relative to a "good score" = threshold + 0.3, capped at 1.0.
        good_score = min(1.0, similarity_threshold + 0.30)
        top_score_signal = min(1.0, top_score / good_score) if good_score > 0 else 0.0
    else:
        scores = []
        top_score = 0.0
        top_score_signal = 0.0

    # --- Signal 3: score spread (differentiation quality) ---
    if n >= 2:
        spread = max(scores) - min(scores)
        spread_signal = min(1.0, spread / _GOOD_SPREAD)
    else:
        spread_signal = 0.0

    # --- Signal 4: source diversity ---
    unique_sources = len({r.get("metadata", {}).get("source", r["id"]) for r in filtered_results})
    diversity_signal = min(1.0, unique_sources / _MIN_SOURCE_DIVERSITY)

    # --- Signal 5: threshold pass rate ---
    if total_candidates > 0:
        pass_rate = n / total_candidates
        pass_rate_signal = min(1.0, pass_rate / _GOOD_PASS_RATE)
    else:
        pass_rate_signal = 0.0 if n == 0 else 1.0

    factors = {
        "result_count": round(count_signal, 4),
        "top_score": round(top_score_signal, 4),
        "score_spread": round(spread_signal, 4),
        "source_diversity": round(diversity_signal, 4),
        "threshold_pass_rate": round(pass_rate_signal, 4),
    }

    score = sum(_WEIGHTS[k] * v for k, v in factors.items())

    if score >= _VERDICT_HIGH:
        verdict: Literal["high", "medium", "low"] = "high"
    elif score >= _VERDICT_MEDIUM:
        verdict = "medium"
    else:
        verdict = "low"

    return RetrievalConfidence(
        score=round(score, 4),
        verdict=verdict,
        factors=factors,
        fallback_recommended=(verdict == "low"),
    )


# ---------------------------------------------------------------------------
# Story 2.2 — Correction policy engine
# ---------------------------------------------------------------------------


@dataclass
class CorrectionDecision:
    """Output of :func:`evaluate_correction_policy`.

    ``trust_local`` — use the local retrieval results.
    ``trigger_web_fallback`` — escalate to web search (requires
        ``truth_grounding=True`` and a configured API key).
    ``reason`` — human-readable explanation for the decision, suitable for
        structured logs and dry-run diagnostics.
    """

    trust_local: bool
    trigger_web_fallback: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "trust_local": self.trust_local,
            "trigger_web_fallback": self.trigger_web_fallback,
            "reason": self.reason,
        }


def evaluate_correction_policy(
    confidence: RetrievalConfidence,
    *,
    has_local_results: bool,
    truth_grounding_enabled: bool,
    crag_lite_threshold: float = _VERDICT_MEDIUM,
) -> CorrectionDecision:
    """Decide whether to trust local retrieval or trigger web fallback.

    Policy rules (heuristic, first version — no LLM call):

    1. **No local results at all** → trigger web fallback if ``truth_grounding``
       is enabled (preserves existing behaviour).
    2. **CRAG-Lite low confidence** (score < ``crag_lite_threshold``) → even with
       some local results, recommend web fallback when ``truth_grounding`` is on.
       This is the key CRAG-Lite extension over the legacy hard-wired guard.
    3. **Medium confidence** → trust local; log a warning about shallow KB.
    4. **High confidence** → trust local silently.

    Args:
        confidence:               Confidence assessment from :func:`assess_confidence`.
        has_local_results:        True if any local results survived thresholding.
        truth_grounding_enabled:  Whether the ``truth_grounding`` config flag is on.
        crag_lite_threshold:      Confidence score below which fallback is recommended
                                  even when some local results exist.  Defaults to the
                                  medium verdict boundary (0.40).

    Returns:
        A :class:`CorrectionDecision`.
    """
    # Case 1: no local results at all — identical to legacy behaviour.
    if not has_local_results:
        if truth_grounding_enabled:
            return CorrectionDecision(
                trust_local=False,
                trigger_web_fallback=True,
                reason="no_local_results",
            )
        return CorrectionDecision(
            trust_local=False,
            trigger_web_fallback=False,
            reason="no_local_results_no_web",
        )

    # Case 2: CRAG-Lite low-confidence — some results exist but quality is weak.
    if confidence.score < crag_lite_threshold:
        if truth_grounding_enabled:
            return CorrectionDecision(
                trust_local=False,
                trigger_web_fallback=True,
                reason=f"low_confidence({confidence.score:.2f}<{crag_lite_threshold:.2f})",
            )
        # Web not available — fall back to local with a warning logged by caller.
        return CorrectionDecision(
            trust_local=True,
            trigger_web_fallback=False,
            reason=f"low_confidence_no_web({confidence.score:.2f})",
        )

    # Case 3: medium confidence — trust local but surface warning in diagnostics.
    if confidence.verdict == "medium":
        return CorrectionDecision(
            trust_local=True,
            trigger_web_fallback=False,
            reason="medium_confidence_local_trusted",
        )

    # Case 4: high confidence — trust local silently.
    return CorrectionDecision(
        trust_local=True,
        trigger_web_fallback=False,
        reason="high_confidence",
    )
