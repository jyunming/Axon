"""Unit tests for OpenReranker (rerank.py).

Covers: LLM score parsing (tolerant regex), clamping, parse-error fallback,
empty-list guard, concurrent scoring correctness, cross-encoder path, and
field preservation.  All LLM/model calls are mocked so tests run offline.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_cfg(provider: str = "llm") -> SimpleNamespace:
    return SimpleNamespace(
        rerank=True,
        reranker_provider=provider,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


def _make_llm_reranker(responses: dict[str, str] | None = None):
    """Return an OpenReranker wired to a mock LLM (no __init__ side effects)."""
    from axon.rerank import OpenReranker

    llm = MagicMock()

    def _complete(prompt: str, **kw: object) -> str:
        if responses:
            for key, resp in responses.items():
                if key in prompt:
                    return resp
        return "5"

    llm.complete.side_effect = _complete
    r = OpenReranker.__new__(OpenReranker)
    r.config = _make_cfg("llm")
    r.llm = llm
    r.model = None
    return r


# ---------------------------------------------------------------------------
# LLM reranker path
# ---------------------------------------------------------------------------


def test_llm_rerank_returns_sorted_by_score():
    docs = [
        {"text": "alpha doc", "id": "a"},
        {"text": "beta doc", "id": "b"},
        {"text": "gamma doc", "id": "c"},
    ]
    reranker = _make_llm_reranker({"alpha": "8", "beta": "5", "gamma": "9"})
    result = reranker._llm_rerank("query", docs)
    assert len(result) == 3
    scores = [d["rerank_score"] for d in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0]["id"] == "c"  # gamma scored 9
    assert result[1]["id"] == "a"  # alpha scored 8
    assert result[2]["id"] == "b"  # beta scored 5


def test_llm_score_parsing_tolerant():
    _SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")
    cases = [
        ("Score: 8", 8.0),
        ("Relevance: 7.5", 7.5),
        ("8/10", 8.0),
        ("6", 6.0),
        ("The score is 9.0 out of 10.", 9.0),
    ]
    for response, expected in cases:
        m = _SCORE_RE.search(response)
        assert m is not None, f"No match for: {response!r}"
        assert float(m.group(0)) == expected, f"Wrong value for: {response!r}"


def test_llm_score_parse_failure_returns_zero():
    reranker = _make_llm_reranker({"doc": "no numbers here!"})
    docs = [{"text": "doc text", "id": "x"}]
    result = reranker._llm_rerank("query", docs)
    assert result[0]["rerank_score"] == 0.0


def test_llm_score_clamped_to_valid_range():
    reranker = _make_llm_reranker({"high": "11", "low": "-1"})
    docs = [
        {"text": "high doc", "id": "h"},
        {"text": "low doc", "id": "l"},
    ]
    result = reranker._llm_rerank("query", docs)
    for d in result:
        assert 0.0 <= d["rerank_score"] <= 10.0


def test_llm_rerank_empty_list_returns_empty():
    reranker = _make_llm_reranker()
    result = reranker._llm_rerank("query", [])
    assert result == []
    reranker.llm.complete.assert_not_called()


def test_llm_rerank_preserves_all_fields():
    docs = [{"text": "doc", "id": "1", "source": "file.txt", "chunk_id": "abc"}]
    reranker = _make_llm_reranker({"doc": "7"})
    result = reranker._llm_rerank("query", docs)
    assert result[0]["source"] == "file.txt"
    assert result[0]["chunk_id"] == "abc"
    assert result[0]["id"] == "1"
    assert "rerank_score" in result[0]


def test_concurrent_scoring_no_race_condition():
    reranker = _make_llm_reranker()  # all docs get "5"
    docs = [{"text": f"doc {i}", "id": str(i)} for i in range(20)]
    result = reranker._llm_rerank("query", docs)
    assert len(result) == 20
    ids_returned = {d["id"] for d in result}
    ids_expected = {str(i) for i in range(20)}
    assert ids_returned == ids_expected


# ---------------------------------------------------------------------------
# Cross-encoder path
# ---------------------------------------------------------------------------


def test_cross_encoder_path_sorts_by_score():
    from axon.rerank import OpenReranker

    r = OpenReranker.__new__(OpenReranker)
    r.config = _make_cfg("cross-encoder")
    r.llm = None

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.3, 0.9, 0.1]
    r.model = mock_model

    docs = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    result = r.rerank("query", docs)
    scores = [d["rerank_score"] for d in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0]["rerank_score"] == 0.9


def test_rerank_returns_input_unchanged_when_disabled():
    from axon.rerank import OpenReranker

    r = OpenReranker.__new__(OpenReranker)
    r.config = SimpleNamespace(rerank=False, reranker_provider="llm")
    r.llm = None
    r.model = None

    docs = [{"text": "doc", "id": "1"}]
    result = r.rerank("query", docs)
    assert result is docs
