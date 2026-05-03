"""Tests for v0.3.2 item (18): structured citation metadata.

Covers the helper that builds the metadata payload (parsing + slim form)
and the API-route surface (include_citations toggle).
"""
from __future__ import annotations

import pytest

from axon.query_router import _build_citation_metadata, _slim_source

# ---------------------------------------------------------------------------
# _slim_source helpers
# ---------------------------------------------------------------------------


class TestSlimSource:
    def test_score_falls_back_to_score_when_no_rerank_score(self):
        row = {"id": "x", "text": "hello", "score": 0.7, "metadata": {"source": "a.md"}}
        out = _slim_source(0, row)
        assert out["score"] == pytest.approx(0.7)
        assert out["index"] == 0
        assert out["title"] == "a.md"
        assert out["source"] == "a.md"
        assert out["id"] == "x"

    def test_rerank_score_wins_when_present(self):
        row = {
            "id": "x",
            "text": "hello",
            "score": 0.4,
            "rerank_score": 0.95,
            "metadata": {"source": "a.md"},
        }
        out = _slim_source(0, row)
        assert out["score"] == pytest.approx(0.95)

    def test_text_truncated_with_ellipsis(self):
        row = {"id": "x", "text": "y" * 1000, "metadata": {"source": "a.md"}}
        out = _slim_source(0, row)
        assert out["text"].endswith("…")
        # 500 chars + ellipsis
        assert len(out["text"]) == 501

    def test_short_text_not_truncated(self):
        row = {"id": "x", "text": "hello", "metadata": {"source": "a.md"}}
        out = _slim_source(0, row)
        assert out["text"] == "hello"
        assert not out["text"].endswith("…")

    def test_web_row_marked(self):
        row = {
            "id": "w",
            "text": "external",
            "is_web": True,
            "metadata": {"url": "https://example.com", "title": "Ex"},
        }
        out = _slim_source(0, row)
        assert out["is_web"] is True
        assert out["url"] == "https://example.com"
        assert out["title"] == "Ex"

    def test_metadata_filtered_to_safe_keys(self):
        row = {
            "id": "x",
            "text": "code",
            "metadata": {
                "source": "main.py",
                "file_path": "/path/to/main.py",
                "page": 0,
                "symbol_name": "foo",
                "symbol_type": "function",
                "source_class": "code",
                "secret_key": "do not leak",
                "auth_token": "ditto",
            },
        }
        out = _slim_source(0, row)
        assert "secret_key" not in out["metadata"]
        assert "auth_token" not in out["metadata"]
        assert out["metadata"]["file_path"] == "/path/to/main.py"
        assert out["metadata"]["symbol_name"] == "foo"

    def test_score_none_when_not_a_number(self):
        row = {"id": "x", "text": "hello", "score": None, "metadata": {"source": "a"}}
        out = _slim_source(0, row)
        assert out["score"] is None


# ---------------------------------------------------------------------------
# _build_citation_metadata
# ---------------------------------------------------------------------------


def _row(idx: int, source: str = "doc.md", text: str = "") -> dict:
    return {"id": f"chunk-{idx}", "text": text or f"row {idx}", "metadata": {"source": source}}


class TestBuildCitationMetadata:
    def test_empty_inputs_return_empty(self):
        out = _build_citation_metadata("", [])
        assert out == {"sources": [], "citations": []}

    def test_non_string_response_returns_sources_and_empty_citations(self):
        """When the LLM mock returns a non-string (e.g. MagicMock), the
        helper must not crash; sources are still surfaced, citations is
        empty."""
        from unittest.mock import MagicMock

        rows = [_row(1, "a.md")]
        out = _build_citation_metadata(MagicMock(), rows)
        assert len(out["sources"]) == 1
        assert out["citations"] == []

    def test_no_markers_returns_sources_but_no_citations(self):
        rows = [_row(1, "a.md"), _row(2, "b.md")]
        out = _build_citation_metadata("plain answer with no markers", rows)
        assert len(out["sources"]) == 2
        assert out["citations"] == []

    def test_bracketed_digit_markers_parsed(self):
        rows = [_row(1, "a.md"), _row(2, "b.md")]
        response = "First fact [1]. Second fact [2]."
        out = _build_citation_metadata(response, rows)
        assert len(out["citations"]) == 2
        assert out["citations"][0]["marker"] == "[1]"
        assert out["citations"][0]["document_index"] == 0
        assert out["citations"][0]["document_title"] == "a.md"
        assert out["citations"][0]["document_id"] == "chunk-1"
        # Char offsets are correct
        assert (
            response[
                out["citations"][0]["start_in_response"] : out["citations"][0]["end_in_response"]
            ]
            == "[1]"
        )
        assert out["citations"][1]["document_index"] == 1

    def test_document_n_marker_parsed(self):
        rows = [_row(1, "a.md")]
        response = "See [Document 1] for details."
        out = _build_citation_metadata(response, rows)
        assert len(out["citations"]) == 1
        assert out["citations"][0]["marker"] == "[Document 1]"
        assert out["citations"][0]["document_index"] == 0

    def test_out_of_range_marker_dropped(self):
        rows = [_row(1, "a.md")]  # only 1 source available
        response = "Stale citation [3]"  # marker N=3 has no matching source
        out = _build_citation_metadata(response, rows)
        # No source for [3] → no citation entry
        assert out["citations"] == []

    def test_repeat_markers_each_get_an_entry(self):
        rows = [_row(1, "a.md")]
        response = "Foo [1] and bar [1] both come from doc 1."
        out = _build_citation_metadata(response, rows)
        assert len(out["citations"]) == 2
        # Both point at the same document
        assert {c["document_index"] for c in out["citations"]} == {0}
        # But char offsets differ
        assert out["citations"][0]["start_in_response"] != out["citations"][1]["start_in_response"]

    def test_zero_marker_is_dropped(self):
        rows = [_row(1, "a.md")]
        out = _build_citation_metadata("answer [0]", rows)
        assert out["citations"] == []

    def test_multiple_markers_in_order(self):
        rows = [_row(1, "a.md"), _row(2, "b.md"), _row(3, "c.md")]
        response = "[2] then [1] then [3]"
        out = _build_citation_metadata(response, rows)
        # Order matches occurrence in the response, not the source order
        assert [c["document_index"] for c in out["citations"]] == [1, 0, 2]


# ---------------------------------------------------------------------------
# API route surface (include_citations toggle on QueryRequest)
# ---------------------------------------------------------------------------


class TestQueryRequestIncludeCitations:
    def test_default_is_true(self):
        from axon.api_schemas import QueryRequest

        req = QueryRequest(query="hi")
        assert req.include_citations is True

    def test_explicit_false_honoured(self):
        from axon.api_schemas import QueryRequest

        req = QueryRequest(query="hi", include_citations=False)
        assert req.include_citations is False
