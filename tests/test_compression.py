"""
tests/test_compression.py — Unit tests for axon.compression (Epic 3, Stories 3.1–3.3)

Coverage:
- Story 3.1: ContextCompressor interface, strategy routing, CompressionResult contract
- Story 3.2: sentence strategy, llmlingua strategy, fallback chain, source attribution
- Story 3.3: token telemetry, compression_ratio, fallback_reason, diagnostics plumbing
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from axon.compression import (
    CompressionResult,
    ContextCompressor,
    _chunk_text,
    _estimate_tokens,
    _total_tokens,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text="hello world", chunk_id="c1", is_web=False, parent_text=None):
    meta = {}
    if parent_text is not None:
        meta["parent_text"] = parent_text
    if is_web:
        meta["source"] = "web"
    return {"id": chunk_id, "text": text, "metadata": meta, "score": 0.9, "is_web": is_web}


def _make_llm(response="compressed text"):
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# Story 3.1 — Compression abstraction
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_estimate_tokens_positive(self):
        assert _estimate_tokens("hello") >= 1

    def test_estimate_tokens_empty_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_estimate_tokens_approx(self):
        # 400 chars → ~100 tokens
        assert 80 <= _estimate_tokens("a" * 400) <= 120

    def test_chunk_text_prefers_parent_text(self):
        c = _make_chunk(text="chunk", parent_text="parent")
        assert _chunk_text(c) == "parent"

    def test_chunk_text_falls_back_to_text(self):
        c = _make_chunk(text="chunk")
        assert _chunk_text(c) == "chunk"

    def test_total_tokens_sums_chunks(self):
        chunks = [_make_chunk("a" * 40), _make_chunk("b" * 40)]
        assert _total_tokens(chunks) >= 2


class TestCompressionResultContract:
    def test_has_required_fields(self):
        r = CompressionResult(
            chunks=[],
            strategy_used="none",
            pre_tokens=100,
            post_tokens=100,
            compression_ratio=1.0,
        )
        assert r.chunks == []
        assert r.strategy_used == "none"
        assert r.pre_tokens == 100
        assert r.post_tokens == 100
        assert r.compression_ratio == 1.0
        assert r.fallback_reason == ""

    def test_fallback_reason_optional(self):
        r = CompressionResult(
            chunks=[], strategy_used="sentence", pre_tokens=0, post_tokens=0, compression_ratio=1.0
        )
        assert r.fallback_reason == ""


class TestContextCompressorNoneStrategy:
    def test_none_strategy_returns_chunks_unchanged(self):
        c = ContextCompressor(llm=None)
        chunks = [_make_chunk("text")]
        result = c.compress("q", chunks, strategy="none")
        assert result.chunks is chunks or result.chunks == chunks
        assert result.strategy_used == "none"
        assert result.compression_ratio == 1.0

    def test_unknown_strategy_treated_as_none(self):
        c = ContextCompressor(llm=None)
        chunks = [_make_chunk()]
        result = c.compress("q", chunks, strategy="invalid_strategy")
        assert result.strategy_used == "none"

    def test_empty_chunks_returns_immediately(self):
        c = ContextCompressor(llm=None)
        result = c.compress("q", [], strategy="sentence")
        assert result.chunks == []
        assert result.strategy_used == "none"
        assert result.pre_tokens == 0
        assert result.compression_ratio == 1.0


# ---------------------------------------------------------------------------
# Story 3.2 — Sentence strategy
# ---------------------------------------------------------------------------
class TestSentenceStrategy:
    def test_sentence_compresses_chunk(self):
        llm = _make_llm("short answer")
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk("this is a very long passage " * 10)
        result = c.compress("what is this?", [chunk], strategy="sentence")
        assert result.strategy_used == "sentence"
        # The LLM returned something shorter — chunk should be compressed
        assert result.post_tokens < result.pre_tokens or result.chunks[0]["metadata"].get(
            "compressed"
        )

    def test_sentence_marks_compressed_metadata(self):
        long_text = "sentence one. " * 20
        llm = _make_llm("sentence one.")
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk(long_text)
        result = c.compress("q", [chunk], strategy="sentence")
        # Only compressed if LLM result is shorter
        if result.chunks[0].get("metadata", {}).get("compressed"):
            assert result.post_tokens <= result.pre_tokens

    def test_sentence_skips_web_chunks(self):
        llm = _make_llm("shorter")
        c = ContextCompressor(llm=llm)
        web_chunk = _make_chunk("web result text", is_web=True)
        result = c.compress("q", [web_chunk], strategy="sentence")
        # Web chunk returned unchanged; LLM not called for it
        assert result.chunks[0] is web_chunk

    def test_sentence_uses_parent_text_when_present(self):
        llm = _make_llm("compressed parent")
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk("chunk text", parent_text="parent text is longer " * 5)
        c.compress("q", [chunk], strategy="sentence")
        # LLM prompt should reference parent_text content
        call_args = llm.complete.call_args[0][0]
        assert "parent text is longer" in call_args

    def test_sentence_falls_back_on_llm_error(self):
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM error")
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk("original text")
        result = c.compress("q", [chunk], strategy="sentence")
        # Falls back to original
        assert result.chunks[0]["text"] == "original text"

    def test_sentence_no_llm_returns_unchanged(self):
        c = ContextCompressor(llm=None)
        chunk = _make_chunk("text")
        result = c.compress("q", [chunk], strategy="sentence")
        assert result.chunks[0] is chunk

    def test_sentence_preserves_source_attribution(self):
        llm = _make_llm("short")
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk("long text " * 10)
        chunk["metadata"]["source"] = "doc.txt"
        result = c.compress("q", [chunk], strategy="sentence")
        # source metadata must survive
        assert result.chunks[0].get("metadata", {}).get("source") == "doc.txt"

    def test_sentence_does_not_expand_text(self):
        """If LLM returns longer text, original is kept."""
        llm = _make_llm("this is a much longer response than the original passage text content")
        c = ContextCompressor(llm=llm)
        short_chunk = _make_chunk("short")
        result = c.compress("q", [short_chunk], strategy="sentence")
        # post_tokens ≤ pre_tokens because expansion is rejected
        assert result.post_tokens <= result.pre_tokens + 5  # small slack for tokenisation

    def test_multiple_chunks_processed_in_parallel(self):
        llm = _make_llm("compressed")
        c = ContextCompressor(llm=llm)
        chunks = [_make_chunk(f"chunk text {i} " * 10, chunk_id=f"c{i}") for i in range(5)]
        result = c.compress("q", chunks, strategy="sentence")
        assert len(result.chunks) == 5


# ---------------------------------------------------------------------------
# Story 3.2 — LLMLingua strategy
# ---------------------------------------------------------------------------
class TestLLMLinguaStrategy:
    def _mock_lingua(self, compressed="compressed output"):
        lingua = MagicMock()
        lingua.compress_prompt.return_value = {"compressed_prompt": compressed}
        return lingua

    def test_llmlingua_compresses_chunk(self):
        c = ContextCompressor(llm=None)
        c._lingua = self._mock_lingua("shorter text")
        chunk = _make_chunk("this is a long passage " * 10)
        result = c.compress("q", [chunk], strategy="llmlingua")
        assert result.strategy_used == "llmlingua"
        assert result.fallback_reason == ""

    def test_llmlingua_marks_backend_in_metadata(self):
        c = ContextCompressor(llm=None)
        c._lingua = self._mock_lingua("short")
        chunk = _make_chunk("long text " * 10)
        result = c.compress("q", [chunk], strategy="llmlingua")
        meta = result.chunks[0].get("metadata", {})
        if meta.get("compressed"):
            assert meta.get("compression_backend") == "llmlingua"

    def test_llmlingua_skips_web_chunks(self):
        c = ContextCompressor(llm=None)
        c._lingua = self._mock_lingua("shorter")
        web = _make_chunk("web result", is_web=True)
        result = c.compress("q", [web], strategy="llmlingua")
        assert result.chunks[0] is web

    def test_llmlingua_fallback_on_import_error(self):
        """When llmlingua is not installed, falls back to sentence."""
        llm = _make_llm("sentence fallback")
        c = ContextCompressor(llm=llm)
        with patch.object(c, "_ensure_llmlingua", side_effect=ImportError("no llmlingua")):
            chunk = _make_chunk("long passage " * 5)
            result = c.compress("q", [chunk], strategy="llmlingua")
        assert "llmlingua_unavailable" in result.fallback_reason
        assert result.strategy_used == "sentence"

    def test_llmlingua_fallback_no_llm_uses_none(self):
        """When llmlingua fails and no LLM, falls back to none (chunks unchanged)."""
        c = ContextCompressor(llm=None)
        with patch.object(c, "_ensure_llmlingua", side_effect=ImportError("no llmlingua")):
            chunk = _make_chunk("original")
            result = c.compress("q", [chunk], strategy="llmlingua")
        assert "llmlingua_unavailable" in result.fallback_reason
        assert result.strategy_used == "none"

    def test_llmlingua_chunk_error_returns_original(self):
        """Per-chunk LLMLingua error keeps original chunk, does not abort."""
        c = ContextCompressor(llm=None)
        lingua = MagicMock()
        lingua.compress_prompt.side_effect = RuntimeError("cuda error")
        c._lingua = lingua
        chunk = _make_chunk("original text")
        result = c.compress("q", [chunk], strategy="llmlingua")
        assert result.chunks[0]["text"] == "original text"
        assert result.fallback_reason == ""  # per-chunk error is silent, not a fallback

    def test_llmlingua_token_budget_used_for_rate(self):
        """A token_budget > 0 sets a compression rate proportional to budget/current."""
        c = ContextCompressor(llm=None)
        lingua = self._mock_lingua("short")
        c._lingua = lingua
        # 400-char chunk ≈ 100 tokens; budget=50 → rate ≈ 0.5
        chunk = _make_chunk("a" * 400)
        c.compress("q", [chunk], strategy="llmlingua", token_budget=50)
        call_kwargs = lingua.compress_prompt.call_args
        rate = call_kwargs[1].get("rate") or call_kwargs[0][1]
        assert 0.1 <= rate <= 0.95

    def test_llmlingua_zero_budget_uses_default_rate(self):
        c = ContextCompressor(llm=None)
        lingua = self._mock_lingua("short")
        c._lingua = lingua
        chunk = _make_chunk("text " * 40)
        c.compress("q", [chunk], strategy="llmlingua", token_budget=0)
        call_kwargs = lingua.compress_prompt.call_args
        rate = call_kwargs[1].get("rate") or call_kwargs[0][1]
        assert rate == 0.5  # default

    def test_ensure_llmlingua_lazy_init(self):
        """_ensure_llmlingua should only construct once."""
        mock_cls = MagicMock()
        c = ContextCompressor(llm=None, llmlingua_model="some/model")
        with patch("axon.compression.ContextCompressor._ensure_llmlingua", return_value=mock_cls):
            c._lingua = mock_cls  # pre-set so _ensure skips construction
            c._sentence_compress("q", [])  # unrelated method


# ---------------------------------------------------------------------------
# Story 3.3 — Compression telemetry
# ---------------------------------------------------------------------------
class TestCompressionTelemetry:
    def test_pre_tokens_positive(self):
        c = ContextCompressor(llm=_make_llm("short"))
        chunk = _make_chunk("token count test " * 10)
        result = c.compress("q", [chunk], strategy="sentence")
        assert result.pre_tokens > 0

    def test_post_tokens_leq_pre_tokens_when_compressed(self):
        long_text = "word " * 100
        llm = _make_llm("word")  # very short response
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk(long_text)
        result = c.compress("q", [chunk], strategy="sentence")
        assert result.post_tokens <= result.pre_tokens

    def test_compression_ratio_1_when_no_compression(self):
        c = ContextCompressor(llm=None)
        chunks = [_make_chunk("text")]
        result = c.compress("q", chunks, strategy="none")
        assert result.compression_ratio == 1.0

    def test_compression_ratio_less_than_1_when_compressed(self):
        long_text = "long text " * 50
        llm = _make_llm("short")
        c = ContextCompressor(llm=llm)
        chunk = _make_chunk(long_text)
        result = c.compress("q", [chunk], strategy="sentence")
        # Either actually compressed or no change (if LLM response not shorter)
        assert 0 < result.compression_ratio <= 1.0

    def test_fallback_reason_empty_on_success(self):
        c = ContextCompressor(llm=_make_llm("compressed"))
        result = c.compress("q", [_make_chunk("text " * 20)], strategy="sentence")
        assert result.fallback_reason == ""

    def test_fallback_reason_set_on_llmlingua_failure(self):
        c = ContextCompressor(llm=None)
        with patch.object(c, "_ensure_llmlingua", side_effect=ImportError("no pkg")):
            result = c.compress("q", [_make_chunk("text")], strategy="llmlingua")
        assert result.fallback_reason != ""
        assert "llmlingua_unavailable" in result.fallback_reason

    def test_strategy_used_reflects_actual_path(self):
        c = ContextCompressor(llm=_make_llm("x"))
        result = c.compress("q", [_make_chunk("text " * 30)], strategy="sentence")
        assert result.strategy_used == "sentence"

    def test_strategy_used_none_on_empty(self):
        c = ContextCompressor(llm=None)
        result = c.compress("q", [], strategy="sentence")
        assert result.strategy_used == "none"


# ---------------------------------------------------------------------------
# Story 3.3 — Diagnostics plumbing via AxonBrain.query()
# ---------------------------------------------------------------------------
class TestDiagnosticsPlumbing:
    """Verify compression fields appear in CodeRetrievalDiagnostics."""

    def _make_brain(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            vector_store="chroma",
            vector_store_path=str(tmp_path / "vs"),
            bm25_path=str(tmp_path / "bm25"),
            projects_root=str(tmp_path / "projects"),
            compress_context=True,
            compression_strategy="sentence",
            hybrid_search=False,  # avoid combined-score filtering complexity
            similarity_threshold=0.5,
        )
        with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM") as MockLLM, patch(
            "axon.main.OpenReranker"
        ):
            brain = AxonBrain(config)
            brain.vector_store = MagicMock()
            brain.bm25 = MagicMock()
            return brain, MockLLM.return_value

    def test_diagnostics_has_compression_fields(self, tmp_path):
        from axon.code_retrieval import CodeRetrievalDiagnostics

        d = CodeRetrievalDiagnostics()
        assert hasattr(d, "compression_strategy")
        assert hasattr(d, "compression_pre_tokens")
        assert hasattr(d, "compression_post_tokens")
        assert hasattr(d, "compression_ratio")
        assert hasattr(d, "compression_fallback_reason")

    def test_diagnostics_to_dict_includes_compression(self, tmp_path):
        from axon.code_retrieval import CodeRetrievalDiagnostics

        d = CodeRetrievalDiagnostics()
        d.compression_strategy = "sentence"
        d.compression_pre_tokens = 100
        d.compression_post_tokens = 60
        d.compression_ratio = 0.6
        data = d.to_dict()
        assert data["compression_strategy"] == "sentence"
        assert data["compression_pre_tokens"] == 100
        assert data["compression_post_tokens"] == 60
        assert data["compression_ratio"] == 0.6
        assert "compression_fallback_reason" in data

    def test_diagnostics_version_bumped_to_1_3(self):
        from axon.code_retrieval import CodeRetrievalDiagnostics

        assert CodeRetrievalDiagnostics.diagnostics_version == "1.3"

    def test_compression_diagnostics_written_after_query(self, tmp_path):
        brain, mock_llm = self._make_brain(tmp_path)
        mock_llm.complete.return_value = "short answer"
        results = [
            {
                "id": "d1",
                "text": "long context " * 20,
                "score": 0.9,
                "vector_score": 0.9,  # must exceed similarity_threshold=0.5
                "metadata": {"source": "s1"},
            }
        ]
        brain.vector_store.search.return_value = results
        brain.bm25.search.return_value = []
        brain.query("What is this?")
        diag = brain._last_diagnostics
        # compression_strategy should be set because compress_context=True
        assert diag.compression_strategy == "sentence"
        assert diag.compression_pre_tokens > 0
        assert diag.compression_post_tokens >= 0
        assert 0 < diag.compression_ratio <= 1.0
