"""Integration tests for the Rust bridge — exercises the real compiled axon_rust module.

Skipped automatically when the Rust extension is not compiled/installed.
Run with:
    python -m pytest tests/test_rust_bridge.py -v
"""
from __future__ import annotations

import hashlib
import os

import pytest

# Skip the entire file when the Rust extension is not available.
axon_rust = pytest.importorskip(
    os.getenv("AXON_RUST_MODULE", "axon.axon_rust"),
    reason="axon_rust native module not compiled",
)

from axon.rust_bridge import get_rust_bridge  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str, **meta) -> dict:
    return {"id": meta.pop("id", text[:8]), "text": text, "metadata": meta}


def _make_code_doc(
    text: str,
    symbol_name: str = "",
    symbol_type: str = "function",
    file_path: str = "module.py",
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    meta: dict = {
        "source_class": "code",
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "file_path": file_path,
    }
    if start_line is not None:
        meta["start_line"] = start_line
    if end_line is not None:
        meta["end_line"] = end_line
    return {"id": symbol_name or text[:8], "text": text, "metadata": meta}


# ---------------------------------------------------------------------------
# RustBridge capability flags
# ---------------------------------------------------------------------------


class TestRustBridgeCapabilities:
    def test_can_bm25(self):
        assert get_rust_bridge().can_bm25()

    def test_can_symbol_search(self):
        assert get_rust_bridge().can_symbol_search()

    def test_can_symbol_index(self):
        assert get_rust_bridge().can_symbol_index()

    def test_can_doc_hash(self):
        assert get_rust_bridge().can_doc_hash()

    def test_can_extract_code_tokens(self):
        assert get_rust_bridge().can_extract_code_tokens()

    def test_can_code_lexical_scores(self):
        assert get_rust_bridge().can_code_lexical_scores()


# ---------------------------------------------------------------------------
# preprocess_documents
# ---------------------------------------------------------------------------


class TestPreprocessDocuments:
    def test_strips_extra_whitespace(self):
        docs = [{"text": "hello   world  "}]
        result = axon_rust.preprocess_documents(docs, 1)
        assert result[0]["text"] == "hello world"

    def test_preserves_other_fields(self):
        docs = [{"text": "foo  bar", "id": "123", "metadata": {"source": "x"}}]
        result = axon_rust.preprocess_documents(docs, 1)
        assert result[0]["id"] == "123"
        assert result[0]["metadata"]["source"] == "x"

    def test_empty_list(self):
        assert axon_rust.preprocess_documents([], 1) == []


# ---------------------------------------------------------------------------
# BM25 index + search
# ---------------------------------------------------------------------------


class TestBm25:
    CORPUS = [
        {"text": "the quick brown fox jumps over the lazy dog"},
        {"text": "python is a high level programming language"},
        {"text": "rust provides memory safety without garbage collection"},
        {"text": "machine learning models require training data"},
    ]

    def test_build_and_search_returns_results(self):
        idx = axon_rust.build_bm25_index(self.CORPUS)
        results = axon_rust.search_bm25(idx, "python programming", 3)
        assert len(results) > 0

    def test_most_relevant_doc_ranks_first(self):
        idx = axon_rust.build_bm25_index(self.CORPUS)
        results = axon_rust.search_bm25(idx, "rust memory safety", 4)
        assert results[0][0] == 2, "doc 2 (rust doc) should rank first"

    def test_no_match_returns_empty(self):
        idx = axon_rust.build_bm25_index(self.CORPUS)
        results = axon_rust.search_bm25(idx, "zzznomatchzzz", 5)
        assert results == []

    def test_empty_corpus_returns_empty(self):
        idx = axon_rust.build_bm25_index([])
        assert axon_rust.search_bm25(idx, "hello", 5) == []

    def test_parity_with_rank_bm25(self):
        """Rust BM25 top-1 result matches rank_bm25 top-1 for the same query."""
        rank_bm25 = pytest.importorskip("rank_bm25")
        corpus_texts = [d["text"] for d in self.CORPUS]
        tokenized = [t.lower().split() for t in corpus_texts]
        py_bm25 = rank_bm25.BM25Okapi(tokenized)
        query = "rust memory safety"
        py_scores = py_bm25.get_scores(query.lower().split())
        py_top = int(py_scores.argmax())

        idx = axon_rust.build_bm25_index(self.CORPUS)
        rust_results = axon_rust.search_bm25(idx, query, 1)
        assert rust_results[0][0] == py_top

    def test_bridge_search_bm25_returns_dicts(self):
        bridge = get_rust_bridge()
        idx = bridge.build_bm25_index(self.CORPUS)
        results = bridge.search_bm25(idx, "python", 3)
        assert results is not None
        assert all(isinstance(r, dict) and "index" in r and "score" in r for r in results)


# ---------------------------------------------------------------------------
# Symbol index
# ---------------------------------------------------------------------------


SYMBOL_CORPORA = [
    [
        _make_doc(
            "def load_file(path):",
            source_class="code",
            symbol_name="load_file",
            qualified_name="loaders.load_file",
        ),
        _make_doc(
            "def save_file(path, data):",
            source_class="code",
            symbol_name="save_file",
            qualified_name="loaders.save_file",
        ),
        _make_doc(
            "class FileManager:",
            source_class="code",
            symbol_name="FileManager",
            qualified_name="loaders.FileManager",
        ),
    ]
]


class TestSymbolIndex:
    def test_build_and_search_exact(self):
        idx = axon_rust.build_symbol_index(SYMBOL_CORPORA)
        results = axon_rust.search_symbol_index(idx, ["load_file"], 5)
        assert len(results) > 0
        assert results[0][0] == 0  # doc index 0
        assert abs(results[0][1] - 1.0) < 1e-9  # exact match → score 1.0

    def test_partial_match(self):
        idx = axon_rust.build_symbol_index(SYMBOL_CORPORA)
        results = axon_rust.search_symbol_index(idx, ["load"], 5)
        indices = [r[0] for r in results]
        assert 0 in indices  # load_file should appear

    def test_empty_tokens_returns_empty(self):
        idx = axon_rust.build_symbol_index(SYMBOL_CORPORA)
        assert axon_rust.search_symbol_index(idx, [], 5) == []

    def test_bridge_build_and_search(self):
        bridge = get_rust_bridge()
        idx = bridge.build_symbol_index(SYMBOL_CORPORA)
        assert idx is not None
        results = bridge.search_symbol_index(idx, ["load_file"], 5)
        assert results is not None
        assert len(results) > 0
        assert results[0]["index"] == 0


# ---------------------------------------------------------------------------
# compute_doc_hash
# ---------------------------------------------------------------------------


class TestComputeDocHash:
    def test_known_value(self):
        # echo -n "hello" | md5sum
        assert axon_rust.compute_doc_hash("hello") == "5d41402abc4b2a76b9719d911017c592"

    def test_empty_string(self):
        assert axon_rust.compute_doc_hash("") == hashlib.md5(b"").hexdigest()

    def test_matches_hashlib(self):
        text = "the quick brown fox jumps over the lazy dog"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert axon_rust.compute_doc_hash(text) == expected

    def test_deterministic(self):
        t = "some document with repeated content"
        assert axon_rust.compute_doc_hash(t) == axon_rust.compute_doc_hash(t)

    def test_bridge_compute_doc_hash(self):
        bridge = get_rust_bridge()
        text = "test content"
        result = bridge.compute_doc_hash(text)
        assert result == hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# extract_code_query_tokens
# ---------------------------------------------------------------------------


class TestExtractCodeQueryTokens:
    def test_camelcase(self):
        tokens = set(axon_rust.extract_code_query_tokens("CodeAwareSplitter"))
        assert "codeawaresplitter" in tokens
        assert "code" in tokens
        assert "aware" in tokens
        assert "splitter" in tokens

    def test_snake_case(self):
        tokens = set(axon_rust.extract_code_query_tokens("split_python_ast"))
        assert "split" in tokens
        assert "python" in tokens

    def test_dotted_qualified(self):
        tokens = set(axon_rust.extract_code_query_tokens("axon.loaders"))
        assert "axon" in tokens
        assert "loaders" in tokens
        assert "axon.loaders" in tokens

    def test_basename_from_extension(self):
        tokens = set(axon_rust.extract_code_query_tokens("loaders.py"))
        assert "loaders" in tokens

    def test_min_length(self):
        tokens = set(axon_rust.extract_code_query_tokens("a ab abc abcd"))
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abcd" in tokens

    def test_empty_query(self):
        assert axon_rust.extract_code_query_tokens("") == []

    def test_bridge_wraps_in_frozenset(self):
        bridge = get_rust_bridge()
        result = bridge.extract_code_query_tokens("CodeAwareSplitter")
        assert result is not None
        assert isinstance(result, frozenset)
        assert "code" in result


# ---------------------------------------------------------------------------
# code_lexical_scores
# ---------------------------------------------------------------------------


class TestCodeLexicalScores:
    def _results(self):
        return [
            _make_code_doc(
                "def load_file(path): ...",
                symbol_name="load_file",
                symbol_type="function",
                file_path="loaders.py",
                start_line=10,
                end_line=20,
            ),
            _make_code_doc(
                "def unrelated(): ...",
                symbol_name="unrelated",
                symbol_type="function",
                file_path="other.py",
            ),
            {"id": "prose", "text": "This document has no code metadata.", "metadata": {}},
        ]

    def test_exact_symbol_hit_scores_higher(self):
        results = self._results()
        scores, max_lex = axon_rust.code_lexical_scores(results, ["load_file", "loaders"])
        assert scores[0] > scores[1], "load_file result should score higher than unrelated"
        assert max_lex > 0.0

    def test_non_code_doc_scores_zero(self):
        results = self._results()
        scores, _ = axon_rust.code_lexical_scores(results, ["load_file"])
        assert scores[2] == 0.0, "prose doc (no source_class=code) must score 0"

    def test_no_token_match_returns_zero_max(self):
        # No symbol/text hits fire; only the line-range tiebreaker (≤0.05) may apply.
        results = self._results()
        scores, max_lex = axon_rust.code_lexical_scores(results, ["zzznomatch"])
        assert max_lex <= 0.05  # no meaningful match — only tiny line-range bonus at most

    def test_empty_results(self):
        scores, max_lex = axon_rust.code_lexical_scores([], ["load_file"])
        assert scores == []
        assert max_lex == 0.0

    def test_empty_tokens_returns_zeros(self):
        results = self._results()
        scores, max_lex = axon_rust.code_lexical_scores(results, [])
        assert all(s == 0.0 for s in scores)
        assert max_lex == 0.0

    def test_bridge_code_lexical_scores(self):
        bridge = get_rust_bridge()
        results = self._results()
        out = bridge.code_lexical_scores(results, ["load_file"])
        assert out is not None
        scores, max_lex = out
        assert scores[0] > 0.0
        assert max_lex > 0.0


# ---------------------------------------------------------------------------
# decode_corpus_json
# ---------------------------------------------------------------------------


class TestCorpusJsonDecode:
    def _dedup_v1_bytes(self, n: int = 5) -> bytes:
        import json

        texts = [f"text body {i}" for i in range(n)]
        docs = [{"id": f"doc{i}", "t": i, "metadata": {"k": i, "tag": f"t{i}"}} for i in range(n)]
        payload = {"format": "dedup_v1", "texts": texts, "docs": docs}
        return json.dumps(payload).encode()

    def _legacy_bytes(self, n: int = 3) -> bytes:
        import json

        docs = [{"id": f"d{i}", "text": f"text {i}", "metadata": {"x": i}} for i in range(n)]
        return json.dumps(docs).encode()

    def test_decodes_dedup_v1_format(self):
        result = axon_rust.decode_corpus_json(self._dedup_v1_bytes(5))
        assert result is not None
        assert len(result) == 5
        assert result[0]["id"] == "doc0"
        assert result[0]["text"] == "text body 0"
        assert result[0]["metadata"]["k"] == 0

    def test_decodes_legacy_list_format(self):
        result = axon_rust.decode_corpus_json(self._legacy_bytes(3))
        assert result is not None
        assert len(result) == 3
        assert result[1]["text"] == "text 1"

    def test_returns_none_on_invalid_json(self):
        result = axon_rust.decode_corpus_json(b"not valid json {{{")
        assert result is None

    def test_parity_with_python_decoder(self):
        import json

        raw = self._dedup_v1_bytes(10)
        rust_result = axon_rust.decode_corpus_json(raw)
        assert rust_result is not None

        payload = json.loads(raw)
        texts = payload["texts"]
        py_result = [
            {"id": d["id"], "text": texts[d["t"]], "metadata": d["metadata"]}
            for d in payload["docs"]
        ]
        assert len(rust_result) == len(py_result)
        for r, p in zip(rust_result, py_result):
            assert r["id"] == p["id"]
            assert r["text"] == p["text"]

    def test_metadata_roundtrip(self):
        import json

        payload = {
            "format": "dedup_v1",
            "texts": ["hello"],
            "docs": [{"id": "x", "t": 0, "metadata": {"nested": {"a": 1}, "lst": [1, 2]}}],
        }
        result = axon_rust.decode_corpus_json(json.dumps(payload).encode())
        assert result is not None
        assert result[0]["metadata"]["nested"]["a"] == 1
        assert result[0]["metadata"]["lst"] == [1, 2]

    def test_bridge_decode_corpus_json(self):
        bridge = get_rust_bridge()
        assert bridge.can_decode_corpus_json()
        result = bridge.decode_corpus_json(self._dedup_v1_bytes(3))
        assert result is not None
        assert len(result) == 3


# ---------------------------------------------------------------------------
# encode/decode_corpus_msgpack
# ---------------------------------------------------------------------------


class TestCorpusMsgpack:
    def _make_payload(self, n: int = 5):
        texts = [f"body text number {i}" for i in range(n)]
        docs = [{"id": f"doc{i}", "t": i, "metadata": {"k": i}} for i in range(n)]
        return texts, docs

    def test_encode_decode_roundtrip(self):
        texts, docs = self._make_payload(10)
        raw = axon_rust.encode_corpus_msgpack(texts, docs)
        result = axon_rust.decode_corpus_msgpack(raw)
        assert result is not None
        assert len(result) == 10
        assert result[3]["id"] == "doc3"
        assert result[3]["text"] == "body text number 3"
        assert result[3]["metadata"]["k"] == 3

    def test_msgpack_smaller_than_json(self):
        import json

        texts, docs = self._make_payload(100)
        payload = {"format": "dedup_v1", "texts": texts, "docs": docs}
        json_bytes = json.dumps(payload).encode()
        mp_bytes = axon_rust.encode_corpus_msgpack(texts, docs)
        assert len(mp_bytes) < len(
            json_bytes
        ), f"msgpack {len(mp_bytes)} should be < json {len(json_bytes)}"

    def test_parity_with_json_decode(self):
        import json

        texts, docs = self._make_payload(20)
        payload = {"format": "dedup_v1", "texts": texts, "docs": docs}
        json_result = axon_rust.decode_corpus_json(json.dumps(payload).encode())
        mp_result = axon_rust.decode_corpus_msgpack(axon_rust.encode_corpus_msgpack(texts, docs))
        assert json_result is not None
        assert mp_result is not None
        assert len(json_result) == len(mp_result)
        for j, m in zip(json_result, mp_result):
            assert j["id"] == m["id"]
            assert j["text"] == m["text"]

    def test_empty_corpus(self):
        raw = axon_rust.encode_corpus_msgpack([], [])
        result = axon_rust.decode_corpus_msgpack(raw)
        assert result is not None
        assert result == []

    def test_metadata_with_nested_dicts(self):
        texts = ["some text"]
        docs = [{"id": "d0", "t": 0, "metadata": {"nested": {"x": 1}, "lst": [True, None, 3.14]}}]
        raw = axon_rust.encode_corpus_msgpack(texts, docs)
        result = axon_rust.decode_corpus_msgpack(raw)
        assert result is not None
        meta = result[0]["metadata"]
        assert meta["nested"]["x"] == 1
        assert meta["lst"][0] is True
        assert meta["lst"][1] is None

    def test_bridge_corpus_msgpack(self):
        bridge = get_rust_bridge()
        assert bridge.can_corpus_msgpack()
        texts, docs = self._make_payload(5)
        raw = bridge.encode_corpus_msgpack(texts, docs)
        assert raw is not None
        result = bridge.decode_corpus_msgpack(raw)
        assert result is not None
        assert len(result) == 5


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


class TestSha256:
    def test_known_value(self):
        import hashlib

        expected = hashlib.sha256(b"hello").hexdigest()
        assert axon_rust.compute_sha256("hello") == expected

    def test_strips_whitespace(self):
        assert axon_rust.compute_sha256("  hello  ") == axon_rust.compute_sha256("hello")

    def test_empty_string(self):
        import hashlib

        assert axon_rust.compute_sha256("") == hashlib.sha256(b"").hexdigest()

    def test_deterministic(self):
        t = "some test content for sha256"
        assert axon_rust.compute_sha256(t) == axon_rust.compute_sha256(t)

    def test_bridge_compute_sha256(self):
        import hashlib

        bridge = get_rust_bridge()
        assert bridge.can_sha256()
        text = "bridge sha256 test"
        result = bridge.compute_sha256(text)
        assert result == hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Binary hash store
# ---------------------------------------------------------------------------


class TestHashStoreBinary:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "hashes.bin")
        hashes = {
            "5d41402abc4b2a76b9719d911017c592",
            "d41d8cd98f00b204e9800998ecf8427e",
            "acbd18db4cc2f85cedef654fccc4a4d8",
        }
        axon_rust.save_hash_store_binary(path, list(hashes))
        loaded = set(axon_rust.load_hash_store_binary(path))
        assert loaded == hashes

    def test_binary_smaller_than_text(self, tmp_path):
        bin_path = str(tmp_path / "h.bin")
        txt_path = str(tmp_path / "h.txt")
        hashes = ["5d41402abc4b2a76b9719d911017c592"] * 1000
        axon_rust.save_hash_store_binary(bin_path, hashes)
        with open(txt_path, "w") as f:
            f.write("\n".join(hashes))
        assert (tmp_path / "h.bin").stat().st_size < (tmp_path / "h.txt").stat().st_size

    def test_probe_found(self, tmp_path):
        path = str(tmp_path / "h.bin")
        target = "5d41402abc4b2a76b9719d911017c592"
        axon_rust.save_hash_store_binary(path, [target, "acbd18db4cc2f85cedef654fccc4a4d8"])
        assert axon_rust.probe_hash_store(path, target) is True

    def test_probe_not_found(self, tmp_path):
        path = str(tmp_path / "h.bin")
        axon_rust.save_hash_store_binary(path, ["5d41402abc4b2a76b9719d911017c592"])
        assert axon_rust.probe_hash_store(path, "acbd18db4cc2f85cedef654fccc4a4d8") is False

    def test_probe_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.bin")
        assert axon_rust.probe_hash_store(path, "5d41402abc4b2a76b9719d911017c592") is None

    def test_bridge_hash_store(self, tmp_path):
        bridge = get_rust_bridge()
        assert bridge.can_hash_store_binary()
        path = str(tmp_path / "h.bin")
        hashes = {"5d41402abc4b2a76b9719d911017c592", "acbd18db4cc2f85cedef654fccc4a4d8"}
        ok = bridge.save_hash_store_binary(path, hashes)
        assert ok is True
        loaded = bridge.load_hash_store_binary(path)
        assert loaded is not None
        assert loaded == hashes


# ---------------------------------------------------------------------------
# Sentence window msgpack codec
# ---------------------------------------------------------------------------


class TestSentenceCodec:
    def _make_records(self):
        return {
            "sent_0": {"chunk_id": "chk_a", "position": 0, "text": "First sentence."},
            "sent_1": {"chunk_id": "chk_a", "position": 1, "text": "Second sentence."},
        }

    def _make_chunk_to_sentences(self):
        return {"chk_a": ["sent_0", "sent_1"], "chk_b": ["sent_2"]}

    def test_sentence_index_roundtrip(self):
        records = self._make_records()
        cts = self._make_chunk_to_sentences()
        raw = axon_rust.encode_sentence_index(records, cts)
        result = axon_rust.decode_sentence_index(raw)
        assert result is not None
        r_records, r_cts = result
        assert r_records["sent_0"]["chunk_id"] == "chk_a"
        assert r_cts["chk_a"] == ["sent_0", "sent_1"]

    def test_sentence_meta_roundtrip(self):
        ids = ["sent_0", "sent_1", "sent_2"]
        meta = [
            {"chunk_id": "a", "pos": 0},
            {"chunk_id": "a", "pos": 1},
            {"chunk_id": "b", "pos": 0},
        ]
        raw = axon_rust.encode_sentence_meta(ids, meta)
        result = axon_rust.decode_sentence_meta(raw)
        assert result is not None
        r_ids, r_meta = result
        assert list(r_ids) == ids
        assert r_meta[1]["chunk_id"] == "a"

    def test_nested_metadata_preserved(self):
        records = {"s0": {"nested": {"x": [1, 2, 3]}, "flag": True}}
        cts = {"c0": ["s0"]}
        raw = axon_rust.encode_sentence_index(records, cts)
        result = axon_rust.decode_sentence_index(raw)
        assert result is not None
        r_records, _ = result
        assert r_records["s0"]["nested"]["x"] == [1, 2, 3]
        assert r_records["s0"]["flag"] is True

    def test_bridge_sentence_codec(self):
        bridge = get_rust_bridge()
        assert bridge.can_sentence_codec()
        records = self._make_records()
        cts = self._make_chunk_to_sentences()
        raw = bridge.encode_sentence_index(records, cts)
        assert raw is not None
        result = bridge.decode_sentence_index(raw)
        assert result is not None


# ---------------------------------------------------------------------------
# segment_text
# ---------------------------------------------------------------------------


class TestSegmentText:
    def test_basic_split(self):
        # Use min_chars=3 so "Hi." (3 chars) doesn't get merged
        parts = axon_rust.segment_text("Hello world. This is another sentence.", 5)
        assert len(parts) == 2
        assert parts[0] == "Hello world."

    def test_merges_short_fragment(self):
        # "Ok." is 3 chars < min_chars=10 → merged into preceding
        parts = axon_rust.segment_text("Hello world. Ok.", 10)
        assert len(parts) == 1
        assert "Ok." in parts[0]

    def test_empty_input(self):
        assert axon_rust.segment_text("", 10) == []
        assert axon_rust.segment_text("   ", 10) == []

    def test_parity_with_python(self):
        from axon.sentence_window import _MIN_SENTENCE_CHARS, _SENTENCE_BOUNDARY

        text = "The quick brown fox jumps. It ran quickly across the field. Then it stopped."
        # Python implementation
        raw = _SENTENCE_BOUNDARY.split(text.strip())
        py_parts: list[str] = []
        for part in raw:
            part = part.strip()
            if not part:
                continue
            if len(part) < _MIN_SENTENCE_CHARS and py_parts:
                py_parts[-1] = py_parts[-1] + " " + part
            else:
                py_parts.append(part)
        py_result = [s for s in py_parts if s.strip()]
        rust_result = axon_rust.segment_text(text, _MIN_SENTENCE_CHARS)
        assert rust_result == py_result

    def test_bridge_segment_text(self):
        bridge = get_rust_bridge()
        assert bridge.can_segment_text()
        result = bridge.segment_text("Hello world. This is a long sentence.", 5)
        assert result is not None
        assert len(result) == 2


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical(self):
        score = axon_rust.cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(score - 1.0) < 1e-9

    def test_orthogonal(self):
        score = axon_rust.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(score - 0.0) < 1e-9

    def test_zero_vector(self):
        score = axon_rust.cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert score == 0.0  # no division by zero

    def test_partial(self):
        import math

        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2.0)
        score = axon_rust.cosine_similarity(a, b)
        assert abs(score - expected) < 1e-9

    def test_bridge_cosine_similarity(self):
        bridge = get_rust_bridge()
        assert bridge.can_cosine_similarity()
        result = bridge.cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert result is not None
        assert abs(result - 1.0) < 1e-9


# ── Phase 3: Score fusion ─────────────────────────────────────────────────────


class TestScoreFusion:
    V = [{"id": "a", "score": 0.9, "text": "hello"}, {"id": "b", "score": 0.5, "text": "world"}]
    B = [{"id": "b", "score": 15.0, "text": "world"}, {"id": "c", "score": 5.0, "text": "foo"}]

    def test_weighted_returns_all_docs(self):
        out = axon_rust.score_fusion_weighted(self.V, self.B, 0.7)
        ids = {d["id"] for d in out}
        assert ids == {"a", "b", "c"}

    def test_weighted_sorted_by_score(self):
        out = axon_rust.score_fusion_weighted(self.V, self.B, 0.7)
        scores = [d["score"] for d in out]
        assert scores == sorted(scores, reverse=True)

    def test_weighted_vector_score_preserved(self):
        out = axon_rust.score_fusion_weighted(self.V, self.B, 0.7)
        a = next(d for d in out if d["id"] == "a")
        assert abs(a["vector_score"] - 0.9) < 1e-9

    def test_weighted_empty_bm25(self):
        out = axon_rust.score_fusion_weighted(self.V, [], 0.7)
        assert len(out) == 2

    def test_weighted_empty_both(self):
        out = axon_rust.score_fusion_weighted([], [], 0.7)
        assert out == []

    def test_rrf_returns_all_docs(self):
        out = axon_rust.score_fusion_rrf(self.V, self.B, 60)
        ids = {d["id"] for d in out}
        assert ids == {"a", "b", "c"}

    def test_rrf_sorted_by_score(self):
        out = axon_rust.score_fusion_rrf(self.V, self.B, 60)
        scores = [d["score"] for d in out]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_vector_score_preserved(self):
        out = axon_rust.score_fusion_rrf(self.V, self.B, 60)
        a = next(d for d in out if d["id"] == "a")
        assert abs(a["vector_score"] - 0.9) < 1e-9

    def test_bridge_can_score_fusion(self):
        bridge = get_rust_bridge()
        assert bridge.can_score_fusion()

    def test_bridge_weighted_parity(self):
        bridge = get_rust_bridge()
        result = bridge.score_fusion_weighted(self.V, self.B, 0.7)
        assert result is not None
        expected = axon_rust.score_fusion_weighted(self.V, self.B, 0.7)
        assert [d["id"] for d in result] == [d["id"] for d in expected]

    def test_bridge_rrf_parity(self):
        bridge = get_rust_bridge()
        result = bridge.score_fusion_rrf(self.V, self.B, 60)
        assert result is not None
        expected = axon_rust.score_fusion_rrf(self.V, self.B, 60)
        assert [d["id"] for d in result] == [d["id"] for d in expected]


# ── Phase 3: MMR reranking ────────────────────────────────────────────────────


class TestMmrRerank:
    RESULTS = [
        {"id": "a", "score": 0.9, "text": "the quick brown fox jumps over the lazy dog"},
        {
            "id": "b",
            "score": 0.8,
            "text": "the quick brown fox jumps over the lazy dog",
        },  # near-dup
        {"id": "c", "score": 0.7, "text": "python programming language syntax variables"},
        {"id": "d", "score": 0.6, "text": "rust memory safety ownership borrowing"},
    ]

    def test_drops_near_duplicate(self):
        out = axon_rust.mmr_rerank(self.RESULTS, 0.5, 0.85)
        ids = [d["id"] for d in out]
        # "a" and "b" are identical text — one should be dropped
        assert not ("a" in ids and "b" in ids), f"Near-dup not dropped: {ids}"

    def test_single_result_unchanged(self):
        single = [{"id": "x", "score": 1.0, "text": "hello world"}]
        out = axon_rust.mmr_rerank(single, 0.5, 0.85)
        assert len(out) == 1
        assert out[0]["id"] == "x"

    def test_empty_returns_empty(self):
        out = axon_rust.mmr_rerank([], 0.5, 0.85)
        assert out == []

    def test_diverse_results_all_kept(self):
        diverse = [
            {"id": "a", "score": 0.9, "text": "cat dog animal pet"},
            {"id": "b", "score": 0.8, "text": "python rust programming language"},
            {"id": "c", "score": 0.7, "text": "quantum physics particle wave"},
        ]
        out = axon_rust.mmr_rerank(diverse, 0.5, 0.85)
        assert len(out) == 3

    def test_bridge_can_mmr_rerank(self):
        bridge = get_rust_bridge()
        assert bridge.can_mmr_rerank()

    def test_bridge_mmr_rerank(self):
        bridge = get_rust_bridge()
        result = bridge.mmr_rerank(self.RESULTS, 0.5, 0.85)
        assert result is not None
        assert len(result) < len(self.RESULTS)  # at least one dup dropped


# ── Phase 3: Code-doc bridge edges ───────────────────────────────────────────


class TestCodeDocBridgeEdges:
    SYM_LOOKUP = {"MyClass": "node_myclass", "parse_token": "node_parse"}
    CHUNKS = [
        {"id": "chunk1", "text": "The MyClass is used here for parsing."},
        {"id": "chunk2", "text": "parse_token handles tokenization."},
        {"id": "chunk3", "text": "No symbols here, just plain text."},
    ]

    def test_emits_edges_for_matches(self):
        edges = axon_rust.build_code_doc_bridge_edges(self.SYM_LOOKUP, self.CHUNKS, [])
        targets = {e["target"] for e in edges}
        assert "chunk1" in targets
        assert "chunk2" in targets

    def test_no_edge_for_non_match(self):
        edges = axon_rust.build_code_doc_bridge_edges(self.SYM_LOOKUP, self.CHUNKS, [])
        targets = {e["target"] for e in edges}
        assert "chunk3" not in targets

    def test_edge_type_is_mentioned_in(self):
        edges = axon_rust.build_code_doc_bridge_edges(self.SYM_LOOKUP, self.CHUNKS, [])
        assert all(e["edge_type"] == "MENTIONED_IN" for e in edges)

    def test_skips_existing_edges(self):
        existing = [("node_myclass", "chunk1", "MENTIONED_IN")]
        edges = axon_rust.build_code_doc_bridge_edges(self.SYM_LOOKUP, self.CHUNKS, existing)
        new_targets = [e["target"] for e in edges if e["source"] == "node_myclass"]
        assert "chunk1" not in new_targets

    def test_no_substring_match(self):
        sym = {"MyClassFoo": "node_foo"}
        chunks = [{"id": "c1", "text": "MyClass is here but not MyClassFoo"}]
        edges = axon_rust.build_code_doc_bridge_edges(sym, chunks, [])
        assert any(e["source"] == "node_foo" for e in edges)

    def test_word_boundary_prevents_partial_match(self):
        sym = {"parse": "node_parse"}
        chunks = [{"id": "c1", "text": "parseToken is a method"}]
        edges = axon_rust.build_code_doc_bridge_edges(sym, chunks, [])
        assert not any(e["source"] == "node_parse" for e in edges)

    def test_bridge_can_code_doc_bridge(self):
        bridge = get_rust_bridge()
        assert bridge.can_code_doc_bridge()

    def test_bridge_parity(self):
        bridge = get_rust_bridge()
        result = bridge.build_code_doc_bridge_edges(self.SYM_LOOKUP, self.CHUNKS, [])
        assert result is not None
        expected = axon_rust.build_code_doc_bridge_edges(self.SYM_LOOKUP, self.CHUNKS, [])
        assert {e["target"] for e in result} == {e["target"] for e in expected}


# ── Phase 3: Entity graph I/O ─────────────────────────────────────────────────


class TestEntityGraphCodec:
    GRAPH = {
        "python": {
            "description": "Programming language",
            "chunk_ids": ["c1", "c2"],
            "type": "LANGUAGE",
            "frequency": 2,
            "degree": 1,
        },
        "rust": {
            "description": "Systems language",
            "chunk_ids": ["c3"],
            "type": "LANGUAGE",
            "frequency": 1,
            "degree": 0,
        },
    }

    def test_roundtrip(self):
        raw = axon_rust.encode_entity_graph(self.GRAPH)
        result = axon_rust.decode_entity_graph(raw)
        assert result is not None
        assert set(result.keys()) == set(self.GRAPH.keys())
        assert result["python"]["chunk_ids"] == ["c1", "c2"]

    def test_msgpack_smaller_than_json(self):
        import json

        raw_mp = axon_rust.encode_entity_graph(self.GRAPH)
        raw_json = json.dumps(self.GRAPH).encode()
        assert len(raw_mp) < len(raw_json)

    def test_invalid_bytes_returns_none(self):
        result = axon_rust.decode_entity_graph(b"\xff\xfe\xfd")
        assert result is None

    def test_empty_graph(self):
        raw = axon_rust.encode_entity_graph({})
        result = axon_rust.decode_entity_graph(raw)
        assert result == {}

    def test_bridge_can_entity_graph_codec(self):
        bridge = get_rust_bridge()
        assert bridge.can_entity_graph_codec()

    def test_bridge_roundtrip(self):
        bridge = get_rust_bridge()
        raw = bridge.encode_entity_graph(self.GRAPH)
        assert raw is not None
        result = bridge.decode_entity_graph(raw)
        assert result is not None
        assert set(result.keys()) == set(self.GRAPH.keys())


class TestEntityEmbeddingsCodec:
    EMBEDDINGS = {
        "python": [0.1, 0.2, 0.3],
        "rust": [0.4, 0.5, 0.6],
    }

    def test_roundtrip(self):
        raw = axon_rust.encode_entity_embeddings(self.EMBEDDINGS)
        result = axon_rust.decode_entity_embeddings(raw)
        assert result is not None
        assert set(result.keys()) == set(self.EMBEDDINGS.keys())
        assert len(result["python"]) == 3

    def test_invalid_bytes_returns_none(self):
        result = axon_rust.decode_entity_embeddings(b"\xde\xad\xbe\xef")
        assert result is None

    def test_bridge_can_entity_embeddings_codec(self):
        bridge = get_rust_bridge()
        assert bridge.can_entity_embeddings_codec()

    def test_bridge_roundtrip(self):
        bridge = get_rust_bridge()
        raw = bridge.encode_entity_embeddings(self.EMBEDDINGS)
        assert raw is not None
        result = bridge.decode_entity_embeddings(raw)
        assert result is not None
        assert set(result.keys()) == set(self.EMBEDDINGS.keys())


class TestRelationGraphCodec:
    GRAPH = {
        "alice": [
            {
                "target": "bob",
                "relation": "knows",
                "chunk_id": "c1",
                "subject": "alice",
                "object": "bob",
                "strength": 7,
                "weight": 2.5,
            }
        ]
    }

    def test_roundtrip(self):
        raw = axon_rust.encode_relation_graph(self.GRAPH)
        result = axon_rust.decode_relation_graph(raw)
        assert result is not None
        assert set(result.keys()) == {"alice"}
        assert result["alice"][0]["target"] == "bob"
        assert result["alice"][0]["strength"] == 7

    def test_invalid_bytes_returns_none(self):
        result = axon_rust.decode_relation_graph(b"\xa1a")
        assert result is None

    def test_bridge_roundtrip(self):
        bridge = get_rust_bridge()
        assert bridge.can_relation_graph_codec()
        raw = bridge.encode_relation_graph(self.GRAPH)
        assert raw is not None
        result = bridge.decode_relation_graph(raw)
        assert result is not None
        assert result["alice"][0]["weight"] == 2.5


class TestResolveEntityAliasGroups:
    def test_bridge_reports_capability(self):
        bridge = get_rust_bridge()
        assert bridge.can_resolve_entity_alias_groups()

    def test_bridge_groups_similar_embeddings(self):
        bridge = get_rust_bridge()
        groups = bridge.resolve_entity_alias_groups(
            [
                [1.0, 0.0, 0.0],
                [0.999, 0.001, 0.0],
                [0.0, 1.0, 0.0],
            ],
            0.99,
        )
        assert groups is not None
        normalized = sorted(sorted(group) for group in groups)
        assert [0, 1] in normalized


class TestGraphRagAcceleration:
    def test_bridge_can_build_graph_edges(self):
        bridge = get_rust_bridge()
        assert bridge.can_build_graph_edges()

    def test_build_graph_edges_roundtrip_shape(self):
        bridge = get_rust_bridge()
        entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "frequency": 1},
        }
        relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1", "weight": 2}]
        }

        nodes, edges = bridge.build_graph_edges(entity_graph, relation_graph)

        assert nodes is not None
        assert edges is not None
        assert "alice" in nodes
        assert ("alice", "bob", 2.0) in edges

    def test_run_louvain_returns_mapping(self):
        bridge = get_rust_bridge()
        mapping = bridge.run_louvain(
            ["alice", "bob", "carol"],
            [("alice", "bob", 1.0), ("bob", "carol", 1.0)],
            resolution=1.0,
        )
        assert mapping is not None
        assert set(mapping.keys()) == {"alice", "bob", "carol"}
        assert all(isinstance(v, int) for v in mapping.values())

    def test_merge_entities_into_graph_mutates_in_place(self):
        bridge = get_rust_bridge()
        entity_graph = {
            "alice": {
                "description": "Existing",
                "type": "PERSON",
                "chunk_ids": ["c1"],
                "frequency": 1,
                "degree": 0,
            }
        }
        results = [
            (
                "c2",
                [
                    {"name": "Alice", "type": "PERSON", "description": "Existing"},
                    {"name": "Bob", "type": "PERSON", "description": "New"},
                ],
            )
        ]

        inserted = bridge.merge_entities_into_graph(entity_graph, results)

        assert inserted is not None
        assert "c2" in entity_graph["alice"]["chunk_ids"]
        assert entity_graph["bob"]["type"] == "PERSON"


# ── Phase 3: Dedup corpus payload ─────────────────────────────────────────────


class TestDedupCorpusPayload:
    CORPUS = [
        {"id": "doc0", "text": "hello world", "metadata": {"source": "a.txt"}},
        {"id": "doc1", "text": "foo bar", "metadata": {"source": "b.txt"}},
        {"id": "doc2", "text": "hello world", "metadata": {"source": "c.txt"}},  # dup text
    ]

    def test_deduplicates_texts(self):
        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        assert len(texts) == 2  # "hello world" deduplicated

    def test_docs_count_matches_corpus(self):
        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        assert len(docs) == 3

    def test_dup_docs_share_text_index(self):
        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        t0 = docs[0]["t"]
        t2 = docs[2]["t"]
        assert t0 == t2

    def test_text_index_in_range(self):
        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        for d in docs:
            assert 0 <= d["t"] < len(texts)

    def test_ids_preserved(self):
        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        assert docs[0]["id"] == "doc0"
        assert docs[1]["id"] == "doc1"
        assert docs[2]["id"] == "doc2"

    def test_metadata_preserved(self):
        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        assert docs[0]["metadata"]["source"] == "a.txt"

    def test_empty_corpus(self):
        texts, docs = axon_rust.build_dedup_corpus_payload([])
        assert texts == []
        assert docs == []

    def test_parity_with_python(self):
        """Rust output matches Python _build_dedup_corpus_payload logic."""
        from axon.retrievers import BM25Retriever

        retriever = BM25Retriever.__new__(BM25Retriever)
        retriever.corpus = self.CORPUS
        retriever._rust = get_rust_bridge()
        # Temporarily disable Rust path to get Python output
        import unittest.mock as mock

        with mock.patch.object(retriever._rust, "can_dedup_corpus_payload", return_value=False):
            py_payload = retriever._build_dedup_corpus_payload()

        texts, docs = axon_rust.build_dedup_corpus_payload(self.CORPUS)
        assert texts == py_payload["texts"]
        for i, d in enumerate(docs):
            assert d["id"] == py_payload["docs"][i]["id"]
            assert d["t"] == py_payload["docs"][i]["t"]

    def test_bridge_can_dedup_corpus_payload(self):
        bridge = get_rust_bridge()
        assert bridge.can_dedup_corpus_payload()

    def test_bridge_dedup_corpus_payload(self):
        bridge = get_rust_bridge()
        result = bridge.build_dedup_corpus_payload(self.CORPUS)
        assert result is not None
        texts, docs = result
        assert len(texts) == 2
        assert len(docs) == 3
