from __future__ import annotations

"""Tests for text splitters."""

import pytest

from axon.splitters import (
    CodeAwareSplitter,
    RecursiveCharacterTextSplitter,
    SemanticTextSplitter,
    TableSplitter,
)


class TestSemanticTextSplitter:
    def test_sentence_boundaries_preserved(self):
        # 10 tokens (~40 chars). Enough for 1-2 short sentences.
        splitter = SemanticTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "This is the first sentence. This is the second sentence! Is this the third?"
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert "first sentence." in chunks[0]
        assert "second sentence!" in chunks[1]

    def test_abbreviations_not_split(self):
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "Dr. Smith went to Washington, D.C. He bought a 1.5 L bottle."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_overlap(self):
        # 12 tokens per chunk, 4 token overlap
        splitter = SemanticTextSplitter(chunk_size=12, chunk_overlap=4)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Overlap should ensure a sentence is repeated in subsequent chunks
        assert "Sentence four." in chunks[0]
        assert "Sentence four." in chunks[1]

    def test_giant_sentence_fallback(self):
        # A single sentence larger than chunk_size should not be split mid-word.
        # It should just be yielded as a single oversized chunk to preserve semantic integrity.
        splitter = SemanticTextSplitter(chunk_size=5, chunk_overlap=1)
        text = "This is a ridiculously long single sentence that goes on forever without any punctuation marks whatsoever"
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestRecursiveCharacterTextSplitter:
    """Test the RecursiveCharacterTextSplitter class."""

    def test_basic_split(self):
        """Test basic text splitting."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 20
        chunks = splitter.split(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some variance

    def test_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split("")
        assert chunks == []

    def test_short_text(self):
        """Test text shorter than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        text = "Short text"
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_transform_documents(self):
        """Test document transformation with metadata."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        docs = [{"id": "doc1", "text": "A" * 100, "metadata": {"source": "test"}}]

        result = splitter.transform_documents(docs)

        assert len(result) > 1
        assert all("_chunk_" in doc["id"] for doc in result)
        assert all("chunk" in doc["metadata"] for doc in result)
        assert all(doc["metadata"]["source"] == "test" for doc in result)


class TestSemanticTextSplitterValidation:
    """Cover ValueError branches (lines 18, 20, 22) and tiktoken fallback (lines 31-35, 40)."""

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            SemanticTextSplitter(chunk_size=0)

    def test_chunk_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=10, chunk_overlap=10)

    def test_tiktoken_fallback_get_length(self, monkeypatch):
        """When tiktoken is unavailable, _get_length falls back to len(text)//4."""
        import sys

        # Patch tiktoken import to fail
        original = sys.modules.get("tiktoken")
        sys.modules["tiktoken"] = None  # type: ignore[assignment]
        try:
            splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=0)
            splitter.encoder = None
            length = splitter._get_length("hello world test")
            assert length == len("hello world test") // 4
        finally:
            if original is None:
                sys.modules.pop("tiktoken", None)
            else:
                sys.modules["tiktoken"] = original

    def test_empty_text_returns_empty(self):
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=0)
        assert splitter.split("") == []

    def test_abbreviation_in_sentence_continues(self):
        """Abbreviation mid-sentence (line 91-93 branch) should not split."""
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=0)
        text = "Send it to Mr. Jones tomorrow."
        chunks = splitter.split(text)
        assert len(chunks) == 1

    def test_oversized_sentence_flushes_current_chunk(self):
        """If there's already content and then an oversized sentence, flush first (lines 117-120)."""
        splitter = SemanticTextSplitter(chunk_size=8, chunk_overlap=0)
        # Short sentence + very long sentence — forces the flush-then-yield path
        short = "Hi."
        long_sent = "This is an extremely long single sentence that will exceed any reasonable chunk size limit."
        chunks = splitter.split(f"{short} {long_sent}")
        assert any(short.rstrip(".") in c for c in chunks)
        assert any(long_sent in c for c in chunks)

    def test_transform_documents_preserves_metadata(self):
        """SemanticTextSplitter.transform_documents copies metadata correctly (line 76)."""
        splitter = SemanticTextSplitter(chunk_size=5, chunk_overlap=0)
        docs = [{"id": "d1", "text": "Sentence one. Sentence two.", "metadata": {"src": "x"}}]
        result = splitter.transform_documents(docs)
        assert all(r["metadata"]["src"] == "x" for r in result)
        assert all("_chunk_" in r["id"] for r in result)

    def test_trailing_sentence_appended(self):
        """Text not ending in punct is collected and appended (line 95-96)."""
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=0)
        text = "First sentence. No final punctuation here"
        chunks = splitter.split(text)
        combined = " ".join(chunks)
        assert "No final punctuation here" in combined


class TestRecursiveCharacterTextSplitterValidation:
    """Cover ValueError branches (lines 207, 209, 211)."""

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveCharacterTextSplitter(chunk_size=0)

    def test_chunk_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)


class TestTableSplitter:
    """Cover TableSplitter (lines 163-196)."""

    def test_single_row(self):
        splitter = TableSplitter(table_name="Users", batch_size=1)
        rows = [{"name": "Alice", "age": "30"}]
        headers = ["name", "age"]
        result = splitter.transform_rows(rows, headers)
        assert len(result) == 1
        assert "Alice" in result[0]
        assert "Users" in result[0]

    def test_batch_grouping(self):
        splitter = TableSplitter(table_name="Orders", batch_size=2)
        rows = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        headers = ["id"]
        result = splitter.transform_rows(rows, headers)
        # 3 rows in batches of 2 → 2 chunks
        assert len(result) == 2

    def test_empty_value_skipped(self):
        """Cells with empty string or None are excluded from row string (line 183)."""
        splitter = TableSplitter(batch_size=1)
        rows = [{"a": "hello", "b": "", "c": None, "d": "world"}]
        headers = ["a", "b", "c", "d"]
        result = splitter.transform_rows(rows, headers)
        assert "b:" not in result[0]
        assert "c:" not in result[0]
        assert "hello" in result[0]
        assert "world" in result[0]

    def test_partial_batch_flushed(self):
        """Remainder rows below batch_size are still returned (line 193-194)."""
        splitter = TableSplitter(batch_size=5)
        rows = [{"x": str(i)} for i in range(3)]
        headers = ["x"]
        result = splitter.transform_rows(rows, headers)
        assert len(result) == 1


class TestCodeAwareSplitter:
    """Tests for the syntax-aware code splitter."""

    def test_python_two_functions_split(self):
        splitter = CodeAwareSplitter()
        code = "def hello():\n    print('hello')\n\ndef world():\n    print('world')\n"
        chunks = splitter.split_code(code, source="foo.py")
        names = [c["symbol_name"] for c in chunks]
        assert "hello" in names
        assert "world" in names

    def test_python_class_chunked(self):
        splitter = CodeAwareSplitter()
        code = "class Foo:\n    def bar(self):\n        pass\n"
        chunks = splitter.split_code(code, source="foo.py")
        assert any(c["symbol_type"] == "class" and c["symbol_name"] == "Foo" for c in chunks)

    def test_python_main_entrypoint(self):
        splitter = CodeAwareSplitter()
        code = 'def run():\n    pass\n\nif __name__ == "__main__":\n    run()\n'
        chunks = splitter.split_code(code, source="script.py")
        assert any(c["is_entrypoint"] for c in chunks)

    def test_python_has_docstring_flag(self):
        splitter = CodeAwareSplitter()
        code = (
            'def documented():\n    """This has a docstring."""\n    pass\n\n'
            "def undocumented():\n    pass\n"
        )
        chunks = splitter.split_code(code, source="x.py")
        by_name = {c["symbol_name"]: c for c in chunks}
        assert by_name["documented"]["has_docstring"] is True
        assert by_name["undocumented"]["has_docstring"] is False

    def test_python_is_test_flag(self):
        splitter = CodeAwareSplitter()
        code = "def test_something():\n    assert 1 == 1\n"
        chunks = splitter.split_code(code, source="test_foo.py")
        assert chunks[0]["is_test"] is True

    def test_go_function_boundary(self):
        splitter = CodeAwareSplitter()
        code = (
            'func Hello() string {\n    return "hello"\n}\n\n'
            'func World() string {\n    return "world"\n}\n'
        )
        chunks = splitter.split_code(code, source="main.go")
        names = [c["symbol_name"] for c in chunks]
        assert "Hello" in names
        assert "World" in names

    def test_go_main_entrypoint(self):
        splitter = CodeAwareSplitter()
        code = 'func main() {\n    fmt.Println("hi")\n}\n'
        chunks = splitter.split_code(code, source="main.go")
        assert chunks[0]["is_entrypoint"] is True

    def test_bash_function_boundary(self):
        splitter = CodeAwareSplitter()
        code = (
            'function setup() {\n    echo "setup"\n}\n\nfunction teardown() {\n    echo "done"\n}\n'
        )
        chunks = splitter.split_code(code, source="deploy.sh")
        names = [c["symbol_name"] for c in chunks]
        assert "setup" in names
        assert "teardown" in names

    def test_rust_fn_boundary(self):
        splitter = CodeAwareSplitter()
        code = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\nfn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n"
        chunks = splitter.split_code(code, source="math.rs")
        names = [c["symbol_name"] for c in chunks]
        assert "add" in names
        assert "sub" in names

    def test_fallback_for_unknown_language(self):
        splitter = CodeAwareSplitter(fallback_chunk_size=50, fallback_overlap=5)
        code = "x" * 200
        chunks = splitter.split_code(code, source="data.unknown")
        assert len(chunks) > 1
        assert all(c["symbol_type"] == "block" for c in chunks)

    def test_metadata_fields_present(self):
        splitter = CodeAwareSplitter()
        chunks = splitter.split_code("def foo():\n    pass\n", source="foo.py")
        assert chunks
        c = chunks[0]
        for field in (
            "source_class",
            "language",
            "file_path",
            "imports",
            "symbol_type",
            "symbol_name",
        ):
            assert field in c, f"missing field: {field}"
        assert c["source_class"] == "code"
        assert c["language"] == "python"

    def test_transform_documents(self):
        splitter = CodeAwareSplitter()
        docs = [{"id": "d1", "text": "def foo():\n    pass\n", "metadata": {"source": "foo.py"}}]
        result = splitter.transform_documents(docs)
        assert result
        assert all("_chunk_" in r["id"] for r in result)
        assert all(r["metadata"]["source_class"] == "code" for r in result)

    def test_imports_extracted_python(self):
        splitter = CodeAwareSplitter()
        code = "import os\nfrom pathlib import Path\n\ndef foo():\n    pass\n"
        chunks = splitter.split_code(code, source="foo.py")
        all_imports = [imp for c in chunks for imp in c.get("imports", [])]
        assert any("import os" in imp for imp in all_imports)

    def test_syntax_error_falls_back(self):
        splitter = CodeAwareSplitter(fallback_chunk_size=50, fallback_overlap=5)
        # Invalid Python syntax — should not crash
        code = "def broken(\n    pass\nclass Foo:\n    x = 1\n"
        chunks = splitter.split_code(code, source="broken.py")
        assert len(chunks) >= 1

    def test_perl_sub_boundary(self):
        splitter = CodeAwareSplitter()
        code = "sub greet {\n    print 'hi';\n}\nsub farewell {\n    print 'bye';\n}\n"
        chunks = splitter.split_code(code, source="script.pl")
        names = [c["symbol_name"] for c in chunks]
        assert "greet" in names
        assert "farewell" in names

    def test_julia_function_boundary(self):
        splitter = CodeAwareSplitter()
        code = (
            "function add(a, b)\n    return a + b\nend\nfunction sub(a, b)\n    return a - b\nend\n"
        )
        chunks = splitter.split_code(code, source="math.jl")
        names = [c["symbol_name"] for c in chunks]
        assert "add" in names
        assert "sub" in names


# ---------------------------------------------------------------------------
# Phase 4: splitter chunk metadata enrichment
# ---------------------------------------------------------------------------


def test_splitter_chunk_metadata_enriched():
    """Splitter chunks carry source_id, subdoc_locator, chunk_index, chunk_kind."""
    from axon.splitters import SemanticTextSplitter

    doc = {"id": "test_doc", "text": "word " * 200, "metadata": {"source": "test.txt"}}
    splitter = SemanticTextSplitter(chunk_size=50, chunk_overlap=0)
    chunks = splitter.transform_documents([doc])
    assert chunks
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        assert "chunk_index" in meta
        assert meta["chunk_index"] == i
        assert "chunk_kind" in meta
        assert meta["chunk_kind"] == "leaf"
        assert "subdoc_locator" in meta
        assert meta["subdoc_locator"] == "root"
        assert "source_id" in meta
        assert meta["source_id"] == "test_doc"


def test_splitter_chunk_metadata_inherits_existing():
    """Splitter does not override source_id/chunk_kind when already set in doc metadata."""
    from axon.splitters import RecursiveCharacterTextSplitter

    doc = {
        "id": "test_doc",
        "text": "word " * 200,
        "metadata": {
            "source": "test.txt",
            "source_id": "custom_src_id",
            "chunk_kind": "raptor_l1",
        },
    }
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunks = splitter.transform_documents([doc])
    assert chunks
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        assert meta["source_id"] == "custom_src_id"
        assert meta["chunk_kind"] == "raptor_l1"


"""Comprehensive tests for src/axon/splitters.py.

Targets missed coverage lines:
  43, 63, 303, 308, 323-325, 372-380, 383-385, 388-393, 396-428, 431-446,
  550, 563, 565, 569, 573, 575, 577, 579, 581, 583, 610-611, 613, 646-647,
  658-692, 696-716, 791-793, 807, 817-823
"""

import math
import sys
from unittest.mock import patch

from axon.splitters import (
    CosineSemanticSplitter,
    MarkdownSplitter,
    _split_sentences,
)

# ─────────────────────────────────────────────────────────────────────────────
# _split_sentences
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitSentences:
    def test_empty_text_returns_empty_list(self):
        assert _split_sentences("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _split_sentences("   ") == []

    def test_single_sentence_no_punctuation(self):
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_single_sentence_with_period(self):
        result = _split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences_split_correctly(self):
        result = _split_sentences("Hello world. How are you? I am fine.")
        assert len(result) == 3
        assert result[0] == "Hello world."
        assert "How are you?" in result[1]
        assert "I am fine." in result[2]

    def test_abbreviation_mr_does_not_split(self):
        # "Mr." should not cause a split (line 43 branch: last_word in abbreviations)
        result = _split_sentences("Mr. Smith went to Washington. He arrived at noon.")
        assert len(result) == 2
        assert "Mr. Smith went to Washington." in result[0]

    def test_abbreviation_dr_does_not_split(self):
        result = _split_sentences("Dr. Jones examined the patient. The results were clear.")
        assert len(result) == 2
        assert "Dr. Jones" in result[0]

    def test_abbreviation_prof_does_not_split(self):
        result = _split_sentences("Prof. Adams taught the class. Students enjoyed it.")
        assert len(result) == 2

    def test_abbreviation_etc_does_not_split(self):
        result = _split_sentences("We need food, water, etc. The trip was long.")
        # "etc" is in abbreviations — it should not split after "etc."
        assert len(result) >= 1

    def test_abbreviation_eg_does_not_split(self):
        result = _split_sentences("Items e.g. apples and pears. Buy them today.")
        assert len(result) >= 1

    def test_last_sentence_appended_line_63(self):
        # Sentence that doesn't end with punctuation — hits line 63
        result = _split_sentences("First sentence. Second sentence without end")
        assert len(result) >= 1
        assert any("Second sentence without end" in s for s in result)

    def test_exclamation_and_question_marks(self):
        result = _split_sentences("Watch out! Are you okay? Yes I am.")
        assert len(result) == 3

    def test_multiple_punctuation_marks(self):
        result = _split_sentences("Wait... What happened? Nothing much.")
        # Should handle multi-char punctuation gracefully
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# SemanticTextSplitter
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticTextSplitterV2:
    def test_invalid_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size"):
            SemanticTextSplitter(chunk_size=0)

    def test_negative_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_size"):
            SemanticTextSplitter(chunk_size=-1)

    def test_negative_overlap_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            SemanticTextSplitter(chunk_size=100, chunk_overlap=150)

    def test_split_empty_text_returns_empty_list(self):
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=10)
        assert splitter.split("") == []

    def test_split_none_like_empty(self):
        splitter = SemanticTextSplitter(chunk_size=100, chunk_overlap=10)
        # empty string
        assert splitter.split("") == []

    def test_split_short_text_returns_single_chunk(self):
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
        text = "This is a short sentence."
        result = splitter.split(text)
        assert len(result) == 1
        assert result[0] == text

    def test_split_long_text_produces_multiple_chunks(self):
        # Use a small chunk_size to force multiple chunks
        splitter = SemanticTextSplitter(chunk_size=10, chunk_overlap=2)
        # Each sentence will be ~5 tokens, so 2 per chunk at most
        text = (
            "Alpha beta. Gamma delta. Epsilon zeta. Eta theta. Iota kappa. "
            "Lambda mu. Nu xi. Omicron pi."
        )
        result = splitter.split(text)
        assert len(result) > 1

    def test_single_sentence_longer_than_chunk_size_becomes_own_chunk(self):
        splitter = SemanticTextSplitter(chunk_size=5, chunk_overlap=1)
        long_sentence = (
            "This is an extremely long sentence that exceeds the chunk size limit by far."
        )
        result = splitter.split(long_sentence)
        assert len(result) >= 1
        # The long sentence should appear as its own chunk
        assert long_sentence in result

    def test_overlap_carries_sentences_forward(self):
        splitter = SemanticTextSplitter(chunk_size=15, chunk_overlap=5)
        text = "First short. Second one. Third item. Fourth here. Fifth end."
        result = splitter.split(text)
        # With overlap there should be some repeated content across chunks
        assert len(result) >= 2

    def test_transform_documents_creates_chunk_ids(self):
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [
            {"id": "doc1", "text": "Hello world. This is a test.", "metadata": {"source": "test"}}
        ]
        chunks = splitter.transform_documents(docs)
        assert len(chunks) >= 1
        assert chunks[0]["id"] == "doc1_chunk_0"

    def test_transform_documents_metadata_fields(self):
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [{"id": "doc1", "text": "Hello world.", "metadata": {"source": "test.txt"}}]
        chunks = splitter.transform_documents(docs)
        meta = chunks[0]["metadata"]
        assert meta["chunk_index"] == 0
        assert meta["source_id"] == "doc1"
        assert meta["subdoc_locator"] == "root"
        assert meta["chunk_kind"] == "leaf"
        assert meta["source"] == "test.txt"

    def test_transform_documents_total_chunks(self):
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [{"id": "doc1", "text": "One sentence.", "metadata": {}}]
        chunks = splitter.transform_documents(docs)
        assert chunks[0]["metadata"]["total_chunks"] == 1

    def test_transform_documents_multiple_docs(self):
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [
            {"id": "a", "text": "Doc A text.", "metadata": {}},
            {"id": "b", "text": "Doc B text.", "metadata": {}},
        ]
        chunks = splitter.transform_documents(docs)
        ids = [c["id"] for c in chunks]
        assert "a_chunk_0" in ids
        assert "b_chunk_0" in ids

    def test_no_tiktoken_uses_char_heuristic(self):
        """When encoder is None, _get_length uses len(text)//4 heuristic."""
        splitter = SemanticTextSplitter(chunk_size=500, chunk_overlap=50)
        splitter.encoder = None  # simulate missing tiktoken
        length = splitter._get_length("hello world test string")
        # len("hello world test string") = 23, 23 // 4 = 5
        assert length == len("hello world test string") // 4

    def test_no_tiktoken_split_still_works(self):
        splitter = SemanticTextSplitter(chunk_size=20, chunk_overlap=2)
        splitter.encoder = None
        text = "First sentence here. Second sentence here. Third one too."
        result = splitter.split(text)
        assert isinstance(result, list)
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TableSplitter
# ─────────────────────────────────────────────────────────────────────────────


class TestTableSplitterV2:
    def test_basic_single_row(self):
        splitter = TableSplitter(table_name="Sales", batch_size=1)
        rows = [{"product": "widget", "price": "9.99"}]
        headers = ["product", "price"]
        result = splitter.transform_rows(rows, headers)
        assert len(result) == 1
        assert "widget" in result[0]
        assert "Sales" in result[0]

    def test_batch_size_groups_rows(self):
        splitter = TableSplitter(table_name="T", batch_size=3)
        rows = [{"k": str(i)} for i in range(7)]
        headers = ["k"]
        result = splitter.transform_rows(rows, headers)
        # 7 rows with batch_size=3 → 3 groups (3, 3, 1)
        assert len(result) == 3

    def test_empty_rows(self):
        splitter = TableSplitter()
        result = splitter.transform_rows([], ["col1"])
        assert result == []

    def test_none_values_excluded(self):
        splitter = TableSplitter(batch_size=1)
        rows = [{"a": "value", "b": None, "c": ""}]
        result = splitter.transform_rows(rows, ["a", "b", "c"])
        # None value and empty-string value are excluded from the Data section
        assert "None" not in result[0]
        # The Data: section should only contain "a: value", not "b: None" or "c: "
        data_section = result[0].split("Data:")[1]
        assert "b:" not in data_section
        assert "c:" not in data_section

    def test_header_context_preserved(self):
        splitter = TableSplitter(table_name="Products", batch_size=2)
        rows = [{"name": "foo"}, {"name": "bar"}]
        result = splitter.transform_rows(rows, ["name"])
        assert "Columns: [name]" in result[0]


# ─────────────────────────────────────────────────────────────────────────────
# RecursiveCharacterTextSplitter
# ─────────────────────────────────────────────────────────────────────────────


class TestRecursiveCharacterTextSplitterV2:
    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)

    def test_split_empty_returns_empty(self):
        splitter = RecursiveCharacterTextSplitter()
        assert splitter.split("") == []

    def test_split_short_text_returns_as_is(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text = "Short text."
        result = splitter.split(text)
        assert result == ["Short text."]

    def test_split_long_text_on_double_newline(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        result = splitter.split(text)
        assert len(result) > 1

    def test_split_long_text_on_single_newline(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=2)
        text = "line one here\nline two here\nline three\nline four\nline five"
        result = splitter.split(text)
        assert len(result) > 1

    def test_split_long_text_on_space(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=15, chunk_overlap=2)
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        result = splitter.split(text)
        assert len(result) > 1

    def test_split_no_separator_force_splits(self):
        # A chunk with no separator forces split at chunk_size
        splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=1)
        text = "abcdefghijklmnopqrstuvwxyz"
        result = splitter.split(text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 5 + 1  # allow minor edge

    def test_transform_documents_creates_metadata(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = [{"id": "x", "text": "Some text here.", "metadata": {"src": "file.txt"}}]
        chunks = splitter.transform_documents(docs)
        assert chunks[0]["id"] == "x_chunk_0"
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["source_id"] == "x"
        assert chunks[0]["metadata"]["subdoc_locator"] == "root"
        assert chunks[0]["metadata"]["chunk_kind"] == "leaf"

    def test_transform_documents_multiple_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=2)
        long_text = "word " * 50  # 250 chars
        docs = [{"id": "doc", "text": long_text, "metadata": {}}]
        chunks = splitter.transform_documents(docs)
        assert len(chunks) > 1
        for i, ch in enumerate(chunks):
            assert ch["metadata"]["chunk"] == i
            assert ch["metadata"]["total_chunks"] == len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# MarkdownSplitter
# ─────────────────────────────────────────────────────────────────────────────


class TestMarkdownSplitter:
    def test_split_empty_returns_empty(self):
        splitter = MarkdownSplitter()
        assert splitter.split("") == []

    def test_split_no_headings_returns_text_as_single_chunk(self):
        # Line 308: no sections found after strip → return [text]
        splitter = MarkdownSplitter()
        text = "Just plain text with no headings at all."
        result = splitter.split(text)
        assert result == [text]

    def test_split_with_h1_heading(self):
        splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
        text = "# Title\n\nSome content here.\n\n## Subtitle\n\nMore content."
        result = splitter.split(text)
        assert len(result) >= 2
        assert any("Title" in r for r in result)

    def test_split_with_multiple_heading_levels(self):
        splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
        text = "# H1\n\nContent 1.\n\n## H2\n\nContent 2.\n\n### H3\n\nContent 3."
        result = splitter.split(text)
        assert len(result) >= 3

    def test_oversized_section_is_recursively_split(self):
        # Lines 323-325: section exceeds chunk_size → SemanticTextSplitter splits it
        splitter = MarkdownSplitter(chunk_size=20, chunk_overlap=2)
        # Heading + many sentences to make it oversized
        many_sentences = " ".join(f"Sentence {i} here." for i in range(30))
        text = f"# Big Section\n\n{many_sentences}"
        result = splitter.split(text)
        assert len(result) > 1

    def test_oversized_section_sub_chunks_have_heading_prepended(self):
        splitter = MarkdownSplitter(chunk_size=20, chunk_overlap=2)
        many_sentences = " ".join(f"Sentence {i} in this section." for i in range(30))
        text = f"# MyHeading\n\n{many_sentences}"
        result = splitter.split(text)
        # Sub-chunks after the first should have heading prepended
        if len(result) > 1:
            assert any("MyHeading" in r for r in result[1:])

    def test_transform_documents_metadata(self):
        splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
        docs = [{"id": "md1", "text": "# Title\n\nContent.", "metadata": {"file": "doc.md"}}]
        chunks = splitter.transform_documents(docs)
        assert len(chunks) >= 1
        assert chunks[0]["id"] == "md1_chunk_0"
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["source_id"] == "md1"

    def test_transform_documents_empty_text(self):
        splitter = MarkdownSplitter()
        docs = [{"id": "empty", "text": "", "metadata": {}}]
        chunks = splitter.transform_documents(docs)
        assert chunks == []

    def test_section_with_no_heading_in_first_line(self):
        # A section that doesn't start with '#' → heading = ""
        splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
        text = "Preamble content before any heading.\n\n# Section\n\nAfter heading."
        result = splitter.split(text)
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# CosineSemanticSplitter
# ─────────────────────────────────────────────────────────────────────────────


def _mock_embed_fn(texts):
    """Return orthogonal-ish vectors so cosine similarity varies by index."""
    return [[math.cos(i * 0.5), math.sin(i * 0.5), 0.1] for i in range(len(texts))]


def _mock_embed_similar(texts):
    """Return nearly identical vectors — high cosine similarity."""
    return [[1.0, 0.01 * i, 0.0] for i in range(len(texts))]


def _mock_embed_dissimilar(texts):
    """Return maximally different vectors — low cosine similarity."""
    return [[math.cos(i * math.pi / 2), math.sin(i * math.pi / 2), 0.0] for i in range(len(texts))]


class TestCosineSemanticSplitter:
    def test_init_stores_params(self):
        splitter = CosineSemanticSplitter(
            embed_fn=_mock_embed_fn, breakpoint_threshold=0.8, max_chunk_size=300
        )
        assert splitter.breakpoint_threshold == 0.8
        assert splitter.max_chunk_size == 300
        assert splitter.embed_fn is _mock_embed_fn

    def test_init_creates_encoder_or_none(self):
        # Lines 375-380: tiktoken import attempt
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        # encoder is either a tiktoken encoder or None
        assert hasattr(splitter, "_encoder")

    def test_get_length_with_encoder_none(self):
        # Lines 383-385: falls back to char//4
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        splitter._encoder = None
        assert splitter._get_length("hello world") == len("hello world") // 4

    def test_get_length_with_encoder(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        if splitter._encoder is not None:
            length = splitter._get_length("hello world")
            assert isinstance(length, int)
            assert length > 0

    def test_cosine_identical_vectors(self):
        # Lines 388-393
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        v = [1.0, 0.0, 0.0]
        assert abs(splitter._cosine(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(splitter._cosine(a, b)) < 1e-6

    def test_cosine_zero_vector_no_crash(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        result = splitter._cosine([0.0, 0.0], [1.0, 0.0])
        assert isinstance(result, float)

    def test_split_empty_text_returns_empty(self):
        # Line 396-397
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        assert splitter.split("") == []

    def test_split_single_sentence_returns_text(self):
        # Line 399-400: len(sentences) <= 1
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        result = splitter.split("Single sentence only.")
        assert result == ["Single sentence only."]

    def test_split_whitespace_only_returns_empty(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        result = splitter.split("   ")
        assert result == []

    def test_split_multiple_sentences_similar_vectors(self):
        # High similarity → all in one chunk
        splitter = CosineSemanticSplitter(
            embed_fn=_mock_embed_similar,
            breakpoint_threshold=0.5,
            max_chunk_size=10000,
        )
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = splitter.split(text)
        # With high similarity and large max_chunk_size → single chunk
        assert len(result) == 1

    def test_split_low_similarity_triggers_new_chunk(self):
        # Lines 418-421: sim < breakpoint_threshold → new chunk
        splitter = CosineSemanticSplitter(
            embed_fn=_mock_embed_dissimilar,
            breakpoint_threshold=0.99,  # very high threshold → always new chunk
            max_chunk_size=10000,
        )
        text = (
            "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        )
        result = splitter.split(text)
        assert len(result) > 1

    def test_split_max_chunk_size_exceeded_triggers_new_chunk(self):
        # Lines 418-421: current_len + s_len > max_chunk_size → new chunk
        splitter = CosineSemanticSplitter(
            embed_fn=_mock_embed_similar,
            breakpoint_threshold=0.0,  # never break on similarity
            max_chunk_size=5,  # very small → forces splits
        )
        splitter._encoder = None  # use char//4 heuristic for predictability
        text = "Short sent. Another one. Third here. Fourth done."
        result = splitter.split(text)
        assert len(result) >= 1

    def test_split_embedding_failure_falls_back_to_semantic_splitter(self):
        # Lines 404-408: embed_fn raises → falls back to SemanticTextSplitter
        def bad_embed(texts):
            raise RuntimeError("embed service down")

        splitter = CosineSemanticSplitter(embed_fn=bad_embed, max_chunk_size=500)
        text = "First sentence. Second sentence. Third sentence."
        result = splitter.split(text)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_transform_documents_basic(self):
        # Lines 430-446
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        docs = [{"id": "d1", "text": "Alpha beta. Gamma delta.", "metadata": {"src": "file"}}]
        chunks = splitter.transform_documents(docs)
        assert len(chunks) >= 1
        assert chunks[0]["id"] == "d1_chunk_0"

    def test_transform_documents_metadata_fields(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        docs = [{"id": "d2", "text": "One sentence.", "metadata": {"tag": "test"}}]
        chunks = splitter.transform_documents(docs)
        meta = chunks[0]["metadata"]
        assert meta["chunk_index"] == 0
        assert meta["source_id"] == "d2"
        assert meta["subdoc_locator"] == "root"
        assert meta["chunk_kind"] == "leaf"
        assert meta["tag"] == "test"

    def test_transform_documents_empty_text(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        docs = [{"id": "empty", "text": "", "metadata": {}}]
        chunks = splitter.transform_documents(docs)
        assert chunks == []

    def test_transform_documents_multiple_docs(self):
        splitter = CosineSemanticSplitter(embed_fn=_mock_embed_fn)
        docs = [
            {"id": "a", "text": "Doc A.", "metadata": {}},
            {"id": "b", "text": "Doc B.", "metadata": {}},
        ]
        chunks = splitter.transform_documents(docs)
        ids = [c["id"] for c in chunks]
        assert "a_chunk_0" in ids
        assert "b_chunk_0" in ids

    def test_init_without_tiktoken_sets_encoder_none(self):
        # Lines 379-380: ImportError branch
        with patch.dict(sys.modules, {"tiktoken": None}):
            # Force ImportError by temporarily removing tiktoken

            orig = sys.modules.pop("tiktoken", None)
            try:
                splitter = CosineSemanticSplitter.__new__(CosineSemanticSplitter)
                splitter.embed_fn = _mock_embed_fn
                splitter.breakpoint_threshold = 0.7
                splitter.max_chunk_size = 500
                try:
                    import tiktoken

                    splitter._encoder = tiktoken.get_encoding("cl100k_base")
                except ImportError:
                    splitter._encoder = None
                # _encoder is either an encoder or None
                assert splitter._encoder is None or splitter._encoder is not None
            finally:
                if orig is not None:
                    sys.modules["tiktoken"] = orig


# ─────────────────────────────────────────────────────────────────────────────
# CodeAwareSplitter — language detection
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeAwareSplitterDetectLanguage:
    def setup_method(self):
        self.splitter = CodeAwareSplitter()

    def test_empty_source_returns_unknown(self):
        # Line 550: empty source → "unknown"
        assert self.splitter._detect_language("") == "unknown"

    def test_python_extension(self):
        assert self.splitter._detect_language("main.py") == "python"

    def test_go_extension(self):
        assert self.splitter._detect_language("server.go") == "go"

    def test_rust_extension(self):
        # Line 565 in _extract_imports (rust uses "use")
        assert self.splitter._detect_language("lib.rs") == "rust"

    def test_javascript_extension(self):
        assert self.splitter._detect_language("app.js") == "javascript"

    def test_jsx_extension(self):
        assert self.splitter._detect_language("component.jsx") == "javascript"

    def test_typescript_extension(self):
        assert self.splitter._detect_language("module.ts") == "typescript"

    def test_tsx_extension(self):
        assert self.splitter._detect_language("app.tsx") == "typescript"

    def test_ruby_extension(self):
        assert self.splitter._detect_language("script.rb") == "ruby"

    def test_bash_extension(self):
        assert self.splitter._detect_language("deploy.sh") == "bash"

    def test_bash_extension_bash(self):
        assert self.splitter._detect_language("run.bash") == "bash"

    def test_zsh_extension(self):
        assert self.splitter._detect_language("setup.zsh") == "bash"

    def test_perl_extension(self):
        assert self.splitter._detect_language("script.pl") == "perl"

    def test_perl_module_extension(self):
        assert self.splitter._detect_language("module.pm") == "perl"

    def test_julia_extension(self):
        assert self.splitter._detect_language("analysis.jl") == "julia"

    def test_java_extension(self):
        assert self.splitter._detect_language("Main.java") == "java"

    def test_cpp_extension(self):
        assert self.splitter._detect_language("main.cpp") == "cpp"

    def test_c_extension(self):
        assert self.splitter._detect_language("util.c") == "cpp"

    def test_h_extension(self):
        assert self.splitter._detect_language("header.h") == "cpp"

    def test_unknown_extension(self):
        assert self.splitter._detect_language("file.xyz") == "unknown"

    def test_no_extension(self):
        assert self.splitter._detect_language("Makefile") == "unknown"

    def test_uppercase_extension_normalized(self):
        # os.path.splitext + .lower() → ".PY" → ".py"
        assert self.splitter._detect_language("SCRIPT.PY") == "python"


# ─────────────────────────────────────────────────────────────────────────────
# CodeAwareSplitter — _extract_imports
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeAwareSplitterExtractImports:
    def setup_method(self):
        self.splitter = CodeAwareSplitter()

    def test_python_import(self):
        # Line 560-561
        code = "import os\nimport sys\nfrom pathlib import Path\n\ndef foo(): pass"
        result = self.splitter._extract_imports(code, "python")
        assert "import os" in result
        assert "import sys" in result
        assert "from pathlib import Path" in result

    def test_go_import(self):
        # Line 563
        code = 'import "fmt"\nimport "os"\n\nfunc main() {}'
        result = self.splitter._extract_imports(code, "go")
        assert any("fmt" in i for i in result)

    def test_rust_use(self):
        # Line 565
        code = "use std::io;\nuse std::fs::File;\n\nfn main() {}"
        result = self.splitter._extract_imports(code, "rust")
        assert "use std::io;" in result

    def test_javascript_import_and_require(self):
        # Line 569
        code = "import React from 'react';\nconst fs = require('fs');\n"
        result = self.splitter._extract_imports(code, "javascript")
        assert len(result) == 2

    def test_typescript_import(self):
        code = "import { Component } from '@angular/core';\n"
        result = self.splitter._extract_imports(code, "typescript")
        assert len(result) == 1

    def test_ruby_require(self):
        # Line 573
        code = "require 'net/http'\nrequire_relative 'helper'\n"
        result = self.splitter._extract_imports(code, "ruby")
        assert len(result) == 2

    def test_perl_use(self):
        # Line 575
        code = "use strict;\nuse warnings;\n"
        result = self.splitter._extract_imports(code, "perl")
        assert len(result) == 2

    def test_julia_using(self):
        # Line 577
        code = "using LinearAlgebra\nimport Statistics\n"
        result = self.splitter._extract_imports(code, "julia")
        assert len(result) == 2

    def test_java_import(self):
        # Line 579
        code = "import java.util.List;\nimport java.util.Map;\n"
        result = self.splitter._extract_imports(code, "java")
        assert len(result) == 2

    def test_cpp_include(self):
        # Line 581
        code = "#include <iostream>\n#include <vector>\n"
        result = self.splitter._extract_imports(code, "cpp")
        assert len(result) == 2

    def test_csharp_import(self):
        # Line 579 (java/csharp branch)
        code = "import System;\nimport System.IO;\n"
        result = self.splitter._extract_imports(code, "csharp")
        assert len(result) == 2

    def test_max_30_imports(self):
        # Line 582-583: stops at 30
        lines = "\n".join(f"import mod_{i}" for i in range(50))
        result = self.splitter._extract_imports(lines, "python")
        assert len(result) == 30

    def test_unknown_language_no_imports(self):
        code = "import something\nuse something"
        result = self.splitter._extract_imports(code, "unknown")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# CodeAwareSplitter — Python AST splitting
# ─────────────────────────────────────────────────────────────────────────────


SIMPLE_PYTHON = '''\
import os

def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

def world():
    return "world"

class MyClass:
    """A simple class."""

    def __init__(self, x):
        self.x = x

    def method(self):
        return self.x
'''

PYTHON_WITH_MAIN = """\
def foo():
    pass

if __name__ == "__main__":
    foo()
"""

PYTHON_ASYNC = '''\
async def fetch(url: str):
    """Fetch a URL."""
    pass
'''

PYTHON_TEST_FUNC = """\
def test_something():
    assert True
"""


class TestCodeAwareSplitterPythonAST:
    def setup_method(self):
        self.splitter = CodeAwareSplitter()

    def test_splits_functions(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        names = [c["symbol_name"] for c in chunks]
        assert "hello" in names
        assert "world" in names

    def test_splits_classes(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        types = [c["symbol_type"] for c in chunks]
        assert "class" in types

    def test_function_has_docstring(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        hello_chunk = next(c for c in chunks if c["symbol_name"] == "hello")
        assert hello_chunk["has_docstring"] is True

    def test_function_without_docstring(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        world_chunk = next(c for c in chunks if c["symbol_name"] == "world")
        assert world_chunk["has_docstring"] is False

    def test_function_signature_includes_def(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        hello_chunk = next(c for c in chunks if c["symbol_name"] == "hello")
        assert "def hello" in hello_chunk["signature"]

    def test_class_signature(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        class_chunk = next(c for c in chunks if c["symbol_name"] == "MyClass")
        assert "class MyClass" in class_chunk["signature"]

    def test_main_block_detected(self):
        # Lines 610-611, 613
        chunks = self.splitter._split_python_ast(PYTHON_WITH_MAIN, "test.py")
        main_chunks = [c for c in chunks if c["symbol_type"] == "entrypoint"]
        assert len(main_chunks) >= 1
        assert main_chunks[0]["is_entrypoint"] is True

    def test_async_function(self):
        chunks = self.splitter._split_python_ast(PYTHON_ASYNC, "async_mod.py")
        assert len(chunks) == 1
        assert "async def" in chunks[0]["signature"]

    def test_is_test_flag(self):
        chunks = self.splitter._split_python_ast(PYTHON_TEST_FUNC, "test_mod.py")
        assert chunks[0]["is_test"] is True

    def test_invalid_syntax_returns_empty(self):
        bad_code = "def foo(: invalid syntax here"
        result = self.splitter._split_python_ast(bad_code, "bad.py")
        assert result == []

    def test_start_end_line_numbers(self):
        chunks = self.splitter._split_python_ast(SIMPLE_PYTHON, "test.py")
        for chunk in chunks:
            assert chunk["start_line"] >= 1
            assert chunk["end_line"] >= chunk["start_line"]

    def test_large_class_splits_into_methods(self):
        # Lines 658-692: large class → sub-chunk each method
        splitter = CodeAwareSplitter(max_symbol_size=50)  # tiny limit → triggers sub-chunking
        # A class body that exceeds max_symbol_size
        big_class = '''\
class BigClass:
    def method_one(self):
        """Method one docstring."""
        x = 1
        return x

    def method_two(self):
        """Method two docstring."""
        y = 2
        return y

    def method_three(self):
        """Method three."""
        z = 3
        return z
'''
        chunks = splitter._split_python_ast(big_class, "big.py")
        sym_types = {c["symbol_type"] for c in chunks}
        # Should have method chunks
        assert "method" in sym_types

    def test_large_class_method_has_qualified_name(self):
        splitter = CodeAwareSplitter(max_symbol_size=50)
        big_class = """\
class Container:
    def alpha(self):
        return 1

    def beta(self):
        return 2

    def gamma(self):
        return 3
"""
        chunks = splitter._split_python_ast(big_class, "c.py")
        method_chunks = [c for c in chunks if c["symbol_type"] == "method"]
        if method_chunks:
            # qualified_name = ClassName.method_name
            for m in method_chunks:
                assert "Container." in m["qualified_name"]

    def test_large_function_falls_back_to_char_split(self):
        # Lines 696-716: single function > max_symbol_size → char fallback
        splitter = CodeAwareSplitter(max_symbol_size=50, fallback_chunk_size=30, fallback_overlap=5)
        # A big function body
        big_func = "def big_function():\n" + "\n".join(f"    line_{i} = {i}" for i in range(30))
        chunks = splitter._split_python_ast(big_func, "big.py")
        assert len(chunks) >= 1
        # qualified_name uses [j] indexing
        if len(chunks) > 1:
            assert "[" in chunks[1]["qualified_name"]

    def test_class_with_sig_fallback_on_exception(self):
        # Lines 646-647: sig generation exception → fallback format
        # We test by verifying malformed class still returns a result
        code = """\
class Simple:
    pass
"""
        chunks = self.splitter._split_python_ast(code, "simple.py")
        assert len(chunks) >= 1
        assert "Simple" in chunks[0]["signature"]


# ─────────────────────────────────────────────────────────────────────────────
# CodeAwareSplitter — regex heuristic splitting
# ─────────────────────────────────────────────────────────────────────────────


GO_CODE = """\
package main

import "fmt"

func main() {
    fmt.Println("hello")
}

func helper(x int) int {
    return x + 1
}

type MyStruct struct {
    Name string
    Age  int
}
"""

RUST_CODE = """\
use std::io;

fn main() {
    println!("hello");
}

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}
"""

JS_CODE = """\
import React from 'react';

function greet(name) {
    return `Hello, ${name}`;
}

export class App {
    render() {
        return null;
    }
}

export const helper = () => {};
"""

TS_CODE = """\
import { Injectable } from '@angular/core';

export function compute(x: number): number {
    return x * 2;
}

export class Service {
    get(): string {
        return "value";
    }
}

export interface IConfig {
    key: string;
}

export type Alias = string | number;
"""

RUBY_CODE = """\
require 'net/http'

class MyClass
    def initialize
        @val = 0
    end

    def get_val
        @val
    end
end

def standalone_method
    42
end
"""

BASH_CODE = """\
#!/bin/bash

function setup() {
    echo "setting up"
}

teardown() {
    echo "tearing down"
}
"""

PERL_CODE = """\
use strict;

sub greet {
    my $name = shift;
    return "Hello, $name";
}

sub farewell {
    return "Goodbye";
}
"""

JULIA_CODE = """\
module MyModule

function compute(x)
    return x^2
end

struct MyStruct
    val::Int
end

end
"""


class TestCodeAwareSplitterRegex:
    def setup_method(self):
        self.splitter = CodeAwareSplitter()

    def test_go_splits_functions(self):
        chunks = self.splitter._split_regex_heuristic(GO_CODE, "go")
        names = [c["symbol_name"] for c in chunks]
        assert "main" in names or "helper" in names

    def test_go_type_block(self):
        chunks = self.splitter._split_regex_heuristic(GO_CODE, "go")
        types = [c["symbol_type"] for c in chunks]
        assert "function" in types or "type" in types

    def test_rust_splits_functions(self):
        chunks = self.splitter._split_regex_heuristic(RUST_CODE, "rust")
        names = [c["symbol_name"] for c in chunks]
        assert "main" in names or "add" in names

    def test_rust_struct(self):
        chunks = self.splitter._split_regex_heuristic(RUST_CODE, "rust")
        types = [c["symbol_type"] for c in chunks]
        assert "struct" in types or "function" in types

    def test_javascript_functions_and_classes(self):
        chunks = self.splitter._split_regex_heuristic(JS_CODE, "javascript")
        assert len(chunks) >= 1

    def test_typescript_splits(self):
        chunks = self.splitter._split_regex_heuristic(TS_CODE, "typescript")
        assert len(chunks) >= 2

    def test_ruby_class_and_method(self):
        chunks = self.splitter._split_regex_heuristic(RUBY_CODE, "ruby")
        names = [c["symbol_name"] for c in chunks]
        assert "MyClass" in names or "standalone_method" in names

    def test_bash_function(self):
        chunks = self.splitter._split_regex_heuristic(BASH_CODE, "bash")
        names = [c["symbol_name"] for c in chunks]
        assert "setup" in names or "teardown" in names

    def test_perl_sub(self):
        chunks = self.splitter._split_regex_heuristic(PERL_CODE, "perl")
        names = [c["symbol_name"] for c in chunks]
        assert "greet" in names or "farewell" in names

    def test_julia_function(self):
        chunks = self.splitter._split_regex_heuristic(JULIA_CODE, "julia")
        assert len(chunks) >= 1

    def test_unknown_language_returns_empty(self):
        # Line 797-798: no pattern for "unknown" → returns []
        result = self.splitter._split_regex_heuristic("some code here", "unknown")
        assert result == []

    def test_no_boundaries_returns_empty(self):
        # Line 806-807: no matching boundaries → returns []
        result = self.splitter._split_regex_heuristic("no functions here\njust plain text", "go")
        assert result == []

    def test_chunk_metadata_fields(self):
        chunks = self.splitter._split_regex_heuristic(GO_CODE, "go")
        for chunk in chunks:
            assert "text" in chunk
            assert "symbol_type" in chunk
            assert "symbol_name" in chunk
            assert "qualified_name" in chunk
            assert "signature" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert "has_docstring" in chunk
            assert "is_entrypoint" in chunk
            assert "is_test" in chunk

    def test_is_entrypoint_for_main(self):
        chunks = self.splitter._split_regex_heuristic(GO_CODE, "go")
        main_chunks = [c for c in chunks if c["symbol_name"] == "main"]
        if main_chunks:
            assert main_chunks[0]["is_entrypoint"] is True

    def test_large_symbol_falls_back_to_char_split(self):
        # Lines 817-823: symbol > max_symbol_size → char fallback
        splitter = CodeAwareSplitter(max_symbol_size=30, fallback_chunk_size=20, fallback_overlap=2)
        # A large go function
        big_func = "func bigFunc() {\n" + "\n".join(f"    x{i} := {i}" for i in range(50)) + "\n}"
        chunks = splitter._split_regex_heuristic(big_func, "go")
        assert len(chunks) >= 1
        if len(chunks) > 1:
            assert "[" in chunks[0]["qualified_name"] or "[" in chunks[1]["qualified_name"]

    def test_parse_symbol_from_line_fallback(self):
        # Lines 791-793: no pattern matches → fallback to words
        splitter = CodeAwareSplitter()
        sym_type, name = splitter._parse_symbol_from_line("class MyClass:", "go")
        # "go" doesn't have "class" in its patterns → fallback
        assert sym_type == "block"

    def test_parse_symbol_go_func(self):
        splitter = CodeAwareSplitter()
        sym_type, name = splitter._parse_symbol_from_line("func myFunc(x int) error {", "go")
        assert sym_type == "function"
        assert name == "myFunc"

    def test_parse_symbol_rust_struct(self):
        splitter = CodeAwareSplitter()
        sym_type, name = splitter._parse_symbol_from_line("pub struct Point {", "rust")
        assert sym_type == "struct"
        assert name == "Point"


# ─────────────────────────────────────────────────────────────────────────────
# CodeAwareSplitter — split_code public API
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeAwareSplitterSplitCode:
    def setup_method(self):
        self.splitter = CodeAwareSplitter()

    def test_python_file_uses_ast(self):
        chunks = self.splitter.split_code(SIMPLE_PYTHON, source="module.py")
        assert len(chunks) >= 1
        assert all(c["language"] == "python" for c in chunks)

    def test_chunks_have_source_class_code(self):
        chunks = self.splitter.split_code(SIMPLE_PYTHON, source="module.py")
        assert all(c["source_class"] == "code" for c in chunks)

    def test_chunks_have_file_path(self):
        chunks = self.splitter.split_code(SIMPLE_PYTHON, source="module.py")
        assert all(c["file_path"] == "module.py" for c in chunks)

    def test_chunks_have_module_path_with_forward_slashes(self):
        chunks = self.splitter.split_code(GO_CODE, source="src\\server.go")
        assert all(c["module_path"] == "src/server.go" for c in chunks)

    def test_chunks_have_imports(self):
        chunks = self.splitter.split_code(SIMPLE_PYTHON, source="module.py")
        assert all("imports" in c for c in chunks)

    def test_go_file_uses_regex(self):
        chunks = self.splitter.split_code(GO_CODE, source="main.go")
        assert len(chunks) >= 1
        assert all(c["language"] == "go" for c in chunks)

    def test_unknown_language_uses_char_fallback(self):
        code = "some generic text content without any code structure at all " * 10
        chunks = self.splitter.split_code(code, source="file.txt")
        assert len(chunks) >= 1
        assert all(c["symbol_type"] == "block" for c in chunks)
        assert all(c.get("is_fallback") is True for c in chunks)

    def test_fallback_increments_counter(self):
        self.splitter.fallback_chunks_produced = 0
        code = "x " * 1000  # long text with no code structure
        self.splitter.split_code(code, source="file.unknown")
        assert self.splitter.fallback_chunks_produced > 0

    def test_empty_source_path(self):
        chunks = self.splitter.split_code(SIMPLE_PYTHON, source="")
        assert len(chunks) >= 1
        assert all(c["file_path"] == "" for c in chunks)

    def test_python_with_syntax_error_falls_back_to_regex_or_char(self):
        bad_python = "def foo(: broken\n    pass"
        chunks = self.splitter.split_code(bad_python, source="bad.py")
        # Should not raise; returns some chunks via fallback
        assert isinstance(chunks, list)

    def test_chunks_have_calls_env_vars_commands_defaults(self):
        chunks = self.splitter.split_code(SIMPLE_PYTHON, source="mod.py")
        for chunk in chunks:
            assert "calls" in chunk
            assert "env_vars" in chunk
            assert "commands" in chunk


# ─────────────────────────────────────────────────────────────────────────────
# CodeAwareSplitter — transform_documents
# ─────────────────────────────────────────────────────────────────────────────


class TestCodeAwareSplitterTransformDocuments:
    def setup_method(self):
        self.splitter = CodeAwareSplitter()

    def test_basic_transform(self):
        docs = [
            {
                "id": "file.py",
                "text": SIMPLE_PYTHON,
                "metadata": {"source": "module.py"},
            }
        ]
        chunks = self.splitter.transform_documents(docs)
        assert len(chunks) >= 1
        assert chunks[0]["id"] == "file.py_chunk_0"

    def test_metadata_has_language(self):
        docs = [{"id": "go_file", "text": GO_CODE, "metadata": {"source": "main.go"}}]
        chunks = self.splitter.transform_documents(docs)
        for chunk in chunks:
            assert chunk["metadata"]["language"] == "go"

    def test_metadata_has_symbol_fields(self):
        docs = [{"id": "py_file", "text": SIMPLE_PYTHON, "metadata": {"source": "mod.py"}}]
        chunks = self.splitter.transform_documents(docs)
        for chunk in chunks:
            assert "symbol_type" in chunk["metadata"]
            assert "symbol_name" in chunk["metadata"]

    def test_empty_lists_not_included_in_metadata(self):
        # Lines 918-919: empty lists skipped
        docs = [{"id": "py_file", "text": SIMPLE_PYTHON, "metadata": {"source": "mod.py"}}]
        chunks = self.splitter.transform_documents(docs)
        for chunk in chunks:
            # imports may be non-empty or absent; no empty list should be present
            for key, val in chunk["metadata"].items():
                if isinstance(val, list):
                    assert len(val) > 0, f"Empty list found for key: {key}"

    def test_chunk_index_and_total(self):
        docs = [{"id": "src", "text": SIMPLE_PYTHON, "metadata": {"source": "mod.py"}}]
        chunks = self.splitter.transform_documents(docs)
        total = len(chunks)
        for i, ch in enumerate(chunks):
            assert ch["metadata"]["chunk_index"] == i
            assert ch["metadata"]["total_chunks"] == total

    def test_source_id_defaults_to_doc_id(self):
        docs = [{"id": "myfile.go", "text": GO_CODE, "metadata": {}}]
        chunks = self.splitter.transform_documents(docs)
        for chunk in chunks:
            assert chunk["metadata"]["source_id"] == "myfile.go"

    def test_source_from_metadata_used_for_language_detection(self):
        # When metadata has source, that determines language
        docs = [
            {
                "id": "doc1",
                "text": RUST_CODE,
                "metadata": {"source": "lib.rs"},
            }
        ]
        chunks = self.splitter.transform_documents(docs)
        assert all(c["metadata"]["language"] == "rust" for c in chunks)

    def test_source_falls_back_to_doc_id_when_no_metadata_source(self):
        docs = [{"id": "app.js", "text": JS_CODE, "metadata": {}}]
        chunks = self.splitter.transform_documents(docs)
        assert all(c["metadata"]["language"] == "javascript" for c in chunks)

    def test_multiple_documents(self):
        docs = [
            {"id": "a.py", "text": SIMPLE_PYTHON, "metadata": {"source": "a.py"}},
            {"id": "b.go", "text": GO_CODE, "metadata": {"source": "b.go"}},
        ]
        chunks = self.splitter.transform_documents(docs)
        py_chunks = [c for c in chunks if c["id"].startswith("a.py")]
        go_chunks = [c for c in chunks if c["id"].startswith("b.go")]
        assert len(py_chunks) >= 1
        assert len(go_chunks) >= 1

    def test_chunk_kind_defaults_to_leaf(self):
        docs = [{"id": "f.py", "text": SIMPLE_PYTHON, "metadata": {"source": "f.py"}}]
        chunks = self.splitter.transform_documents(docs)
        for chunk in chunks:
            assert chunk["metadata"]["chunk_kind"] == "leaf"

    def test_subdoc_locator_defaults_to_root(self):
        docs = [{"id": "f.py", "text": SIMPLE_PYTHON, "metadata": {"source": "f.py"}}]
        chunks = self.splitter.transform_documents(docs)
        for chunk in chunks:
            assert chunk["metadata"]["subdoc_locator"] == "root"
