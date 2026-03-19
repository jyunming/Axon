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
