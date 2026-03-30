"""Extra tests for axon.splitters to increase coverage."""
import pytest
from axon.splitters import RecursiveCharacterTextSplitter, CodeAwareSplitter, MarkdownSplitter

def test_recursive_character_splitter_basic():
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    text = "This is a long string that needs to be split into chunks."
    chunks = splitter.split(text)
    assert len(chunks) > 1

def test_markdown_splitter_basic():
    splitter = MarkdownSplitter(chunk_size=1000, chunk_overlap=100)
    text = "# Header\n\nContent here.\n\n## Subheader\n\nMore content."
    chunks = splitter.split(text)
    assert len(chunks) >= 1

def test_code_aware_splitter_python():
    splitter = CodeAwareSplitter(max_symbol_size=50)
    code = """
def hello():
    print("world")

class Test:
    def method(self):
        pass
"""
    chunks = splitter.split_code(code, source="test.py")
    assert len(chunks) >= 1
    assert any("def hello" in c["text"] for c in chunks)

def test_code_aware_splitter_js():
    splitter = CodeAwareSplitter(max_symbol_size=50)
    code = """
function hello() {
    console.log("world");
}

class Test {
    method() {
    }
}
"""
    chunks = splitter.split_code(code, source="test.js")
    assert len(chunks) >= 1

def test_code_aware_splitter_fallback():
    # Test with unknown language
    splitter = CodeAwareSplitter(max_symbol_size=50)
    text = "Some random text that should just be split normally."
    chunks = splitter.split_code(text, source="test.txt")
    assert len(chunks) >= 1

