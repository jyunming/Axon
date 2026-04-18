import builtins
from unittest.mock import patch

from axon.splitters import (
    CodeAwareSplitter,
    CosineSemanticSplitter,
    RecursiveCharacterTextSplitter,
    _split_sentences,
)


def test_split_sentences_no_trailing_punct():
    text = "Hello world. This is a test"
    sentences = _split_sentences(text)
    assert "This is a test" in sentences


def test_recursive_splitter_empty_sep():
    # Force empty separator into list
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    splitter.separators = ["\n", " ", ""]
    # This should trigger the 'if sep == "": continue' line
    chunks = splitter.split("a b c d e f g")
    assert len(chunks) > 0


def test_cosine_splitter_no_tiktoken():
    real_import = builtins.__import__

    def my_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=my_import):
        splitter = CosineSemanticSplitter(embed_fn=lambda x: [[0.1] * 1536] * len(x))
        assert splitter._encoder is None
        assert splitter._get_length("test") == 1  # "test" len 4 // 4 = 1


def test_code_splitter_ast_unparse_failure():
    splitter = CodeAwareSplitter()
    code = """
def my_func(a, b):
    \"\"\"Docstring\"\"\"
    return a + b

class MyClass:
    def method(self):
        pass
"""
    # Mock ast.unparse to raise an exception
    with patch("ast.unparse", side_effect=Exception("unparse error")):
        chunks = splitter.split_code(code, source="test.py")
        assert len(chunks) > 0
        # Should fallback to def my_func(...) signature
        assert any("def my_func(...)" in c["signature"] for c in chunks)


def test_code_splitter_main_block_unparse_failure():
    splitter = CodeAwareSplitter()
    code = """
if __name__ == "__main__":
    print("hello")
"""
    # Trigger Exception in if __name__ == "__main__" block
    with patch("ast.unparse", side_effect=Exception("unparse error")):
        chunks = splitter.split_code(code, source="test.py")
        # Should have 1 chunk (fallback)
        assert len(chunks) == 1
        assert chunks[0]["symbol_type"] in ("text", "block")
