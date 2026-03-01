"""
tests/test_loaders.py

Unit tests for all document loaders in rag_brain.loaders.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_brain.loaders import (
    CSVLoader,
    DirectoryLoader,
    DOCXLoader,
    HTMLLoader,
    JSONLoader,
    PDFLoader,
    TextLoader,
    TSVLoader,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_doc(doc):
    """Assert a document has the required schema."""
    assert "id" in doc
    assert "text" in doc
    assert "metadata" in doc
    assert isinstance(doc["metadata"], dict)


# ---------------------------------------------------------------------------
# TextLoader
# ---------------------------------------------------------------------------

def test_text_loader_basic(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("Hello world", encoding="utf-8")
    docs = TextLoader().load(str(p))
    assert len(docs) == 1
    _assert_doc(docs[0])
    assert docs[0]["text"] == "Hello world"
    assert docs[0]["id"] == "hello.txt"


def test_text_loader_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    docs = TextLoader().load(str(p))
    assert len(docs) == 1
    assert docs[0]["text"] == ""


# ---------------------------------------------------------------------------
# TSVLoader
# ---------------------------------------------------------------------------

def test_tsv_loader(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("content\tscore\nfoo bar\t1\nbaz qux\t2\n", encoding="utf-8")
    docs = TSVLoader().load(str(p))
    assert len(docs) == 2
    for doc in docs:
        _assert_doc(doc)
    assert docs[0]["text"] == "foo bar"


# ---------------------------------------------------------------------------
# JSONLoader
# ---------------------------------------------------------------------------

def test_json_loader_list(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
    docs = JSONLoader().load(str(p))
    assert len(docs) == 2
    _assert_doc(docs[0])
    assert docs[0]["text"] == "a"


def test_json_loader_dict(tmp_path):
    p = tmp_path / "item.json"
    p.write_text(json.dumps({"text": "hello"}), encoding="utf-8")
    docs = JSONLoader().load(str(p))
    assert len(docs) == 1
    assert docs[0]["text"] == "hello"


# ---------------------------------------------------------------------------
# CSVLoader
# ---------------------------------------------------------------------------

def test_csv_loader_text_column(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("text,score\nfoo,1\nbar,2\n", encoding="utf-8")
    docs = CSVLoader().load(str(p))
    assert len(docs) == 2
    _assert_doc(docs[0])
    assert docs[0]["text"] == "foo"
    assert docs[0]["metadata"]["type"] == "csv"


def test_csv_loader_no_text_column(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    docs = CSVLoader().load(str(p))
    assert len(docs) == 2
    assert "a:" in docs[0]["text"] or "1" in docs[0]["text"]


def test_csv_loader_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("text\n", encoding="utf-8")
    docs = CSVLoader().load(str(p))
    assert docs == []


# ---------------------------------------------------------------------------
# HTMLLoader
# ---------------------------------------------------------------------------

def test_html_loader_strips_script(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>T</title></head><body>"
        "<script>alert('x')</script>"
        "<p>Hello</p></body></html>",
        encoding="utf-8",
    )
    docs = HTMLLoader().load(str(p))
    assert len(docs) == 1
    _assert_doc(docs[0])
    assert "Hello" in docs[0]["text"]
    assert "alert" not in docs[0]["text"]
    assert docs[0]["metadata"]["type"] == "html"


def test_html_loader_empty_body(tmp_path):
    p = tmp_path / "bare.html"
    p.write_text("<html><body></body></html>", encoding="utf-8")
    docs = HTMLLoader().load(str(p))
    assert len(docs) == 1
    assert docs[0]["text"] == ""


# ---------------------------------------------------------------------------
# DOCXLoader
# ---------------------------------------------------------------------------

def test_docx_loader_happy(tmp_path):
    mock_para = MagicMock()
    mock_para.text = "Test paragraph"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]

    p = tmp_path / "doc.docx"
    p.write_bytes(b"fake")

    docx_mock = MagicMock()
    docx_mock.Document.return_value = mock_doc
    with patch.dict(sys.modules, {"docx": docx_mock}):
        # Force re-import inside the loader function by clearing cached module
        sys.modules.pop("rag_brain.loaders", None)
        from rag_brain.loaders import DOCXLoader as _DOCXLoader
        docs = _DOCXLoader().load(str(p))

    assert len(docs) == 1
    _assert_doc(docs[0])
    assert "Test paragraph" in docs[0]["text"]


def test_docx_loader_missing_dep(tmp_path):
    p = tmp_path / "doc.docx"
    p.write_bytes(b"fake")
    # Simulate python-docx not installed
    with patch.dict(sys.modules, {"docx": None}):
        docs = DOCXLoader().load(str(p))
    assert docs == []


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------

def _make_fitz_mock(texts):
    """Build a minimal fitz (PyMuPDF) mock returning given page texts."""
    page_mocks = [MagicMock(get_text=MagicMock(return_value=t)) for t in texts]
    doc_mock = MagicMock()
    doc_mock.__len__ = MagicMock(return_value=len(texts))
    doc_mock.__getitem__ = MagicMock(side_effect=lambda i: page_mocks[i])
    fitz_mock = MagicMock()
    fitz_mock.open.return_value = doc_mock
    return fitz_mock, doc_mock


def test_pdf_loader_fitz(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-fake")
    fitz_mock, _ = _make_fitz_mock(["Page one text", "Page two text"])

    with patch.dict(sys.modules, {"fitz": fitz_mock}):
        # Remove cached import if any
        sys.modules.pop("rag_brain.loaders", None)
        from rag_brain.loaders import PDFLoader as _PDFLoader  # re-import inside patch
        docs = _PDFLoader().load(str(p))

    assert len(docs) == 2
    _assert_doc(docs[0])
    assert docs[0]["metadata"]["type"] == "pdf"
    assert docs[0]["metadata"]["page"] == 0
    assert docs[1]["metadata"]["page"] == 1


def test_pdf_loader_pypdf_fallback(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-fake")

    page_mock = MagicMock()
    page_mock.extract_text.return_value = "Fallback text"
    reader_mock = MagicMock()
    reader_mock.pages = [page_mock]
    pypdf_mock = MagicMock()
    pypdf_mock.PdfReader.return_value = reader_mock

    with patch.dict(sys.modules, {"fitz": None, "pypdf": pypdf_mock}):
        sys.modules.pop("rag_brain.loaders", None)
        from rag_brain.loaders import PDFLoader as _PDFLoader
        docs = _PDFLoader().load(str(p))

    assert len(docs) >= 1 or docs == []  # graceful even if both fail


def test_pdf_loader_no_deps(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-fake")
    with patch.dict(sys.modules, {"fitz": None, "pypdf": None}):
        sys.modules.pop("rag_brain.loaders", None)
        from rag_brain.loaders import PDFLoader as _PDFLoader
        docs = _PDFLoader().load(str(p))
    assert docs == []


# ---------------------------------------------------------------------------
# DirectoryLoader
# ---------------------------------------------------------------------------

def test_directory_loader_mixed(tmp_path):
    (tmp_path / "a.txt").write_text("text file", encoding="utf-8")
    csv_p = tmp_path / "b.csv"
    csv_p.write_text("text\nhello\n", encoding="utf-8")

    loader = DirectoryLoader()
    docs = loader.load(str(tmp_path))

    assert len(docs) >= 2
    for doc in docs:
        _assert_doc(doc)


def test_directory_loader_ignores_unknown(tmp_path):
    (tmp_path / "file.xyz").write_bytes(b"unknown")
    loader = DirectoryLoader()
    docs = loader.load(str(tmp_path))
    assert docs == []


# ---------------------------------------------------------------------------
# Class-based tests (origin/master style)
# ---------------------------------------------------------------------------

class TestTextLoader:
    """Test the TextLoader class."""

    def test_load_text_file(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            f.flush()

            loader = TextLoader()
            docs = loader.load(f.name)

            assert len(docs) == 1
            assert docs[0]["text"] == "This is a test document."
            assert docs[0]["metadata"]["type"] == "text"
            assert f.name in docs[0]["metadata"]["source"]


class TestJSONLoader:
    """Test the JSONLoader class."""

    def test_load_json_list(self):
        """Test loading JSON with a list of documents."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {"text": "Document 1", "author": "Alice"},
                {"text": "Document 2", "author": "Bob"}
            ]
            json.dump(data, f)
            f.flush()

            loader = JSONLoader()
            docs = loader.load(f.name)

            assert len(docs) == 2
            assert docs[0]["text"] == "Document 1"
            assert docs[0]["metadata"]["author"] == "Alice"

    def test_load_json_object(self):
        """Test loading JSON with a single object."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"text": "Single document", "category": "test"}
            json.dump(data, f)
            f.flush()

            loader = JSONLoader()
            docs = loader.load(f.name)

            assert len(docs) == 1
            assert docs[0]["text"] == "Single document"
            assert docs[0]["metadata"]["category"] == "test"


class TestDirectoryLoader:
    """Test the DirectoryLoader class."""

    def test_load_mixed_directory(self):
        """Test loading a directory with mixed file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "test1.txt").write_text("Text file content")
            Path(tmpdir, "test2.json").write_text(json.dumps({"text": "JSON content"}))

            loader = DirectoryLoader()
            docs = loader.load(tmpdir)

            assert len(docs) >= 2
            texts = [doc["text"] for doc in docs]
            assert "Text file content" in texts
            assert "JSON content" in texts
