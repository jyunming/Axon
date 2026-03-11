"""
tests/test_loaders.py

Unit tests for all document loaders in axon.loaders.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axon.loaders import (
    BMPLoader,
    CSVLoader,
    DirectoryLoader,
    DOCXLoader,
    HTMLLoader,
    ImageLoader,
    JSONLoader,
    TextLoader,
    TSVLoader,
    URLLoader,
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
        sys.modules.pop("axon.loaders", None)
        from axon.loaders import DOCXLoader as _DOCXLoader

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
        sys.modules.pop("axon.loaders", None)
        from axon.loaders import PDFLoader as _PDFLoader  # re-import inside patch

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
        sys.modules.pop("axon.loaders", None)
        from axon.loaders import PDFLoader as _PDFLoader

        docs = _PDFLoader().load(str(p))

    assert len(docs) == 1
    assert docs[0]["text"] == "Fallback text"
    assert docs[0]["metadata"]["type"] == "pdf"


def test_pdf_loader_no_deps(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-fake")
    with patch.dict(sys.modules, {"fitz": None, "pypdf": None}):
        sys.modules.pop("axon.loaders", None)
        from axon.loaders import PDFLoader as _PDFLoader

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [
                {"text": "Document 1", "author": "Alice"},
                {"text": "Document 2", "author": "Bob"},
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {"text": "Single document", "category": "test"}
            json.dump(data, f)
            f.flush()

            loader = JSONLoader()
            docs = loader.load(f.name)

            assert len(docs) == 1
            assert docs[0]["text"] == "Single document"
            assert docs[0]["metadata"]["category"] == "test"


class TestImageLoader:
    """Tests for the ImageLoader (VLM-based image captioning) and BMPLoader alias."""

    def test_bmp_loader_is_alias_for_image_loader(self):
        assert BMPLoader is ImageLoader

    def test_image_loader_returns_empty_when_ollama_missing(self, tmp_path):
        """ImageLoader returns [] gracefully when ollama is not installed."""
        loader = ImageLoader()
        loader.ollama = None  # simulate missing package
        img_path = tmp_path / "test.bmp"
        img_path.write_bytes(b"fake")
        result = loader.load(str(img_path))
        assert result == []

    def test_image_loader_calls_ollama_generate(self, tmp_path):
        """ImageLoader calls ollama.generate and returns a captioned document."""
        from PIL import Image as PILImage

        # Create a minimal real BMP image so Pillow can open it
        img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
        img_path = tmp_path / "red.bmp"
        img.save(str(img_path))

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = {"response": "A red square"}

        loader = ImageLoader(ollama_model="llava")
        loader.ollama = mock_ollama

        docs = loader.load(str(img_path))

        assert len(docs) == 1
        assert "A red square" in docs[0]["text"]
        assert docs[0]["metadata"]["type"] == "image"
        assert docs[0]["metadata"]["format"] == "bmp"
        assert docs[0]["metadata"]["model"] == "llava"
        mock_ollama.generate.assert_called_once()
        call_kwargs = mock_ollama.generate.call_args
        assert (
            call_kwargs[1]["model"] == "llava" or call_kwargs[0][0] == "llava"
        )  # positional or kw

    def test_image_loader_handles_png(self, tmp_path):
        """ImageLoader correctly records 'png' as format in metadata."""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (4, 4), color=(0, 255, 0))
        img_path = tmp_path / "green.png"
        img.save(str(img_path))

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = {"response": "A green square"}

        loader = ImageLoader()
        loader.ollama = mock_ollama

        docs = loader.load(str(img_path))
        assert len(docs) == 1
        assert docs[0]["metadata"]["format"] == "png"

    def test_image_loader_handles_pgm(self, tmp_path):
        """ImageLoader converts PGM (grayscale) via Pillow and calls ollama."""
        from PIL import Image as PILImage

        img = PILImage.new("L", (4, 4), color=128)  # grayscale
        img_path = tmp_path / "gray.pgm"
        img.save(str(img_path))

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = {"response": "A gray image"}

        loader = ImageLoader()
        loader.ollama = mock_ollama

        docs = loader.load(str(img_path))
        assert len(docs) == 1
        assert docs[0]["metadata"]["format"] == "pgm"

    def test_image_loader_returns_empty_on_ollama_error(self, tmp_path):
        """ImageLoader returns [] and logs when ollama.generate raises."""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (4, 4))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        mock_ollama = MagicMock()
        mock_ollama.generate.side_effect = RuntimeError("connection refused")

        loader = ImageLoader()
        loader.ollama = mock_ollama

        result = loader.load(str(img_path))
        assert result == []

    def test_directory_loader_registers_image_formats(self):
        """DirectoryLoader registers png, tif, tiff, pgm alongside bmp."""
        dl = DirectoryLoader()
        for ext in (".bmp", ".png", ".tif", ".tiff", ".pgm"):
            assert ext in dl.loaders, f"Extension {ext} not registered in DirectoryLoader"


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


# ---------------------------------------------------------------------------
# New test: DirectoryLoader.aload()
# ---------------------------------------------------------------------------


class TestDirectoryLoaderAload:
    """Test the async aload() method of DirectoryLoader."""

    def test_aload_returns_same_as_load(self, tmp_path):
        """aload() should return the same documents as load() for the same directory."""
        import asyncio

        (tmp_path / "hello.txt").write_text("Hello async world", encoding="utf-8")

        loader = DirectoryLoader()
        sync_docs = loader.load(str(tmp_path))
        async_docs = asyncio.run(loader.aload(str(tmp_path)))

        assert len(async_docs) == len(sync_docs)
        sync_texts = sorted(d["text"] for d in sync_docs)
        async_texts = sorted(d["text"] for d in async_docs)
        assert sync_texts == async_texts

    def test_aload_returns_empty_for_unknown_extensions(self, tmp_path):
        """aload() should return [] when no supported files are found."""
        import asyncio

        (tmp_path / "ignored.xyz").write_bytes(b"binary")

        loader = DirectoryLoader()
        docs = asyncio.run(loader.aload(str(tmp_path)))
        assert docs == []

    def test_aload_multiple_files(self, tmp_path):
        """aload() correctly gathers results from multiple files concurrently."""
        import asyncio

        (tmp_path / "a.txt").write_text("file A content", encoding="utf-8")
        (tmp_path / "b.txt").write_text("file B content", encoding="utf-8")

        loader = DirectoryLoader()
        docs = asyncio.run(loader.aload(str(tmp_path)))

        assert len(docs) == 2
        texts = {d["text"] for d in docs}
        assert "file A content" in texts
        assert "file B content" in texts


# ---------------------------------------------------------------------------
# URLLoader  (P1-B)
# ---------------------------------------------------------------------------


class TestURLLoader:
    """Tests for URLLoader. All network calls are mocked — no real HTTP."""

    def _make_response(
        self,
        text="<html><body>Hello world</body></html>",
        status=200,
        content_type="text/html; charset=utf-8",
    ):
        """Build a minimal mock httpx.Response."""
        resp = MagicMock()
        resp.status_code = status
        resp.headers = {"content-type": content_type}
        resp.text = text
        return resp

    # --- Scheme / SSRF mitigations (no network needed) ---

    def test_blocks_file_scheme(self):
        """file:// URLs are rejected before any network call."""
        loader = URLLoader()
        with pytest.raises(ValueError, match="Scheme 'file'"):
            loader.load("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        loader = URLLoader()
        with pytest.raises(ValueError, match="Scheme 'ftp'"):
            loader.load("ftp://example.com/file.txt")

    def test_blocks_localhost_loopback(self):
        """127.x addresses are rejected via DNS resolution check."""
        loader = URLLoader()
        # Patch socket.getaddrinfo so we don't do real DNS
        with patch("axon.loaders.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("127.0.0.1", 0))]
            with pytest.raises(ValueError, match="blocked private"):
                loader.load("http://localhost")

    def test_blocks_cloud_metadata_ip(self):
        """169.254.169.254 (cloud IMDS) is rejected."""
        loader = URLLoader()
        with patch("axon.loaders.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.169.254", 0))]
            with pytest.raises(ValueError, match="blocked private"):
                loader.load("http://metadata.internal")

    def test_blocks_rfc1918_private_10(self):
        """10.x private range is rejected."""
        loader = URLLoader()
        with patch("axon.loaders.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("10.0.0.1", 0))]
            with pytest.raises(ValueError, match="blocked private"):
                loader.load("http://internal.corp")

    # --- Successful fetch ---

    def test_success_html_strips_tags(self):
        """HTML content is fetched and tags are stripped; doc schema is valid."""
        loader = URLLoader()

        with patch("axon.loaders.socket.getaddrinfo") as mock_gai, patch(
            "axon.loaders.URLLoader._check_ssrf"
        ), patch("httpx.get") as mock_get:
            mock_gai.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
            mock_get.return_value = self._make_response(
                text="<html><head><title>T</title></head><body><p>Hello world</p></body></html>",
                content_type="text/html; charset=utf-8",
            )
            docs = loader.load("https://example.com")

        assert len(docs) == 1
        _assert_doc(docs[0])
        assert "Hello world" in docs[0]["text"]
        assert "<" not in docs[0]["text"], "HTML tags should be stripped"
        assert docs[0]["metadata"]["source"] == "https://example.com"
        assert docs[0]["metadata"]["type"] == "url"

    def test_success_plain_text_not_stripped(self):
        """Plain text responses are returned as-is without HTML stripping."""
        loader = URLLoader()

        with patch("axon.loaders.URLLoader._check_ssrf"), patch("httpx.get") as mock_get:
            mock_get.return_value = self._make_response(
                text="raw text content",
                content_type="text/plain",
            )
            docs = loader.load("https://example.com/readme.txt")

        assert docs[0]["text"] == "raw text content"

    # --- Error conditions ---

    def test_non_200_raises_value_error(self):
        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch("httpx.get") as mock_get:
            mock_get.return_value = self._make_response(status=404)
            with pytest.raises(ValueError, match="HTTP 404"):
                loader.load("https://example.com/missing")

    def test_binary_content_type_raises_value_error(self):
        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch("httpx.get") as mock_get:
            mock_get.return_value = self._make_response(content_type="application/pdf")
            with pytest.raises(ValueError, match="Non-text content type"):
                loader.load("https://example.com/file.pdf")

    def test_timeout_raises_value_error(self):
        import httpx

        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch(
            "httpx.get", side_effect=httpx.TimeoutException("timeout", request=None)
        ):
            with pytest.raises(ValueError, match="timed out"):
                loader.load("https://slow.example.com")

    def test_too_many_redirects_raises_value_error(self):
        import httpx

        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch(
            "httpx.get", side_effect=httpx.TooManyRedirects("redirects", request=None)
        ):
            with pytest.raises(ValueError, match="redirects"):
                loader.load("https://redirect.example.com")

    def test_dns_failure_raises_value_error(self):
        """Unresolvable hostname raises ValueError before any HTTP call."""
        import socket

        loader = URLLoader()
        with patch("axon.loaders.socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            with pytest.raises(ValueError, match="Could not resolve"):
                loader.load("https://nonexistent.example.invalid")
