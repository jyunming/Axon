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
    # ID is now a stable hash (not basename) â€” just verify it's non-empty
    assert docs[0]["id"]
    assert docs[0]["id"] != "hello.txt"


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
    assert "Context: data.tsv" in docs[0]["text"]
    assert "Data: content: foo bar | score: 1" in docs[0]["text"]


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
    assert "Context: data.csv" in docs[0]["text"]
    assert "Data: text: foo | score: 1" in docs[0]["text"]
    assert docs[0]["metadata"]["type"] == "csv"


def test_csv_loader_no_text_column(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    docs = CSVLoader().load(str(p))
    assert len(docs) == 2
    assert "Context: data.csv" in docs[0]["text"]
    assert "Data: a: 1 | b: 2" in docs[0]["text"]


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

    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        f_path = tmp_path / "test.txt"
        f_path.write_text("This is a test document.", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(str(f_path))
        assert len(docs) == 1
        assert docs[0]["text"] == "This is a test document."
        assert docs[0]["metadata"]["type"] == "text"
        assert str(f_path) in docs[0]["metadata"]["source"]


class TestJSONLoader:
    """Test the JSONLoader class."""

    def test_load_json_list(self, tmp_path):
        """Test loading JSON with a list of documents."""
        f_path = tmp_path / "test.json"
        data = [
            {"text": "Document 1", "author": "Alice"},
            {"text": "Document 2", "author": "Bob"},
        ]
        with open(f_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loader = JSONLoader()
        docs = loader.load(str(f_path))

        assert len(docs) == 2
        assert docs[0]["text"] == "Document 1"
        assert docs[0]["metadata"]["author"] == "Alice"

    def test_load_json_object(self, tmp_path):
        """Test loading JSON with a single object."""
        f_path = tmp_path / "single.json"
        data = {"text": "Single document", "category": "test"}
        with open(f_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loader = JSONLoader()
        docs = loader.load(str(f_path))

        assert len(docs) == 1
        assert docs[0]["text"] == "Single document"
        assert docs[0]["metadata"]["category"] == "test"


class TestImageLoader:
    """Tests for the ImageLoader (VLM-based image captioning)."""

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

    def test_load_mixed_directory(self, tmp_path):
        """Test loading a directory with mixed file types."""
        tmpdir = str(tmp_path)
        # Create test files
        Path(tmpdir, "test1.txt").write_text("Text file content")
        Path(tmpdir, "test2.json").write_text(json.dumps({"text": "JSON content"}))

        loader = DirectoryLoader()
        docs = loader.load(tmpdir)

        assert len(docs) >= 2
        texts = [doc["text"] for doc in docs]
        assert any("Text file content" in t for t in texts)
        assert any("JSON content" in t for t in texts)


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
        # Check that breadcrumbs are present in both
        assert "[File Path: hello.txt]" in sync_docs[0]["text"]
        assert "[File Path: hello.txt]" in async_docs[0]["text"]
        assert "Hello async world" in sync_docs[0]["text"]
        assert "Hello async world" in async_docs[0]["text"]

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
        all_text = " ".join(d["text"] for d in docs)
        assert "file A content" in all_text
        assert "file B content" in all_text


# ---------------------------------------------------------------------------
# URLLoader  (P1-B)
# ---------------------------------------------------------------------------


class TestURLLoader:
    """Tests for URLLoader. All network calls are mocked â€” no real HTTP."""

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
        ), patch("httpx.Client") as mock_client_cls:
            mock_gai.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
            mock_http = mock_client_cls.return_value.__enter__.return_value
            mock_http.get.return_value = self._make_response(
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

        with patch("axon.loaders.URLLoader._check_ssrf"), patch("httpx.Client") as mock_client_cls:
            mock_http = mock_client_cls.return_value.__enter__.return_value
            mock_http.get.return_value = self._make_response(
                text="raw text content",
                content_type="text/plain",
            )
            docs = loader.load("https://example.com/readme.txt")

        assert docs[0]["text"] == "raw text content"

    # --- Error conditions ---

    def test_non_200_raises_value_error(self):
        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch("httpx.Client") as mock_client_cls:
            mock_http = mock_client_cls.return_value.__enter__.return_value
            mock_http.get.return_value = self._make_response(status=404)
            with pytest.raises(ValueError, match="HTTP 404"):
                loader.load("https://example.com/missing")

    def test_binary_content_type_raises_value_error(self):
        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch("httpx.Client") as mock_client_cls:
            mock_http = mock_client_cls.return_value.__enter__.return_value
            mock_http.get.return_value = self._make_response(content_type="application/pdf")
            with pytest.raises(ValueError, match="Non-text content type"):
                loader.load("https://example.com/file.pdf")

    def test_timeout_raises_value_error(self):
        import httpx

        loader = URLLoader()
        # Patch both SSRF and getaddrinfo to be doubly safe
        with patch("axon.loaders.URLLoader._check_ssrf"), patch(
            "axon.loaders.socket.getaddrinfo"
        ), patch("httpx.Client") as mock_client_cls:
            mock_http = mock_client_cls.return_value.__enter__.return_value
            mock_http.get.side_effect = httpx.TimeoutException("timeout", request=None)
            with pytest.raises(ValueError, match="timed out"):
                loader.load("https://slow.example.com")

    def test_too_many_redirects_raises_value_error(self):
        import httpx

        loader = URLLoader()
        with patch("axon.loaders.URLLoader._check_ssrf"), patch(
            "axon.loaders.socket.getaddrinfo"
        ), patch("httpx.Client") as mock_client_cls:
            mock_http = mock_client_cls.return_value.__enter__.return_value
            mock_http.get.side_effect = httpx.TooManyRedirects("redirects", request=None)
            with pytest.raises(ValueError, match="redirects"):
                loader.load("https://redirect.example.com")

    def test_dns_failure_raises_value_error(self):
        """Unresolvable hostname raises ValueError before any HTTP call."""
        import socket

        loader = URLLoader()
        with patch("axon.loaders.socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            with pytest.raises(ValueError, match="Could not resolve"):
                loader.load("https://nonexistent.example.invalid")


# ---------------------------------------------------------------------------
# Gap 5 â€” _rewrite_github_url
# ---------------------------------------------------------------------------


class TestRewriteGithubUrl:
    """Tests for the _rewrite_github_url helper (Gap 5)."""

    def setup_method(self):
        from axon.loaders import _rewrite_github_url

        self.rewrite = _rewrite_github_url

    def test_blob_url_rewritten_to_raw(self):
        url = "https://github.com/openai/gpt-4/blob/main/README.md"
        result = self.rewrite(url)
        assert result == "https://raw.githubusercontent.com/openai/gpt-4/main/README.md"

    def test_blob_url_with_subpath_rewritten(self):
        url = "https://github.com/owner/repo/blob/develop/src/utils/helpers.py"
        result = self.rewrite(url)
        assert result == "https://raw.githubusercontent.com/owner/repo/develop/src/utils/helpers.py"

    def test_http_blob_url_still_rewritten(self):
        url = "http://github.com/owner/repo/blob/main/file.txt"
        result = self.rewrite(url)
        assert result == "https://raw.githubusercontent.com/owner/repo/main/file.txt"

    def test_tree_url_raises_value_error(self):
        url = "https://github.com/owner/repo/tree/main/src"
        with pytest.raises(ValueError, match="tree"):
            self.rewrite(url)

    def test_gist_url_rewritten_to_raw(self):
        url = "https://gist.github.com/torvalds/1234abcd5678ef90"
        result = self.rewrite(url)
        assert result == "https://gist.githubusercontent.com/torvalds/1234abcd5678ef90/raw"

    def test_gist_url_with_trailing_slash(self):
        url = "https://gist.github.com/user/abcdef1234567890/"
        result = self.rewrite(url)
        assert result == "https://gist.githubusercontent.com/user/abcdef1234567890/raw"

    def test_non_github_url_unchanged(self):
        url = "https://example.com/some/page.html"
        assert self.rewrite(url) == url

    def test_raw_githubusercontent_url_unchanged(self):
        url = "https://raw.githubusercontent.com/owner/repo/main/file.py"
        assert self.rewrite(url) == url

    def test_github_releases_url_unchanged(self):
        """Release asset URLs pass through (not blob/tree/gist)."""
        url = "https://github.com/owner/repo/releases/download/v1.0/app.tar.gz"
        assert self.rewrite(url) == url

    def test_load_rewrites_github_blob_url(self):
        """URLLoader.load() automatically rewrites GitHub blob URLs before fetching."""
        from unittest.mock import MagicMock, patch

        loader = URLLoader()
        blob_url = "https://github.com/owner/repo/blob/main/README.md"
        raw_url = "https://raw.githubusercontent.com/owner/repo/main/README.md"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "# Hello World"

        with patch(
            "axon.loaders.socket.getaddrinfo", return_value=[("", "", "", "", ("1.2.3.4", 0))]
        ):
            with patch("httpx.Client") as mock_client_cls:
                mock_http = mock_client_cls.return_value.__enter__.return_value
                mock_http.get.return_value = mock_resp
                docs = loader.load(blob_url)
                # httpx.Client.get must have been called with the RAW url, not the blob url
                called_url = mock_http.get.call_args[0][0]
                assert called_url == raw_url
        assert docs[0]["text"] == "# Hello World"


class TestSmartTextLoaderIdFix:
    """SmartTextLoader must use the full source string as the doc id, not basename."""

    def test_agent_doc_id_preserved(self):
        """source='agent_doc_abc12345' must become id='agent_doc_abc12345', not 'agent_doc_abc12345'."""
        from axon.loaders import SmartTextLoader

        loader = SmartTextLoader()
        docs = loader.load_text("Hello world ingestion test.", source="agent_doc_abc12345")
        assert len(docs) == 1
        assert docs[0]["id"] == "agent_doc_abc12345"

    def test_full_path_source_uses_full_string(self):
        """For file-path sources, full path is the id (consistent with other loaders)."""
        from axon.loaders import SmartTextLoader

        loader = SmartTextLoader()
        docs = loader.load_text("Some content.", source="/data/myproject/file.txt")
        assert docs[0]["id"] == "/data/myproject/file.txt"

    def test_windows_path_source_not_mangled(self):
        """Windows-style paths must not be mangled â€” use mixed separators so the test
        is meaningful on all platforms (os.path.basename of a backslash-only path is
        a no-op on POSIX, so we include a forward slash to force a detectable split)."""
        from axon.loaders import SmartTextLoader

        loader = SmartTextLoader()
        # Mixed separator: os.path.basename would return 'notes.txt' if the bug were present.
        source = r"C:\data/project\notes.txt"
        docs = loader.load_text("Some content.", source=source)
        assert docs[0]["id"] == source


# ---------------------------------------------------------------------------
# Phase 3: stable file IDs
# ---------------------------------------------------------------------------


def test_text_loader_id_is_not_basename(tmp_path):
    """TextLoader id should not be just the filename."""
    from axon.loaders import TextLoader

    f = tmp_path / "overview.txt"
    f.write_text("hello world")
    docs = TextLoader().load(str(f))
    assert docs
    assert docs[0]["id"] != "overview.txt"
    assert len(docs[0]["id"]) > 10  # stable hash


def test_two_files_same_name_different_dirs_get_different_ids(tmp_path):
    """Two files with same basename in different dirs get distinct IDs."""
    from axon.loaders import TextLoader

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "doc.txt").write_text("content a")
    (dir_b / "doc.txt").write_text("content b")
    docs_a = TextLoader().load(str(dir_a / "doc.txt"))
    docs_b = TextLoader().load(str(dir_b / "doc.txt"))
    assert docs_a[0]["id"] != docs_b[0]["id"]


# ---------------------------------------------------------------------------
# SmartTextLoader â€” file-path load
# ---------------------------------------------------------------------------


class TestSmartTextLoaderFile:
    def test_load_txt_file(self, tmp_path):
        from axon.loaders import SmartTextLoader

        p = tmp_path / "notes.txt"
        p.write_text("This is plain prose text without many commas.", encoding="utf-8")
        docs = SmartTextLoader().load(str(p))
        assert len(docs) >= 1
        _assert_doc(docs[0])

    def test_load_csv_delegates_to_table(self, tmp_path):
        from axon.loaders import SmartTextLoader

        p = tmp_path / "data.csv"
        p.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,28,Chicago\n", encoding="utf-8")
        docs = SmartTextLoader().load(str(p))
        # should detect as table and delegate
        assert len(docs) >= 1
        _assert_doc(docs[0])


# ---------------------------------------------------------------------------
# CodeFileLoader
# ---------------------------------------------------------------------------


class TestCodeFileLoader:
    def test_basic_python_file(self, tmp_path):
        from axon.loaders import CodeFileLoader

        p = tmp_path / "example.py"
        p.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        docs = CodeFileLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "code"
        assert docs[0]["metadata"]["source"] == str(p)
        assert docs[0]["id"].startswith("code_")
        assert "def hello" in docs[0]["text"]

    def test_id_is_stable_hash(self, tmp_path):
        from axon.loaders import CodeFileLoader

        p = tmp_path / "stable.py"
        p.write_text("x = 1")
        docs1 = CodeFileLoader().load(str(p))
        docs2 = CodeFileLoader().load(str(p))
        assert docs1[0]["id"] == docs2[0]["id"]


# ---------------------------------------------------------------------------
# NotebookLoader
# ---------------------------------------------------------------------------


class TestNotebookLoader:
    def test_basic_notebook(self, tmp_path):
        from axon.loaders import NotebookLoader

        nb = tmp_path / "demo.ipynb"
        nb.write_text(
            json.dumps(
                {
                    "nbformat": 4,
                    "nbformat_minor": 5,
                    "cells": [
                        {"cell_type": "code", "source": "x = 1", "outputs": []},
                        {"cell_type": "markdown", "source": "## heading"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        docs = NotebookLoader().load(str(nb))
        assert len(docs) == 2
        _assert_doc(docs[0])
        assert any("x = 1" in d["text"] for d in docs)
        assert docs[0]["metadata"]["type"] == "notebook"

    def test_empty_cells_skipped(self, tmp_path):
        from axon.loaders import NotebookLoader

        nb = tmp_path / "empty.ipynb"
        nb.write_text(
            json.dumps(
                {
                    "nbformat": 4,
                    "nbformat_minor": 5,
                    "cells": [
                        {"cell_type": "code", "source": "", "outputs": []},
                        {"cell_type": "markdown", "source": "## real content"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        docs = NotebookLoader().load(str(nb))
        assert len(docs) == 1
        assert "real content" in docs[0]["text"]

    def test_malformed_json_returns_empty(self, tmp_path):
        from axon.loaders import NotebookLoader

        nb = tmp_path / "bad.ipynb"
        nb.write_text("not json", encoding="utf-8")
        docs = NotebookLoader().load(str(nb))
        assert docs == []


# ---------------------------------------------------------------------------
# XMLLoader
# ---------------------------------------------------------------------------


class TestXMLLoader:
    def test_basic_xml(self, tmp_path):
        from axon.loaders import XMLLoader

        p = tmp_path / "data.xml"
        p.write_text(
            '<?xml version="1.0"?>'
            '<root><item id="1">Hello world</item><item id="2">Foo bar</item></root>',
            encoding="utf-8",
        )
        docs = XMLLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "xml"
        assert "Hello world" in docs[0]["text"] or "Foo bar" in docs[0]["text"]

    def test_malformed_xml_falls_back_to_regex(self, tmp_path):
        from axon.loaders import XMLLoader

        p = tmp_path / "broken.xml"
        p.write_text("<unclosed><item>some text</item>", encoding="utf-8")
        docs = XMLLoader().load(str(p))
        # Should return something, not crash
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# JSONLLoader
# ---------------------------------------------------------------------------


class TestJSONLLoader:
    def test_basic_jsonl(self, tmp_path):
        from axon.loaders import JSONLLoader

        p = tmp_path / "data.jsonl"
        p.write_text(
            '{"text": "line one"}\n{"text": "line two"}\n{"text": "line three"}\n', encoding="utf-8"
        )
        docs = JSONLLoader().load(str(p))
        assert len(docs) == 3
        _assert_doc(docs[0])
        assert docs[0]["text"] == "line one"
        assert docs[0]["metadata"]["type"] == "jsonl"

    def test_malformed_line_skipped(self, tmp_path):
        from axon.loaders import JSONLLoader

        p = tmp_path / "mixed.jsonl"
        p.write_text('{"text": "good"}\nnot json here\n{"text": "also good"}\n', encoding="utf-8")
        docs = JSONLLoader().load(str(p))
        assert len(docs) == 2
        assert docs[0]["text"] == "good"
        assert docs[1]["text"] == "also good"

    def test_empty_lines_skipped(self, tmp_path):
        from axon.loaders import JSONLLoader

        p = tmp_path / "sparse.jsonl"
        p.write_text('{"text": "hello"}\n\n\n{"text": "world"}\n', encoding="utf-8")
        docs = JSONLLoader().load(str(p))
        assert len(docs) == 2

    def test_non_text_field_uses_full_json(self, tmp_path):
        from axon.loaders import JSONLLoader

        p = tmp_path / "ntext.jsonl"
        p.write_text('{"value": 42, "label": "x"}\n', encoding="utf-8")
        docs = JSONLLoader().load(str(p))
        assert len(docs) == 1
        # should fall back to json.dumps of item
        assert "42" in docs[0]["text"]


# ---------------------------------------------------------------------------
# RTFLoader
# ---------------------------------------------------------------------------


class TestRTFLoader:
    def test_basic_rtf(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_text(r"{\rtf1\ansi Hello World}", encoding="utf-8")
        # Mock striprtf
        mock_striprtf = MagicMock()
        mock_striprtf.striprtf.rtf_to_text.return_value = "Hello World"
        with patch.dict(
            sys.modules, {"striprtf": mock_striprtf, "striprtf.striprtf": mock_striprtf.striprtf}
        ):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import RTFLoader as _RTFLoader

            docs = _RTFLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "rtf"

    def test_missing_dep_returns_empty(self, tmp_path):
        p = tmp_path / "doc.rtf"
        p.write_bytes(b"fake rtf")
        with patch.dict(sys.modules, {"striprtf": None, "striprtf.striprtf": None}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import RTFLoader as _RTFLoader

            docs = _RTFLoader().load(str(p))
        assert docs == []


# ---------------------------------------------------------------------------
# LaTeXLoader
# ---------------------------------------------------------------------------


class TestLaTeXLoader:
    def test_basic_latex(self, tmp_path):
        from axon.loaders import LaTeXLoader

        p = tmp_path / "paper.tex"
        p.write_text(
            r"""\documentclass{article}
\begin{document}
\section{Introduction}
This is the \textbf{introduction} to our paper.
\end{document}""",
            encoding="utf-8",
        )
        docs = LaTeXLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "latex"
        # Should contain the prose without latex commands
        assert "introduction" in docs[0]["text"].lower()

    def test_math_discarded(self, tmp_path):
        from axon.loaders import LaTeXLoader

        p = tmp_path / "math.tex"
        p.write_text(
            r"""\begin{document}
Text before equation.
\begin{equation}
E = mc^2
\end{equation}
Text after equation.
\end{document}""",
            encoding="utf-8",
        )
        docs = LaTeXLoader().load(str(p))
        assert "Text before equation" in docs[0]["text"]
        assert "mc^2" not in docs[0]["text"]


# ---------------------------------------------------------------------------
# EMLLoader
# ---------------------------------------------------------------------------


class TestEMLLoader:
    def test_basic_eml(self, tmp_path):
        from axon.loaders import EMLLoader

        p = tmp_path / "test.eml"
        p.write_bytes(
            b"From: alice@example.com\r\n"
            b"To: bob@example.com\r\n"
            b"Subject: Hello\r\n"
            b"Date: Mon, 1 Jan 2024 12:00:00 +0000\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Hello Bob, this is the email body.\r\n"
        )
        docs = EMLLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "eml"
        assert docs[0]["metadata"]["email_from"] == "alice@example.com"
        assert docs[0]["metadata"]["email_subject"] == "Hello"
        assert "Hello Bob" in docs[0]["text"]

    def test_from_to_subject_in_text(self, tmp_path):
        from axon.loaders import EMLLoader

        p = tmp_path / "meta.eml"
        p.write_bytes(
            b"From: sender@test.com\r\n"
            b"To: receiver@test.com\r\n"
            b"Subject: Test Subject\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Body text.\r\n"
        )
        docs = EMLLoader().load(str(p))
        text = docs[0]["text"]
        assert "From:" in text
        assert "Subject:" in text


# ---------------------------------------------------------------------------
# MSGLoader
# ---------------------------------------------------------------------------


class TestMSGLoader:
    def test_basic_msg(self, tmp_path):
        p = tmp_path / "email.msg"
        p.write_bytes(b"fake msg")
        mock_msg = MagicMock()
        mock_msg.sender = "alice@example.com"
        mock_msg.to = "bob@example.com"
        mock_msg.subject = "Test Subject"
        mock_msg.date = None
        mock_msg.body = "Hello Bob"
        mock_msg.htmlBody = None
        mock_extract_msg = MagicMock()
        mock_extract_msg.Message.return_value = mock_msg
        with patch.dict(sys.modules, {"extract_msg": mock_extract_msg}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import MSGLoader as _MSGLoader

            docs = _MSGLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "msg"
        assert "Hello Bob" in docs[0]["text"]
        assert docs[0]["metadata"]["email_from"] == "alice@example.com"

    def test_missing_dep_returns_empty(self, tmp_path):
        p = tmp_path / "email.msg"
        p.write_bytes(b"fake")
        with patch.dict(sys.modules, {"extract_msg": None}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import MSGLoader as _MSGLoader

            docs = _MSGLoader().load(str(p))
        assert docs == []


# ---------------------------------------------------------------------------
# PPTXLoader
# ---------------------------------------------------------------------------


class TestPPTXLoader:
    def test_basic_pptx(self, tmp_path):
        p = tmp_path / "slides.pptx"
        p.write_bytes(b"fake pptx")
        # Build mock Presentation with 2 slides
        shape1 = MagicMock()
        shape1.text = "Slide one content"
        shape2 = MagicMock()
        shape2.text = "Slide two content"
        slide1 = MagicMock()
        slide1.shapes = [shape1]
        slide2 = MagicMock()
        slide2.shapes = [shape2]
        prs_mock = MagicMock()
        prs_mock.slides = [slide1, slide2]
        mock_pptx = MagicMock()
        mock_pptx.Presentation.return_value = prs_mock
        with patch.dict(sys.modules, {"pptx": mock_pptx}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import PPTXLoader as _PPTXLoader

            docs = _PPTXLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "pptx"
        assert "Slide one content" in docs[0]["text"]
        assert "Slide two content" in docs[0]["text"]

    def test_missing_dep_returns_empty(self, tmp_path):
        p = tmp_path / "slides.pptx"
        p.write_bytes(b"fake")
        with patch.dict(sys.modules, {"pptx": None}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import PPTXLoader as _PPTXLoader

            docs = _PPTXLoader().load(str(p))
        assert docs == []


# ---------------------------------------------------------------------------
# ExcelLoader
# ---------------------------------------------------------------------------


class TestExcelLoader:
    def test_missing_pandas_returns_empty(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"fake xlsx")
        with patch.dict(sys.modules, {"pandas": None}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import ExcelLoader as _ExcelLoader

            docs = _ExcelLoader().load(str(p))
        assert docs == []


# ---------------------------------------------------------------------------
# ParquetLoader
# ---------------------------------------------------------------------------


class TestParquetLoader:
    def test_missing_pandas_returns_empty(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"fake parquet")
        with patch.dict(sys.modules, {"pandas": None}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import ParquetLoader as _ParquetLoader

            docs = _ParquetLoader().load(str(p))
        assert docs == []

    def test_missing_pyarrow_returns_empty(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"fake parquet")
        mock_pd = MagicMock()
        mock_pd.read_parquet.side_effect = ImportError("No module named 'pyarrow'")
        with patch.dict(sys.modules, {"pandas": mock_pd}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import ParquetLoader as _ParquetLoader

            docs = _ParquetLoader().load(str(p))
        assert docs == []


# ---------------------------------------------------------------------------
# EPUBLoader
# ---------------------------------------------------------------------------


class TestEPUBLoader:
    def test_missing_dep_returns_empty(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"fake epub")
        with patch.dict(sys.modules, {"ebooklib": None, "ebooklib.epub": None}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import EPUBLoader as _EPUBLoader

            docs = _EPUBLoader().load(str(p))
        assert docs == []

    def test_basic_epub(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"fake epub")
        # Mock ebooklib
        mock_item = MagicMock()
        mock_item.get_content.return_value = (
            b"<html><body><p>Chapter content here</p></body></html>"
        )
        mock_item.get_name.return_value = "chapter1.xhtml"
        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [mock_item]
        mock_ebooklib = MagicMock()
        mock_ebooklib.ITEM_DOCUMENT = 9
        mock_epub_mod = MagicMock()
        mock_epub_mod.read_epub.return_value = mock_book
        mock_ebooklib.epub = mock_epub_mod
        with patch.dict(sys.modules, {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub_mod}):
            sys.modules.pop("axon.loaders", None)
            from axon.loaders import EPUBLoader as _EPUBLoader

            docs = _EPUBLoader().load(str(p))
        assert len(docs) == 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "epub"
        assert "Chapter content" in docs[0]["text"]


# ---------------------------------------------------------------------------
# SQLLoader
# ---------------------------------------------------------------------------


class TestSQLLoader:
    def test_basic_sql_statements(self, tmp_path):
        from axon.loaders import SQLLoader

        p = tmp_path / "schema.sql"
        p.write_text(
            "CREATE TABLE users (id INT, name VARCHAR(100));\n"
            "INSERT INTO users VALUES (1, 'Alice');\n",
            encoding="utf-8",
        )
        docs = SQLLoader().load(str(p))
        assert len(docs) >= 1
        _assert_doc(docs[0])
        assert docs[0]["metadata"]["type"] == "sql"
        assert any("CREATE TABLE" in d["text"] for d in docs)

    def test_no_semicolons_returns_whole_file(self, tmp_path):
        from axon.loaders import SQLLoader

        p = tmp_path / "query.sql"
        p.write_text("SELECT * FROM products WHERE price > 100", encoding="utf-8")
        docs = SQLLoader().load(str(p))
        assert len(docs) == 1
        assert "SELECT" in docs[0]["text"]

    def test_comment_only_statements_skipped(self, tmp_path):
        from axon.loaders import SQLLoader

        p = tmp_path / "commented.sql"
        p.write_text(
            "-- This is just a comment\n" "CREATE TABLE items (id INT);\n", encoding="utf-8"
        )
        docs = SQLLoader().load(str(p))
        assert any("CREATE TABLE" in d["text"] for d in docs)
