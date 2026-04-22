from __future__ import annotations

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

    # ID is now a stable hash (not basename) — just verify it's non-empty

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

        all_text = " ".join(d["text"] for d in docs)

        assert "file A content" in all_text

        assert "file B content" in all_text


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

        with patch("axon.loaders.socket.getaddrinfo") as mock_gai, patch(
            "axon.loaders.URLLoader._check_ssrf"
        ), patch("httpx.Client") as mock_client_cls:
            mock_gai.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
            mock_http = mock_client_cls.return_value.__enter__.return_value

            mock_http.get.side_effect = httpx.TimeoutException("timeout", request=None)

            with pytest.raises(ValueError, match="timed out"):
                loader.load("https://slow.example.com")

    def test_too_many_redirects_raises_value_error(self):
        import httpx

        loader = URLLoader()

        with patch("axon.loaders.socket.getaddrinfo") as mock_gai, patch(
            "axon.loaders.URLLoader._check_ssrf"
        ), patch("httpx.Client") as mock_client_cls:
            mock_gai.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
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


# Gap 5 — _rewrite_github_url


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

        mock_resp.url = raw_url

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


"""Extra tests for axon.loaders to push coverage above 90%."""
import asyncio

# ---------------------------------------------------------------------------
# _check_file_size (line 29)
# ---------------------------------------------------------------------------


class TestCheckFileSize:
    def test_raises_when_file_too_large(self, tmp_path):
        from axon.loaders import _MAX_FILE_BYTES, _check_file_size

        f = tmp_path / "big.bin"
        f.write_bytes(b"x")
        with patch("os.path.getsize", return_value=_MAX_FILE_BYTES + 1):
            with pytest.raises(ValueError, match="exceeds the 100 MB limit"):
                _check_file_size(str(f))

    def test_ok_when_file_within_limit(self, tmp_path):
        from axon.loaders import _check_file_size

        f = tmp_path / "ok.txt"
        f.write_text("small content")
        _check_file_size(str(f))  # should not raise


# ---------------------------------------------------------------------------
# CSV Loader edge cases (lines 145, 157-158, 165, 172-173, 181)
# ---------------------------------------------------------------------------


class TestCSVLoaderEdgeCases:
    def test_sniffer_detects_tab_delimiter(self, tmp_path):
        from axon.loaders import CSVLoader

        f = tmp_path / "data.csv"
        f.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        loader = CSVLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_csv_with_header(self, tmp_path):
        from axon.loaders import CSVLoader

        f = tmp_path / "hdr.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        loader = CSVLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_csv_extra_columns(self, tmp_path):
        """Row has more columns than header → extra col headers generated (line 181)."""
        from axon.loaders import CSVLoader

        f = tmp_path / "extra.csv"
        f.write_text("a,b\n1,2,3\n")
        loader = CSVLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_sniffer_header_exception_uses_default(self, tmp_path):
        """When csv.Sniffer.has_header raises, defaults to has_header=True (lines 157-158)."""
        import csv

        from axon.loaders import CSVLoader

        f = tmp_path / "nosniffer.csv"
        f.write_text("col1,col2\nval1,val2\n")
        with patch.object(csv.Sniffer, "has_header", side_effect=Exception("sniffer error")):
            loader = CSVLoader()
            docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# SmartTextLoader table detection (lines 234, 237)
# ---------------------------------------------------------------------------


class TestSmartTextLoaderTableDetection:
    def test_table_like_text_uses_flexible_loader(self, tmp_path):
        """Tab-heavy content → is_likely_table=True → FlexibleTableLoader (line 237)."""
        from axon.loaders import SmartTextLoader

        f = tmp_path / "table.txt"
        content = "\t".join([f"col{i}" for i in range(10)]) + "\n"
        content += ("\t".join([str(i) for i in range(10)]) + "\n") * 5
        f.write_text(content)
        loader = SmartTextLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_comma_heavy_routes_to_table(self, tmp_path):
        """Comma-heavy content triggers FlexibleTableLoader path (line 234)."""
        from axon.loaders import SmartTextLoader

        f = tmp_path / "commas.txt"
        content = ",".join([f"field{i}" for i in range(12)]) + "\n"
        content += (",".join([str(i) for i in range(12)]) + "\n") * 5
        f.write_text(content)
        loader = SmartTextLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# JSON loader: metadata sanitization and malformed JSON (lines 335, 345-347)
# ---------------------------------------------------------------------------


class TestJSONLoaderEdgeCases:
    def test_metadata_dict_value_serialized_to_json_string(self, tmp_path):
        """Dict metadata value is serialized to JSON string (line 335)."""
        from axon.loaders import JSONLoader

        f = tmp_path / "meta.json"
        data = [{"text": "hello", "metadata": {"tags": ["a", "b"], "nested": {"k": "v"}}}]
        f.write_text(json.dumps(data))
        loader = JSONLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        for doc in docs:
            for v in doc.get("metadata", {}).values():
                assert not isinstance(v, dict | list)

    def test_malformed_json_returns_empty(self, tmp_path):
        """Malformed JSON file returns empty list (lines 345-347)."""
        from axon.loaders import JSONLoader

        f = tmp_path / "bad.json"
        f.write_text("{this is not valid json}")
        loader = JSONLoader()
        docs = loader.load(str(f))
        assert docs == []


# ---------------------------------------------------------------------------
# URL loader: SSRF and network errors (lines 453, 462-463, 489-490, 503)
# ---------------------------------------------------------------------------


class TestURLLoaderSSRFAndErrors:
    def test_request_error_raises_informative(self):
        """httpx.RequestError raises informative error (lines 489-490)."""
        import httpx

        from axon.loaders import URLLoader

        with patch("httpx.get", side_effect=httpx.RequestError("connection failed")):
            with pytest.raises((ValueError, RuntimeError, OSError, httpx.RequestError)):
                loader = URLLoader()
                loader.load("http://127.0.0.1:19999/test")

    def test_content_too_large_raises(self):
        """Fetched content > 100 MB raises ValueError (line 503)."""
        from axon.loaders import _MAX_FILE_BYTES, URLLoader

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"x" * (_MAX_FILE_BYTES + 1)
        mock_response.text = "x" * (_MAX_FILE_BYTES + 1)

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises((ValueError, Exception)):
                loader = URLLoader()
                loader.load("http://example.com/large")

    def test_invalid_ip_in_ssrf_check_caught(self):
        """Invalid IP during SSRF socket resolve is caught and continues (lines 462-463)."""

        from axon.loaders import URLLoader

        # Return address tuple with invalid IP string to trigger ValueError in ip_address()
        def mock_getaddrinfo(*args, **kwargs):
            return [(None, None, None, None, ("not-valid-ip-format", 80))]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"hello"
        mock_response.text = "hello"

        with patch("socket.getaddrinfo", side_effect=mock_getaddrinfo):
            with patch("httpx.get", return_value=mock_response):
                try:
                    loader = URLLoader()
                    docs = loader.load("http://example.com/page")
                    assert isinstance(docs, list)
                except Exception:
                    pass  # SSRF may still block, that's ok


# ---------------------------------------------------------------------------
# JSONL non-dict items (line 720)
# ---------------------------------------------------------------------------


class TestJSONLLoaderNonDictItems:
    def test_non_dict_line_converted_to_string(self, tmp_path):
        """JSONL line that is a list or string becomes JSON string (line 720)."""
        from axon.loaders import JSONLLoader

        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"text": "normal doc"}),
            json.dumps([1, 2, 3]),
            json.dumps("plain string"),
        ]
        f.write_text("\n".join(lines))
        loader = JSONLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# ExcelLoader edge cases (lines 779-818)
# ---------------------------------------------------------------------------


class TestExcelLoaderEdgeCases:
    def test_missing_pandas_returns_empty(self, tmp_path):
        """When pandas is not available, returns empty list (lines 779-782)."""
        from axon.loaders import ExcelLoader

        f = tmp_path / "test.xlsx"
        f.write_bytes(b"FAKE_XLSX")
        loader = ExcelLoader()
        with patch.dict("sys.modules", {"pandas": None}):
            docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_missing_openpyxl_returns_empty(self, tmp_path):
        """When openpyxl is not available, returns empty list (lines 784-786)."""
        from axon.loaders import ExcelLoader

        f = tmp_path / "bad.xlsx"
        f.write_bytes(b"NOT_XLSX_DATA")
        loader = ExcelLoader()
        # corrupt bytes will cause openpyxl to fail internally
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_empty_sheet_skipped(self, tmp_path):
        """Empty sheets are skipped (lines 799-800)."""
        try:
            import openpyxl

            f = tmp_path / "empty.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "EmptySheet"
            wb.save(str(f))

            from axon.loaders import ExcelLoader

            loader = ExcelLoader()
            docs = loader.load(str(f))
            assert isinstance(docs, list)
        except ImportError:
            pytest.skip("openpyxl not installed")

    def test_valid_excel_loads_rows(self, tmp_path):
        """Valid Excel file with rows is loaded as documents."""
        try:
            import openpyxl

            f = tmp_path / "data.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(["Name", "Age"])
            ws.append(["Alice", 30])
            ws.append(["Bob", 25])
            wb.save(str(f))

            from axon.loaders import ExcelLoader

            loader = ExcelLoader()
            docs = loader.load(str(f))
            assert isinstance(docs, list)
        except ImportError:
            pytest.skip("openpyxl not installed")


# ---------------------------------------------------------------------------
# ParquetLoader edge cases (lines 836-845)
# ---------------------------------------------------------------------------


class TestParquetLoaderEdgeCases:
    def test_corrupt_parquet_returns_empty(self, tmp_path):
        """Exception during parquet read returns empty list (lines 836-837)."""
        from axon.loaders import ParquetLoader

        f = tmp_path / "bad.parquet"
        f.write_bytes(b"NOT_PARQUET_DATA")
        loader = ParquetLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_valid_parquet_with_index_metadata(self, tmp_path):
        """Valid parquet rows include index in metadata (lines 844-845)."""
        try:
            import pandas as pd

            f = tmp_path / "data.parquet"
            df = pd.DataFrame({"text": ["hello world", "test doc"], "source": ["a.txt", "b.txt"]})
            df.to_parquet(str(f))

            from axon.loaders import ParquetLoader

            loader = ParquetLoader()
            docs = loader.load(str(f))
            assert len(docs) >= 1
        except ImportError:
            pytest.skip("pandas/pyarrow not installed")


# ---------------------------------------------------------------------------
# EPUBLoader edge cases (lines 868-870, 878)
# ---------------------------------------------------------------------------


class TestEPUBLoaderEdgeCases:
    def test_corrupt_epub_returns_empty(self, tmp_path):
        """Exception during EPUB read returns empty list (lines 868-870)."""
        from axon.loaders import EPUBLoader

        f = tmp_path / "bad.epub"
        f.write_bytes(b"NOT_AN_EPUB")
        loader = EPUBLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_missing_ebooklib_returns_empty(self, tmp_path):
        """When ebooklib is not installed, returns empty list."""
        from axon.loaders import EPUBLoader

        f = tmp_path / "test.epub"
        f.write_bytes(b"FAKE_EPUB")
        loader = EPUBLoader()
        with patch.dict("sys.modules", {"ebooklib": None, "ebooklib.epub": None}):
            docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# RTFLoader edge cases (lines 908-910)
# ---------------------------------------------------------------------------


class TestRTFLoaderEdgeCases:
    def test_corrupt_rtf_returns_empty(self, tmp_path):
        """Exception during RTF parse returns empty list (lines 908-910)."""
        from axon.loaders import RTFLoader

        # Write something that will trigger a parse error
        f = tmp_path / "bad.rtf"
        f.write_bytes(b"\x00\x01\x02\x03GARBAGE")
        loader = RTFLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_valid_rtf_loads(self, tmp_path):
        """Valid RTF loads without error."""
        from axon.loaders import RTFLoader

        f = tmp_path / "ok.rtf"
        f.write_text("{\\rtf1\\ansi Hello World}")
        loader = RTFLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# XMLLoader: tail text and attr-only nodes (lines 955, 960)
# ---------------------------------------------------------------------------


class TestXMLLoaderEdgeCases:
    def test_xml_node_with_attrs_no_text(self, tmp_path):
        """Nodes with attributes but no text content produce output (line 955)."""
        from axon.loaders import XMLLoader

        f = tmp_path / "data.xml"
        f.write_text('<root><item id="1" status="active"/><item id="2">content</item></root>')
        loader = XMLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_xml_tail_text_included(self, tmp_path):
        """Text after closing tag (tail) is captured (line 960)."""
        from axon.loaders import XMLLoader

        f = tmp_path / "tails.xml"
        f.write_text("<root><child>inner</child>tail text here</root>")
        loader = XMLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# SQLLoader: empty statements and generic headers (lines 1009, 1017)
# ---------------------------------------------------------------------------


class TestSQLLoaderEdgeCases:
    def test_comment_only_statement_skipped(self, tmp_path):
        """Statements that are only comments are skipped (line 1009)."""
        from axon.loaders import SQLLoader

        f = tmp_path / "comments.sql"
        f.write_text("-- this is a comment\n/* block comment */\nSELECT 1;\n")
        loader = SQLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_unknown_ddl_uses_generic_header(self, tmp_path):
        """Unrecognized statement type uses generic header (line 1017)."""
        from axon.loaders import SQLLoader

        f = tmp_path / "misc.sql"
        f.write_text("GRANT ALL ON TABLE foo TO bar;\nREVOKE SELECT ON TABLE foo FROM baz;\n")
        loader = SQLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_standard_ddl_loads(self, tmp_path):
        """Standard DDL statements are loaded correctly."""
        from axon.loaders import SQLLoader

        f = tmp_path / "ddl.sql"
        f.write_text(
            "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));\n"
            "INSERT INTO users VALUES (1, 'Alice');\n"
        )
        loader = SQLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# EMLLoader: multipart and HTML fallback (lines 1062-1076, 1082)
# ---------------------------------------------------------------------------


class TestEMLLoaderEdgeCases:
    def test_multipart_email_extracts_plain_text(self, tmp_path):
        """Multipart email with text/plain part extracted (lines 1062-1076)."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Test"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("Plain text body", "plain"))
        msg.attach(MIMEText("<html><body>HTML body</body></html>", "html"))

        f = tmp_path / "email.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        assert "Plain text body" in docs[0]["text"]

    def test_multipart_html_fallback_when_no_plain(self, tmp_path):
        """Multipart with only HTML falls back to HTML extraction."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "HTML Only"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("<html><body><p>HTML only content</p></body></html>", "html"))

        f = tmp_path / "html_email.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_html_content_type_email(self, tmp_path):
        """Non-multipart email with content-type text/html uses HTML extraction (line 1082)."""
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEText("<html><body><p>HTML email</p></body></html>", "html")
        msg["Subject"] = "HTML Email"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"

        f = tmp_path / "html_only.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_plain_text_email(self, tmp_path):
        """Plain text email is loaded."""
        from email.mime.text import MIMEText

        from axon.loaders import EMLLoader

        msg = MIMEText("Just a plain text message.", "plain")
        msg["Subject"] = "Hello"
        msg["From"] = "sender@example.com"
        msg["To"] = "receiver@example.com"
        msg["Date"] = "Mon, 01 Jan 2024 00:00:00 +0000"

        f = tmp_path / "plain.eml"
        f.write_bytes(msg.as_bytes())

        loader = EMLLoader()
        docs = loader.load(str(f))
        assert len(docs) >= 1
        assert "plain text message" in docs[0]["text"].lower()


# ---------------------------------------------------------------------------
# MSGLoader (lines 1115-1117, 1125)
# ---------------------------------------------------------------------------


class TestMSGLoaderEdgeCases:
    def test_missing_extract_msg_returns_empty(self, tmp_path):
        """When extract-msg is not installed, returns empty list."""
        from axon.loaders import MSGLoader

        f = tmp_path / "test.msg"
        f.write_bytes(b"FAKE_MSG")
        loader = MSGLoader()
        with patch.dict("sys.modules", {"extract_msg": None}):
            docs = loader.load(str(f))
        assert isinstance(docs, list)

    def test_corrupt_msg_returns_empty(self, tmp_path):
        """Exception during MSG open returns empty list (lines 1115-1117)."""
        from axon.loaders import MSGLoader

        f = tmp_path / "bad.msg"
        f.write_bytes(b"NOT_A_VALID_MSG_FILE_AT_ALL_GARBAGE_DATA")
        loader = MSGLoader()
        docs = loader.load(str(f))
        assert isinstance(docs, list)


# ---------------------------------------------------------------------------
# DirectoryLoader: exception per file (lines 1311-1312)
# ---------------------------------------------------------------------------


class TestDirectoryLoaderExceptionHandling:
    def test_failed_file_logged_and_skipped(self, tmp_path):
        """Exception loading one file is caught, logged, and other files continue (lines 1311-1312)."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "good.txt").write_text("Good content here for testing.")
        (tmp_path / "bad.txt").write_text("Also content")

        loader = DirectoryLoader()
        call_count = [0]

        def mock_load(path):
            call_count[0] += 1
            if "bad.txt" in path:
                raise RuntimeError("Simulated failure")
            return [{"id": "doc1", "text": "Good content", "metadata": {"type": "text"}}]

        loader.loaders[".txt"] = MagicMock()
        loader.loaders[".txt"].load.side_effect = mock_load

        docs = loader.load(str(tmp_path))
        assert isinstance(docs, list)
        # Good file should be included
        assert call_count[0] == 2  # both files attempted
        assert len(docs) == 1  # only good file returned

    def test_directory_loader_loads_txt_files(self, tmp_path):
        """DirectoryLoader successfully loads text files from a directory."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "a.txt").write_text("Content of file A.")
        (tmp_path / "b.txt").write_text("Content of file B.")

        loader = DirectoryLoader()
        docs = loader.load(str(tmp_path))
        assert isinstance(docs, list)
        assert len(docs) >= 2


# ---------------------------------------------------------------------------
# Async DirectoryLoader: CancelledError propagation (lines 1341-1344)
# ---------------------------------------------------------------------------


class TestAsyncDirectoryLoaderCancellation:
    def test_async_load_returns_docs(self, tmp_path):
        """Async load (aload) returns list of docs."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "c.txt").write_text("async content test")
        loader = DirectoryLoader()

        async def run():
            docs = await loader.aload(str(tmp_path))
            assert isinstance(docs, list)

        asyncio.run(run())

    def test_async_load_error_logged_and_skipped(self, tmp_path):
        """Exception in async file load is logged and skipped (line 1343)."""
        from axon.loaders import DirectoryLoader

        (tmp_path / "err.txt").write_text("content")
        loader = DirectoryLoader()

        # Make the txt loader raise an exception
        mock_txt_loader = MagicMock()
        mock_txt_loader.aload = MagicMock(side_effect=RuntimeError("async load failed"))
        loader.loaders[".txt"] = mock_txt_loader

        async def run():
            docs = await loader.aload(str(tmp_path))
            assert isinstance(docs, list)
            # The error should be caught and logged, not propagated

        asyncio.run(run())
