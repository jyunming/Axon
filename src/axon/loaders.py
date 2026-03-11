import asyncio
import io
import ipaddress
import json
import logging
import os
import socket
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("Axon.Loaders")

_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def _check_file_size(path: str) -> None:
    """Raise ValueError if the file exceeds _MAX_FILE_BYTES."""
    size = os.path.getsize(path)
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File '{path}' is {size / (1024 * 1024):.1f} MB, " f"which exceeds the 100 MB limit."
        )


def _extract_html_text(html: str) -> str:
    """Extract visible text from an HTML string using stdlib html.parser."""
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._texts = []
            self._skip_tags = {"script", "style", "head", "meta", "link"}
            self._current_skip = False
            self._skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag.lower() in self._skip_tags:
                self._current_skip = True
                self._skip_depth += 1

        def handle_endtag(self, tag):
            if tag.lower() in self._skip_tags and self._skip_depth > 0:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._current_skip = False

        def handle_data(self, data):
            if not self._current_skip:
                stripped = data.strip()
                if stripped:
                    self._texts.append(stripped)

        def get_text(self):
            return " ".join(self._texts)

    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


class BaseLoader:
    """Base class for document loaders."""

    def load(self, path: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def aload(self, path: str) -> list[dict[str, Any]]:
        """Async version of load."""
        return await asyncio.to_thread(self.load, path)


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return [
            {
                "id": os.path.basename(path),
                "text": content,
                "metadata": {"source": path, "type": "text"},
            }
        ]


class TSVLoader(BaseLoader):
    """Loader for tab-delimited files."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        df = pd.read_csv(path, sep="	")
        documents = []
        for i, row in df.iterrows():
            # Assume first column or 'content' column is the text
            text_col = "content" if "content" in df.columns else df.columns[0]
            text = str(row[text_col])
            metadata = row.drop(text_col).to_dict()
            metadata.update({"source": path, "type": "tsv", "row": i})
            documents.append(
                {"id": f"{os.path.basename(path)}_row_{i}", "text": text, "metadata": metadata}
            )
        return documents


class JSONLoader(BaseLoader):
    """Loader for JSON files."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping malformed JSON file {path}: {exc}")
                return []

        if isinstance(data, list):
            documents = []
            for i, item in enumerate(data):
                text = item.get("text", item.get("content", json.dumps(item)))
                metadata = {k: v for k, v in item.items() if k not in ["text", "content"]}
                metadata.update({"source": path, "type": "json", "index": i})
                documents.append(
                    {"id": f"{os.path.basename(path)}_{i}", "text": text, "metadata": metadata}
                )
            return documents
        else:
            text = data.get("text", data.get("content", json.dumps(data)))
            metadata = {k: v for k, v in data.items() if k not in ["text", "content"]}
            metadata.update({"source": path, "type": "json"})
            return [{"id": os.path.basename(path), "text": text, "metadata": metadata}]


class CSVLoader(BaseLoader):
    """Loader for CSV files. Each row becomes a document."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        import csv

        documents = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Use 'text' or 'content' column if present, else join all values
                text_col = next((c for c in ["text", "content", "body"] if c in row), None)
                if text_col:
                    text = row[text_col]
                    metadata = {k: v for k, v in row.items() if k != text_col}
                else:
                    text = " | ".join(f"{k}: {v}" for k, v in row.items())
                    metadata = {}
                metadata.update({"source": path, "type": "csv", "row": i})
                documents.append(
                    {"id": f"{os.path.basename(path)}_row_{i}", "text": text, "metadata": metadata}
                )
        return documents


class HTMLLoader(BaseLoader):
    """Loader for HTML files. Extracts visible text content."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        text = _extract_html_text(content)
        return [
            {
                "id": os.path.basename(path),
                "text": text,
                "metadata": {"source": path, "type": "html"},
            }
        ]


class URLLoader(BaseLoader):
    """Loader for HTTP/HTTPS URLs with SSRF mitigations.

    Fetches the URL using httpx (already in project deps), strips HTML if the
    response is text/html, and returns a single document dict.

    Security:
    - Only http/https schemes are accepted; all others raise ValueError.
    - Resolved IP addresses are checked against blocked private/internal ranges
      to prevent Server-Side Request Forgery (SSRF).
    - Redirects are capped at 5 hops.
    - Response content type must be text/*; binary responses are rejected.
    - Content size is capped at _MAX_FILE_BYTES (100 MB).
    """

    _BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
        ipaddress.ip_network("127.0.0.0/8"),  # loopback
        ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918 private
        ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918 private
        ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918 private
        ipaddress.ip_network("169.254.0.0/16"),  # link-local / cloud metadata (AWS/GCP/Azure)
        ipaddress.ip_network("::1/128"),  # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
        ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ]

    def _check_ssrf(self, url: str) -> None:
        """Raise ValueError if the URL targets a private/internal address."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Scheme '{parsed.scheme}' is not allowed. Only http and https are permitted."
            )
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("No hostname found in URL.")
        try:
            addrinfos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve hostname '{hostname}': {exc}")
        for addrinfo in addrinfos:
            ip_str = addrinfo[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            for blocked in self._BLOCKED_NETWORKS:
                if ip in blocked:
                    raise ValueError(
                        f"Host '{hostname}' resolves to a blocked private/internal address ({ip})."
                    )

    def load(self, url: str) -> list[dict[str, Any]]:
        """Fetch *url* and return its text content as a single document."""
        import uuid

        import httpx

        self._check_ssrf(url)
        try:
            resp = httpx.get(url, timeout=30.0, max_redirects=5, follow_redirects=True)
        except httpx.TimeoutException:
            raise ValueError(f"Request to '{url}' timed out.")
        except httpx.TooManyRedirects:
            raise ValueError(f"Too many redirects fetching '{url}'.")
        except httpx.RequestError as exc:
            raise ValueError(f"Failed to fetch '{url}': {exc}")

        if resp.status_code != 200:
            raise ValueError(f"'{url}' returned HTTP {resp.status_code}.")

        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("text/"):
            raise ValueError(
                f"Non-text content type '{content_type}' for '{url}'. Only text/* is accepted."
            )

        raw = resp.text
        if len(raw.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ValueError(
                f"URL content exceeds the {_MAX_FILE_BYTES // (1024 * 1024)} MB limit."
            )

        text = _extract_html_text(raw) if "html" in content_type else raw
        return [
            {
                "id": uuid.uuid4().hex,
                "text": text,
                "metadata": {"source": url, "type": "url"},
            }
        ]


class DOCXLoader(BaseLoader):
    """Loader for DOCX files using python-docx."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return []

        doc = Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)

        return [
            {
                "id": os.path.basename(path),
                "text": text,
                "metadata": {"source": path, "type": "docx"},
            }
        ]


class PPTXLoader(BaseLoader):
    """Loader for PPTX files using python-pptx."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            from pptx import Presentation
        except ImportError:
            logger.error("python-pptx not installed. Install with: pip install python-pptx")
            return []

        prs = Presentation(path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text.strip())
        text = "\n\n".join([t for t in text_runs if t])

        return [
            {
                "id": os.path.basename(path),
                "text": text,
                "metadata": {"source": path, "type": "pptx"},
            }
        ]


class LegacyOfficeLoader(BaseLoader):
    """Loader for legacy binary Office files (.doc, .ppt) using textract."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            import textract
        except ImportError:
            logger.error("textract not installed. Install with: pip install textract")
            return []

        try:
            text = textract.process(path).decode("utf-8")
            return [
                {
                    "id": os.path.basename(path),
                    "text": text,
                    "metadata": {"source": path, "type": "legacy_office"},
                }
            ]
        except Exception as e:
            logger.error(f"Error extracting text from {path}: {e}")
            return []


class ImageLoader(BaseLoader):
    """
    Loader for raster images (BMP, PNG, JPG, TIF/TIFF, PGM) using Ollama VLM for captioning.

    Images are converted to PNG bytes via Pillow before being sent to the VLM so that
    all formats — including exotic ones like PGM — are handled uniformly.
    """

    def __init__(self, ollama_model: str = "llava"):
        self.ollama_model = ollama_model
        try:
            from PIL import Image

            self._pil = Image
        except ImportError:
            self._pil = None
            logger.warning("Pillow not installed. Image loading will be skipped.")
        try:
            import ollama

            self.ollama = ollama
        except ImportError:
            self.ollama = None
            logger.warning("ollama package not installed. Image loading will be skipped.")

    def load(self, path: str) -> list[dict[str, Any]]:
        if self._pil is None or self.ollama is None:
            return []

        logger.info(f"🖼️ Processing image: {path} with {self.ollama_model}...")

        try:
            # Normalize to PNG bytes via Pillow for maximum VLM compatibility
            img = self._pil.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_data = buf.getvalue()

            fmt = Path(path).suffix.lstrip(".").lower()

            # Call Ollama VLM
            response = self.ollama.generate(
                model=self.ollama_model,
                prompt="Describe this image in detail. Mention any text, objects, people, or patterns visible.",
                images=[image_data],
            )

            description = response["response"]

            return [
                {
                    "id": os.path.basename(path),
                    "text": f"Image Description: {description}",
                    "metadata": {
                        "source": path,
                        "type": "image",
                        "format": fmt,
                        "model": self.ollama_model,
                    },
                }
            ]
        except Exception as e:
            logger.error(f"Error processing image {path}: {e}")
            return []


# Backward-compatible alias kept for existing code that imports BMPLoader directly.
BMPLoader = ImageLoader


class PDFLoader(BaseLoader):
    """Loader for PDF files. Extracts text page-by-page using PyMuPDF (fitz) with pypdf fallback."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(path)
            total = len(doc)
            documents = []
            for page_num in range(total):
                page = doc[page_num]
                text = page.get_text()
                documents.append(
                    {
                        "id": f"{os.path.basename(path)}_page_{page_num}",
                        "text": text,
                        "metadata": {
                            "source": path,
                            "type": "pdf",
                            "page": page_num,
                            "total_pages": total,
                        },
                    }
                )
            doc.close()
            return documents
        except ImportError:
            pass

        try:
            import pypdf

            reader = pypdf.PdfReader(path)
            total = len(reader.pages)
            documents = []
            for page_num in range(total):
                text = reader.pages[page_num].extract_text() or ""
                documents.append(
                    {
                        "id": f"{os.path.basename(path)}_page_{page_num}",
                        "text": text,
                        "metadata": {
                            "source": path,
                            "type": "pdf",
                            "page": page_num,
                            "total_pages": total,
                        },
                    }
                )
            return documents
        except ImportError:
            pass

        logger.error(
            "Neither PyMuPDF (fitz) nor pypdf is installed. "
            "Install with: pip install pymupdf>=1.24.0 or pypdf>=4.0.0"
        )
        return []


class DirectoryLoader:
    """Loader that crawls a directory and uses appropriate loaders for each file."""

    def __init__(self, vlm_model: str = "llava"):
        _image_loader = ImageLoader(ollama_model=vlm_model)
        _legacy_office_loader = LegacyOfficeLoader()
        self.loaders = {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".tsv": TSVLoader(),
            ".json": JSONLoader(),
            ".csv": CSVLoader(),
            ".html": HTMLLoader(),
            ".htm": HTMLLoader(),
            ".docx": DOCXLoader(),
            ".pptx": PPTXLoader(),
            ".doc": _legacy_office_loader,
            ".ppt": _legacy_office_loader,
            ".bmp": _image_loader,
            ".png": _image_loader,
            ".jpg": _image_loader,
            ".jpeg": _image_loader,
            ".tif": _image_loader,
            ".tiff": _image_loader,
            ".pgm": _image_loader,
            ".pdf": PDFLoader(),
        }

    def load(self, directory: str) -> list[dict[str, Any]]:
        all_documents = []
        path = Path(directory)

        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in self.loaders:
                loader = self.loaders[file_path.suffix.lower()]
                try:
                    docs = loader.load(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        return all_documents

    async def aload(self, directory: str) -> list[dict[str, Any]]:
        """Async version of load_directory — caps concurrency at 32 tasks."""
        path = Path(directory)
        semaphore = asyncio.Semaphore(32)

        async def _load_with_semaphore(loader, file_path: str):
            async with semaphore:
                return await loader.aload(file_path)

        tasks = []
        for file_path in path.rglob("*"):
            suffix = file_path.suffix.lower()
            if suffix in self.loaders:
                loader = self.loaders[suffix]
                tasks.append(_load_with_semaphore(loader, str(file_path)))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_documents = []
        for res in results:
            if isinstance(res, BaseException):
                if isinstance(res, asyncio.CancelledError):
                    raise res
                logger.warning("async file load failed: %s", res)
                continue
            all_documents.extend(res)

        return all_documents
