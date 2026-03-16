import asyncio
import io
import ipaddress
import json
import logging
import os
import re
import socket
import urllib.parse
from pathlib import Path
from typing import Any

logger = logging.getLogger("Axon.Loaders")

_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def _check_file_size(path: str) -> None:
    """Raise ValueError if the file exceeds _MAX_FILE_BYTES."""
    size = os.path.getsize(path)
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File '{path}' is {size / (1024 * 1024):.1f} MB, " f"which exceeds the 100 MB limit."
        )


def _rewrite_github_url(url: str) -> str:
    """Rewrite GitHub web URLs to their raw/fetchable equivalents.

    Supported rewrites:
    - ``github.com/<owner>/<repo>/blob/<ref>/<path>``
      → ``raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>``
    - ``gist.github.com/<user>/<gist_id>``
      → ``gist.githubusercontent.com/<user>/<gist_id>/raw``

    Raises ``ValueError`` for GitHub tree (directory listing) URLs because they
    cannot be fetched as a single document.  Use a specific file URL
    (``/blob/...``) or clone the repo and use ``ingest_path`` instead.

    All other URLs are returned unchanged.
    """
    # github.com blob URL → raw URL
    blob_match = re.match(
        r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", url, re.IGNORECASE
    )
    if blob_match:
        owner, repo, ref, path = blob_match.groups()
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

    # github.com tree URL → helpful error (directory listings can't be one doc)
    if re.match(r"^https?://github\.com/[^/]+/[^/]+/tree/", url, re.IGNORECASE):
        raise ValueError(
            "GitHub directory (tree) URLs cannot be fetched as a single document. "
            "Provide a specific file URL ending in /blob/<ref>/<file>, or clone "
            "the repository locally and use ingest_path instead."
        )

    # gist.github.com URL → raw gist URL
    gist_match = re.match(r"^https?://gist\.github\.com/([^/]+)/([a-f0-9]+)/?$", url, re.IGNORECASE)
    if gist_match:
        user, gist_id = gist_match.groups()
        return f"https://gist.githubusercontent.com/{user}/{gist_id}/raw"

    return url


def _extract_html_text(html: str) -> str:
    """Extract visible text from an HTML string using stdlib html.parser."""
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._texts = []
            self._content_skip_tags = {"script", "style"}
            self._current_skip = False

        def handle_starttag(self, tag, attrs):
            if tag.lower() in self._content_skip_tags:
                self._current_skip = True

        def handle_endtag(self, tag):
            if tag.lower() in self._content_skip_tags:
                self._current_skip = False

        def handle_data(self, data):
            if not self._current_skip:
                stripped = data.strip()
                if stripped:
                    self._texts.append(stripped)

        def get_text(self) -> str:
            return " ".join(self._texts)

    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


class BaseLoader:
    """Base class for document loaders."""

    def load(self, path: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def aload(self, path: str) -> list[dict[str, Any]]:
        """Async version of load."""
        return await asyncio.to_thread(self.load, path)


class FlexibleTableLoader(BaseLoader):
    """
    Robust loader for tabular data (CSV, TSV, etc.) that handles:
    1. Automatic delimiter detection (sniffing).
    2. Ragged/jagged rows (varying column counts).
    3. Missing or present headers.
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.load_text(content, source=path)

    def load_text(self, text: str, source: str = "raw_text") -> list[dict[str, Any]]:
        """Process raw text as a table."""
        import csv
        import io

        from axon.splitters import TableSplitter

        sample = text[:8192]

        # Detect delimiter
        try:
            # Priority 1: Sniffer
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
            delimiter = dialect.delimiter
        except Exception:
            # Priority 2: Simple heuristics
            counts = {"\t": sample.count("\t"), ",": sample.count(","), "|": sample.count("|")}
            delimiter = max(counts, key=lambda k: counts[k])
            # If everything is 0, default to comma
            if counts[delimiter] == 0:
                delimiter = ","

        # Detect header
        try:
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            has_header = True

        f = io.StringIO(text)
        reader = csv.reader(f, delimiter=delimiter)
        all_rows = [r for r in reader if r]  # Skip empty rows

        if not all_rows:
            return []

        # Find the max column count across ALL rows
        max_cols = max(len(row) for row in all_rows)

        # Determine headers and data
        if has_header:
            base_headers = all_rows[0]
            data_rows = all_rows[1:]
        else:
            base_headers = [f"Col_{i}" for i in range(len(all_rows[0]))]
            data_rows = all_rows

        # Ensure header count matches max_cols
        headers = list(base_headers)
        while len(headers) < max_cols:
            headers.append(f"Extra_Col_{len(headers)}")
        # If actual headers were shorter than row 0, handle that
        headers = headers[:max_cols]

        final_rows = []
        for row in data_rows:
            row_dict = {}
            for i, val in enumerate(row):
                if i < len(headers):
                    row_dict[headers[i]] = val.strip()
            final_rows.append(row_dict)

        table_name = os.path.basename(source)
        splitter = TableSplitter(table_name=table_name)
        chunks = splitter.transform_rows(final_rows, headers)

        documents = []
        for i, text_chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"{table_name}_row_{i}",
                    "text": text_chunk,
                    "metadata": {"source": source, "type": "table", "row": i},
                }
            )
        return documents


class SmartTextLoader(BaseLoader):
    """
    Delegates to FlexibleTableLoader if the file looks structured,
    otherwise uses TextLoader.
    """

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.load_text(content, source=path)

    def load_text(self, text: str, source: str = "raw_text") -> list[dict[str, Any]]:
        """Intelligently detect if text is a table or prose and load accordingly."""
        # Check if it looks like a table (multiple tabs or commas per line)
        sample = text[:2048]
        lines = sample.strip().split("\n")
        tab_count = sample.count("\t")
        comma_count = sample.count(",")

        is_likely_table = False
        if len(lines) > 1:
            avg_tabs = tab_count / len(lines)
            avg_commas = comma_count / len(lines)
            if avg_tabs > 1.5 or avg_commas > 2.0:
                is_likely_table = True

        if is_likely_table:
            return FlexibleTableLoader().load_text(text, source=source)
        else:
            # Fallback to standard document dictionary format.
            # Use the full source as the id (already unique when source is a
            # UUID doc_id); os.path.basename strips directory context and can
            # produce duplicate ids for non-path sources on Windows.
            return [
                {
                    "id": source,
                    "text": text,
                    "metadata": {"source": source, "type": "text"},
                }
            ]


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
    """Loader for tab-delimited files. Each row becomes an enriched document chunk."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        import csv

        from axon.splitters import TableSplitter

        rows = []
        headers: list[str] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            headers = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)

        table_name = os.path.basename(path)
        splitter = TableSplitter(table_name=table_name)
        chunks = splitter.transform_rows(rows, headers)

        documents = []
        for i, text in enumerate(chunks):
            documents.append(
                {
                    "id": f"{table_name}_row_{i}",
                    "text": text,
                    "metadata": {"source": path, "type": "tsv", "row": i},
                }
            )
        return documents


class JSONLoader(BaseLoader):
    """Loader for JSON files."""

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata values are scalar (str, int, float, bool)."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, dict | list):
                sanitized[k] = json.dumps(v)
            else:
                sanitized[k] = v
        return sanitized

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
                metadata = self._sanitize_metadata(metadata)
                metadata.update({"source": path, "type": "json", "index": i})
                documents.append(
                    {"id": f"{os.path.basename(path)}_{i}", "text": text, "metadata": metadata}
                )
            return documents
        else:
            text = data.get("text", data.get("content", json.dumps(data)))
            metadata = {k: v for k, v in data.items() if k not in ["text", "content"]}
            metadata = self._sanitize_metadata(metadata)
            metadata.update({"source": path, "type": "json"})
            return [{"id": os.path.basename(path), "text": text, "metadata": metadata}]


class CSVLoader(BaseLoader):
    """Loader for CSV files. Each row becomes an enriched document chunk."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        import csv

        from axon.splitters import TableSplitter

        rows = []
        headers: list[str] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)

        table_name = os.path.basename(path)
        splitter = TableSplitter(table_name=table_name)
        chunks = splitter.transform_rows(rows, headers)

        documents = []
        for i, text in enumerate(chunks):
            documents.append(
                {
                    "id": f"{table_name}_row_{i}",
                    "text": text,
                    "metadata": {"source": path, "type": "csv", "row": i},
                }
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
        """Fetch *url* and return its text content as a single document.

        GitHub ``/blob/`` URLs are automatically rewritten to
        ``raw.githubusercontent.com`` before any network contact is made.
        GitHub Gist URLs are rewritten to their raw form.
        GitHub tree (directory) URLs raise ``ValueError`` immediately.
        """
        import uuid

        import httpx

        url = _rewrite_github_url(url)  # Gap-5: rewrite before SSRF check
        self._check_ssrf(url)
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True, max_redirects=5) as client:
                resp = client.get(url)
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

            self._pil: Any = Image
        except ImportError:
            self._pil = None
            logger.warning("Pillow not installed. Image loading will be skipped.")
        try:
            import ollama

            self.ollama: Any = ollama
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


class JSONLLoader(BaseLoader):
    """Loader for newline-delimited JSON files (.jsonl, .ndjson). Each line becomes a document."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        fname = os.path.basename(path)
        documents = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {i + 1} in {path}")
                    continue
                if isinstance(item, dict):
                    text = item.get("text", item.get("content", json.dumps(item)))
                else:
                    text = json.dumps(item)
                documents.append(
                    {
                        "id": f"{fname}_{i}",
                        "text": text,
                        "metadata": {"source": path, "type": "jsonl", "line": i},
                    }
                )
        return documents


class NotebookLoader(BaseLoader):
    """Loader for Jupyter notebooks (.ipynb). Extracts markdown and code cells as documents."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8") as f:
            try:
                nb = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping malformed notebook {path}: {exc}")
                return []

        fname = os.path.basename(path)
        documents = []
        for i, cell in enumerate(nb.get("cells", [])):
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else source
            text = text.strip()
            if not text:
                continue
            if cell_type == "code":
                text = f"```python\n{text}\n```"
            documents.append(
                {
                    "id": f"{fname}_cell_{i}",
                    "text": text,
                    "metadata": {
                        "source": path,
                        "type": "notebook",
                        "cell_type": cell_type,
                        "cell_index": i,
                    },
                }
            )
        return documents


class ExcelLoader(BaseLoader):
    """Loader for Excel files (.xlsx, .xls) using pandas. Each sheet is loaded as a table."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed.")
            return []
        try:
            xl = pd.ExcelFile(path)
        except ImportError:
            logger.error("openpyxl not installed. Install with: pip install openpyxl")
            return []
        except Exception as exc:
            logger.warning(f"Could not open Excel file {path}: {exc}")
            return []

        from axon.splitters import TableSplitter

        fname = os.path.basename(path)
        documents = []
        doc_idx = 0
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name).fillna("").astype(str)
            except Exception as exc:
                logger.warning(f"Skipping sheet '{sheet_name}' in {path}: {exc}")
                continue
            if df.empty:
                continue
            headers = list(df.columns)
            rows = df.to_dict(orient="records")
            splitter = TableSplitter(table_name=f"{fname}[{sheet_name}]")
            for text in splitter.transform_rows(rows, headers):
                documents.append(
                    {
                        "id": f"{fname}_{sheet_name}_{doc_idx}",
                        "text": text,
                        "metadata": {
                            "source": path,
                            "type": "excel",
                            "sheet": sheet_name,
                            "row": doc_idx,
                        },
                    }
                )
                doc_idx += 1
        return documents


class ParquetLoader(BaseLoader):
    """Loader for Parquet files using pandas + pyarrow."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed.")
            return []
        try:
            df = pd.read_parquet(path).fillna("").astype(str)
        except ImportError:
            logger.error("pyarrow not installed. Install with: pip install pyarrow")
            return []
        except Exception as exc:
            logger.warning(f"Could not read Parquet file {path}: {exc}")
            return []

        from axon.splitters import TableSplitter

        fname = os.path.basename(path)
        splitter = TableSplitter(table_name=fname)
        chunks = splitter.transform_rows(df.to_dict(orient="records"), list(df.columns))
        return [
            {
                "id": f"{fname}_{i}",
                "text": text,
                "metadata": {"source": path, "type": "parquet", "row": i},
            }
            for i, text in enumerate(chunks)
        ]


class EPUBLoader(BaseLoader):
    """Loader for EPUB ebooks using ebooklib. Each document item becomes a chunk."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            logger.error("ebooklib not installed. Install with: pip install ebooklib")
            return []
        try:
            book = epub.read_epub(path, options={"ignore_ncx": True})
        except Exception as exc:
            logger.warning(f"Could not read EPUB {path}: {exc}")
            return []

        fname = os.path.basename(path)
        documents = []
        for i, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            html = item.get_content().decode("utf-8", errors="ignore")
            text = _extract_html_text(html).strip()
            if not text:
                continue
            documents.append(
                {
                    "id": f"{fname}_chapter_{i}",
                    "text": text,
                    "metadata": {
                        "source": path,
                        "type": "epub",
                        "chapter": i,
                        "item_name": item.get_name(),
                    },
                }
            )
        return documents


class RTFLoader(BaseLoader):
    """Loader for RTF files using striprtf."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            logger.error("striprtf not installed. Install with: pip install striprtf")
            return []
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            text = rtf_to_text(content).strip()
        except Exception as exc:
            logger.warning(f"Could not parse RTF file {path}: {exc}")
            return []
        return [
            {
                "id": os.path.basename(path),
                "text": text,
                "metadata": {"source": path, "type": "rtf"},
            }
        ]


class XMLLoader(BaseLoader):
    """Loader for XML files. Strips tags and extracts clean text content."""

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            logger.warning(f"Could not parse XML {path}: {exc}")
            # Fall back to stripping tags with regex
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            return [
                {
                    "id": os.path.basename(path),
                    "text": text,
                    "metadata": {"source": path, "type": "xml"},
                }
            ]

        root = tree.getroot()
        parts: list[str] = []

        def _walk(node: ET.Element, depth: int = 0) -> None:
            tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
            attrs = " ".join(f'{k.split("}")[-1]}="{v}"' for k, v in node.attrib.items())
            header = f"<{tag}{' ' + attrs if attrs else ''}>"
            text = (node.text or "").strip()
            if text:
                parts.append(f"{header} {text}")
            elif node.attrib:
                parts.append(header)
            for child in node:
                _walk(child, depth + 1)
            tail = (node.tail or "").strip()
            if tail:
                parts.append(tail)

        _walk(root)
        full_text = "\n".join(parts)
        return [
            {
                "id": os.path.basename(path),
                "text": full_text,
                "metadata": {"source": path, "type": "xml"},
            }
        ]


class SQLLoader(BaseLoader):
    """Loader for SQL files. Extracts statements with natural-language descriptions."""

    # Patterns that identify DDL/DML statement types
    _STMT_RE = re.compile(
        r"^\s*(CREATE\s+(?:TABLE|VIEW|INDEX|PROCEDURE|FUNCTION|TRIGGER)|"
        r"INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|ALTER\s+TABLE|"
        r"DROP\s+(?:TABLE|VIEW)|SELECT\b)",
        re.IGNORECASE | re.MULTILINE,
    )

    def load(self, path: str) -> list[dict[str, Any]]:
        _check_file_size(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split on statement boundaries (semicolons followed by whitespace/newline)
        raw_stmts = re.split(r";\s*\n", content)
        fname = os.path.basename(path)
        documents = []

        for i, stmt in enumerate(raw_stmts):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue

            # Extract comments preceding the statement as context
            comment_lines = [
                line.lstrip("- ").strip()
                for line in stmt.splitlines()
                if line.strip().startswith("--")
            ]
            code = "\n".join(
                line for line in stmt.splitlines() if not line.strip().startswith("--")
            ).strip()
            if not code:
                continue

            # Build a natural-language header from the first keyword + object name
            m = self._STMT_RE.match(code)
            if m:
                first_line = code.splitlines()[0]
                header = f"SQL statement: {first_line.strip()}"
            else:
                header = "SQL statement"

            comment_text = " ".join(comment_lines)
            text = f"{header}\n{comment_text}\n\n{code}".strip()

            documents.append(
                {
                    "id": f"{fname}_stmt_{i}",
                    "text": text,
                    "metadata": {"source": path, "type": "sql", "statement_index": i},
                }
            )

        # If no semicolon-delimited statements found, return the whole file
        if not documents:
            return [
                {
                    "id": fname,
                    "text": content.strip(),
                    "metadata": {"source": path, "type": "sql"},
                }
            ]

        return documents


class DirectoryLoader:
    """Loader that crawls a directory and uses appropriate loaders for each file."""

    def __init__(self, vlm_model: str = "llava"):
        _image_loader = ImageLoader(ollama_model=vlm_model)
        _excel_loader = ExcelLoader()
        self.loaders = {
            ".txt": SmartTextLoader(),
            ".md": TextLoader(),
            ".tsv": FlexibleTableLoader(),
            ".json": JSONLoader(),
            ".jsonl": JSONLLoader(),
            ".ndjson": JSONLLoader(),
            ".ipynb": NotebookLoader(),
            ".csv": FlexibleTableLoader(),
            ".xlsx": _excel_loader,
            ".xls": _excel_loader,
            ".parquet": ParquetLoader(),
            ".epub": EPUBLoader(),
            ".rtf": RTFLoader(),
            ".html": HTMLLoader(),
            ".htm": HTMLLoader(),
            ".xml": XMLLoader(),
            ".sql": SQLLoader(),
            ".docx": DOCXLoader(),
            ".pptx": PPTXLoader(),
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
                    # Native Path Enrichment (Breadcrumbs) for textual docs
                    for doc in docs:
                        if doc["metadata"].get("type") not in (
                            "csv",
                            "tsv",
                            "excel",
                            "parquet",
                            "image",
                        ):
                            rel_path = os.path.relpath(str(file_path), directory)
                            doc["text"] = f"[File Path: {rel_path}]\n{doc['text']}"
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
        file_paths_for_tasks = []
        for file_path in path.rglob("*"):
            suffix = file_path.suffix.lower()
            if suffix in self.loaders:
                loader = self.loaders[suffix]
                tasks.append(_load_with_semaphore(loader, str(file_path)))
                file_paths_for_tasks.append(file_path)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_documents = []
        for file_path, res in zip(file_paths_for_tasks, results):
            if isinstance(res, BaseException):
                if isinstance(res, asyncio.CancelledError):
                    raise res
                logger.warning("async file load failed: %s", res)
                continue
            for doc in res:
                if doc["metadata"].get("type") not in ("csv", "tsv", "image"):
                    rel_path = os.path.relpath(str(file_path), directory)
                    doc["text"] = f"[File Path: {rel_path}]\n{doc['text']}"
            all_documents.extend(res)

        return all_documents
