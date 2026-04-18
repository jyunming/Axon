import logging
import os
import re
from typing import Any

logger = logging.getLogger("Axon.Splitters")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex, re-joining common abbreviations."""
    # Split on period, exclamation, or question mark followed by whitespace and an uppercase letter.
    # We capture the punctuation and space to preserve it, then clean up.
    parts = re.split(r"([.!?]+)\s+", text.strip())

    sentences = []
    current_sentence = ""

    abbreviations = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "rev",
        "st",
        "gen",
        "rep",
        "sen",
        "ph.d",
        "m.d",
        "b.a",
        "m.a",
        "b.s",
        "m.s",
        "e.g",
        "i.e",
        "etc",
    }

    for i in range(0, len(parts), 2):
        text_part = parts[i].strip()
        if not text_part:
            continue

        punct_part = parts[i + 1] if i + 1 < len(parts) else ""

        # If current_sentence is empty, just start it. Otherwise, append.
        if current_sentence:
            current_sentence += " " + text_part + punct_part
        else:
            current_sentence = text_part + punct_part

        # Check if the text_part ends with a known abbreviation
        words = text_part.split()
        last_word = words[-1].lower() if words else ""

        # We don't want to split if it's an abbreviation
        if last_word not in abbreviations:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    return sentences


class SemanticTextSplitter:
    """
    Split text into chunks along semantic boundaries (sentences) while
    respecting a maximum token limit using tiktoken.
    """

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be > 0.")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            import tiktoken

            self.encoder = tiktoken.get_encoding(encoding_name)
        except ImportError:
            logger.warning(
                "tiktoken not found, falling back to character counts (//4 heuristic) for SemanticTextSplitter."
            )
            self.encoder = None  # type: ignore[assignment]

    def _get_length(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text, disallowed_special=()))
        return len(text) // 4  # Rough character-to-token heuristic fallback

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences. Delegates to module-level ``_split_sentences``."""
        return _split_sentences(text)

    def split(self, text: str) -> list[str]:
        """Split text into semantically cohesive chunks."""
        if not text:
            return []

        sentences = self._split_sentences(text)
        # Pre-calculate lengths to avoid redundant tokenization in the overlap loop
        sentence_data = [(s, self._get_length(s)) for s in sentences]

        chunks = []
        current_chunk_sentences: list[tuple[str, int]] = []
        current_length = 0

        for sentence, sentence_len in sentence_data:
            # If a single sentence is larger than the chunk size, we yield it as its own chunk
            # rather than falling back to character splitting, preserving semantic integrity.
            if sentence_len > self.chunk_size:
                if current_chunk_sentences:
                    chunks.append(" ".join([s for s, _ in current_chunk_sentences]))
                    current_chunk_sentences = []
                    current_length = 0
                chunks.append(sentence)
                continue

            if current_length + sentence_len <= self.chunk_size:
                current_chunk_sentences.append((sentence, sentence_len))
                current_length += sentence_len
            else:
                chunks.append(" ".join([s for s, _ in current_chunk_sentences]))

                # Handle overlap: take sentences from the end of the current chunk
                # until we hit the overlap limit
                overlap_sentences: list[tuple[str, int]] = []
                overlap_length = 0
                for s, s_len in reversed(current_chunk_sentences):
                    if overlap_length + s_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, (s, s_len))
                        overlap_length += s_len
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [(sentence, sentence_len)]
                current_length = overlap_length + sentence_len

        if current_chunk_sentences:
            chunks.append(" ".join([s for s, _ in current_chunk_sentences]))

        return chunks

    def transform_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split a list of documents into semantic chunks."""
        all_chunks = []
        for doc in documents:
            doc_id = doc["id"]
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                chunk_meta = {**metadata}
                chunk_meta.setdefault("source_id", doc.get("id", doc_id))
                chunk_meta.setdefault("subdoc_locator", "root")
                chunk_meta["chunk_index"] = i
                chunk_meta.setdefault("chunk_kind", "leaf")
                all_chunks.append(
                    {"id": f"{doc_id}_chunk_{i}", "text": chunk, "metadata": chunk_meta}
                )
        return all_chunks


class TableSplitter:
    """Specialized splitter for tabular data (CSV, TSV, Markdown).

    Converts rows into enriched natural language strings to preserve
    header-value context during embedding and retrieval.
    """

    def __init__(self, table_name: str = "Table", batch_size: int = 1):
        self.table_name = table_name
        self.batch_size = batch_size

    def transform_rows(self, rows: list[dict], headers: list[str]) -> list[str]:
        """Convert a list of row dictionaries into enriched searchable strings."""
        chunks = []
        header_str = ", ".join(headers)

        current_batch = []
        for row in rows:
            # Enriched string format: [Table Context] [Column Schema] [Row Data]
            # This format is proven to work best with SBERT/BGE embedding models.
            row_items = [f"{k}: {v}" for k, v in row.items() if v is not None and v != ""]
            row_str = f"Context: {self.table_name} | Columns: [{header_str}] | Data: " + " | ".join(
                row_items
            )
            current_batch.append(row_str)

            if len(current_batch) >= self.batch_size:
                chunks.append("\n".join(current_batch))
                current_batch = []

        if current_batch:
            chunks.append("\n".join(current_batch))

        return chunks


class RecursiveCharacterTextSplitter:
    """
    Split text into chunks based on character length and overlap,
    recursively trying different separators to maintain context.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be > 0.")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # If remaining text is smaller than chunk_size, take it all and finish
            if text_len - start <= self.chunk_size:
                chunks.append(text[start:].strip())
                break

            end = start + self.chunk_size

            # Find the best separator within the current window
            split_at = -1
            for sep in self.separators:
                if sep == "":
                    continue
                # Search backwards from end to start
                idx = text.rfind(sep, start, end)
                if idx != -1:
                    split_at = idx + len(sep)
                    break

            # If no separator found OR split_at would not make progress beyond overlap,
            # force split at the full chunk_size
            if split_at <= start + self.chunk_overlap:
                split_at = end

            chunks.append(text[start:split_at].strip())
            # Next chunk starts after accounting for overlap
            start = split_at - self.chunk_overlap

        return [c for c in chunks if c]

    def transform_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split a list of documents into chunks."""
        all_chunks = []
        for doc in documents:
            doc_id = doc["id"]
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                chunk_meta = {**metadata}
                chunk_meta.setdefault("source_id", doc.get("id", doc_id))
                chunk_meta.setdefault("subdoc_locator", "root")
                chunk_meta["chunk_index"] = i
                chunk_meta.setdefault("chunk_kind", "leaf")
                all_chunks.append(
                    {"id": f"{doc_id}_chunk_{i}", "text": chunk, "metadata": chunk_meta}
                )
        return all_chunks


class MarkdownSplitter:
    """Splits Markdown text on heading boundaries (ATX headings: # through ######).

    Each section (heading + body) becomes a chunk. Sections exceeding max_chunk_size
    tokens are recursively split with SemanticTextSplitter to prevent oversized chunks.
    The heading is prepended to sub-chunks so each chunk retains section context.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on ATX heading boundaries, keeping the heading with its section
        sections = re.split(r"(?=\n#{1,6} )", "\n" + text)
        sections = [s.strip() for s in sections if s.strip()]
        if not sections:
            return [text]

        fallback = SemanticTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks: list[str] = []
        for section in sections:
            lines = section.splitlines()
            heading = lines[0] if lines and lines[0].startswith("#") else ""
            # Check token size via fallback splitter's length method
            section_len = fallback._get_length(section)
            if section_len <= self.chunk_size:
                chunks.append(section)
            else:
                # Recursively split oversized section; prepend heading to sub-chunks
                sub_chunks = fallback.split(section)
                for i, sc in enumerate(sub_chunks):
                    chunks.append(f"{heading}\n{sc}".strip() if heading and i > 0 else sc)
        return chunks

    def transform_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            doc_id = doc["id"]
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                chunk_meta = {**metadata}
                chunk_meta.setdefault("source_id", doc.get("id", doc_id))
                chunk_meta.setdefault("subdoc_locator", "root")
                chunk_meta["chunk_index"] = i
                chunk_meta.setdefault("chunk_kind", "leaf")
                all_chunks.append(
                    {"id": f"{doc_id}_chunk_{i}", "text": chunk, "metadata": chunk_meta}
                )
        return all_chunks


class CosineSemanticSplitter:
    """Semantic chunker using cosine similarity between consecutive sentences.

    Sentences are split into a new chunk when the cosine similarity to the next
    sentence drops below ``breakpoint_threshold``. A ``max_chunk_size`` guard
    prevents runaway chunks on repetitive text.

    The ``embed_fn`` must be a callable matching the OpenEmbedding.embed signature:
    ``embed_fn(texts: list[str]) -> list[list[float]]``. All sentences are embedded
    in a single batch call to minimise latency.

    Args:
        embed_fn: Embedding callable injected at construction time.
        breakpoint_threshold: Cosine similarity below which a new chunk starts (default 0.7).
        max_chunk_size: Maximum tokens per chunk (default 500). Uses tiktoken when available.
        encoding_name: tiktoken encoding name (default "cl100k_base").
    """

    def __init__(
        self,
        embed_fn,
        breakpoint_threshold: float = 0.7,
        max_chunk_size: int = 500,
        encoding_name: str = "cl100k_base",
    ):
        self.embed_fn = embed_fn
        self.breakpoint_threshold = breakpoint_threshold
        self.max_chunk_size = max_chunk_size
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(encoding_name)
        except ImportError:
            self._encoder = None  # type: ignore[assignment]

    def _get_length(self, text: str) -> int:
        if self._encoder:
            return len(self._encoder.encode(text, disallowed_special=()))
        return len(text) // 4

    def _cosine(self, a: list[float], b: list[float]) -> float:
        from axon.rust_bridge import get_rust_bridge

        bridge = get_rust_bridge()
        if bridge.can_cosine_similarity():
            result = bridge.cosine_similarity(a, b)
            if result is not None:
                return result
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-9)

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        try:
            embeddings = self.embed_fn(sentences)
        except Exception as exc:
            logger.warning(
                f"CosineSemanticSplitter: embedding failed ({exc}), falling back to SemanticTextSplitter."
            )
            return SemanticTextSplitter(chunk_size=self.max_chunk_size).split(text)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = self._get_length(sentences[0])

        for i in range(1, len(sentences)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            s_len = self._get_length(sentences[i])
            # Start new chunk on semantic break OR max size exceeded
            if sim < self.breakpoint_threshold or current_len + s_len > self.max_chunk_size:
                chunks.append(" ".join(current))
                current = [sentences[i]]
                current_len = s_len
            else:
                current.append(sentences[i])
                current_len += s_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    def transform_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_chunks = []
        for doc in documents:
            doc_id = doc["id"]
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                chunk_meta = {**metadata}
                chunk_meta.setdefault("source_id", doc.get("id", doc_id))
                chunk_meta.setdefault("subdoc_locator", "root")
                chunk_meta["chunk_index"] = i
                chunk_meta.setdefault("chunk_kind", "leaf")
                all_chunks.append(
                    {"id": f"{doc_id}_chunk_{i}", "text": chunk, "metadata": chunk_meta}
                )
        return all_chunks


class CodeAwareSplitter:
    """Syntax-aware code splitter that chunks by symbol/block boundaries.

    Language support:
    - Python: stdlib ``ast`` (full symbol extraction + rich metadata)
    - Go, Rust, C++, Bash, Ruby, Perl, Julia, JS/TS: regex boundary detection
    - All others: ``RecursiveCharacterTextSplitter`` character fallback

    Each chunk carries code-specific metadata (``source_class``, ``language``,
    ``symbol_type``, ``symbol_name``, ``qualified_name``, ``signature``,
    ``start_line``, ``end_line``, ``imports``, ``has_docstring``,
    ``is_entrypoint``, ``is_test``) per the code graph schema from the
    SCRIPT_CHUNKING_AND_CODE_GRAPH_REPORT_2026_03_18 qualification report.
    """

    LANGUAGE_MAP: dict[str, str] = {
        ".py": "python",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rb": "ruby",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".pl": "perl",
        ".pm": "perl",
        ".jl": "julia",
        ".java": "java",
        ".kt": "kotlin",
        ".cs": "csharp",
        ".php": "php",
        ".scala": "scala",
        ".swift": "swift",
    }

    # Regex patterns matching the first line of a top-level symbol definition.
    # Only applied to lines with no leading whitespace (indentation level 0).
    _SYMBOL_PATTERNS: dict[str, re.Pattern] = {
        "go": re.compile(r"^(?:func|type)\s+\w"),
        "rust": re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|trait|impl|mod)\s+\w"),
        "javascript": re.compile(
            r"^(?:(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+\w"
            r"|(?:export\s+)?class\s+\w"
            r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\()"
        ),
        "typescript": re.compile(
            r"^(?:(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+\w"
            r"|(?:export\s+)?class\s+\w"
            r"|(?:export\s+)?(?:interface|type)\s+\w"
            r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*[=:])"
        ),
        "ruby": re.compile(r"^(?:def|class|module)\s+\w"),
        "bash": re.compile(r"^(?:function\s+\w+\s*(?:\(\s*\)\s*)?(?:\{|$)|\w+\s*\(\)\s*(?:\{|$))"),
        "perl": re.compile(r"^sub\s+\w"),
        "julia": re.compile(r"^(?:function|struct|mutable struct|module|macro)\s+\w"),
        "java": re.compile(
            r"^(?:(?:public|private|protected|static|abstract|final|synchronized)\s+)*"
            r"(?:class|interface|enum|record)\s+\w"
        ),
        "cpp": re.compile(
            r"^(?:(?:template\s*<[^>]*>\s*)?(?:class|struct|namespace|enum)\s+\w"
            r"|(?:(?:inline|static|virtual|explicit|constexpr|friend)\s+)*\w[\w:*&<>]+\s+\w+\s*\()"
        ),
        "kotlin": re.compile(
            r"^(?:(?:fun|class|object|interface|data\s+class|sealed\s+class)\s+\w)"
        ),
        "csharp": re.compile(
            r"^(?:(?:public|private|protected|internal|static|abstract|sealed|partial|override|virtual)\s+)*"
            r"(?:class|interface|struct|enum|void|\w+)\s+\w+\s*[({<]"
        ),
        "scala": re.compile(r"^(?:def|class|object|trait|case\s+class|case\s+object)\s+\w"),
        "swift": re.compile(
            r"^(?:(?:public|private|internal|open|fileprivate|final|override|static|class)\s+)*"
            r"(?:func|class|struct|enum|protocol|extension)\s+\w"
        ),
    }

    def __init__(
        self,
        max_symbol_size: int = 4000,
        fallback_chunk_size: int = 800,
        fallback_overlap: int = 100,
    ) -> None:
        self.max_symbol_size = max_symbol_size
        self.fallback_chunk_size = fallback_chunk_size
        self.fallback_overlap = fallback_overlap
        # Telemetry: counts blocks produced via character fallback (not AST).
        # Reset to 0 before each ingest run if you want per-file stats.
        self.fallback_chunks_produced: int = 0

    # ── language detection ────────────────────────────────────────────────

    def _detect_language(self, source: str) -> str:
        if not source:
            return "unknown"
        ext = os.path.splitext(source)[1].lower()
        return self.LANGUAGE_MAP.get(ext, "unknown")

    # ── import extraction ─────────────────────────────────────────────────

    def _extract_imports(self, text: str, language: str) -> list[str]:
        imports: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if language == "python" and (s.startswith("import ") or s.startswith("from ")):
                imports.append(s)
            elif language == "go" and s.startswith("import "):
                imports.append(s)
            elif language == "rust" and s.startswith("use "):
                imports.append(s)
            elif language in ("javascript", "typescript") and (
                s.startswith("import ") or "require(" in s
            ):
                imports.append(s)
            elif language == "ruby" and (
                s.startswith("require ") or s.startswith("require_relative ")
            ):
                imports.append(s)
            elif language == "perl" and (s.startswith("use ") or s.startswith("require ")):
                imports.append(s)
            elif language == "julia" and (s.startswith("using ") or s.startswith("import ")):
                imports.append(s)
            elif language in ("java", "csharp") and s.startswith("import "):
                imports.append(s)
            elif language == "cpp" and s.startswith("#include"):
                imports.append(s)
            if len(imports) >= 30:
                break
        return imports

    # ── Python AST chunking ───────────────────────────────────────────────

    def _split_python_ast(self, text: str, source: str) -> list[dict]:
        import ast as _ast

        try:
            tree = _ast.parse(text)
        except SyntaxError:
            return []

        lines = text.splitlines()
        chunks: list[dict] = []

        for node in tree.body:
            if not isinstance(
                node,
                _ast.FunctionDef | _ast.AsyncFunctionDef | _ast.ClassDef | _ast.If,
            ):
                continue

            # Only handle if __name__ == "__main__": blocks
            if isinstance(node, _ast.If):
                try:
                    test_src = _ast.unparse(node.test) if hasattr(_ast, "unparse") else ""
                except Exception:
                    test_src = ""
                if "__name__" not in test_src or "__main__" not in test_src:
                    continue
                start, end = node.lineno - 1, node.end_lineno
                chunks.append(
                    {
                        "text": "\n".join(lines[start:end]),
                        "symbol_type": "entrypoint",
                        "symbol_name": "__main__",
                        "qualified_name": "__main__",
                        "signature": 'if __name__ == "__main__":',
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "has_docstring": False,
                        "is_entrypoint": True,
                        "is_test": False,
                    }
                )
                continue

            start, end = node.lineno - 1, node.end_lineno
            name = node.name
            is_class = isinstance(node, _ast.ClassDef)
            symbol_type = "class" if is_class else "function"

            try:
                if isinstance(node, _ast.ClassDef):
                    bases = (
                        [_ast.unparse(b) for b in node.bases] if hasattr(_ast, "unparse") else []
                    )
                    sig = f"class {name}({', '.join(bases)})" if bases else f"class {name}"
                else:
                    args = _ast.unparse(node.args) if hasattr(_ast, "unparse") else "..."
                    prefix = "async def" if isinstance(node, _ast.AsyncFunctionDef) else "def"
                    sig = f"{prefix} {name}({args})"
            except Exception:
                sig = f"{'class' if is_class else 'def'} {name}(...)"

            has_doc = False
            if node.body and isinstance(node.body[0], _ast.Expr):
                _expr_val = node.body[0].value
                if isinstance(_expr_val, _ast.Constant) and isinstance(_expr_val.value, str):
                    has_doc = True
            body_text = "\n".join(lines[start:end])

            # Large class: sub-chunk its methods individually
            if is_class and len(body_text) > self.max_symbol_size:
                sub_chunks: list[dict] = []
                for child in node.body:
                    if isinstance(child, _ast.FunctionDef | _ast.AsyncFunctionDef):
                        cs, ce = child.lineno - 1, child.end_lineno
                        ct = "\n".join(lines[cs:ce])
                        try:
                            ca = _ast.unparse(child.args) if hasattr(_ast, "unparse") else "..."
                            prefix = (
                                "async def" if isinstance(child, _ast.AsyncFunctionDef) else "def"
                            )
                            csig = f"{prefix} {name}.{child.name}({ca})"
                        except Exception:
                            csig = f"def {name}.{child.name}(...)"
                        child_has_doc = False
                        if child.body and isinstance(child.body[0], _ast.Expr):
                            _cv = child.body[0].value
                            if isinstance(_cv, _ast.Constant) and isinstance(_cv.value, str):
                                child_has_doc = True
                        sub_chunks.append(
                            {
                                "text": ct,
                                "symbol_type": "method",
                                "symbol_name": child.name,
                                "qualified_name": f"{name}.{child.name}",
                                "signature": csig,
                                "start_line": child.lineno,
                                "end_line": child.end_lineno,
                                "has_docstring": child_has_doc,
                                "is_entrypoint": child.name in ("main", "__init__"),
                                "is_test": child.name.startswith("test_"),
                            }
                        )
                if sub_chunks:
                    chunks.extend(sub_chunks)
                    continue

            # Large single symbol: character fallback
            if len(body_text) > self.max_symbol_size:
                for j, part in enumerate(
                    RecursiveCharacterTextSplitter(
                        chunk_size=self.fallback_chunk_size,
                        chunk_overlap=self.fallback_overlap,
                    ).split(body_text)
                ):
                    chunks.append(
                        {
                            "text": part,
                            "symbol_type": symbol_type,
                            "symbol_name": name,
                            "qualified_name": f"{name}[{j}]",
                            "signature": sig,
                            "start_line": node.lineno,
                            "end_line": node.end_lineno,
                            "has_docstring": has_doc and j == 0,
                            "is_entrypoint": name == "main",
                            "is_test": name.startswith("test") or name.startswith("Test"),
                        }
                    )
                continue

            chunks.append(
                {
                    "text": body_text,
                    "symbol_type": symbol_type,
                    "symbol_name": name,
                    "qualified_name": name,
                    "signature": sig,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "has_docstring": has_doc,
                    "is_entrypoint": name == "main",
                    "is_test": name.startswith("test") or name.startswith("Test"),
                }
            )

        return chunks

    # ── Regex heuristic chunking ──────────────────────────────────────────

    def _parse_symbol_from_line(self, line: str, language: str) -> tuple[str, str]:
        """Return (symbol_type, symbol_name) from a definition line."""
        patterns_by_lang: dict[str, list[tuple[str, str]]] = {
            "go": [
                (r"^func\s+(?:\([^)]+\)\s+)?(\w+)", "function"),
                (r"^type\s+(\w+)", "type"),
            ],
            "rust": [
                (r"^(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)", "function"),
                (r"^(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)", "struct"),
                (r"^(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)", "trait"),
                (r"^(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)", "enum"),
                (r"^(?:pub(?:\([^)]*\))?\s+)?impl\s+(?:<[^>]+>\s+)?(\w+)", "impl"),
                (r"^(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)", "module"),
            ],
            "javascript": [
                (
                    r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)",
                    "function",
                ),
                (r"^(?:export\s+)?class\s+(\w+)", "class"),
                (r"^(?:export\s+)?(?:const|let|var)\s+(\w+)", "variable"),
            ],
            "typescript": [
                (
                    r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)",
                    "function",
                ),
                (r"^(?:export\s+)?class\s+(\w+)", "class"),
                (r"^(?:export\s+)?interface\s+(\w+)", "interface"),
                (r"^(?:export\s+)?type\s+(\w+)", "type"),
                (r"^(?:export\s+)?(?:const|let|var)\s+(\w+)", "variable"),
            ],
            "ruby": [
                (r"^def\s+(\w+)", "method"),
                (r"^class\s+(\w+)", "class"),
                (r"^module\s+(\w+)", "module"),
            ],
            "bash": [
                (r"^function\s+(\w+)", "function"),
                (r"^(\w+)\s*\(\)", "function"),
            ],
            "perl": [(r"^sub\s+(\w+)", "function")],
            "julia": [
                (r"^function\s+(\w+)", "function"),
                (r"^(?:mutable\s+)?struct\s+(\w+)", "struct"),
                (r"^module\s+(\w+)", "module"),
                (r"^macro\s+(\w+)", "macro"),
            ],
        }
        for pattern_str, sym_type in patterns_by_lang.get(language, []):
            m = re.match(pattern_str, line)
            if m:
                return sym_type, m.group(1)
        words = [w for w in line.split() if re.match(r"^\w+$", w)]
        name = words[1] if len(words) > 1 else (words[0] if words else "unknown")
        return "block", name

    def _split_regex_heuristic(self, text: str, language: str) -> list[dict]:
        pattern = self._SYMBOL_PATTERNS.get(language)
        if pattern is None:
            return []
        lines = text.splitlines()
        # Only match at indentation level 0 (no leading whitespace)
        boundaries = [
            i
            for i, line in enumerate(lines)
            if not line[:1].isspace() and pattern.match(line.rstrip())
        ]
        if not boundaries:
            return []

        chunks: list[dict] = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
            body = "\n".join(lines[start:end]).rstrip()
            first_line = lines[start].strip()
            sym_type, sym_name = self._parse_symbol_from_line(first_line, language)

            if len(body) > self.max_symbol_size:
                for j, part in enumerate(
                    RecursiveCharacterTextSplitter(
                        chunk_size=self.fallback_chunk_size,
                        chunk_overlap=self.fallback_overlap,
                    ).split(body)
                ):
                    chunks.append(
                        {
                            "text": part,
                            "symbol_type": sym_type,
                            "symbol_name": sym_name,
                            "qualified_name": f"{sym_name}[{j}]",
                            "signature": first_line[:200],
                            "start_line": start + 1,
                            "end_line": end,
                            "has_docstring": False,
                            "is_entrypoint": sym_name == "main",
                            "is_test": sym_name.startswith("test") or sym_name.startswith("Test"),
                        }
                    )
            else:
                chunks.append(
                    {
                        "text": body,
                        "symbol_type": sym_type,
                        "symbol_name": sym_name,
                        "qualified_name": sym_name,
                        "signature": first_line[:200],
                        "start_line": start + 1,
                        "end_line": end,
                        "has_docstring": body.lstrip()[:3] in ('"""', "'''", "/**"),
                        "is_entrypoint": sym_name == "main",
                        "is_test": sym_name.startswith("test") or sym_name.startswith("Test"),
                    }
                )
        return chunks

    # ── Public API ────────────────────────────────────────────────────────

    def split_code(self, text: str, source: str = "") -> list[dict]:
        """Split code text into symbol-aware chunks with rich metadata.

        Returns a list of dicts, each with ``text`` plus code metadata fields.
        """
        language = self._detect_language(source)
        imports = self._extract_imports(text, language)

        chunks: list[dict] = []
        if language == "python":
            chunks = self._split_python_ast(text, source)
        if not chunks:
            chunks = self._split_regex_heuristic(text, language)
        if not chunks:
            # Character fallback for unknown languages or files with no symbol boundaries
            raw = RecursiveCharacterTextSplitter(
                chunk_size=self.fallback_chunk_size,
                chunk_overlap=self.fallback_overlap,
            ).split(text)
            chunks = [
                {
                    "text": part,
                    "symbol_type": "block",
                    "symbol_name": f"block_{i}",
                    "qualified_name": "",
                    "signature": "",
                    "start_line": None,
                    "end_line": None,
                    "has_docstring": False,
                    "is_entrypoint": False,
                    "is_test": False,
                    "is_fallback": True,
                }
                for i, part in enumerate(raw)
            ]
            self.fallback_chunks_produced += len(chunks)

        module_path = source.replace("\\", "/") if source else ""
        for chunk in chunks:
            chunk.setdefault("source_class", "code")
            chunk.setdefault("language", language)
            chunk.setdefault("file_path", source)
            chunk.setdefault("module_path", module_path)
            chunk.setdefault("imports", imports)
            chunk.setdefault("calls", [])
            chunk.setdefault("env_vars", [])
            chunk.setdefault("commands", [])
        return chunks

    def transform_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split a list of code documents into symbol-aware chunks."""
        all_chunks: list[dict[str, Any]] = []
        for doc in documents:
            doc_id = doc["id"]
            source = doc.get("metadata", {}).get("source", "") or doc.get("id", "")
            code_chunks = self.split_code(doc["text"], source=source)
            for i, chunk_data in enumerate(code_chunks):
                metadata = doc.get("metadata", {}).copy()
                for k, v in chunk_data.items():
                    if k == "text":
                        continue
                    # Skip empty lists — vector stores (e.g. ChromaDB) reject them
                    if isinstance(v, list) and not v:
                        continue
                    metadata[k] = v
                metadata.update({"chunk": i, "total_chunks": len(code_chunks)})
                chunk_meta = {**metadata}
                chunk_meta.setdefault("source_id", doc.get("id", doc_id))
                chunk_meta.setdefault("subdoc_locator", "root")
                chunk_meta["chunk_index"] = i
                chunk_meta.setdefault("chunk_kind", "leaf")
                all_chunks.append(
                    {
                        "id": f"{doc_id}_chunk_{i}",
                        "text": chunk_data["text"],
                        "metadata": chunk_meta,
                    }
                )
        return all_chunks
