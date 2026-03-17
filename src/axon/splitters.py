import logging
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
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                all_chunks.append(
                    {"id": f"{doc['id']}_chunk_{i}", "text": chunk, "metadata": metadata}
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
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                all_chunks.append(
                    {"id": f"{doc['id']}_chunk_{i}", "text": chunk, "metadata": metadata}
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
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                all_chunks.append(
                    {"id": f"{doc['id']}_chunk_{i}", "text": chunk, "metadata": metadata}
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
            text_chunks = self.split(doc["text"])
            for i, chunk in enumerate(text_chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata.update({"chunk": i, "total_chunks": len(text_chunks)})
                all_chunks.append(
                    {"id": f"{doc['id']}_chunk_{i}", "text": chunk, "metadata": metadata}
                )
        return all_chunks
