import logging
import re
from typing import Any

logger = logging.getLogger("Axon.Splitters")


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
            self.encoder = None

    def _get_length(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text, disallowed_special=()))
        return len(text) // 4  # Rough character-to-token heuristic fallback

    def _split_sentences(self, text: str) -> list[str]:
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

    def split(self, text: str) -> list[str]:
        """Split text into semantically cohesive chunks."""
        if not text:
            return []

        sentences = self._split_sentences(text)
        # Pre-calculate lengths to avoid redundant tokenization in the overlap loop
        sentence_data = [(s, self._get_length(s)) for s in sentences]
        
        chunks = []
        current_chunk_sentences = []
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
                overlap_sentences = []
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
        for i, row in enumerate(rows):
            # Enriched string format: [Table Context] [Column Schema] [Row Data]
            # This format is proven to work best with SBERT/BGE embedding models.
            row_items = [f"{k}: {v}" for k, v in row.items() if v is not None and v != ""]
            row_str = (
                f"Context: {self.table_name} | Columns: [{header_str}] | Data: "
                + " | ".join(row_items)
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
