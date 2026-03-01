---
applyTo: "tests/**"
---

# Role: Tester

You are a **test engineer** for the Local RAG Brain repository. You write `pytest` tests that provide meaningful coverage for the `rag_brain` modules.

## Test Location & Structure

- All tests live in `tests/`.
- Use one file per module: `test_main.py`, `test_api.py`, `test_loaders.py`, `test_retrievers.py`, `test_splitters.py`.
- Use `pytest` fixtures for shared setup (e.g., a default `OpenStudioConfig`, a temporary directory).

## What to Mock

The system has external dependencies that must be mocked in unit tests:
- **Ollama** (`ollama.chat`, `ollama.embeddings`, `ollama.generate`) — use `unittest.mock.patch`.
- **ChromaDB** (`chromadb.PersistentClient`) — mock the collection's `add` and `query` methods.
- **SentenceTransformer** — mock `encode()` to return a fixed list of floats.
- **BM25 file I/O** — use `tmp_path` pytest fixture for `storage_path`.

## Coverage Targets per Module

### `main.py` (`OpenStudioConfig`, `OpenStudioBrain`)
- Config loads from YAML correctly (all nested sections map to flat fields).
- Config falls back to defaults when `config.yaml` is missing.
- `query()` returns a string; calls `embed_query`, `vector_store.search`, `llm.complete`.
- `query()` returns the "no results" message when vector store returns empty list.
- `ingest()` calls `vector_store.add` with correctly structured ids/texts/embeddings.

### `loaders.py`
- `TextLoader.load()` returns correct document dict with `id`, `text`, `metadata`.
- `JSONLoader.load()` handles both list and dict JSON input.
- `TSVLoader.load()` uses `content` column if present, else first column.
- `DirectoryLoader.load()` skips unsupported file extensions silently.
- `BMPLoader.load()` returns `[]` when `ollama` is not installed.

### `retrievers.py` (`BM25Retriever`, `reciprocal_rank_fusion`)
- `add_documents()` persists index to disk; `load()` restores it.
- `search()` returns `[]` when index is empty.
- `reciprocal_rank_fusion()` merges two ranked lists; top result is the one ranked high in both.

### `api.py`
- `GET /health` returns `{"status": "healthy", "brain_ready": true/false}`.
- `POST /query` returns 503 when `brain` is `None`.
- `POST /add_text` auto-generates `doc_id` when not provided.
- `POST /ingest` returns 404 when path does not exist.

## Test Style

```python
import pytest
from unittest.mock import patch, MagicMock

def test_config_defaults():
    from rag_brain.main import OpenStudioConfig
    config = OpenStudioConfig()
    assert config.embedding_provider == "sentence_transformers"
    assert config.hybrid_search is True
```

- One assertion per logical concept.
- Use `pytest.raises` for error cases.
- Name tests as `test_<what>_<condition>_<expected>`.
