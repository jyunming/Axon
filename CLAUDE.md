# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run services
rag-brain "your question"          # CLI query
rag-brain --ingest ./docs/         # CLI ingest
rag-brain-api                      # FastAPI on :8000
rag-brain-ui                       # Streamlit on :8501
docker-compose up --build          # both services via Docker

# Tests
pytest tests/ -v                   # all tests
pytest tests/test_main.py -v       # single file
pytest tests/ -v -m "not slow"     # skip slow tests
make test-cov                      # with coverage (htmlcov/)

# Code quality
make lint        # ruff
make format      # black (100-char line length)
make type-check  # mypy
make ci          # all of the above + tests
```

## Architecture

Three entry points (CLI, FastAPI, Streamlit) share one `OpenStudioBrain` instance:

```
config.yaml
  └─► OpenStudioConfig.load()   (flattens nested YAML → dataclass fields)
        └─► OpenStudioBrain     (src/rag_brain/main.py)
                ├─ OpenEmbedding    → sentence_transformers / ollama / fastembed / openai
                ├─ OpenLLM         → ollama / gemini / ollama_cloud / openai
                ├─ OpenVectorStore → chroma (default) / qdrant
                ├─ BM25Retriever   → rank_bm25 keyword search (src/rag_brain/retrievers.py)
                └─ OpenReranker    → cross-encoder or LLM (optional, off by default)
```

**Query flow** (`OpenStudioBrain.query()`):
1. Embed query → vector search via `OpenVectorStore`
2. If `hybrid_search=true`: BM25 search → merge with `reciprocal_rank_fusion()`
3. Filter by `similarity_threshold`, optionally rerank
4. Build context → `OpenLLM.complete(system_prompt, chat_history)`

**Ingestion flow**:
`DirectoryLoader` (dispatches by extension) → `RecursiveCharacterTextSplitter` → `OpenVectorStore.add()` + `BM25Retriever.add()`

## Key Conventions

**Document schema** — used everywhere (loaders, ingest, vector store):
```python
{"id": str, "text": str, "metadata": dict}
```
Duplicate IDs silently overwrite in ChromaDB.

**Config flattening** — `OpenStudioConfig.load()` maps nested YAML to flat fields:
- `embedding.provider` → `embedding_provider`, `embedding.model` → `embedding_model`
- `llm.provider` → `llm_provider`, `chunk.size` → `chunk_size`, `rerank.enabled` → `rerank`

When adding new config fields, update both the dataclass and `load()`.

**Adding a loader** — subclass `BaseLoader` in `src/rag_brain/loaders.py`, implement `load(path) -> List[Dict]`, register extension in `DirectoryLoader.loaders`. Async via `aload()` is free.

**Adding a vector store** — add `_init_store()`, `add()`, `search()` branches in `OpenVectorStore` (`src/rag_brain/main.py`).

**BM25 persistence** — stored as `bm25_corpus.json` (legacy `.pkl` auto-migrates on load). Corpus is rebuilt in memory on every `add_documents()` call.

**ChromaDB collection** — always named `"rag_brain"` with cosine distance.

**Gemma/Gemini** — Gemma models don't support `system_instruction`; handled in `OpenLLM.complete()`.

## FastAPI Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /query` | Synthesized answer (calls `brain.query()`) |
| `POST /query/stream` | Streaming SSE answer |
| `POST /search` | Raw chunk retrieval (no LLM) |
| `POST /add_text` | Inject knowledge (ID: `agent_doc_<uuid8>`) |
| `POST /delete` | Remove docs by ID: `{"doc_ids": [...]}` |
| `POST /ingest` | Background directory ingestion |

Path security: `RAG_INGEST_BASE` env var (default: cwd) restricts `/ingest` to prevent traversal.

## Agent Tools

`src/rag_brain/tools.py` exports `get_rag_tool_definition(api_base_url)` — 6 OpenAI-compatible tool schemas: `query_knowledge_base`, `search_documents`, `add_knowledge`, `delete_documents`, `ingest_directory`, `stream_query`. See `examples/agent_simple.py` and `examples/agent_orchestration.py`.

## Branch Strategy

- `main` — tagged releases only
- `develop` — integration branch
- `feature/<name>` / `fix/<name>` → PR to `develop`
- `hotfix/<name>` → PR to **both** `main` and `develop`

CI runs on PR: syntax check + pytest. Release triggered by `v*.*.*` tag push.

> **IMPORTANT for Claude Code**: Always create and check out a feature/fix branch before making changes.
> Never commit directly to `master` or `main`. Use `git checkout -b feature/<name>` before editing any files.
