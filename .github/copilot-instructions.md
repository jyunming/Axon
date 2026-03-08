# Copilot Instructions – Local RAG Brain

## Setup & Install

```bash
pip install -e .
```

Requires Ollama running locally. Pull models before first use:
```bash
ollama pull gemma            # default LLM (or use rag-brain --pull <model>)
ollama pull nomic-embed-text # optional, higher-quality embeddings
ollama pull llava            # optional, image captioning (BMP, PNG, TIF/TIFF, PGM)
```

## Running the Project

| Command | Description |
|---|---|
| `rag-brain "question"` | Single-shot query |
| `rag-brain --stream "question"` | Stream response token-by-token |
| `rag-brain` | **Interactive REPL** (no args = chat mode) |
| `rag-brain --model gemma:2b "question"` | Override model at runtime (auto-pulls if missing) |
| `rag-brain --provider gemini --model gemini-1.5-flash "q"` | Use cloud LLM |
| `rag-brain --pull gemma:2b` | Explicitly pull an Ollama model |
| `rag-brain --list-models` | List providers and locally installed Ollama models |
| `rag-brain --ingest ./docs/` | Ingest a directory via CLI |
| `rag-brain --list` | List all ingested sources and chunk counts |
| `rag-brain-api` | Start FastAPI server (port 8000) |
| `rag-brain-ui` | Launch Streamlit web UI (port 8501) |
| `docker-compose up --build` | Launch API + UI via Docker |
| `pytest tests/ -v` | Run test suite (all tests must pass before merge) |
| `make lint` / `make format` | Ruff + Black (100-char line length) |
| `make ci` | lint + type-check + tests |

## Architecture

```
config.yaml
    └─► OpenStudioConfig (dataclass, loaded once at startup)
            └─► OpenStudioBrain (src/rag_brain/main.py)
                    ├─ OpenEmbedding  → sentence_transformers / ollama / fastembed / openai
                    ├─ OpenLLM        → ollama / gemini / ollama_cloud / openai
                    ├─ OpenVectorStore → chroma / qdrant
                    ├─ BM25Retriever  → rank_bm25 (keyword search)
                    └─ OpenReranker   → cross-encoder or LLM (optional, off by default)
```

**Retrieval flow** (`OpenStudioBrain.query()`):
1. Optional HyDE (`hyde=true`) or multi-query expansion (`multi_query=true`)
2. Embed query → vector search (`OpenVectorStore`); cosine scores stored as `vector_score`
3. If `hybrid_search=true`: BM25 search → merge via Reciprocal Rank Fusion
4. If `truth_grounding=true` and max cosine score < `similarity_threshold`: Brave Search fires as fallback; web docs tagged `is_web: True`
5. Filter by `similarity_threshold`, optionally rerank, truncate to `top_k`
6. Build context (web results labeled `[Web Result N]`, local labeled `[Document N]`) → `OpenLLM.complete()`
7. If no docs and `discussion_fallback=true`: LLM answers from general knowledge

**Agent API** (src/rag_brain/api.py):
- `POST /query` – synthesized answer
- `POST /query/stream` – streaming SSE answer
- `POST /search` – raw chunk retrieval
- `POST /add_text` – inject knowledge (ID: `agent_doc_<uuid8>`)
- `POST /delete` – delete by ID: `{"doc_ids": [...]}`
- `POST /ingest` – background directory ingestion
- `GET /collection` – list ingested sources with chunk counts

## Key Conventions

### Document format
```python
{"id": str, "text": str, "metadata": dict}
```
All loaders, `ingest()`, and vector store calls use this schema. Duplicate IDs silently overwrite in ChromaDB.

### config.yaml → dataclass mapping
`OpenStudioConfig.load()` flattens nested YAML: `embedding.provider` → `embedding_provider`, `llm.provider` → `llm_provider`, `chunk.size` → `chunk_size`, `rerank.enabled` → `rerank`. Add new fields to both the dataclass and `load()`.

### Adding a new loader
Subclass `BaseLoader` (src/rag_brain/loaders.py), implement `load(path) -> List[Dict]`, register extension in `DirectoryLoader.loaders`. Async via `aload()` is free.

### Supported file types
`.txt` `.md` `.json` `.tsv` `.csv` `.docx` `.html` `.pdf` `.bmp` `.png` `.tif` `.tiff` `.pgm`

### Adding a new vector store
Add `_init_store()`, `add()`, `search()` branches in `OpenVectorStore` (src/rag_brain/main.py). Add client as optional extra in `setup.py`.

### Streaming protocol
`query_stream()` yields both `str` chunks and `dict` markers (e.g. `{"type":"sources"}`). All consumers must skip `dict` items — only forward `str` to the user.

### Gemma system prompt workaround
Gemma models don't support `system_instruction`. `OpenLLM.complete()` detects Gemma model names and prepends system prompt to the user message instead.

### Documentation — always update in the same PR
When adding/modifying a function, endpoint, config option, or CLI flag:
- `README.md` — user-facing features and API table
- `QUICKREF.md` — CLI examples and config reference
- `.github/copilot-instructions.md` — architectural or convention changes
- Docstrings — Google-style on all public functions

### Environment variables
- `RAG_BRAIN_HOST` (default: `0.0.0.0`), `RAG_BRAIN_PORT` (default: `8000`)
- `RAG_INGEST_BASE` — restricts `/ingest` and UI to this base path (403 if outside)
- `BRAVE_API_KEY` — required for truth grounding / web search
- `GEMINI_API_KEY` / `OPENAI_API_KEY` — required for respective cloud providers

### BM25 persistence
Stored as `bm25_corpus.json`. Legacy pickle auto-migrates on first load. Rebuilt in memory on every `add_documents()` call.

### Observability
API logs `query_complete` (query, latency_ms, num_results, scores) and `ingest_complete` (path, latency_ms, docs_added, collection_size) as structured JSON at INFO level.

## Branch Strategy & Workflow

> ⚠️ **Never commit directly to `master`.** Always branch first:
> ```bash
> git checkout master && git pull
> git checkout -b feature/<name>   # or fix/, docs/, hotfix/
> ```

> 🎯 **One branch at a time.** Each branch has a single scoped purpose. Complete and merge before starting the next.

- `master` — protected; tagged releases only
- `feature/<name>` / `fix/<name>` / `docs/<name>` → PR to `master`
- `hotfix/<name>` → emergency fix; PR to `master`

**CI/CD:**
- `ci.yml` — runs on every PR: install + pytest
- `release.yml` — triggers on `v*.*.*` tag: changelog + GitHub Release
- `security.yml` — PR + weekly: `pip-audit` + `bandit`

