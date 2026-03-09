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
| `rag-brain` | **Interactive REPL** (no args = chat mode with session persistence) |
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

### Interactive REPL Features

When running `rag-brain` with no arguments, you get:
- **Session Persistence:** Chat history auto-saves to `~/.rag_brain/sessions/session_<timestamp>.json`. Resume any past session on startup.
- **Live Tab Completion:** Slash commands, filesystem paths, and Ollama model names auto-complete as you type (via prompt_toolkit or readline).
- **Animated Init Spinner:** Braille spinner (⠋⠙⠹…) shows progress inside a box during initialization.
- **Thinking Spinner:** Same spinner displays `Brain: ⠙ thinking…` while waiting for LLM response.
- **REPL Slash Commands:**
  - `/help [cmd]` – Show all commands or detailed help for specific commands (model, embed, ingest, rag, sessions)
  - `/list` – List ingested documents with chunk counts
  - `/ingest <path|glob>` – Ingest files/directories with glob pattern support
  - `/model [provider/model]` – Switch LLM provider and model; bare `/model <name>` auto-detects provider (`gemini-*`→gemini, `gpt-*/o1-*/o3-*`→openai, else→ollama)
  - `/embed [provider/model]` – Switch embedding provider and model
  - `/pull <name>` – Pull an Ollama model with progress
  - `/search` – Toggle Brave web search (truth_grounding)
  - `/discuss` – Toggle discussion_fallback mode
  - `/rag [option]` – Show/modify RAG settings: `topk <n>`, `threshold <0-1>`, `hybrid`, `rerank`, `hyde`, `multi`
  - `/vllm-url [url]` – Show or set the vLLM server base URL at runtime
  - `/compact` – Summarize chat history via LLM to free context
  - `/context` – Display token usage bar, model info, RAG settings, chat history, retrieved sources
  - `/sessions` – List recent saved sessions
  - `/resume <id>` – Load a previous session by timestamp ID
  - `/retry` – Re-send the last query (useful after switching model or settings)
  - `/project [list|new|switch|delete|folder]` – Manage named projects with isolated knowledge bases
  - `/keys [set provider]` – Show API key status or interactively set a key (saved to `~/.rag_brain/.env`)
  - `/clear` – Clear current chat history
  - `/quit`, `/exit` – Exit the REPL

## Architecture

```
config.yaml
    └─► OpenStudioConfig (dataclass, loaded once at startup)
            └─► OpenStudioBrain (src/rag_brain/main.py)
                    ├─ OpenEmbedding  → sentence_transformers / ollama / fastembed / openai
                    ├─ OpenLLM        → ollama / gemini / ollama_cloud / openai / vllm
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
6. Build context (web results labeled `[Web Result N]`, local labeled `[Document N]`) → injected into **system prompt** (not user message) so `chat_history` stays as plain Q/A pairs for consistent multi-turn conversation
7. `OpenLLM.complete(plain_query, system_prompt_with_context, chat_history)` — user message is always the raw query
8. If no docs and `discussion_fallback=true`: LLM answers from general knowledge

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

`OpenStudioConfig.load()` flattens nested YAML structure into flat fields:

**Embedding:**
- `embedding.provider` → `embedding_provider` (sentence_transformers, ollama, fastembed, openai)
- `embedding.model` → `embedding_model`
- `embedding.base_url` → `embedding_base_url` (optional, for Ollama)

**LLM:**
- `llm.provider` → `llm_provider` (ollama, gemini, ollama_cloud, openai, vllm)
- `llm.model` → `llm_model` (default: gemma)
- `llm.base_url` → `llm_base_url`
- `llm.api_key` → `llm_api_key`
- `llm.temperature` → `temperature`
- `llm.max_tokens` → `max_tokens`
- `llm.vllm_base_url` → `vllm_base_url` (default: `http://localhost:8000`; also read from `VLLM_BASE_URL` env)

**Vector Store:**
- `vector_store.provider` → `vector_store` (chroma or qdrant)
- `vector_store.path` → `vector_store_path`

**RAG:**
- `rag.top_k` → `top_k`
- `rag.similarity_threshold` → `similarity_threshold`
- `rag.hybrid_search` → `hybrid_search`

**Chunking:**
- `chunk.size` → `chunk_size`
- `chunk.overlap` → `chunk_overlap`

**Reranking:**
- `rerank.enabled` → `rerank`
- `rerank.provider` → `rerank_provider`
- `rerank.model` → `rerank_model`

**Query Transformations:**
- `query_transformations.multi_query` → `multi_query`
- `query_transformations.hyde` → `hyde`
- `query_transformations.discussion_fallback` → `discussion_fallback` (default: true)

**Web Search:**
- `web_search.enabled` → `truth_grounding`

When adding new config fields, update both the `OpenStudioConfig` dataclass and its `load()` method.

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

> 🧪 **Golden Rule: every bug fix and every new feature must include at least one test.**
> Bug fix → add a regression test that would have caught the bug.
> New feature → add tests covering the happy path and key edge cases.
> No PR is complete without the accompanying test(s).

- `master` — protected; tagged releases only
- `feature/<name>` / `fix/<name>` / `docs/<name>` → PR to `master`
- `hotfix/<name>` → emergency fix; PR to `master`

**CI/CD:**
- `ci.yml` — runs on every PR: install + pytest
- `release.yml` — triggers on `v*.*.*` tag: changelog + GitHub Release
- `security.yml` — PR + weekly: `pip-audit` + `bandit`

