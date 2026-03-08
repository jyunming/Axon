# Copilot Instructions – Local RAG Brain

## Setup & Install

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Ollama running locally. Pull models before first use:
```bash
ollama pull gemma            # default LLM
ollama pull nomic-embed-text # optional, higher-quality embeddings
ollama pull llava            # optional, for image captioning (BMP, PNG, TIF/TIFF, PGM)
```

## Running the Project

| Command | Description |
|---|---|
| `rag-brain --ingest ./docs/` | Ingest a directory via CLI |
| `rag-brain --list` | List all ingested sources and chunk counts |
| `rag-brain "your question"` | Query from CLI |
| `rag-brain-api` | Start FastAPI server (port 8000) |
| `rag-brain-ui` | Launch Streamlit web UI (port 8501) |
| `docker-compose up --build` | Launch both API + UI via Docker |
| `python migrate.py import --dir ./docs/` | Bulk import from legacy system |
| `pytest tests/ -v` | Run comprehensive test suite |
| `pytest tests/test_main.py -v` | Run a single test file |
| `pytest tests/ -v -m "not slow"` | Skip slow tests |
| `make lint` | Ruff + Black check (100-char line length) |
| `make format` | Auto-format with Black + Ruff |
| `make type-check` | mypy on `src/rag_brain/` |
| `make ci` | lint + type-check + tests (no auto-format) |
| `make test-cov` | Tests with HTML coverage report (`htmlcov/`) |

## Test Suite

Pytest tests are located in `tests/`:
- **`test_main.py`:** Core RAG pipeline (ingestion, querying, document deletion, config loading)
- **`test_api.py`:** FastAPI endpoint tests
- **`test_loaders.py`:** All file loader classes
- **`test_retrievers.py`:** BM25 retriever and reciprocal rank fusion (RRF)
- **`test_splitters.py`:** Text splitting
- **`test_streaming.py`:** Streaming query responses
- **`test_tools.py`:** Agent tool schema definitions
- **`test_config.py`:** Config loading and YAML flattening

Run with `pytest tests/ -v`. All tests must pass before merging; CI runs pytest automatically on PR.

## Architecture

The system is a fully local RAG pipeline with three entry points (CLI, FastAPI, Streamlit) all sharing the same `OpenStudioBrain` core object.

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

**Retrieval flow** (inside `OpenStudioBrain.query()`):
1. Optional query transformations: HyDE (`hyde=true`) or multi-query expansion (`multi_query=true`)
2. Embed query → vector search (`OpenVectorStore`)
3. If `hybrid_search=true`: also BM25 search, then merge with Reciprocal Rank Fusion (`retrievers.reciprocal_rank_fusion`)
4. If `truth_grounding=true`: also fetch live web results via Brave Search API (web docs have `is_web: True` in metadata)
5. Filter by `similarity_threshold`
6. Optionally rerank with cross-encoder or LLM reranker
7. Truncate to `top_k`, build context string, call `OpenLLM.complete()`
8. If no documents found and `discussion_fallback=true`: LLM answers from general knowledge

**Agent API** (src/rag_brain/api.py – FastAPI):
- `POST /query` – synthesized answer (calls `brain.query()`)
- `POST /query/stream` – streaming answer via Server-Sent Events (calls `brain.query()` with streaming)
- `POST /search` – raw chunk retrieval (calls `brain.vector_store.search()` directly)
- `POST /add_text` – real-time knowledge injection (auto-generates doc ID: `agent_doc_<uuid8>`)
- `POST /delete` – delete documents by ID (body: `{"doc_ids": ["id1","id2"]}`, returns `{"status":"success","deleted":N,"doc_ids":[...]}`)
- `POST /ingest` – background directory/file ingestion via `BackgroundTasks`
- `GET /collection` – list all ingested sources with chunk counts (returns `{total_files, total_chunks, files:[{source,chunks}]}`)

## Key Conventions

### Document format
Every document throughout the pipeline is a plain dict:
```python
{"id": str, "text": str, "metadata": dict}
```
All loaders, `ingest()`, and vector store calls use this schema. The `id` must be unique per document (duplicates silently overwrite in ChromaDB).

### config.yaml → dataclass mapping
The YAML uses nested sections; `OpenStudioConfig.load()` flattens them:
- `embedding.provider` → `embedding_provider`
- `embedding.model` → `embedding_model`
- `llm.provider` → `llm_provider`
- `chunk.size` → `chunk_size`
- `rerank.enabled` → `rerank`

When adding new config fields, add to both `OpenStudioConfig` and the `load()` method's flattening logic.

### Adding a new loader
Subclass `BaseLoader` (src/rag_brain/loaders.py), implement `load(path) -> List[Dict]`, then register the file extension in `DirectoryLoader.loaders`. Async support is free via `BaseLoader.aload()` which wraps `load()` with `asyncio.to_thread`.

### Supported file types (DirectoryLoader)
`.txt`, `.md`, `.json`, `.tsv`, `.csv`, `.docx`, `.html`, `.pdf`, `.bmp`, `.png`, `.tif`, `.tiff`, `.pgm`
- **.txt, .md, .json, .tsv:** Text extraction via corresponding loaders
- **.csv:** Each row becomes a document (uses text/content/body column if present)
- **.docx:** Extracts paragraphs via `python-docx`
- **.html:** Extracts visible text (skips script/style tags) via HTML parser
- **.pdf:** Extracts text page-by-page using PyMuPDF (fitz) with pypdf fallback
- **.bmp, .png, .tif, .tiff, .pgm:** Uses Ollama VLM for visual captioning (images normalised to PNG via Pillow before sending)

### Documentation must stay in sync
**Whenever you add or modify a function, API endpoint, config option, or loader, update the relevant docs in the same PR:**
- `README.md` — user-facing features and API table
- `SETUP.md` — setup steps affected by the change
- `QUICKREF.md` — config examples and environment variables
- `CONTRIBUTING.md` — workflow or coding standard changes
- `.github/copilot-instructions.md` — architectural or convention changes
- Docstrings — use Google-style for all public functions

### Adding a new vector store
Implement `_init_store()`, `add()`, and `search()` branches inside `OpenVectorStore` (src/rag_brain/main.py). Install the client as an optional extra in `setup.py`.

### Agent tool definitions
`src/rag_brain/tools.py` exports `get_rag_tool_definition(api_base_url)`, returning OpenAI-compatible tool schemas for 6 tools: `query_knowledge_base`, `search_documents`, `add_knowledge`, `delete_documents`, `ingest_directory`, and `stream_query`. Update this when adding new API endpoints that agents should use. See `examples/agent_simple.py` for a minimal agent integration and `examples/agent_orchestration.py` for a multi-step planner pattern.

### Streaming protocol
`query_stream()` is a generator that yields both `str` chunks and `dict` markers (e.g., `{"type": "sources"}`). All consumers (API SSE handler, CLI, Streamlit) must filter out `dict` items to avoid broken output — only forward `str` chunks to the user.

### Gemma / Gemini system prompt workaround
Gemma models do not support `system_instruction` in the Gemini SDK. `OpenLLM.complete()` detects Gemma model names and prepends the system prompt to the user message instead of passing it to the constructor.

### Query transformation flags
Three optional config flags (all off by default):
- `multi_query: true` — generates multiple query reformulations and merges results
- `hyde: true` — generates a hypothetical document then embeds that instead of the raw query
- `discussion_fallback: true` — if no documents pass the similarity threshold, LLM answers from general knowledge

### Truth grounding (web search)
Set `web_search.enabled: true` in config (or `truth_grounding: true` in the dataclass) and provide `BRAVE_API_KEY` env var. Live web results are merged with local retrieval results. Web-sourced documents have `is_web: True` in their metadata.

### Environment variables for API server
- `RAG_BRAIN_HOST` (default: `0.0.0.0`)
- `RAG_BRAIN_PORT` (default: `8000`)
- `RAG_INGEST_BASE` (default: current working directory) — Restricts `/ingest` endpoint and Streamlit UI ingestion sidebar to files/directories within this path. Any path outside throws 403 Forbidden (API) or displays an error (Streamlit).

### ChromaDB collection name
The collection is always named `"rag_brain"` with cosine distance. If you change the vector store provider mid-project, existing ChromaDB data will not carry over.

### BM25 persistence
The BM25 corpus is stored as `bm25_corpus.json` in the storage directory. Legacy pickle files (`bm25_index.pkl`) are automatically migrated to JSON on first load—no manual action needed. The corpus is rebuilt in memory on every `add_documents()` call (not incremental). Deleting the JSON file resets keyword search without affecting the vector store.

### Observability — Structured Logging

The API emits structured JSON logs for both query and ingest operations:
- **`query_complete`** event: Includes `query` (string), `latency_ms` (int), `num_results` (int), `scores` (list of floats)
- **`ingest_complete`** event: Includes `path` (string), `latency_ms` (int), `docs_added` (int), `collection_size` (int)

These events are logged at INFO level in the format:
```python
{"event": "query_complete", "query": "...", "latency_ms": 234, "num_results": 5, "scores": [...]}
{"event": "ingest_complete", "path": "...", "latency_ms": 1200, "docs_added": 42, "collection_size": 987}
```

Useful for monitoring, latency tracking, and debugging retrieval quality.

### RAGAS Evaluation

RAGAS (Retrieval-Augmented Generation Assessment) is available in `scripts/evaluate.py` for benchmarking RAG quality:

```bash
python scripts/evaluate.py --testset examples/eval_testset.jsonl --output results.csv
```

Requires optional dependencies:
```bash
pip install ragas langchain-community datasets
```

A sample test set is provided in `examples/eval_testset.jsonl` (JSONL format with `question` and `ground_truth` keys). Evaluation metrics include retrieval precision, context relevance, and answer correctness.

## Development Workflow & Agent Roles

The repo uses role-specific Copilot instruction files in `.github/instructions/`. Toggle a role with `/instructions` in the CLI, or reference the file with `@.github/instructions/<role>.instructions.md` in your prompt.

### Release Cycle

```
[Planner]        → Break down feature/bug into ordered tasks
[Developer]      → Implement on feature/* or fix/* branch
CI Workflow      → Auto-runs on PR: syntax check + pytest
[Tester]         → Write/expand tests in tests/
[Security Auditor] → Review before merge to main
[Code Reviewer]  → PR approval
[Docs Writer]    → Update README, docstrings, copilot-instructions.md
[Release Manager] → Bump version, tag → triggers Release Workflow

── Incident ──
[Hotfix Responder] → hotfix/* branch → Hotfix Workflow → patch tag
```

### Agent Roles Quick Reference

| Role | Instruction File | When to Use |
|---|---|---|
| **Planner** | `planner.instructions.md` | Starting a new feature or bug fix |
| **Developer** | `developer.instructions.md` | Writing implementation code |
| **Code Reviewer** | `reviewer.instructions.md` | Reviewing a PR diff |
| **Tester** | `tester.instructions.md` | Writing pytest test cases |
| **Release Manager** | `release-manager.instructions.md` | Preparing a release |
| **Security Auditor** | `security-auditor.instructions.md` | Pre-release security gate |
| **Docs Writer** | `docs-writer.instructions.md` | Updating documentation |
| **Hotfix Responder** | `hotfix-responder.instructions.md` | Production incident response |

### GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR to `main`/`develop` | Install, syntax check, pytest |
| `release.yml` | Tag push `v*.*.*` | Auto-changelog + GitHub Release |
| `security.yml` | PR + weekly Monday | `pip-audit` CVEs + `bandit` SAST |
| `hotfix.yml` | Push to `hotfix/*` | Fail-fast CI + dual-PR reminder |

### Branch Strategy

> ⚠️ **Never commit directly to `master`.** Always create a feature or fix branch first.
> ```bash
> git checkout master && git pull
> git checkout -b feature/<name>   # or fix/, docs/, hotfix/
> ```

> 🎯 **One branch at a time.** Each branch must have a single, clearly scoped purpose.
> Complete and merge (or explicitly abandon) the current branch before creating a new one.
> Mixing unrelated changes across a branch makes PRs hard to review and increases risk.

- `master` — protected; tagged releases only. Direct commits are forbidden.
- `feature/<name>` → PR to `master`
- `fix/<name>` → PR to `master`
- `docs/<name>` → PR to `master`
- `hotfix/<name>` → emergency fix; PR to `master`
