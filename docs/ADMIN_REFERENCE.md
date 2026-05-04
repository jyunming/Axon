# Axon Admin Reference

Complete operator reference for Axon deployments.
Covers all four surfaces: CLI, REPL, REST API, and MCP server.

For installation and first-run setup, see [SETUP.md](SETUP.md).
For interactive API docs, start `axon-api` and open `http://localhost:8000/docs`.

---

## 1. Entry Points

| Command | Starts | Default Port | Best For |
|---------|--------|-------------|---------|
| `axon` | Interactive REPL | — | Day-to-day exploration, power users |
| `axon-api` | FastAPI REST server + WebGUI | `8000` | Agents, scripts, CI pipelines, browser UI |
| `axon-mcp` | MCP stdio server | — | GitHub Copilot agent mode, Claude Code |
| `axon-ui` | Streamlit web UI | `8501` | Alternative browser-based exploration |
| `axon-ext` | Install VS Code extension | — | Registers the bundled VSIX in VS Code |

When `axon-api` is running, the built-in WebGUI is at `http://localhost:8000/gui/`.

Start flags for the API server:

```bash
axon-api --host 0.0.0.0 --port 8000   # bind to all interfaces
axon-api --reload                       # dev mode: auto-reload on source change
axon-api --config /path/to/config.yaml  # explicit config file
```

---

## 2. CLI Reference

```bash
axon [options] ["query string"]
```

If no query string is given, the interactive REPL starts. If a query string is given, a single query runs and exits.

### 2.1 Global Options

| Flag | Description |
|------|-------------|
| `--version` | Print version and exit |
| `--config PATH` | Path to config YAML (default: `~/.config/axon/config.yaml`) |
| `--quiet` / `-q` | Suppress spinners and progress (auto-enabled when stdin is not a TTY) |
| `--non-interactive` | Run in headless mode — do not start the interactive REPL (useful for scripts/CI) |

### 2.2 Query & Output

| Flag | Description |
|------|-------------|
| `"query string"` | Run a single query (non-interactive) |
| `--stream` | Stream response token-by-token |
| `--cite` / `--no-cite` | Enable/disable inline `[Document N]` citations in the response |
| `--dry-run` | Skip LLM; run retrieval only and print diagnostics and ranked chunks (requires a query argument) |

### 2.3 RAG Flags (single-query overrides)

| Flag | Description |
|------|-------------|
| `--hyde` / `--no-hyde` | Enable/disable HyDE (hypothetical document embedding) |
| `--multi-query` / `--no-multi-query` | Enable/disable multi-query retrieval (3 paraphrases) |
| `--step-back` / `--no-step-back` | Enable/disable step-back query abstraction |
| `--decompose` / `--no-decompose` | Enable/disable query decomposition into sub-questions |
| `--compress` / `--no-compress` | Enable/disable LLM context compression |
| `--rerank` / `--no-rerank` | Enable/disable cross-encoder reranking |
| `--reranker-model MODEL` | Reranker model (e.g. `BAAI/bge-reranker-v2-m3`; default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `--hybrid` / `--no-hybrid` | Enable/disable BM25+vector hybrid search |
| `--graph-rag` / `--no-graph-rag` | Enable/disable GraphRAG entity-graph retrieval |
| `--graph-rag-mode MODE` | GraphRAG traversal mode: `local`, `global`, or `hybrid` |
| `--graph-rag-max-hops N` | Max relation-hop depth from matched entity (`0` = direct only) |
| `--graph-rag-hop-decay F` | Score multiplier per hop (default `0.7`) |
| `--no-graph-rag-weighted` | Use plain BFS hop count instead of Dijkstra weighted traversal |
| `--raptor` / `--no-raptor` | Enable/disable RAPTOR hierarchical summary retrieval |
| `--raptor-group-size N` | Chunks grouped per RAPTOR summary (default: `5`) |
| `--sentence-window` / `--no-sentence-window` | Enable/disable sentence-window retrieval |
| `--sentence-window-size N` | Surrounding sentences to include per matched chunk (1–10) |
| `--crag-lite` / `--no-crag-lite` | Enable/disable CRAG-Lite corrective retrieval |
| `--discuss` / `--no-discuss` | Enable/disable general-knowledge fallback when no docs match |
| `--search` / `--no-search` | Enable/disable Brave web search fallback (requires `BRAVE_API_KEY`) |
| `--cache` / `--no-cache` | Enable/disable in-memory query result caching |
| `--top-k N` | Number of chunks to retrieve (default: from config, usually `10`) |
| `--threshold F` | Similarity threshold for retrieval, `0.0`–`1.0` (default: `0.3`) |
| `--temperature F` | LLM temperature for generation, `0.0`–`2.0` (default: from config) |
| `--code-graph` / `--no-code-graph` | Enable/disable structural code-symbol graph construction during ingest |
| `--code-graph-bridge` / `--no-code-graph-bridge` | Enable/disable Phase-3 code-graph bridge (prose-to-code symbol links; requires `--code-graph`) |

### 2.4 Model & Provider

| Flag | Description |
|------|-------------|
| `--provider PROVIDER` | LLM provider: `ollama`, `openai`, `gemini`, `grok`, `vllm`, `github_copilot`, `ollama_cloud` |
| `--model NAME` | LLM model name (e.g. `gemma:2b`, `gemini-1.5-flash`, `gpt-4o`). Also accepts `provider/model` format |
| `--embed MODEL` | Embedding model (e.g. `all-MiniLM-L6-v2` or `ollama/nomic-embed-text`). Accepts `provider/model` format |
| `--list-models` | List supported providers and locally installed Ollama models, then exit |
| `--pull MODEL` | Pull an Ollama model by name, then exit (e.g. `--pull gemma:2b`) |

### 2.5 Ingestion

| Flag | Description |
|------|-------------|
| `--ingest PATH` | Ingest a file or directory into the knowledge base, then exit |
| `--no-dedup` | Disable ingest deduplication (allow re-ingesting identical content) |
| `--chunk-strategy STRATEGY` | Chunking strategy: `recursive` or `semantic` |
| `--parent-chunk-size N` | Enable small-to-big retrieval: index child chunks but return parent passages of N tokens as LLM context. `0` = disabled |
| `--refresh` | Re-ingest any tracked files whose content has changed since last ingest, then exit |
| `--list-stale` | List documents not re-ingested within `--stale-days` days, then exit |
| `--stale-days N` | Age threshold in days for `--list-stale` (default: `7`) |
| `--delete-doc SOURCE` | Delete all chunks for a source (matched by source path/name), then exit |
| `--delete-doc-id ID...` | Delete specific chunk IDs directly (space-separated), then exit |

### 2.6 Collection & Listing

| Flag | Description |
|------|-------------|
| `--list` | List all ingested sources with chunk counts, then exit |
| `--session-list` | List saved conversation sessions for the current project, then exit |

### 2.7 Project Management

| Flag | Description |
|------|-------------|
| `--project NAME` | Use an existing named project (`"default"` = global knowledge base) |
| `--project-new NAME` | Create a new project (if it does not exist) and use it. Combine with `--ingest` to populate in one step |
| `--project-list` | List all projects and exit |
| `--project-delete NAME` | Delete a project and all its data, then exit |

### 2.8 Graph Operations

| Flag | Description |
|------|-------------|
| `--graph-status` | Print knowledge graph status (entity count, code nodes, community state), then exit |
| `--graph-finalize` | Rebuild community summaries and finalize the knowledge graph, then exit |
| `--graph-export [PATH]` | Export the entity graph as an HTML file to PATH (default: active project dir/graph.html), then exit |
| `--graph-conflicts` | List facts with `status='conflicted'` from the active graph backend (dynamic_graph or federated), then exit |
| `--graph-retrieve QUERY [--graph-at TS]` | Run the active graph backend's `retrieve()` directly. `--graph-at` passes an ISO-8601 `point_in_time` (only honoured by bi-temporal backends). Then exit |

### 2.9 Vector Index Management

| Flag | Description |
|------|-------------|
| `--optimize-index` | Build or rebuild the ANN vector index (LanceDB IVF_PQ) for the active project, then exit |
| `--migrate-vectors [CHROMA_PATH]` | Migrate vectors from ChromaDB at CHROMA_PATH (or `auto` to auto-detect) to LanceDB, then exit |

### 2.10 Config Management

| Flag | Description |
|------|-------------|
| `--config-validate` | Validate `config.yaml` and print issues; exits with code `1` if any errors found |
| `--config-reset` | Reset `config.yaml` to built-in defaults and exit |
| `--setup` | Run the interactive config setup wizard and exit |
| `--doctor` | Run health checks (Python ≥ 3.10, Ollama daemon, model pulled, store writable, recommended extras) and print a colored checklist; exits non-zero on any required-check failure |

### 2.11 AxonStore (Sealed Store)

| Flag | Description |
|------|-------------|
| `--store-init PATH` | Initialise AxonStore multi-user mode at PATH (e.g. `~/axon_data`), then exit |
| `--store-status` | Show sealed-store init/unlock state, then exit |
| `--store-bootstrap PASSPHRASE` | One-time sealed-store init with PASSPHRASE. Generates the master key and persists to OS keyring. Losing this passphrase means losing every sealed project — no recovery |
| `--store-unlock PASSPHRASE` | Unlock the sealed-store for sealed-project queries. Required after every process restart before `--project-seal` works |
| `--store-lock` | Clear the in-process master key cache, then exit |
| `--store-change-passphrase OLD NEW` | Rotate the sealed-store passphrase (O(1) — project DEKs are not re-encrypted) |
| `--project-seal NAME` | Encrypt every content file in project NAME at rest, then exit. Requires store to be unlocked first |
| `--passphrase-generate [--passphrase-words N]` | **v0.4.0 Item 1** — print a Diceware passphrase from the bundled EFF wordlist (CC BY 3.0 US, 7,776 words) and exit. `N` is 4-12 (default 6 ≈ 77 bits of entropy). Use the printed phrase as input to `--store-bootstrap` or `--store-unlock` |

### 2.12 Share Lifecycle

| Flag | Description |
|------|-------------|
| `--share-list` | List shares issued by and received by this user, then exit |
| `--share-generate PROJECT GRANTEE` | Generate a read-only share key for PROJECT and GRANTEE, then exit |
| `--share-redeem SHARE_STRING` | Redeem a share string and mount the shared project, then exit |
| `--share-revoke KEY_ID` | Revoke a previously issued share by KEY_ID. For sealed shares (`KEY_ID` starts with `ssk_`), pair with `--share-project` |
| `--share-rotate` | Hard-revoke a sealed share: rotate the project DEK and re-encrypt every content file (invalidates all share wraps). Pair with `--share-revoke` and `--share-project` |
| `--share-project NAME` | Project name companion to `--share-revoke` when revoking a sealed share. Required for `ssk_` key IDs |

---

## 3. REPL Commands

Start the REPL with `axon`. All commands begin with `/`. Use `!<cmd>` for shell passthrough (controlled by `repl.shell_passthrough` in config).

### 3.1 Navigation & General

| Command | Description | Example |
|---------|-------------|---------|
| `/help [cmd]` | Show all commands, or detailed help for a specific command | `/help rag` |
| `/quit` or `/exit` | Exit the REPL | `/quit` |
| `/clear` | Clear conversation history (does not delete the saved session) | `/clear` |
| `/debug` | Toggle verbose debug logging on/off | `/debug` |
| `/theme [NAME]` | Switch syntax-highlighting theme for code blocks (e.g. `monokai`, `dracula`, `solarized-dark`) | `/theme dracula` |
| `/keys [set PROVIDER]` | Show API key status for all providers; `/keys set <provider>` saves interactively | `/keys set openai` |

### 3.2 Ingestion & Collection

| Command | Description | Example |
|---------|-------------|---------|
| `/ingest PATH` | Ingest a file, directory, or glob pattern | `/ingest ./docs/` |
| `/ingest URL` | Fetch and ingest a public URL | `/ingest https://example.com/page` |
| `/refresh` | Re-ingest all tracked files whose content has changed (SHA-256 dedup) | `/refresh` |
| `/stale [N]` | List documents not refreshed in N days (default: 7) | `/stale 14` |
| `/list` | List all ingested sources with chunk counts | `/list` |
| `/compact` | Compact storage (rebuild BM25 index and defrag vector store) | `/compact` |

### 3.3 Query & Search

| Command | Description | Example |
|---------|-------------|---------|
| `<any text>` | Send a query to the RAG pipeline and get a synthesised answer | `What is RAPTOR?` |
| `/search QUERY` | Vector search without LLM synthesis — shows raw ranked chunks | `/search entity graph` |
| `/retry` | Re-send the last query (useful after changing model or RAG flags) | `/retry` |
| `/discuss` | Toggle `discussion_fallback` — allow general-knowledge answers when no docs match | `/discuss` |
| `/search` | Toggle Brave web search fallback (`truth_grounding`) | `/search` |
| `/agent` | Toggle agent mode (LLM calls Axon tools) | `/agent` |

### 3.4 RAG Configuration

All RAG flags can be toggled at runtime without restarting.

| Command | Flag | Description |
|---------|------|-------------|
| `/rag` | — | Show all current RAG settings |
| `/rag hybrid` | `hybrid_search` | Toggle BM25+vector hybrid search |
| `/rag hyde` | `hyde` | Toggle HyDE (hypothetical document embedding) |
| `/rag multi` | `multi_query` | Toggle multi-query (3 query paraphrases) |
| `/rag step-back` | `step_back` | Toggle step-back abstraction |
| `/rag decompose` | `query_decompose` | Toggle query decomposition into sub-questions |
| `/rag compress` | `compress_context` | Toggle context compression |
| `/rag rerank` | `rerank` | Toggle cross-encoder reranking |
| `/rag rerank-model MODEL` | `reranker_model` | Set reranker model |
| `/rag sentence-window` | `sentence_window` | Toggle sentence-window retrieval |
| `/rag crag-lite` | `crag_lite` | Toggle CRAG-Lite corrective retrieval |
| `/rag cite` | `cite` | Toggle inline citations |
| `/rag raptor` | `raptor` | Toggle RAPTOR hierarchical summaries |
| `/rag graph-rag` | `graph_rag` | Toggle GraphRAG entity-graph retrieval |
| `/rag topk N` | `top_k` | Set number of retrieved chunks (e.g. `/rag topk 20`) |
| `/rag threshold F` | `similarity_threshold` | Set minimum similarity score (e.g. `/rag threshold 0.4`) |

### 3.5 Model Configuration

| Command | Description | Example |
|---------|-------------|---------|
| `/model [provider/model]` | Switch LLM provider and model. Auto-detects provider from model name (`gemini-*` → gemini, `gpt-*` → openai, else → ollama). Auto-pulls Ollama models if not present | `/model gemini/gemini-2.0-flash` |
| `/llm [temp=N]` | Show or set LLM temperature | `/llm temp=0.3` |
| `/embed [provider/model]` | Switch embedding provider and model | `/embed ollama/nomic-embed-text` |
| `/pull NAME` | Pull an Ollama model with a progress indicator | `/pull gemma3:12b` |
| `/vllm-url [URL]` | Show or set the vLLM server base URL | `/vllm-url http://gpu-box:8000/v1` |

### 3.6 Configuration

| Command | Description | Example |
|---------|-------------|---------|
| `/config` | Show current effective configuration (all fields and values) | `/config` |
| `/config KEY VALUE` | Set a single configuration field for the session | `/config top_k 20` |
| `/context` | Show token usage bar, model info, RAG settings, and last retrieved sources | `/context` |

### 3.7 Projects

| Command | Description | Example |
|---------|-------------|---------|
| `/project list` | List all projects with metadata and tree structure | `/project list` |
| `/project NAME` | Switch to a named project (creates it if it does not exist) | `/project my-docs` |
| `/project new NAME` | Create a new project explicitly | `/project new research/2026` |
| `/project delete NAME` | Delete a project and all its data | `/project delete old-proj` |
| `/project folder` | Show the filesystem path of the active project | `/project folder` |

### 3.8 Sessions

| Command | Description | Example |
|---------|-------------|---------|
| `/sessions` | List saved conversation sessions (up to 20 most recent) | `/sessions` |
| `/resume ID` | Load a previous session by its timestamp ID | `/resume 20260426-143022` |
| `/compact` | Summarise entire chat history via LLM to free context window space | `/compact` |

### 3.9 Sharing & AxonStore

| Command | Description | Example |
|---------|-------------|---------|
| `/store init PATH` | Change the store base path (e.g. to a shared drive) | `/store init /mnt/teamdrive` |
| `/store whoami` | Show AxonStore identity and current store path | `/store whoami` |
| `/store status` | Show sealed-store init/unlock state | `/store status` |
| `/share list` | List outgoing and incoming shares | `/share list` |
| `/share generate PROJECT GRANTEE` | Generate a read-only HMAC share key | `/share generate my-proj alice` |
| `/share redeem KEY` | Mount a shared project (read-only) | `/share redeem axon-share-...` |
| `/share revoke KEY_ID` | Revoke an outgoing share | `/share revoke sk_abc123` |

### 3.10 Graph

| Command | Description | Example |
|---------|-------------|---------|
| `/graph status` | Show GraphRAG entity/community build status | `/graph status` |
| `/graph build` | Trigger entity extraction and community detection | `/graph build` |
| `/graph finalize` | Trigger explicit community rebuild; reports `not_applicable` on backends without a community step (e.g. `dynamic_graph`) | `/graph finalize` |
| `/graph conflicts` | List facts with `status='conflicted'` (dynamic_graph or federated backend); `graphrag` reports unsupported | `/graph conflicts` |
| `/graph retrieve <q>` | Run the active backend's `retrieve()` directly. Flags: `--at ISO-TIMESTAMP` (point-in-time), `--top-k N` | `/graph retrieve who leads acme --at 2025-06-01` |
| `/graph viz` | Open the interactive 3D graph in VS Code webview (or default browser outside VS Code) | `/graph viz` |
| `/graph-viz [PATH]` | Export entity–relation graph as standalone HTML file; omit path to open in browser immediately | `/graph-viz /tmp/graph.html` |

### 3.11 Inline Context Attachment

Attach file or folder contents inline to any query using `@`:

```
axon> Explain this code @./src/axon/main.py
axon> What changed in @./src/axon/
axon> Compare @report.pdf with @notes.docx
```

`@file` and `@folder` can appear anywhere in the prompt; the contents are injected at that position.

---

## 4. REST API Reference

**68 endpoints** across 12 route files. Base URL: `http://localhost:8000` (default).

Interactive docs: `/docs` (Swagger UI), `/redoc`.

### 4.1 Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/live` | Liveness probe — always returns `{"status": "alive"}` with `200` while the process is running; does not check brain state |
| `GET` | `/health/ready` | Readiness probe — returns `{"status": "ok", "project": "<active-project>"}` with `200` when ready; returns `{"status": "initializing"}` with `503` during cold start |
| `GET` | `/health` | Backward-compatible alias for `/health/ready` |

> **Kubernetes:** use `livenessProbe → /health/live` and `readinessProbe → /health/ready`.

### 4.2 Query & Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Full RAG synthesis with optional citations, streaming, and per-request flag overrides |
| `POST` | `/query/stream` | SSE streaming version of `/query` — returns `text/event-stream` |
| `POST` | `/search` | Semantic/hybrid search — raw document chunks with scores; no LLM synthesis |
| `POST` | `/search/raw` | Raw retrieval with optional diagnostics trace; no synthesis |
| `POST` | `/clear` | Wipe all vectors and BM25 index for the active project |

**`POST /query` parameters:**

> Omit a field or pass `null` to inherit from the server's global config. Pass an explicit value to override for this request only.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | *(required)* Query text |
| `project` | string | `null` | Must match active project if set; `null` = no check |
| `stream` | bool | `false` | SSE streaming response |
| `top_k` | int | `10` | Chunks to retrieve |
| `threshold` | float | `0.3` | Minimum cosine similarity |
| `hyde` | bool | `false` | HyDE hypothetical document expansion |
| `rerank` | bool | `false` | Cross-encoder reranking |
| `multi_query` | bool | `false` | Multi-query paraphrase expansion |
| `step_back` | bool | `false` | Step-back query abstraction |
| `decompose` | bool | `false` | Decompose compound query into sub-questions |
| `compress` | bool | `false` | LLM context compression post-retrieval |
| `include_citations` | bool | `true` | **v0.3.2** — when true, response carries `sources` + `citations` arrays with character offsets (Claude `cite_sources` / OpenAI `file_citation` shape). Set `false` for high-throughput callers that only need the answer string |
| `discuss` | bool | `true` | Allow general-knowledge fallback |
| `graph_rag` | bool | `false` | GraphRAG entity-graph retrieval. Note: per-call `federation_weights` override is **not** on this endpoint — it lives on `POST /graph/retrieve` only |
| `raptor` | bool | `false` | RAPTOR hierarchical summary retrieval |
| `crag_lite` | bool | `false` | CRAG-Lite corrective retrieval |
| `temperature` | float | `0.7` | LLM temperature |
| `timeout` | int | `60` | LLM request timeout in seconds |
| `include_diagnostics` | bool | `false` | Add `diagnostics` object to response |
| `dry_run` | bool | `false` | Skip LLM synthesis; return retrieved chunks only |
| `chat_history` | list | `[]` | Prior turns for multi-turn context |

### 4.3 Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Async ingest of file/directory path — returns `{"job_id": "..."}` |
| `POST` | `/ingest/refresh` | Re-ingest all tracked files whose SHA-256 hash has changed |
| `GET` | `/ingest/status/{job_id}` | Poll async ingest job: `pending | running | completed | failed` |
| `POST` | `/ingest/upload` | Upload a file directly for ingest (multipart form; no path required) |
| `POST` | `/ingest_url` | Fetch and ingest content from a public URL |
| `POST` | `/add_text` | Ingest a single text string with optional metadata |
| `POST` | `/add_texts` | Batch ingest an array of `{text, metadata}` objects |
| `GET` | `/collection` | Source count and total chunk count for the active project |
| `GET` | `/collection/stale` | List documents not refreshed in N days (query param: `days=7`) |
| `GET` | `/tracked-docs` | Full manifest: all ingested sources with hashes and timestamps |
| `POST` | `/delete` | Delete specific document chunks by `doc_ids` list |

### 4.4 Projects & Config

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config` | Read current effective configuration |
| `POST` | `/config/update` | Apply runtime overrides (not persisted to disk) |
| `GET` | `/config/validate` | Validate the config file and return a list of issues |
| `POST` | `/config/reset` | Reset config to built-in defaults |
| `POST` | `/config/set` | Set a single config field by key and value |
| `GET` | `/projects` | List all projects with tree structure and metadata |
| `POST` | `/project/new` | Create a new named project |
| `POST` | `/project/switch` | Switch the active project |
| `POST` | `/project/delete/{name}` | Delete a project and all its stored data |
| `POST` | `/project/rotate-keys` | Rotate sealed-project DEK and re-encrypt all content files |
| `POST` | `/project/seal` | Encrypt all content files in a project at rest |
| `POST` | `/mount/refresh` | Force-refresh a mounted shared project's handles |
| `GET` | `/sessions` | List saved conversation sessions (most recent first) |
| `GET` | `/session/{session_id}` | Retrieve a full session transcript |

### 4.5 Sharing & AxonStore

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/store/init` | Change the store base path (e.g. to a shared drive) |
| `GET` | `/store/status` | Show sealed-store init/unlock state and cipher suite |
| `GET` | `/store/whoami` | Return current AxonStore identity and store path |
| `POST` | `/share/generate` | Generate an HMAC-SHA256 read-only share key |
| `POST` | `/share/redeem` | Mount a shared project (read-only) |
| `POST` | `/share/revoke` | Revoke an outgoing share key |
| `POST` | `/share/extend` | Extend the expiry of an existing share key |
| `GET` | `/share/list` | List outgoing and incoming shares with revocation status |

### 4.6 Security (Sealed Store)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/security/status` | Sealed-store init state and cipher suite |
| `POST` | `/security/bootstrap` | One-time sealed-store init with a passphrase |
| `POST` | `/security/unlock` | Unlock the sealed-store for sealed-project operations |
| `POST` | `/security/lock` | Lock the sealed-store (clear in-process key cache) |
| `POST` | `/security/change-passphrase` | Rotate the sealed-store passphrase |

### 4.7 Graph

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graph/status` | Entity count, edge count, community count, rebuild status |
| `POST` | `/graph/finalize` | Trigger community detection rebuild. v0.3.2: returns capability-flagged status (`ok` \| `not_applicable` \| `error`) + `community_summary_count` + `backend_id` |
| `POST` | `/graph/retrieve` | **v0.3.2** — run active backend's `retrieve()` directly. Body: `query` (req), `top_k`, `point_in_time` (ISO 8601), `federation_weights` (`{graphrag, dynamic_graph}`). Bi-temporal-only fields silently ignored on non-bi-temporal backends |
| `GET`  | `/graph/conflicts` | **v0.3.2** — list `status='conflicted'` facts. Query: `?limit=N` (default 100). Returns `{supported: bool, conflicts: [...]}` |
| `GET` | `/graph/data` | Full entity/relation graph as JSON (`nodes` + `links`) |
| `GET` | `/code-graph/data` | Structural code graph as JSON (file/class/function nodes) |
| `GET` | `/graph/visualize` | Interactive 3D graph as self-contained HTML |
| `GET` | `/graph/backend/status` | Active graph backend type and health metrics |
| `POST` | `/query/visualize` | Run a query and return HTML with LLM answer, citations, and highlighted graph |
| `POST` | `/search/visualize` | Same as `/query/visualize` but skips LLM generation |

### 4.8 Governance & Operations

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/governance/overview` | Active project count, session count, active lease count |
| `GET` | `/governance/audit` | Per-query event log. Params: `limit`, `project`, `surface` |
| `GET` | `/governance/copilot/sessions` | Active Copilot agent sessions |
| `GET` | `/governance/projects` | Per-project audit statistics |
| `POST` | `/governance/graph/rebuild` | Rebuild entity graph for a specific project (admin use) |
| `POST` | `/governance/project/maintenance` | Set project maintenance state: `normal | draining | readonly | offline` |
| `POST` | `/governance/copilot/session/{session_id}/expire` | Force-expire a Copilot session |
| `POST` | `/project/maintenance` | Set maintenance state for the active project |
| `GET` | `/project/maintenance` | Get current maintenance state of the active project |
| `GET` | `/registry/leases` | Active write-lease counts per project |

**Maintenance state lifecycle:**

```
normal → draining  (new writes rejected; reads and in-flight ops allowed)
draining → readonly (all writes blocked; reads allowed)
readonly → offline  (fully offline: all operations rejected)
offline → normal    (restore: resumes accepting traffic)
```

Always check `/registry/leases` before transitioning to `readonly`. Wait for `active_leases == 0`.

### 4.9 Copilot Bridge (Internal)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/copilot/agent` | VS Code Copilot bridge — relay LLM tasks from extension to engine |
| `GET` | `/llm/copilot/tasks` | Poll pending Copilot LLM tasks |
| `POST` | `/llm/copilot/result/{task_id}` | Report Copilot LLM task result |

These endpoints are used internally by the VS Code extension. External callers should use `/query` instead.

### 4.10 Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/metrics` | Prometheus exposition format metrics (`text/plain; version=0.0.4`); returns `503` if `prometheus-client` is not installed |

Exposed metrics: `axon_requests_total` (Counter, labels: `path/method/status`), `axon_request_duration_seconds` (Histogram, labels: `path/method`), `axon_query_total` (Counter, labels: `project/surface`), `axon_ingest_total` (Counter, labels: `project/surface`), `axon_brain_ready` (Gauge — `1` when brain is initialized).

> **Rate limiting:** `/share`, `/ingest`, and `/security` endpoints enforce per-IP rate limiting (default: 10 requests per 60-second window). Requests exceeding the limit receive HTTP `429 Too Many Requests`.

> **Request ID tracking:** every API response includes an `X-Request-ID` header. Governance write routes also emit this value into the audit trail for end-to-end log correlation.

---

## 5. MCP Tools Reference

44 tools are registered when running `axon-mcp`.

### 5.1 Ingestion (6)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `ingest_text` | `text` (str, required), `metadata` (dict, optional), `project` (str, optional) | Ingest a text string |
| `ingest_texts` | `docs` (list of `{text, metadata}`, required), `project` (str, optional) | Batch ingest multiple documents |
| `ingest_url` | `url` (str, required), `metadata` (dict, optional), `project` (str, optional) | Fetch and ingest a public URL |
| `ingest_path` | `path` (str, required) | Ingest a file or directory by absolute path |
| `refresh_ingest` | `project` (str, optional) | Re-ingest all tracked files whose SHA-256 hash has changed |
| `get_job_status` | `job_id` (str, required) | Poll async ingest job status |

Returns: `ingest_path` / `ingest_url` → `{"job_id": "..."}`. `get_job_status` → `{"status": "pending|running|completed|failed", "phase": "...", "chunks_embedded": N}`.

### 5.2 Search & Query (2)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_knowledge` | `query` (str), `top_k` (int, default `5`), `threshold` (float, optional), `hybrid` (bool, optional), `project` (str, optional) | Semantic/hybrid search — returns raw chunks with scores, no synthesis |
| `query_knowledge` | `query` (str), `top_k` (int, optional), `hyde` (bool, optional), `rerank` (bool, optional), `project` (str, optional), `chat_history` (list, default `[]`) | Full RAG query with LLM synthesis |

### 5.3 Knowledge Base Management (5)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_knowledge` | — | List all ingested sources with chunk counts |
| `delete_documents` | `doc_ids` (list of str, required) | Delete specific chunk IDs |
| `clear_knowledge` | — | Wipe entire knowledge base for active project (irreversible) |
| `get_stale_docs` | `days` (int, default `7`) | List documents not refreshed in N days |
| `get_active_leases` | — | List active write-lease counts per project |

### 5.4 Project Management (4)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_projects` | — | List all projects with metadata |
| `switch_project` | `project_name` (str, required) | Switch to a named project |
| `create_project` | `name` (str, required), `description` (str, default `""`) | Create a new project |
| `delete_project` | `name` (str, required) | Delete a project and all its data |

### 5.5 Settings (2)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_current_settings` | — | Return all current config values |
| `update_settings` | `hyde`, `rerank`, `graph_rag`, `cite`, `top_k`, `threshold`, `sentence_window_size`, `llm_provider`, `llm_model` (all optional) | Update config fields for the session |

### 5.6 Sessions (2)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_sessions` | — | List up to 20 most recent sessions |
| `get_session` | `session_id` (str, required) | Retrieve a full session transcript |

### 5.7 AxonStore & Sharing (8)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_store_status` | — | Check AxonStore init state, path, version |
| `init_store` | `base_path` (str, required), `persist` (bool, default `false`) | Initialise or move the store to a shared filesystem |
| `share_project` | `project` (str), `grantee` (str), `expires_at` (str, optional) | Generate a read-only share key |
| `redeem_share` | `share_string` (str, required) | Mount a shared project (read-only) |
| `revoke_share` | `key_id` (str, required) | Revoke an outgoing share |
| `extend_share` | `key_id` (str, required), `ttl_days` (int, optional) | Extend share expiry |
| `list_shares` | — | List outgoing and incoming shares |
| `refresh_mount` | — | Force-refresh a mounted shared project's handles |

### 5.8 Security (Sealed Store) (6)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `security_status` | — | Show sealed-store init/unlock state |
| `security_bootstrap` | `passphrase` (str, required) | One-time sealed-store init |
| `security_unlock` | `passphrase` (str, required) | Unlock for sealed-project operations |
| `security_lock` | — | Lock the sealed-store (clear in-process key cache) |
| `security_change_passphrase` | `old_passphrase` (str), `new_passphrase` (str) | Rotate the sealed-store passphrase |
| `seal_project` | `project_name` (str), `migration_mode` (str, default `"in_place"`) | Encrypt all content files in a project at rest |

### 5.9 Graph (4)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `graph_status` | — | Entity count, edge count, community count, rebuild status |
| `graph_finalize` | — | Trigger community detection rebuild |
| `graph_data` | — | Return full nodes+links JSON |
| `graph_backend_status` | — | Active graph backend type and health metrics |

---

## 6. Configuration Reference

Config file location: `~/.config/axon/config.yaml`. Override with `--config PATH` or `AXON_CONFIG_PATH` env var.

Run `axon --config-validate` to check for errors, or `axon --setup` to run the interactive wizard.

### 6.1 LLM Settings

YAML section: `llm:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm.provider` | str | `ollama` | LLM provider: `ollama`, `openai`, `gemini`, `grok`, `vllm`, `github_copilot`, `ollama_cloud` |
| `llm.model` | str | `llama3.1:8b` | Model name |
| `llm.temperature` | float | `0.7` | Generation temperature (`0.0` = deterministic, `2.0` = max creative) |
| `llm.max_tokens` | int | `2048` | Max tokens in the generated answer |
| `llm.timeout` | int | `60` | LLM request timeout in seconds |
| `llm.base_url` | str | `http://localhost:11434` | Ollama server URL (also: `OLLAMA_HOST` env var) |
| `llm.models_dir` | str | `""` | Ollama model root directory (also: `OLLAMA_MODELS` env var) |
| `llm.openai_api_key` | str | `""` | OpenAI API key (also: `OPENAI_API_KEY` env var) |
| `llm.gemini_api_key` | str | `""` | Gemini API key (also: `GEMINI_API_KEY` env var) |
| `llm.grok_api_key` | str | `""` | xAI Grok API key (also: `XAI_API_KEY` env var) |
| `llm.vllm_base_url` | str | `http://localhost:8000/v1` | vLLM server base URL (also: `VLLM_BASE_URL` env var) |
| `llm.ollama_cloud_url` | str | `https://ollama.com/api` | Remote Ollama endpoint URL (also: `OLLAMA_CLOUD_URL` env var) |
| `llm.ollama_cloud_key` | str | `""` | Remote Ollama API key (also: `OLLAMA_CLOUD_KEY` env var) |

LLM provider transport table:

| Provider | Transport | Required credentials |
|----------|-----------|---------------------|
| `ollama` | HTTP to `OLLAMA_HOST` (default `localhost:11434`) | None |
| `openai` | HTTPS to `api.openai.com` | `OPENAI_API_KEY` or `llm.openai_api_key` |
| `grok` | HTTPS to `api.x.ai/v1` | `XAI_API_KEY` / `GROK_API_KEY` or `llm.grok_api_key` |
| `gemini` | HTTPS to Google AI | `GEMINI_API_KEY` or `llm.gemini_api_key` |
| `vllm` | HTTP to your vLLM server | `llm.vllm_base_url` |
| `github_copilot` | HTTPS to Copilot API | `GITHUB_COPILOT_PAT` (OAuth token) |
| `ollama_cloud` | HTTPS to remote Ollama | `OLLAMA_CLOUD_URL` + `OLLAMA_CLOUD_KEY` |

See [MODEL_GUIDE.md](MODEL_GUIDE.md) for per-provider `config.yaml` examples.

### 6.2 Embedding Settings

YAML section: `embedding:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding.provider` | str | `sentence_transformers` | Embedding provider: `sentence_transformers`, `ollama`, `fastembed`, `openai` |
| `embedding.model` | str | `all-MiniLM-L6-v2` | Embedding model name |
| `embedding.model_path` | str | `""` | Local path override for the embedding model (takes precedence over `model` when set) |
| `embedding.dim` | int | `0` | Override embedding vector dimension (`0` = auto-detect from model) |

Embedding provider install requirements:

| Provider | Description | Install |
|----------|-------------|---------|
| `sentence_transformers` | Local CPU inference (default) | Bundled |
| `ollama` | Via local Ollama endpoint | Ollama running + model pulled |
| `fastembed` | Quantised ONNX models (BGE, BAAI) | `pip install 'axon[fastembed]'` |
| `openai` | OpenAI embedding API | `OPENAI_API_KEY` |

### 6.3 Vector Store Settings

YAML section: `vector_store:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vector_store.provider` | str | `turboquantdb` | Backend: `turboquantdb`, `lancedb`, `chroma`, `qdrant` |
| `vector_store.tqdb_bits` | int | `4` | TurboQuantDB quantization bits per coordinate (`2`, `4`, or `8`) |
| `vector_store.tqdb_fast_mode` | bool | `false` | Trade index build CPU for faster queries; slightly lowers recall |
| `vector_store.tqdb_rerank` | bool | `true` | Enable internal ANN rerank pass; improves recall at small CPU cost |
| `vector_store.tqdb_rerank_precision` | str\|null | `null` | Rerank precision: `null` = dequant, `"f16"` or `"f32"` = exact |
| `vector_store.tqdb_ef_construction` | int | `200` | HNSW build quality — higher = better recall, slower build |
| `vector_store.tqdb_max_degree` | int | `32` | HNSW graph degree — higher = better recall, larger index |
| `vector_store.tqdb_search_list_size` | int | `128` | ANN candidate list size at query time |
| `vector_store.tqdb_alpha` | float\|null | `null` | HNSW pruning aggressiveness (`null` = TQDB default 1.2) |
| `vector_store.tqdb_n_refinements` | int\|null | `null` | HNSW refinement passes during build (`null` = TQDB default 5) |
| `vector_store.qdrant_url` | str | `""` | Qdrant server URL for remote mode (empty = local file mode) |
| `vector_store.qdrant_api_key` | str | `""` | Qdrant API key for remote mode |
| `vector_store.qdrant_collection` | str | `""` | Qdrant collection name override |

### 6.4 Retrieval Settings

YAML section: `rag:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rag.top_k` | int | `10` | Chunks retrieved per query (recommended: 5–30) |
| `rag.similarity_threshold` | float | `0.3` | Min cosine similarity to include a chunk (`0.0`–`1.0`) |
| `rag.hybrid_search` | bool | `true` | Combine vector + BM25 sparse retrieval |
| `rag.hybrid_weight` | float | `0.7` | Hybrid fusion weight (`1.0` = pure vector, `0.0` = pure BM25; only used in `weighted` mode) |
| `rag.hybrid_mode` | str | `rrf` | Hybrid fusion mode: `rrf` (Reciprocal Rank Fusion, default) or `weighted` |
| `rag.sparse_retrieval` | bool | `false` | Enable SPLADE learned sparse retrieval alongside dense+BM25 (requires `pip install 'axon-rag[sparse]'`) |
| `rag.sparse_model` | str | `naver/splade-cocondenser-ensembledistil` | HuggingFace model ID for the SPLADE sparse encoder |
| `rag.sparse_weight` | float | `0.3` | Weight applied to sparse scores during hybrid fusion (`0.0`–`1.0`); values outside this range are rejected at startup |
| `rag.rerank` | bool | `false` | BGE cross-encoder reranker |
| `rag.rerank_top_k` | int | `5` | Max results to return after re-ranking |
| `rag.sentence_window` | bool | `false` | Expand retrieved chunks with surrounding sentences |
| `rag.sentence_window_size` | int | `3` | Sentences of context on each side (recommended: 1–5) |
| `rag.mmr` | bool | `false` | Maximal Marginal Relevance deduplication — reorders and removes near-duplicate chunks |
| `rag.mmr_lambda` | float | `0.5` | MMR diversity–relevance trade-off (`1.0` = pure relevance, `0.0` = pure diversity) |
| `rag.parent_doc` | bool | `false` | Small-to-big retrieval (index child chunks, return parent passages) |
| `rag.parent_chunk_size` | int | `1500` | Parent passage size in tokens when `parent_doc` is enabled |
| `rag.cite` | bool | `true` | Include inline `[Document N]` citations in answers |
| `rag.discuss` | bool | `true` | Allow general-knowledge fallback when KB has no hits (`discussion_fallback`) |
| `rag.query_cache` | bool | `false` | In-memory query result caching |
| `rag.query_cache_size` | int | `128` | Maximum cached entries |
| `rag.query_cache_ttl` | int | `1800` | Cache entry expiry in seconds (`0` = no expiry) |
| `rag.dedup_on_ingest` | bool | `true` | Skip ingest for content already in the knowledge base (SHA-256 dedup) |
| `rag.query_router` | str | `heuristic` | Multi-class query router: `heuristic` (zero latency), `llm` (one classifier call), `off` |
| `rag.truth_grounding` | bool | `false` | Enable Brave web search fallback (requires `BRAVE_API_KEY`) |
| `rag.ingest_engine` | str | `python` | Ingest pipeline engine: `python` or `rust` |
| `rag.bm25_engine` | str | `python` | BM25 index engine: `python` or `rust` |
| `rag.symbol_index_engine` | str | `python` | Symbol index engine: `python` or `rust` |
| `rag.rust_fallback_enabled` | bool | `true` | Fall back to Python engine if Rust fails |
| `rag.rust_batch_size` | int | `512` | Batch size for Rust ingest operations |

### 6.5 Query Transformation Settings

YAML section: `query_transformations:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query_transformations.hyde` | bool | `false` | Hypothetical document embedding — generates a hypothetical answer and retrieves by its embedding |
| `query_transformations.multi_query` | bool | `false` | Generate 3 query paraphrases and merge results (adds 1 LLM call) |
| `query_transformations.step_back` | bool | `false` | Abstract query to a higher-level concept before retrieval (adds 1 LLM call) |
| `query_transformations.query_decompose` | bool | `false` | Split compound query into sub-questions (adds 1 LLM call) |
| `query_transformations.discussion_fallback` | bool | `true` | Fall back to general LLM answer when retrieval confidence is low |

### 6.6 Chunking Settings

YAML section: `chunk:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunk.strategy` | str | `semantic` | Chunking strategy: `recursive`, `semantic`, `markdown`, or `cosine_semantic` |
| `chunk.size` | int | `1000` | Target chunk size in tokens (recommended: 400–2000) |
| `chunk.overlap` | int | `200` | Token overlap between adjacent chunks (recommended: 50–400) |
| `chunk.cosine_semantic_threshold` | float | `0.7` | Cosine similarity threshold for `cosine_semantic` strategy |
| `chunk.cosine_semantic_max_size` | int | `500` | Max chunk size in tokens for `cosine_semantic` strategy |
| `chunk.parent_chunk_size` | int | `1500` | Parent passage size when using small-to-big retrieval |

### 6.7 Reranker Settings

YAML section: `rerank:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rerank.enabled` | bool | `false` | Enable cross-encoder reranking |
| `rerank.provider` | str | `cross-encoder` | Reranker provider: `cross-encoder` or `llm` |
| `rerank.model` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. Use `BAAI/bge-reranker-v2-m3` for SOTA multilingual accuracy |
| `rerank.top_k` | int | `5` | Max results to return after re-ranking |

### 6.8 Graph RAG Settings

YAML section: `rag:` (graph_rag_* keys)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rag.graph_rag` | bool | `false` | Enable GraphRAG entity-centric retrieval (config.yaml default; dataclass default is `true`) |
| `rag.graph_rag_mode` | str | `local` | Traversal mode: `local` (entity/relation context), `global` (community summaries), `hybrid` (both) |
| `rag.graph_rag_depth` | str | `standard` | Extraction tier: `light` (regex, no LLM), `standard` (LLM-based NER), `deep` (standard + claims) |
| `rag.graph_rag_budget` | int | `3` | Max graph-expanded chunks injected beyond normal `top_k` |
| `rag.graph_rag_relations` | bool | `true` | Extract SUBJECT\|RELATION\|OBJECT triples during ingest |
| `rag.graph_rag_relation_budget` | int | `30` | Max chunks per batch for relation extraction (`0` = unlimited) |
| `rag.graph_rag_community` | bool | `true` | Build Louvain/Leiden community summaries |
| `rag.graph_rag_community_backend` | str | `louvain` | Community detection algorithm: `louvain` (safe default), `leidenalg`, or `auto` |
| `rag.graph_rag_entity_min_frequency` | int | `2` | Prune entities appearing in fewer than N chunks (`1` = no pruning) |
| `rag.graph_rag_max_hops` | int | `2` | Max relation-hop depth from a matched entity (`0` = direct only) |
| `rag.graph_rag_hop_decay` | float | `0.7` | Score multiplier per hop (1-hop = 0.7×, 2-hop = 0.49×) |
| `rag.graph_rag_distance_weighted` | bool | `true` | Use Dijkstra weighted traversal (false = plain BFS) |
| `rag.graph_rag_global_top_communities` | int | `0` | Cap communities entering map-reduce (`0` = no cap) |
| `rag.graph_rag_auto_route` | str | `off` | Auto-route queries based on complexity: `off`, `heuristic`, or `llm` |
| `rag.graph_rag_ner_backend` | str | `llm` | NER backend: `llm` (default) or `gliner` (no LLM for entity extraction) |
| `rag.graph_rag_relation_backend` | str | `llm` | Relation extraction backend: `llm` or `rebel` |
| `rag.graph_rag_entity_resolve` | bool | `false` | Merge near-duplicate entity names via cosine similarity |
| `rag.graph_backend` | str | `graphrag` | Graph backend: `graphrag`, `dynamic_graph` (bi-temporal SQLite), or `federated` (RRF over both). Value is `dynamic_graph` — `dynamic` fails Literal validation. (YAML key lives under `rag:`; flat on the dataclass.) |
| `rag.graph_federation_weights` | dict[str, float] | `{}` | Per-backend RRF weights when `graph_backend: federated`. Keys: `graphrag`, `dynamic_graph`. Override per-call via `federation_weights` on `POST /graph/retrieve` |
| `rag.code_graph` | bool | `false` | Build File/Symbol nodes with CONTAINS/IMPORTS edges from code chunk metadata |
| `rag.code_graph_bridge` | bool | `false` | Add MENTIONED_IN edges linking prose chunks to code symbols |

### 6.9 RAPTOR Settings

YAML section: `rag:` (raptor_* keys)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rag.raptor` | bool | `false` | Enable RAPTOR hierarchical summarisation nodes at ingest (config.yaml default; dataclass default is `true`) |
| `rag.raptor_chunk_group_size` | int | `5` | Consecutive chunks grouped per RAPTOR summary |
| `rag.raptor_max_levels` | int | `2` | Recursive summarization depth (note: previous docs incorrectly stated `1`) |
| `rag.raptor_min_source_size_mb` | float | `5.0` | Skip RAPTOR for sources smaller than this MB (`0` = no filter) |
| `rag.raptor_cache_summaries` | bool | `true` | Skip LLM when window content unchanged |
| `rag.raptor_drilldown` | bool | `true` | Replace summary hits with leaf chunks at retrieval time |
| `rag.raptor_drilldown_top_k` | int | `5` | Max leaf chunks substituted per summary hit |

### 6.10 Context Compression Settings

YAML section: `context_compression:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context_compression.enabled` | bool | `false` | Enable LLM context compression post-retrieval |
| `context_compression.strategy` | str | `sentence` | Compression algorithm: `none`, `sentence` (LLM extraction), or `llmlingua` (requires `pip install 'axon[llmlingua]'`) |
| `context_compression.token_budget` | int | `0` | Target output tokens for `llmlingua` (`0` = model default ratio) |

### 6.11 CRAG-Lite & Web Search Settings

YAML section: `rag:` / `web_search:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rag.crag_lite` | bool | `false` | Activate CRAG-Lite corrective retrieval |
| `rag.crag_lite_confidence_threshold` | float | `0.4` | Below this score, CRAG-Lite triggers fallback |
| `rag.truth_grounding` | bool | `false` | Escalate to Brave web search when CRAG-Lite fires |
| `web_search.enabled` | bool | `false` | Enable Brave Search web fallback |
| `web_search.brave_api_key` | str | `""` | Brave API key (also: `BRAVE_API_KEY` env var) |
| `web_search.num_results` | int | `10` | Web search results per query |

### 6.12 Offline / Air-gapped Settings

YAML section: `offline:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offline.enabled` | bool | `false` | Block all outbound calls (HF Hub, cloud LLMs, web search); also disables RAPTOR/GraphRAG |
| `offline.local_assets_only` | bool | `false` | Block HF Hub downloads only; RAPTOR/GraphRAG remain enabled with a local Ollama |
| `offline.local_models_dir` | str | `""` | Root directory for all pre-downloaded HF model files |
| `offline.embedding_models_dir` | str | `""` | sentence-transformers / fastembed model root (overrides `local_models_dir`) |
| `offline.hf_models_dir` | str | `""` | GLiNER, REBEL, LLMLingua, and cross-encoder reranker model root |
| `offline.tokenizer_cache_dir` | str | `""` | tiktoken BPE encoding cache directory |

### 6.13 Security Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api.key` | str | `""` | API key required on protected endpoints (empty = no auth) |
| `api.allow_origins` | list | `[]` | CORS origins allowed by the REST API server |

### 6.14 API Server Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api.key` | str | `""` | Bearer token required for write endpoints |
| `api.allow_origins` | list | `[]` | CORS allowed origins (e.g. `["http://localhost:3000"]`) |
| `api.max_upload_bytes` | int | `524288000` | Maximum file size per `/ingest/upload` multipart upload in bytes (default 500 MiB); requests exceeding this receive HTTP `413` |
| `api.max_files_per_request` | int | `1000` | Maximum files per `/ingest/upload` batch request; requests exceeding this receive HTTP `422` |

Environment variables for the API server:

| Variable | Default | Description |
|----------|---------|-------------|
| `AXON_HOST` | `127.0.0.1` | Bind address for `axon-api` |
| `AXON_PORT` | `8000` | Listen port for `axon-api` |

### 6.15 Mount & Share Settings

YAML section: top-level (or under `rag:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mount_refresh_mode` | str | `switch` | Staleness detection for shared mounts: `off` (never auto-refresh), `switch` (cache on project switch, TTL-based re-check), `per_query` (re-read marker before every retrieval, adds ~1 ms/query) |
| `mount_refresh_ttl_s` | int | `300` | Seconds before a `switch`-mode mount auto-refreshes during a query |
| `mount_sync_retry_max` | int | `5` | Max mid-sync retry attempts before raising `MountSyncPendingError` |
| `mount_sync_retry_backoff_s` | float | `0.5` | Base backoff seconds between sync retries (exponential: `backoff * 2 ** attempt`) |

### 6.16 Store Settings

YAML section: `store:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `store.base` | str | `~/.axon` | Shared filesystem path for the AxonStore (also: `AXON_STORE_BASE` env var) |

See [AXON_STORE.md](AXON_STORE.md) for the full sharing lifecycle.

### 6.17 REPL Settings

YAML section: `repl:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `repl.shell_passthrough` | str | `local_only` | Controls `!<cmd>` shell passthrough: `local_only` (allow for local/default projects only), `always` (all projects), `off` (disabled) |

### 6.18 Concurrency & Performance

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_workers` | int | `8` | Thread pool size for all background ingest and retrieval tasks |
| `graph_rag_map_workers` | int | `0` | Dedicated pool for GraphRAG map-reduce phase (`0` = share `max_workers`) |
| `ingest_batch_mode` | bool | `false` | Defer BM25, entity graph, and relation graph saves to `finalize_ingest()`. Reduces O(N²) disk writes to O(1) per session |
| `max_chunks_per_source` | int | `0` | Per-source chunk count cap after splitting (`0` = unlimited) |
| `bloom_filter_hash_store` | bool | `false` | Use a bloom filter for dedup hash membership testing (saves ~6MB RAM at 100k docs; ~0.1% false-positive rate) |

### 6.19 Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | LLM + embedding | Ollama server URL (default: `http://localhost:11434`; also accepted: `OLLAMA_BASE_URL`) |
| `OPENAI_API_KEY` | LLM + embedding | OpenAI API key |
| `XAI_API_KEY` | LLM | xAI Grok API key (also accepted: `GROK_API_KEY`) |
| `GEMINI_API_KEY` | LLM | Google Gemini API key |
| `GITHUB_COPILOT_PAT` | LLM | Copilot OAuth token (also: `GITHUB_TOKEN`) |
| `VLLM_BASE_URL` | LLM | vLLM server base URL |
| `OLLAMA_CLOUD_URL` | LLM | Remote Ollama endpoint URL |
| `OLLAMA_CLOUD_KEY` | LLM | Remote Ollama API key |
| `OLLAMA_MODELS` | LLM | Ollama model root directory |
| `BRAVE_API_KEY` | Web search | Brave Search API key |
| `AXON_STORE_BASE` | Sharing | AxonStore shared filesystem path |
| `AXON_PROJECTS_ROOT` | Projects | Override the projects root directory directly |
| `AXON_HOST` | API server | API server bind address (default: `127.0.0.1`) |
| `AXON_PORT` | API server | API server port (default: `8000`) |
| `RAG_INGEST_BASE` | Security | Restrict ingest to this directory tree (default: cwd) |
| `AXON_DEBUG` | Logging | Set to any non-empty value to enable DEBUG-level console logging |
| `PYTHONUTF8` | Windows | Set to `1` to force UTF-8 mode on Windows |

---

## 7. Governance Runbook

### 7.1 Check System Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/governance/overview
```

### 7.2 Audit Query Activity

```bash
# Last 50 queries across all projects
curl "http://localhost:8000/governance/audit?limit=50"
# Filter by project
curl "http://localhost:8000/governance/audit?project=my-project&limit=20"
# Filter by surface (api | mcp | vscode | repl | cli)
curl "http://localhost:8000/governance/audit?surface=mcp"
```

### 7.3 Put a Project Into Maintenance

```bash
# Check leases first
curl http://localhost:8000/registry/leases
# Wait for active_leases == 0, then transition
curl -X POST http://localhost:8000/governance/project/maintenance \
  -H "Content-Type: application/json" \
  -d '{"project": "my-project", "state": "readonly"}'
# Fully offline
curl -X POST http://localhost:8000/governance/project/maintenance \
  -d '{"project": "my-project", "state": "offline"}'
# Restore
curl -X POST http://localhost:8000/governance/project/maintenance \
  -d '{"project": "my-project", "state": "normal"}'
```

### 7.4 Force-Expire a Stuck Session

```bash
curl http://localhost:8000/governance/copilot/sessions
curl -X POST http://localhost:8000/governance/copilot/session/<session_id>/expire
```

### 7.5 Rebuild GraphRAG Communities

```bash
curl http://localhost:8000/graph/status
curl -X POST http://localhost:8000/graph/finalize
```

---

## 8. Surface Capability Matrix

| Capability | API | REPL | CLI | MCP |
|-----------|-----|------|-----|-----|
| Query (synthesis) | ✓ | ✓ | ✓ | ✓ |
| Query (streaming) | ✓ | — | ✓ | — |
| Search (raw) | ✓ | ✓ | — | ✓ |
| Dry-run retrieval | ✓ | — | ✓ | — |
| Ingest text | ✓ | — | — | ✓ |
| Ingest file/dir | ✓ | ✓ | ✓ | ✓ |
| Ingest URL | ✓ | ✓ | — | ✓ |
| Ingest upload | ✓ | — | — | — |
| Smart refresh | ✓ | ✓ | ✓ | ✓ |
| Stale detection | ✓ | ✓ | ✓ | ✓ |
| Collection inspect | ✓ | ✓ | ✓ | ✓ |
| Delete documents | ✓ | — | ✓ | ✓ |
| Clear knowledge base | ✓ | ✓ | — | ✓ |
| Project list | ✓ | ✓ | ✓ | ✓ |
| Project switch | ✓ | ✓ | ✓ | ✓ |
| Project create | ✓ | ✓ | ✓ | ✓ |
| Project delete | ✓ | ✓ | ✓ | ✓ |
| Config read | ✓ | ✓ | — | ✓ |
| Config update | ✓ | ✓ | — | ✓ |
| Config validate | ✓ | — | ✓ | — |
| Session list | ✓ | ✓ | ✓ | ✓ |
| Session load | ✓ | ✓ | — | ✓ |
| Share generate | ✓ | ✓ | ✓ | ✓ |
| Share redeem | ✓ | ✓ | ✓ | ✓ |
| Share revoke | ✓ | ✓ | ✓ | ✓ |
| Share list | ✓ | ✓ | ✓ | ✓ |
| Share extend | ✓ | — | — | ✓ |
| AxonStore init | ✓ | ✓ | ✓ | ✓ |
| Sealed store bootstrap | ✓ | — | ✓ | ✓ |
| Sealed store unlock/lock | ✓ | — | ✓ | ✓ |
| Project seal | ✓ | — | ✓ | ✓ |
| Mount refresh | ✓ | ✓ | — | ✓ |
| Graph status | ✓ | ✓ | ✓ | ✓ |
| Graph finalize | ✓ | ✓ | ✓ | ✓ |
| Graph data | ✓ | — | — | ✓ |
| Graph visualize | ✓ | ✓ | ✓ | — |
| Graph backend status | ✓ | — | — | ✓ |
| Optimize index | — | — | ✓ | — |
| Migrate vectors | — | — | ✓ | — |
| Maintenance state | ✓ | — | — | — |
| Lease registry | ✓ | — | — | ✓ |
| Governance audit | ✓ | — | — | — |

---

*See [API_REFERENCE.md](API_REFERENCE.md) for full request/response schemas.*
*See [MCP_TOOLS.md](MCP_TOOLS.md) for MCP tool signatures.*
*See [GOVERNANCE_CONSOLE.md](GOVERNANCE_CONSOLE.md) for the full audit and monitoring runbook.*
*See [EVALUATION.md](EVALUATION.md) for RAGAS metrics, smoke tests, and building testsets.*
*See [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) for air-gap setup, model pre-download, and local-assets-only mode.*
*See [SHARING.md](SHARING.md) for plaintext and sealed sharing setup, OneDrive/Dropbox/Google Drive walkthroughs, and the filesystem compatibility matrix.*
