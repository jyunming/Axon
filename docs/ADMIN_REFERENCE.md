# Axon Operator & Admin Reference

Complete function reference for operators and administrators of an Axon deployment.
This document covers every exposed entry point across all four surfaces: REST API, REPL, CLI, and MCP Server.

For installation and initial setup, see [SETUP.md](SETUP.md).

For interactive API docs, start `axon-api` and open `http://localhost:8000/docs`.

---

## 1. Entry Points

| Command | Starts | Default Port | Best For |
|---------|--------|-------------|---------|
| `axon` | Interactive REPL | — | Day-to-day exploration, power users |
| `axon-api` | FastAPI REST server | `8000` | Agents, scripts, CI pipelines |
| `axon-mcp` | MCP stdio server | — | GitHub Copilot agent mode, Claude Code |
| `axon-ui` | Streamlit web UI | `8501` | Browser-based exploration |

Start flags:

```bash
axon-api --host 0.0.0.0 --port 8000   # bind to all interfaces
axon-api --reload                       # dev mode: auto-reload on source change
axon-api --config /path/to/config.yaml  # explicit config file
```

---

## 2. REST API — Full Endpoint Reference

**54 endpoints** across 8 route files. Base URL: `http://localhost:8000` (default).

Interactive docs: `/docs` (Swagger), `/redoc`.

> For compact endpoint tables without request/response schemas, see [API_REFERENCE.md](API_REFERENCE.md).

### 2.1 Health

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Liveness probe — returns `{"status": "ok", "project": "<active-project>"}` |

### 2.2 Query & Search

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/query` | Optional | Full RAG synthesis with optional citations, streaming, and per-request flag overrides |
| `POST` | `/query/stream` | Optional | SSE streaming version of `/query` — returns `text/event-stream` |
| `POST` | `/search` | Optional | Semantic/hybrid search — raw document chunks with scores; no LLM synthesis |
| `POST` | `/search/raw` | Optional | Raw retrieval with optional diagnostics trace; no synthesis |
| `POST` | `/clear` | Required | Wipe all vectors and BM25 index for the active project |

**`POST /query` parameters:**

> Omit a field or pass `null` to inherit from the server's global config. Pass an explicit value to override for this request only.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | *(required)* Query text |
| `project` | string | `null` | Must match active project if set; `null` = no check |
| `stream` | bool | `false` | SSE streaming response (`text/event-stream`) |
| `top_k` | int | `10` | Chunks to retrieve |
| `threshold` | float | `0.3` | Minimum cosine similarity for a chunk to qualify |
| `hyde` | bool | `false` | HyDE hypothetical document expansion |
| `rerank` | bool | `false` | Cross-encoder reranking |
| `multi_query` | bool | `false` | Multi-query paraphrase expansion |
| `step_back` | bool | `false` | Step-back query abstraction |
| `decompose` | bool | `false` | Decompose compound query into sub-questions |
| `compress` | bool | `false` | LLM context compression post-retrieval |
| `cite` | bool | `true` | Inline `[source: file]` citations |
| `discuss` | bool | `true` | Allow general-knowledge fallback when KB has no hits |
| `graph_rag` | bool | `false` | GraphRAG entity-graph retrieval |
| `raptor` | bool | `false` | RAPTOR hierarchical summary retrieval |
| `crag_lite` | bool | `false` | CRAG-Lite corrective retrieval on low-confidence results |
| `temperature` | float | `0.7` | LLM temperature |
| `timeout` | int | `60` | LLM request timeout in seconds |
| `include_diagnostics` | bool | `false` | Add `diagnostics` object (confidence, fallback_triggered) to response |
| `dry_run` | bool | `false` | Skip LLM synthesis; return retrieved chunks only |
| `chat_history` | list | `[]` | Prior turns for multi-turn context |

**`POST /query` response:**

```json
{
  "query": "...",
  "response": "...",
  "settings": { "provider": "ollama", "model": "llama3.1:8b", ... },
  "provenance": {
    "answer_source": "local_kb | discussion_fallback | web_snippet",
    "retrieved_count": 5,
    "web_snippet_fallback": false
  }
}
```

When `include_diagnostics: true`, adds `"diagnostics": { "confidence": 0.82, "fallback_triggered": false }`.

### 2.3 Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Async ingest of file/directory path — returns `{"job_id": "..."}` |
| `POST` | `/ingest/refresh` | Re-ingest all tracked files whose SHA-256 hash has changed |
| `GET` | `/ingest/status/{job_id}` | Poll async ingest job: `pending \| running \| completed \| failed` |
| `POST` | `/ingest_url` | Fetch and ingest content from a public URL |
| `POST` | `/add_text` | Ingest a single text string with optional metadata |
| `POST` | `/add_texts` | Batch ingest an array of `{text, metadata}` objects |
| `GET` | `/collection` | Source count and total chunk count for the active project |
| `GET` | `/collection/stale` | List documents not refreshed in N days (query param: `days=7`) |
| `GET` | `/tracked-docs` | Full manifest: all ingested sources with hashes and last-seen timestamps |
| `POST` | `/delete` | Delete specific document chunks by `doc_ids` list |

**`POST /ingest` parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | — | *(required)* Absolute path to file or directory |
| `project` | string | `null` | Expected active project — error on mismatch; `null` = no check |
| `raptor` | bool | `false` | Enable RAPTOR hierarchical indexing for this ingest |
| `graph_rag` | bool | `false` | Enable GraphRAG entity extraction for this ingest |

### 2.4 Projects & Configuration

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config` | Read current effective configuration |
| `POST` | `/config/update` | Apply runtime overrides (scoped to this request; not persisted) |
| `GET` | `/projects` | List all projects with tree structure and metadata |
| `POST` | `/project/new` | Create a new named project |
| `POST` | `/project/switch` | Switch the active project |
| `POST` | `/project/delete/{name}` | Delete a project and all its stored data |
| `GET` | `/sessions` | List saved conversation sessions (most recent first) |
| `GET` | `/session/{session_id}` | Retrieve a full session transcript |

### 2.5 Sharing & AxonStore

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/store/init` | Change the store base path (e.g. to a shared drive) |
| `GET` | `/store/whoami` | Return current AxonStore identity and store path |
| `POST` | `/share/generate` | Generate an HMAC-SHA256 read-only share key |
| `POST` | `/share/redeem` | Mount a shared project (read-only) |
| `POST` | `/share/revoke` | Revoke an outgoing share key |
| `GET` | `/share/list` | List outgoing and incoming shares with revocation status |

**`POST /share/generate` parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | string | required | Name of the project to share |
| `grantee` | string | required | OS username of the recipient |
| `expires_at` | string | `null` | ISO 8601 expiry timestamp — `null` means no expiry |

### 2.6 Graph

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graph/status` | Entity count, edge count, community count, rebuild status |
| `POST` | `/graph/finalize` | Trigger explicit community detection rebuild |
| `GET` | `/graph/data` | Full entity/relation graph as JSON (`nodes` + `links`) |
| `GET` | `/code-graph/data` | Structural code graph as JSON (file/class/function nodes) |
| `GET` | `/graph/visualize` | Interactive 3D graph as HTML — opens in browser. Click a node to see its description and source evidence. (VS Code extension uses the embedded webview which also supports opening files in the editor.) |
| `POST` | `/query/visualize` | Run a query and return a self-contained HTML page with LLM answer, citations, and highlighted graph — nodes matched by the query are golden; first-degree neighbours are orange. |
| `POST` | `/search/visualize` | Same as `/query/visualize` but skips LLM generation — shows raw retrieved chunks instead of an answer. Useful for inspecting retrieval quality without spending LLM tokens. |

### 2.7 Governance & Operations

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/governance/overview` | Active project count, session count, active lease count |
| `GET` | `/governance/audit` | Per-query event log — project, surface, query, latency. Params: `limit`, `project`, `surface` |
| `GET` | `/governance/copilot/sessions` | Active Copilot agent sessions (opened_at, last_query, surface) |
| `GET` | `/governance/projects` | Per-project audit statistics |
| `POST` | `/governance/graph/rebuild` | Rebuild entity graph for a specific project (admin use) |
| `POST` | `/governance/project/maintenance` | Set project maintenance state: `normal \| draining \| readonly \| offline` |
| `POST` | `/governance/copilot/session/{session_id}/expire` | Force-expire a Copilot session |
| `GET` | `/registry/leases` | Active write-lease counts per project — check before maintenance |

**Maintenance state lifecycle:**

```
normal → draining  (new writes rejected; reads and in-flight ops allowed)
draining → readonly (all writes blocked; reads allowed)
readonly → offline  (fully offline: all operations rejected)
offline → normal    (restore: resumes accepting traffic)
```

Always check `/registry/leases` before transitioning to `readonly`. Wait for `active_leases == 0`.

### 2.8 Internal / Copilot Bridge

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/copilot/agent` | VS Code Copilot bridge — relay LLM tasks from extension to engine |
| `GET` | `/llm/copilot/tasks` | Poll pending Copilot LLM tasks |
| `POST` | `/llm/copilot/result/{task_id}` | Report Copilot LLM task result |

These endpoints are used internally by the VS Code extension. External callers should use `/query` instead.

---

## 3. REPL Commands — Full Reference

Start the REPL with `axon`. All commands begin with `/`. Use `!<cmd>` for shell passthrough.

### 3.1 Ingestion & Collection

| Command | Description |
|---------|-------------|
| `/ingest <path\|glob>` | Ingest a file, directory, or glob pattern |
| `/ingest <url>` | Fetch and ingest a public URL |
| `/refresh` | Re-ingest all tracked files whose content has changed (SHA-256 dedup) |
| `/stale [N]` | List documents not refreshed in N days (default: 7) |
| `/list` | List all ingested sources with chunk counts |
| `/clear` | Wipe the active project's knowledge base (irreversible) |
| `/compact` | Compact storage (rebuild BM25 index and defrag vector store) |

### 3.2 Query & Search

| Command | Description |
|---------|-------------|
| `<any text>` | Send a query to the RAG pipeline and get a synthesised answer |
| `/search <query>` | Vector search without LLM synthesis — shows raw ranked chunks |
| `/retry` | Re-send the last query (useful after changing model or RAG flags) |
| `/discuss` | Toggle `discussion_fallback` — allow general-knowledge answers when no documents match |
| `/search` | Toggle Brave web search fallback (`truth_grounding`) |

### 3.3 RAG Configuration

All RAG flags can be toggled at runtime without restarting. Changes persist for the session.

| Command | Flag | Description |
|---------|------|-------------|
| `/rag hyde` | `hyde` | Toggle HyDE (hypothetical document embedding) |
| `/rag multi` | `multi_query` | Toggle multi-query (3 query paraphrases) |
| `/rag step-back` | `step_back` | Toggle step-back abstraction |
| `/rag decompose` | `decompose` | Toggle query decomposition into sub-questions |
| `/rag compress` | `compress` | Toggle context compression |
| `/rag rerank` | `rerank` | Toggle cross-encoder reranking |
| `/rag rerank-model <model>` | `rerank_model` | Set reranker model |
| `/rag sentence-window` | `sentence_window` | Toggle sentence-window retrieval |
| `/rag crag-lite` | `crag_lite` | Toggle CRAG-Lite corrective retrieval |
| `/rag cite` | `cite` | Toggle inline citations |
| `/rag raptor` | `raptor` | Toggle RAPTOR hierarchical summaries |
| `/rag graph-rag` | `graph_rag` | Toggle GraphRAG entity-graph retrieval |
| `/rag topk <n>` | `top_k` | Set number of retrieved chunks (e.g. `/rag topk 20`) |
| `/rag threshold <0-1>` | `similarity_threshold` | Set minimum similarity score |
| `/rag hybrid` | `hybrid_search` | Toggle BM25+vector hybrid search |
| `/rag` | — | Show all current RAG settings |

### 3.4 Model Configuration

| Command | Description |
|---------|-------------|
| `/model [provider/model]` | Switch LLM provider and model. Auto-detects provider from model name (`gemini-*` → gemini, `gpt-*` → openai, else → ollama). Auto-pulls Ollama models if not present. |
| `/llm [temp=N]` | Show or set LLM temperature |
| `/embed [provider/model]` | Switch embedding provider and model |
| `/pull <name>` | Pull an Ollama model with a progress indicator |
| `/vllm-url [url]` | Show or set the vLLM server base URL |
| `/keys [set <provider>]` | Show API key status for all providers; `/keys set <provider>` saves interactively |

### 3.5 Projects

| Command | Description |
|---------|-------------|
| `/project list` | List all projects |
| `/project <name>` | Switch to a named project (creates it if it does not exist) |
| `/project new <name>` | Create a new project explicitly |
| `/project delete <name>` | Delete a project and all its data |
| `/project folder` | Show the filesystem path of the active project |

### 3.6 Sessions

| Command | Description |
|---------|-------------|
| `/sessions` | List saved conversation sessions (up to 20 most recent) |
| `/resume <id>` | Load a previous session by its timestamp ID |
| `/compact` | Summarise entire chat history via LLM to free context window space |
| `/context` | Show token usage bar, model info, RAG settings, chat history, last retrieved sources |

### 3.7 Sharing & AxonStore

| Command | Description |
|---------|-------------|
| `/store init <path>` | Change the store base path (e.g. to a shared drive) |
| `/store whoami` | Show AxonStore identity and current store path |
| `/share list` | List outgoing and incoming shares |
| `/share generate <project> <grantee>` | Generate a read-only HMAC share key |
| `/share redeem <key>` | Mount a shared project (read-only) |
| `/share revoke <key_id>` | Revoke an outgoing share |

### 3.8 Graph

| Command | Description |
|---------|-------------|
| `/graph status` | Show GraphRAG entity/community build status |
| `/graph finalize` | Trigger explicit community rebuild |
| `/graph viz` | Open the interactive 3D graph in VS Code webview (or default browser outside VS Code) |
| `/graph-viz [path]` | Export entity–relation graph as standalone HTML file; omit path to open in browser immediately |

### 3.9 Utility

| Command | Description |
|---------|-------------|
| `/help [cmd]` | Show all commands or detailed help for a specific command |
| `/clear` | Clear chat history (does not delete the saved session) |
| `/quit` / `/exit` | Exit the REPL |
| `! <command>` | Execute a shell command directly (controlled by `repl.shell_passthrough` in config) |

**`@file` / `@folder` inline context:** Attach file or folder contents inline to any query:

```
axon> Explain this code @./src/axon/main.py
axon> What changed in @./src/axon/
axon> Compare @report.pdf with @notes.docx
```

---

## 4. CLI Flags — Full Reference

```bash
axon [options] ["query string"]
```

If no query string is given, the interactive REPL starts. If a query string is given, a single query runs and the process exits.

### 4.1 Query & Output

| Flag | Description |
|------|-------------|
| `"query"` | Run a single query (non-interactive) |
| `--stream` | Stream response token-by-token |
| `--cite` / `--no-cite` | Enable/disable inline citations |
| `--no-answer` | Skip LLM synthesis; show only retrieved chunks |

### 4.2 RAG Flags (single-query overrides)

| Flag | Description |
|------|-------------|
| `--hyde` | Enable HyDE for this query |
| `--multi-query` | Enable multi-query expansion |
| `--step-back` | Enable step-back abstraction |
| `--decompose` | Enable query decomposition |
| `--compress` | Enable context compression |
| `--rerank` | Enable cross-encoder reranking |
| `--graph-rag` | Enable GraphRAG entity-graph retrieval |
| `--raptor` | Enable RAPTOR hierarchical summaries |

### 4.3 Ingestion

| Flag | Description |
|------|-------------|
| `--ingest <path>` | Ingest a file or directory and exit |
| `--raptor --ingest` | Ingest with RAPTOR hierarchical indexing |
| `--graph-rag --ingest` | Ingest with GraphRAG entity extraction |
| `--code-graph --ingest <path>` | Build structural code graph from source directory |

### 4.4 Model & Provider

| Flag | Description |
|------|-------------|
| `--model <name>` | Set LLM model for this run |
| `--provider <name>` | Set LLM provider: `ollama \| openai \| gemini \| grok \| vllm \| copilot \| github_copilot \| ollama_cloud` |
| `--pull <name>` | Pull an Ollama model and exit |
| `--list-models` | List available providers and locally installed Ollama models |

### 4.5 Project Management

| Flag | Description |
|------|-------------|
| `--project <name>` | Use a named project |
| `--project-new <name>` | Create a new project |
| `--project-list` | List all projects |
| `--project-delete <name>` | Delete a project |

### 4.6 Collection Management

| Flag | Description |
|------|-------------|
| `--list` | List all ingested documents |

---

## 5. MCP Tools — Full Reference

30 tools are registered when running `axon-mcp`.

### Ingestion (6)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `ingest_text` | `text` | string | required | Text to ingest |
| | `metadata` | dict | `{}` | Key/value metadata attached to all chunks |
| | `project` | string | `null` | Expected active project (error on mismatch) |
| `ingest_texts` | `docs` | `[{text, metadata}]` | required | List of `BatchDocItem` objects |
| | `project` | string | `null` | Expected active project |
| `ingest_url` | `url` | string | required | Public URL to fetch and ingest |
| | `metadata` | dict | `{}` | Metadata for ingested chunks |
| | `project` | string | `null` | Expected active project |
| `ingest_path` | `path` | string | required | Absolute path to file or directory |
| `refresh_ingest` | `project` | string | `null` | Re-ingest all tracked files whose SHA-256 hash has changed |
| `get_job_status` | `job_id` | string | required | Job ID from `ingest_path` or `ingest_url` |

**Returns:** `ingest_path` / `ingest_url` → `{"job_id": "..."}`. `get_job_status` → `{"status": "pending|running|completed|failed", "phase": "...", "chunks_embedded": N}`.

### Search & Query (2)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `search_knowledge` | `query` | string | required | Search query |
| | `top_k` | int | `5` | Number of chunks to return |
| | `threshold` | float | `null` | Minimum similarity score (null = config default 0.3) |
| | `hybrid` | bool | `null` | BM25+vector hybrid mode (null = config default) |
| | `project` | string | `null` | Expected active project |
| `query_knowledge` | `query` | string | required | RAG query |
| | `top_k` | int | `null` | Chunks to retrieve (null = config default 10) |
| | `hyde` | bool | `null` | HyDE expansion (null = config default) |
| | `rerank` | bool | `null` | Cross-encoder reranking (null = config default) |
| | `project` | string | `null` | Expected active project |
| | `chat_history` | list | `[]` | Prior turns for multi-turn context |

**Returns:** `search_knowledge` → `[{text, score, metadata}]`. `query_knowledge` → `{response, provenance, settings}`.

### Knowledge Base Management (5)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `list_knowledge` | — | — | — | No parameters |
| `delete_documents` | `doc_ids` | `[string]` | required | Internal chunk ID list |
| `clear_knowledge` | — | — | — | No parameters — irreversible |
| `get_stale_docs` | `days` | int | `7` | Documents not refreshed in this many days |
| `get_active_leases` | — | — | — | No parameters |

### Project Management (4)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `list_projects` | — | — | — | No parameters |
| `switch_project` | `project_name` | string | required | Name of project to activate |
| `create_project` | `name` | string | required | New project name |
| | `description` | string | `""` | Optional human-readable description |
| `delete_project` | `name` | string | required | Project to delete (all data removed) |

### Settings (2)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `get_current_settings` | — | — | — | No parameters |
| `update_settings` | `hyde` | bool | `null` | HyDE toggle (null = no change) |
| | `rerank` | bool | `null` | Rerank toggle |
| | `graph_rag` | bool | `null` | GraphRAG toggle |
| | `cite` | bool | `null` | Citations toggle |
| | `top_k` | int | `null` | Retrieved chunk count |
| | `threshold` | float | `null` | Minimum similarity score |
| | `sentence_window_size` | int | `2` | Window size for sentence-window retrieval |
| | `llm_provider` | string | `null` | LLM provider name |
| | `llm_model` | string | `null` | LLM model name |

### Sessions (2)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `list_sessions` | — | — | — | No parameters — returns up to 20 most recent |
| `get_session` | `session_id` | string | required | Timestamp-based session ID |

### AxonStore & Sharing (6)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `get_store_status` | — | — | — | Check whether AxonStore is initialised; returns path, version, `created_at`. Safe to call before brain is ready. |
| `init_store` | `base_path` | string | required | Absolute path to shared filesystem location |
| | `persist` | bool | `false` | Write store path to config.yaml |
| `share_project` | `project` | string | required | Project to share |
| | `grantee` | string | required | Recipient identifier |
| | `expires_at` | string | `null` | ISO 8601 expiry timestamp (null = no expiry) |
| `redeem_share` | `share_string` | string | required | Full share string from `share_project` |
| `revoke_share` | `key_id` | string | required | Key ID from `list_shares` |
| `list_shares` | — | — | — | No parameters |

### Graph (3)

| Tool | Parameter | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `graph_status` | — | — | — | No parameters |
| `graph_finalize` | — | — | — | No parameters — triggers community rebuild |
| `graph_data` | — | — | — | No parameters — returns full nodes+links JSON |

---

## 6. Configuration Reference

### 6.1 LLM Providers

Set `llm.provider` in `config.yaml`:

| Provider value | Transport | Required credentials |
|---------------|-----------|---------------------|
| `ollama` | HTTP to `OLLAMA_HOST` (default `localhost:11434`) | None |
| `openai` | HTTPS to `api.openai.com` | `OPENAI_API_KEY` or `llm.openai_api_key` |
| `grok` | HTTPS to `api.x.ai/v1` | `XAI_API_KEY` / `GROK_API_KEY` or `llm.grok_api_key` |
| `gemini` | HTTPS to Google AI | `GEMINI_API_KEY` or `llm.gemini_api_key` |
| `vllm` | HTTP to your vLLM server | `llm.vllm_base_url` |
| `copilot` | VS Code extension bridge | Active VS Code Copilot subscription |
| `github_copilot` | HTTPS to Copilot API | `GITHUB_COPILOT_PAT` (OAuth token) |
| `ollama_cloud` | HTTPS to remote Ollama | `OLLAMA_CLOUD_URL` + `OLLAMA_CLOUD_KEY` |

See [MODEL_GUIDE.md](MODEL_GUIDE.md) for per-provider `config.yaml` examples.

### 6.2 Embedding Providers

Set `embedding.provider` in `config.yaml`:

| Provider value | Description | Install |
|---------------|-------------|---------|
| `sentence_transformers` | Local CPU inference (default) | `pip install sentence-transformers` |
| `ollama` | Via local Ollama endpoint | Ollama running + model pulled |
| `fastembed` | Quantised ONNX models (BGE, BAAI) | `pip install 'axon[fastembed]'` |
| `openai` | OpenAI embedding API | `OPENAI_API_KEY` |

### 6.3 Vector Store Backends

Set `vector_store.provider` in `config.yaml`:

| Provider value | Description | Install |
|---------------|-------------|---------|
| `turboquantdb` | Quantized embedded store — fastest ingest, smallest disk *(default)* | Bundled (`tqdb` on PyPI) |
| `lancedb` | Embedded columnar store | Bundled |
| `chroma` | Local persistent store | Bundled |
| `qdrant` | Local or remote Qdrant | `pip install 'axon[qdrant]'` |

**TurboQuantDB config fields** (only used when `provider: turboquantdb`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tqdb_bits` | int | `4` | Quantization bits per coordinate (`2`, `4`, or `8`) |
| `tqdb_fast_mode` | bool | `false` | Trade index build CPU for faster queries; slightly lowers recall |
| `tqdb_rerank` | bool | `true` | Enable internal ANN rerank pass; improves recall at small CPU cost |
| `tqdb_rerank_precision` | str\|null | `null` | Rerank precision: `null` = dequant, `"f16"` or `"f32"` = exact (uses more disk) |
| `tqdb_ef_construction` | int | `200` | HNSW build quality — higher = better recall, slower build |
| `tqdb_max_degree` | int | `32` | HNSW graph degree — higher = better recall, larger index |
| `tqdb_search_list_size` | int | `128` | ANN candidate list size (ef_construction alias for build; ef_search at query time) |
| `tqdb_alpha` | float\|null | `null` | HNSW pruning aggressiveness (`null` = TQDB default 1.2) |
| `tqdb_n_refinements` | int\|null | `null` | HNSW refinement passes during build (`null` = TQDB default 5) |

### 6.4 RAG Flags — Complete List

All flags are `false` in the shipped `config.yaml`. All can be set in config, as CLI flags, via `/rag` in the REPL, or as per-request fields in `POST /query`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `hybrid_search` | bool | false | Combine BM25 + vector scores |
| `hyde` | bool | false | Hypothetical document embedding |
| `multi_query` | bool | false | Generate 3 query paraphrases |
| `step_back` | bool | false | Abstract query to higher-level concept |
| `decompose` | bool | false | Split compound query into sub-questions |
| `compress` | bool | false | LLM context compression post-retrieval |
| `rerank` | bool | false | Cross-encoder (BGE) reranking |
| `sentence_window` | bool | false | Sentence-granularity indexing + window expansion |
| `crag_lite` | bool | false | Corrective retrieval on low-confidence results |
| `cite` | bool | true | Inline `[source: file]` citations in answers |
| `raptor` | bool | false | Hierarchical clustering summaries (ingest-time) |
| `graph_rag` | bool | false | Entity/relation graph retrieval |
| `discuss` | bool | true | Allow general-knowledge fallback when KB has no hits |
| `top_k` | int | 10 | Retrieved chunk count |
| `similarity_threshold` | float | 0.3 | Minimum cosine similarity for a chunk to qualify |
| `rerank_model` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `sentence_window_size` | int | 3 | ±N sentences around each hit |
| `crag_lite_confidence_threshold` | float | 0.4 | Below this score, CRAG-Lite triggers fallback |
| `query_router` | str | `heuristic` | `heuristic \| llm \| off` |
| `code_graph` | bool | false | Build File+Symbol nodes with CONTAINS/IMPORTS edges during code ingest |
| `code_graph_bridge` | bool | false | Add MENTIONED\_IN edges linking prose docs to code symbols |
| `graph_rag_mode` | str | `local` | `local \| global \| hybrid` |
| `graph_rag_depth` | str | `standard` | `light (no LLM) \| standard \| deep (+claims)` |
| `graph_rag_budget` | int | 3 | Max graph-expanded chunks injected into context |
| `graph_rag_relations` | bool | true | Extract relation triples (LLM-heavy) |
| `graph_rag_community` | bool | true | Build Louvain/Leiden community summaries |
| `graph_rag_community_backend` | str | `louvain` | `louvain` (safe default) or `leidenalg` |
| `graph_rag_relation_budget` | int | 30 | Max chunks per batch for relation extraction (`0` = unlimited) |
| `graph_rag_entity_min_frequency` | int | 2 | Prune entities appearing fewer than N times |
| `raptor_max_levels` | int | 1 | Summary tree depth |
| `raptor_min_source_size_mb` | float | 5.0 | Skip RAPTOR for sources smaller than this |

### 6.5 Web Search / CRAG-Lite

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `web_search.enabled` | bool | `false` | Enable Brave Search web fallback |
| `web_search.brave_api_key` | string | — | Brave API key (or set `BRAVE_API_KEY` env var) |
| `rag.crag_lite` | bool | `false` | Activate CRAG-Lite corrective retrieval |
| `rag.crag_lite_confidence_threshold` | float | `0.4` | Below this score, CRAG-Lite triggers fallback |
| `rag.truth_grounding` | bool | `false` | When CRAG-Lite fires, escalate to Brave web search |

Obtain a Brave API key at `https://brave.com/search/api/`. The free tier allows 2,000 queries/month.

### 6.6 Offline / Air-gapped Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offline.enabled` | bool | `false` | Block all outbound calls (HF Hub, cloud LLMs, web search); disables RAPTOR/GraphRAG automatically |
| `offline.local_assets_only` | bool | `false` | Block HF Hub downloads only; RAPTOR/GraphRAG remain enabled with a local Ollama |
| `offline.local_models_dir` | string | `""` | Root directory for all pre-downloaded HF model files |
| `offline.embedding_models_dir` | string | `""` | sentence-transformers / fastembed model root (overrides `local_models_dir`) |
| `offline.hf_models_dir` | string | `""` | GLiNER, REBEL, LLMLingua, and cross-encoder reranker model root |
| `offline.tokenizer_cache_dir` | string | `""` | tiktoken BPE encoding cache directory |
| `llm.models_dir` | string | `""` | Ollama model root directory |

Example `config.yaml` for a strict air-gap:

```yaml
offline:
  enabled: true
  local_models_dir: "/mnt/aimodels"
  embedding_models_dir: /mnt/aimodels/embedding
  hf_models_dir: /mnt/aimodels/hf
  tokenizer_cache_dir: /mnt/aimodels/tiktoken
llm:
  models_dir: /mnt/aimodels/ollama
```

**Compatible providers:** LLM: `ollama`, `vllm` · Embedding: `sentence_transformers`, `fastembed`, `ollama` · Vector store: `lancedb`, `chroma`

**Incompatible:** `openai`, `gemini`, `grok`, `github_copilot`, `ollama_cloud`, Brave web search

See [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) for the full setup walkthrough, model pre-download instructions, and troubleshooting.

### 6.7 Shell Passthrough Policy

Controls what `! <command>` runs in the REPL.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repl.shell_passthrough` | string | `local_only` | `local_only` — allow only for local writable projects · `always` — allow for all projects (not recommended in shared deployments) · `off` — disable `!` passthrough entirely |

### 6.8 AxonStore (Multi-User Sharing)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store.base` | string | `~/.axon` | Shared filesystem path for the AxonStore (also: `AXON_STORE_BASE` env var) |

See [AXON_STORE.md](AXON_STORE.md) for the full sharing lifecycle.

### 6.9 Environment Variables — Complete List

| Variable | Used By | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | LLM + embedding | Ollama server URL (default: `http://localhost:11434`) |
| `OPENAI_API_KEY` | LLM + embedding | OpenAI API key |
| `XAI_API_KEY` | LLM | xAI Grok API key (also accepted: `GROK_API_KEY`) |
| `GEMINI_API_KEY` | LLM | Google Gemini API key |
| `GITHUB_COPILOT_PAT` | LLM | Copilot OAuth token (also: `GITHUB_TOKEN`) |
| `OLLAMA_CLOUD_URL` | LLM | Remote Ollama endpoint URL |
| `OLLAMA_CLOUD_KEY` | LLM | Remote Ollama API key |
| `BRAVE_API_KEY` | Web search | Brave Search API key |
| `AXON_STORE_BASE` | Sharing | AxonStore shared filesystem path |
| `AXON_HOST` | API server | API server bind address (default: `127.0.0.1`) |
| `AXON_PORT` | API server | API server port (default: `8000`) |
| `RAG_INGEST_BASE` | Security | Restrict ingest to this directory tree (default: cwd) |
| `LOG_LEVEL` | Logging | `DEBUG \| INFO \| WARNING \| ERROR` (default: `INFO`) |
| `PYTHONUTF8` | Windows | Set to `1` to force UTF-8 mode on Windows |

### 6.10 Concurrency & Performance Tuning

Axon handles concurrency at two independent levels.

#### Web-server level (I/O concurrency)

The API server is built on **FastAPI + Uvicorn**, with all endpoints written as `async def`.

This means the server can accept and process hundreds of concurrent HTTP requests without blocking —

a query, an ingest job, and a share operation can all run simultaneously on a single process.

Start the server with multiple OS-level workers for true multi-process parallelism:

```bash
uvicorn axon.api:app --host 0.0.0.0 --port 8000 --workers 4
# or via the CLI entry-point env vars:
AXON_HOST=0.0.0.0 AXON_PORT=8000 axon-api
```

#### Engine level (CPU thread concurrency)

Heavy work (chunking, embedding calls, BM25 indexing, GraphRAG extraction) runs inside a

**shared background thread pool** inside `AxonBrain`.  The pool size is controlled by:

| `config.yaml` key | Default | Description |
|-------------------|---------|-------------|
| `max_workers` | `8` | Thread pool size for all background ingest and retrieval tasks |
| `graph_rag_map_workers` | `0` | Dedicated pool for GraphRAG map-reduce phase (`0` = share `max_workers`) |

```yaml
# Scale up on a 32-core server
max_workers: 16
# Isolate GraphRAG map-reduce so large graph jobs don't starve normal queries
graph_rag_map_workers: 8
```

**When to adjust `max_workers`:**

| Machine | Recommended value |
|---------|------------------|
| Laptop / CI (8 cores) | `4`–`8` (default) |
| Workstation (16–32 cores) | `12`–`24` |
| Server (32+ cores, multiple users) | `24`–`32` |

**When to set `graph_rag_map_workers`:**

Set this when running GraphRAG community finalization (`axon-api` or `/graph/finalize`) on

a machine that also serves live query traffic.  A dedicated pool ensures map-reduce LLM

batches do not block incoming search requests.

---

## 7. Governance Runbook

### 7.1 Check System Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/governance/overview
```

`/governance/overview` returns:

```json
{
  "active_projects": 3,
  "session_count": 12,
  "active_leases": 0,
  "maintenance_projects": []
}
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
# List active sessions
curl http://localhost:8000/governance/copilot/sessions
# Force-expire
curl -X POST http://localhost:8000/governance/copilot/session/<session_id>/expire
```

### 7.5 Rebuild GraphRAG Communities

```bash
# Check if rebuild needed
curl http://localhost:8000/graph/status
# Trigger rebuild
curl -X POST http://localhost:8000/graph/finalize
```

---

## 8. Surface Capability Matrix

Which operations are available on which surface:

| Capability | API | REPL | CLI | MCP |
|-----------|-----|------|-----|-----|
| Query (synthesis) | ✓ | ✓ | ✓ | ✓ |
| Query (streaming) | ✓ | — | ✓ | — |
| Search (raw) | ✓ | ✓ | — | ✓ |
| Ingest text | ✓ | — | — | ✓ |
| Ingest file/dir | ✓ | ✓ | ✓ | ✓ |
| Ingest URL | ✓ | ✓ | — | ✓ |
| Smart refresh | ✓ | ✓ | — | — |
| Stale detection | ✓ | ✓ | — | ✓ |
| Collection inspect | ✓ | ✓ | ✓ | ✓ |
| Delete documents | ✓ | — | — | ✓ |
| Clear knowledge base | ✓ | ✓ | — | ✓ |
| Project list | ✓ | ✓ | ✓ | ✓ |
| Project switch | ✓ | ✓ | ✓ | ✓ |
| Project create | ✓ | ✓ | ✓ | ✓ |
| Project delete | ✓ | ✓ | ✓ | ✓ |
| Config read | ✓ | ✓ | — | ✓ |
| Config update | ✓ | ✓ | — | ✓ |
| Session list | ✓ | ✓ | — | ✓ |
| Session load | ✓ | ✓ | — | ✓ |
| Share generate | ✓ | ✓ | — | ✓ |
| Share redeem | ✓ | ✓ | — | ✓ |
| Share revoke | ✓ | ✓ | — | — |
| Share list | ✓ | ✓ | — | ✓ |
| AxonStore init | ✓ | ✓ | — | ✓ |
| Graph status | ✓ | ✓ | — | ✓ |
| Graph finalize | ✓ | ✓ | — | ✓ |
| Graph data | ✓ | — | — | ✓ |
| Graph visualize | ✓ | ✓ | — | — |
| Maintenance state | ✓ | — | — | — |
| Lease registry | ✓ | — | — | ✓ |
| Governance audit | ✓ | — | — | — |

---

*See [API_REFERENCE.md](API_REFERENCE.md) for full request/response schemas.*

*See [MCP_TOOLS.md](MCP_TOOLS.md) for MCP tool signatures.*

*See [GOVERNANCE_CONSOLE.md](GOVERNANCE_CONSOLE.md) for the full audit and monitoring runbook.*

*See [EVALUATION.md](EVALUATION.md) for RAGAS metrics, smoke tests, and building testsets.*

*See [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) for air-gap setup, model pre-download, and local-assets-only mode.*
