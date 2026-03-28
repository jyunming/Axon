# Axon MCP Tools Reference

Axon exposes a Model Context Protocol (MCP) server (`axon-mcp`) with **30 tools**.

> **Which integration should I use?**
> - **`@axon` chat participant** — install the VS Code extension (VSIX). Gives you a conversational `@axon` inside Copilot Chat. No `.vscode/mcp.json` needed.
> - **MCP tools** (this reference) — configure `axon-mcp` in your editor or agent CLI. Gives programmatic tool access to Claude Code, Codex CLI, Gemini CLI, Cursor, and VS Code Copilot agent mode.

`axon-api` must be running before any MCP client connects. See [SETUP.md § 10](SETUP.md#10-mcp-server-setup) for connection instructions per client.

---

## Ingestion (5)

### `ingest_text`

Ingest a single text string. Prefer `ingest_texts` for multiple documents — it uses one embedding call.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | The text content to ingest |
| `metadata` | dict | `null` | Key/value metadata; always set `"source"` so the collection can be audited |
| `project` | string | `null` | Target project — omit to use the active project |

**Returns:** `{"status": "ok", "chunks": N}` or `{"status": "skipped"}` for duplicate content.

### `ingest_texts`

Batch ingest a list of documents in a single embedding call. Preferred over calling `ingest_text` in a loop.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `docs` | `[{text, metadata?}]` | required | Each item must have `"text"`; optional keys: `"doc_id"`, `"metadata"` |
| `project` | string | `null` | Target project applied to all docs |

**Returns:** `{"status": "ok", "total": N}`

### `ingest_url`

Fetch an HTTP/HTTPS URL and ingest its text content. HTML is stripped automatically. Private/internal addresses are blocked server-side.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | Public HTTP or HTTPS URL |
| `metadata` | dict | `null` | Extra metadata merged with page metadata |
| `project` | string | `null` | Target project |

**Returns:** `{"job_id": "..."}` — poll with `get_job_status`.

### `ingest_path`

Walk and ingest a local file or directory (always async). Path must be within `RAG_INGEST_BASE`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | required | Absolute or relative path to a file or directory |

**Returns:** `{"job_id": "..."}` — poll with `get_job_status`.

### `get_job_status`

Poll the status of an async ingest job started by `ingest_path` or `ingest_url`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `job_id` | string | required | Job ID from `ingest_path` or `ingest_url` |

**Returns:** `{"job_id": "...", "status": "processing|completed|failed", "started_at": "...", "completed_at": "...", "path": "...", "error": null}`

---

## Search & Query (2)

### `search_knowledge`

Retrieve raw document chunks with scores. Best for multi-step reasoning where you want to inspect chunks before synthesising an answer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The search query |
| `top_k` | int | `5` | Number of chunks to return (must be ≥ 1) |
| `filters` | dict | `null` | Optional metadata filters, e.g. `{"source": "https://..."}` |
| `project` | string | `null` | Expected active project — returns 409 on mismatch; use `switch_project` to change |

**Returns:** `[{"text": "...", "score": 0.87, "metadata": {...}}]`

### `query_knowledge`

Full RAG query — retrieval + generation in one call. Use `search_knowledge` if you need to inspect chunks first.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The question to answer |
| `top_k` | int | `null` | Chunks to retrieve; null uses global config default |
| `filters` | dict | `null` | Optional metadata filters for retrieval |
| `project` | string | `null` | Expected active project — returns 409 on mismatch; use `switch_project` to change |

**Returns:** `{"response": "...", "provenance": {"answer_source": "...", "retrieved_count": N}, "settings": {...}}`

---

## Knowledge Base Management (5)

### `list_knowledge`

List indexed sources with chunk counts for the active project. No parameters.

**Returns:** `[{"source": "file.md", "chunks": 12}]`

### `delete_documents`

Remove documents from the index by chunk ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `doc_ids` | `[string]` | required | List of document IDs to delete |

**Returns:** `{"deleted": N}`

### `clear_knowledge`

Wipe the active project's vector store and BM25 index entirely. No parameters. **Irreversible.**

**Returns:** `{"status": "cleared"}`

### `get_stale_docs`

List documents not re-ingested within N days. Only tracks documents seen in the current server process — staleness resets on server restart.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | int | `7` | Staleness threshold in days |

**Returns:** `[{"source": "...", "last_seen": "...", "days_old": N}]`

### `get_active_leases`

List active write-lease counts per project. Use to check whether it is safe to put a project into maintenance state (wait for `active_leases` to reach 0 first). No parameters.

**Returns:** `{"project_name": {"active_leases": N, "draining": false, "epoch": 1}}`

---

## Project Management (4)

### `list_projects`

List all local projects and mounted shares. No parameters.

**Returns:** `[{"name": "...", "path": "...", "chunk_count": N}]`

### `switch_project`

Switch the active project. **Warning:** mutates global server state — do not call from concurrent handlers. Prefer passing `project` directly to ingest/query tools instead.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | string | required | Name of the project to activate |

**Returns:** `{"active_project": "..."}`

### `create_project`

Create a new named project with an isolated knowledge base.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Project name (max 5 slash-separated segments) |
| `description` | string | `""` | Optional human-readable description |

**Returns:** `{"name": "..."}`

### `delete_project`

Delete a project and all its stored data permanently. **Irreversible.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Project name to delete |

**Returns:** `{"status": "deleted"}`

---

## Settings (2)

### `get_current_settings`

Return active RAG flags, model config, and runtime settings. No parameters.

**Returns:** `{"provider": "ollama", "model": "llama3.1:8b", "top_k": 10, "hyde": false, ...}`

### `update_settings`

Toggle RAG flags and model settings at runtime. Changes are session-scoped and not persisted to `config.yaml`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | `null` | Retrieved chunk count (1–50) |
| `similarity_threshold` | float | `null` | Minimum match score (0.0–1.0) |
| `hybrid_search` | bool | `null` | BM25 + vector hybrid search |
| `rerank` | bool | `null` | Cross-encoder reranking |
| `hyde` | bool | `null` | Hypothetical Document Embeddings |
| `multi_query` | bool | `null` | Multi-query retrieval (3 rephrased queries merged) |
| `step_back` | bool | `null` | Step-back prompting (abstract query before retrieval) |
| `query_decompose` | bool | `null` | Query decomposition into sub-questions |
| `compress_context` | bool | `null` | LLM context compression before generation |
| `raptor` | bool | `null` | RAPTOR hierarchical summaries |
| `graph_rag` | bool | `null` | GraphRAG entity-graph retrieval |
| `graph_rag_mode` | string | `null` | GraphRAG query mode: `"local"`, `"global"`, or `"hybrid"` |
| `code_graph` | bool | `null` | Code-graph retrieval for code queries |
| `sentence_window` | bool | `null` | Sentence-window retrieval |
| `sentence_window_size` | int | `null` | Surrounding sentences per side (1–10) |
| `crag_lite` | bool | `null` | CRAG-lite corrective retrieval on low-confidence chunks |
| `truth_grounding` | bool | `null` | Truth-grounding enforcement on retrieved chunks |
| `discussion_fallback` | bool | `null` | Allow general-knowledge fallback when no chunks found |
| `cite` | bool | `null` | Inline source citations in generated answers |

**Returns:** `{"status": "updated"}`

---

## Sessions (2)

### `list_sessions`

List saved conversation sessions for the active project. Returns up to 20 most recent. No parameters.

**Returns:** `[{"id": "20260326_120000", "timestamp": "...", "turns": 5}]`

### `get_session`

Retrieve a full session transcript by ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | string | required | Session ID from `list_sessions` |

**Returns:** `{"id": "...", "history": [{"role": "user", "content": "..."}]}`

---

## AxonStore & Sharing (5)

### `get_store_status`

Check whether the AxonStore has been initialised on this machine. Returns store metadata when ready,

or `{"initialized": false}` on a fresh install. Call this before any other tool on first use to

decide whether to prompt the user to run `init_store`.

**No parameters.**

**Returns:** `{"initialized": true, "path": "~/.axon/...", "store_version": 2, "store_id": "...", "created_at": "...", "username": "alice"}` or `{"initialized": false, ...}`

---

### `init_store`

Move the store to a different base path (e.g. a shared network drive). Your data always uses the AxonStore directory structure — call this only when changing where it lives. Safe to call repeatedly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | string | required | New base path (e.g. `/data` → data lives at `/data/AxonStore/<user>/`) |
| `persist` | bool | `false` | Write the new path to `config.yaml` so it survives restarts |

**Returns:** `{"store_path": "...", "user_dir": "...", "username": "..."}`

### `share_project`

Generate a read-only share key for a project. Returns a `share_string` to send to the recipient out-of-band.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | string | required | Project to share (must exist) |
| `grantee` | string | required | OS username of the recipient |

**Returns:** `{"share_string": "axon-share:v1:...", "key_id": "..."}`

### `redeem_share`

Mount a shared project using a share string (read-only). After redemption the project appears as `mounts/{owner}_{project}`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `share_string` | string | required | Full share string from `share_project` |

**Returns:** `{"mount_path": "..."}`

### `list_shares`

List outgoing shares (projects this user has shared, with revocation status) and incoming shares (projects shared with this user, with mount names). No parameters.

**Returns:** `{"sharing": [...], "shared": [...]}`

---

## Graph (3)

### `graph_status`

Return entity count, edge count, community summary count, rebuild status, and readiness for graph-augmented retrieval. Call before `graph_finalize` to check if a rebuild is needed. No parameters.

**Returns:** `{"entities": 340, "edges": 820, "communities": 12, "rebuild_in_progress": false, "ready": true}`

### `graph_finalize`

Trigger an explicit community detection rebuild. Call after a large ingest batch to ensure graph-augmented answers reflect the latest knowledge. No parameters.

**Returns:** `{"communities_built": N}`

### `graph_data`

Return the full entity/relation graph as JSON for inspection, export, or custom visualisations. No parameters.

**Returns:** `{"nodes": [...], "links": [...]}`

---

## Usage Notes

- All tools operate on the **active project**. Most ingest, search, and query tools accept an optional `project` parameter validated against the active project (returns 409 on mismatch). Tools that do **not** accept `project`: `ingest_path`, `list_sessions`, `get_session`, `list_shares`. Use `switch_project` to change the active project.

- `ingest_path` and `ingest_url` are always async — they return a `job_id`. Poll `get_job_status` until `status == "completed"` or `"failed"`.

- `clear_knowledge` and `delete_project` are irreversible.

- Mounted shares (via `redeem_share`) are always **read-only**. Ingest calls against a mount return an error.

- `update_settings` changes are scoped to the current session and are not persisted to `config.yaml`.

- AxonStore tools (`init_store`, `share_project`, `redeem_share`, `list_shares`) are always available. Call `init_store` only if you want to move the store to a shared drive.
