# Axon REST API Reference

Full endpoint reference for the Axon FastAPI server (`axon-api`, default port 8000).

For interactive exploration, open `http://localhost:8000/docs` (Swagger UI) or
`http://localhost:8000/redoc` after starting the server.

---

## Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` — use for liveness probes |

---

## Query & Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Full RAG query — returns synthesised answer with optional citations |
| `POST` | `/query/stream` | Streaming RAG query — returns Server-Sent Events (`text/event-stream`) |
| `POST` | `/search` | Semantic / hybrid search — returns raw document chunks with scores |
| `POST` | `/search/raw` | Retrieval without LLM synthesis — returns chunks plus optional trace |

**`POST /query` body:**
```json
{
  "query": "What is the ingestion pipeline?",
  "project": "my-project",
  "hyde": true,
  "rerank": true,
  "top_k": 20
}
```

> **Project field:** `project` must match the brain's currently active project. If it
> differs, the server returns `409 Conflict`. Use `POST /project/switch` first to change
> the active project. Omit `project` to query whichever project is currently active.

**`POST /query` response** includes a `provenance` object on every non-dry-run response:
```json
{
  "query": "...",
  "response": "...",
  "settings": {...},
  "provenance": {
    "answer_source": "local_kb",
    "retrieved_count": 5,
    "web_count": 0
  }
}
```
`answer_source` values: `"local_kb"` (retrieved from knowledge base), `"web_snippet_fallback"` (Brave web results used), `"no_context_fallback"` (no retrieval; LLM answered from general knowledge because `discussion_fallback=true`), `"no_results"` (no retrieval and strict mode returned a refusal).

**`POST /query/stream` body:** same as `/query`.
Response: Server-Sent Events stream. Each event is `data: <json>\n\n` where `<json>` is a
text chunk string. The stream closes when generation completes. Errors emit
`data: [ERROR] <message>\n\n` instead of a normal chunk.

**`POST /search` body:**
```json
{
  "query": "AxonBrain",
  "top_k": 5,
  "project": "my-project"
}
```

> **Project field:** same enforcement as `/query` — returns `409` on mismatch.

**`POST /search/raw` body:** same as `/search`. Always returns a `diagnostics` object.
Accepts optional `?include_trace=true` query parameter to additionally include the full
retrieval `trace` object (pipeline step timings and intermediate results).

---

## Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Async file/directory ingest — returns `job_id` immediately |
| `GET` | `/ingest/status/{job_id}` | Poll async ingest job status |
| `POST` | `/ingest/refresh` | Re-ingest files whose content has changed |
| `POST` | `/ingest_url` | Fetch and ingest content from a remote URL |
| `POST` | `/add_text` | Ingest a single text string |
| `POST` | `/add_texts` | Batch ingest a list of text strings |

**`POST /ingest` body:**
```json
{"path": "/path/to/documents"}
```
Response: `{"message": "Ingestion started", "job_id": "abc123", "status": "processing"}`

**`GET /ingest/status/{job_id}` response:**
```json
{
  "job_id": "abc123",
  "status": "processing",
  "phase": "embedding",
  "path": "/path/to/documents",
  "started_at": "2026-03-23T10:00:00Z",
  "completed_at": null,
  "files_total": 42,
  "chunks_total": 318,
  "chunks_embedded": 128,
  "documents_ingested": null,
  "error": null,
  "community_build_in_progress": false
}
```
`phase` values: `loading` → `chunking` → `raptor` → `graph_build` → `embedding` → `code_graph` → `finalizing` → `completed` (or `failed`). Phases that are disabled by config are still reported as the job moves past them. `chunks_embedded` updates per 32-chunk batch during the `embedding` phase.

**`POST /ingest_url` body:**
```json
{"url": "https://example.com/page"}
```

**`POST /add_texts` body:**
```json
{
  "docs": [
    {"doc_id": "a", "text": "First document", "metadata": {"source": "a.txt"}},
    {"doc_id": "b", "text": "Second document", "metadata": {"source": "b.txt"}}
  ]
}
```
`doc_id` is optional; if omitted it is auto-generated as `agent_doc_<8-hex>`. Each item in `docs` is a `BatchDocItem`.

---

## Collection Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collection` | Source and chunk counts for the active project |
| `GET` | `/collection/stale` | Documents not refreshed in N days (`?days=30`) |
| `GET` | `/tracked-docs` | Full tracked-document manifest with hashes and timestamps |
| `POST` | `/delete` | Remove documents by internal chunk ID list |
| `POST` | `/clear` | Wipe entire active project store and index (irreversible) |

**`POST /delete` body:**
```json
{"doc_ids": ["chunk-abc123", "chunk-def456"]}
```
**Important:** `/delete` accepts **internal chunk IDs** only — not the source-level `doc_id`
returned by `/add_text`, `/ingest_url`, or `/add_texts`. Those ingest responses identify the
source document, not individual vector chunks. `GET /collection/stale` returns source-dedup
metadata (source path and ingest timestamp); its `doc_id` field is a source identifier, **not**
a chunk ID usable with `/delete`. `GET /tracked-docs` returns content hashes and timestamps and
also does not expose chunk IDs. Currently there is no public endpoint that maps a source document
to its chunk IDs; delete by source is not yet a clean public contract. Use `GET /collection` to
inspect ingested sources and chunk counts.

---

## Projects

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/projects` | List all local and mounted projects |
| `POST` | `/project/new` | Create a new named project |
| `POST` | `/project/switch` | Switch the active project |
| `POST` | `/project/delete/{name}` | Delete a project and all its data |

Use `POST /project/switch` to change the active project before querying, searching, or ingesting.
All routes that accept a `"project"` field validate it against the active project and return
`409` if it does not match. The field is not a cross-project targeting mechanism — it is a safety
check to prevent accidental writes or reads to the wrong corpus.

---

## Configuration (Runtime)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config` | Return active RAG flags, model config, and runtime settings (sensitive fields masked as `"***"`: `api_key`, `gemini_api_key`, `ollama_cloud_key`, `copilot_pat`, `brave_api_key`, `qdrant_api_key`) |
| `POST` | `/config/update` | Update LLM/embedding provider or RAG flags without restart |

**`POST /config/update` body (all fields optional):**
```json
{
  "llm_model": "llama3.1:8b",
  "llm_provider": "ollama",
  "embedding_provider": "fastembed",
  "hyde": true,
  "rerank": true
}
```

Changes are scoped to the current server session by default. To write them to `config.yaml`,
add `"persist": true` to the request body.

---

## Sessions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/sessions` | List saved conversation sessions (up to 20 most recent) |
| `GET` | `/session/{id}` | Retrieve a full session transcript by timestamp ID |

---

## GraphRAG

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graph/status` | Graph readiness and community build status |
| `POST` | `/graph/finalize` | Trigger explicit community rebuild |
| `GET` | `/graph/data` | Full knowledge graph payload as JSON (nodes + links) |
| `GET` | `/graph/visualize` | Render the knowledge graph as a self-contained HTML page |
| `GET` | `/code-graph/data` | Code structure graph payload for the VS Code code-graph panel |

`/graph/status` response: `{"community_build_in_progress": false, "community_summary_count": 12, "entity_count": 340, "code_node_count": 0, "graph_ready": true}`. `graph_ready` is `true` once the entity graph or code graph has nodes — use this to know whether graph queries will return data before the full ingest job completes.

`/graph/visualize` returns `text/html` — open in a browser or embed in an iframe.
`/graph/data` and `/code-graph/data` return `{"nodes": [...], "links": [...]}` JSON payloads
consumed by the VS Code extension's graph panels.

---

## Maintenance

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/project/maintenance` | Set maintenance state for a project |
| `GET` | `/project/maintenance` | Check maintenance state (`?name=<project>`) |

**`POST /project/maintenance` body:**
```json
{"name": "my-project", "state": "readonly"}
```
Valid states: `normal`, `draining`, `readonly`, `offline`.

---

## Copilot / VS Code Bridge

These endpoints support GitHub Copilot agent integrations and the optional VS Code Copilot LLM bridge (`axon.useCopilotLlm=true`). The VS Code extension's standard query and tool flows use `/query` and `/search` directly — not `/copilot/agent`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/copilot/agent` | Internal SSE chat endpoint for direct Copilot agent integrations — not used by the VS Code extension's primary query/tool flows |
| `GET` | `/llm/copilot/tasks` | Poll for pending LLM tasks queued by the backend for VS Code to execute |
| `POST` | `/llm/copilot/result/{task_id}` | Submit the result of a Copilot LLM task back to the backend |

**`POST /copilot/agent` body:**
```json
{
  "messages": [{"role": "user", "content": "/search AxonBrain architecture"}],
  "agent_request_id": "req-abc123"
}
```

Response: Server-Sent Events stream (`text/event-stream`). Event sequence:

| Event type | Payload | Description |
|------------|---------|-------------|
| `created` | `{"type": "created", "id": "<request_id>"}` | Stream opened |
| `text` | `{"type": "text", "content": "<markdown>"}` | Response chunk |
| `error` | `{"type": "error", "message": "<msg>"}` | Error during processing |
| `[DONE]` | raw string | Stream closed |

**Supported slash commands in the `content` field:**

| Command | Effect |
|---------|--------|
| `/search <query>` | Run hybrid retrieval and return top-5 results |
| `/ingest <url>` | Fetch and ingest a URL into the active project |
| `/projects` | List available Axon projects |
| _(any other text)_ | Full RAG query with chat history |

**`GET /llm/copilot/tasks`** returns `{"tasks": [...]}` — the list is consumed and cleared on
each poll. VS Code calls this to receive tasks queued while the user was offline.

**`POST /llm/copilot/result/{task_id}` body:**
```json
{"result": "<llm_response_text>", "error": null}
```

---

## Registry & Leases

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/registry/leases` | List active read/write leases held via AxonStore |

---

## AxonStore & Sharing

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/store/whoami` | AxonStore identity — returns `username` and `store_path` |
| `POST` | `/store/init` | Initialise AxonStore at a shared filesystem base path |
| `POST` | `/share/generate` | Generate a read-only share key for a project and grantee |
| `POST` | `/share/redeem` | Mount a shared project using a share string |
| `POST` | `/share/revoke` | Revoke a share by `key_id` |
| `GET` | `/share/list` | List outgoing and incoming shares |

See [AXON_STORE.md](AXON_STORE.md) for the full sharing lifecycle guide.

---

## Error Responses

All endpoints return standard HTTP status codes:

| Code | Meaning |
|------|---------|
| `200` | Success |
| `202` | Accepted (async job started) |
| `400` | Bad request — invalid body or parameters |
| `403` | Forbidden — attempted write to a read-only mount |
| `404` | Not found — project, job, or session ID does not exist |
| `409` | Conflict — duplicate project name or ongoing ingest |
| `503` | Service unavailable — brain not initialized (start the server first) |
| `500` | Internal server error — check server logs |

Error bodies always include a `"detail"` field with a human-readable message.

---

## Governance Routes `/governance/*`

Operator console routes for audit log, project state, and Copilot session management.
Requires no additional auth beyond the standard `X-API-Key` header.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/governance/overview` | Aggregated operator status (project, graph, leases, Copilot) |
| `GET` | `/governance/audit` | Audit log (`?project=&action=&surface=&status=&since=&limit=50`) |
| `GET` | `/governance/copilot/sessions` | Active + recent Copilot bridge sessions |
| `GET` | `/governance/projects` | All projects with maintenance + graph state |
| `POST` | `/governance/graph/rebuild` | Audited graph community rebuild |
| `POST` | `/governance/project/maintenance` | Audited maintenance state change (`?name=&state=`) |
| `POST` | `/governance/copilot/session/{id}/expire` | Force-close a stuck Copilot session |

All write routes emit `X-Request-ID` into the audit trail for end-to-end traceability.

See [GOVERNANCE_CONSOLE.md](GOVERNANCE_CONSOLE.md) for full schema, runbooks, and the VS Code panel guide.
