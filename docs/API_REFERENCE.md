# Axon REST API Reference

Full endpoint reference for the Axon FastAPI server (`axon-api`, default port 8000).

For interactive exploration, open `http://localhost:8000/docs` (Swagger UI) or
`http://localhost:8000/redoc` after starting the server.

> Every endpoint below is also reachable under `/v1/...` (every router is dual-mounted at the root and the `/v1` prefix). The `/v1` mount is the recommended path for clients that pin to a major API version.

> **Operators:** for governance runbooks, maintenance state workflows, and the complete CLI/REPL/MCP reference, see [ADMIN_REFERENCE.md](ADMIN_REFERENCE.md).

---

## Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/live` | Liveness probe — returns `{"status": "alive"}` with `200` as long as the ASGI process is responding; does **not** check whether the brain is initialised |
| `GET` | `/health/ready` | Readiness probe — returns `{"status": "ok", "project": "<active-project>"}` with `200` only when the brain is fully initialised; returns `{"status": "initializing"}` with `503` during cold start |
| `GET` | `/health` | Backward-compatible alias for `/health/ready` — returns the same payload; preserved so existing uptime checkers and VS Code extension probes continue to work |

> **Kubernetes usage:** point `livenessProbe` at `/health/live` (restart only on a fully wedged process) and `readinessProbe` at `/health/ready` (hold traffic until the brain and vector store are ready).

---

## Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/metrics` | Prometheus exposition format metrics (`text/plain; version=0.0.4`) — returns `503` with a plain message if `prometheus-client` is not installed |

**Exposed metrics:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `axon_requests_total` | Counter | `path`, `method`, `status` | Total HTTP requests handled; lets operators alert on 5xx spikes per route |
| `axon_request_duration_seconds` | Histogram | `path`, `method` | Wall-clock request latency; enables p50/p95 dashboards |
| `axon_query_total` | Counter | `project`, `surface` | Total RAG queries; tracks volume per project and calling surface (`api`, `mcp`, `repl`, etc.) |
| `axon_ingest_total` | Counter | `project`, `surface` | Total ingest requests per project and surface |
| `axon_brain_ready` | Gauge | — | `1` when `axon.api.brain` is initialised, `0` otherwise; refreshed on every scrape |

`prometheus-client` ships in the base `axon-rag` dependency set (`>=0.20`), so the `503` fallback is not expected under normal install conditions.

---

## Query & Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Full RAG query — returns synthesised answer with optional citations |
| `POST` | `/query/stream` | Streaming RAG query — returns Server-Sent Events (`text/event-stream`) |
| `POST` | `/query/visualize` | Full RAG query — returns a self-contained HTML page with answer, citations, and highlighted graph nodes |
| `POST` | `/search` | Semantic / hybrid search — returns raw document chunks with scores |
| `POST` | `/search/raw` | Retrieval without LLM synthesis — returns chunks plus optional trace |
| `POST` | `/search/visualize` | Like `/query/visualize` but without LLM synthesis — shows raw retrieved chunks in HTML |

**`POST /query` body:**
```json
{
  "query": "What is the ingestion pipeline?",   // required
  "project": null,           // optional — must match active project if set; omit to use active
  "stream": false,           // stream tokens via SSE instead of single response
  "top_k": null,             // chunks to retrieve — null inherits global config (default 10)
  "threshold": null,         // minimum similarity score — null inherits global config (default 0.3)
  "hyde": null,              // HyDE query expansion — null inherits global config
  "rerank": null,            // cross-encoder reranking — null inherits global config
  "multi_query": null,       // 3-paraphrase expansion — null inherits global config
  "step_back": null,         // step-back abstraction — null inherits global config
  "decompose": null,         // sub-question decomposition — null inherits global config
  "compress": null,          // context compression — null inherits global config
  "cite": null,              // inline citations — null inherits global config (default true)
  "discuss": null,           // general-knowledge fallback — null inherits global config (default true)
  "graph_rag": null,         // entity-graph retrieval — null inherits global config
  "raptor": null,            // RAPTOR hierarchical retrieval — null inherits global config
  "crag_lite": null,         // corrective retrieval — null inherits global config
  "temperature": null,       // LLM temperature override — null inherits global config
  "timeout": null,           // request timeout in seconds — null inherits global config
  "include_diagnostics": false,  // include confidence scores in response
  "include_citations": true, // v0.3.2 — when true, response carries sources + citations arrays (Claude / OpenAI compatible). Set false for high-throughput agents
  "dry_run": false,          // skip LLM synthesis; return retrieved chunks only
  "chat_history": []         // prior turns for multi-turn conversation context
}
```

> **`null` vs explicit value:** All RAG flag fields default to `null`, meaning they inherit the
> server's current global config. Pass an explicit `true`/`false` to override for this request only.
> **Project field:** `project` must match the brain's currently active project. If it
> differs, the server returns `409 Conflict`. Use `POST /project/switch` first to change
> the active project. Omit `project` to query whichever project is currently active.

**`POST /query` response** includes a `provenance` object on every non-dry-run response, and (when `include_citations: true` — the default) a `sources` array plus a `citations` array:
```json
{
  "query": "...",
  "response": "First fact [1]. Second fact [Document 2].",
  "settings": {...},
  "provenance": {
    "answer_source": "local_kb",
    "retrieved_count": 5,
    "web_count": 0
  },
  "sources": [
    {
      "index": 0,
      "id": "chunk-abc",
      "source": "README.md",
      "title": "README.md",
      "score": 0.91,
      "is_web": false,
      "url": null,
      "text": "Truncated chunk text… (max 500 chars + ellipsis)",
      "metadata": {"file_path": "...", "page": 0, "symbol_name": "..."}
    }
  ],
  "citations": [
    {
      "marker": "[1]",
      "document_index": 0,
      "document_title": "README.md",
      "document_id": "chunk-abc",
      "start_in_response": 11,
      "end_in_response": 14
    }
  ]
}
```
`answer_source` values: `"local_kb"` (retrieved from knowledge base), `"web_snippet_fallback"` (Brave web results used), `"no_context_fallback"` (no retrieval; LLM answered from general knowledge because `discussion_fallback=true`), `"no_results"` (no retrieval and strict mode returned a refusal).

`sources` is the slim view of every chunk made available to the LLM (indexed 0..N-1; markers `[N]` map to `sources[N-1]`). `citations` is parsed from the response — one entry per `[N]` or `[Document N]` marker — with character offsets so frontends can render inline citations without re-scanning the response. Out-of-range markers are silently dropped. Set `include_citations: false` on the request body to skip both arrays (e.g. high-throughput agents that only need the answer string).

**`POST /query/stream` body:** same as `/query`.
Response: Server-Sent Events stream. Each event is `data: <json>\n\n` where `<json>` is a
text chunk string. The stream closes when generation completes. Errors emit
`data: [ERROR] <message>\n\n` instead of a normal chunk.

**`POST /search` body:**
```json
{
  "query": "AxonBrain",        // required
  "top_k": null,               // chunks to return — null inherits global config (default 10)
  "threshold": null,           // minimum similarity — null inherits global config (default 0.3)
  "project": null,             // optional — must match active project if set
  "filters": null,             // optional metadata key/value filters
  "hybrid": null               // BM25+vector hybrid mode — null inherits global config
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
| `POST` | `/ingest/upload` | **Synchronous** multipart file upload — used by VS Code extension and webapp drag-drop. Ingests inline and returns `{status, files, ingested_files, ingested_chunks}`; no `job_id` polling required (unlike `/ingest`) |
| `GET` | `/ingest/status/{job_id}` | Poll async ingest job status |
| `POST` | `/ingest/refresh` | Re-ingest files whose content has changed |
| `POST` | `/ingest_url` | Fetch and ingest content from a remote URL |
| `POST` | `/add_text` | Ingest a single text string |
| `POST` | `/add_texts` | Batch ingest a list of text strings |
| `POST` | `/delete` | Delete documents matching a metadata filter or doc_id list |

**`POST /ingest` body:**
```json
{
  "path": "/path/to/documents",  // required — absolute path to file or directory
  "project": null,                // optional — must match active project if set
  "raptor": null,                 // enable RAPTOR during ingest — null inherits global config
  "graph_rag": null               // enable entity extraction during ingest — null inherits global config
}
```
Response: `{"message": "Ingestion started for /path/to/documents", "job_id": "abc123", "status": "processing"}`

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
{
  "url": "https://example.com/page",  // required
  "metadata": {},                      // optional key/value metadata attached to all chunks
  "project": null                      // optional — must match active project if set
}
```

**`POST /add_text` body:**
```json
{
  "text": "The content to ingest",  // required
  "metadata": {},                    // optional key/value metadata (default: empty dict)
  "doc_id": null,                    // optional stable ID; auto-generated as agent_doc_<hex> if omitted
  "project": null                    // optional — must match active project if set
}
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
| `POST` | `/project/seal` | Seal an unsealed project — encrypt every content file in place with AES-256-GCM. Idempotent on already-sealed projects. Requires `[sealed]`/`[starter]` extra |
| `POST` | `/project/rotate-keys` | Rotate the project DEK; re-encrypt every sealed content file; selectively re-wrap surviving share KEKs. Equivalent to a hard-revoke without a specific key_id |
| `POST` | `/mount/refresh` | Refresh shared-store mount descriptors — call after the owner has issued/revoked a share so the grantee's `mounts/` reflects the current set |

Use `POST /project/switch` to change the active project before querying, searching, or ingesting.
All routes that accept a `"project"` field validate it against the active project and return
`409` if it does not match. The field is not a cross-project targeting mechanism — it is a safety
check to prevent accidental writes or reads to the wrong corpus.

---

## Configuration (Runtime)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config` | Return active RAG flags, model config, and runtime settings (sensitive fields masked as `"***"`: `api_key`, `gemini_api_key`, `ollama_cloud_key`, `copilot_pat`, `brave_api_key`, `qdrant_api_key`) |
| `GET` | `/config/validate` | Validate the current `config.yaml` — returns `{"valid": bool, "issue_count": int, "issues": [...]}` |
| `POST` | `/config/update` | Update LLM/embedding provider or RAG flags without restart |
| `POST` | `/config/set` | Set a single config field using dot-notation (e.g. `chunk.strategy`) — accepts `{"key": "rag.top_k", "value": 20, "persist": false}` |
| `POST` | `/config/reset` | Reset `config.yaml` to built-in defaults — returns `{"written_to": "<path>"}`. Running brain is not reloaded; call `POST /config/update` to apply |

**`POST /config/update` body (all fields optional):**
```json
{
  "llm_model": null,           // LLM model name (default: null — no change)
  "llm_provider": null,        // LLM provider: ollama | openai | gemini | grok | vllm | copilot | github_copilot | ollama_cloud (default: null — no change)
  "embedding_provider": null,  // Embedding provider (default: null — no change)
  "hyde": null,                // Override HyDE flag (default: null — no change)
  "rerank": null,              // Override rerank flag (default: null — no change)
  "multi_query": null,         // Override multi-query flag (default: null — no change)
  "top_k": null,               // Override retrieved chunk count (default: null — no change)
  "persist": false             // Write changes to config.yaml (default: false — session-scoped only)
}
```

Changes are scoped to the current server session by default (`persist: false`). Set
`"persist": true` to write the changes to `config.yaml` for durability across restarts.

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
| `GET` | `/graph/backend/status` | Active graph backend type and health (distinguishes graphrag vs dynamic backend) |
| `GET` | `/graph/conflicts` | List facts with `status='conflicted'` (incompatible exclusive-relation facts) |
| `POST` | `/graph/retrieve` | Run the active graph backend's `retrieve()` directly — point-in-time + per-query federation weights |

`/graph/status` response: `{"community_build_in_progress": false, "community_summary_count": 12, "entity_count": 340, "code_node_count": 0, "graph_ready": true}`. `graph_ready` is `true` once the entity graph or code graph has nodes — use this to know whether graph queries will return data before the full ingest job completes.

`/graph/finalize` response includes `status` (`"ok"` | `"not_applicable"` | `"error"`) and `detail` so callers can tell "ran and built nothing" apart from "this backend has no finalize step". The `dynamic_graph` backend always returns `not_applicable`; `federated` aggregates from sub-backends.

`/graph/conflicts` query params: `project` (optional, must match active project), `limit` (1-1000, default 100). Response: `{"backend": "...", "supported": bool, "conflicts": [{"fact_id", "subject", "relation", "object", "valid_at", "invalid_at", "scope_key", ...}, ...]}`. Backends without conflict tracking return `supported: false`.

`/graph/retrieve` request body: `{"query": "...", "top_k"?: int, "point_in_time"?: ISO-8601, "federation_weights"?: {"graphrag": 0.5, "dynamic_graph": 1.5}, "project"?: "..."}`. Returns graph contexts only (no LLM call). Use this to surface historical queries on bi-temporal backends (`dynamic_graph`) and per-question federation weight overrides; the main `/query` endpoint still routes through the legacy GraphRAG mixin and does not yet expose `point_in_time`.

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

## Security (sealed-mount lifecycle)

Master-key + passphrase lifecycle for sealed projects. All routes under `/security/*`.
Requires the `[sealed]` extra (already bundled in `[starter]`).

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/security/status` | Sealed store status — `{initialized, unlocked, sealed_hidden_count, public_key_fingerprint, cipher_suite}` |
| `POST` | `/security/bootstrap` | One-time setup — generate the master key under a passphrase. Body: `{passphrase}` |
| `POST` | `/security/unlock` | Unlock the sealed store with the bootstrap passphrase. Body: `{passphrase}` |
| `POST` | `/security/lock` | Lock the sealed store — drops the master from process memory. No body |
| `POST` | `/security/change-passphrase` | Re-wrap the master under a new passphrase. Body: `{old_passphrase, new_passphrase}` |
| `GET`  | `/suggestions/passphrase` | **v0.4.0 Item 1** — generate a Diceware passphrase from the bundled EFF wordlist (CC BY 3.0 US, 7,776 words). Query: `?words=N&separator=S` (default `words=6` ≈ 77 bits, `separator=-`). Returns `{passphrase, n_words, entropy_bits, separator, source}`. Subject to global X-API-Key middleware. |
| `POST` | `/security/keyring-mode` | **v0.4.0 Item 2** — change DEK storage mode at runtime. Body `{"mode": "persistent"\|"session"\|"never"}`. 422 on invalid mode. Caveat: previously stored secrets are NOT migrated. For permanent change, set `security.keyring_mode` in config.yaml and restart. `GET /security/status` now also returns `keyring_mode` + `session_cache_size`. |

> Sealed-mount admin routes (`/project/seal`, `/project/rotate-keys`) live in the **Projects** section but require an unlocked store — call `/security/unlock` first if your shell or process restarted. See [SHARING.md](SHARING.md) for the full owner/grantee flow.

---

## AxonStore & Sharing

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/store/status` | AxonStore initialisation status and metadata — safe to call before the brain is ready; polls `store_meta.json` directly |
| `GET` | `/store/whoami` | AxonStore identity — returns `username` and `store_path` |
| `POST` | `/store/init` | Change the store base path (e.g. to a shared drive) |
| `POST` | `/share/generate` | Generate a read-only share key. Auto-detects sealed vs plaintext based on whether the project is sealed |
| `POST` | `/share/redeem` | Mount a shared project using a share string |
| `POST` | `/share/revoke` | Revoke a share by `key_id`. Pass `rotate: true` for hard revoke (rotates DEK) |
| `POST` | `/share/extend` | Push out the expiry of an issued share. **Plaintext shares only.** Sealed (`ssk_`) `key_id`s are not in the plaintext share manifest and currently return `404 Key not found`. To extend a sealed share, mint a fresh sealed share with the desired `ttl_days` and revoke the old one — sealed expiry lives in an Ed25519-signed sidecar that cannot be re-signed in place. Body: `{key_id, ttl_days}` |
| `GET`  | `/share/list` | List outgoing shares (sharing) and incoming mounts (shared) |

**`POST /store/init` body:**
```json
{
  "base_path": "/shared/axon-store",  // required — absolute path to shared filesystem location
  "persist": false                     // write store path to config.yaml (default: false — session-scoped)
}
```

**`POST /share/generate` body:**
```json
{
  "project": "my-project",             // required — project to share
  "grantee": "alice",                  // required — identifier of the recipient
  "ttl_days": null                     // optional positive integer (days until expiry, default null = no expiry).
                                       // v0.4.0: honoured by BOTH sealed (ssk_) and plaintext (sk_) shares.
                                       // ttl_days <= 0 returns 422.
}
```
Response carries `key_id` (`sk_` for plaintext, `ssk_` for sealed), `share_string` (base64 envelope; v0.4.0+ sealed shares always decode to `SEALED2:...` — the legacy 6-field `SEALED1:...` format is accepted on redeem but no longer emitted), and `expires_at` (ISO 8601 Z-suffixed UTC when `ttl_days` was set, else `null`). When `ttl_days` is set on a sealed share, an Ed25519-signed expiry sidecar is written next to the wrap and a TTL check fires on every grantee mount; an expired share auto-destroys the local DEK + cache + mount descriptor on next mount attempt.

**`POST /share/redeem` body:**
```json
{
  "share_string": "axon-share:v1:..."  // required — full share string from /share/generate
}
```

**`POST /share/revoke` body:**
```json
{
  "key_id": "ssk_a4f9c1d2",   // required — key_id from /share/list (sk_ plaintext, ssk_ sealed)
  "project": "research",      // required for sealed shares (ssk_); optional for plaintext
  "rotate": false             // hard revoke — rotate project DEK + re-encrypt + selective re-wrap
}
```

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
| `409` | Conflict — duplicate project name, ongoing ingest, or `project` field mismatch |
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
