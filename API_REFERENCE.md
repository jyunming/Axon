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
| `POST` | `/search` | Semantic / hybrid search — returns raw document chunks with scores |

**`POST /query` body:**
```json
{
  "query": "What is the ingestion pipeline?",
  "project": "my-project",
  "chat_history": [],
  "overrides": {"hyde": true, "rerank": true, "top_k": 20}
}
```

**`POST /search` body:**
```json
{
  "query": "AxonBrain",
  "top_k": 5,
  "project": "my-project"
}
```

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
`doc_id` is optional and defaults to a content hash. Each item in `docs` is a `BatchDocItem`.

---

## Collection Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collection` | Source and chunk counts for the active project |
| `GET` | `/collection/stale` | Documents not refreshed in N days (`?days=30`) |
| `GET` | `/tracked-docs` | Full tracked-document manifest with hashes and timestamps |
| `POST` | `/delete` | Remove documents by source path or URL list |
| `POST` | `/clear` | Wipe entire active project store and index (irreversible) |

**`POST /delete` body:**
```json
{"doc_ids": ["chunk-abc123", "chunk-def456"]}
```
Pass the `doc_ids` returned by `/tracked-docs` or `/collection`. To delete by source path,
look up the doc IDs with `GET /tracked-docs` first.

---

## Projects

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/projects` | List all local and mounted projects |
| `POST` | `/project/new` | Create a new named project |
| `POST` | `/project/switch` | Switch the active project |
| `POST` | `/project/delete/{name}` | Delete a project and all its data |

Pass `"project": "<name>"` in any request body to target a specific project instead of the
active default.

---

## Configuration (Runtime)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/config` | Return active RAG flags, model config, and runtime settings |
| `POST` | `/config/update` | Update LLM/embedding provider or RAG flags without restart |

**`POST /config/update` body (all fields optional):**
```json
{
  "llm_model": "gemma3:27b",
  "llm_provider": "ollama",
  "embedding_provider": "fastembed",
  "hyde": true,
  "rerank": true
}
```

Changes are scoped to the current server session and are not persisted to `config.yaml`.

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
| `GET` | `/graph/status` | Community detection build status (pending / complete / error) |
| `POST` | `/graph/finalize` | Trigger explicit community rebuild |
| `GET` | `/graph/data` | Full knowledge graph payload (used by VS Code panel) |

---

## Maintenance

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/project/maintenance` | Enter maintenance mode — blocks ingest/query while rebuilding |
| `GET` | `/project/maintenance` | Check maintenance mode status |

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
| `500` | Internal server error — check server logs |

Error bodies always include a `"detail"` field with a human-readable message.
