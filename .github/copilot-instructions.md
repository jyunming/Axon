# Axon Knowledge Center — Copilot Instructions

These instructions give GitHub Copilot persistent context about the Axon
knowledge center so you can ingest and retrieve information correctly across
every session.

---

## What Axon Is

Axon is a local RAG (Retrieval-Augmented Generation) system exposed as a
REST API (`localhost:8000`) and as an MCP stdio server. It stores embeddings
in a local vector store (ChromaDB by default) and a BM25 index. All storage
is free and local; only the LLM call (if using Ollama) runs locally too.

---

## Project Namespaces

Projects group related knowledge. The active project determines which vector
store collection and BM25 index are read and written.

| Project name | Use for |
|---|---|
| `default` | General-purpose knowledge, catch-all |
| (add project-specific rows here as namespaces are created) |

Switch the active project with `POST /project/switch` before ingesting if the
content belongs to a specific namespace.

### Hierarchical projects (up to 3 levels)

Projects support slash-separated nesting up to 3 levels deep:

```
research/               ← top-level parent
research/papers/        ← child
research/papers/2024    ← grandchild (max depth)
```

**Searching a parent automatically searches all its descendants.** When you
`switch_project("research")`, Axon builds a `MultiVectorStore` and
`MultiBM25Retriever` that fan out queries across `research`, `research/papers`,
and `research/papers/2024` — results are merged and ranked together.

Writes (ingestion) always go to the parent's own store. Reads fan out.

**Pattern for searching across unrelated topics:** If you need to search both
`react-docs` and `python-stdlib` together, place them under a shared parent:
`docs/react-docs` and `docs/python-stdlib`. Then `switch_project("docs")`
searches both in one call.

Use `list_projects` to discover available namespaces before switching.

---

## Ingestion Workflow

### Always prefer batch ingestion

Use `POST /add_texts` (batch) instead of `POST /add_text` (single) whenever
you have more than one document to ingest. A single batch call embeds all
documents in one round-trip.

```
POST /add_texts
{
  "docs": [
    {"text": "...", "metadata": {"source": "https://...", "topic": "..."}},
    {"text": "...", "metadata": {"source": "https://..."}}
  ]
}
```

### Always set `metadata.source`

Every document should carry a `source` field so the collection can be
audited and stale content can be identified later.

```json
{"source": "https://react.dev/reference/hooks", "topic": "react"}
```

### Ingest a URL directly

Use `POST /ingest_url` instead of manually fetching and pasting content.
Axon fetches the page, strips HTML, and embeds the text:

```
POST /ingest_url
{"url": "https://example.com/docs/page", "metadata": {"topic": "example"}}
```

Private and cloud-metadata URLs (`127.x`, `169.254.x`, `10.x`, `192.168.x`,
`172.16–31.x`) are blocked by the loader — do not attempt to ingest them.

### Check for duplicates before ingesting

Call `GET /collection` first to see what sources are already indexed.
The API also performs content-level deduplication automatically (SHA-256 of
the text) — a second ingest of identical content returns `status: skipped`
without re-embedding.

### Poll job status for directory ingestion

`POST /ingest` (directory ingestion) is asynchronous. After posting, poll
`GET /ingest/status/{job_id}` until `status` is `completed` or `failed`.
*(Available after P1-C is merged.)*

---

## Query Workflow

| Scenario | Endpoint / tool |
|---|---|
| Direct answer synthesis | `POST /query` |
| Multi-step reasoning over raw chunks | `POST /search` |
| Real-time / streaming output | `POST /query/stream` |

Use `/search` when you need to inspect individual document chunks yourself
before synthesising an answer. Use `/query` when you want Axon to produce
the final answer.

---

## MCP Tool Names (agent mode)

When using Copilot in **agent mode** with the Axon MCP server, use these
tool names (they differ deliberately from the OpenAI-format `tools.py` names):

| MCP tool | Does |
|---|---|
| `ingest_text` | Single document ingest |
| `ingest_texts` | Batch ingest (prefer this) |
| `ingest_url` | Fetch URL and ingest |
| `ingest_path` | Ingest a local file/directory |
| `get_job_status` | Poll async ingest job |
| `search_knowledge` | Raw chunk retrieval |
| `query_knowledge` | Synthesised answer |
| `list_knowledge` | List indexed sources |
| `switch_project` | Change active project |
| `delete_documents` | Remove by ID |
| `list_projects` | List all project namespaces |
| `get_stale_docs` | Find docs not re-ingested in N days |

---

## Dos and Don'ts

- **Do** use `ingest_texts` (batch) for multiple documents — never call
  `ingest_text` in a loop.
- **Do** set `metadata.source` on every document.
- **Do** call `list_knowledge` before a large ingest to check what's already
  indexed.
- **Do** use hierarchical projects (`docs/react`, `docs/python`) when you want
  to search multiple topics together — switching to the parent searches all
  descendants automatically.
- **Do** call `list_projects` to discover available namespaces before switching.
- **Don't** ingest private network URLs — they are blocked server-side.
- **Don't** call `POST /project/switch` from concurrent request handlers —
  use the `project` parameter on ingest endpoints instead (available after
  P1-D is merged).
