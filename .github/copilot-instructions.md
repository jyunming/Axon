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
| `default` | General-purpose knowledge, catch-all (canonical name; was `_default` in earlier versions) |
| (add project-specific rows here as namespaces are created) |

Switch the active project with `POST /project/switch` before ingesting if the
content belongs to a specific namespace.

### Hierarchical projects (up to 5 levels)

Projects support slash-separated nesting up to 5 levels deep:

```
research/               ← top-level parent
research/papers/        ← child
research/papers/2024    ← grandchild
research/papers/2024/q1 ← great-grandchild (max depth: 5 segments total)
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

### Merged read-only scopes

Three special project names give a unified read-only view across multiple stores:

| Scope | What it searches |
|---|---|
| `@projects` | All authoritative local projects |
| `@mounts` | All mounted (shared) projects |
| `@store` | `default` + `@projects` + `@mounts` |

Switch using `switch_project("@store")` etc. Ingest is blocked while a merged scope
is active — switch to a specific project before writing.

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

### Ingesting code repositories

Use `ingest_path` on any local source directory — Axon automatically detects
code files (`.py`, `.go`, `.rs`, `.ts`, `.js`, `.java`, `.cpp`, `.rb`, `.sh`,
`.pl`, `.jl`, and more) and routes them through the syntax-aware
`CodeAwareSplitter`. No path prefix is injected into code content.

```
POST /ingest
{"path": "/path/to/repo/src", "metadata": {"project": "my-service"}}
```

**For cross-file link awareness** (recommended for large repos), set in
`config.yaml`:

```yaml
rag:
  code_graph: true        # File + Symbol nodes with CONTAINS/IMPORTS edges
  code_graph_bridge: true # MENTIONED_IN edges linking prose docs to symbols
```

With `code_graph: true`, a query that retrieves a function will automatically
expand to include the file it belongs to, files it imports, and any callers
in the index — at zero extra LLM cost.

---

## Query Router

Axon automatically selects the cheapest retrieval strategy per query. The
default mode is `heuristic` (zero LLM calls — keyword + query-length signals):

| Route | When activated | Retrieval strategy |
|---|---|---|
| `factual` | Short lookup, specific fact (default fallback) | Hybrid BM25+dense only |
| `synthesis` | Summarise / compare / explain | + RAPTOR + parent-doc |
| `table_lookup` | Numbers, statistics, rows/columns | Dense only, tabular path |
| `entity_relation` | Relationship between X and Y | + GraphRAG (graph-light) |
| `corpus_exploration` | Main themes, key topics across all docs | + RAPTOR + multi-query |

Change the mode in `config.yaml`:

```yaml
rag:
  query_router: heuristic   # heuristic | llm | off
```

`llm` mode uses a single tight classification prompt — more accurate but adds
one LLM call per query. `off` falls back to the legacy `graph_rag_auto_route`
binary flag.

---

## Query Workflow

| Scenario | Endpoint / tool |
|---|---|
| **Copilot synthesises** (recommended) | `search_knowledge` / `POST /search` |
| Axon's local LLM synthesises | `query_knowledge` / `POST /query` |
| Real-time / streaming output | `POST /query/stream` |
| **Visualise + answer in VS Code panel** | `axon_showGraph` (VS Code extension LM tool) |

**Prefer `search_knowledge` in agent mode.** This returns raw ranked chunks
to Copilot, which then synthesises the answer using its own LLM. Axon handles
only retrieval (ChromaDB + BM25 fan-out); Copilot handles reasoning. This is
faster, avoids Ollama entirely, and scales naturally — Ollama can only generate
one response at a time while Copilot's LLM has no such limit.

Use `query_knowledge` / `POST /query` only when you want Axon to be
self-contained with its local Ollama model (e.g. offline, air-gapped).

**Context window tip:** `search_knowledge` returns `top_k` chunks. Keep
`top_k` between 5–8 for focused queries, 10–15 for broad/exploratory ones
to avoid burning the context window.

---

## VS Code Extension LM Tool Names (27 total)

When using the Axon VS Code extension in Copilot Chat (`@workspace` or inline), these LM tools are available:

### Search & Query

| Tool | Does |
|---|---|
| `axon_searchKnowledge` | Raw chunk retrieval — best for discovery; Copilot synthesises the answer |
| `axon_queryKnowledge` | Retrieval + answer via local LLM (requires Ollama) |
| `axon_showGraph` | Open Graph Panel — shows answer, citations, and 3D entity/code graph in VS Code |
| `axon_showGraphStatus` | Return GraphRAG community build status (in-progress flag + summary count) |
| `axon_getSettings` | Read current active configuration (top_k, rerank, hyde, model, etc.) |
| `axon_updateSettings` | Toggle RAG flags for the current session (hyde, rerank, graph_rag, etc.) |

### Ingestion

| Tool | Does |
|---|---|
| `axon_ingestText` | Ingest a raw text snippet |
| `axon_ingestUrl` | Fetch a web page and ingest it |
| `axon_ingestPath` | Ingest a local file or directory (async, returns `job_id`) |
| `axon_getIngestStatus` | Poll ingest job status until `completed` |
| `axon_refreshIngest` | Re-ingest tracked files whose content changed since last ingest |
| `axon_ingestImage` | Describe an image via Copilot vision model and ingest the description |

### Knowledge Base Management

| Tool | Does |
|---|---|
| `axon_getCollection` | List indexed sources and chunk counts for the active project |
| `axon_deleteDocuments` | Remove specific documents by ID |
| `axon_clearCollection` | Wipe all data from the active project (irreversible) |
| `axon_clearKnowledgeBase` | Alias for clearCollection — use only when a full re-ingest is intended |
| `axon_listStaleDocs` | Return docs not re-ingested within N days (default 7) |
| `axon_finalizeGraph` | Trigger community rebuild on the knowledge graph for global-mode GraphRAG |

### Project Management

| Tool | Does |
|---|---|
| `axon_listProjects` | List all project namespaces and mounted shares |
| `axon_switchProject` | Switch active project |
| `axon_createProject` | Create a new named project |
| `axon_deleteProject` | Delete a project and all its data permanently |

### AxonStore & Sharing

| Tool | Does |
|---|---|
| `axon_initStore` | Initialise AxonStore multi-user mode at a given base directory |
| `axon_shareProject` | Generate an HMAC share key for a grantee |
| `axon_redeemShare` | Mount a shared project using a share string (read-only) |
| `axon_revokeShare` | Revoke a previously issued share key by `key_id` |
| `axon_listShares` | List outgoing shares and incoming mounts with revocation status |

Use `axon_showGraph` when the user asks to "show the graph", "visualise", or "see connections" for a topic. The tool opens the split panel inside VS Code — **no browser is opened**.

---

## MCP Tool Names (27 total — agent mode)

When using Copilot in **agent mode** with the Axon MCP server, use these
tool names (they differ deliberately from the OpenAI-format `tools.py` names):

### Ingestion

| MCP tool | Does |
|---|---|
| `ingest_text` | Single document ingest |
| `ingest_texts` | Batch ingest (prefer this) |
| `ingest_url` | Fetch URL and ingest |
| `ingest_path` | Ingest a local file/directory (async, returns `job_id`) |
| `get_job_status` | Poll async ingest job |

### Search & Query

| MCP tool | Does |
|---|---|
| `search_knowledge` | Raw chunk retrieval (threshold fallback: retries without threshold if zero results) |
| `query_knowledge` | Synthesised answer via local LLM |

### Knowledge Base Management

| MCP tool | Does |
|---|---|
| `list_knowledge` | List indexed sources with chunk counts |
| `delete_documents` | Remove documents by `doc_ids` list |
| `clear_knowledge` | Wipe active project's vector store and BM25 index (irreversible) |
| `get_stale_docs` | Find docs not refreshed in N days (default 30) |
| `get_active_leases` | List active read/write leases held via AxonStore |

### Project Management

| MCP tool | Does |
|---|---|
| `list_projects` | List all project namespaces and mounted shares |
| `switch_project` | Change active project |
| `create_project` | Create a new named project |
| `delete_project` | Delete a project and all its data permanently |

### Settings

| MCP tool | Does |
|---|---|
| `get_current_settings` | Return active RAG flags, model config, and runtime settings |
| `update_settings` | Toggle RAG flags at runtime (session-scoped, not persisted) |

### Sessions

| MCP tool | Does |
|---|---|
| `list_sessions` | List saved conversation sessions (up to 20 most recent) |
| `get_session` | Retrieve a full session transcript by timestamp ID |

### GraphRAG

| MCP tool | Does |
|---|---|
| `graph_status` | Return entity count, edge count, community count, and rebuild state |
| `graph_finalize` | Trigger community detection rebuild for global-mode GraphRAG |
| `graph_data` | Return the full entity/relation graph as a JSON nodes+links payload |

### AxonStore & Sharing

| MCP tool | Does |
|---|---|
| `init_store` | Initialise AxonStore at a shared filesystem base path |
| `share_project` | Generate a read-only share key for a grantee |
| `redeem_share` | Mount a shared project using a share string (read-only) |
| `list_shares` | List outgoing and incoming shares, including revoked status |

---

## Local Model Configuration Fields

The following `config.yaml` fields under the `offline:` key control local model routing:

| Field | Type | Purpose |
|---|---|---|
| `offline.enabled` | bool | Full offline mode — locks HF network access; disables RAPTOR + GraphRAG |
| `offline.local_models_dir` | str | Legacy fallback root for both embedding and HF models |
| `offline.local_assets_only` | bool | Enforce local HF files **without** disabling RAPTOR or GraphRAG |
| `offline.embedding_models_dir` | str | Root directory for sentence-transformers / fastembed model files |
| `offline.hf_models_dir` | str | Root directory for GLiNER, REBEL, LLMLingua, and cross-encoder reranker |
| `offline.tokenizer_cache_dir` | str | tiktoken BPE encoding cache directory (maps to `TIKTOKEN_CACHE_DIR`) |

When `local_assets_only: true`, Axon runs a preflight model audit at startup and logs
`[local]`, `[hf_cache]`, `[remote]`, or `[MISSING]` for each active model. Startup is
aborted with a `RuntimeError` if any active model is `[remote]` or `[MISSING]`.

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
  use the `project` parameter on ingest endpoints instead.
- **Don't** set `graph_rag: true` for code corpora — code-to-code links use
  the code graph (`code_graph: true`); prose GraphRAG is disabled for code
  datasets by design.
- **Do** set `code_graph: true` when ingesting a source code repository if
  you want cross-file link traversal at query time.
- **Do** use `query_router: heuristic` (the default) for most deployments —
  it requires no LLM calls and selects the right retrieval strategy
  automatically.
