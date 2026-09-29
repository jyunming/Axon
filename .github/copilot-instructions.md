# Axon Knowledge Center — Copilot Instructions

These instructions give GitHub Copilot persistent context about the Axon
knowledge center so you can ingest and retrieve information correctly across
every session.

---

## What Axon Is

Axon is a local RAG (Retrieval-Augmented Generation) system exposed as a
REST API (`localhost:8420`) and as an MCP stdio server. It stores embeddings
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

Use `ingest_knowledge(path=...)` (or `POST /ingest`) on any local source directory — Axon automatically detects
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

## Agent Tool Names (0.5.0) — MCP: 18, VS Code LM tools: 20

The MCP server (`axon-mcp`, agent mode) and the VS Code extension's Copilot LM tools
expose the **same 18 tools with the same names and parameters**; VS Code adds
`show_graph` and `ingest_image`. Full parameter tables: `docs/REFERENCE.md` (section 9, MCP server).

| Tool | Does |
|---|---|
| `query_knowledge` | Retrieval + synthesised answer via the configured LLM |
| `search_knowledge` | Raw chunk retrieval — best for discovery; you synthesise the answer |
| `ingest_knowledge` | Ingest exactly one of `text`, `docs` (batch — prefer it for many docs), `url`, `path` (async, returns `job_id`), or `refresh: true` (re-ingest changed files, async) |
| `get_job_status` | Poll an async ingest job until `completed` |
| `list_knowledge` | List indexed sources and chunk counts for the active project |
| `delete_documents` | Remove chunks or whole documents by `doc_ids` |
| `list_projects` | List project namespaces and mounted shares |
| `switch_project` | Switch the active project (`project_name`) — global server state |
| `create_project` | Create a project (optional `graph_backend`: `graphrag` / `dynamic_graph` / `none`) |
| `get_config` | Read the active config (secrets masked); `validate: true` adds config.yaml findings |
| `set_config` | Set several config keys in one call (`settings`); `persist` defaults to false; all-or-nothing |
| `graph_retrieve` | Graph-backend retrieval, point-in-time capable (`point_in_time`) |
| `update_fact` | Assert or correct a graph fact (subject, relation, object) in a `dynamic_graph` / `federated` project |
| `share_project` | Generate a read-only share key for a grantee (optional `ttl_days`) |
| `redeem_share` | Mount a shared project using a share string |
| `list_shares` | List outgoing shares and incoming mounts |
| `revoke_share` | Soft-revoke a share (`project` required for sealed `ssk_` keys) — no key rotation |
| `extend_share` | Renew or clear a plaintext share's expiry |
| `show_graph` | *(VS Code only)* Open the Graph Panel for a query — answer, citations, 3D graph. No browser is opened |
| `ingest_image` | *(VS Code only)* Describe an image with a Copilot vision model and ingest the description |

A tool's `project` parameter is an **assertion**: a mismatch with the active project is a 409,
never a silent switch — call `switch_project` first.

Human-only (no agent tool — use the CLI, REPL or REST API): clearing a project
(`axon --clear --yes`, REPL `/clear`), deleting a project, stale-doc listing, sessions, store
init/status, the sealed store, seal/pack/unpack, mount refresh, hard share revocation with key
rotation, and graph status/finalize/conflicts.

Use `show_graph` when the user asks to "show the graph", "visualise", or "see connections" for a topic.

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

- **Do** use `ingest_knowledge(docs=[...])` (batch) for multiple documents — never
  call `ingest_knowledge(text=...)` in a loop.
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

---

## PyPI Publishing

- Package name on PyPI: `axon-rag` — install with `pip install axon-rag`
- Publishing is tag-triggered, NOT merge-triggered — merging to main does nothing to PyPI
- Full release sequence: `python scripts/bump_version.py X.Y.Z` (Cargo.toml — the single version source — plus the VS Code package.json, index.html, the rebuilt VSIX and Cargo.lock) → `python scripts/audit_packaging.py --expected-version X.Y.Z` → commit → PR → merge → `git tag vX.Y.Z && git push origin vX.Y.Z`
- NEVER bump version without a functional reason — packaging/doc/readme fixes alone do not justify a bump
- PyPI releases are immutable — description cannot be updated after upload; always verify `README.md` renders correctly before tagging
- `README.md` is the PyPI description (`readme = "README.md"` in `pyproject.toml`) — it uses absolute URLs (`https://raw.githubusercontent.com/jyunming/Axon/main/...` for images, `https://github.com/jyunming/Axon/blob/main/...` for docs) because PyPI cannot resolve relative paths or repo-relative images
- GitHub release notes use `generate_notes: true` in `release.yml` — do NOT replace with custom git-log scripts (they dump entire history on first tag and misattribute authorship)

---

## Packaging Rules

- `src/__init__.py` must NOT exist — `packages.find where=["src"]` would ship a bare `src` namespace package into user environments
- Files imported by `axon/__init__.py` (e.g. `llm.py`) cannot do `from axon import __version__` — circular import; use `importlib.metadata.version("axon-rag")` directly instead
- When moving a dep from base to an optional extra, guard its top-level import — e.g. `try: import streamlit as st; _AVAIL=True` / `except ImportError: _AVAIL=False`
- Windows mypy outputs backslash paths; `_is_allowed()` in `test_lint.py` normalises with `.replace("\\", "/")` before matching the allowlist

---

## Branch & PR Rules

- Always create a feature branch before touching any file — never commit directly to `main`
- **One branch at a time** — never split work across multiple branches; consolidate into one before committing
- **Before every push** — check for unresolved PR review comments first (`gh api repos/.../pulls/<n>/comments`); fix all findings before pushing to avoid wasting CI runs
- **NEVER push or create a PR without explicit user approval**
- Version bumps only for functional changes — doc/readme/HTML fixes do not justify a bump
- No tag = no PyPI release; merging to main alone does nothing to PyPI
