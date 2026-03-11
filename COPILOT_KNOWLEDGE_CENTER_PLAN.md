# Copilot Knowledge Center — Implementation Plan

A plan to equip GitHub Copilot agent to ingest knowledge into the Axon
via MCP tools, closing all identified API gaps and wiring in the workflow
instruction layer.

---

## Background & Design Rationale

### The Core Idea

GitHub Copilot charges per request, not per token — meaning a single request
can read a 500-page docs site at the same cost as renaming a variable. The
strategy is to use Copilot for the expensive "reading" step, then hand off to
the Axon for all embedding, storage, and retrieval — which stays 100%
local and free at query time.

```
Copilot Agent Mode (VS Code)
        │
        │  reads / fetches / cleans content
        │  calls MCP tools (stdio)
        ▼
  MCP Server  ──── new: thin wrapper ~150 lines
  (stdio process)
        │
        │  HTTP calls
        ▼
  Axon API  ── already exists at localhost:8000
  (FastAPI)
        │
        ├── Vector store (default: ChromaDB; also Qdrant, LanceDB) ← free, local
        └── BM25 index                ← free, local
```

### What's Already There

| Asset | State |
|-------|-------|
| `api.py` | 309 lines — `/health`, `/add_text`, `/ingest`, `/query`, `/query/stream`, `/search`, `/collection`, `/delete`, `/project/switch` |
| `tools.py` | OpenAI-format tool schemas — very close to MCP, wrong transport |
| `loaders.py` | File-only: `.txt .tsv .json .csv .md .html .htm .docx .pdf .png .bmp .tif .tiff .pgm` |
| `requirements.txt` | `httpx` already present — URL fetching costs zero new deps |
| `mcp` package | **Not installed, not in deps** |
| `.github/copilot-instructions.md` | **Does not exist** |
| `.vscode/mcp.json` | **Does not exist** |
| `src/axon/mcp_server.py` | **Does not exist** |
| Test suite | 11 test files covering api, loaders, main, projects, retrievers |

---

## Identified Gaps

### Read Side (Ingestion)

**Gap 1 — No URL/Web Loader**
`loaders.py` only handles local file extensions. Copilot has to manually copy
page content and paste it. A `URLLoader` using `httpx` (already in deps) would
let Copilot POST a URL directly.

**Gap 2 — No Batch Ingestion Endpoint (Critical Throughput Blocker)**
`/add_text` handles one document per call. 10 pages = 10 separate API calls,
10 separate embed+store operations. Need `POST /add_texts` accepting a list.

**Gap 3 — Fire-and-Forget Ingestion, No Status Tracking**
`/ingest` uses `BackgroundTasks` and returns `"status": "processing"` with no
job ID. Copilot cannot tell if ingestion succeeded, failed, or is still running.
Need a job ID and `GET /ingest/status/{job_id}`.

**Gap 4 — No Source-Level Tracking / Skip Signaling at API**
The core ingest pipeline already does content-based dedup (hashing `doc["text"]`
when `dedup_on_ingest` is enabled), so duplicate vectors are usually avoided at
the chunk level. What is missing is source-level tracking and clear UX: Copilot
cannot tell if a re-ingested URL was newly stored or quickly recognised and
skipped. Need API fields (e.g., stable `source_id` / URL) and explicit statuses
like `ingested` vs `skipped_duplicate` for observability — avoiding unnecessary
embedding work and giving Copilot actionable feedback.

**Gap 5 — No GitHub-Specific Loader**
No native support for GitHub URLs, repo file trees, issues, or PRs. Copilot
can reference these natively but the system can't consume them directly.

**Gap 6 — No Per-Call Project Targeting**
`/add_text` has no `project` parameter. Copilot must call `/project/switch`
first — a race condition under concurrent ingestion. Need `project` field in
ingest requests.

### Write Side (Retrieval)

**Gap 7 — No Knowledge Freshness / TTL**
No timestamp-based eviction or staleness tracking. Stale content stays forever
with no way to identify candidates for refresh.

**Gap 8 — No Cross-Project Search**
`/search` and `/query` operate on the currently active project only. A
knowledge center should be able to search across all namespaces.

### Integration Layer

**Gap 9 — No MCP Server**
`tools.py` defines tools in OpenAI function-calling format but there is no MCP
transport wrapper. Copilot agent mode requires MCP stdio for tool calling.

**Gap 10 — No Copilot Workflow Instructions**
`.github/copilot-instructions.md` does not exist. Without it, Copilot has no
persistent context about how to use the tools correctly across sessions.

---

## Phase 0 — Preparation (Before Any Code)

Hard blockers. Validate all of these before writing a single line.

### P0-A: VS Code Version Check

MCP tool calling in Copilot agent mode requires **VS Code ≥ 1.99**.
Below that, `.vscode/mcp.json` is silently ignored.

```bash
code --version   # must show 1.99.x or higher
```

If below 1.99: update VS Code first. Everything else is blocked by this.

### P0-B: GitHub Copilot Extension Version

MCP tool calling requires **Copilot Chat extension ≥ 0.22** (the version that
introduced agent mode tool support). Check Extensions panel → GitHub Copilot
Chat → version number.

### P0-C: MCP Python SDK Availability

```bash
pip install mcp
python -c "import mcp; print(mcp.__version__)"
```

If this fails, Phase 2 is fully blocked. `mcp` is not in `requirements.txt`
yet — it will be added as part of Phase 2.

### P0-D: Confirm Axon API Boots

```bash
axon-api &
curl http://localhost:8000/health
```

The MCP server proxies every call to this process. If the API doesn't start
cleanly, nothing in Phase 2 will work.

### P0-E: Confirm `RAG_INGEST_BASE` Is Set

`/ingest` path validation uses the `RAG_INGEST_BASE` env var (defaults to `.`).
Decide and document what this should be set to in the dev environment before
building the URL loader. The URL loader design in Phase 1 fetches and returns
documents directly in memory (no temp file). If a future implementation variant
writes to disk instead, ensure its temp directory lives under `RAG_INGEST_BASE`.

---

## Phase 1 — API Gaps (Backend)

Estimated effort: **5–6 hours**

Build and test these before touching MCP. The MCP server is only as good as
the endpoints it calls.

### P1-A: `POST /add_texts` — Batch Ingestion _(1 hr)_

**Files:** `api.py`, `tools.py`

Add a new Pydantic model `BatchTextIngestRequest` and a `/add_texts` endpoint
that calls `brain.ingest([...])` with a list. One HTTP call, one embedding
batch, N documents.

**ID assignment:** Each item has the shape `{"text": str, "doc_id": Optional[str], "metadata": Optional[dict]}`.
If `doc_id` is omitted, the server generates a UUID4. The response returns a
per-item list: `[{"id": "<assigned-id>", "status": "created"|"updated", "error": Optional[str]}]`.

**Pre-test:** Send `[{"text": "doc1"}, {"text": "doc2"}]` via curl. Confirm the
response contains 2 distinct `id` values and both appear in `/collection`.

**Unit test:** `tests/test_api.py::test_add_texts_batch` — post 3 docs, assert
the response contains 3 unique `id` values and `/collection` count increases by 3.

---

### P1-B: URL Loader + `POST /ingest_url` _(1.5 hrs)_

**Files:** `loaders.py`, `api.py`, `tools.py`

Add a `URLLoader` class to `loaders.py`:
1. Accept only `http`/`https` URLs; reject all other schemes.
2. Parse hostname and reject requests to private, loopback, link-local, and
   cloud metadata ranges (`127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`,
   `192.168.0.0/16`, `169.254.169.254`) — prevents SSRF.
3. Fetch URL with `httpx.get()` (already in deps) with a timeout and a
   capped redirect limit (e.g., `max_redirects=5`).
4. Strip HTML with stdlib `html.parser` (zero new deps).
5. Return `{"id", "text", "metadata"}` with `source` = URL.

Add `POST /ingest_url` in `api.py` calling this loader and feeding into
`brain.ingest()`. The endpoint must enforce the same scheme and IP-range checks
server-side and should be gated behind API key auth (already enforced by
`RAG_API_KEY`) to prevent untrusted callers from triggering arbitrary
server-side requests.

**Pre-test:** Fetch a plain-text URL (e.g., `https://example.com`). Verify text
is readable, not raw HTML. Also verify that `http://127.0.0.1` and
`http://169.254.169.254` return a 400 error, not a fetch.

**Unit test:** `tests/test_loaders.py::test_url_loader` with `httpx` mocked —
no real network calls in CI. Include cases that assert blocked hosts/schemes
raise errors before any network call is made.

**Edge cases to handle:** redirect following (capped), non-200 status, non-text
content type (reject binary), max content size (reuse `_MAX_FILE_BYTES` pattern
from existing loaders), and SSRF mitigations (scheme allowlist, IP range check).

---

### P1-C: Job Status Tracking for `/ingest` _(1.5 hrs)_

**File:** `api.py`

Right now `/ingest` returns `{"status": "processing"}` with no way to poll.
Changes:
- Module-level `_jobs: Dict[str, dict]` dict
- `/ingest` generates `job_id = uuid4().hex[:12]`, stores
  `{status, started_at, path, error}`, returns `job_id` immediately
- Background task updates `_jobs[job_id]` → `completed` or `failed` with
  timestamp
- New `GET /ingest/status/{job_id}` returns the job dict

**Pre-test:** POST to `/ingest`, grab `job_id`, poll
`/ingest/status/{job_id}` until `completed`.

**Unit test:** `tests/test_api.py::test_ingest_job_status` — mock
`brain.load_directory`, assert status transitions `processing` → `completed`.

**Note:** In-memory only — jobs are lost on server restart and the `_jobs`
dict is not shared across multiple API workers (e.g., `uvicorn --workers > 1`),
so job status will appear inconsistent in multi-worker deployments. For v1,
assume a single API worker and document both limitations explicitly in the
deployment docs.

**Cleanup / TTL:** Without a retention policy `_jobs` grows without bound. Apply
a simple max-size cap (e.g., 1 000 entries) or a TTL sweep: on every write,
evict entries older than N minutes (e.g., 60 min). Document the chosen policy
in the deployment docs alongside the single-worker requirement.

---

### P1-D: Per-Call Project Targeting _(0.5 hrs)_

**File:** `api.py`

Add optional `project: Optional[str]` to `TextIngestRequest` and
`BatchTextIngestRequest`. If provided, route the request to the correct
project-scoped brain instance — do **not** call `brain.switch_project()`
from inside a request handler.

**Concurrency requirement:** `brain.switch_project()` mutates shared global
state (vector store, BM25 path, hash store, cache) and MUST NOT be called
directly from request handlers. Implement one of:
- A `Dict[str, AxonBrain]` mapping project name → dedicated brain
  instance, selecting the right one per request (preferred).
- A process-wide `asyncio.Lock` guarding any global-state switch + ingest/query
  pair (simpler but serialises all ingestion).

**Pre-test:** Ingest a doc with `project: "test-project"` and a doc with
`project: "other-project"` concurrently. Confirm each appears only in its own
project's `/collection` and neither leaks into the other.

---

### P1-E: Source-Level Deduplication _(0.5 hrs)_

**File:** `api.py`

Add a `_source_hashes: Dict[str, Dict[str, dict]]` dict keyed by **project
name**, so deduplication is scoped per-project and a document ingested into one
project does not block the same source from being ingested into a different
project. This mirrors the existing `.content_hashes` mechanism that
`switch_project()` reloads per project.

Each entry stores `{content_hash, doc_id, last_ingested_at}`. The hash is
always over the **fetched, normalised content** — never the URL string alone.
Hashing the URL would incorrectly skip re-ingestion when a page's content
changes; hashing the content enables proper change detection. Before ingesting
in `/add_text`, `/add_texts`, `/ingest_url`: compute `sha256(content)`, check
`_source_hashes[project]`, and return:

```json
{"status": "skipped", "reason": "already_ingested", "doc_id": "<existing>"}
```

without calling `brain.ingest()`.

**Pre-test:** Call `/ingest_url` twice with the same URL. Second call must
return `skipped`.

**Unit test:** `tests/test_api.py::test_add_text_dedup`.

---

## Phase 2 — MCP Server (Integration Layer)

Estimated effort: **2.5–3 hours**

Build only after all Phase 1 endpoints pass their tests.

### P2-A: `src/axon/mcp_server.py` _(2 hrs)_

A stdio process using the `mcp` Python SDK. Tool definitions adapted from the
existing `tools.py` (already almost MCP-compatible). Each tool handler calls
the local REST API via `httpx`.

Tools to expose:

| Tool | Maps to |
|------|---------|
| `ingest_text(text, metadata?, project?)` | `POST /add_text` |
| `ingest_texts(docs[], project?)` | `POST /add_texts` ← new |
| `ingest_url(url, metadata?, project?)` | `POST /ingest_url` ← new |
| `ingest_path(path)` | `POST /ingest` |
| `get_job_status(job_id)` | `GET /ingest/status/{id}` ← new |
| `search_knowledge(query, top_k?, filters?)` | `POST /search` |
| `query_knowledge(query, filters?)` | `POST /query` |
| `list_knowledge()` | `GET /collection` |
| `switch_project(project_name)` | `POST /project/switch` |
| `delete_documents(doc_ids[])` | `POST /delete` |

**Naming note:** MCP tool names above (e.g., `search_knowledge`,
`query_knowledge`) differ deliberately from the existing `tools.py` names
(`search_documents`, `query_knowledge_base`). The MCP server will define its
own, more concise names optimised for agent-mode ergonomics. The existing
`tools.py` schemas (OpenAI format, used by other LLM callers) are kept
unchanged. Do not treat the two sets of names as interchangeable.

**API key auth:** `api.py` enforces `X-API-Key` on all endpoints except
`/health` when `RAG_API_KEY` is set. The MCP server must read `RAG_API_KEY`
from its environment and include it as an `X-API-Key` header in every `httpx`
call. Omitting this header will cause 401 errors in any secured deployment.

**Pre-test:** Run `python -m axon.mcp_server` — confirm it starts without
error and responds to an MCP `tools/list` request.

---

### P2-B: `.vscode/mcp.json` _(15 min)_

```json
{
  "servers": {
    "axon": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "axon.mcp_server"],
      "env": {
        "RAG_API_BASE": "http://localhost:8000",
        "RAG_API_KEY": ""
      }
    }
  }
}
```

**Pre-test:** Reload VS Code window. Open Copilot Chat → Agent mode → Tools
icon. Confirm `axon` tools appear in the tool list.

---

### P2-C: Dependency Updates _(15 min)_

**Files:** `requirements.txt`, `setup.py`, `pyproject.toml`

- **Reconcile first:** `setup.py` and `pyproject.toml` are already divergent —
  `pyproject.toml` declares FastAPI/Uvicorn while `setup.py` does not. Before
  adding `mcp`, decide which file is the single source of truth and bring the
  other into sync. Adding `mcp` to only one file will cause broken installs
  depending on the build path used.
- Add `mcp>=1.0.0` to `requirements.txt`, to `install_requires` in `setup.py`,
  **and** to the appropriate dependency table in `pyproject.toml`.
- Add console script `axon-mcp = axon.mcp_server:main` to
  `entry_points["console_scripts"]` in `setup.py` and mirror the same script
  in `pyproject.toml` so both build paths expose the CLI.

---

## Phase 3 — Copilot Workflow Layer (Instructions)

Estimated effort: **1 hour**. No code. Can start Day 1 in parallel with Phase 1.

### P3-A: `.github/copilot-instructions.md` _(45 min)_

This file does not currently exist. It provides persistent context to Copilot
across all sessions without requiring the user to re-explain the workflow.

Key sections to include:
- **Knowledge Center overview**: what projects exist, what each namespace is for
- **Ingestion workflow**: always use `ingest_texts` (batch) over `ingest_text`
  (single), always set `metadata.source`, always poll `get_job_status` after
  async operations, always call `list_knowledge` first to check for duplicates
- **Query workflow**: use `search_knowledge` for multi-step reasoning,
  `query_knowledge` for direct answers
- **Project naming conventions**: standard namespaces and what goes in each

### P3-B: Update `.github/instructions/developer.instructions.md` _(15 min)_

Add a section "Adding a new MCP tool" that mirrors the existing "Adding a new
loader" and "API endpoints" patterns so the developer Copilot role stays
consistent as the codebase grows.

---

## Phase 4 — Testing

Estimated effort: **2–3 hours**

### Unit Tests (Automated, Run in CI)

| Test | File | Covers |
|------|------|--------|
| `test_add_texts_batch` | `test_api.py` | P1-A |
| `test_url_loader_mock` | `test_loaders.py` | P1-B (httpx mocked) |
| `test_ingest_job_status` | `test_api.py` | P1-C |
| `test_per_call_project` | `test_api.py` | P1-D |
| `test_add_text_dedup` | `test_api.py` | P1-E |
| `test_mcp_tools_list` | `test_tools.py` | P2-A (subprocess) |

### Manual Integration Tests (Copilot-in-the-Loop)

These require a human with VS Code open in Copilot agent mode:

1. Ask: `"Ingest the content at [URL] into the react-docs project"` — confirm
   `ingest_url` tool is called with correct params
2. Ask: `"What do we know about React hooks?"` — confirm `search_knowledge` is
   called and returns relevant results
3. Ask: `"Ingest these three summaries: [A, B, C]"` — confirm `ingest_texts`
   is called as one batch, not three separate calls
4. Ingest the same URL twice — confirm second call returns `skipped`
5. Ingest a doc, poll `get_job_status` — confirm status transitions to
   `completed`

### Regression Check

After all phases, run the existing test suite to confirm nothing broke:

```bash
pytest tests/ -m "not slow and not integration" --cov=axon
```

---

## Summary: Effort & Build Order

| Phase | Task | Effort | Notes |
|-------|------|--------|-------|
| P0 | Prerequisites | 1–2 hrs | Must pass before any code |
| P1-A | Batch endpoint | 1 hr | Lowest risk, highest throughput gain |
| P1-B | URL loader | 1.5 hr | Needs httpx mock in tests |
| P1-C | Job status | 1.5 hr | Makes pipeline observable |
| P1-D | Project param | 0.5 hr | Simple, enables parallel namespaces |
| P1-E | Source dedup | 0.5 hr | Simple hash map check |
| P2-A | MCP server | 2 hr | Requires Phase 1 endpoints to be stable |
| P2-B | mcp.json | 15 min | Requires VS Code ≥ 1.99 (P0-A) |
| P2-C | Deps update | 15 min | Before P2-A to avoid import errors |
| P3-A | copilot-instructions | 45 min | Can run in parallel with Phase 1 |
| P3-B | Dev instructions update | 15 min | Can run in parallel with Phase 1 |
| P4 | Testing | 2–3 hr | After each phase completes |
| **Total** | | **~11–13 hrs** | |

### Critical Dependency Chain

```
P0-A (VS Code ≥ 1.99)     ──→  P2-B (mcp.json visible to Copilot)
P0-C (mcp SDK installs)   ──→  P2-A (mcp_server.py can import mcp)
P1-A + P1-B + P1-C        ──→  P2-A (MCP server needs stable endpoints)
P2-A + P2-B               ──→  Phase 4 manual integration tests
P3-A                       ──→  independent, start Day 1
```

### Phases That Can Run in Parallel

- **P3-A + P3-B** can be written on Day 1 alongside P0 checks
- **P1-A through P1-E** are independent of each other within Phase 1
- **P2-C** (dependency update) should be done before P2-A to avoid import
  errors during development
