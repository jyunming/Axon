# Copilot Knowledge Center — Implementation Plan

A plan to equip GitHub Copilot agent to ingest knowledge into the RAG Brain
via MCP tools, closing all identified API gaps and wiring in the workflow
instruction layer.

---

## Background & Design Rationale

### The Core Idea

GitHub Copilot charges per request, not per token — meaning a single request
can read a 500-page docs site at the same cost as renaming a variable. The
strategy is to use Copilot for the expensive "reading" step, then hand off to
the RAG Brain for all embedding, storage, and retrieval — which stays 100%
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
  RAG Brain API  ── already exists at localhost:8000
  (FastAPI)
        │
        ├── ChromaDB (vector store)   ← free, local
        └── BM25 index                ← free, local
```

### What's Already There

| Asset | State |
|-------|-------|
| `api.py` | 309 lines — `/add_text`, `/ingest`, `/query`, `/search`, `/collection` |
| `tools.py` | OpenAI-format tool schemas — very close to MCP, wrong transport |
| `loaders.py` | File-only: `.txt .tsv .json .csv .html .docx .pdf .png` |
| `requirements.txt` | `httpx` already present — URL fetching costs zero new deps |
| `mcp` package | **Not installed, not in deps** |
| `.github/copilot-instructions.md` | **Does not exist** |
| `.vscode/mcp.json` | **Does not exist** |
| `src/rag_brain/mcp_server.py` | **Does not exist** |
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

**Gap 4 — No Source-Level Deduplication at API**
Chunk-level dedup exists inside the pipeline, but if Copilot re-ingests the
same URL twice, you get duplicate vectors degrading retrieval quality. Need a
source fingerprint registry (`sha256(content)`) checked before embedding.

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

### P0-D: Confirm RAG Brain API Boots

```bash
rag-brain-api &
curl http://localhost:8000/health
```

The MCP server proxies every call to this process. If the API doesn't start
cleanly, nothing in Phase 2 will work.

### P0-E: Confirm `RAG_INGEST_BASE` Is Set

`/ingest` path validation uses the `RAG_INGEST_BASE` env var (defaults to `.`).
Decide and document what this should be set to in the dev environment before
building the URL loader, which writes fetched content to a temp path.

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

**Pre-test:** Send `[{"text": "doc1"}, {"text": "doc2"}]` via curl. Confirm
both appear in `/collection`.

**Unit test:** `tests/test_api.py::test_add_texts_batch` — post 3 docs, assert
`/collection` count increases by 3.

---

### P1-B: URL Loader + `POST /ingest_url` _(1.5 hrs)_

**Files:** `loaders.py`, `api.py`, `tools.py`

Add a `URLLoader` class to `loaders.py`:
1. Fetch URL with `httpx.get()` (already in deps)
2. Strip HTML with stdlib `html.parser` (zero new deps)
3. Return `{"id", "text", "metadata"}` with `source` = URL

Add `POST /ingest_url` in `api.py` calling this loader and feeding into
`brain.ingest()`.

**Pre-test:** Fetch a plain-text URL (e.g., `https://example.com`). Verify
text is readable, not raw HTML.

**Unit test:** `tests/test_loaders.py::test_url_loader` with `httpx` mocked —
no real network calls in CI.

**Edge cases to handle:** redirect following, non-200 status, non-text content
type (reject binary), max content size (reuse `_MAX_FILE_BYTES` pattern from
existing loaders).

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

**Note:** In-memory only — jobs are lost on server restart. Acceptable for v1;
document this limitation explicitly.

---

### P1-D: Per-Call Project Targeting _(0.5 hrs)_

**File:** `api.py`

Add optional `project: Optional[str]` to `TextIngestRequest` and
`BatchTextIngestRequest`. If provided, call `brain.switch_project(project)`
before ingesting.

**Pre-test:** Ingest a doc with `project: "test-project"`, switch to that
project via `/project/switch`, confirm the doc appears in `/collection`.

**Race condition warning:** `brain.switch_project()` mutates global state.
Under concurrent calls this is unsafe. For v1: document this. For v2: use a
per-request brain instance scoped to the project.

---

### P1-E: Source-Level Deduplication _(0.5 hrs)_

**File:** `api.py`

Add module-level `_source_hashes: Dict[str, str]` dict. Before ingesting in
`/add_text`, `/add_texts`, `/ingest_url`: compute `sha256(text or url)`, check
if already seen. Return:

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

### P2-A: `src/rag_brain/mcp_server.py` _(2 hrs)_

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

**Pre-test:** Run `python -m rag_brain.mcp_server` — confirm it starts without
error and responds to an MCP `tools/list` request.

---

### P2-B: `.vscode/mcp.json` _(15 min)_

```json
{
  "servers": {
    "rag-brain": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "rag_brain.mcp_server"],
      "env": {
        "RAG_API_BASE": "http://localhost:8000",
        "RAG_API_KEY": ""
      }
    }
  }
}
```

**Pre-test:** Reload VS Code window. Open Copilot Chat → Agent mode → Tools
icon. Confirm `rag-brain` tools appear in the tool list.

---

### P2-C: Dependency Updates _(15 min)_

**Files:** `requirements.txt`, `pyproject.toml`

- Add `mcp>=1.0.0` to `requirements.txt`
- Add `[mcp]` optional extra in `pyproject.toml`
- Add entry point `rag-brain-mcp = "rag_brain.mcp_server:main"` in
  `[project.scripts]`

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
pytest tests/ -m "not slow and not integration" --cov=rag_brain
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
