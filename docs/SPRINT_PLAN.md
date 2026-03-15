# Sprint Plan — Axon Integration Alternatives

**Branch:** `claude/axon-integration-alternatives-EnT0C`
**Date:** 2026-03-13

## Overview

This sprint delivers 9 improvements to the Axon RAG engine, covering configuration
consistency, performance tuning, content-aware chunking, and smart ingestion.

---

## Items

### [x] ITEM 1 — Embedding Config Consistency (Priority: High)

- Changed `AxonConfig` defaults from `fastembed/BAAI/bge-large-en-v1.5` to
  `sentence_transformers/all-MiniLM-L6-v2` to match `config.yaml`
- Added `_KNOWN_DIMS` dict for accurate embedding dimension lookup
- Replaced hardcoded Ollama dimension (768) with `_KNOWN_DIMS` lookup
- Replaced partial fastembed dimension logic with `_KNOWN_DIMS` lookup

### [x] ITEM 2 — Health 503 Fix + Max Workers Config (Priority: High)

- Fixed `/health` endpoint to return 503 when brain is not initialized
- Added `max_workers: int = 8` to `AxonConfig` dataclass
- Replaced hardcoded `ThreadPoolExecutor(max_workers=8)` in `__init__` and
  `switch_project` with `self.config.max_workers`
- Added `max_workers: 8` to `config.yaml`

### [x] ITEM 3 — RRF Config Flag (Priority: Medium)

- Added `hybrid_mode: Literal["weighted", "rrf"] = "weighted"` to `AxonConfig`
- Updated hybrid retrieval section to branch on `cfg.hybrid_mode`
- `"rrf"` invokes `reciprocal_rank_fusion`; `"weighted"` (default) uses
  `weighted_score_fusion` as before
- Added `hybrid_mode: weighted` to `config.yaml` rag section

### [x] ITEM 4 — Content-Based Dataset Type Detection (Priority: Medium)

- Added `dataset_type: Literal[...]` config field (default: `"auto"`)
- Added class-level regex patterns: `_CODE_EXTENSIONS`, `_CODE_LINE_PATTERNS`,
  `_PAPER_SIGNALS`, `_DOC_SIGNALS`
- Implemented `_detect_dataset_type(doc)` heuristic with 7-priority detection chain:
  file extension → JSON discussion → tabular → code ratio → paper signals →
  doc signals → extension fallback
- Added `_get_splitter_for_type(dataset_type, has_code)` method

### [x] ITEM 5 — Type-Specific Chunking (Priority: Medium)

- Updated `ingest()` to call `_detect_dataset_type` per document
- Stores `dataset_type` and `has_code` in chunk metadata
- Uses `_get_splitter_for_type()` to select the right splitter per document
- Falls back to the default configured splitter for standard doc types

### [x] ITEM 6 — Continuous Ingestion / Smart Re-Ingest (Priority: Medium)

- Added `smart_ingest: bool = False` to `AxonConfig`
- Added `_doc_versions` dict and `_doc_versions_path` (`.doc_versions.json`) to `AxonBrain.__init__`
- Added `_load_doc_versions()`, `_save_doc_versions()`, `get_doc_versions()` methods
- Added `GET /tracked-docs` endpoint — lists all tracked document sources
- Added `POST /ingest/refresh` endpoint — checks file hashes and reports changed files

### [x] ITEM 7 — Qdrant Remote Mode (Priority: Low)

- Added `qdrant_url: str = ""` and `qdrant_api_key: str = ""` to `AxonConfig`
- Added parsing for both fields in `AxonConfig.load()`
- Updated `OpenVectorStore._init_store()` to use remote URL when `qdrant_url` is set
- Added commented-out remote Qdrant config block to `config.yaml`

### [x] ITEM 8 — Public/Private dbstore Layout (Priority: Low)

- Added `projects_base` alias parsing in `AxonConfig.load()` — maps to `projects_root`
- Added commented-out `projects_base` example to `config.yaml`

### [x] ITEM 9 — Sprint Plan Document (Priority: Low)

- Created this file: `docs/SPRINT_PLAN.md`

---

## Additional — New Tests

- [x] `test_detect_dataset_type` — code files, papers, tables, discussion JSON
- [x] `test_health_503_when_brain_none` — verify 503 response
- [x] `test_health_200_when_brain_initialized` — verify 200 response
- [x] `test_hybrid_mode_rrf` — verify RRF fusion path is called
- [x] `test_known_dims_ollama` — Ollama dimension lookup via `_KNOWN_DIMS`
- [x] `test_get_doc_versions` — doc version storage and retrieval
- [x] `test_load_doc_versions` — persistence roundtrip

---

---

## ITEM 10 — AxonStore Multi-User Shared Storage (Priority: High)

- **`src/axon/shares.py`** (new): HMAC-SHA256 share key system — `generate_share_key`, `redeem_share_key`, `revoke_share_key`, `list_shares`, `validate_received_shares`
- **`src/axon/projects.py`**: Added `ensure_user_namespace`, `_ensure_single_project_at`, `_make_share_link`, `_remove_share_link`, `list_share_mounts`; `_MAX_DEPTH = 5`; reserved names: `{"sharemount", "_default", ".shares"}`; `list_descendants` rewritten as recursive DFS with cycle detection
- **`src/axon/main.py`**: `AxonConfig.axon_store_base`, `axon_store_mode`; `__post_init__` reads `AXON_STORE_BASE` env var and derives project/vector/bm25 paths; `AxonBrain.__init__` calls `ensure_user_namespace` when in store mode
- **`src/axon/api.py`**: New endpoints: `POST /store/init`, `GET /store/whoami`, `POST /share/generate`, `POST /share/redeem`, `POST /share/revoke`, `GET /share/list`; `GET /projects` validates received shares; `_get_user_dir()` helper
- **`src/axon/mcp_server.py`**: Added `share_project`, `redeem_share`, `list_shares` MCP tools
- **`integrations/vscode-axon/src/extension.ts`**: `ensureServerRunning` sets `RAG_INGEST_BASE=/` and passes `AXON_STORE_BASE`; new commands: `axon.initStore`, `axon.shareProject`, `axon.redeemShare`, `axon.revokeShare`, `axon.listShares`, `axon.ingestWorkspace`, `axon.ingestFolder`; new LM tool `AxonListSharesTool`
- **`integrations/vscode-axon/package.json`**: New settings `axon.storeBase`, `axon.ingestBase`; all new commands and LM tool declared
- **`tests/test_shares.py`** (new): 17 unit tests; Linux symlink tests skip on non-Linux

## ITEM 11 — Test Quality & Coverage (Priority: Medium)

- Registered `demo`, `perf`, `stress` marks in `pyproject.toml`
- Excluded `webapp.py` from coverage (Streamlit UI, untestable without browser)
- Fixed `aload()` path enrichment bug: async directory loader now applies `[File Path: ...]` breadcrumbs (parity with sync `load()`)
- Added API tests for `/store/whoami`, `/share/list`, `/share/generate`, `/share/revoke` (503, 404, 200 cases)
- Added project tests for `ensure_user_namespace`, reserved names, `list_share_mounts`
- Test suite: 504 passing, 6 skipped (Linux symlink), 0 failing

## ITEM 12 — Temperature Control (CLI / REPL / API) (Priority: Low)

**Date:** 2026-03-15

- Added `--temperature <float>` CLI argument to `axon` command
- Added `/llm temperature <0.0–2.0>` REPL command (separate from `/rag`)
- Added `/llm` tab completion to `_PTCompleter`
- Added `temperature: float | None` field to `QueryRequest` in `api.py`
- API maps `temperature` to `overrides["llm_temperature"]` for per-request override
- `/` commands list in REPL and `/help` output sorted alphabetically
- Tests: `TestTemperatureCLI`, `TestReplLlmCommand`, `TestSlashCommandOrder` in `test_main.py`; `TestQueryTemperature` in `test_api.py`

## ITEM 13 — VS Code Extension: Python Auto-Detection (Priority: Medium)

**Date:** 2026-03-15

- Added `_write_python_discovery()` to `main.py` — writes interpreter path to `~/.axon/.python_path` on first `axon` CLI run
- Added `discoverPythonPath()` to `extension.ts` — priority chain: explicit `axon.pythonPath` setting → `~/.axon/.python_path` → pipx venv → workspace venv → system Python with warning notification
- Changed `axon.pythonPath` default from `"python"` to `""` in `package.json`
- Supports pip, pipx, venv, virtualenv install scenarios without user configuration

## ITEM 14 — Version Alignment (Priority: Low)

**Date:** 2026-03-15

- Python package: `pyproject.toml` `version` set to `1.0.0` (was `2.0.0`)
- VS Code extension: `package.json` `version` set to `1.0.0` (was `0.1.0`)
- Repacked as `axon-copilot-1.0.0.vsix`
- Updated `QUICKREF.md` version footer

## ITEM 15 — Bug Fixes (Priority: High)

**Date:** 2026-03-15

### Bug 2: axon_ingestText 500 [Errno 22] Invalid argument

- **Root cause:** `BM25Retriever.save()` only caught `PermissionError`; Windows `os.replace()` can raise `OSError` (errno 22 / WinError 87 ERROR_INVALID_PARAMETER) on some file systems
- **Fix:** Changed `except PermissionError:` to `except OSError:` in `BM25Retriever.save()` — the shutil.copy2 fallback now activates for any OS-level replace failure
- **Tests:** `TestBM25SaveOsError.test_save_falls_back_on_oserror`, `test_save_falls_back_on_permission_error` in `test_retrievers.py`

### Bug 1: Path ingestion accepted but content not retrievable

- **Root cause:** `axon_ingestPath` returns a `job_id` immediately (background job) but the VS Code extension had no tool to poll job status; Copilot proceeded to search before ingestion completed
- **Fix:** Added `axon_getIngestStatus` language model tool to `extension.ts` and `package.json` — polls `GET /ingest/status/{job_id}` and returns `processing` / `completed` / `failed`; updated `axon_ingestPath` model description to require polling before searching
- **Tests:** `TestIngestStatusEndpoint` (4 tests) in `test_api.py`; `TestAddTextBug500` (2 tests) in `test_api.py`

## Test Status

Run with: `python -m pytest tests/ -v --tb=short`

Current counts (2026-03-15): **~526 passed, 6 skipped, 0 failed**
