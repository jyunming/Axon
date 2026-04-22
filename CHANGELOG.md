# Changelog

## [Unreleased]

### ✨ New Features

- **REPL agent mode**: The interactive REPL now supports LLM-driven tool calls — Gemini and other providers can invoke Axon tools directly from a conversation turn.
- **REPL UI overhaul**: Full redesign with pinned input toolbar, conversation-area spinner, 4-char margins, separator lines, gray user-message background, icon and color polish, and code-fencing in responses.
- **`/debug` toggle**: New `/debug` slash command in the REPL suppresses or surfaces markdown-it and asyncio debug logs on demand.
- **TurboQuantDB (tqdb) as default vector store**: `tqdb` (b=4 + rerank) replaces ChromaDB as the default backend; moved from optional to required dependency.
- **Rust bridge**: Optional Rust-accelerated BM25 and retrieval helpers integrated via `axon_rust` extension module with benchmark suite.
- **AxonStore cross-user sharing**: `init_store`, `share_project`, `redeem_share`, `revoke_share`, and `list_shares` tools enable multi-user read-only knowledge sharing over a shared filesystem path.
- **Fleet audit — tools parity**: All MCP and API tools audited for parity; sharing and maintenance tools added to the fleet.
- **WebGUI redesign**: New dynamic-graph 3D visualisation panel, updated layout and component structure.
- **`--non-interactive` CLI flag**: Prevents the REPL from starting in scripts and CI environments.
- **`--dry-run` hardening**: Dry-run mode now disables all LLM transforms (HyDE, multi-query, step-back, compression, RAPTOR, GraphRAG) to guarantee zero LLM calls.
- **Lightweight CLI startup**: Metadata-only commands (`project-list`, `share-list`, `session-list`, etc.) skip full `AxonBrain` initialisation for faster response.

### 🐛 Bug Fixes

- **Agent mode REPL**: Fixed google_genai debug log spam, malformed JSON tool-call blobs, `create_project` failing to switch context, and corrupt binary hash store on write.
- **Gemini agent mode**: Corrected tool-call dispatch and response parsing for the Gemini provider.
- **`@file` expansion**: Fixed Windows path normalisation in `_expand_at_files`; corrected regex to avoid incorrectly expanding email addresses as file paths.
- **Ingest refresh scoped to active project**: `POST /ingest/refresh` now re-ingests only sources belonging to the active project, preventing cross-project contamination.
- **REPL full-screen / toolbar**: Switched `Application` to `full_screen=False`; pinned input and toolbar to bottom; silenced httpcore debug logs.
- **Math formula detection**: Wired formula detection into the REPL render pipeline so LaTeX blocks are fenced correctly.
- **`init_llm` kwarg removed**: Stale `init_llm` keyword argument removed from `AxonBrain` instantiation in CLI entry point.
- **RRF threshold**: Fixed reciprocal-rank-fusion score threshold being applied before fusion, causing valid results to be dropped.
- **Graph-data endpoint**: Fixed `/graph/data` returning empty payload when no community rebuild had been triggered.
- **Rust loader and CI**: Updated tests and CI configuration for Rust bridge loader path changes.
- **`tempfile.mkdtemp()`**: Replaced bare `mkdtemp()` calls in benchmark scripts with `TemporaryDirectory()` context managers to ensure cleanup.
- **`brain` initialisation in REPL**: `AxonBrain` is now instantiated before entering the interactive REPL when none is available, fixing non-TTY and test runs.
- **Stale gitlinks removed**: Dangling gitlinks `_wt_v010` and `tmp_graphiti_research` removed from the index.
- **Pre-commit hook stability**: Resolved hook failures caused by coverage file regeneration; `--no-verify` is no longer needed for normal commits.

### 🔒 Security

- **URLLoader SSRF hardening**: Blocked requests to private and link-local address ranges (127.x, 169.254.x, 10.x, 192.168.x, 172.16–31.x) in the URL loader.
- **LanceDB injection hardening**: Sanitised user-supplied filter expressions passed to LanceDB queries to prevent expression injection.
- **Multipart upload OOM prevention**: `POST /ingest` now streams uploaded files to disk in 1 MB chunks and enforces a 500 MB per-file cap, eliminating memory exhaustion from large uploads.
- **Vector store input validation**: `OpenVectorStore.add()` now raises `ValueError` immediately on mismatched `ids`/`texts`/`embeddings`/`metadatas` lengths.
- **40 code-review findings addressed**: Security, correctness, and performance issues identified in a full fleet code-review pass were resolved.

### ⚡ Performance

- **Rust-accelerated BM25**: Optional Rust bridge (`axon_rust`) provides a compiled BM25 implementation; benchmark suite included for comparison with the Python fallback.
- **BM25 JSONL corpus log**: BM25 corpus is now streamed to/from JSONL format, reducing peak memory during large index loads.
- **GraphRAG incoming index + batch map-reduce**: Pre-built incoming-edge index and batched community map-reduce cut GraphRAG query latency significantly on large graphs.
- **Bloom filter hash store**: Document deduplication now uses a bloom filter backed hash store, reducing memory and speeding up ingestion of large corpora.
- **Regex cache**: Compiled regex patterns are cached at module load, eliminating repeated recompilation in the hot query path.
- **Unified query transforms**: HyDE, multi-query, step-back, and compression transforms share a single transform pipeline, reducing redundant LLM calls.
- **`graph_lock` thread safety**: Added a dedicated lock around GraphRAG graph mutations to prevent data races under concurrent requests.

### 🔧 CI / Build

- **Multi-platform CI matrix**: CI now runs on Ubuntu, macOS, and Windows runners in parallel.
- **`shell: bash` on Windows CI**: Added explicit `shell: bash` to all CI steps that use bash syntax, fixing failures on Windows runners.
- **Rust step skipped on non-Windows**: Rust build step conditionally skipped on non-Windows CI runners where the pre-built `.pyd` is not applicable.
- **ChromaDB in `dev` extra**: Moved `chromadb` from required to the `dev` optional extra to keep the base install slim.
- **PyPI trusted publisher**: Switched to OIDC trusted-publisher workflow for PyPI releases; dropped long-lived API tokens.
- **Single-package Python + Rust wheels**: Build system updated to ship a unified wheel containing both Python and the compiled Rust extension with one-source versioning.
- **Version sync**: Version bumped to `0.2.0`; synced across `pyproject.toml`, VS Code extension `package.json`, and Cargo manifest.

### 📚 Documentation

- **MCP tool count corrected**: Fixed tool count in docs (30 → 31) and added `bloom_filter_hash_store` to the config reference.
- **Dynamic Graph roadmap**: Added design doc and GitHub Project setup for the dynamic graph feature track.
- **Sprint process doc**: Added `sprint_process.md` with GitHub Project field and iteration IDs for team planning.
- **Fleet docs and website**: Updated fleet agent documentation and project website alongside tools-parity audit.
- **Config reference expanded**: `ADMIN_REFERENCE.md` updated with new config flags introduced in this release (bloom filter, tqdb defaults, offline model dirs).

### 🧪 Tests

- **Fleet agent test suite**: Sharing, maintenance, and tools-parity tests added as part of the fleet audit.
- **GraphRAG regression fixes**: Resolved 40 pre-existing `test_main.py` GraphRAG failures and stubbed `test_code_retrieval`.
- **Test isolation hardened**: REPL e2e tests, CLI tests, and Rust bridge tests stabilised to avoid inter-test state leakage.
- **BM25 persistence tests**: Added round-trip tests for BM25 JSONL corpus serialisation and bloom filter hash store.
- **Vector store input-validation tests**: Added `pytest` cases asserting `ValueError` for mismatched `add()` input lengths.
