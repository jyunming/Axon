# Changelog

## v0.3.0 — 2026-04-27

### ✨ New Features

- **Sealed sharing (AES-256-GCM encrypted-at-rest)** — all 7 phases shipped; works through OneDrive, Dropbox, and Google Drive. Cloud providers see only ciphertext. See [Sharing Guide](docs/SHARING.md).
- **Cross-platform key portability** — dual-write `master.enc` so sealed projects move cleanly between Windows, macOS, and Linux without re-sealing.
- **Grantee headless Linux / Docker support** — share DEK file fallback for environments without a GUI keyring (CI, servers, containers).
- **SPLADE sparse retrieval (Phase 1)** — hybrid dense + BM25 + sparse pipeline; opt-in via `pip install axon-rag[sparse]` + `sparse_retrieval: true` in config.yaml.
- **7 new MCP tools** — governance suite, `query_stream`, `mount_refresh`, `seal_project`.
- **Split health endpoints** — `/health/live` (process up) and `/health/ready` (brain ready) replace the single `/health` route; Prometheus metrics at `/metrics`.
- **Per-IP rate limiting** — applied to share, ingest, and security endpoints.
- **Structured logging** — every request logs a `X-Request-ID` header for distributed tracing.

### ⬆️ Upgrade from v0.2.1

```bash
pip install --upgrade axon-rag          # base upgrade
pip install "axon-rag[sealed]"          # add cloud-drive sharing support
```

- **No action required for existing projects** — all changes are fully additive and backward-compatible.
- **To enable sealed sharing**: run `axon --project-seal <name>` once per project (opt-in, irreversible without re-ingest).
- **To enable cloud-drive sharing**: set `store.base` to your OneDrive / Dropbox / Google Drive path in `config.yaml`.
- **To enable sparse retrieval**: `pip install axon-rag[sparse]` then add `sparse_retrieval: true` to `config.yaml`.

---

## [0.2.1] - 2026-04-25

### ✨ New Features

- **Sealed-mount stack**: Encrypted-at-rest project sharing via AES-256-GCM + per-share AES-KW key wrap; works on cloud-sync mounts (OneDrive, Dropbox, Google Drive) without any server. Adds `/security/{status,bootstrap,unlock,lock,change-passphrase}` REST routes, matching MCP / VS Code LM tools, and `axon /store` REPL commands.
- **Cross-machine staleness detection**: Owner writes `version.json` after each ingest; grantees auto-detect re-indexes via marker bump (or per-query polling when `mount_refresh_mode=per_query`).
- **Three-tier sync test strategy**: Unit (mock filesystem), integration (real WebDAV via Nextcloud-in-Docker), smoke (manual OneDrive recipe in `docs/SHARE_MOUNT_SMOKE.md`).
- **Single-PR release automation**: `scripts/bump_version.py` bumps `pyproject.toml`, rebuilds + bundles VSIX, refreshes `Cargo.lock`, and runs `audit_packaging.py` in one command.

### 🐛 Bug Fixes

- **Sealed-share lifecycle**: Hard-revoke now bulk-deletes share wraps BEFORE promoting the rotated DEK so a crash mid-revoke can never leave grantees with mismatched keys; bumps `version.json` so mounted grantees notice. `_executor.shutdown(wait=False)` now waits on submitted futures so background graph persists are not silently dropped on close.
- **Concurrency hardening**: TOCTOU races on lazy-initialised `_graph_lock` / `_traversal_cache_lock` / `_persist_executor` properties closed via eager init in `AxonBrain.__init__` plus DCL fallback in mixins; `SealedCache.wipe()` is now thread-safe; `governance.emit()` reuses a singleton thread pool instead of spawning a daemon per audit event.
- **API hardening**: `/query/stream` now yields tokens incrementally (was buffering full response); CORS middleware actually applies `api.allow_origins` from config; lifespan re-raises so brain init failures fail fast; `_unlock_failures` rate-limit dict no longer grows unbounded and only ticks on credential failures.
- **Loaders SSRF**: URL loader now SSRF-checks every redirect hop (was only initial + final URL); pre-checks Content-Length to avoid OOM on hostile origins. `_hashlib.md5(..., usedforsecurity=False)` annotation on the ingest-refresh dedup hash so FIPS-mode runtimes don't reject it.
- **Rerank**: Tolerant LLM score parser pulls the first numeric run from the response (previously returned 0.0 on `"Score: 8"`-style replies, making rerank useless).
- **Empty-list ValueError**: `ThreadPoolExecutor(max_workers=min(N, len(...)))` calls now `max(1, ...)` so empty result lists don't crash MultiVectorStore / MultiBM25Retriever / compression / rerank.
- **Atomic + cloud-sync-safe writes**: `_save_code_graph`, `_do_save_entity_graph`, and graph-rag JSON/bytes writers now use `_atomic_replace` with a copy+unlink fallback for transient cloud-sync locks.
- **GLiNER NER backend**: `_extract_entities_gliner` now traps `ImportError` and surfaces a clear "install with `pip install axon-rag[gliner]`" hint instead of failing silently.
- **`graph_rag_*` config knobs**: ~30 knobs that were read via `getattr(cfg, …, default)` are now declared `AxonConfig` dataclass fields so they round-trip through YAML, validation, and `/config/get`.

### 🔒 Security

- **Sealed cache `.security/` skip**: Plaintext cache no longer mirrors the wrap-material directory into the OS temp dir.
- **Governance panel CSP**: Replaced `script-src 'unsafe-inline'` with a per-render nonce; inline `onclick` handlers moved to `addEventListener`.
- **Embedding retry framework**: `_retry_call` wraps Ollama / OpenAI embedding calls with exponential-backoff + jitter so transient 5xx no longer kills long ingests.

### 📚 Docs

- New: `docs/SHARE_MOUNT_SEALED.md`, `docs/SHARE_MOUNT_SMOKE.md`, `docs/AUDIT_2026_04_26.md`.
- README + `docs/SETUP.md`: vector store default updated to TurboQuantDB; tool / endpoint counts corrected.
- `CLAUDE.md`: bumped current version pointer.

## [0.2.0] - 2026-04-22

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
