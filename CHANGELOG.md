# Changelog

## [Unreleased]

### ✨ New Features

- **Graph backend capability flags** — `FinalizationResult.status` is now `"ok"`, `"not_applicable"`, or `"error"` so callers can tell "ran and built nothing" apart from "this backend has no finalize step". `dynamic_graph` returns `not_applicable`; the federated backend aggregates statuses from sub-backends. Surfaced via `POST /graph/finalize` and the `graph_finalize` MCP tool.
- **Point-in-time graph retrieval surface** — new `POST /graph/retrieve` REST route, `graph_retrieve` MCP tool, `/graph retrieve <q> [--at TS]` REPL command, and `--graph-retrieve QUERY [--graph-at TS]` CLI flag run the active backend's `retrieve()` directly with a `RetrievalConfig`, surfacing `point_in_time` historical queries that were already implemented internally. Also exposed as a VS Code LM tool (`graph_retrieve`).
- **Conflict inspection** — new `GET /graph/conflicts` REST route, `graph_conflicts` MCP tool, `/graph conflicts` REPL command, `--graph-conflicts` CLI flag, and VS Code LM tool return facts with `status='conflicted'` (incompatible exclusive-relation facts in the same scope). Backends without conflict tracking return `supported: false` instead of an empty list.
- **Per-query federation weight override** — `RetrievalConfig.federation_weights` and the `federation_weights` field on `POST /graph/retrieve` / `graph_retrieve` MCP tool override the project-level `graph_federation_weights` for a single retrieve. Lets agents shift weight toward `graphrag` or `dynamic_graph` per-question without changing config. Validated to reject unknown keys and negative values.
- **LangChain `BaseRetriever` adapter** — new `axon-rag[langchain]` extra ships `axon.integrations.langchain.AxonRetriever`, a drop-in `BaseRetriever` subclass that wraps `AxonBrain.search_raw()`. Any LangChain agent can now use Axon as its local retrieval backend without REST round-trips. Per-call overrides via `with_overrides({...})`.
- **LlamaIndex `BaseRetriever` adapter** — new `axon-rag[llama-index]` extra ships `axon.integrations.llama_index.AxonLlamaRetriever`, returning native `NodeWithScore` for use in any LlamaIndex query engine.
- **Structured citation metadata** — `POST /query` now returns `sources` (slim view of every retrieved chunk made available to the LLM) and `citations` (structured spans parsed from the response, one per `[N]` / `[Document N]` marker, with character offsets). Lets agents render clickable citations without re-running retrieval. Disable with `include_citations: false` for high-throughput callers that only need the answer string.
- **`axon-rag[starter]` install bundle** — recommended one-line install for first-time users. Pulls Streamlit UI, sealed-mount sharing (cryptography + keyring), and the optional document loaders (EPUB, RTF, .msg) so a fresh `pip install "axon-rag[starter]"` covers >90% of beginner workflows. Power users keep the granular extras.
- **First-run setup auto-trigger** — running plain `axon` on a fresh checkout (no config file at the default path, no projects under the AxonStore base) now sends the user through the setup wizard before dropping into the REPL. Existing installs are unaffected. Press Ctrl+C to skip and configure later via `axon --setup`.
- **`axon --doctor` health-check command** — non-destructive sanity check that prints a colored checklist for: Python version (≥ 3.10), Ollama daemon reachable, default LLM model pulled, AxonStore base directory writable, recommended extras present. Each warning carries a one-line "do this next" hint. Exits non-zero on any required-check failure so CI and shell scripts can use it as a precondition.

### 🐛 Bug Fixes

- `BACKEND_ID` is now a class attribute on `DynamicGraphBackend` and `GraphRagBackend` (was only a module-level constant). The federated backend's `b.BACKEND_ID` lookup in `retrieve()` now works as documented.

### 📚 Doc cleanup (vector-store default consistency)

The dataclass default for the vector store has been `turboquantdb` since v0.2.1, but several user-facing docs and the (unreferenced) `config.yaml.template` still showed `provider: lancedb` as the default in their example configs. Fixed in `docs/OFFLINE_GUIDE.md`, `docs/MODEL_GUIDE.md`, `docs/SETUP.md`, and `config.yaml.template` so every example aligns with the actual code default. LanceDB remains a fully-supported alternative (covered in the same examples) — only the labelling changes.

### 🛠️ Developer Experience

- **Pre-commit pytest now uses pytest-testmon** — selective test runs based on per-file coverage tracking. Typical local commit drops from ~45 min to ~30 s for doc-only changes; source edits to widely-imported modules still take a few minutes. CI is unaffected (full suite still runs on every push). Cache file `.testmondata` is gitignored. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#pre-commit-pytest-testmon-accelerated) for cache-recovery commands when the selection is wrong.

---

## [0.3.1] - 2026-04-29

### ✨ New Features

- **Dynamic Graph temporal queries** — `RetrievalConfig.point_in_time` filters facts valid at any past timestamp; backed by the `idx_facts_temporal` index for indexed temporal lookups.
- **Dynamic Graph conflict detection** — exclusive-relation facts with the same scope_key and valid_at timestamp (±1 s) are marked `conflicted` rather than silently superseded; surfaced in `status()` as `conflicted_facts`.
- **Federated graph backend** — `graph_backend: "federated"` runs `graphrag` + `dynamic_graph` concurrently via `ThreadPoolExecutor` and fuses results with per-backend weighted Reciprocal Rank Fusion. Wall-clock latency ≈ max(t_graphrag, t_dynamic). Weights tunable via `graph_federation_weights` in `config.yaml`.
- **Code AST extraction** — Python code chunks are parsed with `ast` (stdlib) instead of an LLM call; extracts graph entities such as `CONCEPT`/`PRODUCT` and relation facts such as `IMPORTS`/`INHERITS`. Faster and deterministic.
- **Dynamic Graph visualization enrichment** — `graph_data()` now attaches node colors, tooltips, and `valid_at`/`invalid_at` temporal labels to links for the 3D renderer.
- **REPL `/project rotate-keys`** — rotates the sealed project DEK and invalidates all outstanding shares from the REPL (previously REST-only).
- **TurboQuantDB v0.7.0/v0.8.0** — hybrid BM25+dense search (`tqdb_hybrid: true`, `tqdb_hybrid_weight` in config), `tqdb.aio.AsyncDatabase` for non-blocking FastAPI query paths, `delete_batch` support, and `tqdb.migrate` toolkit for importing existing ChromaDB/LanceDB collections.

### 🐛 Bug Fixes

- Fixed `None` metadata entries crashing `tqdb` inserts by normalising to `{}` before write.
- Fixed `FakeVectorStore.search()` signature in e2e conftest missing `query_text` parameter.
- Fixed `MultiVectorStore.search/batch_search` not forwarding `query_text`/`query_texts` to sub-stores.

### ⬆️ Upgrade from v0.3.0

```bash
pip install --upgrade axon-rag
```

- **No breaking changes** — all additions are fully backward-compatible.
- **To enable federated backend**: set `graph_backend: "federated"` under `rag:` in `config.yaml`.
- **To enable hybrid search**: set `tqdb_hybrid: true` under `vector_store:` in `config.yaml` (TurboQuantDB only).
- **Minimum tqdb version**: `0.7.0` (updated from `0.1.0`).

---

## [0.3.0] - 2026-04-27

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
pip install --upgrade "axon-rag[sealed]"   # upgrades base + adds cloud-drive sharing support
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
