# Changelog

## [Unreleased]

### 🛠 Tooling — `index.html` version is now single-source

The landing page used to hand-roll five version strings; every release bump chased them with regexes and Copilot caught stragglers twice (PRs #104, #105). PR I closes that loop.

- **`index.template.html`** — the source-of-truth landing page; every release-version slot uses `{{AXON_VERSION}}`. Historical attribution (e.g. "v0.3.2 graph backend changes" educational content) stays verbatim.
- **`index.html`** — committed rendered output. GitHub Pages serves the repo root with no build step, so we keep the rendered file checked in and let the audit script catch drift.
- **`scripts/render_index.py`** — reads version from `src/axon/Cargo.toml`, substitutes `{{AXON_VERSION}}`, writes `index.html`. `--check` mode compares without writing (returns 1 on drift).
- **`scripts/bump_version.py`** — replaces its hand-rolled regex pair with a `render_index.py` subprocess call.
- **`scripts/audit_packaging.py`** — invokes `render_index.py --check`; reports `index.html vs template: in sync | OUT OF SYNC` and fails the audit on drift.
- 9 tests in `tests/test_render_index.py`: substitutes-all, version override, count-in-stdout, --check pass + fail + no-write, missing-placeholder error, missing-template error, real-repo drift guard.

### 🔒 Security — Item 4: Metadata leakage hardening

Two of three sub-items from the plan; **4b deferred to v0.5.0**.

#### 4a — Hostname → store-scoped UUID node_id

`version.json` markers now stamp the writer's identity with a UUID4 (`owner_node_id`) minted once at store-init time and cached in `store_meta.json::node_id`. The legacy `owner_host` field is retained as an empty string for schema continuity with v0.3.x readers; new writers no longer leak `socket.gethostname()` through the synced filesystem volume. `axon.projects.get_or_create_node_id` migrates pre-v0.4.0 stores in-place on first read. `axon.version_marker` no longer imports `socket` at all.

#### 4c — Random padding in AXSL sealed files (`security.seal_padding_bytes`)

New `AxonConfig.seal_padding_bytes: int = 0` (off by default, fully backward-compatible). When `> 0`, every `SealedFile.write` / `write_stream` / `write_stream_from_path` appends a random number of bytes between `0` and `seal_padding_bytes` (inclusive) **after** the GCM tag. Reader slices the padding off via the new `padding_length` field stamped into 4 bytes of the previously-reserved header region (preserves the 16-byte header size). The bound: a 1024-byte budget hides plaintext length to within ±1 KiB; for share wraps and KEK files (~40 bytes) this is enough to mask whether a wrap is "small metadata" or "an unusual share". Plumbed through `project_seal` so the existing seal pipeline picks up the config; share-wrap and KEK callers can opt in incrementally.

#### 4b — Hashed key_id filenames in `.security/shares/` — DEFERRED to v0.5.0

Implementing this cleanly requires an encrypted index file (so owners can still enumerate shares for `list_sealed_shares` and `hard_revoke`), which is a non-trivial design surface. The existing leak (filenames carry plaintext `key_id`s) remains in v0.4.0; documented as known limitation.

#### Tests

`tests/test_metadata_hardening.py` — 17 tests:
- 4a: `ensure_user_project` writes `node_id`; `get_or_create_node_id` round-trip + legacy migration; missing `store_meta` → empty string; `version_marker.bump` writes `owner_node_id` + empty `owner_host`; defensive regression against `import socket` re-appearing.
- 4c: round-trip with padding; baseline file size unchanged when `padding_bytes=0`; padding distribution check (200 writes, no length > 30%); streaming write supports padding; negative `padding_bytes` rejected; truncated trailing padding fails cleanly via `SealedFormatError` instead of misaligned `InvalidTag`.
- Config: `seal_padding_bytes` default 0; YAML round-trip; negative value rejected at load.

### 🔒 Security — Item 3: Ephemeral plaintext cache mode

- New `security.seal_cache_ephemeral: bool = false` config (off by default).
- When ON for a sealed project: every retrieval runs inside a per-query mount/unmount cycle. The plaintext-on-disk window collapses from "entire session" to "one query execution time" (~1s). Cost: re-decrypt per query (vs. once per session today).
- `_execute_retrieval` in `query_router.py` now wraps the body in `AxonBrain._ephemeral_query_window()` so all three call sites — `search_raw`, `query`, `query_stream` — share the same per-query lifetime. LLM synthesis runs against in-memory chunks **after** the window closes, so the wipe is safe.
- New manual wipe API for "scrub now" (works regardless of `seal_cache_ephemeral`):
  - **CLI**: `axon --wipe-sealed-cache` (also `--seal-cache-ephemeral` flag override)
  - **REPL**: `/store wipe-cache`
  - **REST**: `POST /security/wipe-sealed-cache` → `{wiped: bool}`
  - **MCP**: `wipe_sealed_cache` (51st tool)
- Cache re-materialises on the next query via stored remount args (`_sealed_remount_args`).
- 13 tests in `tests/test_ephemeral_cache.py`: config round-trip, wipe semantics (3 cases), context-manager behaviour (4 cases — pass-through outside ephemeral, remount+wipe inside, wipe still fires when body raises), REST contract (3 cases including no-brain shape).

### 🔒 Security — Item 2: Keyring hardening (3 modes)

Per-share DEK and master-key storage now obeys a configurable `security.keyring_mode`:

- **`persistent`** (default, current v0.3.x behaviour) — DEK lives in the OS keyring (DPAPI / Keychain / Secret Service) until revoked, expired, or auto-destroyed.
- **`session`** — DEK lives only in a process-local `SessionDEKCache` (thread-safe `dict`). OS keyring is never touched. Wiped on process exit. Practical for server / Docker / CI deployments where `persistent` would fail with `KeyringUnavailableError`.
- **`never`** — DEK is never cached anywhere. Every `get_grantee_dek` returns `None` from cache; callers must re-derive from the share string. Suitable for air-gapped / high-security deployments where any persistent DEK material is unacceptable.

Cross-interface parity:
- **CLI** — `axon --keyring-mode {persistent|session|never}` (per-invocation override of config)
- **REPL** — `/store keyring-mode [persistent|session|never]` (read-or-set; shows session cache size)
- **REST** — `GET /security/status` now returns `keyring_mode` + `session_cache_size`; `POST /security/keyring-mode {mode}` (50th endpoint after Item 1)
- **MCP** — `set_keyring_mode(mode)` (50th tool)
- **Config** — `security.keyring_mode` field in YAML + `AxonConfig`

24 tests in `tests/test_keyring_modes.py`: `SessionDEKCache` thread-safety (16 threads × 100 ops), mode dispatch (persistent → OS keyring, session → in-memory, never → silent drop), config round-trip + invalid-mode rejection, REST status + setter contract.

### 🔒 Security — Item 1: Diceware passphrase generation

- **EFF large wordlist** (CC BY 3.0 US, 7,776 words) bundled under `src/axon/security/data/`. License preserved as `LICENSE-EFF-WORDLIST.txt`.
- **`axon.security.generate_passphrase(n_words=6)`** — uses `secrets.choice` for cryptographic randomness. 6 words ≈ 77.5 bits of entropy (`log2(7776**6)`), enough to make scrypt brute force infeasible.
- **Cross-interface parity** — exposed on every surface:
  - **CLI** — `axon --passphrase-generate [--passphrase-words N]`
  - **REPL** — `/passphrase generate [N]`
  - **REST** — `GET /suggestions/passphrase?words=N&separator=S`
  - **MCP** — `suggest_passphrase(words=6, separator="-")` (49th tool)
- 32 tests in `tests/test_passphrase.py` covering wordlist parse, entropy, format, edge cases, no-duplicate-in-1000-runs, and REST contract.
- Default separator is space (4 EFF entries are themselves hyphenated, so `-` would be visually ambiguous as a word delimiter). REST endpoint defaults to `-` for URL-friendliness.

---

## [0.4.0] - 2026-05-04

### 🔒 Security — TTL-gated sealed shares with auto-destruction

Closes the v0.3.x security gap where a redeemed sealed-share DEK lived in the grantee's OS keyring **indefinitely**. v0.4.0 adds:

- **Ed25519 signing keypair** — derived deterministically from the owner's master via HKDF-SHA256. Domain-separated from the per-share KEK derivation. No new files on disk.
- **`SEALED2:` share-string envelope** — extends `SEALED1:` with a 7th field carrying the owner's signing pubkey hex (64 chars). Backward-compatible — older `SEALED1:` strings sent before v0.4.0 keep redeeming.
- **Signed expiry sidecar** at `<project>/.security/shares/<key_id>.expiry` — JSON `{key_id, expires_at, sig}`. Owner signs `b"key_id:expires_at"` with the Ed25519 privkey; grantee verifies on every mount using the pubkey from the SEALED2 envelope.
- **Mount-time TTL check** in `get_grantee_dek()` — seven failure modes (expired, tampered date, rename attack, missing pubkey, malformed JSON, non-dict JSON, bad signature) all raise `ShareExpiredError`.
- **Auto-destroy on expiry** in `_mount_sealed_project()` — wipes DEK from keyring + file fallback, releases active plaintext cache, removes mount descriptor. Encrypted source files on the synced filesystem are **deliberately not touched** (would propagate destruction back to the owner via OneDrive sync).

### 🛠️ Surfaces

`ttl_days` now flows through every share-generation surface:
- **REST** — `POST /share/generate` accepts `ttl_days: N` for sealed projects too (was plaintext-only). Response carries `expires_at` (canonical ISO 8601 UTC, `Z` suffix).
- **CLI** — `axon --share-generate research alice --share-ttl-days 30` (works for both modes; help text updated).
- **REPL** — `/share generate research alice --ttl-days 30` (works for both modes).
- **MCP** — `share_project(..., ttl_days=N)` already wired in v0.3.2; now also propagates through the sealed branch on the server side.

`POST /share/generate` rejects `ttl_days <= 0` with HTTP 422.

### 🛡️ Wire-format invariants

- `SEALED1:` envelope (6 fields) — unchanged. Continues to be accepted on redeem.
- `SEALED2:` envelope adds field 7: lowercase hex Ed25519 pubkey (exactly 64 chars).
- Signed message format: `f"{key_id}:{expires_at_iso}".encode()` — bumping requires a new envelope version.
- `expires_at` is always written as ISO 8601 UTC with `Z` suffix (never `+00:00`).
- Sidecar atomic write with `.sealing` tmp + `os.replace` + `0o600` perms (matches the wrap-file convention).

### 🐛 Bug fixes

- `_check_expiry_or_raise` now defends against non-dict JSON (`[]`, `null`, `42`) and non-string field values — all funneled through `ShareExpiredError` per the contract.
- `_auto_destroy_expired_share` strips the `mounts/` prefix before calling `remove_mount_descriptor` (the helper expects the bare mount name; descriptor would otherwise stay orphaned).

### ⚙️ Developer experience

- `scripts/precommit_pytest_scoped.py` replaces the testmon-based pre-commit hook. Path-prefix mapping picks a tight subset of test files per change area: `axon/security/*` → `tests/test_sealed*`, `axon/api_routes/*` → `tests/test_api*`, etc. Predictable runtime — sub-minute for any change area, vs the 30+ min testmon worst case on foundational-module edits. CI still runs the full suite as the safety net.

### 📦 Packaging

- `axon-rag` 0.4.0 on PyPI.
- VS Code extension `axon-copilot-0.4.0.vsix` rebuilt + bundled under `src/axon/extensions/`.
- `scripts/audit_packaging.py --expected-version 0.4.0` passes — Cargo, package.json, index.html (hero + terminal), bundled VSIX all in sync.

### ⚠️ Known limitations

- **Clock skew**: TTL relies on the grantee's local clock. Ed25519 prevents tampering with the date but not clock manipulation. NTP oracle out of scope.
- **Encrypted sync files NOT deleted**: deleting them would propagate destruction back to the owner via OneDrive — destructive failure mode.
- **Pre-v0.4.0 grantees**: client-side enforcement model. Older grantees won't perform the TTL check; they pre-date the security gap closure.

### ⬆️ Upgrade from v0.3.2

```bash
pip install --upgrade axon-rag
```

- **No breaking changes** for existing share strings — `SEALED1:` envelopes keep working forever.
- **Owner side**: pass `--share-ttl-days N` (or `ttl_days=N` via API/MCP) to mint a TTL-gated share.
- **Grantee side**: TTL is enforced automatically — no config or migration needed. Once you upgrade, expired shares auto-destroy on next mount.

---

## [0.3.2] - 2026-05-03

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
