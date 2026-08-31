# Changelog

## [Unreleased]

A third capabilities-audit cycle, this time targeting v0.4.4 itself via an
independent `/code-review` pass over its full diff rather than waiting for
the next scheduled audit — one security fix and several outright
regressions introduced by that same release.

### 🔐 Security fixes

- **Atomic file replacement silently widened restrictive file permissions.**
  `_atomic_replace()`'s temp file always carried the process umask's default
  mode, not the target's — a `0600` file (the sealed-share key material this
  helper's own docstring names as an intended use case) rewritten through it
  would silently widen to world-readable. The target's existing mode, if
  any, is now copied onto the replacement before the swap.

### 🐛 Fixes

- **`--project-pack`/`--project-unpack` bypassed the single-instance
  server-detection gate** that `--ingest`/`--project-new`/`--project-delete`
  already use, despite writing/reading a project's on-disk footprint
  directly — unpacking into a project a live `axon-api` is serving could
  race the server's open file handles. Both now route through the server
  (new `remote_project_pack`/`remote_project_unpack`) when one is running.
- **REPL `/clear` always failed when reusing a running server** (the new
  single-instance default) — it called local-brain-only internals
  (`_assert_write_allowed`, `vector_store`) that a `RemoteBrain` proxy
  doesn't implement. Added `RemoteBrain.clear()`, routed through the
  server's `/clear`.
- **`RemoteBrain` silently dropped `chat_history`** at `logger.debug` —
  invisible by default, so a multi-turn follow-up against a reused server
  lost conversational context with no indication why. Promoted to a
  once-per-instance `logger.warning()`.
- **The RAPTOR summary cache could be reset without its own lock** on
  project switch/clear — a race with the lock this same release added for
  every other access to that cache, that could silently drop or reintroduce
  an entry right after a user-initiated wipe.
- **`add_text`/`ingest_url`/`ingest_texts`/`refresh_ingest` MCP tools
  reported false success on a fully-deduped ingest** — the sibling
  `ingest_path` tool was already fixed for this in v0.4.4; these four
  discarded `brain.ingest()`'s actual chunk count and reported
  `len(docs)` (or, for `refresh_ingest`, an unconditional increment)
  instead.
- **A narrow two-process race could let two `axon-api` servers both start**
  against the same store — the read-then-decide-then-write gap between
  `find_live_server_for_store()` and `write_store_lock()` had no atomicity.
  The lock is now claimed via an exclusive file create, so at most one
  near-simultaneous starter wins it; the loser aborts the same way it would
  have if it had detected the other server on the earlier check.
- **A `graph_data()` failure rendered as a silent empty graph** across every
  export surface (CLI, REPL, REST) with zero diagnostic trail. Now logged.
- **The web GUI's chat crashed with a raw traceback on a `query_stream`
  error** (a server error, or a dropped connection to a reused server) —
  the REPL's equivalent loop already degraded gracefully; the web GUI now
  does too.
- **The shared LRU+TTL cache evicted an unrelated entry when updating an
  already-present key at capacity** — overwriting a key doesn't grow the
  store, so the at-capacity check shouldn't fire for it.
- **The VS Code extension's Copilot LLM background worker used a
  once-per-activation `apiBase` snapshot** for its whole lifetime, unlike
  every command elsewhere in the extension (already fixed to resolve fresh
  per call) — a server address that changes after activation left the
  worker polling a stale address indefinitely.

### Notes on findings assessed but not changed

- A Windows CRLF→LF change in `config.yaml` writes (from the atomic-write
  helper introduced in v0.4.4) is cosmetic only — YAML parses identically
  either way, `config.yaml` isn't a tracked file, and the existing test
  suite already relies on the raw-bytes, no-translation behavior for
  deterministic cross-platform digest matching. Left as-is.
- `lifespan()`'s host/port bookkeeping is only fully correct when launched
  via `main()` (or the VS Code extension, which already sets the env vars
  itself) — a raw manual `uvicorn axon.api:app` invocation bypassing both
  can still desync the lock file's recorded port. Both supported launch
  paths are unaffected; hardening the unsupported path would need deeper
  ASGI/uvicorn socket introspection than fits a patch release.
- `GraphRagEngine` retaining a live back-reference to `brain` (the M2
  ownership-inversion refactor not being a full decoupling) and a ~65-line
  duplicated GraphRAG context-assembly block between `query()`/
  `query_stream()` are simplification opportunities, not bugs — deferred to
  avoid destabilizing the graph subsystem under a patch-release fix cycle.

## [0.4.4] - 2026-08-31

A feature release — project backup/restore, self-update, and single-instance
coordination — alongside a second capabilities-audit cycle: a path-traversal
fix, a missing system-path guard on the web UI's ingest picker, five
concurrency/locking bugs, and the graph-engine ownership-inversion refactor
that several of the audit's own findings had to route around.

### ⚡ Single-instance behaviour — one brain per store

Multiple Axon surfaces used to each build their own full `AxonBrain` on the same
store. Two processes writing one TurboQuantDB store race on the native files and
crash, and each re-loads the embedding model on startup. Axon now coordinates a
single owner per store:

- **CLI store-mutating commands route through a running server.** If an
  `axon-api` server is already serving the same store, `axon --ingest` and
  project create / delete / switch are sent to it over HTTP instead of opening a
  second in-process store — so the running server stays the single writer. The
  target store is verified against the server's `projects_root` (via `/config`)
  so a client never writes to the wrong corpus. `--local` forces an in-process
  brain; `AXON_API_BASE` (or the new `api_host` / `api_port` config fields) points
  at a specific server.
- **`axon-api` refuses a second server on the same store.** A per-store lockfile
  records the serving process (host / port / pid); a second `axon-api` pointed at
  that store exits with a clear message instead of opening a competing brain.
  Stale locks (dead server) are ignored. Override with
  `AXON_ALLOW_MULTIPLE_SERVERS=1`.
- **Streamlit UI + CLI REPL reuse a running server.** A new `get_brain(config)`
  factory returns a lightweight `RemoteBrain` HTTP proxy when a same-store server
  is live (queries, ingest, project switch, etc. routed over HTTP; the embedding
  model is never re-loaded), or a local `AxonBrain` otherwise. So `axon-ui` and
  the interactive `axon` REPL no longer stand up a second full brain alongside a
  running `axon-api`. `--local` (REPL) forces in-process. `axon-mcp` already
  worked this way. (Reuse is single-turn against the server; a few REPL slash-
  commands that need direct store internals fall back with a clear message.)
- **Per-process CLI log files** (`axon-YYYYMMDD-<pid>.log`). A shared daily log
  could not be rotated on Windows while another `axon` process held it open —
  every rollover then raised `PermissionError`, flooding the logs and letting the
  file grow without bound (observed at 500+ MB). Each process now owns its file,
  so rotation always succeeds.

### ⚠️ `axon-api`'s default port moved from 8000 to 8420

8000 collided with Django's dev server and with this codebase's own
`vllm_base_url` default — a machine running both `axon-api` and a local vLLM
server on defaults already had a latent port clash. The port is also now
genuinely configurable, not just env-var-only:

- `axon-api --host`/`--port`/`--config` flags now exist for real (previously
  documented in ADMIN_REFERENCE.md but not implemented).
- Resolution precedence: `--port` > `AXON_PORT` env var > `config.yaml`'s
  `api.port` > `8420` default (`--host`/`AXON_PORT` only, not `config.yaml`'s
  client-oriented `api_host`, to avoid silently narrowing the bind from all-
  interfaces to localhost-only). Previously `config.yaml`'s `api.port` had
  zero effect on what port the server actually bound to.
- A bind-time "address already in use" now exits with a clear message
  suggesting `--port <other>` instead of a raw socket traceback.
- If you scripted around the old `:8000` default (curl examples, MCP
  `RAG_API_BASE`, VS Code `axon.apiBase`, etc.), update to `:8420` or set
  `--port 8000` explicitly to keep the old value.
- **The VS Code extension now launches `axon-api` on a dynamically chosen
  free port** instead of always trying the same static one and giving up if
  it's occupied. Resolution order: an explicit `axon.apiBase` override (used
  as-is, never auto-spawned); the last-known port from a previous session
  (adopted if still alive); otherwise a fresh OS-assigned free port. Two real
  bugs were fixed alongside this: the resolved port wasn't written into the
  spawned process's environment, so the server-side lock file always recorded
  `8420` regardless of what port `uvicorn` actually bound to; and ~25 call
  sites across 10 files each independently re-read the raw `apiBase` setting
  instead of sharing one resolved address.

### 📦 `pack_project` / `unpack_project` — back up and restore a project

No export/import/backup mechanism existed anywhere in this codebase before
this. `pack_project` zips a project's entire on-disk footprint (including a
sealed project's ciphertext and DEK wraps, restored still sealed); `unpack_project`
restores a zip back into AxonStore. Zip-slip-safe from scratch — every archive
member is validated (absolute paths, `..` traversal, symlinks) before
anything is written, extraction lands in a staging directory first, and the
final swap is atomic on full success only. Wired across all 7 surfaces: CLI
(`--project-pack`/`--project-unpack`), REPL (`/project pack|unpack`), REST
(`POST /project/pack`, `/project/unpack`), MCP (`pack_project`,
`unpack_project` — 55 → 57 tools), and the VS Code extension (39 → 41 LM
tools).

### ✨ `axon update` — self-update, one command

`axon` (CLI/REPL) and `axon-api` now check PyPI for a newer `axon-rag`
release at startup — non-blocking, rate-limited to once/day via an on-disk
cache, silent when `offline_mode` is on or the check fails. `axon`/REPL
print a one-line suggestion after the startup banner; `axon-api` logs it.
`axon-mcp`/`axon-ui` are intentionally not checked (no human watching
their stdout).

- **`axon update`** (new bare subcommand, plus `/update` in the REPL)
  detects your install method (pip / pipx / conda), runs the matching
  upgrade command, then re-installs the bundled VS Code extension so both
  stay in sync — one command instead of two. Prompts for confirmation
  (`-y`/`--yes` for scripted use).
- Refuses inside a container (`docker compose pull && docker compose up
  -d` is the right move there) and while an `axon-api` server is live
  against the active store, to avoid upgrading a running server's package
  out from under it.
- CLI/REPL only — deliberately no REST or MCP equivalent (see
  [ADMIN_REFERENCE.md §2.10a](docs/ADMIN_REFERENCE.md#210a-self-update)).
  Does not attempt config-schema migration; read this file (or the release
  notes) for breaking config changes, same as before.

### 🔐 Security fixes

- **Path traversal in REPL `/resume`.** The raw argument went straight to the
  session loader with no validation (`/resume ../../../../etc/passwd` reached
  it unmodified) — the equivalent REST route already validated session ids
  against a filesystem-safe pattern; the REPL bypassed it by calling the
  loader directly instead of going through the API layer. Both now share one
  validation pattern.
- **The web GUI's ingest-directory picker had no system-path guard.** The
  REST `/ingest` route already blocks a curated list of system directories;
  the picker only checked containment inside `RAG_INGEST_BASE`, so a broad or
  unset `RAG_INGEST_BASE` left this surface with no system-directory
  protection at all.

### 🐛 Fixes

- **Web GUI chat dropped spaces between words** in streamed answers
  ("HowcanIhelp"). The client trimmed every SSE frame, stripping the leading
  space each tokenizer token carries. It now preserves token whitespace and only
  JSON-parses structured control frames (sources / errors). The Graph Explorer's
  node colors (by entity type), entity search, hop-distance highlighting, and
  legend swatches were restored in the same pass.
- **REPL `/share list` was missing sealed shares entirely** — `axon
  --share-list` already showed every sealed share you've issued; the REPL
  equivalent silently showed "(none)" for the same project. Both surfaces now
  render from one shared listing function.
- **`axon --refresh` compared the wrong hash.** It recomputed a SHA-256 to
  decide whether a file had changed, but `ingest()`'s own dedup uses MD5 — the
  two never agreed, so an unchanged file could still be silently re-ingested.
- **REPL `/governance` and CLI `--governance` never sent the API key.** Both
  hand-rolled their own HTTP call instead of using the shared client helper
  that attaches `X-API-Key`, so either command failed against a
  key-protected `axon-api`.
- **Deleting a multi-chunk document could leave a stale dedup entry**, silently
  blocking re-ingestion of that source afterward — purge now matches by the
  resolved source id, not just the raw chunk ids passed to `/delete`.
- **`AxonBrain.ingest()` returned `None` instead of the chunk count**, which
  crashed the MCP `ingest_path` tool with a `TypeError` on every real,
  non-forced ingest (masked by the tool dispatcher stringifying the
  exception). It now returns the count, and `add_text`/`add_texts`/
  `ingest_url` report "already ingested" instead of a false success when
  dedup drops everything.
- **Fresh installs and config resets got `llm.max_tokens: 2048`, not `8192`**,
  re-triggering the exact truncation-on-reasoning-models bug 0.4.3 said it
  fixed. That fix only ever updated the dataclass default; the literal
  first-run config template, the setup wizard's fallback, and the tracked
  `config.yaml.template` reference file all still said `2048`. All three are
  now `8192`.
- **`compute_doc_hash()`'s Python fallback hashed with SHA-256 instead of
  MD5**, disagreeing with the native Rust implementation (and every other
  ingest-dedup call site) — a document could get a different dedup hash
  depending on whether the Rust extension loaded.
- **RAPTOR's summary cache had no size cap and no lock**, despite being
  written from multiple worker threads during summary generation — unbounded
  growth plus a real (if latent) race. Now capped at `raptor_summary_cache_size`
  (default 500, configurable) and lock-protected.
- **LLMLingua's model cache had no cross-instance sharing or lock**, unlike
  its two siblings (GLiNER, REBEL) — concurrent `AxonBrain` instances could
  each load their own compressor instead of sharing one.
- **`OpenLLM`'s five client-factory caches (Gemini, OpenAI, Grok, Copilot,
  plus Copilot's token refresh) had no lock**, despite being shared by
  concurrent request handlers — two requests hitting a cold cache at once
  (e.g. right after an API-key rotation) could each build a duplicate client
  or fire a duplicate token exchange.
- **`config.yaml` writes were not atomic** — save, the first-run scaffold, and
  all three "reset to defaults" paths (CLI, REST, REPL) wrote straight to the
  live file, so a crash mid-write could leave a truncated, unparsable config
  that the next launch couldn't load.
- **REPL confirmation prompts were six inconsistent copies** (`/clear`,
  `/project new`/`delete`, `/update`, `/config reset`, agent-mode tool
  confirmation) — some accepted `"yes"`, most crashed on Ctrl+C/EOF instead of
  treating it as "no". Consolidated to one shared prompt helper.

### 🏗 Internal

- **Graph-engine ownership inversion (M2 backend-boundary refactor).**
  `AxonBrain` no longer inherits `GraphRagMixin` directly — its ~90 methods
  and graph-state now live on a new `GraphRagEngine`, composed into
  `GraphRagBackend` behind the `GraphBackend` protocol. No behavior change;
  affects only code that reached into `GraphRagMixin` internals directly
  rather than through `AxonBrain`/`AxonConfig`. See `CLAUDE.md`'s
  Architecture section.
- The query router's query cache and the graph backend's traversal cache —
  previously two independent hand-rolled LRU+TTL implementations — now share
  one extracted algorithm.
- The local-LLM reachability probe (duplicated between `llm.py` and
  `doctor.py`) and the CLI's two duplicated knowledge-base table renderers
  were each consolidated to a single implementation.

### 🧪 Tests

- VS Code e2e mocks updated for the port-fix's new `config.inspect()` /
  `workspaceState` calls.

## [0.4.3] - 2026-08-09

Two things at once: the browser UI question is settled — the native web GUI
served by `axon-api` at `/gui/` is now the maintained surface and the Streamlit
UI is deprecated — and Axon gains a `local` LLM provider for any
OpenAI-compatible server you already run, plus the config fixes that verifying
it against a real local model turned up.

0.4.2 is the previous published release, so everything below lands in one step.

### ⚠️ Read this first

Upgrading from 0.4.2, two changes alter existing behaviour despite the
patch-level version:

- **`llm.max_tokens` now defaults to 8192, up from 2048 — for every provider.**
  Reasoning models (Gemma 4, GPT-OSS, DeepSeek-R1 derivatives) spend the budget
  on `reasoning_content` before emitting any `content`, and 2048 truncates them
  mid-thought. This also raises the per-call output ceiling on paid APIs
  (OpenAI, Gemini, Grok, Copilot) for anyone relying on the default. Set
  `llm.max_tokens` explicitly to keep the old value.
- **`streamlit` is no longer in the `[starter]` or `[all]` extras.**
  `pip install "axon-rag[all]" && axon-ui` now fails until you add `[ui]`. The
  browser UI that ships with `axon-api` at `/gui/` needs no extra dependency and
  is the maintained surface.

### ✨ Features

- **`local` LLM provider — point Axon at any OpenAI-compatible server on your machine.**
  llama.cpp (`llama-server`), vLLM, LM Studio, text-generation-inference, LocalAI:
  anything exposing `POST /chat/completions` and `GET /models`. Configure with
  `llm.local_base_url` (default `http://localhost:8080/v1`, also
  `AXON_LOCAL_LLM_BASE_URL`) and the optional `llm.local_api_key`.
  Axon **never loads or unloads models** — an unloaded model fails fast instead of
  stalling behind a 30–90 s hot-swap.
- **Endpoint ping.** `axon --doctor` gains a "Local LLM endpoint" check that reports
  reachability and lists served models; `/local-url ping` does the same from the REPL.
  "Up but serving no models" is reported as a warning, since that is a real state for
  a router-mode server with nothing resident.
- **REPL `/local-url [URL|ping]`** to show, set, or ping the endpoint.
- **Config tools on MCP** — `get_config`, `set_config`, `update_config` and
  `validate_config` (51 -> 55 tools). MCP was the only surface with no config
  access at all, even though `surface_contract.py` already declared
  `config_read` / `config_update` as supported on every surface.

- **RAPTOR and GraphRAG verified against a local model.** Both work end to end
  with llama.cpp serving Gemma 4 26B: RAPTOR built a 2-level summary hierarchy
  (420 s ingest, 145 s query) and GraphRAG answered a multi-hop question with
  citations from two documents (0.5 s ingest at `graph_rag_depth: light`,
  71 s query). MODEL_GUIDE records the settings and the measured costs.
- **Config validation warns on GraphRAG + a local model.** LLM-based graph
  extraction makes one call per chunk; on a slow local model that is impractical
  (measured: >10 min for a single extraction, with `llm.timeout` unable to cut it
  short). The warning points at `rag.graph_rag_depth: light`, which extracted the
  same entities in 0.6 s with no LLM calls.

### ⚠️ Deprecations

- **`axon-ui` (Streamlit) is deprecated** and will be removed in a future
  release. `main_ui()` emits a `DeprecationWarning` plus a stderr notice, and
  the app renders an in-sidebar banner pointing at `http://localhost:8000/gui/`.
  `Surface.WEBAPP` in `surface_contract.py` is marked deprecated — no new
  capability should be exposed there.

### 📦 Packaging

- **`streamlit` dropped from the `[starter]` and `[all]` extras.** The web GUI
  ships with the API server and needs no extra dependency. Users who still want
  the Streamlit UI must install it explicitly: `pip install "axon-rag[ui]"`.
  The `[ui]` extra is unchanged. The `docker-compose` `axon-ui` service keeps
  working — `requirements.txt` still pins `streamlit` for that image.

### 🐛 Fixes

- **`AxonConfig.save()` silently dropped 156 of 241 fields.** Only 85 were
  written, so everything else reverted to its default on the next load —
  `graph_rag_depth: light` came back `standard`, a custom `ollama_base_url`
  came back `localhost`, `mmr` came back `False`. Anything set via
  `axon --setup`, `/config set` or `POST /config/update` with `persist: true`
  was quietly lost on restart. save() now emits every remaining field under
  `rag:`, which load() already maps verbatim onto field names.
- **Three credential fields had no load mapping at all.** save() wrote
  `llm.gemini_api_key`, `llm.ollama_cloud_key` and `llm.ollama_cloud_url`;
  load() produced `llm_gemini_api_key` etc., which match no dataclass field,
  and dropped them. `openai_api_key` was written only under the legacy
  `llm.api_key`, which loads into the separate `api_key` field, so it degraded
  on every round-trip.
- **139 of 240 config fields were unreachable from every API surface.**
  `POST /config/set` resolved keys through a 101-entry alias table and returned
  400 for anything else, so `graph_rag_depth`, `chunk_size`, `llm_temperature`
  and 136 others could only be changed by hand-editing config.yaml — including
  from the VS Code extension and web GUI, which both call that route. The new
  `resolve_config_key()` accepts a curated alias, a bare field name, or the
  last dotted segment, and is shared with REPL `/config set`.
- **`POST /config/update` silently ignored unmodelled keys.** It declares a
  curated 32-field subset for live RAG tuning; Pydantic discarded anything else
  before the route saw it, so the call returned 200 `success` with the field
  untouched. Unknown keys are now reported in an `ignored` list (and still not
  applied — use `/config/set` for the rest).
- **Reasoning models no longer return empty answers.** Gemma 4, GPT-OSS and
  DeepSeek-R1 derivatives put their chain of thought in a non-standard
  `reasoning_content` field and leave `content` empty when the token budget runs out
  mid-thought. Axon read `content` only, so those responses arrived as `""` with no
  error — silently degrading every internal RAG step that parses model output rather
  than displaying it (HyDE, multi-query, step-back, decompose, context compression,
  GraphRAG NER, RAPTOR summaries, LLM rerank). All OpenAI-compatible providers now
  fall back to `reasoning_content`, `vllm` included.
- **`--doctor` no longer warns about a missing `streamlit`.** The check told
  users to install `[starter]`, which no longer ships it — so the hint pointed
  at a bundle that could not resolve the warning. `check_optional_extras()` now
  covers only `cryptography` + `keyring` (sealed sharing).

### ⚠️ Other behaviour changes

- **`llm.timeout` resolves to 300 s when `llm.provider` is `local`** (60 s
  elsewhere, unchanged). Locally served models are slow enough that the cloud
  default broke `step_back` and `query_decompose` outright. An explicit
  `llm.timeout` always wins. Note this bound is per-read, not wall-clock — see
  MODEL_GUIDE.

### 📚 Documentation

- README and `SETUP.md` now document the built-in web GUI as *the* browser
  surface, with the launch line (`axon-api` → `http://localhost:8000/gui/`).
- `axon-ui` marked deprecated across README, `SETUP.md`, `GETTING_STARTED.md`,
  `QUICKREF.md`, `ADMIN_REFERENCE.md`, `DEVELOPMENT.md`, `TROUBLESHOOTING.md`
  and the entry-points diagram. `Makefile` and `docker-compose.yml` annotated.

### 🧪 Tests

- Deprecation banner renders in the Streamlit sidebar; `main_ui()` emits the
  `DeprecationWarning` and stderr notice while still launching.
- Onboarding extras tests updated for the doctor change, including a regression
  test asserting a missing `streamlit` is *not* reported.

## [0.4.2] - 2026-05-19

A maintenance + audit release. **30 bug fixes** across nine parallel codebase-audit units (PRs #122–125, #127–131), plus brand polish on the auto-generated REST docs and the VS Code marketplace tile, plus seven documentation staleness fixes. Several findings were security-relevant — operators on shared / cloud-synced storage should upgrade.

### 🔐 Security fixes

- **`graph_rag.py` pickle RCE mitigated** ([#128](https://github.com/jyunming/Axon/pull/128)) — the opt-in `.relation_graph.cache.pkl` was loaded via `pickle.load()` inside a silent `except Exception: pass`. On a shared / cloud-synced `bm25_path`, an attacker with write access could swap in a malicious pickle and gain RCE on the next query. Added HMAC-SHA256 integrity check keyed by a fresh 32-byte file at `~/.axon/.relation_pickle_hmac.key` (mode 0600); verified before `pickle.loads`, tagged by the existing shard-signature `cache_key` to prevent cross-state replay.
- **`agent.py:_tool_get_config` 8-key secret leak** ([#130](https://github.com/jyunming/Axon/pull/130)) — `dataclasses.fields(AxonConfig)` was dumped into the agent-tool result, which is fed back to the LLM. A cloud LLM (OpenAI/Gemini/Grok/GitHub Copilot/Ollama Cloud) would receive every *other* configured provider's API key. 8 fields masked with `***` while preserving the field name: `api_key`, `openai_api_key`, `grok_api_key`, `gemini_api_key`, `ollama_cloud_key`, `copilot_pat`, `brave_api_key`, `qdrant_api_key`.
- **`SealedCache` materialised tampered plaintext as authoritative** ([#131](https://github.com/jyunming/Axon/pull/131)) — non-AXSL files were copied as-is into the cache. An attacker with write access to the synced filesystem could replace `meta.json` / `bm25_index/*` with plaintext and the backend would read it as authoritative — bypassing authenticated encryption for that file. `SealedCache.create` now consults `seal._should_seal(rel)`; files in the must-seal set that lack the AXSL header raise `SealedFileTamperError`.
- **`/config` and `/config/update` API-key leakage** ([#124](https://github.com/jyunming/Axon/pull/124)) — `old_value` / `new_value` echoed the previous + incoming values verbatim, including API keys. Added shared `_mask_if_sensitive` helper used by all three config endpoints; `_SENSITIVE_FIELDS` extended to include `openai_api_key` + `grok_api_key` (previously leaked through the read paths).
- **Path traversal in `/share/revoke`, `/project/rotate-keys`, `/project/seal`, `GET /session/{id}`** ([#124](https://github.com/jyunming/Axon/pull/124)) — `request.project` flowed unvalidated into `_resolve_owned_project_dir`, which joins segments verbatim under `user_dir`. Added regex gate matching `/share/generate`. Session ID was interpolated into `session_<id>.json`; constrained to `[A-Za-z0-9_-]{1,128}`.
- **`/copilot/agent` IndexError on whitespace-only query** ([#124](https://github.com/jyunming/Axon/pull/124)) — the empty-check ran pre-strip, then `parts[0]` blew up. Now returns the documented 400.
- **DoS caps on unbounded auth inputs** ([#124](https://github.com/jyunming/Axon/pull/124)) — `ShareRedeemRequest.share_string` capped at 16 KB; all three passphrase fields capped at 4 KB so scrypt can't be fed gigabyte inputs.
- **`/security/unlock` lockout used raw `req.client.host`** ([#124](https://github.com/jyunming/Axon/pull/124)) — every other rate limiter honors XFF; switched to the shared `_get_ip` helper.
- **`access.py:check_write_allowed` silent except masking maintenance state** ([#125](https://github.com/jyunming/Axon/pull/125)) — bare `except Exception: pass` swallowed any failure of the maintenance-state lookup, bypassing readonly/offline/draining guards with no audit trail. Now logs at `WARNING`; fail-open behaviour preserved for back-compat.
- **`AxonBrain.close()` left sealed plaintext on disk after backend error** ([#127](https://github.com/jyunming/Axon/pull/127)) — a raise in any backend `store.close()` (e.g. Windows file lock on turboquantdb) terminated the loop early AND skipped the sealed-cache cleanup. Each `store.close()` now wrapped in try/except.
- **Stale `.expiry` sidecar after sealed-share revoke** ([#131](https://github.com/jyunming/Axon/pull/131)) — `_soft_revoke` + `_hard_revoke` left `.expiry` files on disk. Re-issuing the same `key_id` without `expires_at` silently inherited the old TTL, or worse — a past-expired sidecar denied access to a freshly-minted share (DoS). Both revoke paths now delete `.expiry`.
- **`SealedFile` writers had no upper bound on `padding_bytes`** ([#131](https://github.com/jyunming/Axon/pull/131)) — reader's `_unpack_header` rejected `padding_length > 1 MiB`, so a direct caller bypassing the config layer could write files the reader silently treats as data-loss. `MAX_PADDING_BYTES = 1 MiB` constant + shared `_validate_padding_bytes()` helper; same constant referenced by both writer and reader.

### 🐛 Correctness fixes

- **`_apply_overrides` mutated global `self.config`** ([#129](https://github.com/jyunming/Axon/pull/129)) — returned `self.config` by reference when called with no overrides. Route-profile logic in `query()` / `query_stream()` then mutated the shared config via `object.__setattr__`, corrupting brain state across requests (first factual query permanently flipped `self.config.hyde` to `False`). Now always returns a `copy.copy()`.
- **Query cache hit dropped citation / provenance / diagnostics state** ([#129](https://github.com/jyunming/Axon/pull/129)) — `query()` reset `_last_citations` at entry, then cache hits returned the stored response without restoring state. REST `/query` and ReDoc/agent callers reading those attributes after `brain.query()` returned saw empty sources for cached queries. Cache tuple now stores `(time, response, citations, provenance, diagnostics)` and restores all three on hit, with deep-copies on both store and restore so mutations can't reach back.
- **Citation regex missed `[Document N — label]` and `[Document N (ID: ...)]` forms** ([#129](https://github.com/jyunming/Axon/pull/129)) — the exact marker `_build_context` emits, and the form the SYSTEM_PROMPT documents. Dual-alternation pattern that accepts trailing label content only when the `Document` prefix is present (keeps bare-digit form strict to avoid false positives in source-quoted code).
- **`query_stream()` never reset `_last_citations`** ([#129](https://github.com/jyunming/Axon/pull/129)) — stale citations from prior `query()` calls leaked through. Reset at entry; populate `.sources` after retrieval completes.
- **`SentenceVectorStore.search(top_k=0)` returned all rows sorted** ([#123](https://github.com/jyunming/Axon/pull/123)) — `np.argpartition(scores, -0)[-0:]` evaluates to `scores[0:]`, the entire array. Guard added.
- **Cross-encoder rerank crashed the query pipeline on `predict()` failure** ([#123](https://github.com/jyunming/Axon/pull/123)) — OOM, model errors, or a missing `"text"` key escaped to the caller while the LLM rerank path already degraded gracefully. Wrapped predict in `try/except`, switched to `doc.get("text", "")`.
- **`fuse_sparse` mutated caller's `dense_results` metadata** ([#123](https://github.com/jyunming/Axon/pull/123)) — shallow `dict(r)` shared the inner `metadata` reference, so adding `"sparse_score"` leaked into the original list. Deep-copy via a small inline helper.
- **`AxonBrain.switch_project` did not rebind `_doc_versions_path`** ([#127](https://github.com/jyunming/Axon/pull/127)) — the path was set once in `__init__` and never updated. After switching projects, ingest wrote `.doc_versions.json` into the wrong project's directory and `get_doc_versions()` returned stale data. Path now rebinds + reloads in `switch_project()`; in-memory dict cleared in `_switch_to_scope()` to prevent cross-project leakage.
- **`governance.py:_query_jsonl` dropped subsequent rows after one malformed line** ([#125](https://github.com/jyunming/Axon/pull/125)) — `AuditEvent(**d)` lived inside the outer try/except. Per-row catch added; forward-compat columns no longer drop subsequent valid rows.
- **`shares.py:validate_received_shares` TOCTOU + KeyError** ([#125](https://github.com/jyunming/Axon/pull/125)) — `_read_json` happened outside the module lock, so a concurrent `redeem_share_key()` could append a record between the read and the in-lock write, clobbering it. Also `record["mount_name"]` raised `KeyError` on corrupted state and abandoned stale descriptors mid-loop. Lock now spans the whole read-modify-write; switched to `.get()` with a drop-malformed-record path.
- **`config_wizard.py:_pick` crashed on bool / int choices** ([#122](https://github.com/jyunming/Axon/pull/122)) — `AttributeError: 'bool' object has no attribute 'lower'` when reaching the turboquantdb knobs in `--setup full` mode (choices `[False, True]`, `[4, 8]`).
- **`OpenAI` / `Grok` clients cached by `base_url` only** ([#122](https://github.com/jyunming/Axon/pull/122)) — rotating API keys at runtime silently reused stale clients. Gemini already had the correct pattern; OpenAI / Grok now match.
- **`agent.py:_tool_refresh_ingest` used `sha256` while ingest stores `md5`** ([#130](https://github.com/jyunming/Axon/pull/130)) — every comparison failed; every refresh re-ingested every tracked file even when content was unchanged. Wasted embedding + LLM budget on no-op refreshes. Switched to `md5` to match `_doc_versions` (`main.py:2681`). REPL `/refresh` already used `md5` correctly.
- **Dynamic graph backend `_load_snapshot` skipped version validation** ([#128](https://github.com/jyunming/Axon/pull/128)) — a newer-schema snapshot would be silently replayed against v1 SQLite. Added `snapshot_version` check + per-row `isinstance(dict)` + up-front `isinstance(list)` checks for entities/facts.
- **Federated graph backend silently swallowed sub-backend exceptions** ([#128](https://github.com/jyunming/Axon/pull/128)) — ingest / clear / delete_documents hid partial-success state from operators. Now logged at WARNING via a module-level `logger`; four inline `import logging` calls consolidated.

### ✨ Branding

- **VS Code marketplace tile** ([#126](https://github.com/jyunming/Axon/pull/126)) — `integrations/vscode-axon/package.json` now declares `"icon": "media/axon-icon.png"`. 256×256 RGB PNG rendered from the existing `docs/assets/brand/axon-icon.svg` on the brand-dark background (`#050a14`). VSIX rebuilt: `src/axon/extensions/axon-copilot-0.4.2.vsix` is 502 KB / 23 files (was 482 KB / 22 in v0.4.1).
- **FastAPI auto-generated docs** ([#126](https://github.com/jyunming/Axon/pull/126)) — `GET /docs` (Swagger UI) and `GET /redoc` now render with `rel="shortcut icon"` pointing at `/brand/axon-favicon.svg` instead of `fastapi.tiangolo.com/img/favicon.png`. `GET /favicon.ico` 302-redirects to the same canonical SVG so browsers that hammer `/favicon.ico` unprompted stop 404-spamming logs. `/brand/` is mounted as `StaticFiles` from a new in-package directory `src/axon/brand/` — the SVGs ship inside the wheel/sdist (via `[tool.maturin].include`) so `pip install axon-rag` users get the same behaviour as repo developers. All five surfaces (`/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/brand/*`) added to the X-API-Key middleware bypass list so they work when `RAG_API_KEY` is set.

### 📚 Documentation

- 7 staleness fixes ([#132](https://github.com/jyunming/Axon/pull/132)) — `SECURITY.md` Supported Versions table corrected (was claiming `0.9.x`); `CHANGELOG.md` duplicate `[0.4.0]` header consolidated and a broken anchor link redirected; `CONTRIBUTING.md` pre-commit testmon claim updated to the scope-aware selector + nonexistent `setup.py` dropped from the project tree; `docs/DEVELOPMENT.md` "Pre-commit pytest" section rewritten to match `.pre-commit-config.yaml`; `docs/API_REFERENCE.md` + `docs/ADMIN_REFERENCE.md` extended with the new DoS caps and the branded docs / favicon / brand-mount endpoints; `README.md` LangChain section extended to mention `await retriever.aretrieve(query, hyde=True, top_k=8)`.

### ✅ Surface counts (verified, no drift)

- **51 MCP tools** · **73 REST endpoints** · **39 VS Code LM tools** · **8 LLM providers**

## [0.4.1] - 2026-05-19

### ✨ Integrations — `AxonRetriever.aretrieve()` async API

LangChain's `BaseRetriever.ainvoke(input, config, **kwargs)` does not forward extra kwargs to `_aget_relevant_documents`, so async-first callers that want to vary RAG flags per request had to materialise a new retriever via `with_overrides({...}).ainvoke(query)` for every call.

- **`AxonRetriever.aretrieve(query, *, filters=None, **overrides)`** — accepts AxonConfig override flags directly as kwargs (`top_k`, `rerank`, `sentence_window`, `hyde`, `multi_query`, `hybrid_search`, `graph_rag`, etc.) and forwards them to `search_raw` without rebuilding the retriever. Per-call kwargs win over constructor defaults; `self.overrides` / `self.top_k` / `self.filters` are never mutated.
- Existing sync `invoke()` and LangChain's built-in `ainvoke()` → `_aget_relevant_documents` paths are untouched; the async path was already wired but undocumented.
- Module + class docstrings now surface all three call patterns (`invoke`, `ainvoke`, `aretrieve`).
- 6 new tests in `tests/test_v032_integrations.py`: `ainvoke` regression, kwargs-as-overrides, per-call-wins, no-mutation across calls, filters replacement, no-event-loop-blocking under `gather()`.

### 🛠 Tooling — `scripts/bump_version.py` cargo crate name fix

`_refresh_cargo_lock` called `cargo update -p axon` but the crate is named `axon_rust` in `src/axon/Cargo.toml`. Every prior bump silently swallowed the "package ID specification did not match any packages" error and noted "cargo not available", leaving `Cargo.lock` stale. Fixed to `cargo update -p axon_rust`.

## [0.4.0] - 2026-05-04

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

- **Pre-commit pytest now uses pytest-testmon** — selective test runs based on per-file coverage tracking. Typical local commit drops from ~45 min to ~30 s for doc-only changes; source edits to widely-imported modules still take a few minutes. CI is unaffected (full suite still runs on every push). Cache file `.testmondata` is gitignored. (Superseded by `scripts/precommit_pytest_scoped.py` in v0.4.0 — see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#pre-commit-pytest-scope-aware-selector).)

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
