# Axon Capabilities Registry

A curated map of the reusable functions, classes, and shared machinery already
built in `src/axon/` — grouped by subsystem, not an exhaustive function-level
index (well over a thousand functions across 90 files makes a literal index
unmaintainable). First built by a 10-subsystem parallel code audit
(2026-08-28, against `2aa84f6`); every entry re-verified against current source
on 2026-09-28 (against `4238183`, after the 0.5.0 slim-down PRs #154–#169).

**How to use this doc:** before implementing new functionality, skim the
relevant subsystem section below to check whether something close already
exists. If it does, prefer extending it or building on top of it over writing
a parallel implementation — the "Notable Duplication" section right below
exists precisely because that discipline wasn't always followed. If nothing
close exists, or the existing thing is a poor fit, search the code further
(grep, read the file) before writing new code — this doc is a starting point,
not a substitute for verifying against current source. It will drift; when
you notice it's wrong, fix it in the same PR as the code change that made it
wrong. Line numbers are as of the last refresh and shift with every edit — use
them as a hint, search by symbol name.

Companion doc: `src/axon/surface_contract.py`'s `REGISTRY` is the sibling
source of truth for *cross-interface parity* (which surfaces — API/REPL/
CLI/VS Code — support which user-facing capability). This doc is about
internal reuse; `surface_contract.py` is about external surface coverage.
Check both where relevant.

---

## Known Issues

Not duplication — correctness issues found while cataloging. Flagged here so
they aren't lost; fixing them is a separate decision from this doc.

- **`RemoteBrain.ingest()` returns `None`; `AxonBrain.ingest()` returns the
  number of chunks written.** A caller that branches on the count (e.g. "0
  means everything was a duplicate") silently loses that signal when handed a
  `RemoteBrain` (`remote_brain.py`). Proxy-parity gap, not duplication.
- **More config/getattr-fallback drift may be latent in `graph_rag.py`.**
  `graph_defaults.py`'s own docstring records that the group-D audit
  (`59491d2`) found further drifts of the kind the #160–#164 collapse fixed
  (some inverting a boolean) in the still-uncollapsed `graph_rag_*` surface.
  Source-documented, not independently re-verified here.

*Resolved since the 2026-08-28 audit:* `self.llm.generate()` didn't exist, so
LLM route classification and contextual retrieval silently did nothing — both
call sites now use `self.llm.complete()` (#156, `532ab0a`).

---

## Notable Duplication & Consolidation Candidates

Each subsystem section below ends with its own "Possible internal overlap"
notes — full detail lives there. This is a scannable index.

**Cross-subsystem:**
- **Atomic file writes are still hand-rolled in several places.**
  `_atomic_persist.py` now has `write_json_if_changed()`,
  `write_bytes_if_changed()` and `write_text_if_changed()` (the latter two
  added in `2427be1`), used by `graph_rag.py`, `code_graph.py` and all three
  config-reset paths. Still not using them:
  - `security/crypto.py`, `security/master.py`, `security/seal.py`,
    `security/share.py` — hand-rolled tempfile+`os.replace` (some need
    crash-safe two-phase writes the helpers don't support; `2427be1`
    explicitly deferred them).
  - `retrievers.py`'s `BM25Retriever.save()` — the same pattern hand-rolled
    four times inside one method.
  - `shares.py`, `mounts.py`, `projects.py` — `meta.json` / mount descriptor /
    share manifest written with a bare `write_text()`, **no atomicity at all**:
    a crash mid-write leaves a truncated file. The worst case of the three.
- **Private helpers used as public API.** `seal._should_seal` (imported by
  `cache.py` and `share.py`) and `BM25Retriever._ensure_corpus_materialized`
  (called by `AxonBrain.delete_documents` via `getattr`, because
  `BM25Retriever` has no public accessor for its materialized corpus).

**Within one subsystem** (see that section for full detail):
- **Core Engine** — `_switch_to_scope`/`switch_project` duplicate `Multi*`
  fan-out store construction 3 times in `main.py`.
- **Config/Diagnostics** — `is_newer()` vs `is_newer_than()` naming
  collision across two unrelated domains; store-health-checking split across
  `AxonConfig.validate()` and `doctor.check_store_writable()` with no shared
  helper; three independent "render for terminal, degrade gracefully"
  implementations (`config_wizard.py` ×2, `doctor.py`).
- **Retrieval Pipeline** — 5 splitter classes duplicate `transform_documents`
  metadata-assembly boilerplate; two independent sentence segmenters with
  different rules (`splitters.py` vs `sentence_window.py`).
- **Graph RAG** — three independent "build a code structure graph"
  extractors (`CodeGraphMixin`, `DynamicGraphBackend`, `GraphRagMixin`'s
  generic extraction); three independent query-time graph-expansion BFS
  routines with drifted feature sets (the LRU+TTL cache behind one of them is
  now the shared `_lru_ttl_cache.py`, but the traversals are still separate);
  three independent graph→viz-payload builders; two independent
  `nx.Graph`-from-edges builders; `QueryRouterMixin` and `GraphRagMixin` keep
  parallel lazy-init definitions of the same graph lock / token-index
  properties (deliberate, commented).
- **Query Routing & LLM** — `complete()`/`complete_with_tools()`/`stream()`
  in `llm.py` each re-implement the same per-provider message-building
  boilerplate independently (~15 near-duplicate copies across 3 methods ×
  5 OpenAI-dialect providers).
- **Projects/Sharing** — the same defensive "scan a directory, parse each
  JSON, skip on error" loop is duplicated 4 times across
  `list_projects`/`_list_sub_projects`/`list_mount_descriptors`/
  `list_share_mounts`; two independent "is this share still valid" checks
  answer a similar question from different sources of truth with no
  cross-reference; "resolve a project dir against an explicit `user_dir`" now
  exists three ways (`projects.project_dir`, `seal._resolve_project_dir`,
  `project_pack._resolve_project_dir` — the last two deliberately, see their
  docstrings).
- **Security** — the "keyring write + AES-KW-wrapped file fallback" pattern
  is duplicated independently between `master.py` and `share.py`.
- **REST API** — `POST /graph/finalize` vs `POST /governance/graph/rebuild`
  are near-duplicate handlers with drifted response shapes; `POST
  /project/maintenance` vs `POST /governance/project/maintenance` likewise;
  `/config` endpoints are split across `projects.py` and `config_routes.py`
  with no naming cue, and `_reinitialize_runtime_components` duplicates
  (rather than shares) `projects.py`'s inline reinit-on-change logic.
- **User Surfaces** — two independent OpenAI-format tool-schema sets
  (`agent.py::REPL_TOOLS`, `mcp_server.py`) name the same operations
  differently (`add_text` vs. `ingest_text`; `clear_project` vs.
  `clear_knowledge`), have drifted parameters for the same tool
  (`update_settings`), and now differ in coverage (`pack_project`/
  `unpack_project` are MCP-only); `cli.py`/`repl.py` still reimplement project
  CRUD, share generate/redeem/revoke/extend, store lifecycle, refresh and the
  governance proxy end-to-end (only share *listing* is shared now); one
  confirmation prompt (`cli.py`'s `axon update`) still bypasses
  `repl.py::_confirm()`.
- **Infra** — the LangChain and LlamaIndex adapters duplicate the same
  `search_raw`-row-to-framework-document normalization logic.

*Resolved since the 2026-08-28 audit* (so don't re-report them): the third
tool-schema set `tools.py` and `webapp.py`'s separate session store (both
deleted, #157); `fuse_sparse` reimplementing `weighted_score_fusion`
(SPLADE removed, #158); the dead `_build_nx_graph` alias, byte-identical
`can_entity_merge`/`can_merge_entities_into_graph`, the shared
`compute_doc_hash`/`compute_sha256` fallback body, and the 6 hand-rolled
confirmation prompts (`df49862`); `llm.py`'s five unlocked client-factory
caches (`15806a1`); `OpenLLM.ping_local()` vs `doctor.check_local_llm_reachable()`
(`8d3e933`); per-instance LLMLingua loading (`ccf2c16`); two PID-liveness checks
(`_pid_check.py`); per-surface delete logic (`AxonBrain.delete_documents`, #169).

---

## Contents

1. [Core Engine & Runtime](#1-core-engine--runtime)
2. [Config, Setup & Diagnostics](#2-config-setup--diagnostics)
3. [Retrieval & Vector Pipeline](#3-retrieval--vector-pipeline)
4. [Graph RAG & Code Graph](#4-graph-rag--code-graph)
5. [Query Routing & LLM Providers](#5-query-routing--llm-providers)
6. [Projects, Sessions, Sharing & Governance](#6-projects-sessions-sharing--governance)
7. [Security (Sealed Store)](#7-security-sealed-store)
8. [REST API Layer](#8-rest-api-layer)
9. [User-facing Surfaces](#9-user-facing-surfaces)
10. [Infra, Rust Bridge & Integrations](#10-infra-rust-bridge--integrations)

---

## 1. Core Engine & Runtime

### src/axon/main.py
Role: The central orchestrator. Defines `AxonBrain`, which wires together embedding, vector store, BM25, reranker, and LLM, owns project/scope switching (including sealed/encrypted and mounted-share projects), dedup/version tracking, and the full `ingest()` pipeline (chunking, RAPTOR, GraphRAG, code graph). Ends with a "Phase 3/4 re-export" block that re-exports mixins (`CodeGraphMixin`, `CodeRetrievalMixin`, `GraphRagMixin`, `GraphRenderMixin`, `QueryRouterMixin`) and CLI/REPL/session symbols for backward compatibility — a pattern worth knowing before adding new top-level symbols elsewhere in the package.

- `_BloomHashStore(capacity, fp_rate)` — probabilistic (bloom-filter) hash set for RAM-constrained ingest dedup; `.add`/`.update`/`__contains__`/`.discard` — `main.py:117`
- `AxonBrain(config)` — construct the engine; loads config, applies offline/local-assets-only lockdown, resolves model paths, initializes embedding/LLM/vector store/reranker/BM25/splitter/query cache/graph backend/sentence-window index — `main.py:369`
- `AxonBrain.rebuild_vector_store(*, dry_run=False)` — recovery path for a vector store that exists but cannot be opened (unreadable/corrupt/incompatible-format). Re-embeds the active project's chunk text (read from the BM25 corpus, not the vector store) into a fresh store; drops rows whose id collides with one already seen and reports the count rather than aborting; moves the old store dir aside (`.old-<timestamp>`) rather than deleting it, and restores it automatically if the rebuild fails partway (`BaseException`, not `Exception`, so Ctrl+C is covered too); raises if there's no chunk text to rebuild from (an empty rebuild would silently replace a broken store with an empty one) — `main.py:629`
- `AxonBrain._restore_store_backup(vs_path, backup)` (staticmethod) — best-effort "put a moved-aside store back" used by `rebuild_vector_store`'s failure path; never raises, since it runs while another failure is already propagating — `main.py:757`
- `AxonBrain.close()` — releases all resources: shuts down executor, closes vector/BM25 stores + graph backend, force-GCs and wipes sealed plaintext cache. Also usable as context manager (`__enter__`/`__exit__`) — `main.py:790` (enter/exit `main.py:784`/`787`)
- `AxonBrain._resolve_model_path(model_name, kind="hf"|"embedding")` — resolves a bare HF model ID to a local path under configured model roots when offline/local-only — `main.py:240`
- `AxonBrain._preflight_model_audit()` — logs source classification (local/hf_cache/remote) for every model asset; raises `RuntimeError` fast if `local_assets_only` is on but an asset isn't local. The embedding row is classified separately from reranker/gliner/rebel/llmlingua/tokenizer rows when `embedding_provider == "fastembed"`, checking fastembed's own `cache_dir` rather than the HF hub cache layout the other rows use — `main.py:277`
- `AxonBrain.should_recommend_project()` — True iff active project is "default" and no other named projects exist yet (onboarding nudge) — `main.py:865`
- `AxonBrain._switch_to_scope(scope)` — builds a merged, read-only `MultiVectorStore`/`MultiBM25Retriever` view across `@projects`/`@mounts`/`@store`; marks brain read-only — `main.py:887`
- `AxonBrain._is_mounted_share()` — True if active project is a received (always read-only) share mount — `main.py:1033`
- `AxonBrain.refresh_mount()` — public API: re-reads the owner's version marker and reopens project handles (via `switch_project`) if the owner has advanced; raises `MountSyncPendingError` mid-sync — `main.py:1041`
- `AxonBrain._assert_write_allowed(operation="write")` — raises `PermissionError` if current project is read-only (scope/mounted/draining) — `main.py:1113`
- `AxonBrain.clear()` — public API: clears the active project's vector store, BM25 index, hash store and entity graph. Enforces write access itself (`_assert_write_allowed`), then delegates to `collection_ops.clear_active_project()`; also purges the API layer's `_source_hashes` dedup cache for the project. Mirrors `RemoteBrain.clear()` — same verb, same return shape, so callers don't need an `isinstance` branch — `main.py:1124`
- `AxonBrain.delete_documents(doc_ids)` — the single delete implementation behind `POST /delete`, the CLI and the agent tool. Each id may be a chunk id or a whole-document id; a non-chunk id is expanded to every chunk whose `metadata.source`/`metadata.source_id` matches it, including `<id>_p<n>` parent-doc-split children. Beyond removing from the vector store/BM25/graph, it forgets each deleted chunk's dedup hash (via `metadata["dedup_hash"]`, stamped by `ingest()` — see below), the source-level dedup records in `axon.api._source_hashes`, and any `_doc_versions` entry left with no chunks — so the same text can be re-ingested afterwards. Only the active project's own stores are touched; on a parent project, chunks living in a descendant come back in `not_found`. Returns `{"status", "deleted", "doc_ids", "not_found"}` — `main.py:1143`
- `AxonBrain._project_is_sealed(project_root)` — cheap probe for the `.security/.sealed` marker — `main.py:1292`
- `AxonBrain._auto_destroy_expired_share(mount_name, share_key_id, cause)` — on `ShareExpiredError`, wipes the grantee's cached DEK, plaintext cache, and mount descriptor (never touches owner's encrypted source files) — `main.py:1309`
- `AxonBrain._mount_sealed_project(name, project_root, share_key_id=None)` — decrypts a sealed project into an ephemeral plaintext cache (owner path via master-unwrapped DEK, grantee path via keyring DEK); stashes `_sealed_cache`/`_sealed_remount_args` — `main.py:1384`
- `AxonBrain.wipe_sealed_cache()` — public API (also CLI `--wipe-sealed-cache`, REPL `/store wipe-cache`, REST, MCP) to scrub the active sealed plaintext cache; returns whether the wipe actually succeeded — `main.py:1471`
- `AxonBrain._ensure_sealed_cache_mounted()` — re-materialises a wiped sealed cache by replaying the last mount args; called at the start of each query in ephemeral mode — `main.py:1507`
- `AxonBrain._ephemeral_query_window()` — context manager: wraps one query in a sealed mount/unmount cycle when `security.seal_cache_ephemeral` is on (pass-through otherwise) — `main.py:1533`
- `AxonBrain.switch_project(name)` — **the** project-switching orchestrator: handles `@scope` (delegates to `_switch_to_scope`), `mounts/...` (plain and sealed), `default`, and local (plain/sealed) projects; closes and rebuilds vector_store/BM25/graph_backend, reloads dedup/doc-version/code-graph state, builds `Multi*` fan-out over descendants when applicable, bumps write-lease epoch on the old project — `main.py:1562`
- `AxonBrain._load_hash_store()` / `_save_hash_store()` — persist ingest-dedup content hashes; prefers Rust binary format (`.content_hashes.bin`) with plain-text fallback — `main.py:1873` / `1895`
- `AxonBrain._load_doc_versions()` / `_save_doc_versions()` / `get_doc_versions()` — per-source content-hash + chunk-count tracking that powers `/tracked-docs` and smart re-ingest — `main.py:1917` / `1931` / `1942`
- `AxonBrain._embedding_meta_path` (property) / `_load_embedding_meta()` / `_save_embedding_meta()` / `_validate_embedding_meta(on_mismatch="raise"|"warn")` — guards against silently corrupting a collection by mixing embedding models; raises/warns on mismatch between persisted and current embedding provider+model. A mismatch is not reported when the stored and current provider/model normalize to the same underlying weights via `embedding_identity()` (e.g. `sentence_transformers` vs. fastembed's ONNX export of the same `sentence-transformers/...` model) — `main.py:1951` / `1956` / `1969` / `1989`
- `AxonBrain.finalize_ingest()` — flush deferred BM25/code-graph saves and trigger GraphRAG community rebuild; call once after the last `ingest()` under `ingest_batch_mode=True` (safe no-op flush otherwise) — `main.py:2027`
- `AxonBrain._raptor_group_by_structure(chunks, n)` — groups chunks into RAPTOR summarization windows using heading/section detection, falling back to fixed-size windows — `main.py:2047`
- `AxonBrain._generate_raptor_summaries(documents)` — generates RAPTOR summary nodes for already-split leaf chunks; the in-memory summary cache is lock-guarded (`_raptor_cache_lock`) and capped at `config.raptor_summary_cache_size` (oldest-inserted evicted first) — `main.py:2088`
- `AxonBrain._raptor_drilldown(query, results, cfg=None)` — replaces RAPTOR summary hits with their underlying leaf chunks via stored `children_ids` lineage (or filtered search fallback), deduplicated by ID — `main.py:2240`
- `AxonBrain._apply_artifact_ranking(results, cfg=None)` — re-orders retrieval results by artifact type (leaf/raptor/community) per `raptor_retrieval_mode` score multipliers — `main.py:2342`
- `AxonBrain._detect_dataset_type(doc)` — content-based heuristic classifier: codebase/manifest/reference/discussion/knowledge/paper/doc, with `has_code` flag — `main.py:2378`
- `AxonBrain._get_splitter_for_type(dataset_type, has_code, source="")` — returns a type-appropriate text splitter (code-aware, semantic, markdown, recursive) — `main.py:2490`
- `AxonBrain._index_sentence_windows(documents)` — segments eligible chunks into sentences and adds them to the sentence-window secondary index (embeds + persists) — `main.py:2521`
- `AxonBrain._split_with_parents(documents)` — parent-document (small-to-big) chunking: splits into large parent chunks, then type-aware child chunks carrying `parent_text` in metadata — `main.py:2573`
- `AxonBrain._maybe_rust_preprocess_documents(documents)` — optional Rust-accelerated ingest preprocessing with Python fallback, gated on `config.ingest_engine == "rust"` — `main.py:2613`
- `AxonBrain.ingest(documents, progress_callback=None) -> int` — the full ingest pipeline: write-allowed check, write-lease acquire, embedding-meta validation, chunking (type-specific or parent-child), per-source chunk cap, project-namespacing of chunk IDs, dedup (stamps each new chunk's `metadata["dedup_hash"]` so `delete_documents()` can later forget it even after contextual retrieval rewrites the stored text), contextual retrieval, RAPTOR, BM25/vector commit, doc-version + code-graph + GraphRAG updates, version-marker bump. Returns the number of chunks actually written to the vector store/BM25 in this call — `0` if the input was empty or everything was filtered out by dedup, distinct from a nonzero count — `main.py:2634`

**Possible internal overlap**
- `AxonBrain._switch_to_scope()` (`main.py:966-988`), `AxonBrain.switch_project()`'s own-store construction (`main.py:1728-1740`), and `switch_project()`'s descendant-fan-out branch (`main.py:1747-1772`) each independently construct `OpenVectorStore`/`BM25Retriever` per project-dir pair and reassign `self.vector_store`/`self.bm25`/`self._own_vector_store`/`self._own_bm25`; the first and third also each build a `MultiVectorStore`/`MultiBM25Retriever` wrapper over the resulting list. Three sites, same pattern, no shared helper — could be factored into one "build fan-out store" function.
- `AxonBrain.wipe_sealed_cache()` (`main.py:1471`, crypto-layer plaintext wipe), `AxonBrain.clear()` (`main.py:1124`, content-layer full wipe, thin wrapper over `collection_ops.clear_active_project()`), and `collection_ops.clear_active_project()` (`collection_ops.py:15`) all "erase project data" but at different layers — not true duplicates, but three similarly-named "wipe this" entry points is easy to reach for the wrong one from. `AxonBrain.clear()`'s docstring cross-references `RemoteBrain.clear()`; nothing cross-references `wipe_sealed_cache()`.

### src/axon/runtime.py
Role: Process-local, in-memory per-project write-lease registry. Backs "drain mode" maintenance transitions and epoch-based fencing so an in-flight `ingest()` started before a `switch_project()` can detect it became stale and abort its commit.

- `get_registry()` — returns the process-wide `LeaseRegistry` singleton — `runtime.py:274`
- `LeaseRegistry.acquire(project)` — acquire a `_WriteLease`; raises `PermissionError` if the project is draining. `"default"` is always allowed/untracked — `runtime.py:149`
- `LeaseRegistry.start_drain(project)` / `stop_drain(project)` — enter/exit maintenance drain mode (blocks new `acquire()` while draining) — `runtime.py:171` / `188`
- `LeaseRegistry.bump_epoch(project)` — increments the epoch counter on project switch, fencing writes started under the previous activation — `runtime.py:197`
- `LeaseRegistry.wait_for_drain(project, timeout=30.0)` — blocks until active leases hit zero or timeout — `runtime.py:214`
- `LeaseRegistry.active_lease_count(project)` — current in-flight write count — `runtime.py:227`
- `LeaseRegistry.snapshot(project)` / `snapshot_all()` — serialisable status dicts (`project`, `epoch`, `active_leases`, `draining`) for governance/ops endpoints — `runtime.py:235` / `250`
- `LeaseRegistry.reset(project)` — drops tracking state for a project (tests/teardown) — `runtime.py:261`
- `_WriteLease` — lease token returned by `acquire()`; `.close()` (idempotent, also via `__del__`/context-manager), `.is_stale()` (True if epoch advanced since acquisition) — `runtime.py:60`

### src/axon/remote_brain.py
Role: HTTP-backed `RemoteBrain` proxy implementing the subset of `AxonBrain`'s interface the CLI REPL uses, so that surface does NOT construct a second in-process `AxonBrain` on a store an `axon-api` server is already serving (two processes racing on the TurboQuantDB files would crash).

- `get_brain(config, allow_remote=True)` — factory: returns a `RemoteBrain` if `server_client.detect_server` finds a live same-store server, else constructs a local `AxonBrain` — `remote_brain.py:432`
- `RemoteBrain(config, server_info=None)` — proxy object; `.config` is local, `.llm`/`.embedding`/`.reranker` are lazily-constructed local lightweight clients (settable) — `remote_brain.py:68`, `__init__` `remote_brain.py:77`, properties `remote_brain.py:95/107/119`
- `RemoteBrain.switch_project(name)` — POSTs `/project/switch`, updates local `_active_project` — `remote_brain.py:222`
- `RemoteBrain.query(query, filters=None, chat_history=None, overrides=None)` — POSTs `/query` (single-turn only — drops `chat_history`, warning once per instance) — `remote_brain.py:244`
- `RemoteBrain.query_stream(...)` — POSTs `/query/stream`, yields decoded SSE token/sources/error payloads — `remote_brain.py:251`
- `RemoteBrain.ingest(documents, progress_callback=None)` — batch-ingests already-loaded `{id,text,metadata}` dicts via `/add_texts`; returns `None` (does not proxy `AxonBrain.ingest()`'s written-chunk count) — `remote_brain.py:280`
- `RemoteBrain.load_directory(directory)` — async wrapper around `server_client.remote_ingest` for path-based ingest — `remote_brain.py:310`
- `RemoteBrain.clear()` — POSTs `/clear`. Same return shape as `AxonBrain.clear()`; write access is enforced server-side (there is no local vector_store/bm25/graph state in this process to check) — `remote_brain.py:319`
- `RemoteBrain.delete_documents(doc_ids)` — POSTs `/delete`. Same return shape as `AxonBrain.delete_documents()`, write access enforced server-side — `remote_brain.py:330`
- `RemoteBrain.list_documents()` / `get_doc_versions()` — GET `/collection` / `/tracked-docs` — `remote_brain.py:341` / `346`
- `RemoteBrain.finalize_graph(force=False)` / `refresh_mount()` — POST `/graph/finalize` / `/mount/refresh` — `remote_brain.py:354` / `357`
- `RemoteBrain._build_query_body(query, filters, overrides)` — maps local `AxonConfig` RAG toggles to `QueryRequest` override field names (`_OVERRIDE_TO_QUERYREQ` table) — `remote_brain.py:191`
- Unsupported-member pattern: explicit methods raise `NotImplementedError` for un-proxyable public/loud-private members (`vector_store`, `should_recommend_project`, `export_graph_html`, `wipe_sealed_cache`, `_build_system_prompt`, `_resolve_model_path`, `_assert_write_allowed`, `_is_mounted_share`); `__getattr__` raises `AttributeError` for other private/dunder names (so `getattr(brain, "_x", default)` probing degrades gracefully) and `NotImplementedError` for other public names — `remote_brain.py:366-401`
- `_parse_sse_event(event)` — extracts the payload from one `data:`-framed SSE event block — `remote_brain.py:419`

### src/axon/server_client.py
Role: Single-instance detection and thin `urllib`-based HTTP client helpers so the CLI (and `remote_brain.py`) can discover and route through an already-running `axon-api` server, plus a per-store lockfile mechanism that makes `axon-api` itself refuse to run twice against the same store.

- `resolve_api_base(config)` — resolves server base URL: `AXON_API_BASE`/`RAG_API_BASE` env > `config.api_host`/`api_port` (default `127.0.0.1:8420`) — `server_client.py:35`
- `detect_server(config, timeout=1.5)` — probes `GET /health/ready`, then verifies the server's `projects_root` (via `GET /config`) matches this client's before returning its info dict; `None` on any mismatch/failure so callers safely fall back to a local brain — `server_client.py:100`
- `remote_project_new(base, name, headers, graph_backend=None)` / `remote_project_switch(...)` / `remote_project_delete(...)` — routed project-lifecycle ops — `server_client.py:151` / `160` / `164`
- `remote_project_pack(base, name, headers, *, out_path=None)` / `remote_project_unpack(base, zip_path, headers, *, as_name=None, force=False)` — routed project export/import, POST `/project/pack` / `/project/unpack` — `server_client.py:168` / `177`
- `remote_ingest(base, path, headers, on_progress=None, poll_interval=2.0, max_wait_s=3600.0)` — POSTs `/ingest`, polls `/ingest/status/{job_id}` to completion; raises `RuntimeError` on failure/timeout — `server_client.py:192`
- `find_live_server_for_store(config, timeout=1.5)` — reads the per-store lockfile and verifies the recorded server is actually alive (treats stale/dead locks as absent) — `server_client.py:256`
- `write_store_lock(config, host, port, *, force=False) -> bool` — atomically claims this store's lock via exclusive file creation (`O_CREAT|O_EXCL`); returns whether this process actually won it (`False` means a live server already holds it and the caller must not construct a brain). A losing attempt retries once if the contending lock looks stale (recorded pid confirmed dead *and* the recorded server fails `/health/ready` — both, because `AxonBrain` construction runs inside the ASGI lifespan and can take many seconds loading models before the server answers, so a slow-but-legitimate owner must not look dead). `force=True` (set from `AXON_ALLOW_MULTIPLE_SERVERS`) skips all of that and unconditionally overwrites the lock file. Narrows, but does not fully close, the TOCTOU gap against a second racing process — `server_client.py:293`
- `_lock_owner_process_is_alive(lock_path)` — best-effort: is the pid recorded in the lock file still running (via the shared `axon._pid_check.pid_alive()`, also used by `security/cache.py`'s sealed-cache orphan check)? An unreadable/malformed lock file or missing pid is treated as "can't confirm alive" (the safer, non-stealing outcome) — `server_client.py:368`
- `release_store_lock(config)` — unregister this process as the store's server (PID-checked, best-effort) — `server_client.py:395`
- `ServerRequestError(status, detail)` — exception carrying HTTP status for a failed routed operation — `server_client.py:82`

### src/axon/paths.py
Role: Pure, filesystem-I/O-free path-classification predicates identifying storage locations unsafe for SQLite WAL / atomic rename (consumer cloud-sync folders, Windows UNC shares, WSL Windows-drive mounts). Used by SQLite-backed components (governance audit, dynamic graph) to gate WAL usage and redirect hot state to a safe local path.

- `is_cloud_sync_path(p)` — True under OneDrive (Personal/Business)/Dropbox/Google Drive/iCloud Drive folder segments (case-insensitive) — `paths.py:52`
- `is_unc_path(p)` — True for `\\server\share\...` / `//server/share/...` — `paths.py:69`
- `is_wsl_windows_mount_path(p)` — True for `/mnt/<letter>/...` or `//wsl$/...` / `//wsl.localhost/...` — `paths.py:77`
- `cloud_sync_path_reason(p)` — short human-readable reason string, or `""` if safe — `paths.py:93`
- `is_cloud_sync_or_mount_path(p)` — the single union predicate callers should use — `paths.py:109`
- `safe_local_path(p)` — coerces an unsafe path to `~/.axon/<basename>`; returns `Path(p)` unchanged if already safe — `paths.py:117`

### src/axon/collection_ops.py
Role: Small shared helper module (one function) so both the REST API and the REPL wipe a project's entire data (vector store, BM25, dedup hashes, doc versions, graph backend, code graph, RAPTOR cache, embedding meta) through one code path instead of each surface reimplementing the per-backend clear logic. `AxonBrain.clear()` (Core Engine, above) is now the one public wrapper around it.

- `clear_active_project(brain)` — provider-aware vector-store wipe (chroma/qdrant/lancedb branches), BM25 corpus reset, resets `_ingested_hashes`/`_doc_versions`/`_code_graph`/`_raptor_summary_cache` (taken under `brain._raptor_cache_lock` when present, so a reset can't race a background RAPTOR summarization thread), delegates to `brain._graph_backend.clear(persist=True)` (backend-agnostic — correctly handles non-GraphRAG backends like `dynamic_graph`), deletes the embedding-meta sidecar file — `collection_ops.py:15`

---

## 2. Config, Setup & Diagnostics

### src/axon/config.py
Role: Defines `AxonConfig`, the single flat dataclass (~162 fields) that every entry point (CLI, REPL, API, MCP, VS Code ext) constructs and reads. Owns YAML load/save round-tripping (nested YAML → flat dataclass fields and back), env-var overrides, path derivation from the AxonStore layout, and non-raising 3-pass validation.

- `AxonConfig` (dataclass, ~162 fields) — central runtime configuration for embeddings, LLM providers, vector store/TQDB tuning, chunking, hybrid/rerank retrieval, RAPTOR, GraphRAG (33 knobs — the switches, backend choices, and levers TROUBLESHOOTING.md names; internal tuning constants live in `axon/graph_defaults.py` instead — see `_DEMOTED_GRAPH_TUNING` below), offline mode, security/keyring/seal settings, API limits — `config.py:625`
  - `AxonConfig.__post_init__()` — populates API keys/URLs from env vars (OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY, OLLAMA_CLOUD_KEY, VLLM_BASE_URL, etc.), derives `axon_store_base`/`projects_root`/`vector_store_path`/`bm25_path` from the AxonStore layout — `config.py:737`
  - `AxonConfig.load(path=None)` (classmethod) — loads `~/.config/axon/config.yaml` (or an explicit path), auto-writes `_DEFAULT_CONFIG_YAML` on first run (via the shared `_atomic_persist.write_text_if_changed()` helper — see Infra), flattens nested YAML sections onto dataclass field names, applies high-priority env var overrides (OLLAMA_HOST, VLLM_BASE_URL, AXON_PROJECTS_ROOT, OLLAMA_MODELS), logs a warning (via `_REMOVED_FIELDS`) for any key that names a field removed in a past release, never raises on missing/malformed file — `config.py:1202`
  - `AxonConfig.save(path=None)` — persists config back to structured YAML via the shared `_atomic_persist.write_text_if_changed()` helper (atomic rename; skips the write entirely when content is unchanged); hand-maps ~85 renamed fields (e.g. `llm_provider`→`llm.provider`) then a "completion pass" (`_unsaved_field_names()`) dumps every remaining field verbatim under `rag:` so nothing silently reverts to its default; refuses to write into the system temp dir (guards against tests clobbering the real user config) — `config.py:1467`
  - `AxonConfig.validate(path=None)` (classmethod) — non-raising 3-pass validator: (1) structural — unknown YAML keys vs `_KNOWN_YAML_KEYS` unioned with `_derived_yaml_keys()` (see below), with `_REMOVED_FIELDS` keys reported as "no longer valid" (not a typo suggestion) and `difflib` close-match suggestions for genuine typos; (2) semantic — enum/range checks (`chunk_strategy`, `graph_rag_mode`, `graph_rag_depth`, `graph_backend`, `query_router`, `top_k`, `similarity_threshold`, `sentence_window_size`), missing-API-key warnings per provider, a slow-local-LLM+GraphRAG warning, and cloud-sync-unsafe-path warnings via `axon.paths.cloud_sync_path_reason`; (3) store health — AxonStore base/`store_meta.json` existence and env/config-base conflict detection. Returns `list[ConfigIssue]` — `config.py:1658`
- `ConfigIssue(level, section, field, message, suggestion)` (dataclass) — one validation finding; `level` is `"error"|"warn"|"info"` — `config.py:605`
  - `ConfigIssue.to_dict()` — serializes to a plain dict for JSON API responses (`/config/validate`) — `config.py:614`
- `_REMOVED_FIELDS` (dict[str, str]) — maps a config key removed in a past release to a human-readable migration message: the 3 SPLADE fields (`sparse_retrieval`/`sparse_model`/`sparse_weight`), the GraphRAG tuning knobs demoted to constants (see `_DEMOTED_GRAPH_TUNING`), and 3 dead GraphRAG fields deleted outright. Consulted by both `load()` (warns instead of silently dropping the key) and `validate()` (reports "no longer valid", not a typo guess) — `config.py:305`
- `_DEMOTED_GRAPH_TUNING` (tuple[str, ...]) — the GraphRAG context-assembly/community/extraction/cache/parallelism field names moved to constants in `axon/graph_defaults.py`; feeds `_REMOVED_FIELDS`. The switches/backend-choice fields (`graph_rag`, `graph_rag_community`, `graph_rag_relation_backend`, `graph_rag_ner_backend`, etc.) deliberately stay real dataclass fields — only the tuning internals moved — `config.py:320`
- `_derived_yaml_keys(cls, section)` — returns the YAML keys `load()` genuinely accepts for a section, derived directly from the dataclass (the whole dataclass, minus leading-underscore bookkeeping fields, for `rag:`; prefix-stripped field names via `_PREFIXED_SECTIONS` for `embedding:`/`llm:`/`chunk:`) rather than relying solely on the hand-written `_KNOWN_YAML_KEYS`, which drifts independently of the dataclass — `config.py:449`
- `_PREFIXED_SECTIONS` (dict[str, str]) — `{"embedding": "embedding_", "llm": "llm_", "chunk": "chunk_"}`, the prefix table `load()` and `_derived_yaml_keys()` both key off — `config.py:442`
- `_unsaved_field_names()` — returns dataclass fields not covered by `_SAVE_EXPLICIT_FIELDS`/`_SAVE_DERIVED_FIELDS`; drives `save()`'s completion pass and is enforced by `tests/test_config_roundtrip.py`. **Any new config field must either be added to `_SAVE_EXPLICIT_FIELDS` or it flows here automatically** — read before adding a field — `config.py:155`
- `_SAVE_EXPLICIT_FIELDS` — frozenset of field names `save()` writes under a bespoke YAML name/section — `config.py:59`
- `_SAVE_DERIVED_FIELDS` — frozenset of path fields recomputed in `__post_init__` and never persisted (`vector_store_path`, `bm25_path`, `projects_root`, `axon_store_base`) — `config.py:150`
- `_KNOWN_YAML_KEYS` — dict of section → known key set, the hand-written half of `validate()`'s structural (unknown-key) pass — unioned with `_derived_yaml_keys()` rather than being the sole source of truth — `config.py:478`
- `_DEFAULT_CONFIG_YAML` — canonical starter config text; single source of truth used by `axon --config-reset`, REPL `/config reset`, `POST /config/reset`, and the wizard's reset path — `config.py:191`
- `_USER_CONFIG_PATH` — canonical default config file path, `~/.config/axon/config.yaml` (XDG-style, cross-platform) — `config.py:176`
- `DEFAULT_LLM_TIMEOUT` / `DEFAULT_LOCAL_LLM_TIMEOUT` — 60s / 300s soft read-timeout constants used to pick the higher local-provider default only when `llm_timeout` is untouched — `config.py:49-50`

### src/axon/config_wizard.py
Role: Terminal-facing presentation layer for config — an interactive numbered-menu editor (`run_wizard`) plus two rich-table renderers for validation results and current-config display. Never touches disk itself; callers persist `AxonConfig` separately.

- `run_wizard(brain=None, config_path="")` — interactive CLI wizard across 16 sections (LLM, Embedding, Vector Store, Chunking, Retrieval, Sentence-Window/CRAG-Lite, Reranking, Query Transformations, RAPTOR, GraphRAG, Code Graph, Output, Performance, REPL, Offline, Store); three depth modes (`quick`/`standard`/`full`); pre-fills from `brain.config` when given; returns `dict[str, Any]` of changed fields only — caller is responsible for saving. The embedding-provider prompt's default is `fastembed` — `config_wizard.py:153`
- `render_issues(issues: list[ConfigIssue])` — prints a rich-formatted (or plain-text fallback) validation report table for `AxonConfig.validate()` output — `config_wizard.py:21`
- `render_config_table(config)` — prints the current `AxonConfig` grouped into 8 display sections (LLM/Embedding/Vector Store/Chunk/RAG/Store/Offline) as rich tables (or plain fallback) — `config_wizard.py:75`

### src/axon/doctor.py
Role: `axon --doctor` first-run health check — a set of independent, non-destructive `Check` probes (Python version, Ollama reachability, model presence, store writability, local-LLM endpoint, optional extras, PyPI update) aggregated into one `DoctorReport`. Designed for reuse from REPL `/doctor`, the wizard, and future GUIs, not just the CLI.

- `Check(name, status, detail="", hint="")` (dataclass) — single check result; `status` is `"ok"|"warning"|"error"`; `.passed` property — `doctor.py:49`
- `DoctorReport(overall, checks=[])` (dataclass) — aggregate result across all checks — `doctor.py:63`
- `check_python_version()` — compares `sys.version_info` against the `_MIN_PYTHON = (3, 10)` floor (mirrors `pyproject.toml`) — `doctor.py:76`
- `check_ollama_reachable(base_url=None)` — HTTP-probes the Ollama daemon root; falls back to `OLLAMA_HOST` env var / `http://localhost:11434`; warning (not error) since cloud providers don't need Ollama — `doctor.py:95`
- `check_default_model(model_name=None, base_url=None)` — checks the configured LLM model exists in Ollama's local tag cache via `/api/tags` — `doctor.py:125`
- `check_store_writable(store_base=None)` — creates/mkdir's the AxonStore base dir and does an atomic write-probe to confirm write permission — `doctor.py:158`
- `check_optional_extras()` — reports whether `cryptography`/`keyring` (sealed sharing) are installed; nudges toward `axon-rag[starter]` — `doctor.py:183`
- `check_local_llm_reachable(provider, base_url, api_key=None)` — pings a configured OpenAI-compatible `local` provider's `/models` endpoint with optional bearer auth via the shared `axon.llm._probe_openai_compatible_models()` probe (also used by `OpenLLM.ping_local()`); distinguishes "unreachable" (error) from "reachable but no model loaded" (warning) — `doctor.py:217`
- `check_update_available(offline=False)` — wraps `update_check.check_for_update()`, reported at `"ok"` severity even when an update exists or the check fails/opts-out (non-alarming by design) — `doctor.py:274`
- `run_doctor(config=None)` — runs all `_CHECK_FUNCS` (tolerates both flat `AxonConfig` and nested-namespace test fixtures via `getattr` chains), returns the aggregate `DoctorReport` (`overall` = worst of `ok`/`warning`/`error`) — `doctor.py:312`
- `render_report(report, use_color=None)` — renders a `DoctorReport` as a colorized (or plain, honoring `NO_COLOR`) multiline terminal checklist — `doctor.py:378`

### src/axon/update_check.py
Role: Two cleanly-separated concerns — a read-only, rate-limited PyPI version check (`check_for_update`, cached 24h on disk) consumed by startup banners and `doctor.py`; and `run_update()`, the one function here allowed to mutate the environment (detects pip/pipx/conda, shells out to upgrade, then reinstalls the bundled VS Code extension), with hard refusals inside Docker or against a live `axon-api` server.

- `current_version()` — installed `axon-rag` version via `importlib.metadata`, or `"0.0.0+dev"` placeholder — `update_check.py:43`
- `is_newer(latest, current)` — semver-ish comparison using a numeric-prefix parse (not full PEP 440); equal numeric prefixes with differing suffixes are treated as equal — `update_check.py:67`
- `UpdateCheckResult(current, latest, update_available, skipped_reason=None, from_cache=False)` (dataclass) — result of a version check — `update_check.py:118`
- `check_for_update(offline=False, force=False, timeout=5.0, ttl_s=86400)` — queries PyPI JSON API, never raises (network/parse errors become `skipped_reason="error"`), honors a 24h on-disk cache (`~/.axon/.update_check_cache.json`) unless `force=True` — `update_check.py:126`
- `format_suggestion(result)` — one-line "Update available: X → Y · run `axon update`" banner text, or `None` when there's nothing to say — `update_check.py:165`
- `detect_install_method()` — best-effort `"pipx"|"conda"|"pip"` detection via env vars / `sys.prefix` — `update_check.py:179`
- `is_running_in_docker()` — checks `/.dockerenv` and `/proc/1/cgroup` — `update_check.py:189`
- `upgrade_command_for(method)` — returns the subprocess argv for the given install method (always `sys.executable -m pip`, never a bare `pip`, to avoid upgrading the wrong venv) — `update_check.py:202`
- `UpdateRunResult(status, detail, package_before="", package_after="", vsix_status="")` (dataclass) — `status` is `"already_current"|"refused"|"upgraded"|"failed"` — `update_check.py:220`
- `run_update(config=None, force_check=True)` — full `axon update` sequence: refuses in Docker or against a live server for the active store (via `axon.server_client.find_live_server_for_store`), checked before any network/cache activity, else runs the detected upgrade command and reinstalls the VS Code extension in-process — `update_check.py:228`

### src/axon/version_marker.py
Role: Cross-machine staleness detection for shared-filesystem (cloud-sync) Axon mounts. The project owner writes a `version.json` marker (schema v1) atomically after every ingest; grantees read/compare it to detect the owner re-indexed while they were idle, without leaking hostnames. Pure stdlib, safe to import anywhere.

- `MountSyncPendingError(RuntimeError)` — raised when the marker says the owner advanced but the underlying index files haven't fully synced yet (mid-sync race); callers should surface as transient (e.g. HTTP 503) — `version_marker.py:72`
- `VERSION_MARKER_FILENAME` / `SCHEMA_VERSION` / `MANIFEST_FILES` — `"version.json"`, `1`, and the tuple of per-backend manifest/log files whose hashes are rolled up into the marker — `version_marker.py:85,86,94`
- `rollup_hashes(project_dir, *, files=MANIFEST_FILES, hash_algo="sha256")` — hashes each existing manifest file (1 MiB chunked reads), returns `{relpath: hexdigest}`; missing files silently skipped — `version_marker.py:106`
- `bump(project_dir, *, seq=None, hash_algo="sha256", node_id=None)` — computes and atomically persists (`.tmp` + `os.replace`) a fresh marker with an auto-incremented `seq`; call after every successful ingest — `version_marker.py:135`
- `read(project_dir)` — returns the marker dict or `None` if missing/unreadable/malformed — `version_marker.py:244`
- `artifacts_match(project_dir, marker)` — re-hashes on-disk artifacts and compares against the marker's recorded hashes to detect the mid-sync race (marker says seq=N but files are still at seq=N-1); `True` means safe to reopen handles — `version_marker.py:263`
- `is_newer_than(current, cached)` — compares two marker dicts by `seq` (falling back to artifact-dict inequality when `seq` ties) to decide whether `current` represents a later ingest than `cached` — `version_marker.py:283`
- `_atomic_replace(...)` — the crash/OneDrive-safe tempfile+`os.replace` primitive this module's own `bump()` uses; also imported directly by `_atomic_persist.write_json_if_changed()` (Infra) — the one atomic-write primitive that *is* properly shared today. Preserves the destination file's existing permission bits across the swap (`os.chmod` before `os.replace`), so a 0600 sealed-share key file isn't silently widened to the umask default on every rewrite — `version_marker.py` (see Infra section)

### src/axon/logging_setup.py
Role: Small, dependency-free structured logging bootstrap shared by every entry point — idempotent handler installation plus a `contextvars`-based per-request ID that gets stamped into every log line for correlating concurrent API requests.

- `configure_logging(level=logging.INFO)` — idempotently installs one named `StreamHandler` (`rid=<request_id>` format) on the root logger; safe to call repeatedly from multiple entry points; updates level/formatter in place if already installed rather than duplicating — `logging_setup.py:23`
- `RequestIdFilter` (logging.Filter) — injects the current `request_id` contextvar into each `LogRecord` — `logging_setup.py:15`
- `set_request_id(rid)` / `reset_request_id(token)` / `get_request_id()` — get/set/reset the request-id contextvar (used to correlate a single API request's log lines across async/threaded code) — `logging_setup.py:53,58,63`

**Possible internal overlap**
- **`is_newer()` (update_check.py:67) vs `is_newer_than()` (version_marker.py:283)** — both compare "is A newer than B" and share a near-identical name, but operate on unrelated data: `is_newer()` compares dotted PyPI version *strings*, `is_newer_than()` compares mount-marker *dicts* by `seq`/artifact-hash. Not a true duplication (different domains, different logic), but the naming makes them easy to conflate — a docstring cross-reference or more distinct naming (e.g. `marker_is_newer_than`) would reduce confusion for a future engineer grepping for "is_newer".
- **Store health checking is split across two modules with different semantics** — `AxonConfig.validate()`'s store-health pass (`config.py:1957-2056`) does read-only checks (does the base dir exist? is `store_meta.json` present?) while `doctor.check_store_writable()` (`doctor.py:158`) actually performs a write-probe. Both surface overlapping "is my AxonStore OK" information through different call paths (`axon --doctor` vs `axon --validate-config`/`/config validate`) with no shared helper — a future consolidation could have `validate()`'s store pass call `check_store_writable()` (or vice versa) instead of re-deriving the effective store base independently in both places.
- **Three independent "render for terminal, gracefully degrade" implementations** — `render_issues()` and `render_config_table()` (config_wizard.py) both hand-roll a `try: import rich / except ImportError: plain-text` pattern, and `render_report()` (doctor.py) implements a third, ANSI-code-based renderer for the same "checklist with color-coded severity" shape. No shared rendering helper exists between them despite near-identical intent (list of leveled findings → colored terminal output).

---

## 3. Retrieval & Vector Pipeline

### src/axon/retrievers.py
Role: Keyword (BM25) retrieval engine with a lazy-rebuild index, Rust-accelerated backend, msgpack/zstd persistence, corpus dedup, and JSONL delta-log incremental saves; also hosts the two classic dense+lexical rank-fusion algorithms.

- `BM25Retriever(storage_path, engine, rust_fallback_enabled)` — keyword/BM25 index over a document corpus, Python or Rust-backed — `retrievers.py:48`
- `BM25Retriever.close()` — release index/corpus references — `retrievers.py:143`
- `BM25Retriever.add_documents(documents, save_deferred)` — add docs to corpus; index rebuild deferred to next `search()` — `retrievers.py:155`
- `BM25Retriever.flush()` — persist deferred batch, using JSONL delta-log append when below compaction threshold — `retrievers.py:248`
- `BM25Retriever.search(query, top_k)` — tokenize + BM25-score query, rebuilding index if dirty — `retrievers.py:282`
- `BM25Retriever.delete_documents(doc_ids)` — remove docs by id, save immediately — `retrievers.py:344`
- `BM25Retriever.save()` — persist corpus (msgpack+zst preferred, else json/json.zst) — `retrievers.py:364`
- `BM25Retriever.load()` — load corpus trying msgpack.zst → json.zst → msgpack → json, then replay JSONL log — `retrievers.py:603`
- `weighted_score_fusion(vector_results, bm25_results, weight)` — min-max-normalized convex combination of dense+BM25 scores (Rust fast-path) — `retrievers.py:775`
- `reciprocal_rank_fusion(vector_results, bm25_results, k)` — RRF merge of dense+BM25 rankings, preserves original cosine score as `vector_score` — `retrievers.py:817`

### src/axon/vector_store.py
Role: Backend-agnostic vector store facade over Chroma/Qdrant/LanceDB/TurboQuantDB (CRUD + search + async search + index maintenance) that degrades to a warned-empty read rather than crashing or silently overwriting data when an existing store can't be opened (`is_unreadable`), plus read-only multi-project fan-out wrappers used when a merged/parent project view is active.

- `OpenVectorStore(config)` — unified vector store client; lazily opens the configured backend (`chroma`/`qdrant`/`lancedb`/`turboquantdb`) — `vector_store.py:53`
- `OpenVectorStore.is_unreadable` (property) — True when a store exists on disk but this process failed to open it (unreadable on-disk format, corruption) — distinct from `client is None` ("no store yet, create on first `add()`"); `search`/`batch_search`/`get_by_ids`/`list_documents`/`optimize_index` degrade to empty+warn when true, `add`/`delete_by_ids` raise `RuntimeError` instead of risking data loss — `vector_store.py:67`
- `OpenVectorStore._warn_unreadable(operation)` — logs why a degraded read came back empty, pointing the operator at `axon --rebuild-vector-store` — `vector_store.py:77`
- `OpenVectorStore.close()` — release backend client/handles — `vector_store.py:248`
- `OpenVectorStore.search_async(query_embedding, top_k, filter_dict, query_text)` — async search; uses `tqdb.aio.AsyncDatabase` for TurboQuantDB, else thread-pool offload — `vector_store.py:297`
- `OpenVectorStore.add(ids, texts, embeddings, metadatas)` — batched insert with per-backend batching/limits and length validation — `vector_store.py:375`
- `OpenVectorStore.list_documents()` — enumerate unique sources with chunk counts and doc_ids — `vector_store.py:568`
- `OpenVectorStore.search(query_embedding, top_k, filter_dict, query_text)` — kNN search; `query_text` enables TQDB-side hybrid BM25+dense RRF — `vector_store.py:630`
- `OpenVectorStore.batch_search(query_embeddings, top_k, filter_dict, query_texts)` — multi-query search (native batch for Chroma, thread pool otherwise) — `vector_store.py:735`
- `OpenVectorStore.get_by_ids(ids)` — fetch docs by exact id (score=1.0), used by GraphRAG expansion — `vector_store.py:789`
- `OpenVectorStore.optimize_index()` — build/rebuild ANN index (LanceDB IVF_PQ, TurboQuantDB HNSW); no-op message for Chroma/Qdrant — `vector_store.py:877`
- `OpenVectorStore.delete_by_ids(ids)` — delete docs by id — `vector_store.py:942`
- `MultiVectorStore(stores)` — read-only fan-out search/list across descendant project stores, merged by score — `vector_store.py:987`
- `MultiVectorStore.search(...)` / `.batch_search(...)` / `.list_documents()` / `.get_by_ids(ids)` — parallel fan-out + merge over child stores — `vector_store.py:1001,1026,1066,1078`
- `MultiVectorStore.add/delete_by_ids/delete_documents(...)` — raise `RuntimeError` (writes unsupported on merged parent view) — `vector_store.py:1063,1085,1088`
- `MultiBM25Retriever(retrievers)` — read-only fan-out over multiple `BM25Retriever`s, merged by score — `vector_store.py:1092`
- `MultiBM25Retriever.search(query, top_k)` — parallel fan-out + merge — `vector_store.py:1106`
- `MultiBM25Retriever.close()` / `.add_documents(...)` / `.delete_documents(doc_ids)` — `close()` releases sub-retrievers; `add_documents`/`delete_documents` raise `RuntimeError` (writes unsupported on merged parent view) — `vector_store.py:1101,1128,1124`

### src/axon/embeddings.py
Role: Unified embedding client abstracting sentence-transformers/Ollama/FastEmbed/OpenAI providers — `fastembed` (ONNX runtime, no torch import) is the default provider as of 0.5.0, cutting cold start from ~20s to ~2s; `sentence_transformers` moved to the optional `[sentence-transformers]` extra. Also hosts a known-dimension registry, HF-hub/fastembed cache-presence probes so callers can pass `local_files_only` and skip redundant network freshness checks, a cross-provider embedding-identity canonicalizer, and retry-with-backoff for network providers.

- `OpenEmbedding(config)` — loads/configures the embedding model per `config.embedding_provider`; resolves `dimension` via `_KNOWN_DIMS` registry or model probe — `embeddings.py:180`
- `OpenEmbedding.embed(texts)` — batch-embed texts (provider-specific call); honors `AXON_DRY_RUN` — `embeddings.py:303`
- `OpenEmbedding.embed_query(query)` — embed a single query string — `embeddings.py:345`
- `_retry_call(fn, attempts, base_delay, max_delay)` — exponential-backoff+jitter retry wrapper for provider network calls (module-private but generic/reusable pattern) — `embeddings.py:21`
- `_KNOWN_DIMS` — dict mapping known embedding model names → vector dimension, avoids a model download just to learn dims — `embeddings.py:59`
- `is_hf_model_cached(model_id, guess_st_prefix=True)` — checks the local HF hub cache dir for a model id, optionally guessing the `sentence-transformers/` org prefix for bare short names; used by `rerank.py`'s cross-encoder loader and `main.py`'s cache-status check — `embeddings.py:85`
- `fastembed_default_cache_dir(axon_store_base)` — stable fastembed cache path under the AxonStore root, replacing fastembed's own default (the OS temp dir, which gets swept and would re-trigger downloads) — `embeddings.py:108`
- `is_fastembed_model_cached(model_id, cache_dir)` — resolves fastembed's public catalog id to its actual backing HF repo (they often differ, e.g. the shipped default) before checking the cache dir — `embeddings.py:124`
- `embedding_identity(provider, model)` — canonicalizes `(provider, model)` so numerically-identical embeddings (verified via cosine similarity — currently just `all-MiniLM-L6-v2` across sentence_transformers/fastembed) compare equal across providers, so switching providers on an existing project doesn't false-positive as a mismatch — `embeddings.py:159`

### src/axon/rerank.py
Role: Post-retrieval document reranker; supports a local cross-encoder model (now behind the optional `[sentence-transformers]` extra) or a pointwise LLM-based scorer, with graceful failure fallback to unranked input.

- `OpenReranker(config)` — loads cross-encoder (`sentence_transformers.CrossEncoder`, using `embeddings.is_hf_model_cached` to pass `local_files_only` when already cached) or wires an `OpenLLM` for pointwise LLM reranking, per `config.reranker_provider` — `rerank.py:23`
- `OpenReranker.rerank(query, documents)` — reranks documents (cross-encoder pair scoring, or LLM 1–10 pointwise scoring in parallel threads via `_llm_rerank`); returns input unchanged on failure or if disabled — `rerank.py:64`

### src/axon/splitters.py
Role: Chunking/text-splitting strategies used at ingest time — semantic (token-budget), recursive-character, markdown-aware, embedding-based cosine-semantic, table-row, and AST/regex code-aware splitters — each exposing `split()`/`transform_documents()`.

- `SemanticTextSplitter(chunk_size, chunk_overlap, encoding_name)` — sentence-boundary-respecting chunking to a tiktoken token budget, with overlap — `splitters.py:61`
- `SemanticTextSplitter.split(text)` / `.transform_documents(documents)` — `splitters.py:99,142`
- `TableSplitter(table_name, batch_size)` — converts tabular rows into enriched "Context/Columns/Data" natural-language strings for embedding — `splitters.py:162`
- `TableSplitter.transform_rows(rows, headers)` — row-dicts → searchable strings, batched — `splitters.py:172`
- `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` — character-length chunking that recursively tries `\n\n`/`\n`/` ` separators — `splitters.py:193`
- `RecursiveCharacterTextSplitter.split(text)` / `.transform_documents(documents)` — `splitters.py:212,244`
- `MarkdownSplitter(chunk_size, chunk_overlap)` — splits on ATX heading boundaries, recursively sub-splits oversized sections via `SemanticTextSplitter` — `splitters.py:264`
- `MarkdownSplitter.split(text)` / `.transform_documents(documents)` — `splitters.py:275,301`
- `CosineSemanticSplitter(embed_fn, breakpoint_threshold, max_chunk_size, encoding_name)` — embedding-based semantic chunking; new chunk starts when adjacent-sentence cosine similarity drops below threshold — `splitters.py:320`
- `CosineSemanticSplitter.split(text)` / `.transform_documents(documents)` — falls back to `SemanticTextSplitter` on embedding failure — `splitters.py:372,403`
- `CodeAwareSplitter(max_symbol_size, fallback_chunk_size, fallback_overlap)` — syntax-aware code chunker: Python `ast`-based symbol extraction, regex boundary detection for Go/Rust/JS/TS/Ruby/Bash/Perl/Julia/Java/C++/Kotlin/C#/Scala/Swift, else character fallback — `splitters.py:422`
- `CodeAwareSplitter.LANGUAGE_MAP` — file-extension → language-name registry driving splitter dispatch — `splitters.py:435`
- `CodeAwareSplitter.split_code(text, source)` — returns symbol-aware chunks with rich metadata (`symbol_type`, `qualified_name`, `signature`, line ranges, `is_entrypoint`, `is_test`, etc.) — `splitters.py:809`
- `CodeAwareSplitter.transform_documents(documents)` — `splitters.py:855`
- `_split_sentences(text)` — abbreviation-aware (Mr./Dr./e.g./etc.) sentence splitter used by `SemanticTextSplitter`/`CosineSemanticSplitter` (module-private but reusable) — `splitters.py:11`

### src/axon/sentence_window.py
Role: Sentence-window retrieval — a secondary, finer-grained index that maps individual sentences back to their parent chunk so retrieval hits can be expanded into a coherent ±N-sentence context window while citations still resolve to the stable parent `chunk_id`.

- `is_eligible(chunk)` — filters out code / RAPTOR-summary / parent-marker chunks from sentence indexing — `sentence_window.py:40`
- `segment_text(text)` — regex (Rust-bridge-accelerated) sentence segmentation with short-fragment merging — `sentence_window.py:69`
- `SentenceRecord(sentence_id, chunk_id, source, sentence_idx, total_sentences, text)` — one indexed sentence — `sentence_window.py:104`
- `segment_chunk(chunk)` — turns an eligible chunk dict into a list of `SentenceRecord` — `sentence_window.py:124`
- `SentenceWindowIndex()` — sentence-id → record and chunk-id → sentence-id-list linkage store — `sentence_window.py:154`
- `SentenceWindowIndex.add_records(records)` — register sentence records for one chunk — `sentence_window.py:173`
- `SentenceWindowIndex.get_record(sentence_id)` / `.get_all_for_chunk(chunk_id)` — lookups — `sentence_window.py:187,194`
- `SentenceWindowIndex.get_window(sentence_id, window_size)` — reconstruct a ±N-sentence context window around a hit — `sentence_window.py:199`
- `SentenceWindowIndex.save(directory)` / `.load(directory)` — msgpack (Rust) or JSON persistence — `sentence_window.py:227,255`
- `SentenceVectorStore(directory)` — numpy-backed float32 sentence-embedding matrix for cosine search, no external vector DB needed — `sentence_window.py:288`
- `SentenceVectorStore.add(ids, embeddings, metadatas)` — append sentence embeddings — `sentence_window.py:310`
- `SentenceVectorStore.search(query_embedding, top_k)` — cosine-similarity top-k via single matrix–vector product — `sentence_window.py:337`
- `SentenceVectorStore.save()` / `.load()` — persist `.npy` matrix + msgpack/JSON metadata — `sentence_window.py:381,408`

### src/axon/compression.py
Role: Post-retrieval context-compression gateway that shrinks chunk text before it's fed to the generation LLM, with a defined fallback chain (`llmlingua` → `sentence` → original) and telemetry on the compression achieved.

- `CompressionResult(chunks, strategy_used, pre_tokens, post_tokens, compression_ratio, fallback_reason)` — output/telemetry contract of a compression call — `compression.py:60`
- `ContextCompressor(llm, llmlingua_model)` — compression gateway; instantiate once per query — `compression.py:80`
- `ContextCompressor.compress(query, chunks, strategy, token_budget)` — dispatches to `"none"`/`"sentence"`/`"llmlingua"` strategy, with automatic fallback to `"sentence"` (or `"none"`) if LLMLingua is unavailable/fails — `compression.py:107`

### src/axon/crag.py
Role: CRAG-Lite corrective-retrieval layer — deterministic, LLM-free heuristics that score retrieval-result quality and decide whether to trust local results or escalate to web fallback, with a full diagnostic audit trail.

- `RetrievalConfidence(score, verdict, factors, fallback_recommended)` — weighted 5-signal confidence assessment (result count, top score, score spread, source diversity, threshold pass rate) — `crag.py:62`
- `RetrievalConfidence.to_dict()` — `crag.py:77`
- `assess_confidence(filtered_results, total_candidates, similarity_threshold)` — computes the `RetrievalConfidence` for a result set — `crag.py:86`
- `CorrectionDecision(trust_local, trigger_web_fallback, reason)` — policy decision output — `crag.py:162`
- `CorrectionDecision.to_dict()` — `crag.py:176`
- `evaluate_correction_policy(confidence, has_local_results, truth_grounding_enabled, crag_lite_threshold)` — decides trust-local vs. web-fallback from a confidence assessment — `crag.py:184`

### src/axon/loaders.py
Role: One `BaseLoader` subclass per source/file type (24 loaders), all dispatched by extension through `DirectoryLoader` for directory-crawl ingest; includes a hardened URL loader (SSRF-checked, redirect-chain validated) and shared HTML/ID-generation utilities.

- `BaseLoader.load(path)` / `.aload(path)` — sync + async (via `asyncio.to_thread`) loader interface all loaders implement — `loaders.py:104,107`
- `_stable_file_id(path, kind)` — SHA-256-based stable doc id from absolute path, used by nearly every loader — `loaders.py:19`
- `_extract_html_text(html)` — stdlib `html.parser`-based visible-text extractor, reused by HTML/URL/EML/MSG/EPUB loaders — `loaders.py:68`
- `_rewrite_github_url(url)` — rewrites GitHub blob/gist URLs to raw-fetchable form; raises on `tree` (directory) URLs — `loaders.py:34`
- `FlexibleTableLoader` — CSV/TSV with delimiter sniffing + ragged-row handling — `loaders.py:112` (`.load`/`.load_text`)
- `SmartTextLoader` — auto-detects table vs. prose text and delegates accordingly — `loaders.py:202`
- `TextLoader` — plain text file loader — `loaders.py:243`
- `CodeFileLoader` — source code loader; tags `type="code"` so downstream skips path-prefix enrichment (keeps AST chunking valid) — `loaders.py:259`
- `TSVLoader` — tab-delimited file → `TableSplitter` rows — `loaders.py:284`
- `JSONLoader` — JSON (single object or list) → documents with sanitized metadata — `loaders.py:316`
- `CSVLoader` — CSV → `TableSplitter` rows — `loaders.py:360`
- `HTMLLoader` — HTML file → extracted visible text — `loaders.py:392`
- `URLLoader` — HTTP(S) fetch with SSRF protection (blocked-network checks re-applied per redirect hop, ≤5 hops), content-type/size validation, GitHub URL rewriting — `loaders.py:409` (`.load(url)` at `:459`, `._check_ssrf` at `:433`)
- `DOCXLoader` — python-docx paragraph extraction — `loaders.py:532`
- `PPTXLoader` — python-pptx shape-text extraction — `loaders.py:554`
- `ImageLoader` (alias `BMPLoader`) — Ollama VLM image captioning (Pillow-normalized to PNG) — `loaders.py:580`
- `CustomImageLoader(vision_fn)` — pluggable custom vision-function image captioning — `loaders.py:656`
- `PDFLoader` — PyMuPDF (fitz) page-by-page extraction with pypdf fallback — `loaders.py:682`
- `JSONLLoader` — newline-delimited JSON (.jsonl/.ndjson), one doc per line — `loaders.py:742`
- `NotebookLoader` — Jupyter `.ipynb` markdown/code cell extraction — `loaders.py:773`
- `ExcelLoader` — multi-sheet `.xlsx`/`.xls` via pandas → `TableSplitter` rows — `loaders.py:810`
- `ParquetLoader` — pandas+pyarrow Parquet → `TableSplitter` rows — `loaders.py:861`
- `EPUBLoader` — ebooklib chapter/item extraction — `loaders.py:894`
- `RTFLoader` — striprtf-based RTF → text — `loaders.py:932`
- `XMLLoader` — structured tag-walk text extraction, falls back to regex tag-stripping on parse error — `loaders.py:958`
- `SQLLoader` — splits `.sql` files into per-statement docs with preceding comments as context — `loaders.py:1010`
- `EMLLoader` — stdlib `email` module `.eml` parsing (From/To/Subject/Date + body, prefers text/plain) — `loaders.py:1072`
- `MSGLoader` — Outlook `.msg` via extract-msg — `loaders.py:1129`
- `LaTeXLoader` — strips LaTeX commands/math/comments, keeps prose (discards equations) — `loaders.py:1174`
- `DirectoryLoader(vlm_model, vision_fn)` — extension → loader dispatch registry (`self.loaders`), directory crawl, path-prefix "breadcrumb" enrichment — `loaders.py:1239`
- `DirectoryLoader.load(directory)` / `.aload(directory)` — sync crawl vs. async crawl (semaphore-capped at 32 concurrent) — `loaders.py:1310,1338`

**Possible internal overlap**
- **Duplicated `transform_documents` boilerplate** — `SemanticTextSplitter`, `RecursiveCharacterTextSplitter`, `MarkdownSplitter`, `CosineSemanticSplitter`, and `CodeAwareSplitter` (`splitters.py:142,244,301,403,855`) each hand-roll nearly identical chunk-metadata assembly (`source_id`, `subdoc_locator`, `chunk_index`, `chunk_kind`, id suffixing). A shared `_build_chunk_metadata()`/mixin would remove ~5x copy-pasted logic. Still present — `splitters.py` had zero commits between 2aa84f6 and current `main`.
- **Two independent sentence segmenters** — `splitters.py:_split_sentences` (`splitters.py:11`, abbreviation-aware, used by `SemanticTextSplitter`/`CosineSemanticSplitter`) vs. `sentence_window.py:segment_text` (`sentence_window.py:69`, simpler regex + Rust-bridge fast path, short-fragment merging). Both solve "split prose into sentences" but with different rules and no shared implementation. Still present — neither file changed in the audited range.
- **`BM25Retriever.save()` hand-rolls tempfile+`os.replace` atomic writes independently, 4 separate times in one method** (`retrievers.py:376-379` msgpack.zst, `:397-400` msgpack, `:421-424` json.zst, `:435-453` json-with-shutil-fallback), instead of calling the shared `_atomic_persist.write_bytes_if_changed()` / `write_text_if_changed()` helpers that already exist for exactly this shape of write and are used elsewhere (Graph RAG, Config). This is the same "atomic file writes reimplemented independently" pattern flagged cross-subsystem for the Security module, but `retrievers.py` isn't in that list — it should be added to it.
- **`BM25Retriever._ensure_corpus_materialized()` is effectively public API reached through `getattr`** — `main.py`'s document-delete path (`main.py:1194-1202`) needs the raw in-memory corpus to expand document ids into chunk ids, and does so by calling the underscore-prefixed `_ensure_corpus_materialized()` via `getattr(bm25, "_ensure_corpus_materialized", None)` + a callable check, then reads `bm25.corpus` directly, rather than `BM25Retriever` exposing a public accessor. Same shape as the `seal._should_seal` leaky-underscore case already flagged in the Security section.

---

## 4. Graph RAG & Code Graph

Covers: `src/axon/graph_rag.py`, `src/axon/graph_defaults.py`, `src/axon/graph_render.py`, `src/axon/graph_backends/*.py`, `src/axon/dynamic_graph/models.py`, `src/axon/code_graph.py`, `src/axon/code_retrieval.py`.

### `src/axon/graph_rag.py` — `GraphRagMixin`

**Role:** The GraphRAG engine proper — Microsoft-GraphRAG-style entity/relation extraction, hierarchical community detection (Leiden/Louvain), LLM community summarization, and local-search/global-search (map-reduce) retrieval, plus all on-disk persistence for that state. ~4360 lines, the largest file in the repo. Not mixed into `AxonBrain` directly — composed into `GraphRagEngine` (`graph_backends/graphrag_engine.py`), which `GraphRagBackend` wraps behind the `GraphBackend` Protocol. Many `_`-prefixed methods here are the de-facto public API other layers call (via `GraphRagEngine`/`GraphRagBackend`), so they're catalogued despite the leading underscore. Since v0.5.0's GraphRAG config collapse (#160-#164), most tuning knobs formerly on `AxonConfig` are read from `graph_defaults` (imported as `_gd`) instead of `getattr(self.config, ...)` — see that module below.

#### Security — pickle-cache integrity
- `_get_or_create_relation_pickle_hmac_key()` — per-machine HMAC key at `~/.axon/.relation_pickle_hmac.key` (mode 0600) — `graph_rag.py:46`
- `_compute_relation_pickle_hmac(payload, cache_key)` — HMAC-SHA256 binding the relation-graph pickle cache to this host; verified before any `pickle.load` (prevents RCE from a tampered/synced pickle) — `graph_rag.py:76`

#### Shared-model lazy init
- `_ensure_shared_model(obj, instance_attr, shared_dict, shared_lock, cache_key, loader)` — module-level: lazy-init a class-level, cross-instance-shared local ML model. Fast path reads `obj`'s instance attribute lock-free; a cold instance falls through to the shared, lock-guarded dict keyed by `cache_key` so multiple `AxonBrain`/`GraphRagEngine` instances in one process don't each hold a separate copy of identical model weights, and concurrent first-callers converge on one load. Deliberately module-level (not a `GraphRagMixin` method) because `_ensure_gliner`/`_ensure_rebel`/`_ensure_llmlingua` are exercised in tests via the unbound `GraphRagEngine._ensure_gliner(brain)` pattern where `brain` may not itself inherit `GraphRagMixin`. Extracted from three near-identical hand-rolled implementations — see `_ensure_llmlingua`/`_ensure_gliner`/`_ensure_rebel` below, all three now delegate to it — `graph_rag.py:116`

#### Locking, lazy state & background persistence infra
- `_graph_lock` (property) — `RLock` guarding all entity/relation/community graph state — `graph_rag.py:157`
- `_gr_cache_lock` (property) — leaf-level lock for the LLM/extraction cache; strictly never nested under `_graph_lock` (deadlock-avoidance) — `graph_rag.py:165`
- `_entity_token_index` / `_rebuild_entity_token_index()` / `_token_index_add(eid)` / `_token_index_remove(eid)` — inverted token→entity-name index so query-entity matching scans candidates instead of all `|V|` nodes — `graph_rag.py:179,224,237,245`
- `_traversal_cache` (+ `_traversal_cache_lock` / `_traversal_cache_maxsize` / `_traversal_cache_ttl`) — LRU+TTL (512 entries / 15 min) cache of BFS multi-hop traversal results, keyed by matched-entity set + hop params; the LRU+TTL bookkeeping itself is now shared with `query_router.py`'s query-response cache via `axon._lru_ttl_cache.lru_ttl_get`/`lru_ttl_put` (see Query Routing §5) — `graph_rag.py:194-219`
- `_reset_graph_state()` — single source of truth for wiping all in-memory graph state (used by `GraphRagBackend.clear()` and read-only scope switching) — `graph_rag.py:257`
- `_track_persist_future(future)` / `_flush_pending_saves()` — register and block-until-complete for background graph-persist writes — `graph_rag.py:292,302`
- `_persist_executor` (property) — dedicated single-worker `ThreadPoolExecutor` that serializes all graph persist I/O (entity/relation/claims) — `graph_rag.py:323`

#### LLM / extraction caching
- `_gr_cache_get(bucket, key)` / `_gr_cache_put(bucket, key, value)` — capped, bucketed cache for entity/relation/LLM outputs (entities, relations, `llm:*`, `global_answer`, `global_map`) — `graph_rag.py:503,508`
- `_load_graph_rag_extraction_cache()` / `_save_graph_rag_extraction_cache()` — msgpack (Rust codec) with JSON fallback persistence of the extraction cache — `graph_rag.py:438,461`
- `_graph_rag_entity_cache_key(text)` / `_graph_rag_relation_cache_key(text)` — cache-key builders (depth + backend + text-hash) — `graph_rag.py:646,651`
- `_extract_graph_llm_batches(chunks_to_process, ...)` — batched/parallel entity+relation extraction pipeline: cache-hit skipping, relation budgeting/selection, fused single-call extraction dispatch — `graph_rag.py:656`
- `_gr_llm_complete_cached(bucket, prompt, system_prompt=None, **kwargs)` — LLM completion wrapper with a semantic response cache keyed by prompt+options — `graph_rag.py:802`
- `_gr_write_json_if_changed(path, payload)` / `_gr_write_bytes_if_changed(path, payload)` / `_gr_json_load_path(path)` — atomic, digest-gated, cloud-sync-safe JSON/bytes persistence primitives (orjson-aware read) — `graph_rag.py:825,842,877`

#### Persistence (entity / relation / community / embedding / claims graphs)
- `_get_incoming_relation_index()` / `_get_incoming_relation_count_map()` — cached reverse-edge (incoming-relation) index, persisted to `.relation_graph.incoming.json` for fast startup — `graph_rag.py:906,967`
- `_normalize_entity_graph(raw)` (static) — apply defaults, drop malformed entries — `graph_rag.py:987`
- `_build_extracted_chunk_ids()` — set of chunk IDs already present in the entity graph, from `entity_data["chunk_ids"]` across `_entity_graph` — used to skip already-extracted chunks on ingest — `graph_rag.py:1001`
- `_load_entity_graph()` / `_save_entity_graph()` (+ `_do_save_entity_graph` background worker) — persist `entity → {description, type, chunk_ids, frequency, degree}` (msgpack, JSON fallback) — `graph_rag.py:1009,1043,1060`
- `_normalize_relation_graph(raw)` (static) — `graph_rag.py:1088`
- `_load_relation_graph()` / `_save_relation_graph()` (+ `_do_save_relation_graph`) — sharded/msgpack relation-graph persistence with an HMAC-verified pickle fast-cache and parallel shard load; sharding/msgpack/pickle/worker-count knobs now come from `graph_defaults` (`RELATION_SHARD_*`, `RELATION_MSGPACK_PERSIST`, `RELATION_PICKLE_CACHE*`) — `graph_rag.py:1133,1316,1361`
- `_load_community_levels()` / `_save_community_levels()` — `{level: {entity: community_id}}` — `graph_rag.py:1554,1571`
- `_load_community_hierarchy()` / `_save_community_hierarchy()` — `{cluster_id: parent_cluster_id}` — `graph_rag.py:1583,1615`
- `_load_community_summaries()` / `_save_community_summaries()` — LLM-generated per-community reports (compact-key persisted) — `graph_rag.py:1629,1653`
- `_load_entity_embeddings()` / `_save_entity_embeddings()` — `graph_rag.py:1675,1702`
- `_load_claims_graph()` / `_save_claims_graph()` — `{chunk_id: [claim, ...]}` — `graph_rag.py:1728,1745`
- `_build_synthetic_community_hierarchy(community_levels)` (static) — derive parent/child hierarchy from independently-clustered per-level community maps (Louvain multi-resolution fallback path) — `graph_rag.py:1761`

#### Graph construction / connected components
- `_graph_connected_components(nodes, edges)` (static) — union-find style connected-components over a generic edge list — `graph_rag.py:548`
- `_build_graph_edge_payload()` — build `(nodes, edges)` from entity+relation graphs, Rust-accelerated, applies the `graph_rag_entity_min_frequency` filter — `graph_rag.py:574`
- `_build_networkx_graph_from_edges(nodes, edges)` (static) — generic weighted `nx.Graph` builder with a Dijkstra `distance` edge attribute — `graph_rag.py:629`
- `_build_networkx_graph()` — dirty-flag-cached `nx.Graph` for the whole entity/relation graph — `graph_rag.py:1794` (its old `_build_nx_graph()` one-line test-only alias was removed in #df49862 — see Known Issues in the report below)

#### Community detection
- `_run_community_detection()` — single-level Louvain (Rust bridge, else `networkx.algorithms.community`) — `graph_rag.py:1806`
- `_run_hierarchical_community_detection()` — multi-level detection with a fallback chain: graspologic hierarchical Leiden → leidenalg (multi-resolution CPM) → multi-resolution Louvain; cluster-size/seed/use-lcc knobs now come from `graph_defaults` (`COMMUNITY_MAX_CLUSTER_SIZE`, `LEIDEN_SEED`, `COMMUNITY_USE_LCC`); returns `(community_levels, hierarchy, children)` — `graph_rag.py:1838`
- `_rebuild_communities()` — orchestrates alias resolution → detection → summary generation → vector-store indexing, with a skip-if-unchanged signature guard — `graph_rag.py:2081`
- `finalize_graph(force=False)` — **public** entry point to trigger a community rebuild (use after batch ingest with `graph_rag_community_defer=True`) — `graph_rag.py:2136`

#### Community summarization & global (map-reduce) search
- `_generate_community_summaries(query_hint="")` — LLM-generated per-community reports, processed finest→coarsest so parent summaries can cite child reports; budget-capped (triage by size/rank/query-relevance, now via `graph_defaults.COMMUNITY_LLM_TOP_N_PER_LEVEL`/`COMMUNITY_LLM_MAX_TOTAL`) with template fallback for uncapped communities — `graph_rag.py:2290`
- `_index_community_reports_in_vector_store()` — index community reports as synthetic `__community__*` documents so they're retrievable like any chunk — `graph_rag.py:2564`
- `_global_search_map_reduce(query, cfg)` — GraphRAG global search: chunk community reports → parallel map phase (LLM key-point extraction, cached, optionally batched N-per-call via `graph_defaults.MAP_BATCH_SIZE`) → top-k point heap → reduce phase (LLM synthesis) → cached answer — `graph_rag.py:2606`

#### Local search / query-time context assembly
- `_get_incoming_relations(entity)` — all relation entries where `entity` is the target — `graph_rag.py:2991`
- `_local_search_context(query, matched_entities, cfg)` — unified joint-ranked local-search context builder: scores entities/relations/communities/text-units/claims onto one weighted scale (weights now `graph_defaults.LOCAL_ENTITY_WEIGHT`/`LOCAL_RELATION_WEIGHT`/`LOCAL_COMMUNITY_WEIGHT`/`LOCAL_TEXT_UNIT_WEIGHT`), greedy-fills a shared token budget instead of a fixed per-type split — `graph_rag.py:3013`
- `_entity_matches(q_entity, g_entity)` — Jaccard-based fuzzy entity-name match score (0.0–1.0) — `graph_rag.py:3325`
- `_classify_query_needs_graphrag(query, mode)` — heuristic-keyword or LLM classifier deciding whether a query warrants global (holistic) vs local GraphRAG search — `graph_rag.py:3365`

#### Entity/relation extraction backends
- `_ensure_llmlingua()` / `_ensure_gliner()` / `_ensure_rebel()` — lazy, cross-instance-shared model loaders (LLMLingua-2 compressor, GLiNER NER, REBEL relation-extraction pipeline), all three now built on the shared `_ensure_shared_model()` helper above — `graph_rag.py:3395,3427,3463`
- `_parse_rebel_output(text)` (static) — parse REBEL's `<triplet>…<subj>…<obj>` token format into relation dicts — `graph_rag.py:3503`
- `_extract_relations_rebel(text)` — no-LLM structured relation extraction via REBEL — `graph_rag.py:3550`
- `_extract_entities_gliner(text)` — no-LLM NER via GLiNER — `graph_rag.py:3597`
- `_extract_entities_light(text)` — regex noun-phrase heuristic, zero-model "light" tier — `graph_rag.py:3634`
- `_parse_extracted_entities(raw)` / `_normalize_extracted_entities_payload(payload)` — parse pipe-delimited or structured-JSON entity output — `graph_rag.py:3654,3691`
- `_parse_extracted_relations(raw)` / `_normalize_extracted_relations_payload(payload)` — parse pipe-delimited or structured-JSON relation output — `graph_rag.py:3723,3767`
- `_extract_entities_and_relations_combined(text)` — single fused LLM call returning both entities and relations as one JSON bundle (perf optimization over two calls) — `graph_rag.py:3799`
- `_extract_entities(text)` / `_extract_relations(text)` — top-level dispatchers: cache check → depth-tier ("light") short-circuit → backend selection (`gliner`/`rebel`/`llm`) → LLM prompt fallback — `graph_rag.py:3887,3938`

#### Entity embedding matching
- `_embed_entities(entity_keys)` — embed entity `"name: description"` strings, store in `_entity_embeddings` — `graph_rag.py:3987`
- `_match_entities_by_embedding(query, top_k=5)` — cosine-similarity semantic entity match (complements exact/Jaccard matching) — `graph_rag.py:4008`

#### Claims extraction
- `_extract_claims(text)` — LLM extraction of factual claims (subject/object/type/status/dates/source-quote) — `graph_rag.py:4036`

#### Entity/relation resolution & canonicalization
- `_resolve_entity_aliases()` — merge near-duplicate entity nodes via cosine similarity on name embeddings (Rust union-find or numpy O(n²) fallback, backend/ceiling/threshold now `graph_defaults.ENTITY_RESOLVE_*`); returns count merged — `graph_rag.py:4098`
- `_canonicalize_entity_descriptions()` — LLM-synthesize one comprehensive description from multiple per-entity descriptions collected across chunks — `graph_rag.py:4228`
- `_canonicalize_relation_descriptions()` — same synthesis for repeated `(subject, object)` relation pairs — `graph_rag.py:4275`

#### Pruning
- `_prune_entity_graph(deleted_ids)` — remove deleted chunk IDs from entity/relation/claims graphs, deleting now-empty nodes, persisting changes — `graph_rag.py:4320`

#### Visualization payload
- `build_graph_payload()` — **public** renderer-neutral `{nodes, links}` payload built from `_entity_graph`/`_relation_graph`/`_community_levels`, including per-node vector-store-backed `evidence` (chunk excerpts) and HTML tooltips — `graph_rag.py:2163`

### `src/axon/graph_defaults.py`

**Role:** NEW in v0.5.0 (#160–#164). Internal tuning constants for GraphRAG that used to be `AxonConfig` dataclass fields — none shipped in `config.yaml`, none was reachable from `/config/get`/`/config/update`, and `AxonConfig.validate()` rejected most of them as unknown keys even while `load()` quietly accepted them (a user who set one got a working override *and* a "did you mean X" warning pointing at an unrelated real field). Demoting 76 fields here (112→33 net across four PRs) closed that inconsistency and shrank the settable config surface without changing any default behavior — every constant here takes the value the dataclass field defaulted to, not the (sometimes drifted) `getattr(..., fallback)` value call sites used to read. `graph_rag.py` and `graphrag_engine.py` import it as `from axon import graph_defaults as _gd` and read constants directly (`_gd.LOCAL_ENTITY_WEIGHT`, etc.) instead of `getattr(self.config, "graph_rag_...", default)`. If a value ever needs to be user-settable again, the module's own docstring is explicit: promote it back to a real `AxonConfig` field with a `_KNOWN_YAML_KEYS` entry, don't reach for it via `getattr`.

Grouped by section (module docstring documents the full rationale per group):
- **Caches, parallelism & profiling** — `EXTRACTION_CACHE(_SIZE)`, `LLM_CACHE(_SIZE)`, `LLM_FUSED_EXTRACTION`, `REBUILD_SKIP_IF_UNCHANGED`, `PROFILE`, `MAP_BATCH_SIZE`, `MAP_USE_DEDICATED_POOL`, `MAP_AUTO_WORKERS`, `REPORT_COMPRESS_RATIO`, `RUST_BUILD_EDGES`, `RUST_MERGE_ENTITIES`, `LARGE_GRAPH_THRESHOLD` — `graph_defaults.py:43-77`
- **Entity/relation extraction internals** — `CANONICALIZE_MIN_OCCURRENCES`, `CANONICALIZE_RELATIONS_MIN_OCCURRENCES`, `ENTITY_EMBEDDING_MATCH`, `ENTITY_MATCH_THRESHOLD`, `EXACT_ENTITY_BOOST`, `ENTITY_RESOLVE_BACKEND`/`_MAX`/`_THRESHOLD` — `graph_defaults.py:90-106`
- **Relation-graph persistence** — sharding/serialization: `RELATION_COMPACT_PERSIST`, `RELATION_MSGPACK_PERSIST`, `RELATION_PICKLE_CACHE(_PROTOCOL)`, `RELATION_SHARD_PERSIST`/`_COUNT`/`_LIST_MANIFEST`/`_SELECTIVE_REWRITE`/`_PARALLEL_LOAD`/`_LOAD_WORKERS`/`_PARALLEL_SIGNATURES`/`_SIGNATURE_WORKERS`/`_PARALLEL_WRITES`/`_WRITE_WORKERS` — `graph_defaults.py:115-141`
- **Community detection & summarization shape** — `COMMUNITY_MAX_CLUSTER_SIZE`, `COMMUNITY_USE_LCC`, `LEIDEN_SEED`, `COMMUNITY_MIN_SIZE`, `COMMUNITY_LLM_TOP_N_PER_LEVEL`, `COMMUNITY_LLM_MAX_TOTAL`, `COMMUNITY_MAX_CONTEXT_TOKENS`, `COMMUNITY_INCLUDE_CLAIMS`, `COMMUNITY_LEVEL`, `INDEX_COMMUNITY_REPORTS`, `COMMUNITY_SUMMARY_COMPACT_PERSIST`, `COMMUNITY_REBUILD_DEBOUNCE_S` — `graph_defaults.py:152-178`
- **Global search (map-reduce)** — `GLOBAL_MIN_SCORE`, `GLOBAL_TOP_POINTS`, `GLOBAL_REDUCE_MAX_TOKENS`, `GLOBAL_MAP_MAX_LENGTH`, `GLOBAL_REDUCE_MAX_LENGTH`, `GLOBAL_ALLOW_GENERAL_KNOWLEDGE`, `GLOBAL_MAX_MAP_CHUNKS`, `GLOBAL_REDUCE_SKIP_IF_TOP_POINTS_LE`/`_TOP_SCORE_GTE`, `GLOBAL_ANSWER_CACHE(_SIZE)`, `GLOBAL_MAP_CACHE(_SIZE)` — `graph_defaults.py:183-215`
- **Local search context assembly** — `LOCAL_MAX_CONTEXT_TOKENS`, `LOCAL_TOP_K_ENTITIES`/`_RELATIONSHIPS`, `LOCAL_INCLUDE_RELATIONSHIP_WEIGHT`, `LOCAL_ENTITY_WEIGHT`/`_RELATION_WEIGHT`/`_COMMUNITY_WEIGHT`/`_TEXT_UNIT_WEIGHT`, plus performance-path toggles `LOCAL_BATCH_FETCH`, `LOCAL_CACHED_INCOMING(_COUNTS)`, `LOCAL_EARLY_CUTOFF(_FACTOR)`, `LOCAL_ENTITY_DEGREE_FAST`, `LOCAL_RELATION_SUPPORT_FAST` — `graph_defaults.py:220-250`

Kept as real `AxonConfig` fields (deliberately not moved here): the on/off switches and backend choices an operator actually reaches for — `graph_rag_relations`, `_relation_backend`, `_ner_backend`, `_relation_budget`, `_entity_min_frequency`, `_canonicalize`, `_canonicalize_relations`, `_claims`, `_entity_resolve`, `_min_entities_for_relations`, `graph_rag_community` (on/off) + `_async`/`_backend`/`_defer`/`_lazy`/`_levels`, and model-id fields (rewritten in place with a resolved local path under `local_assets_only` by `main.py`, which a module constant can't carry per-brain).

### `src/axon/graph_render.py` — `GraphRenderMixin`

**Role:** Turns graph payloads (from `build_graph_payload()` / `build_code_graph_payload()` / a `GraphBackend.graph_data()`) into standalone, self-contained HTML visualizations — a 3D force-graph viewer and a richer query+graph explorer page shared with the VS Code extension.

- `build_code_graph_payload()` — code-structure graph (`_code_graph`, populated by `CodeGraphMixin`) → `{nodes, links}` payload with `file`/`class`/`function`/`method`/`module` node types and `CONTAINS`/`IMPORTS` edges — `graph_render.py:29`
- `_render_graph_html(graph)` (static) — render a `{nodes, links}` payload as a self-contained 3D viewer (three.js + 3d-force-graph via CDN), with per-node evidence panel — `graph_render.py:76`
- `render_query_graph_html(query, answer, sources, kg, cg)` (static) — **public** rich standalone HTML page combining query/answer/citations with tabbed KG + code-graph panes; reuses the VS Code extension's `graph-panel.js` when found on disk, falls back to a minimal inline renderer otherwise — `graph_render.py:218`
- `_resolve_graph_payload()` — fetch the active `_graph_backend`'s `graph_data()` payload (mirrors `GET /graph/data` precedence); as of #c7ee0e0 (v0.4.4 review-fix batch) a `graph_data()` failure is logged via `logger.warning(..., exc_info=True)` before falling back to an empty payload, instead of being silently swallowed — every export surface (CLI `--graph-export`, REPL `/graph export`, REST `GET /graph/export`) used to be unable to distinguish a genuinely empty graph from a broken one — `graph_render.py:467`
- `export_graph_html(path=None, json_path=None, open_browser=True)` — **public** entry point: resolve payload → render → optionally write files and open the default browser — `graph_render.py:497`

### `src/axon/graph_backends/base.py`

**Role:** Defines the stable `GraphBackend` Protocol and its shared data types — the seam every graph backend (`graphrag`, `dynamic_graph`, `federated`, `none`) implements against, so callers (query router, REST/MCP graph routes) never branch on backend type.

- `GraphContext` (dataclass) — one retrieved context item: `context_id`, `context_type` ("entity"/"relation"/"community"/"fact"), `text`, `score`, `rank`, `backend_id`, temporal `valid_at`/`invalid_at`, `evidence_ids`, `matched_entity_names`, multi-hop `hop_count`/`path` — `base.py:20`
- `IngestResult` (dataclass) — `entities_added`, `relations_added`, `chunks_processed`, `backend_id` — `base.py:45`
- `FinalizationResult` (dataclass) — `communities_built`, `time_elapsed`, `status` ("ok"/"error"/"not_applicable"), `detail` — `base.py:55`
- `GraphDataFilters` (dataclass) — `entity_types`, `min_degree`, `limit` filters for `graph_data()` — `base.py:73`
- `RetrievalConfig` (dataclass) — `top_k`, `point_in_time`, `federation_weights` (consumed only by `FederatedGraphBackend`) — `base.py:82`
- `GraphPayload` (dataclass + `to_dict()`) — renderer-neutral `{nodes, links}` container — `base.py:95`
- `GraphBackend` (runtime-checkable `Protocol`) — the pluggable graph-strategy interface: `ingest(chunks)`, `retrieve(query, cfg, existing_results)`, `finalize(force)`, `clear(persist=False)`, `delete_documents(chunk_ids)`, `status()`, `graph_data(filters)`, `has_entities()`, `has_community_summaries()`; optional capability `list_conflicts` (hasattr-guarded) — `base.py:137`

### `src/axon/graph_backends/factory.py`

**Role:** Config-driven constructor for `GraphBackend` instances — the single place new backend types get registered.

- `get_graph_backend(brain)` — **public**; reads `brain.config.graph_backend` ("graphrag" default / "dynamic_graph" / "federated" / "none") and instantiates the matching `GraphBackend` — `factory.py:48`
- `_registry()` — lazily populates and returns the `_BACKEND_REGISTRY` dict (config string → backend class), importing each backend module only on first call to avoid import-time circular deps — `factory.py:31`

### `src/axon/graph_backends/none_backend.py`

**Role:** Explicit no-op `GraphBackend` for projects with `graph_backend: "none"` — every method returns an empty/inert result, nothing persisted.

- `NoneGraphBackend` — full `GraphBackend` Protocol implementation, all no-ops — `none_backend.py:24`

### `src/axon/graph_backends/graphrag_backend.py`

**Role:** Adapts a composed `GraphRagEngine` (which owns all GraphRAG state via `GraphRagMixin`) to the `GraphBackend` Protocol, plus exposes GraphRAG-specific extras that `query_router.py`/`main.py` call directly (hasattr-guarded) beyond the shared Protocol surface.

- `GraphRagBackend(brain)` — `graphrag_backend.py:31`
  - `load()` — reload all GraphRAG state from disk (called by `switch_project()`) — `graphrag_backend.py:47`
  - `merge_descendants(descendants)` — merge sub-project graph state into this project's in-memory graph — `graphrag_backend.py:51`
  - `expand_with_entity_graph(query, results, cfg)` — delegates to `GraphRagEngine.expand_with_entity_graph` — `graphrag_backend.py:55`
  - `local_search_context(query, matched_entities, cfg)` — delegates to `GraphRagMixin._local_search_context` — `graphrag_backend.py:63`
  - `global_search_map_reduce(query, cfg)` — delegates to `GraphRagMixin._global_search_map_reduce` — `graphrag_backend.py:67`
  - `classify_query_needs_graphrag(query, mode)` — delegates to `GraphRagMixin._classify_query_needs_graphrag` — `graphrag_backend.py:71`
  - `ensure_community_summaries(query_hint, index_community_reports=True)` — lazy, double-checked-locking summary build on first global query — `graphrag_backend.py:78`
  - `flush_ingest_saves()` — persist entity/relation/claims graphs + extraction cache after a batch-mode ingest — `graphrag_backend.py:100`
  - `flush()` — flush dirty cache + pending background persists without shutting down — `graphrag_backend.py:125`
  - `close()` — flush then shut down the engine's persist executor — `graphrag_backend.py:138`
  - `ingest(chunks)` / `retrieve(query, cfg, existing_results)` / `finalize(force)` / `clear(persist)` / `delete_documents(chunk_ids)` / `status()` / `graph_data(filters)` / `has_entities()` / `has_community_summaries()` — full `GraphBackend` Protocol implementation — `graphrag_backend.py:152-292`

### `src/axon/graph_backends/graphrag_engine.py`

**Role:** Ownership-inversion class — composes `GraphRagMixin` so `AxonBrain` no longer inherits it directly. Owns all GraphRAG graph state on `self`; proxies brain-owned resources (LLM, embedding, vector store, executor, source-policy tables) back via properties. `GraphRagBackend` holds one instance as `self._engine`. Imports `graph_defaults` as `_gd` (e.g. `_gd.ENTITY_EMBEDDING_MATCH`, `_gd.LARGE_GRAPH_THRESHOLD`) and the shared `axon._lru_ttl_cache.lru_ttl_get`/`lru_ttl_put` for the traversal cache.

- `GraphRagEngine(brain)` (extends `GraphRagMixin`) — `graphrag_engine.py:39`
  - `load()` — (re)load all `_load_*` GraphRAG state from disk; resets transient state (traversal cache, in-progress flags) — `graphrag_engine.py:122`
  - `merge_descendants(descendants)` — merge entity/relation/embedding/claims/community-summary state from descendant projects' own graph files into this project's in-memory state — `graphrag_engine.py:159`
  - `ingest_chunks(documents)` — the full GraphRAG ingest pipeline: RAPTOR-aware chunk eligibility filtering, source-policy filtering, cross-restart dedup (via `_build_extracted_chunk_ids`), batched extraction (`_extract_graph_llm_batches`), entity/relation graph merge (Rust-accelerated), claims extraction, canonicalization, and community-rebuild scheduling (sync/async/deferred) — `graphrag_engine.py:310`
  - `expand_with_entity_graph(query, results, cfg)` — query-time expansion: extract query entities (LLM + embedding union) → token-index candidate matching → Jaccard scoring → multi-hop BFS traversal over the relation graph (hop-decay scored, capped at 1 hop above `graph_defaults.LARGE_GRAPH_THRESHOLD` entities, cached via `lru_ttl_get`/`lru_ttl_put`) → fetch and tag not-yet-present chunks. Docstring notes it was "moved verbatim from `QueryRouterMixin._expand_with_entity_graph()`" — which is why `QueryRouterMixin` still carries fallback `_graph_lock`/`_traversal_cache_lock`/`_entity_token_index` properties for when `GraphRagMixin` isn't in the brain's MRO (see §5) — `graphrag_engine.py:774`

### `src/axon/graph_backends/dynamic_graph_backend.py`

**Role:** Alternative `GraphBackend` implementation — a SQLite bi-temporal fact graph (Graphiti-style: episodes/entities/facts/fact_evidence) instead of GraphRAG's in-memory dict graph. Supports point-in-time queries and explicit fact supersession/conflict tracking. Share-mount safe: DELETE journal mode + owner DB relocated off cloud-synced paths + JSON snapshot export for grantees.

- `_sha8(text)` / `_now_iso()` / `_norm(name)` — deterministic-ID, UTC-timestamp, and name-normalization helpers — `dynamic_graph_backend.py:154,159,163`
- `_is_code_chunk(chunk)` — detect source-code chunks by metadata/extension — `dynamic_graph_backend.py:201`
- `_extract_python_entities_and_facts(text)` — `ast`-based (no-LLM) extraction of classes/functions/imports and `INHERITS`/`IMPORTS` facts from Python source — `dynamic_graph_backend.py:216`
- `DynamicGraphBackend(brain)` — `dynamic_graph_backend.py:292`
  - `_resolve_db_path(base)` / `_maybe_migrate_legacy_db(base)` — relocate the owner's SQLite file off cloud-sync/network paths to `~/.axon/graphs/<id>/`, one-shot-migrating any pre-existing DB — `dynamic_graph_backend.py:343,383`
  - `_export_snapshot()` / `_load_snapshot()` — owner writes a compact JSON snapshot after ingest; grantees load it into an in-memory SQLite instead of touching the owner's DB file — `dynamic_graph_backend.py:435,495`
  - `_extract_entities(text)` / `_extract_facts(text)` — LLM pipe-delimited extraction (own prompt copy, same wire format as `GraphRagMixin`) — `dynamic_graph_backend.py:616,638`
  - `_upsert_entity(name, entity_type, description, now)` — `dynamic_graph_backend.py:671`
  - `_upsert_fact(subject, relation, obj, description, confidence, chunk_id, episode_id, now)` — bi-temporal insert with exclusive-relation (`_EXCLUSIVE_RELATIONS`) supersession/conflict detection (same-timestamp contradictions marked `conflicted`) — `dynamic_graph_backend.py:692`
  - `ingest(chunks)` — extract entities/facts per chunk, store as episodes — `dynamic_graph_backend.py:780`
  - `retrieve(query, cfg, existing_results)` — query-term matching against facts, optional point-in-time filtering, multi-hop BFS traversal with SQLite `IN`-clause chunking, returns ranked `GraphContext` list — `dynamic_graph_backend.py:846`
  - `_build_nx_graph_from_db()` — build a cached (TTL) `nx.Graph` from all active facts (used in tests) — `dynamic_graph_backend.py:1004`
  - `finalize(force)` — no-op (`status="not_applicable"`); dynamic graph has no community-detection step — `dynamic_graph_backend.py:1035`
  - `list_conflicts(limit=100)` — return facts marked `status='conflicted'` for UI/agent resolution — `dynamic_graph_backend.py:1043`
  - `clear(persist)` / `delete_documents(chunk_ids)` / `status()` / `has_entities()` / `has_community_summaries()` / `graph_data(filters)` / `close()` — remaining `GraphBackend` Protocol surface — `dynamic_graph_backend.py:1079-1234`

### `src/axon/graph_backends/federated_backend.py`

**Role:** Composite `GraphBackend` that runs `GraphRagBackend` + `DynamicGraphBackend` concurrently and fuses their retrieval results with weighted Reciprocal Rank Fusion, so both static-corpus (GraphRAG) and temporal-fact (dynamic) retrieval contribute to the same query. As of #7126dbb (Sept 2026) this backend has its own dedicated test suite (37 tests) covering fusion math, per-query weight overrides, and fault isolation — previously untested despite being load-bearing.

- `_weighted_rrf(per_backend, weights, k=60)` — fuse per-backend `GraphContext` lists via weighted RRF, dedup by `context_id` then by `source_chunk_id` (keeping the higher-scored duplicate) — `federated_backend.py:36`
- `FederatedGraphBackend(brain)` — `federated_backend.py:91`
  - `retrieve(query, cfg, existing_results)` — concurrent fan-out (`ThreadPoolExecutor`) to both sub-backends, per-query weight override via `cfg.federation_weights`, then `_weighted_rrf` — `federated_backend.py:122`
  - `ingest(chunks)` / `finalize(force)` / `list_conflicts(limit)` / `clear(persist)` / `delete_documents(chunk_ids)` / `close()` — sequential delegation to each sub-backend, each failure logged and isolated (partial-failure surfaced via `status="error"` on `finalize`) — `federated_backend.py:158-269`
  - `status()` / `graph_data(filters)` — merge sub-backend results — `federated_backend.py:274,283`
  - `has_entities()` / `has_community_summaries()` — true if any sub-backend is true — `federated_backend.py:298,314`
  - `_graphrag_sub_backend()` — locate the wrapped `graphrag`-typed sub-backend by `BACKEND_ID`, or `None` — `federated_backend.py:334`
  - `expand_with_entity_graph` / `local_search_context` / `global_search_map_reduce` / `classify_query_needs_graphrag` / `ensure_community_summaries` — GraphRAG-specific bridge methods delegated to the wrapped `graphrag` sub-backend via `_graphrag_sub_backend()` (hasattr-guarded by callers) — `federated_backend.py:340-363`

### `src/axon/dynamic_graph/models.py`

**Role:** Pydantic schema for `DynamicGraphBackend`'s bi-temporal SQLite model (Graphiti-style Episode/Entity/Fact/FactEvidence) — used for validation/typing rather than as the live storage layer (the backend talks SQL directly).

- `Episode` — one ingested chunk as the atomic unit of temporal knowledge (`episode_id`, `source_chunk_id`, `content`, `reference_time`) — `models.py:29`
- `Entity` — canonical named entity persisting across episodes (`entity_id`, `canonical_name`, `entity_type`, `first_seen_at`/`last_seen_at`) — `models.py:39`
- `Fact` (+ `is_active` property) — bi-temporal `(subject, relation, object)` with `valid_at`/`invalid_at`/`status` — `models.py:51`
- `FactEvidence` — join table linking a `Fact` to the chunk/episode that asserts it — `models.py:73`
- `RelationRegistry(extra=None)` — registry of known relation types (12 defaults: `WORKS_FOR`, `KNOWS`, …) with `register(relation)`, `__contains__`, `all()` — `models.py:102`

### `src/axon/code_graph.py` — `CodeGraphMixin`

**Role:** Builds and persists a lightweight structural code graph (files, symbols, `CONTAINS`/`IMPORTS` edges) from the metadata a code-aware splitter already attaches to chunks — no AST parsing or LLM calls of its own. Feeds `GraphRenderMixin.build_code_graph_payload()` and `CodeRetrievalMixin._expand_with_code_graph()`.

- `_load_code_graph()` / `_save_code_graph()` — atomic, digest-gated, cloud-sync-safe JSON persistence of `{nodes, edges}` (shares `_atomic_persist.write_json_if_changed` with `GraphRagMixin`) — `code_graph.py:12,27`
- `_build_code_graph_from_chunks(chunks)` — build/update File and Symbol nodes plus `CONTAINS` (file→symbol) and `IMPORTS` (file→file, two-pass so forward references resolve) edges from `source_class == "code"` chunk metadata — `code_graph.py:48`
- `_resolve_import_to_file(stmt)` — resolve an `import`/`from...import` statement string to a file `node_id` already in the graph — `code_graph.py:149`

### `src/axon/code_retrieval.py` — `CodeRetrievalMixin`

**Role:** Code-aware retrieval layer sitting alongside the generic BM25/vector pipeline — identifier-aware query tokenization, lexical re-scoring/boosting of code chunks, a dedicated symbol-name search channel, code↔prose linking, and code-graph-based result expansion. Also defines a versioned diagnostics/trace contract for retrieval-quality benchmarking.

- `_extract_code_query_tokens(query)` — identifier tokenizer: CamelCase/snake_case splitting, dotted-path parts, filename stems, Rust-accelerated — `code_retrieval.py:33`
- `_looks_like_code_query(query)` — heuristic: does this query look like it's about code identifiers/files — `code_retrieval.py:76`
- `_build_code_bm25_queries(query, query_tokens)` — expand identifier tokens into spaced-word BM25 sub-queries (CamelCase/snake_case/dotted-path/basename variants) — `code_retrieval.py:96`
- `_classify_retrieval_failure(results, query_tokens, expected_symbol=None)` — CI-benchmark failure-mode labeling (`exact_symbol_missed`, `right_file_wrong_block`, `too_many_broad_chunks`, `fallback_chunk_involved`) — `code_retrieval.py:311`
- `CodeRetrievalDiagnostics` (dataclass, versioned `diagnostics_version`, `to_dict()`/`to_json()`) — stable external contract for code-retrieval observability, returned via `include_diagnostics=True` API responses — `code_retrieval.py:230`
- `CodeRetrievalTrace` (dataclass, `to_dict()`) — internal debug trace (per-result score breakdown, channel counts, diversity-cap deferrals) — `code_retrieval.py:289`
- `CodeRetrievalMixin` — `code_retrieval.py:365`
  - `_build_code_doc_bridge(prose_chunks)` — add `MENTIONED_IN` edges from code symbol/file nodes to prose chunks that reference them by name (Rust-accelerated) — `code_retrieval.py:372`
  - `_get_symbol_cache(corpora)` — build/reuse cached `(flat_docs, entries, exact-lookup)` structures over BM25 corpora for repeated symbol-channel queries — `code_retrieval.py:421`
  - `_apply_code_lexical_boost(results, query_tokens, cfg, diagnostics, trace)` — re-score `source_class == "code"` results by exact/partial symbol-name, basename, qualified-name, and text-token hits; blends with original score, then enforces a per-file diversity cap — `code_retrieval.py:452`
  - `_symbol_channel_search(query_tokens, top_k, filters=None)` — dedicated `symbol_name`/`qualified_name` exact+partial retrieval channel (Rust symbol index or Python fallback), handles both single- and multi-project BM25 retrievers — `code_retrieval.py:588`
  - `_expand_with_code_graph(query, results, cfg)` — match query tokens against code-graph node names, 1-hop traverse `CONTAINS`/`IMPORTS`/`MENTIONED_IN` edges, fetch and tag not-yet-present chunks — `code_retrieval.py:735`

**Possible internal overlap**

1. **Three independent "build a code structure graph" extractors.** `CodeGraphMixin._build_code_graph_from_chunks` (metadata-driven `CONTAINS`/`IMPORTS`, `code_graph.py:48`), `DynamicGraphBackend._extract_python_entities_and_facts` (raw `ast`-walk producing `INHERITS`/`IMPORTS` facts, `dynamic_graph_backend.py:216`), and `GraphRagMixin`'s generic LLM/GLiNER/REBEL entity-relation extraction (which also runs over code chunks unless `source_policy` excludes them) all derive "what relates to what" graphs from the same source files, into three unrelated stores (`_code_graph`, the dynamic-graph SQLite DB, `_entity_graph`/`_relation_graph`). Still present — unchanged since the original audit. A future code-structure feature should check all three before adding a fourth extractor; `DynamicGraphBackend` in particular re-derives `IMPORTS` facts via AST when `CodeGraphMixin` already computes the same edge type from chunk metadata.

2. **Query-time graph expansion reimplemented three times.** `GraphRagEngine.expand_with_entity_graph` (BFS over the entity/relation graph with hop-decay scoring + an LRU+TTL cache, `graphrag_engine.py:774`), `DynamicGraphBackend.retrieve` (BFS over the facts table, no cache, `dynamic_graph_backend.py:846`), and `CodeRetrievalMixin._expand_with_code_graph` (1-hop only, no cache, `code_retrieval.py:735`) all follow the same shape — match query tokens/entities against graph nodes, traverse edges N hops, fetch chunks not already in `results`, tag and merge back — with separate, drifted implementations. The BFS/traversal logic itself is still triplicated; what *was* fixed (#8eb1d7f) is only the LRU+TTL cache algorithm backing `expand_with_entity_graph`'s traversal cache, which now shares `axon._lru_ttl_cache.lru_ttl_get`/`lru_ttl_put` with `query_router.py`'s query-response cache instead of hand-rolling its own — the other two BFS implementations still have no cache at all. A shared "graph BFS expansion" helper parameterized by an edge-lookup callback could still serve all three.

3. **Three separate "graph → viz payload" builders.** `GraphRagMixin.build_graph_payload` (`graph_rag.py:2163`), `DynamicGraphBackend.graph_data` (`dynamic_graph_backend.py:1156`), and `GraphRenderMixin.build_code_graph_payload` (`graph_render.py:29`) each hand-roll their own node dict (`id`/`name`/`label`/`type`/`color`/`val`/`tooltip`) and link dict (`source`/`target`/`label`/`relation`/`value`/`width`) construction. Still present — unchanged. `DynamicGraphBackend.graph_data` already imports `graph_render._VIZ_TYPE_COLORS` for color mapping, showing the pieces are meant to be shared — a common "to viz payload" helper would remove the remaining duplicated tooltip/dict-shaping code across all three.

4. **Two independent nx.Graph-from-edges builders.** `GraphRagMixin._build_networkx_graph_from_edges` (generic static helper taking `(nodes, edges)`, `graph_rag.py:629`) vs `DynamicGraphBackend._build_nx_graph_from_db` (queries its own facts table then inline-builds an `nx.Graph` with the same weight-accumulation + `distance`-attribute logic, `dynamic_graph_backend.py:1004`). Still present — unchanged. The latter could call the former instead of duplicating it.

---

## 5. Query Routing & LLM Providers

### src/axon/query_router.py — `QueryRouterMixin`

**Role:** Mixed into `AxonBrain`. Owns the whole `query()`/`query_stream()` orchestration: query-route classification, parallel query-transformation (HyDE/multi-query/step-back/decompose), the multi-channel retrieval pipeline (dense + BM25 + sentence-window + code/symbol + graph expansion), MMR dedup, context compression, context assembly, citation extraction, and query caching. Since GraphRAG's ownership-inversion (`GraphRagMixin` is no longer in `AxonBrain`'s MRO — see §4), `QueryRouterMixin` carries its own lazily-initialized `_graph_lock`/`_traversal_cache_lock`/`_entity_token_index` properties and a no-op `_rebuild_entity_token_index()` as **fallbacks** for when `GraphRagMixin` isn't mixed in (e.g. standalone tests); `GraphRagMixin`, when present, overrides all four with the real implementations — `query_router.py:187-215`.

Module-level helpers:
- `_looks_like_llm_error(text)` — true when a completion is one of `OpenLLM`'s returned-not-raised errors (the `copilot` provider reports bridge timeouts/task errors as plain strings rather than raising); checked by both `_classify_query_route_llm` and `_prepend_contextual_context` before trusting an `llm.complete()` result — `query_router.py:40`
- `_slim_source(idx, row)` — build a JSON-friendly citation source entry (id/title/score/truncated text) from one retrieved chunk — `query_router.py:98`
- `_build_citation_metadata(response, citation_results)` — parse `[N]` / `[Document N]` markers out of an LLM response into `{"sources": [...], "citations": [...]}` with char offsets; tolerates hallucinated out-of-range markers — `query_router.py:133`
- `_ROUTE_PROFILES` — dict mapping the 5 query-route names (`factual`, `synthesis`, `table_lookup`, `entity_relation`, `corpus_exploration`) to config-flag overrides (raptor/graph_rag/parent_doc/hyde/multi_query/step_back/query_decompose); applied by `query()`/`query_stream()` — `query_router.py:45`
- `_ROUTE_RESPONSE_MAX_CHARS` / `_CONTEXTUAL_PREFIX_MAX_CHARS` — guard constants (200 / 400 chars) capping the two `self.llm.complete()` call sites below, since `complete()` has no `max_tokens` parameter and always spends the full `llm_max_tokens` budget (8192 by default) unless capped by the caller — `query_router.py:36,37`

Query entry points:
- `query(query, filters=None, chat_history=None, overrides=None)` — the main RAG call: applies overrides → routes → cache lookup → retrieval → rerank → RAPTOR drilldown → GraphRAG local/global context → context compression → LLM synth → citation/provenance capture → cache store. Returns the answer string — `query_router.py:1464`
- `query_stream(query, filters=None, chat_history=None, overrides=None)` — streaming twin of `query()`; first yields `{"type": "sources", "sources": [...]}` then yields text chunks from `llm.stream()`. Not cached (no full response to key on) — `query_router.py:1713`
- `search_raw(query, filters=None, overrides=None)` — retrieval-only, no LLM call; returns `(results, diagnostics, trace)` already sliced to effective top-k. Used by `/search/raw`, `--dry-run` CLI, benchmark harnesses — `query_router.py:1342`
- `load_directory(directory)` (async) — scan a directory via `DirectoryLoader` and ingest — `query_router.py:1873`
- `list_documents()` — unique source files + chunk counts from the vector store — `query_router.py:1882`

Retrieval pipeline internals (still worth reusing/extending rather than reimplementing):
- `_execute_retrieval(query, filters=None, cfg=None)` — outer wrapper: mount-revocation check, mount refresh, and (v0.4.0) wraps the retrieval body in the sealed-project ephemeral mount/unmount window — `query_router.py:842`
- `_execute_retrieval_body(query, filters=None, cfg=None)` — the actual multi-channel retrieval: runs query transforms, dense vector search (+HyDE embedding, +sentence-window channel), BM25/hybrid fusion (RRF or weighted), similarity-threshold filtering, CRAG-Lite confidence correction / legacy web-fallback, MMR dedup, code-lexical-boost, GraphRAG entity expansion, code-graph expansion — `query_router.py:878` (SPLADE sparse fusion, previously an optional stage here, was removed entirely in #7126dbb — never enabled once in five months, no REST/MCP/CLI/REPL surface)
- `_check_mount_revocation()` — raises `PermissionError` if the active mounted share was revoked or has expired since switch — `query_router.py:752`
- `_maybe_refresh_mount()` — per-query or TTL-gated (`"switch"` mode) re-check of the owner's version marker before retrieval; lets `MountSyncPendingError` propagate — `query_router.py:790`
- `_execute_web_search(query, count=5)` — Brave Search API call, normalized into the same result-dict shape as local retrieval hits (`is_web=True`) — `query_router.py:713`
- `_merge_graph_slots(results, top_k, budget)` (staticmethod) — merges base + graph-expanded results after rerank while guaranteeing at least `min(budget, n_expanded)` graph-expanded chunks survive top-k truncation — `query_router.py:1305`
- `_pre_filter_for_rerank(results, top_k)` — trims the reranking candidate pool to `max(top_k*3, 20)` before scoring, to cut reranker calls — `query_router.py:1331`
- `_mmr_deduplicate(results, cfg)` — Maximal Marginal Relevance reorder + near-duplicate (Jaccard ≥0.85) removal; Rust fast path via `rust_bridge.mmr_rerank`, pure-Python fallback — `query_router.py:661`

Query-transformation techniques (each is a standalone, independently reusable LLM call):
- `_get_hyde_document(query)` — HyDE: generate a hypothetical passage that answers the query, to embed instead of the raw query — `query_router.py:572`
- `_get_multi_queries(query)` — generate 3 alternative phrasings for multi-query retrieval; always returns `[original] + up to 3 variants` — `query_router.py:579`
- `_get_step_back_query(query)` — generate a broader/more abstract version of the query (step-back prompting) — `query_router.py:560`
- `_decompose_query(query)` — break a complex query into 2–4 atomic sub-questions, deduped, original always first — `query_router.py:511`
- `_get_all_transforms_unified(query, enabled, multi_count=3)` — single LLM call that produces whichever of {multi_queries, step_back, decomposed, hyde_doc} are enabled, as one JSON response; used instead of 4 separate calls when ≥2 transforms are enabled (see overlap note below) — `query_router.py:586`
- `_classify_query_route(query, cfg)` — dispatches to heuristic or LLM classifier per `cfg.query_router` — `query_router.py:270`
- `_classify_query_route_heuristic(query)` — keyword/length-based classifier into factual/synthesis/table_lookup/entity_relation/corpus_exploration — `query_router.py:276`
- `_classify_query_route_llm(query)` — LLM-based version of the same 5-way classification. **Fixed in #532ab0a** (previously called a nonexistent `self.llm.generate()` inside a bare `except: pass`, so this silently degraded to the heuristic default on every query since it shipped — see the report below). Now calls `self.llm.complete(prompt)`, rejects `_looks_like_llm_error` responses, caps/truncates to `_ROUTE_RESPONSE_MAX_CHARS`, and accepts either a bare label or an unambiguous whole-word label mention in prose (e.g. "Category: synthesis") — `query_router.py:303`
- `_prepend_contextual_context(chunk, whole_doc_text)` — Anthropic "contextual retrieval" method: prepends an LLM-generated situating sentence to a chunk before indexing. **Fixed in #532ab0a** — same `generate()`-doesn't-exist bug as above meant this feature was a complete no-op from the day it shipped; now calls `self.llm.complete(prompt)`, rejects `_looks_like_llm_error` results (so a bridge timeout can't get embedded and permanently poison the corpus), and truncates to `_CONTEXTUAL_PREFIX_MAX_CHARS` — `query_router.py:353`

Context / compression / caching / logging:
- `_compress_context(query, results, cfg=None)` — delegates to `axon.compression.ContextCompressor`; returns `(compressed_chunks, CompressionResult)` for telemetry — `query_router.py:534`
- `_build_context(results)` — assembles the labelled context string (`[Document N — label]` / `[Web Result N — title]`), distinguishes code chunks (file/symbol label) from prose, returns `(context_str, has_web_results)` — `query_router.py:1378`
- `_build_system_prompt(has_web, cfg=None, no_context=False)` — selects strict vs. permissive system prompt, strips citation instructions when `cfg.cite=False`, appends web-search / no-context disclaimers — `query_router.py:1413`
- `_apply_overrides(overrides)` — returns a shallow copy of `self.config` with per-request overrides applied; contract is it never mutates the shared brain config — `query_router.py:1446`
- `_make_cache_key(query, filters, cfg)` — MD5 key over every RAG-affecting config flag + query + filters, for the query cache — `query_router.py:382`
- `_log_query_metrics(...)` — structured `logger.info` event (`event: query_complete`) with retrieval counts, latency, transforms applied — `query_router.py:431`
- `_doc_hash(doc)` — MD5 hex digest of a document's text (Rust fast path via `rust_bridge.compute_doc_hash`, hashlib fallback) — `query_router.py:339`

**Possible internal overlap:** `_get_all_transforms_unified` duplicates the combined purpose of `_get_hyde_document` + `_get_multi_queries` + `_get_step_back_query` + `_decompose_query` — it's an intentional single-call consolidation used when ≥2 transforms are enabled (`_execute_retrieval_body` picks unified vs. the four parallel-executor calls based on `cfg.unified_query_transforms` and how many transforms are on). This is deliberate, not accidental duplication, but it means any change to one transform's prompt/parsing needs to be mirrored in the other path — a future edit to (say) the HyDE prompt in `_get_hyde_document` should also update the `hyde_doc` section of `_get_all_transforms_unified`, and vice versa. Separately, the query-response cache (`_query_cache`) and GraphRAG's traversal cache (§4) no longer duplicate their LRU+TTL bookkeeping independently — both now call the shared `axon._lru_ttl_cache.lru_ttl_get`/`lru_ttl_put` (`query_router.py:16`, added in #8eb1d7f), so this specific overlap is resolved even though it was never listed as a named item in this doc's original audit.

### src/axon/llm.py — `OpenLLM` and provider abstraction

**Role:** Unifies all 9 configured LLM providers (`ollama`, `gemini`, `ollama_cloud`, `openai`, `vllm`, `local`, `github_copilot`, `grok`, `copilot`) behind three methods (`complete`, `complete_with_tools`, `stream`), plus the GitHub Copilot VS Code-extension bridge (OAuth device flow, session-token refresh, task queue) and the `message_text()` accessor needed for reasoning-model OpenAI-dialect responses. As of #15806a1, all client-factory caching goes through two shared helpers (`_cached`/`_cached_rebuild_on_change`) guarded by one `self._clients_lock`, replacing 5 independently hand-rolled cached-factory methods that had no locking despite `OpenLLM` being a single process-wide instance shared by concurrent FastAPI request handlers.

Module-level:
- `message_text(message)` — **the required accessor** for any OpenAI-dialect response message; prefers `message.content`, falls back to `reasoning_content`/`reasoning` (attribute, then `model_extra`) so reasoning models (Gemma 4, GPT-OSS, DeepSeek-R1 derivatives) that dump chain-of-thought into a non-standard field don't silently return `""`. Must be used instead of reading `.content` directly anywhere an OpenAI-compatible response is parsed — `llm.py:265`
- `_probe_openai_compatible_models(url, api_key, timeout)` — **NEW** (#8d3e933): shared `GET <url>/models` network-probe core, extracted from `OpenLLM.ping_local()` and `doctor.check_local_llm_reachable()` (`doctor.py`, outside this section's scope), which previously each hand-rolled the same ~30-line "parse the OpenAI-shape-or-bare-list response into model ids" logic. Assumes `url` is already non-empty/stripped (callers own that check, since their error messaging differs); never raises. Returns `{"reachable": bool, "models": list[str], "error": str | None}` — `llm.py:294`
- `_openai_tool_to_gemini_declaration(tool_dict, genai_types)` — converts one OpenAI-format tool schema dict into a Gemini `FunctionDeclaration` (recursive schema conversion incl. nested objects/arrays) — `llm.py:33`
- `_copilot_device_flow()` — runs the GitHub OAuth device-code flow interactively (prints code/URL, polls), returns the OAuth token — `llm.py:104`
- `_refresh_copilot_session(oauth_token)` — exchanges a GitHub OAuth token for a ~30-min Copilot session token via `copilot_internal/v2/token` — `llm.py:153`
- `_get_copilot_session_token(llm)` — returns a valid (auto-refreshing, buffered by `_COPILOT_SESSION_REFRESH_BUFFER`) Copilot session token, caching it on `llm._openai_clients` — `llm.py:174`
- `_fetch_copilot_models(llm)` — lists chat-capable model IDs from `https://api.githubcopilot.com/models`; falls back to `_COPILOT_MODELS_FALLBACK` on any error or missing token — `llm.py:224`
- `_COPILOT_MODELS_FALLBACK` — static fallback model-id list for the above — `llm.py:213`

`OpenLLM` class:
- `_cached(key, build)` — **NEW** (#15806a1): thread-safe get-or-build for `self._openai_clients[key]`, for caches with no separate invalidation criterion (a fixed slot name, or a key that already encodes everything that should bust the cache, e.g. `f"{base_url}|{api_key}"`). Lock-free fast path on a hit, `self._clients_lock`-guarded double-checked build on a miss — `llm.py:359`
- `_cached_rebuild_on_change(slot_key, sentinel_key, sentinel_value, build)` — **NEW** (#15806a1): thread-safe get-or-rebuild for a fixed slot whose *content* invalidates when a tracked value (an API key, a session token) changes; the sentinel is stored separately and compared each call — `llm.py:373`
- `complete(prompt, system_prompt=None, chat_history=None) -> str` — synchronous single-shot completion; one `if/elif` branch per provider, all 9 supported. OpenAI-dialect branches (`openai`, `vllm`/`local`, `github_copilot`, `grok`) all route their final response through `message_text()` — `llm.py:576`
- `complete_with_tools(prompt, tools=None, system_prompt=None, chat_history=None)` — native function/tool calling; returns `list[ToolCall]` when the model calls a tool, else a plain string. Implemented for `gemini` (native `FunctionDeclaration`s + `thought_signature` round-tripping), `ollama` (native `tools=`), and the OpenAI-compatible family (`openai`/`vllm`/`local`/`github_copilot`/`grok`, native `tool_choice="auto"`). Falls back to `self.complete()` for providers without tool support (`ollama_cloud`, `copilot`) — `llm.py:751`
- `stream(prompt, system_prompt=None, chat_history=None)` — generator yielding text chunks; implemented per-provider mirroring `complete()`. For `vllm`/`local`, reasoning-model deltas are deliberately NOT yielded (only `delta.content`, not `reasoning_content`) — the scratchpad stays server-side; `copilot` fakes streaming by yielding one chunk from `complete()` — `llm.py:892`
- `ping_local(base_url=None, timeout=5.0) -> dict` — health-check + model listing for a local OpenAI-compatible endpoint; now a thin wrapper that resolves `base_url`/handles the "not configured" short-circuit itself, then delegates the actual request/parse to `_probe_openai_compatible_models()` above — used by `axon --doctor` and the REST config route — `llm.py:475`
- `_effective_timeout()` — resolves the request timeout, transparently raising it to `DEFAULT_LOCAL_LLM_TIMEOUT` for the `local` provider unless `llm.timeout` was explicitly set away from the global default — `llm.py:460`
- `_get_openai_client(base_url=None, api_key=None)` — cached `openai.OpenAI` client factory built on `_cached()`, cache keyed by `base_url|resolved_key` so runtime API-key rotation (`POST /config/update`) rebuilds the client instead of reusing a stale credential — `llm.py:510`
- `_local_client()` — `_get_openai_client` pre-bound to `config.local_base_url` / `local_api_key` (or dummy key) for the `local` provider — `llm.py:498`
- `_get_grok_client()` — cached OpenAI-compatible client (via `_cached()`) pointed at `https://api.x.ai/v1`, keyed on `grok_api_key` — `llm.py:532`
- `_get_copilot_client()` — OpenAI client authenticated with the auto-refreshed Copilot session token (via `_cached_rebuild_on_change()`), rebuilt only when the token changes — `llm.py:552`
- `_get_gemini_sdk()` — resolves+caches (via `_cached()`) the `google.genai` / `google.genai.types` modules — `llm.py:388`
- `_get_gemini_client(genai_sdk)` — cached (via `_cached_rebuild_on_change()`) `google.genai.Client`, rebuilt if `gemini_api_key` changes — `llm.py:398`
- `_build_gemini_contents(prompt, system_prompt, history, is_gemma)` (staticmethod) — builds the Gemini `contents` payload from chat history, including native tool-call/tool-response round-tripping (`__tool_calls__` history entries → `function_call`/`function_response` parts, with `thought_signature` echoing for Gemini thinking models) and the Gemma special case (no `system_instruction` support, so system prompt gets prepended to the user turn) — `llm.py:406`

**Possible internal overlap:** `complete()`, `complete_with_tools()`, and `stream()` each still re-implement, per provider, the same "build `messages` list from `system_prompt` + `chat_history` + trailing user `prompt`" boilerplate (identical ~6-line block repeated for `ollama`, `openai`, `vllm`/`local`, `github_copilot`, `grok` inside all three methods — roughly 15 near-duplicate copies total). Confirmed still present — no `_build_openai_messages`-style helper exists anywhere in `llm.py`; the client-*factory* consolidation (#15806a1, above) fixed a different, narrower duplication (per-provider client construction/caching) and did not touch this one. None of the OpenAI-compatible branches share a helper for message-list building. A shared `_build_openai_messages(system_prompt, history, prompt, tool_results=None)` helper would remove most of the duplication across `complete`/`complete_with_tools`/`stream` and reduce the risk of the three methods drifting (e.g. tool-result role-handling only exists in `complete_with_tools`, not in `complete`/`stream`, which is fine since those don't take `tools`, but any future per-provider fix — like a header tweak or role-mapping change — has to be applied in up to 5 places per method).

---

## 6. Projects, Sessions, Sharing & Governance

### src/axon/projects.py
Role: Module-level (no class) functions implementing the "AxonStore" on-disk layout — project directory naming/nesting (up to 5 levels via `subs/`), project CRUD, project/store identity IDs, maintenance state, active-project tracking, and listing of received share mounts. This is the ground-truth storage model every other file in this set (sessions, shares, mounts, access, project_pack) builds on.

- `set_projects_root(path)` — override the global `PROJECTS_ROOT` at runtime (called by `AxonBrain` from `config.yaml`) — `projects.py:73`
- `is_reserved_top_level_name(name)` — check if a name's top segment collides with a reserved root (`projects`, `mounts`, `sharemount`, `_default`, `.shares`) — `projects.py:105`
- `build_project_id(prefix="ns")` — generate a random `{prefix}_{uuid4hex}` namespace ID — `projects.py:120`
- `build_source_id(project_id, source_kind, canonical_source_locator)` — deterministic `src_`-prefixed SHA-256-derived ID for a document source — `projects.py:132`
- `build_chunk_id(project_id, source_id, subdoc_locator, chunk_index, chunk_kind="leaf")` — deterministic `chk_`-prefixed SHA-256-derived globally-unique chunk ID — `projects.py:150`
- `ProjectHasChildrenError(ValueError)` — raised by `delete_project` when sub-projects still exist — `projects.py:173`
- `project_dir(name)` — resolve a project name (incl. `a/b/c` nesting) to its filesystem root path — `projects.py:211`
- `project_vector_path(name)` — path to a project's `vector_store_data/` dir — `projects.py:227`
- `project_bm25_path(name)` — path to a project's `bm25_index/` dir — `projects.py:232`
- `project_sessions_path(name)` — path to a project's `sessions/` dir — `projects.py:237`
- `ensure_project(name, description="", security_mode=None, graph_backend=None)` — idempotently create a project (and all ancestor projects) with `meta.json`; enforces `graph_backend` immutability once set — `projects.py:242`
- `get_project_id(name)` — read `project_id` out of a project's `meta.json` — `projects.py:338`
- `get_project_graph_backend(name)` — read a project's stored `graph_backend` (default `"graphrag"`) — `projects.py:354`
- `get_store_id(user_dir)` — read `store_id` from a user's `store_meta.json` — `projects.py:369`
- `get_or_create_node_id(user_dir)` — read or mint-and-persist the per-store `node_id` UUID used to stamp `version.json` markers (v0.4.0 Item 4a) — `projects.py:384`
- `list_descendants(name, visited=None)` — recursive DFS (with symlink-cycle guard) returning all sub-project names under `name` — `projects.py:427`
- `has_children(name)` — cheap short-circuit check for whether a project has any direct sub-projects — `projects.py:461`
- `get_maintenance_state(name)` — read a project's maintenance state (`normal`/`draining`/`readonly`/`offline`) — `projects.py:507`
- `set_maintenance_state(name, state)` — persist + audit-log a maintenance-state transition; validates `state` — `projects.py:525`
- `list_projects()` — return all top-level projects (sorted newest-first) with recursively-nested `children`, each dict carrying name/description/created_at/path/maintenance_state/graph_backend — `projects.py:556`
- `get_active_project(projects_root=None)` — return the currently active project name (defaults `"default"`), read from `~/.axon/.active_project`. **Signature grew a param**: when `projects_root` is passed, the stored name is validated against that root and a pointer to a deleted project self-heals back to `"default"` (persisted via `set_active_project`); omitted, it's a plain unvalidated read — deliberately opt-in because the lightweight CLI early-exit paths never call `set_projects_root()`, so validating against the stale module-global would wrongly reset a perfectly good active-project pointer — `projects.py:593`
- `set_active_project(name)` — persist the active project name to disk (non-fatal on write failure) — `projects.py:643`
- `delete_project(name)` — delete a project's directory tree; refuses `"default"` and projects with children (`ProjectHasChildrenError`), retries on `PermissionError`, resets active project if needed — `projects.py:657`
- `ensure_user_project(user_dir)` — idempotently scaffold a fresh AxonStore user namespace (`default/`, `mounts/`, `.shares/`, `projects/`, `store_meta.json`) — `projects.py:695`
- `list_share_mounts(user_dir)` — list all received share mounts for a user by reading `mounts/` descriptors, flagging broken ones via `validate_mount_descriptor` — `projects.py:783`

### src/axon/sessions.py
Role: Lightweight JSON-file persistence for REPL chat sessions (per-project directory, capped at 50 most-recent, auto-evicting oldest). Distinct from AxonStore's project/session *directories* — this is the actual read/write/list layer for session transcripts.

- `_SESSION_ID_RE` — `^[A-Za-z0-9_-]{1,128}$`; the single filesystem-safe-alphabet pattern every session-ID entry point must validate a caller-supplied ID against before it reaches `_session_path()`'s `f"session_{session_id}.json"` interpolation (a path-traversal sink otherwise). Relocated here from `api_routes/projects.py` (route-specific, only guarded the REST route) so REPL `/resume` validates against the exact same pattern instead of calling `_load_session()` directly and bypassing the check entirely — `sessions.py:17`
- `_new_session(brain)` — build a new session dict (timestamp ID, provider/model, active project, empty history) — `sessions.py:32`
- `_save_session(session)` — persist a session JSON file and evict oldest beyond the 50-session cap (`_MAX_SESSIONS`) — `sessions.py:50`
- `_list_sessions(limit=20, project=None)` — return the most recent saved sessions for a project (or global fallback dir) — `sessions.py:69`
- `_load_session(session_id, project=None)` — load one session by ID, or `None` if missing/corrupt — `sessions.py:86`
- `_print_sessions(sessions)` — REPL-formatted table printer for a session list — `sessions.py:98`

Note: all functions are underscore-prefixed (semi-private) but are the only entry points for session persistence — REPL and any future surface wanting session history should call these rather than reimplementing JSON I/O. Any caller accepting a session ID from outside (REST route, REPL, MCP) must validate it against `_SESSION_ID_RE` first — this was a real path-traversal bug in REPL `/resume` until it was fixed to do so.

### src/axon/shares.py
Role: AxonStore share-key lifecycle — generation, redemption, revocation, TTL extension, and listing of read-only project shares between OS users, backed by two files per user (`.share_manifest.json` public, `.share_keys.json` private) with HMAC-bound tokens.

- `generate_share_key(owner_user_dir, project, grantee, *, ttl_days=None)` — mint a read-only share key + base64 `share_string` for out-of-band transmission; optional TTL expiry — `shares.py:136`
- `redeem_share_key(grantee_user_dir, share_string)` — validate a share string (revocation/expiry/HMAC checks) and create the grantee's `mounts/` descriptor via `mounts.create_mount_descriptor` — `shares.py:217`
- `revoke_share_key(owner_user_dir, key_id)` — lazily revoke a share key in both the private key store and public manifest (grantee descriptor removed on next validation) — `shares.py:308`
- `list_shares(user_dir)` — return `{"sharing": [...], "shared": [...]}` — both issued and received shares, with computed `expired` flag — `shares.py:366`
- `validate_received_shares(user_dir)` — scan all received shares against owners' manifests; remove mount descriptors for revoked/expired shares (lock-guarded to avoid TOCTOU with concurrent redemption) — `shares.py:399`
- `extend_share_key(owner_user_dir, key_id, *, ttl_days)` — renew or clear (`ttl_days=None`) a share key's expiry, mirrored to manifest — `shares.py:458`

Internal helpers worth knowing about (not typically called externally, but relevant to correctness): `_is_expired(expires_at, now=None)` fail-closed TTL check with 5-minute clock-skew leeway (`shares.py:50`); `_compute_hmac(...)` binds token to key_id/project/grantee/owner_store_path (`shares.py:125`); `_write_json(path, data)` writes a share file **without any atomic-replace step** — a bare `path.write_text(...)` followed by a `chmod` (`shares.py:110`) — see "Possible internal overlap" below.

### src/axon/mounts.py
Role: CRUD for the canonical `mount.json` descriptor model representing a received share on the grantee's filesystem (`{user_dir}/mounts/{mount_name}/mount.json`). Pure descriptor I/O — no crypto, no HTTP; `shares.py` and `projects.py` both call into this.

- `mounts_root(user_dir)` — path to `{user_dir}/mounts/` — `mounts.py:38`
- `mount_descriptor_dir(user_dir, mount_name)` — per-mount subdirectory path, with path-traversal guard (`is_relative_to` check) — `mounts.py:43`
- `mount_descriptor_path(user_dir, mount_name)` — path to a specific `mount.json` — `mounts.py:53`
- `create_mount_descriptor(grantee_user_dir, mount_name, owner, project, owner_user_dir, target_project_dir, share_key_id)` — write a new `mount.json` (reads owner's `project_id`/`graph_backend`/`store_id` where available); always `readonly: true`; the write is a bare `write_text()`, not atomic-replaced — `mounts.py:63`
- `load_mount_descriptor(user_dir, mount_name)` — load one descriptor, or `None` if absent/corrupt — `mounts.py:128`
- `list_mount_descriptors(user_dir)` — return all active, non-revoked mount descriptors under a user — `mounts.py:139`
- `remove_mount_descriptor(user_dir, mount_name)` — delete a mount's descriptor directory (`rmtree`) — `mounts.py:160`
- `validate_mount_descriptor(descriptor)` — check revoked/state/target-existence and return `(bool, reason)` — `mounts.py:172`

### src/axon/access.py
Role: Small, centralized write-permission policy for projects — consolidates checks that used to be ad-hoc `AxonBrain` methods. Two functions only; this is the single place to extend write-gating logic (e.g. new maintenance states, new read-only scopes) rather than re-deriving it per call site.

- `is_mounted_share_path(project_name)` — detect whether a project name refers to a received share mount (`mounts/<mount_name>` convention) — `access.py:16`
- `check_write_allowed(operation, active_project, read_only_scope, is_mounted)` — raise `PermissionError` if a write is disallowed, checking in order: merged read-only scope (`@projects`/`@mounts`/`@store`) → mounted-share (always read-only) → project maintenance state (`readonly`/`offline`/`draining`); fails open (with a WARNING log) if the maintenance-state lookup itself errors unexpectedly — `access.py:25`

### src/axon/project_pack.py
Role: `axon --project-pack` / `axon --project-unpack` — zip a project's entire on-disk footprint for backup, restore, or relocation. New in this release cycle (not present at the 2026-08-28 audit) — confirmed via repo history to be genuinely new functionality; no export/import/backup mechanism existed anywhere in this codebase before it. Brain-agnostic by design (project name / zip path + an explicit `user_dir` in, no `AxonBrain` needed) so CLI's early-exit flag handlers, which run before `AxonBrain` is constructed, can call these directly — patterned on `security.seal`'s cross-surface wiring style. Wired across all 7 user-facing surfaces (CLI, REPL, REST, MCP, VS Code, plus the two Copilot-bridge paths); CLI's `--project-pack`/`--project-unpack` route through the single-instance server-detection gate the same way `--ingest` does, via `remote_project_pack()`/`remote_project_unpack()` in `server_client.py`.

- `ProjectPackError(Exception)` — raised for any pack/unpack failure: bad project name, missing project, unsafe archive contents, or an existing unpack target without `force` — `project_pack.py:44`
- `pack_project(project_name, user_dir, *, out_path=None)` — zips a project's entire on-disk footprint (including `.security/` verbatim for sealed projects — ciphertext + DEK wraps, never decrypted) into `{out_path or ~/.axon/packs/{name}-{timestamp}.axonpack.zip}`; also packs an externally-relocated `.dynamic_graph.db` (cloud-sync/WSL-mount case) in place of the local copy when one is found. Returns `{"status": "packed", "project", "out_path", "sealed", "file_count", "bytes"}`. Caller-owned: does not switch away from the active project itself — a live writer mid-pack would produce a torn copy — `project_pack.py:101`
- `unpack_project(zip_path, user_dir, *, as_name=None, force=False)` — restores a pack: validates every archive member (rejects absolute paths, `..` traversal, and symlink entries) *before* writing anything, extracts into a staging directory (never the real target), and only does the final `os.replace` on full success — zip-slip-safe by construction, not a post-hoc filter. Refuses an existing target unless `force=True` (full replace, no merge). Returns `{"status": "unpacked", "project", "sealed", "file_count", "manifest"}` — `project_pack.py:220`
- `MANIFEST_NAME` (`"_axon_pack_manifest.json"`), `PACK_FORMAT_VERSION` — the pack's self-describing manifest entry name/schema version, written by `pack_project` and read by `_read_manifest` — `project_pack.py:38-39`

Internal helpers worth knowing about: `_resolve_project_dir(name, user_dir)` mirrors `projects.project_dir()`'s `subs/` nesting anchored at an explicit `user_dir` — deliberately duplicated rather than imported, for the same reason `security.seal._resolve_project_dir` is (see "Possible internal overlap" below) — `project_pack.py:49`; `_validate_member_path(name, staging)` is the zip-slip guard used by the two-pass `unpack_project` — `project_pack.py:188`; `_external_dynamic_graph_db(project_dir)` mirrors `DynamicGraphBackend._resolve_db_path`'s two-tier `project_id` resolution so a relocated live graph DB is found even for a sealed project — `project_pack.py:76`.

### src/axon/governance.py
Role: Governance/audit backend for the Operator Console — a SQLite-first (JSONL-fallback) append-only audit log plus an in-memory tracker for active/recent "Copilot bridge" sessions. Fire-and-forget writes via a small process-wide thread pool so callers never block.

- `AuditEvent` (dataclass) — one audit record: `event_id`, `timestamp`, `actor`, `surface`, `project`, `action`, `target_type`, `target_id`, `status`, `details`, `request_id` — `governance.py:90`
- `VALID_ACTIONS` — frozenset of the allowed `action` vocabulary (`ingest_started/completed/failed`, `delete`, `graph_finalize`, `maintenance_changed`, `share_generated/redeemed/revoked/extended`, `copilot_session_opened/closed/failed`) — `governance.py:62`
- `AuditStore(db_path, retention_days=90)` — thread-safe SQLite (DELETE journal mode, share-mount safe) audit store with automatic JSONL fallback — `governance.py:112`
  - `.append(event)` — persist one event, thread-safe, swallows errors silently — `governance.py:189`
  - `.query(project=None, action=None, surface=None, status=None, since=None, limit=100)` — filtered, newest-first event query (works against SQLite or JSONL fallback) — `governance.py:223`
  - `.prune(days=90)` — delete events older than N days, returns count deleted (no-op under JSONL fallback) — `governance.py:332`
- `CopilotSession` (dataclass) — one Copilot bridge session record with `.is_active` property — `governance.py:354`
- `CopilotSessionStore(max_recent=50)` — in-memory active/recent session tracker with capacity-based eviction (closed sessions evicted before active ones) — `governance.py:370`
  - `.open(session_id, request_id, project)` — register a new session — `governance.py:402`
  - `.close(session_id, *, error=None)` — mark a session closed (normal or errored) — `governance.py:413`
  - `.expire(session_id)` — force-close a stuck session (operator action); returns whether it was found/active — `governance.py:422`
  - `.list_active()` — sessions not yet closed — `governance.py:432`
  - `.list_recent(limit=20)` — most-recently-opened sessions (active + closed) — `governance.py:437`
- `get_store()` — return the process-wide `AuditStore` singleton, lazily created under `PROJECTS_ROOT` — `governance.py:453`
- `get_session_store()` — return the process-wide `CopilotSessionStore` singleton — `governance.py:472`
- `emit(action, target_type, target_id, *, project, actor="api", surface="api", status="completed", details=None, request_id="")` — the fire-and-forget audit-write entry point everything else should call instead of touching `AuditStore` directly — `governance.py:477`

**Possible internal overlap**

- **Project listing walks meta.json twice, in two shapes.** `projects.list_projects()` / `_list_sub_projects()` (`projects.py:556`, `:471`) build a recursive tree of project dicts (name/description/created_at/path/maintenance_state/graph_backend/children) purely by reading `meta.json` files directly. `mounts.list_mount_descriptors()` and `projects.list_share_mounts()` independently do a similar directory-scan-plus-JSON-parse pattern over `mounts/`. There's no shared "scan a directory of dirs, parse each meta/descriptor JSON, skip on parse error" helper — the same defensive try/except JSON-read loop is duplicated four times (`_list_sub_projects`, `list_projects`, `list_mount_descriptors`, `list_share_mounts`). Still present, unchanged since the 2026-08-28 audit. A shared internal iterator could reduce duplication if this file set is ever refactored.
- **Two independent "is this share still good" checks.** `mounts.validate_mount_descriptor()` (`mounts.py:172`) checks `revoked`/`state`/target-existence on a descriptor, while `shares.validate_received_shares()` (`shares.py:399`) separately re-derives revoked/expired status by re-reading the *owner's* manifest and comparing key_ids. These overlap conceptually (both answer "is this mount still valid?") but check different sources of truth (local descriptor state vs. owner's manifest) — worth documenting clearly so a future caller doesn't assume one implies the other, since a descriptor can look locally "active" while the owner's manifest already shows it revoked (that's exactly the gap `validate_received_shares` exists to close, but it's not obvious from `validate_mount_descriptor` alone). Still present, unchanged.
- **Non-atomic writes in this file set are a strictly weaker variant of the "atomic file writes reimplemented" duplication flagged for Security below.** `shares._write_json()` (`shares.py:110`), `mounts.create_mount_descriptor()` (`mounts.py:122`), and `projects._ensure_single_project()`/`_ensure_single_project_at()`/`ensure_user_project()` (`projects.py:301-314`, `713-730`, `743-754`) all persist JSON via a bare `path.write_text(...)` — no tempfile, no `os.replace`, so a crash mid-write can leave a truncated/unparsable descriptor or `meta.json`. This is worse than Security's hand-rolled tempfile+replace pattern, not just a duplicate of it. It was explicitly named as deferred follow-up work in the commit that added `_atomic_persist.write_bytes_if_changed()`/`write_text_if_changed()` (which currently only `config.py` uses): "Remaining 14 lower-risk sites from the audit (shares.py, sessions.py, projects.py's meta.json, mounts.py, sentence_window.py, etc.) are left for a follow-up pass." Migrating these three files to `write_json_if_changed()` (or the new bytes/text siblings) would both fix the atomicity gap and remove the duplicated scan-and-persist boilerplate in one pass.
- **`project_pack._resolve_project_dir()` is a third independent copy of the "resolve a project name against an explicit user_dir, honoring `subs/` nesting" pattern**, after `projects.project_dir()` (resolves against the process-global `PROJECTS_ROOT`) and `security.seal._resolve_project_dir()` (`seal.py:256`, resolves against an explicit `user_dir` for the same reason: CLI's early-exit path never calls `set_projects_root()`). `project_pack.py:49`'s docstring explicitly acknowledges duplicating rather than importing `seal`'s version. Not a bug — each copy exists for a documented reason — but a shared `resolve_project_dir_at(name, user_dir)` helper (in `projects.py`, imported by both `seal.py` and `project_pack.py`) would remove the third near-identical implementation.

---

## 7. Security (Sealed Store)

Subsystem: `src/axon/security/` — envelope-encryption "sealed project" store: AES-256-GCM file crypto, owner master-key + per-project DEK lifecycle, OS-keyring wrapper with session/never/file-fallback modes, per-share KEK wrapping for grantee sharing (soft/hard revoke), Ed25519 signing for TTL expiry sidecars, ephemeral plaintext mount cache, and a Diceware passphrase generator. All modules require the optional `[sealed]` extra (`cryptography` + `keyring`); each raises a friendly `ImportError` at import time if missing, and `axon/security/__init__.py` re-wraps that as `SecurityError` for its facade functions. `security/data/__init__.py` is an empty marker file for the bundled wordlist resource package — no functions to catalog.

### `src/axon/security/__init__.py`
Role: Public facade / stub-preserving router. Every function here lazily imports the real implementation module and converts `ImportError` (extra not installed) into `SecurityError`, so callers (REST, MCP, REPL, CLI) get one stable import surface regardless of whether `[sealed]` is installed.

- `generate_passphrase(n_words=6, separator=" ")` — re-export of `wordlist.generate_passphrase`, lazily imported — `__init__.py:42`
- `estimate_entropy_bits(n_words)` — re-export of `wordlist.estimate_entropy_bits` — `__init__.py:53`
- `class SecurityError(Exception)` — base exception for all security operation failures — `__init__.py:60`
- `class ShareExpiredError(SecurityError)` — raised when a signed TTL expiry sidecar shows a share has elapsed or fails signature verification; triggers auto-destroy of grantee DEK/cache/mount — `__init__.py:64`
- `store_status(user_dir)` — returns `{initialized, unlocked, sealed_hidden_count, public_key_fingerprint, cipher_suite}` dict for the sealed store — `__init__.py:86`
- `bootstrap_store(user_dir, passphrase)` — routes to `master.bootstrap_store` — `__init__.py:113`
- `unlock_store(user_dir, passphrase)` — routes to `master.unlock_store` — `__init__.py:128`
- `lock_store(user_dir)` — routes to `master.lock_store`; never throws, safe shutdown hook — `__init__.py:143`
- `change_passphrase(user_dir, old_passphrase, new_passphrase)` — routes to `master.change_passphrase`, O(1) regardless of sealed-project count — `__init__.py:156`
- `is_unlocked(user_dir)` — routes to `master.is_unlocked` — `__init__.py:172`
- `get_sealed_project_record(project, user_dir)` — routes to `seal.get_sealed_project_record`, returns `None` on minimal install — `__init__.py:183`
- `generate_sealed_share(owner_user_dir, project, grantee, key_id, *, expires_at=None)` — routes to `share.generate_sealed_share` — `__init__.py:197`
- `redeem_sealed_share(user_dir, share_string)` — routes to `share.redeem_sealed_share` — `__init__.py:226`
- `revoke_sealed_share(owner_user_dir, project, key_id, *, rotate=False)` — routes to `share.revoke_sealed_share` (soft or hard) — `__init__.py:244`
- `validate_received_sealed_shares(user_dir)` — walks every sealed mount descriptor, removes any whose owner-side wrap file has disappeared (soft-revoke detection); returns removed mount names — `__init__.py:268`
- `list_sealed_shares(user_dir)` — returns `{"sharing": [...], "shared": [...]}` — owned projects' active wraps + redeemed sealed mounts — `__init__.py:303`
- `resolve_owned_sealed_project_path(project_name, user_dir)` — resolves on-disk path for a sealed project this user owns; raises `SecurityError` if not sealed — `__init__.py:343`
- `project_rotate_keys(project_root)` — thin wrapper over `share._hard_revoke` with `key_id=""`, i.e. "rotate DEK without revoking a specific share" — `__init__.py:359`
- `project_seal(project_name, user_dir, *, migration_mode="in_place", config=None, embedding=None)` — routes to `seal.project_seal` — `__init__.py:378`

### `src/axon/security/crypto.py`
Role: Lowest-level cryptographic primitives — no state, no disk-path knowledge beyond the file it's told to write. Everything else in `axon.security` is built on top of these.

- `class SealedFormatError(Exception)` — malformed AXSL header / bad magic / unsupported schema-version or cipher-id; distinct from `InvalidTag` (tamper/wrong-key) — `crypto.py:128`
- `class SealedFile` — atomic AES-256-GCM file wrapper (format: `AXSL` magic + 16B header + 12B nonce + ciphertext + 16B tag [+ optional random padding]) — `crypto.py:196`
  - `SealedFile.write(path, plaintext, key, *, aad=b"", padding_bytes=0)` — buffered one-shot encrypt+atomic-write; supports random trailing padding to defeat file-size leaks — `crypto.py:222`
  - `SealedFile.write_stream(path, plaintext_iter, key, *, aad=b"", chunk_size=1MiB, padding_bytes=0)` — streaming encrypt via low-level `Cipher` API, bounded memory for large payloads — `crypto.py:277`
  - `SealedFile.write_stream_from_path(src_path, dst_path, key, *, aad=b"", chunk_size=1MiB, padding_bytes=0)` — convenience: chunked-read a source file straight into `write_stream` — `crypto.py:376`
  - `SealedFile.read(path, key, *, aad=b"")` — decrypt+return plaintext; raises `SealedFormatError` or `cryptography.exceptions.InvalidTag` — `crypto.py:417`
- `generate_dek()` — fresh 256-bit Data Encryption Key from OS CSPRNG — `crypto.py:459`
- `derive_kek(token, key_id, *, info=b"axon-share-v1")` — HKDF-SHA256 32-byte Key Encryption Key from a share token, salted by key_id — `crypto.py:464`
- `wrap_key(key, kek)` — AES-256 Key Wrap (RFC 3394); wraps a 32-byte key under a 32-byte KEK → 40 bytes — `crypto.py:493`
- `unwrap_key(wrapped, kek)` — reverse of `wrap_key`; raises `InvalidUnwrap` on tamper/wrong KEK — `crypto.py:503`
- `make_aad(key_id, relpath)` — builds recommended AAD (`key_id \0 relpath`) binding GCM tag to project+path, preventing cross-project file swaps — `crypto.py:516`
- Constants: `MAGIC`, `SCHEMA_VERSION`, `CIPHER_AES_256_GCM`, `HEADER_LEN`, `NONCE_LEN`, `TAG_LEN`, `DEK_LEN`, `STREAMING_CHUNK_SIZE`, `MAX_PADDING_BYTES` — `crypto.py:73-95`
- `_self_check()` — internal diagnostic round-trip (wrap/unwrap + file roundtrip), intended for future `axon doctor` — `crypto.py:531` (semi-private but reusable for diagnostics)

### `src/axon/security/cache.py`
Role: Ephemeral plaintext materialisation for sealed projects — decrypts a whole sealed project into an OS-temp scratch dir so mmap-based backends (TurboQuantDB/LanceDB/BM25) work unmodified, then securely wipes it on close. Also handles orphaned-cache cleanup after crashes. Its PID-liveness probe (`list_orphans`'s "is the owning process still alive" check) was extracted out to the shared `axon._pid_check.pid_alive()` so `server_client.py`'s single-instance store lock could reuse the exact same logic instead of a second hand-rolled copy — `cache.py` now just re-binds `_pid_alive = pid_alive` for its one internal caller.

- `class CacheCapacityError(RuntimeError)` — insufficient free disk (< project size × 1.1) to materialise the cache — `cache.py:88`
- `class SealedFileTamperError(RuntimeError)` — a file in the must-seal set lacks the AXSL header at mount time (possible on-disk tamper) — `cache.py:97`
- `is_sealed_file(path)` — cheap 4-byte magic-header probe to decide decrypt-vs-copy per file — `cache.py:115`
- `class SealedCache` — context-manager-style handle to a decrypted project cache — `cache.py:252`
  - `SealedCache.create(sealed_dir, dek, *, key_id, cache_root=None)` — decrypts every sealed file (skips `.security/`) into a fresh `tempfile.mkdtemp` dir, copies plaintext passthrough files, writes a PID sentinel; raises `CacheCapacityError`/`SealedFileTamperError`/`InvalidTag`/`SealedFormatError` — `cache.py:275`
  - `cache.path` (property) — the materialised plaintext dir to point backends at — `cache.py:391`
  - `cache.wipe()` — idempotent, thread-safe secure-delete of every cache file + dir removal — `cache.py:395`
  - Also usable as `with SealedCache.create(...) as cache: ...` (`__enter__`/`__exit__` call `wipe()`) — `cache.py:410`
- `list_orphans(cache_root=None)` — finds `axon-sealed-*` cache dirs whose owner PID is no longer alive — `cache.py:425`
- `cleanup_orphans(cache_root=None)` — wipes every orphan cache dir found by `list_orphans`; called from `AxonBrain.__init__`; never raises — `cache.py:453`
- Constants: `CACHE_PREFIX`, `PID_SENTINEL_FILENAME`, `CACHE_HEADROOM_FRACTION` — `cache.py:79-85`
- `_self_check()` — internal diagnostic round-trip for future `axon doctor` — `cache.py:475`

### `src/axon/security/fallback_store.py`
Role: Headless/no-keyring persistence for the owner's master record — same scrypt-wrapped JSON shape the OS keyring would hold, written to `<user_dir>/.security/master.enc` (0600 perms) when DPAPI/Keychain/Secret Service is unavailable.

- `fallback_master_path(user_dir)` — `<user_dir>/.security/master.enc` — `fallback_store.py:55`
- `is_present(user_dir)` — cheap existence probe — `fallback_store.py:60`
- `read_master_record(user_dir)` — returns raw JSON string or `None`; re-raises `OSError` (not swallowed) if file exists but unreadable, to avoid a caller mistakenly re-bootstrapping over an existing master — `fallback_store.py:65`
- `write_master_record(user_dir, payload)` — validates payload is JSON, atomically writes + chmods 0600 — `fallback_store.py:93`
- `delete_master_record(user_dir)` — removes the fallback file; returns whether it existed — `fallback_store.py:124`
- Constant: `FALLBACK_MASTER_FILENAME = "master.enc"` — `fallback_store.py:52`

### `src/axon/security/keyring.py`
Role: Thin cross-platform wrapper over the `keyring` package (DPAPI/Keychain/Secret Service), plus a v0.4.0 mode-dispatch layer (`persistent` / `session` / `never`) that lets callers avoid touching the OS keyring entirely.

- `class KeyringUnavailableError(RuntimeError)` — OS keyring backend missing/unusable (no D-Bus, fail-stub backend, etc.) — `keyring.py:59`
- `master_service(owner)` — builds `axon.master.<owner>` service name — `keyring.py:80`
- `share_service(key_id)` — builds `axon.share.<key_id>` service name — `keyring.py:87`
- `is_available()` — lightweight write/read/delete round-trip probe; never raises, returns bool — `keyring.py:112`
- `store_secret(service, username, secret)` — mode-dispatched write (OS keyring / in-memory session cache / no-op) — `keyring.py:142`
- `get_secret(service, username)` — mode-dispatched read; returns `None` for "not found" — `keyring.py:174`
- `delete_secret(service, username)` — mode-dispatched delete; no-ops on "not found" — `keyring.py:202`
- `class SessionDEKCache` — thread-safe in-process dict substitute for the OS keyring under `keyring_mode="session"` — `keyring.py:239`
  - `.set/.get/.delete(service, username[, secret])`, `.clear()`, `__len__` — `keyring.py:261-282`
- `set_keyring_mode(mode)` — sets process-wide active mode (`persistent`/`session`/`never`); clears the session cache on transition away from `session` — `keyring.py:290`
- `get_keyring_mode()` — returns the active mode — `keyring.py:322`
- `session_cache()` — returns the process-singleton `SessionDEKCache` (mainly for test inspection) — `keyring.py:328`
- Constants: `MASTER_SERVICE_PREFIX`, `SHARE_SERVICE_PREFIX` — `keyring.py:76-77`
- `_self_check()` — diagnostic dict (`available`, `backend`) for future `axon doctor` — `keyring.py:342`

### `src/axon/security/master.py`
Role: Owner master-key bootstrap/unlock/lock/rotate + per-project DEK lifecycle — the envelope-encryption KMS core. Master key is 32 random bytes, persisted only in scrypt-wrapped form (keyring + fallback file); project DEKs are wrapped under the in-memory master.

- `class BadPassphraseError(SecurityError)` — wrong passphrase on unlock/change — `master.py:102`
- `bootstrap_store(user_dir, passphrase)` — first-time setup: generates master, scrypt-wraps under passphrase-derived KEK, persists to keyring+file, caches master in-process; raises if already bootstrapped or passphrase < 8 chars — `master.py:262`
- `unlock_store(user_dir, passphrase)` — verifies passphrase, unwraps master, caches in-process (`BadPassphraseError` on mismatch) — `master.py:306`
- `lock_store(user_dir)` — clears in-memory cached master for this owner — `master.py:330`
- `is_unlocked(user_dir)` — bool check against the in-process cache — `master.py:344`
- `change_passphrase(user_dir, old_passphrase, new_passphrase)` — re-wraps master under a fresh KEK; project DEKs untouched (O(1)) — `master.py:350`
- `get_master_key(user_dir)` — returns cached unlocked master bytes; raises `SecurityError` if locked — `master.py:389`
- `get_or_create_project_dek(user_dir, project_dir)` — reads+unwraps `.security/dek.wrapped` if present, else mints+wraps+persists a new DEK atomically (0600) — `master.py:411`
- `get_project_dek(user_dir, project_dir)` — read-only DEK lookup; raises if the wrapped-DEK file doesn't exist (no silent minting) — used on mount/read paths — `master.py:460`
- `is_bootstrapped(user_dir)` — probes whether a master record exists (keyring or fallback) — `master.py:252`
- Constants: `MASTER_USERNAME`, `PROJECT_DEK_FILENAME`, `SCHEMA_VERSION`, scrypt params (`_SCRYPT_N=2**15` etc.) — `master.py:83-99`
- `_self_check(user_dir=None)` — isolated bootstrap/unlock/lock/DEK round-trip diagnostic, never touches the real keyring — `master.py:487`

### `src/axon/security/mount.py`
Role: Thin glue layer translating "open this sealed project for reading" into a plaintext cache path, composing `master` (DEK) + `seal` (marker) + `cache` (materialisation) so `AxonBrain` doesn't need to know crypto details.

- `materialize_for_read(project_dir, user_dir, *, cache_root=None, dek=None)` — validates the project is sealed, resolves the DEK (from master for owners, or a pre-fetched grantee DEK), calls `SealedCache.create`, unifies all underlying exceptions as `SecurityError` — `mount.py:42`
- `release_cache(cache)` — None-safe wrapper around `cache.wipe()`, so `AxonBrain.close()` can call unconditionally — `mount.py:136`

### `src/axon/security/seal.py`
Role: `axon project seal <name>` — walks a plaintext project and converts it in-place to sealed_v1 (AES-256-GCM per content file), with crash-safe resume via a `.security/.sealing` in-progress marker. Also the read-only "is this project sealed" probe used across the codebase.

- `is_project_sealed(project_dir)` — cheap probe: does `<project>/.security/.sealed` exist — no unlock required — `seal.py:125`
- `read_sealed_marker(project_dir)` — reads+validates the sealed marker JSON (`v`, `cipher_suite`, `seal_id`, `sealed_at`, counts); raises `SecurityError` if malformed — `seal.py:134`
- `project_seal(project_name, user_dir, *, migration_mode="in_place", config=None, embedding=None)` — encrypts every content file (`bm25_index/`, `vector_store_data/`, `meta.json`, `.dynamic_graph.snapshot.json`) atomically per-file, idempotent, crash-resumable (persists `seal_id` before any file write), writes the sealed marker on success — `seal.py:267`
- `get_sealed_project_record(project, user_dir)` — cheap probe returning the marker dict or `None`; used to decide whether the sealed-share path applies — `seal.py:419`
- `_should_seal(rel)` — policy function deciding which relative paths must be encrypted vs stay plaintext (`bm25_index/`, `vector_store_data/`, `meta.json`, `.dynamic_graph.snapshot.json` sealed; `.security/`, `version.json`, `store_meta.json`, `.sealed` never sealed) — reused by `cache.py` (tamper check, imported at `cache.py:327`) and `share._hard_revoke` — `seal.py:107` (semi-private but cross-module reused; worth knowing about)
- `_resolve_project_dir(project_name, user_dir)` — resolves a project path against an explicit `user_dir` (not the process-global `PROJECTS_ROOT`), honoring `subs/` nesting — also duplicated by `project_pack._resolve_project_dir` for the same reason (see Section 6's "Possible internal overlap") — `seal.py:256`
- Constants: `SEALED_MARKER_PATH`, `SEAL_SCHEMA_VERSION`, `CIPHER_SUITE_AES_256_GCM_V1`, `_SEAL_MAX_FILE_BYTES` (1 GiB per-file cap — also imported by `share._hard_revoke`) — `seal.py:67-76`

### `src/axon/security/share.py`
Role: The largest module — sealed-share generation/redemption/revocation between an owner and a grantee, using per-share KEK wrapping so the owner's master never leaves their machine. Covers SEALED1 (legacy) and SEALED2 (v0.4.0+, carries an Ed25519 owner pubkey for TTL-sidecar verification) envelope formats, soft revoke (delete wrap) and crash-safe hard revoke (DEK rotation + selective re-wrap of surviving shares), plus grantee-side DEK storage with OS-keyring/session/file-fallback resolution.

- `is_sealed_share_envelope(decoded)` — True if a base64-decoded string starts with `SEALED1:` or `SEALED2:` — centralizes prefix routing so callers don't hardcode either — `share.py:106`
- `generate_sealed_share(owner_user_dir, project, grantee, key_id, *, expires_at=None)` — mints a per-share token+KEK, wraps a copy of the project DEK, persists the KEK under the owner's master (enables selective re-wrap later), builds a SEALED2 `share_string` embedding the owner's Ed25519 pubkey, optionally writes a signed expiry sidecar — `share.py:563`
- `redeem_sealed_share(grantee_user_dir, share_string)` — parses SEALED1/SEALED2 envelopes, unwraps the DEK via the share token, persists it to the grantee's keyring (or file fallback per `keyring_mode`), writes a `mount_type="sealed"` mount descriptor — `share.py:727`
- `get_grantee_dek(key_id, user_dir=None)` — fetches the grantee's cached DEK; runs the TTL expiry check first (`ShareExpiredError` if elapsed/tampered/unverifiable), resolves via keyring then file fallback — the read path used at `switch_project` time — `share.py:1032`
- `delete_grantee_dek(key_id, user_dir=None)` — removes both the keyring entry and file-fallback copy of a grantee DEK; idempotent, returns whether anything was deleted — `share.py:1752`
- `revoke_sealed_share(owner_user_dir, project, key_id, *, rotate=False)` — dispatches to soft or hard revoke — `share.py:1164`
  - Soft (`_soft_revoke`, internal but conceptually reusable): deletes the wrap/KEK/expiry sidecar for one key_id; cached grantee DEKs still work until a hard rotate — `share.py:1228`
  - Hard (`_hard_revoke`, internal): rotates the project DEK + `seal_id`, re-encrypts every content file (crash-safe via staged `dek.wrapped.rotating` + `.sealing.rotation` marker), selectively re-wraps surviving shares using their persisted KEKs, invalidates shares lacking a persisted KEK (legacy projects) — `share.py:1419`
- `list_sealed_share_key_ids(project_dir)` — enumerates active `.wrapped` key_ids under `.security/shares/`, filtering out sync-engine conflict-file artifacts (OneDrive/Dropbox/GDrive naming patterns) — `share.py:1132`
- `share_wrap_path(project_dir, key_id)` / `share_kek_path(project_dir, key_id)` / `share_expiry_path(project_dir, key_id)` — path builders for the three per-share sidecar files, each validating `key_id` against a strict filename-safe pattern — `share.py:177`, `share.py:390`, `share.py:183`
- Constants: `SEALED_SHARE_PREFIX` (`SEALED1`), `SEALED_SHARE_PREFIX_V2` (`SEALED2`), `SHARE_KEYRING_PREFIX`, `SHARE_DIR_NAME`, `HKDF_INFO` — `share.py:126-145`

### `src/axon/security/signing.py`
Role: Deterministic Ed25519 signing-keypair derivation from the owner's master key — no separate key file to lose; used to sign/verify the sealed-share TTL expiry sidecars introduced in v0.4.0.

- `derive_signing_keypair(master)` — HKDF-SHA256(master, fixed salt, info=`"axon-share-signing-v1"`) → 32-byte seed → `Ed25519PrivateKey`; deterministic per master, domain-separated from the share-KEK HKDF — `signing.py:80`
- `pubkey_to_hex(pubkey)` — encodes an Ed25519 public key as 64-char lowercase hex for embedding in SEALED2 share strings — `signing.py:110`
- `pubkey_from_hex(hex_str)` — decodes/validates a 64-char hex pubkey string back to `Ed25519PublicKey`; raises `SecurityError` on malformed input — `signing.py:122`
- `get_signing_pubkey_hex(owner_user_dir)` — convenience wrapper: loads the master + derives + hex-encodes in one call — `signing.py:152`
- Constants: `SIGNING_HKDF_INFO`, `SIGNING_PUBKEY_HEX_LEN` — `signing.py:53-77`

### `src/axon/security/wordlist.py`
Role: Bundled EFF large Diceware wordlist (7,776 words) for generating human-typeable, high-entropy sealed-store passphrases.

- `generate_passphrase(n_words=6, separator=" ")` — draws `n_words` random words via `secrets.choice` (~12.92 bits/word, 4–12 words allowed) — `wordlist.py:58`
- `estimate_entropy_bits(n_words)` — returns `n_words * log2(7776)` rounded to 1 decimal, for UI "your passphrase has X bits" hints — `wordlist.py:90`
- `_load_wordlist()` — `lru_cache`d loader of the bundled EFF wordlist resource file — `wordlist.py:32` (internal but the only source of truth for the word pool; worth knowing if extending)

**Possible internal overlap**

- **`master.get_or_create_project_dek` / `master.get_project_dek` vs `share._hard_revoke`'s inline DEK rotation** — `_hard_revoke` (share.py:1419) re-implements DEK-wrap staging (`_stage_rotation`, `_read_staged_dek`) rather than reusing `master.get_or_create_project_dek`'s atomic-write pattern, because it needs crash-safe two-phase promotion that `master.py` doesn't support. Not true duplication (different atomicity requirements) but a future generalization of "atomic wrapped-key persist with resume" in `master.py` could serve both call sites. Still present, unchanged since the 2026-08-28 audit — this file was not touched by any of the 31 commits since.
- **`master._write_keyring_record`/`_read_keyring_record` vs `share._write_grantee_dek_fallback`/`_read_grantee_dek_fallback`** — both implement the same "keyring write, and also/instead write an AES-KW-wrapped-under-master file fallback, 0600 perms, atomic tempfile+replace" pattern independently (`master.py:211` and `share.py:510`/`541`). Same shape, duplicated logic across two files — a shared helper (e.g. `keyring_or_fallback_write(service, path, payload)`) would remove the duplication. Still present, unchanged.
- **`seal._should_seal` is a "private" function reused across three modules** (`cache.py:327` imports it for the tamper check — line moved from `:358` after an unrelated `_pid_alive` extraction in `cache.py`, `share.py:1440` imports it for hard-revoke's re-seal loop) — it is effectively public API despite the underscore prefix and the leading-underscore naming undersells its cross-module importance for anyone searching for "what counts as sealed content." Still present.
- **Three near-identical atomic-write-with-tempfile-then-os.replace idioms** appear independently in `crypto.SealedFile.write`, `master.get_or_create_project_dek`, `seal._write_sealed_marker`/`project_seal`, and `share.py` (wrap/KEK/expiry sidecar writers) — no shared `atomic_write_bytes()`/`atomic_write_text()` helper exists in this file set; each call site reimplements `tmp = path.with_suffix(...); write; os.replace(tmp, path)` (with inconsistent cleanup-on-failure handling in a couple of spots). **Still present and, as of this audit, explicitly deferred rather than fixed**: a separate 2026-08-28 fix (commit `2427be1`) added exactly this kind of shared primitive — `_atomic_persist.write_bytes_if_changed()` / `write_text_if_changed()`, siblings to the existing JSON-only `write_json_if_changed()` — but only migrated `config.py`'s writers to it; its own commit message names `security/*.py`'s hand-rolled writers as one of the "remaining lower-risk sites... left for a follow-up pass." None of `crypto.py`, `master.py`, `seal.py`, or `share.py` import `_atomic_persist` as of this audit. See also the top-level "Cross-subsystem" note for the equivalent JSON-case helper already used elsewhere (`graph_rag.py`/`code_graph.py`).

---

## 8. REST API Layer

Covers `src/axon/api.py`, `api_schemas.py`, `surface_contract.py`, and all modules under `src/axon/api_routes/`. Verified endpoint count for this file set: **75** distinct routes across 12 routers, each mounted twice (bare + `/v1` prefix) in `api.py` — `grep -rE '@router\.(get|post|put|delete|patch)\(' src/axon/api_routes/*.py` returns 76 hits, but one (`_rate_limit.py:12`) is inside that module's own usage-example docstring, not a live route. CLAUDE.md's currently-stated "76 REST endpoints" appears to be this same raw grep count including that docstring line; worth a follow-up fix to CLAUDE.md's recount, out of scope for this doc.

### src/axon/api.py

**Role**: FastAPI app factory — owns app-lifecycle (`lifespan`), the global `brain` singleton, cross-cutting middleware (request-id, Prometheus, API-key auth, CORS), static/docs mounting, router registration, and the `axon-api` CLI entry point (`main()`). Nearly everything else in this subsystem depends on module-level state defined here.

- `get_brain()` — FastAPI dependency; returns the global `AxonBrain` or raises 503 if not initialized — `api.py:50`
- `get_brain_optional()` — same but returns `None` instead of raising, for routes that tolerate an uninitialized brain — `api.py:57`
- `_evict_old_jobs()` — TTL/cap eviction for the in-memory `_jobs` ingest-job-status store (never evicts `processing` jobs) — `api.py:81`
- `_check_dedup(text, project)` / `_record_dedup(text, doc_id, project)` / `_purge_dedup(doc_ids, project)` — module-level source-level content-hash dedup store (`_source_hashes: dict[project, dict[hash, {doc_id, last_ingested_at}]]`); shared by every ingest route (text, batch, URL) and — via `AxonBrain.delete_documents()` (`main.py:1143`), not called directly from the route anymore — by delete too (see `POST /delete` below) — `api.py:108,125,134`
- `_get_user_dir()` — returns `Path(brain.config.projects_root)` or raises 503 — `api.py:166`
- `lifespan(app)` — async context manager: loads `AxonConfig`, enforces single-instance-per-store guard via `server_client.find_live_server_for_store`/`write_store_lock`, auto-inits the store, constructs the global `AxonBrain`, fires a background PyPI update-check, and releases the lock + closes the brain on shutdown — `api.py:181`
- `_auto_init_store(config)` — first-run helper; calls `projects.ensure_user_project()` only when `store_meta.json` is missing — `api.py:263`
- `add_request_id` middleware — stamps `X-Request-ID`/`X-Axon-Surface` on every request + response and threads the request id into the logging contextvar — `api.py:353`
- `metrics_middleware` — records per-request count/latency into the Prometheus exporter via `axon.api_routes.metrics.record_request`, using the matched route template (not raw path) to avoid label cardinality blowup — `api.py:377`
- `api_key_middleware` — enforces `X-API-Key` header (constant-time compare) when `RAG_API_KEY` env var is set; bypass list (`_AUTH_BYPASS_EXACT`/`_AUTH_BYPASS_PREFIX`) covers health/metrics/docs/gui/brand assets — `api.py:435`
- `_load_cors_origins_from_disk()` — reads `api.allow_origins` straight from config.yaml (not via `AxonConfig.load()`) to wire `CORSMiddleware` without import-time side effects; defined as a nested function inside the `try:` block that imports `CORSMiddleware`, so it only exists (and only runs) when `fastapi.middleware.cors` is importable — `api.py:313`
- `_resolve_bind_address(args, env)` — resolves `(host, port)` for `axon-api`: `--host`>`AXON_HOST`>`0.0.0.0`; `--port`>`AXON_PORT`>`config.yaml api_port`>`8420` — `api.py:581`
- `main(argv)` — `axon-api` CLI entry point; resolves bind address, writes it back to `AXON_HOST`/`AXON_PORT` env vars (so `lifespan()`'s independent read stays in sync for the lock file), then `uvicorn.run()`; catches `EADDRINUSE` with a friendly message — `api.py:618`
- `_ROUTERS` tuple (12 routers) + registration loop — every router is mounted twice: bare and under `/v1` prefix, giving REST/versioned parity for free — `api.py:529-546`
- Branded `/docs`, `/redoc`, `/favicon.ico`, `/gui/` static mount, `/brand/` static mount — `api.py:455-509`

### src/axon/api_schemas.py

**Role**: Pydantic request/response models for the whole REST surface plus a handful of pure security/validation helpers (path traversal guard, content hashing) reused by multiple route modules. This is the schema contract layer — new endpoints should add models here rather than inlining `BaseModel` subclasses in route files (though a few route files, e.g. `config_routes.py`, `security_routes.py`, still do that for locally-scoped request bodies).

- `_validate_ingest_path(path)` — resolves and validates a filesystem path against `RAG_INGEST_BASE` and a blocked-system-path list (`C:/Windows`, `/etc`, `/proc`, etc.); raises 403 — `api_schemas.py:71`
- `_compute_content_hash(text)` — SHA-256 of normalized text, via the Rust bridge when available, else `hashlib` fallback — `api_schemas.py:97`
- `_BLOCKED_PATH_PREFIXES` — tuple of resolved system-root paths blocked from ingest — `api_schemas.py:45`
- `_VALID_PROJECT_NAME_RE` — regex enforcing 1-5 slash-separated segments, `[a-z0-9_-]{1,50}` each — the canonical project-name validator, reused across `projects.py`, `shares.py`, `maintenance.py`, `governance.py` — `api_schemas.py:68`
- `QueryRequest` / `SearchRequest` / `QueryVisualizeRequest` / `SearchVisualizeRequest` — query/search bodies with RAG-toggle overrides (hyde, multi_query, step_back, rerank, hybrid, top_k, threshold, dry_run, include_diagnostics/citations) — `api_schemas.py:118,173,191,199`
- `GraphRetrieveRequest` — body for `/graph/retrieve`; validates `federation_weights` keys against `_VALID_FEDERATION_KEYS = {graphrag, dynamic_graph}` and rejects negative weights in `__init__` — `api_schemas.py:212`
- `IngestRequest` / `TextIngestRequest` / `BatchDocItem` / `BatchTextIngestRequest` / `URLIngestRequest` / `DeleteRequest` — ingest/delete payloads — `api_schemas.py:262-328`
- `ProjectSwitchRequest` (with `.final_name` property reconciling `project_name`/`name` aliases) / `ProjectCreateRequest` — `api_schemas.py:337,352`
- `ProjectRotateKeysRequest` / `ProjectSealRequest` — `api_schemas.py:544,548`
- `ProjectPackRequest` (`project_name`, optional `out_path` — defaults to `~/.axon/packs/<name>-<timestamp>.axonpack.zip`) / `ProjectUnpackRequest` (`zip_path`, optional `as_name`, `force`) — bodies for the new `POST /project/pack`/`POST /project/unpack` endpoints (v0.4.6, commit `f2e81f9`) — `api_schemas.py:553,564`
- `StoreInitRequest`, `ShareGenerateRequest`, `ShareRedeemRequest`, `ShareRevokeRequest`, `ShareExtendRequest` — store/share lifecycle bodies (note `ttl_days` semantics documented inline) — `api_schemas.py:377-436`
- `CopilotMessage` / `CopilotAgentRequest` / `CopilotTaskResult` — GitHub Copilot bridge payloads — `api_schemas.py:444,449,503`
- `ConfigUpdateRequest` — curated ~32-field subset of `AxonConfig` with `extra="allow"` so unmodelled keys can be reported as `ignored` rather than silently dropped — `api_schemas.py:457`
- `MaintenanceStateRequest`, `SecurityBootstrapRequest`/`SecurityUnlockRequest`/`SecurityChangePassphraseRequest` (using `SecretStr` to keep passphrases out of logs/`repr`) — `api_schemas.py:508,516-535`
- `SearchResult` — response model for `/search` — `api_schemas.py:370`
- Field caps: `MAX_QUERY_FIELD_CHARS`, `MAX_TEXT_FIELD_CHARS`, `MAX_URL_FIELD_CHARS`, `MAX_SHARE_STRING_CHARS`, `MAX_PASSPHRASE_CHARS` — centralized DoS-guard constants — `api_schemas.py:19-34`

### src/axon/surface_contract.py

**Role**: Declarative registry mapping every user-facing capability to which surfaces (API/REPL/CLI/VSCode) support it, with documented intentional exceptions. This is the cross-interface-parity source of truth the project's own CLAUDE.md/MEMORY.md rules ("cross-interface parity" development rule) point back to — a new feature's surface coverage should be registered here, not just implemented. **`Surface.WEBAPP` was removed in `#157`** (the Streamlit UI deletion) — the enum now has 4 members, not 5, and `ALL_SURFACES`/registry entries shrank accordingly with no `intentional_exceptions` loss (every capability's `supported_surfaces` had inherited `WEBAPP` only via `ALL_SURFACES`, never named it explicitly).

- `Tier` enum (`ONE`/`TWO`/`API_ONLY`) and `Surface` enum (`API`/`REPL`/`CLI`/`VSCODE`) — classification vocabulary — `surface_contract.py:29,35`
- `Capability` dataclass — `{id, name, category, tier, description, supported_surfaces, intentional_exceptions, api_route, docs_targets, test_targets}` — `surface_contract.py:52`
- `ALL_SURFACES` / `NO_VSCODE` (API+REPL+CLI, no VSCODE) / `PRIMARY_SURFACES` (API+REPL+CLI+VSCODE) — reusable `frozenset[Surface]` shorthands for `supported_surfaces` — `surface_contract.py:42,45,48`
- `REGISTRY: list[Capability]` — **42** registered capabilities (up from ~35 at the last audit) spanning query/ingest/collection/project/config/share/store/graph/session/governance/maintenance/security categories, each cross-referencing its `api_route` — `surface_contract.py:74-538`
- `capabilities_by_category()` — groups registry entries by `category` — `surface_contract.py:540`
- `tier1_capabilities()` — filters `Tier.ONE` entries (required on every surface) — `surface_contract.py:548`
- `surface_capabilities(surface)` — capabilities supported on a given surface — `surface_contract.py:553`
- `unsupported_on(surface)` — `(capability, reason)` pairs for Tier 1/2 capabilities missing from a surface, using the documented exception or a "no explicit exception" fallback — used by parity-audit tooling/tests — `surface_contract.py:558`

### src/axon/api_routes/__init__.py

**Role**: Tiny shared-helper module for the route package — two cross-cutting guards every write/query route composes with.

- `_enforce_write_access(brain, operation)` — calls `brain._assert_write_allowed(operation)`, translating `PermissionError` (e.g. maintenance-mode readonly/draining) into HTTP 403 — used by `query.py`, `ingest.py`, `graph.py`, `governance.py` — `__init__.py:5`
- `enforce_project(requested, brain)` — raises 409 if the caller's `project` field doesn't match the brain's single active project (the brain is a singleton serving one project at a time); points callers at `POST /project/switch` — used by `query.py`, `ingest.py`, `graph.py` — `__init__.py:13`

### src/axon/api_routes/_rate_limit.py

**Role**: Self-contained in-process sliding-window rate limiter (no external deps/Redis) shared across any route that needs per-IP throttling. Named "buckets" keep independent endpoints' counters from bleeding into each other.

- `enforce_rate_limit(request, *, bucket, max_hits=10, window_seconds=60.0)` — raises HTTP 429 if the caller's IP exceeds `max_hits` within the sliding window for `bucket`; used by `ingest_url`, `ingest/upload`, `share_generate`, `share_redeem`, `security_bootstrap`, `security_change_passphrase` — `_rate_limit.py:47`
- `_get_ip(request)` — best-effort client IP, honoring `X-Forwarded-For` first segment — reused directly by `security_routes.py`'s unlock-lockout logic to keep the two IP-identity notions consistent — `_rate_limit.py:35`
- Internal: `_buckets` dict capped at `_MAX_IPS = 10_000` with oldest-IP eviction to bound memory under unique-IP churn — `_rate_limit.py:27,32`
- The module docstring's usage example (`@router.post("/my/endpoint")`) is not a real route — don't count it when grepping this file for endpoint totals.

### src/axon/api_routes/health.py

**Role**: Kubernetes-style liveness/readiness probes, split out from a legacy single `/health` endpoint.

- `GET /health/live` — `health_live()` — always 200 if the ASGI process is responding; does not check the brain — `health.py:41`
- `GET /health/ready` — `health_ready()` — 200 only once `axon.api.brain` is initialized, else 503 `{"status": "initializing"}` — `health.py:52`
- `GET /health` — `health_check()` — backward-compat alias for `/health/ready` — `health.py:67`
- `_readiness_payload()` — shared body/status-code logic for `/health/ready` and `/health` — `health.py:24`

### src/axon/api_routes/metrics.py

**Role**: Prometheus exposition endpoint + the metric definitions/recorder functions called from `api.py`'s middleware and from `query.py`/`ingest.py` handlers. Owns the dedicated `CollectorRegistry` (not the global one) so test reloads of `axon.api` don't trip duplicate-registration errors.

- `GET /metrics` — `metrics_endpoint()` — returns Prometheus 0.0.4 text exposition, or 503 plain text if `prometheus-client` isn't installed — `metrics.py:175`
- `record_request(path, method, status, duration_seconds)` — increments `axon_requests_total` + observes `axon_request_duration_seconds`; called from `api.py`'s `metrics_middleware` — `metrics.py:120`
- `record_query(project, surface)` — increments `axon_query_total`; called at the start of `POST /query` — `metrics.py:135`
- `record_ingest(project, surface)` — increments `axon_ingest_total`; called at the start of every ingest route — `metrics.py:150`
- `update_brain_ready(is_ready)` — sets `axon_brain_ready` gauge; refreshed on every `/metrics` scrape — `metrics.py:165`
- All four are safe no-ops when `prometheus-client` is missing (`PROMETHEUS_AVAILABLE` guard) — reusable pattern for any future optional-dependency metric.

### src/axon/api_routes/query.py

**Role**: The core RAG query/search/clear surface — the primary "ask a question" and "retrieve chunks" entry points most future features will call or wrap.

- `POST /query` — `query_brain(request)` — full RAG pipeline: builds an overrides dict from RAG toggles, runs `brain.query()` or (dry_run) `brain.search_raw()` off the event loop via `run_in_executor` with a configurable timeout (`AXON_QUERY_TIMEOUT`, default 120s), returns `response`+`settings`+`provenance`+optional `diagnostics`/`sources`/`citations`; maps `MountSyncPendingError` to 503 + `X-Axon-Mount-Sync-Pending` header — `query.py:22`
- `POST /query/stream` — `query_brain_stream(request)` — SSE streaming variant; bridges `brain.query_stream()` (a sync generator) into the async event loop one chunk at a time via a single dedicated `ThreadPoolExecutor(max_workers=1)` so provider-client thread-local state stays consistent across `next()` calls — `query.py:147`
- `POST /search` — `search_brain(request)` — returns ranked chunks only (no LLM), via `brain._execute_retrieval()` — `query.py:231`
- `POST /search/raw` — `search_raw_endpoint(request, include_trace=False)` — retrieval + full diagnostics (and optional retrieval trace) without an LLM call, via `brain.search_raw()` — `query.py:259`
- `POST /clear` — `clear_brain()` — wipes the active project's vector store/BM25/hash-store/entity-graph via `collection_ops.clear_active_project(brain)`, then purges the in-memory dedup bucket for that project — `query.py:296`

### src/axon/api_routes/ingest.py

**Role**: All document-ingestion surfaces (path, text, batch text, URL, multipart upload) plus the async-job status poller, collection listing, staleness check, and chunk deletion. The largest route file; establishes the async-job pattern (`job_id` + `GET /ingest/status/{job_id}`) reused only here today but is the template for any future long-running operation.

- `POST /ingest/refresh` — `refresh_docs(background_tasks)` — background job that re-ingests any tracked file whose MD5 content hash has drifted from `brain.get_doc_versions()`'s recorded hash; returns `job_id` immediately — `ingest.py:64`
- `POST /ingest` — `ingest_data(request, background_tasks, req)` — path-based ingest (file or directory) as a background job with progress callback (`phase`, `files_total`, `chunks_total`, `chunks_embedded`); emits `governance.emit()` events (`ingest_started`/`completed`/`failed`) — `ingest.py:163`
- `GET /ingest/status/{job_id}` — `get_ingest_status(job_id)` — polls `_api._jobs`; also merges in `community_build_in_progress` from the graph backend — `ingest.py:292`
- `GET /tracked-docs` — `list_tracked_docs()` — lists `brain.get_doc_versions()` — `ingest.py:312`
- `GET /collection` — `get_collection()` — lists ingested files + chunk counts via `brain.list_documents()` — `ingest.py:323`
- `GET /collection/stale` — `get_stale_docs(days=7)` — scans the module-level `_api._source_hashes` dedup store for docs not re-ingested within `days` — `ingest.py:342`
- `POST /add_text` — `add_text(request)` — single-doc text ingest with dedup check via `_api._check_dedup`/`_record_dedup`, uses `SmartTextLoader` — `ingest.py:371`
- `POST /add_texts` — `add_texts(request)` — batch text ingest in one embedding call; dedups both against the persisted store and within the same batch — `ingest.py:418`
- `POST /ingest_url` — `ingest_url(request, req)` — fetches + ingests a URL via `URLLoader`; rate-limited (`ingest_url` bucket, 20/60s) — `ingest.py:481`
- `POST /ingest/upload` — `ingest_upload(req, files, project)` — multipart upload; streams each file to a temp dir in 1MB chunks enforcing `AxonConfig.max_upload_bytes`/`max_files_per_request` caps, sanitizes filenames via `_normalise_uploaded_filename`, dedupes collisions, then batch-ingests; rate-limited (`ingest_upload`, 30/60s) — `ingest.py:532`
- `POST /delete` — `delete_documents(request, req)` — **as of `#169`, a thin wrapper**: calls `brain.delete_documents(request.doc_ids)` (`main.py:1143`) and emits the `delete` governance audit event; no longer contains its own chunk-ID-expansion/dedup-purge logic inline — that moved into `AxonBrain.delete_documents()` itself so the CLI (`--delete-doc`/`--delete-doc-id`) and `agent.py::_tool_delete_documents` share the exact same behavior instead of three drifting copies — `ingest.py:684`
- `_normalise_uploaded_filename(filename, index)` — safe-basename sanitizer for uploads (alnum + `-_.` only) — `ingest.py:54`
- `_project_label(brain, request_project)` — stable project label for Prometheus counters — `ingest.py:37`
- `MAX_UPLOAD_BYTES` / `MAX_FILES_PER_REQUEST` — module-level fallback caps used before brain config is available — `ingest.py:33,34`

### src/axon/api_routes/projects.py

**Role**: Project CRUD/switching, session listing, sealed-project key rotation/sealing, project pack/unpack (backup/restore), and — despite the file name — the general `/config` read/update endpoints (config validation/reset/set-by-key live in `config_routes.py` instead; see overlap note below).

- `GET /config` — `get_config()` — returns `asdict(brain.config)` with `_SENSITIVE_FIELDS` masked to `"***"` — `projects.py:81`
- `POST /config/update` — `update_config(request)` — applies a curated ~32-field `ConfigUpdateRequest` subset to `brain.config`, reports `applied`/`ignored` keys, reinitializes `brain.llm`/`brain.embedding`/`brain.reranker` when relevant fields change, optional `persist` to config.yaml — `projects.py:101`
- `GET /projects` — `get_projects()` — lists on-disk projects + memory-only projects + share mounts; also validates received shares/sealed-shares and merges security-store lock status — `projects.py:166`
- `POST /project/new` — `create_project(request)` — validates name against `_VALID_PROJECT_NAME_RE`, gates `security_mode="sealed_v1"` on local turboquantdb, calls `projects.ensure_project()` — `projects.py:239`
- `POST /project/switch` — `switch_project(request)` — switches the brain's active project under `_switch_lock`; validates mount revocation for `mounts/*` projects before switching — `projects.py:278`
- `POST /mount/refresh` — `refresh_mount()` — re-reads the owner's version marker for a mounted share and reopens handles if newer; 503 + `X-Axon-Mount-Sync-Pending` header on `MountSyncPendingError` — `projects.py:326`
- `POST /project/delete/{name}` — `delete_project_endpoint(name)` — blocks deleting projects with active shares (409) or `mounts/*` entries (400); switches away first if it's the active project — `projects.py:372`
- `POST /project/rotate-keys` — `rotate_project_keys(request)` — rotates crypto keys for a sealed project via `security.project_rotate_keys` — `projects.py:413`
- `POST /project/seal` — `seal_project(request)` — converts an open project to `sealed_v1` via `security.project_seal`; switches away/back if it's the active project — `projects.py:443`
- `POST /project/pack` — `pack_project_route(request)` — zips a project (`axon.project_pack.pack_project`) to a server-side path, defaulting to `~/.axon/packs/`; switches off the target project first if it's active, switches back after — new in commit `f2e81f9` (v0.4.6) — `projects.py:485`
- `POST /project/unpack` — `unpack_project_route(request)` — restores a `.axonpack.zip` (`axon.project_pack.unpack_project`) into `AxonStore` as a new/overwritten project; same switch-away-if-active guard, only when `force=true` collides with the active project — new in commit `f2e81f9` — `projects.py:513`
- `GET /sessions` — `list_sessions()` — lists saved chat sessions for the active project — `projects.py:541`
- `GET /session/{session_id}` — `get_session(session_id)` — loads one session; validates id against `_SESSION_ID_RE`, now imported from `axon.sessions` (no longer defined locally in this file) — `projects.py:554`
- `_require_local_turboquantdb(brain, action)` — guard raising 400 unless vector store is local turboquantdb (no remote qdrant) — gates sealing/seal-related ops — `projects.py:32`
- `_mask_if_sensitive(flat_key, value)` — masks a config value if its key is in `_SENSITIVE_FIELDS`; explicitly shared with (imported by) `config_routes.py` to keep masking behavior in lock-step across `/config`, `/config/update`, `/config/set` — `projects.py:68`
- `_SENSITIVE_FIELDS` — frozenset of API-key-shaped config fields always masked in responses — `projects.py:51`

### src/axon/api_routes/config_routes.py

**Role**: Config validation, reset-to-defaults, and the generic dot-notation "set any AxonConfig field" endpoint — the general escape hatch for the ~139 config fields not covered by `/config/update`'s curated subset.

- `GET /config/validate` — `validate_config()` — runs `AxonConfig.validate(config_path)` and returns structured issues (`level`, etc.) — `config_routes.py:215`
- `POST /config/reset` — `reset_config()` — **as of `#128`/the config.yaml atomic-write consolidation**, writes `_DEFAULT_CONFIG_YAML` to the user config path via the shared `axon._atomic_persist.write_text_if_changed()` (digest-cached, crash-safe rename) instead of a plain `open(...).write()`; still does NOT reload the running brain — `config_routes.py:236`
- `POST /config/set` — `set_config_field(request)` — sets a single field via dot-notation key (e.g. `chunk.strategy`) resolved through `resolve_config_key`; reinitializes runtime components as needed; optional persist; masks sensitive old/new values in the response — `config_routes.py:256`
- `resolve_config_key(key)` — 3-tier resolution: curated `_DOT_TO_FLAT` alias → bare `AxonConfig` field name → last dotted segment; returns `None` if nothing matches — the mechanism that makes `/config/set` accept effectively any of the ~240 `AxonConfig` fields, not just the curated ~101 in `_DOT_TO_FLAT` — `config_routes.py:148`
- `_DOT_TO_FLAT` — large curated dict mapping friendly dotted keys (`llm.provider`, `rag.graph_rag_depth`, …) to flat dataclass field names; the canonical reference for "what dotted alias maps to what config field" — `config_routes.py:27`
- `_reinitialize_runtime_components(brain, changed_keys)` — re-constructs `brain.llm`/`brain.embedding`/`brain.reranker` when their backing config fields change — mirrors (does not share code with) the analogous inline logic in `projects.py::update_config` — `config_routes.py:189`
- `ConfigSetRequest` — `{key, value, persist=True}` — locally-defined Pydantic model (not in `api_schemas.py`) — `config_routes.py:183`

### src/axon/api_routes/graph.py

**Role**: GraphRAG/knowledge-graph status, finalize (community rebuild), conflict inspection, direct backend retrieval, and HTML/JSON graph export/visualization — including combined query+graph visualization pages.

- `GET /graph/status` — `get_graph_status()` — community build progress, summary count, entity count, code-node count, `graph_ready` flag — `graph.py:21`
- `POST /graph/finalize` — `finalize_graph(request)` — triggers `backend.finalize(True)` off-thread; emits `graph_finalize` governance event; returns `not_applicable` status for backends without a community step — `graph.py:45`
- `GET /graph/conflicts` — `graph_conflicts(project, limit)` — lists facts with `status="conflicted"` via `backend.list_conflicts()`; returns `supported: false` for backends lacking the method (e.g. graphrag) — `graph.py:99`
- `POST /graph/retrieve` — `graph_retrieve(request)` — runs the active backend's `retrieve()` directly with a `RetrievalConfig` (supports `point_in_time` + per-query `federation_weights`), bypassing the full `/query` pipeline/LLM — `graph.py:138`
- `POST /query/visualize` — `query_visualize(request)` — runs a query (RAPTOR/GraphRAG off by default) and returns a self-contained HTML page combining the answer, sources, and highlighted entity+code graph via `brain.render_query_graph_html()` — `graph.py:234`
- `POST /search/visualize` — `search_visualize(request)` — same HTML visualization but retrieval-only (no LLM answer) — `graph.py:281`
- `GET /graph/visualize` — `get_graph_visualization()` — standalone entity-graph HTML export via `brain.export_graph_html()` — `graph.py:331`
- `GET /graph/backend/status` — `graph_backend_status()` — raw `backend.status()` dict — `graph.py:348`
- `GET /graph/data` — `graph_data(project)` — entity/relation graph as JSON nodes+links, for VS Code webview — `graph.py:366`
- `GET /code-graph/data` — `code_graph_data(project)` — code-structure graph JSON via `brain.build_code_graph_payload()` — `graph.py:378`
- `_resolve_graph_payload(brain)` — normalizes `backend.graph_data()` output (handles `.to_dict()`, dict, or missing) to `{nodes, links}` — shared by `/graph/data` and both `/*/visualize` HTML routes — `graph.py:191`
- `_serialise_context(ctx)` — converts a `GraphContext` dataclass to a JSON-friendly dict (ISO timestamps, etc.) for `/graph/retrieve` — `graph.py:213`

### src/axon/api_routes/governance.py

**Role**: Operator-console read APIs (aggregated overview, audit log, Copilot session list, per-project state) plus a small set of "audited operator wrapper" control endpoints that re-invoke logic also reachable via plainer routes elsewhere (see overlap note).

- `GET /governance/overview` — `governance_overview()` — aggregates project/maintenance/graph/stale-doc/active-job/lease/Copilot-session counts via `_build_overview` — `governance.py:89`
- `GET /governance/audit` — `governance_audit(project, action, surface, status, since, limit)` — filtered audit-log query via `governance.get_store().query()` — `governance.py:98`
- `GET /governance/copilot/sessions` — `governance_copilot_sessions(limit)` — active + recent Copilot bridge sessions — `governance.py:139`
- `GET /governance/projects` — `governance_projects()` — all projects with maintenance state + graph stats — `governance.py:168`
- `POST /governance/graph/rebuild` — `governance_graph_rebuild(req)` — audited wrapper around `backend.finalize(True)`; **near-duplicate of `POST /graph/finalize`** in `graph.py` (see overlap note, still present) — `governance.py:204`
- `POST /governance/project/maintenance` — `governance_set_maintenance(name, state, req)` — audited wrapper around `apply_maintenance_state`; **near-duplicate of `POST /project/maintenance`** in `maintenance.py` (see overlap note, still present) — `governance.py:260`
- `POST /governance/copilot/session/{session_id}/expire` — `governance_expire_session(session_id, req)` — force-closes a stuck Copilot bridge session — `governance.py:310`
- `_build_overview(brain, jobs)` — the aggregation helper backing `/governance/overview`; pulls from `runtime.get_registry()`, `maintenance.get_maintenance_status()`, `projects.list_projects()`, the graph backend, `brain.get_stale_docs()`, and `governance.get_session_store()` — reusable pattern for any future "operator dashboard" aggregation — `governance.py:18`

### src/axon/api_routes/maintenance.py

**Role**: Project maintenance-state get/set, the GitHub Copilot chat-agent bridge (SSE streaming with slash-command dispatch), and the Copilot LLM task queue/result submission endpoints.

- `POST /copilot/agent` — `copilot_agent_handler(request, body)` — SSE handler for GitHub Copilot chat; dispatches `/search`, `/ingest`, `/projects` slash-commands or falls through to `brain.query()` with chat history — `maintenance.py:23`
- `GET /llm/copilot/tasks` — `get_copilot_tasks()` — drains the module-level `_copilot_task_queue` (from `axon.main`) for the VS Code Copilot bridge to poll — `maintenance.py:98`
- `POST /llm/copilot/result/{task_id}` — `submit_copilot_result(task_id, body)` — resolves a pending Copilot task's `asyncio.Event` with the result/error — `maintenance.py:109`
- `POST /project/maintenance` — `set_project_maintenance(request, req)` — sets maintenance state via `apply_maintenance_state`; emits governance event; **near-duplicate of `POST /governance/project/maintenance`** — `maintenance.py:123`
- `GET /project/maintenance` — `get_project_maintenance(name)` — reads current maintenance state via `get_maintenance_status`; 404s if `meta.json` missing — `maintenance.py:152`

### src/axon/api_routes/registry.py

**Role**: Single-purpose write-lease inspection endpoint for the governance/maintenance system.

- `GET /registry/leases` — `get_registry_leases()` — returns active write-lease counts per tracked project via `runtime.get_registry().snapshot_all()` — used to confirm it's safe to enter maintenance state — `registry.py:9`

### src/axon/api_routes/security_routes.py

**Role**: Sealed-store security lifecycle — status, keyring-mode switching, cache wiping, passphrase bootstrap/unlock/lock/change, and a passphrase-suggestion helper. Owns its own IP-based lockout tracker for unlock attempts (separate from the generic `_rate_limit.py` bucket system, but reuses its `_get_ip` helper).

- `GET /security/status` — `security_status()` — sealed-store status + (if the optional `[sealed]` extra is installed) keyring mode + session cache size — `security_routes.py:63`
- `POST /security/keyring-mode` — `security_set_keyring_mode(request)` — changes the in-process keyring mode (`persistent`/`session`/`never`); 400 with install hint if `[sealed]` extra missing — `security_routes.py:89`
- `POST /security/wipe-sealed-cache` — `security_wipe_sealed_cache()` — wipes the active sealed-project plaintext cache; always 200, `wiped: bool` — `security_routes.py:146`
- `GET /suggestions/passphrase` — `suggest_passphrase(words=6, separator="-")` — Diceware passphrase generator from the bundled EFF wordlist; pure helper, no store access — `security_routes.py:170`
- `POST /security/bootstrap` — `security_bootstrap(request, req)` — bootstraps the sealed store with a passphrase; rate-limited (`security_bootstrap`, 10/60s) — `security_routes.py:200`
- `POST /security/unlock` — `security_unlock(request, req)` — unlocks the sealed store; custom IP-lockout (5 failures / 300s window, only credential failures count) separate from the generic rate limiter — `security_routes.py:213`
- `POST /security/lock` — `security_lock()` — locks the sealed store — `security_routes.py:249`
- `POST /security/change-passphrase` — `security_change_passphrase(request, req)` — rotates the store passphrase; rate-limited — `security_routes.py:259`
- `_current_user_dir()` — resolves the user's projects_root from the live brain or a fresh `AxonConfig.load(None)` when no brain is up yet — `security_routes.py:53`
- `_evict_stale_unlock_failures(now)` — amortized eviction of the `_unlock_failures` dict, capped at `_UNLOCK_FAILURES_MAX_IPS = 10000` — `security_routes.py:31`

### src/axon/api_routes/shares.py

**Role**: AxonStore initialization/status/identity plus the full share-key lifecycle (generate/redeem/revoke/extend/list), branching internally between legacy "open" shares and `sealed_v1` shares (SEALED1/SEALED2 envelope detection).

- `POST /store/init` — `store_init(request)` — moves the AxonStore to a new base path, rebuilds `AxonConfig` paths, replaces the global `brain`, clears `_source_hashes`/`_jobs`; warns about projects that become unreachable — `shares.py:31`
- `GET /store/status` — `store_status()` — safe to call before brain is ready; reads `store_meta.json` directly to drive first-run setup UI — `shares.py:98`
- `GET /store/whoami` — `store_whoami()` — current OS username, store path, user dir, active project — `shares.py:146`
- `POST /share/generate` — `share_generate(request, req)` — branches on whether the project is sealed (`ssk_` key_id via `security.generate_sealed_share`) or open (`shares.generate_share_key`); validates `ttl_days > 0`; rate-limited (10/60s) — `shares.py:165`
- `POST /share/redeem` — `share_redeem(request, req)` — detects `SEALED1:`/`SEALED2:` envelope (with legacy JSON-shape fallback) and routes to `security.redeem_sealed_share` or `shares.redeem_share_key` — `shares.py:263`
- `POST /share/revoke` — `share_revoke(request, req)` — routes on `ssk_` key_id prefix to sealed revoke (`security.revoke_sealed_share`, with soft vs. `rotate=true` hard revoke) or legacy `shares.revoke_share_key` — `shares.py:333`
- `POST /share/extend` — `share_extend(request, req)` — renews or clears a share key's `ttl_days` via `shares.extend_share_key` — `shares.py:429`
- `GET /share/list` — `share_list()` — merges open + sealed shares (`sharing` and `shared` lists), tagging each with `security_mode`; also runs revocation validation and reports `removed_stale` — `shares.py:472`

**Possible internal overlap**

1. **Graph community rebuild duplicated across two endpoints — still present.** `POST /graph/finalize` (`graph.py:45`, `finalize_graph`) and `POST /governance/graph/rebuild` (`governance.py:204`, `governance_graph_rebuild`) both call `_enforce_write_access(brain, "finalize_graph")`, fetch `brain._graph_backend`, invoke `backend.finalize(True)` via `asyncio.to_thread`, and emit a `graph_finalize` governance event — with slightly different response shapes (`graph.py` includes `status`/`backend_id`/`detail`; `governance.py` includes only `status`/`community_summary_count`, and additionally emits a `status="started"` event first). No consolidation commit touched this between `2aa84f6` and `main`. Still worth consolidating into one shared helper both routes call.

2. **Maintenance-state set duplicated across two endpoints — still present.** `POST /project/maintenance` (`maintenance.py:123`, `set_project_maintenance`) and `POST /governance/project/maintenance` (`governance.py:260`, `governance_set_maintenance`) both validate the project name against `_VALID_PROJECT_NAME_RE`, call `apply_maintenance_state(name, state)`, and emit a `maintenance_changed` governance event — differing only in that the governance variant takes query params instead of a JSON body and additionally emits a `status="started"` event before the outcome. Same consolidation opportunity as #1; unchanged since the last audit.

3. **`/config` endpoints split across two files with no obvious naming cue — still present.** `GET /config` and `POST /config/update` live in `projects.py` (not `config_routes.py`, despite that file's name), while `GET /config/validate`, `POST /config/reset`, and `POST /config/set` live in `config_routes.py`. `config_routes.py`'s `set_config_field` already imports `_mask_if_sensitive` from `projects.py` to bridge the split, which is a sign the two modules are really one logical "config routes" surface. Not a functional bug (both are correctly registered in `api.py`), but a future engineer searching `config_routes.py` for "where do I add a config endpoint" would miss half of them, and `_reinitialize_runtime_components` (`config_routes.py:189`) still duplicates (rather than shares) the inline reinit-on-change logic in `projects.py::update_config` (`projects.py:101-ish`) — both independently reconstruct `brain.llm`/`brain.embedding`/`brain.reranker` on the same trigger conditions. `POST /config/reset` did get one real fix in this window (see its entry above — now atomic via `write_text_if_changed`), but that's orthogonal to the split itself.

---

## 9. User-facing Surfaces

Covers: `src/axon/cli.py`, `src/axon/repl.py`, `src/axon/mcp_server.py`, `src/axon/agent.py`,
`src/axon/ext_install.py`. **`src/axon/tools.py`, `src/axon/webapp.py`, and `src/axon/webapp_launcher.py`
were deleted in `#157` ("chore: delete the Streamlit UI, tools.py, and other dead code")** — see the
end of this section for what happened to each.

### src/axon/cli.py

**Role**: `argparse`-based entry point for the `axon` command. Parses ~90 flags into an `AxonConfig`
override, decides whether a heavy `AxonBrain` init is needed, routes store-mutating one-shot
commands either to a local brain or to a detected running `axon-api` server, and falls through to
`_interactive_repl` when no one-shot flag/query is given. Shares `_print_project_tree` and (new)
`_print_shares_listing` with `repl.py`.

- `_print_project_tree(proj_list, active, indent=0)` — recursive console renderer for the project tree (active marker, timestamp, `[merged]`/`[state]` tags) — `cli.py:24`. Reused verbatim by `repl.py` (`/project list`).
- `_print_knowledge_base_listing(rows)` — **new** (`#TBD`, commit `fcf6d68`): renders the `--list` "Knowledge Base — N file(s), M chunk(s)" table from any list of `{"source", "chunks"}` dicts; extracted after an audit found `main()`'s two `--list` code paths (the `doc_versions.json` fast path with no brain, and the `brain.list_documents()` path) printing byte-identical tables from independently-written loops — `cli.py:42`
- `_print_shares_listing(sharing, shared, sealed_sharing, *, indent="  ")` — **new** (commit `8dca2c3`): pure-rendering helper for the "Shares — issued by me (legacy/sealed)" / "Shares — received" table; extracted so `cli.py`'s `--share-list` and `repl.py`'s `/share list` can't drift again — a real parity bug was found and fixed in the same commit (REPL's `/share list` had no sealed-shares section at all). Callers own their own data fetch and `indent` (cli.py uses 2sp, REPL uses 4sp) — `cli.py:62`
- `_run_via_server(server, args, config)` — mirrors local `--project-new`/`--project`/`--project-delete`/`--ingest` handling but over HTTP against a detected `axon-api` server, so a second local `AxonBrain` never opens the same store — `cli.py:101`
- `_write_python_discovery()` — writes the current Python executable path to `~/.axon/.python_path` so the VS Code extension can locate the interpreter regardless of pip/venv/pipx install — `cli.py:181`
- `_cli_migrate_vectors_to_tqdb(brain, source_path_arg)` / `_cli_migrate_vectors(brain, chroma_path_arg)` — ChromaDB/LanceDB → TurboQuantDB or LanceDB migration, auto-detecting source backend by directory contents — `cli.py:195`, `cli.py:239`
- `_is_first_run(args)` / `_snapshot_first_run_state(args)` — detects a fresh checkout (no config file, empty `AxonStore/`) to trigger the setup wizard; snapshot must be taken **before** `AxonConfig.load()` auto-creates the config file — `cli.py:321`, `cli.py:373`
- `_run_axon_update(argv)` — handles the bare `axon update [-y]` subcommand (intercepted before argparse); confirmation gate lives here as a plain inline `input(...).strip().lower()` (not routed through `repl.py`'s new `_confirm` helper — see overlap note), `axon.update_check.run_update()` assumes permission already granted — `cli.py:381`
- `main()` — the `axon` entry point; builds the full argparse surface (RAG toggles, graph/sealed-store/share/governance/session/index-management flags), resolves `AxonConfig`, sets up per-PID rotating file logging, decides `need_brain`, and dispatches to REPL or one-shot handlers — `cli.py:441`

Notable one-shot flag groups implemented inline in `main()` (each pairs with an equivalent REPL slash-command and/or MCP tool — see "Possible internal overlap"):
- Store lifecycle: pre-brain fast path (`--wipe-sealed-cache`, `--passphrase-generate`) at `cli.py:1166-1210`; post-brain block (`--store-bootstrap/-unlock/-lock/-change-passphrase`, `--project-seal`) at `cli.py:1710-1800`; **`--store-init` is handled twice** — once pre-brain at `cli.py:1710` and again in a post-brain-init duplicate path at `cli.py:2442-2467` (mirrors the sharing duplication below).
- Sharing: `--share-list/-generate/-redeem/-revoke/-extend`, `--share-ttl-days`, `--share-rotate` — pre-brain block `cli.py:1856-2070`, post-brain-init duplicate path `cli.py:2469-2650`. The **listing** render (`--share-list`) is no longer duplicated (both call `_print_shares_listing`); generate/redeem/revoke/extend logic is still two independent copies within this file, plus a third in `repl.py`.
- Governance proxy: `--governance [overview|audit|sessions|projects|graph-rebuild]` — `cli.py:2075-2105`; **as of commit `3b1b7ef`, no longer raw `urllib.request`** — now routes through `server_client._request()`/`_headers()`/`resolve_api_base()` so it sends `X-API-Key` like every other server_client-backed command (the old hand-rolled version silently 401'd against a key-protected `axon-api`).
- Graph ops: `--graph-status/-finalize/-conflicts/-retrieve/-export`, `--graph-at` — `cli.py:2311-2415`.
- Doc lifecycle: `--refresh`, `--list-stale`, `--delete-doc`, `--delete-doc-id`, `--optimize-index`, `--migrate-vectors` — `cli.py:2240-2300`, `:2456-2472`, `:2667`. **`--delete-doc`/`--delete-doc-id` now both call `brain.delete_documents(...)` (`main.py:1143`)**, the same single implementation the REST `POST /delete` route and `agent.py::_tool_delete_documents` call (fixed in `#169`) — no more separate chunk-ID-expansion logic in the CLI. `--refresh` still hand-rolls its own `hashlib.md5` comparison inline (see overlap note; the hash *algorithm* itself was fixed in `3b1b7ef` to actually match `AxonBrain.ingest()`'s MD5 — previously it used `api_schemas._compute_content_hash()`'s SHA-256, which never matched, so "skip unchanged" silently never fired).

### src/axon/repl.py

**Role**: Interactive REPL — markdown/LaTeX rendering pipeline for terminal display, animated
header/init UI, prompt_toolkit-based input with live tab completion, `@file`/`@folder` context
attachment, `!shell` passthrough, and the single giant slash-command dispatcher
(`_process_input_sync`, ~2500 lines, one `elif cmd ==` branch per command — see `_SLASH_COMMANDS`
at `repl.py:38` for the full command list, already documented in `docs/` as the REPL reference).
This catalog lists the **underlying reusable machinery**, not each of the ~35 top-level commands.

**Markdown / math rendering pipeline** (chainable text-processing helpers, used by `_preprocess_markdown`):
- `_fence_unfenced_code(text)` — auto-detects and fences bare code blocks by language signature (Python/JS/TS/Rust/Go/SQL/bash/Java/C) — `repl.py:165`.
- `_normalize_bullets(text)` — converts `•` bullets to markdown `-` — `repl.py:291`.
- `_mathify(text)` / `_is_formula_line(line)` / `_fence_math_formulas(text)` / `_inline_math_symbols(text)` — ASCII-math → Unicode conversion (superscript/subscript maps, Greek-word substitution) and formula-line detection/fencing — `repl.py:394`, `:448`, `:489`, `:526`.
- `_preprocess_markdown(text)` — top-level pipeline combining task-list/strikethrough/callout substitution + the above — `repl.py:548`.
- `_latex_to_unicode(formula)` — LaTeX command table (Greek, operators, blackboard-bold, `\frac`, super/subscripts) → Unicode terminal string — `repl.py:737`.
- `_make_math_renderable(text)` / `_render_rich_with_math(text, console)` — returns/prints a Rich renderable that splits `$$block$$`/`$inline$` math into bordered panels vs. backtick-wrapped code spans — `repl.py:777`, `:823`.

**Credential / provider helpers**:
- `_save_env_key(env_name, key)` — persists a key to `~/.axon/.env` + `os.environ` — `repl.py:828`. Used by `/keys set`, `/model`, `_prompt_key_if_missing`.
- `_prompt_key_if_missing(provider, brain)` — prompts (getpass, or GitHub OAuth device flow for `github_copilot`) when a cloud provider's key/token is unset, patches `brain.config` in place — `repl.py:839`.
- `_confirm(prompt, *, read_fn=input) -> bool` — **new shared helper (commit `df49862`)**: yes/no prompt parser, treats EOF/Ctrl+C as "no" rather than crashing the session, accepts `y`/`yes`. Routed through `_read_input` at REPL call sites so scripted-test input and prompt_toolkit terminal handling still apply. See overlap note — this collapsed most (not all) of the REPL's duplicated confirmation logic — `repl.py:990`.
- `_infer_provider(model)` — guesses LLM provider (`gemini`/`openai`/`ollama`) from a bare model name; imported directly by `cli.py:main()` for `--model` — `repl.py:1006`.

**Completion**:
- `_make_completer(brain)` — readline-based completer (slash commands, `/ingest` paths, `/model`/`/pull` model names via Ollama or GitHub Copilot listing) — `repl.py:907`. Superseded interactively by the richer `_PTCompleter` class defined inline in `_interactive_repl` when `prompt_toolkit` is available.

**Context/token display**:
- `_estimate_tokens(text)` / `_token_bar(used, total, width=20)` — rough 4-chars/token estimate + a filled-bar string — `repl.py:1021`, `:1026`.
- `_show_context(brain, chat_history, last_sources, last_query)` — renders the full `/context` box (model info, token-usage bar, RAG settings, last 10 chat turns, last 8 retrieved sources, full system prompt) — `repl.py:1035`.
- `_do_compact(brain, chat_history)` — LLM-summarizes chat history into a single `[Conversation summary]:` turn, replacing history in place — `repl.py:1231`.

**Header / banner UI**:
- `_box_width()` — terminal-width-aware inner box width (min 43) — `repl.py:1273`.
- `_get_brain_anim_row(row_idx, frame, width)` / `_anim_pad` — animated ASCII "brain" pulse-path renderer for the header — `repl.py:1314`.
- `_build_header(brain, tick_lines=None)` / `_draw_header(brain, tick_lines=None)` — build/print the full pinned welcome header (model, embedding, search/discuss/hybrid/top-k status, init ticks) — `repl.py:1402`, `:1454`.
- `_print_recent_turns(history, n_turns=2)` — prints the last N Q&A turns below the header — `repl.py:1474`.
- `class _InitDisplay(logging.Handler)` — intercepts init-time log records and renders an animated alternate-screen spinner box during heavy `AxonBrain` construction; collects `tick_lines` for the final header — `repl.py:1497`. Used by both `cli.py:main()` and (implicitly) any long-running init.

**Context attachment**:
- `_expand_at_files(text)` — expands `@file`/`@folder/` references in user input into inlined file contents (loader-aware for `.docx/.pptx/.pdf/.xlsx/...`, byte-capped per-file and per-folder) — `repl.py:1699`.

**Config / shell helpers**:
- `_handle_config_cmd(arg, brain, cfg_path)` — `/config [show|validate|wizard|reset|set <key> <value>]` dispatcher; `set` resolves dot-notation keys via the same `resolve_config_key` used by `POST /config/set`; `reset` now goes through the shared `write_text_if_changed()` atomic writer (same fix as `POST /config/reset`, commit `2427be1`) instead of a plain file write — `repl.py:1820`.
- `_find_bash()` — auto-detects bash (`bash` in PATH → `wsl` → known Git Bash paths on Windows) — `repl.py:1914`. Reused by `agent.py`'s `_tool_run_shell`.
- `_resolve_bash(setting)` — resolves `repl.shell` config value (`native`/`wsl`/`gitbash`/`bash`/`auto`) to a command prefix — `repl.py:1940`.

**Main entry**:
- `_interactive_repl(brain, stream=True, init_display=None, quiet=False, _scripted_inputs=None)` — the REPL loop itself: builds a `prompt_toolkit` full-screen `Application` (fallback: readline), session persistence via `axon.sessions`, background PyPI update check, and the `_process_input_sync(user_input)` closure that is the single dispatch point for every slash command, `!shell` passthrough, `@file` expansion, and plain queries (streaming/non-streaming, agent-mode) — `repl.py:1964`, dispatcher at `:2530`.
  - Confirmation-prompt pattern: **mostly consolidated now** — `/clear` (`:3031`), `/project new` (`:2800`), `/project delete` (`:3467`), `/update` (`:3734`), `/config reset` (`:1857`), and the agent-mode tool-confirmation callback `_confirm_cb` (`:4772`) all now call the shared `_confirm()` helper (`repl.py:990`) instead of hand-parsing `input(...)`. `cli.py`'s `axon update` flow (`cli.py:423`) is the one remaining holdout still doing its own inline `input("...").strip().lower()` — see overlap note.
  - Agent-mode tool-confirmation callback `_confirm_cb(msg)` — suspends the `prompt_toolkit` Application via `in_terminal()` before reading stdin, then calls `_confirm(...)`; passed to `agent.run_agent_loop` as `confirm_cb` — `repl.py:4759`.
  - `_agent_step_cb(tool_name, result)` — collects per-tool-call output for display after an agent-mode turn completes — `repl.py:4812`.
  - `/share` slash-command block — `repl.py:3747-4027` (now imports and calls `_print_shares_listing` from `cli.py` for `/share list`; generate/redeem/revoke/extend remain independently implemented here).
  - `/store` slash-command block — `repl.py:4028-4197`.
  - `/governance` slash-command block — `repl.py:4583-4658`; **as of `3b1b7ef`, routes through `server_client._request()`/`_headers()`** (was raw `urllib.request`) — same X-API-Key fix as `cli.py --governance`, but each surface still independently maintains its own subcommand-to-route table and output formatting (REPL renders per-subcommand tables for `audit`/`projects`; `cli.py` just JSON-dumps everything).
  - `/refresh` slash-command — `repl.py:4227` — still an independent third MD5-hash-comparison implementation (see overlap note).

### src/axon/agent.py

**Role**: Shared in-process agentic tool-calling loop used by the REPL (`/agent` mode). Defines one
OpenAI-format tool schema set, a dispatcher that executes each tool directly against a live
`AxonBrain`, and the multi-turn loop driving `OpenLLM.complete_with_tools`. (Its own docstring's
claim that the Streamlit web GUI also used this module is now moot — that GUI was deleted in `#157`;
`WEBAPP_TOOLS` was deleted in the same commit as dead code, see below.)

- `REPL_TOOLS: list[dict]` — 21 OpenAI/Ollama-format tool schemas (`ingest_path`, `list_knowledge`, `search_knowledge`, `add_text`, `purge_source`, `delete_documents`, `clear_project`, `list_projects`, `switch_project`, `create_project`, `get_config`, `ingest_url`, `update_settings`, `ingest_texts`, `get_stale_docs`, `delete_project`, `graph_status`, `graph_finalize`, `refresh_ingest`, `run_shell`, `write_file`, `read_file`) — unchanged set since the last audit; note it still has **no** `pack_project`/`unpack_project` equivalent even though `mcp_server.py` gained both in commit `f2e81f9` (new cross-surface parity gap, see overlap note) — `agent.py:33`.
- `_DESTRUCTIVE_TOOLS` — `{purge_source, delete_documents, clear_project, delete_project, run_shell}`, gated behind `confirm_cb` — `agent.py:609`.
- `ToolCall = namedtuple("ToolCall", ["name", "args", "thought_signature"], defaults=[None])` — return type of `OpenLLM.complete_with_tools()`; `thought_signature` now defaults to `None` so older/simpler call sites don't have to pass it — `agent.py:27`.
- `dispatch_tool(brain, tool_name, args, *, confirm_cb=None)` — executes one tool call, running the confirm-message lookup + `confirm_cb(msg)` gate for destructive tools before dispatching to a `_tool_*` implementation — `agent.py:622`.
- `_make_vision_fn(brain)` — returns an OCR callable `(image_bytes) -> str` backed by `brain.llm.complete_with_image`, or `None` when vision OCR is disabled/unsupported; shared by `_tool_ingest_path`, `_tool_purge_source`, `_tool_refresh_ingest` — `agent.py:710`.
- `_tool_ingest_path/_tool_list_knowledge/_tool_search_knowledge/_tool_add_text/_tool_purge_source/_tool_delete_documents/_tool_clear_project/_tool_list_projects/_tool_switch_project/_tool_create_project/_tool_get_config/_tool_ingest_url/_tool_update_settings/_tool_ingest_texts/_tool_get_stale_docs/_tool_delete_project/_tool_graph_status/_tool_graph_finalize/_tool_refresh_ingest/_tool_write_file/_tool_read_file/_tool_run_shell` — one implementation per `REPL_TOOLS` entry, each operating directly on the passed `brain` (in-process, not HTTP) — `agent.py:735-1450`.
  - `_tool_delete_documents` — as of `#169`, resolves the source's chunk IDs then calls `brain.delete_documents(ids_to_delete)` (`main.py:1143`), the same single implementation `POST /delete` and the CLI's `--delete-doc`/`--delete-doc-id` call — `agent.py:1020`.
  - `_tool_refresh_ingest` — still its own independent MD5-hash-comparison implementation (a third copy alongside `cli.py --refresh` and `repl.py /refresh`), correctly matching `AxonBrain.ingest()`'s MD5 digest (a comment at `agent.py:1310` documents the earlier SHA-256 mismatch bug this avoided) — `agent.py:1280`.
- `_SENSITIVE_CONFIG_FIELDS` — frozenset of `AxonConfig` field names masked as `***` by `_tool_get_config` so a cloud LLM driving the agent loop never sees a raw API key — `agent.py:1092`.
- `_UPDATABLE_SETTINGS: dict[str, type]` — field→caster map consumed by `_tool_update_settings` — `agent.py:1146`.
- `_AGENT_SYSTEM_PROMPT` — the fixed system prompt steering tool choice (ingestion decision tree, project handling, retry-loop avoidance rules) — `agent.py:1481`.
- `run_agent_loop(llm, brain, prompt, chat_history, *, tools=None, confirm_cb=None, step_cb=None, max_steps=8)` — the multi-turn loop: calls `complete_with_tools`, dispatches tool calls, detects identical-call retry loops (forces an LLM summary after 2 identical calls), feeds `<tool_results>` back into `chat_history` (with a `__tool_calls__` side-channel for providers like Gemini needing native function-call parts) — `agent.py:1545`.

### src/axon/mcp_server.py

**Role**: `FastMCP`-based stdio MCP server (`axon-mcp` console script). Every tool is a **thin async
HTTP proxy** (`_get`/`_post`) onto a running `axon-api` REST server — no direct `AxonBrain` access,
unlike `agent.py`'s in-process dispatch. **56 tools total** (verified: `grep -c "@mcp.tool()"
mcp_server.py`, matches CLAUDE.md); grouped below by concern rather than listed individually since
the full reference lives in `docs/`. Two changes since the last audit: the `refresh_mount` tool was
**removed** in `#157` (it posted to `/mount/refresh` with no arguments, which `mount_refresh(project=None)`
already does as a strict superset — 57→56), and `pack_project`/`unpack_project` were **added** in commit `f2e81f9`.

- `_headers()` — builds request headers with `X-API-Key` (from `RAG_API_KEY` env) and `X-Axon-Surface: mcp` attribution — `mcp_server.py:70`.
- `_get(path, params=None)` / `_post(path, body)` — shared async `httpx` GET/POST helpers (60s timeout) used by every `@mcp.tool()` function — `mcp_server.py:78`, `:85`.
- `API_BASE` (`RAG_API_BASE` env, default `http://localhost:8420`) / `API_KEY` (`RAG_API_KEY` env) — module-level config — `mcp_server.py:61,64`.
- Tool groups (all `@mcp.tool()` async functions, name → REST route):
  - Ingest/retrieval: `ingest_text`, `ingest_texts`, `ingest_url`, `ingest_path`, `refresh_ingest`, `get_job_status`, `search_knowledge`, `query_knowledge`, `list_knowledge` — `mcp_server.py:100-246`; `query_stream` (accumulates SSE chunks into one response) — `mcp_server.py:908`.
  - Project mgmt: `switch_project`, `delete_documents`, `list_projects`, `get_stale_docs`, `create_project`, `delete_project`, `clear_knowledge` — `mcp_server.py:255-333`; `pack_project`, `unpack_project` — new in commit `f2e81f9`, `mcp_server.py:696,709`.
  - Config: `get_current_settings`, `update_settings` — `mcp_server.py:341,349`; `get_config`, `set_config`, `update_config`, `validate_config` — `mcp_server.py:981-1029`.
  - Sessions: `list_sessions`, `get_session` — `mcp_server.py:400,406`.
  - Sharing: `share_project`, `redeem_share`, `list_shares`, `revoke_share`, `extend_share` — `mcp_server.py:415-501`.
  - Sealed-store security: `get_store_status`, `init_store`, `security_status`, `wipe_sealed_cache`, `set_keyring_mode`, `suggest_passphrase`, `security_bootstrap`, `security_unlock`, `security_lock`, `security_change_passphrase`, `seal_project` — `mcp_server.py:515-672`.
  - Graph: `graph_status`, `graph_finalize`, `graph_data`, `graph_backend_status`, `graph_conflicts`, `graph_retrieve` — `mcp_server.py:724-785`.
  - Governance/ops: `get_active_leases`, `governance_overview`, `governance_audit`, `governance_sessions`, `governance_projects`, `governance_graph_rebuild`, `mount_refresh` — `mcp_server.py:814-958`. (`refresh_mount` removed — see above.)
- `main()` — `axon-mcp` console-script entry point (`configure_logging()` + `mcp.run()`) — `mcp_server.py:1046`.

### src/axon/ext_install.py

**Role**: `axon-ext` console-script — installs the VS Code extension VSIX bundled inside the
`axon-rag` package (`src/axon/extensions/`) via the `code` CLI. Unchanged since the last audit.

- `_find_vsix()` — locates the bundled `.vsix` under `axon.extensions` package resources (`importlib.resources.files`), picking the lexicographically-latest if multiple exist — `ext_install.py:19`.
- `_find_code_cmd()` — resolves `code` or `code-insiders` on PATH via `shutil.which` — `ext_install.py:32`.
- `install_vscode_extension()` — `axon-ext` entry point; supports `--path` (print VSIX path), `--version` (print extension version from filename), or runs `code --install-extension <vsix>` — `ext_install.py:39`.

### What happened to the deleted files (`#157`)

- **`src/axon/tools.py`** (704 LOC, `get_rag_tool_definition()`) — deleted outright. It had zero importers anywhere in `src/`; it was a third, drifted copy of the tool schema `mcp_server.py` and `agent.py` already defined independently. `examples/agent_orchestration.py` was updated in the same PR to import `axon.agent.REPL_TOOLS` instead.
- **`src/axon/webapp.py`** (1,319 LOC, the deprecated Streamlit UI / `axon-ui`) and **`src/axon/webapp_launcher.py`** — deleted outright, along with the `axon-ui` console script and the `[ui]` extra. No module in `src/axon` imported `webapp.py`. Its capability is superseded by the native web GUI `axon-api` already serves at `/gui/` — see `api.py`'s `gui_dir`/`StaticFiles` mount in section 8. That GUI (`src/axon/gui/`) is static assets only (`index.html`, `css/`, `js/`, `assets/`) — no Python, nothing to catalog here.
- `agent.py::WEBAPP_TOOLS` (a `REPL_TOOLS` minus `_DESTRUCTIVE_TOOLS` subset built for the webapp integration) was deleted in the same commit — it was dead code even before `webapp.py`'s removal, since `webapp.py` never imported `axon.agent`.
- `Surface.WEBAPP` was retired from `surface_contract.py`'s registry (see section 8).

**Possible internal overlap**

1. **Two independent OpenAI-format tool-schema sets describing largely the same Axon capability surface, under inconsistent names — down from three, still present.** `tools.py`'s third copy is gone (see above), leaving:
   - `agent.py::REPL_TOOLS` (21 tools, direct in-process dispatch against `AxonBrain` — used by REPL `/agent` mode).
   - `mcp_server.py` (56 `@mcp.tool()` functions, thin HTTP proxy to `axon-api` — used by MCP clients).
   Concepts still overlap almost 1:1 (ingest/search/list/switch-project/config) but with **different tool names** for the same operation — e.g. add-text is `add_text` (agent.py) vs. `ingest_text` (mcp_server.py); clear is `clear_project` (agent.py) vs. `clear_knowledge` (mcp_server.py). The two sets have also drifted on parameters for the *same*-named tool: `update_settings` in `mcp_server.py` (`:349`) accepts `crag_lite`, `code_graph`, `graph_rag_mode`, `cite`, and `persist`; `agent.py`'s `REPL_TOOLS` `update_settings` schema (`:319`) stops at `sentence_window_size` and has none of those five. And the two sets have diverged on *coverage*, not just naming: `mcp_server.py` gained `pack_project`/`unpack_project` in commit `f2e81f9` with no `agent.py` equivalent (agent-mode/REPL users can't pack or unpack a project; only MCP/REST clients can) — a genuine new cross-surface parity gap, not just a naming drift.

2. **`agent.py::WEBAPP_TOOLS` dead code — fixed.** Deleted in `#157` along with `webapp.py` itself (see above); no longer applicable.

3. **No shared confirmation-prompt helper — mostly fixed (commit `df49862`).** `repl.py` now has `_confirm(prompt, *, read_fn=input) -> bool` (`repl.py:990`) and routes `/clear`, `/project new`, `/project delete`, `/update`, `/config reset`, and the agent-mode `_confirm_cb` through it. The one remaining holdout is `cli.py`'s `axon update` flow, which still does its own `input("...").strip().lower() == "y"` at `cli.py:423` rather than importing `repl.py::_confirm` (or a shared module both could import from without a cross-file dependency in the other direction). Worth finishing: move `_confirm` to a location both `cli.py` and `repl.py` can import from (today `repl.py` already imports from `cli.py`, so `cli.py` importing back from `repl.py` would be a new cycle — a `axon/cli_shared.py`-style module, per item 4 below, would resolve this cleanly).

4. **`cli.py` and `repl.py` still re-implement several features end-to-end**, though two of the five originally identified have had their *rendering* layer consolidated:
   - Project CRUD: `cli.py:main()`'s `--project`/`--project-new`/`--project-delete` blocks (fast-path pre-brain block `cli.py:1684-1730`, post-brain block `cli.py:2189-2230`, plus `_run_via_server`'s own copy `cli.py:101-180`) vs. `repl.py`'s `/project new|switch|delete` (`repl.py:3309-3480`). **Still fully duplicated control flow**, unlike shares below.
   - Share lifecycle: generate/redeem/revoke/extend (including sealed-share auto-detection via base64-decode + `SEALED1:`/`SEALED2:` prefix sniffing) is still duplicated **three times** — `cli.py`'s pre-brain block (`:1873-2070`), `cli.py`'s post-brain block (`:2505-2650`), and `repl.py`'s `/share` (`:3747-4027`). The **listing** sub-feature (`--share-list` / `/share list`) is no longer duplicated — both now call the shared `_print_shares_listing()` (`cli.py:62`, commit `8dca2c3`), which also fixed a real parity bug (REPL's listing was silently missing sealed shares).
   - `/store` (init/status/bootstrap/unlock/lock/change-passphrase): `cli.py:1166-1210` + `:1710-1800` + `:2442-2467` (duplicate `store_init`) vs. `repl.py:4028-4197`. Unchanged, still duplicated.
   - `/refresh` (re-ingest changed docs by content-hash comparison): all **three** surfaces still hand-roll their own `hashlib.md5` comparison independently — `cli.py:2240-2283`, `repl.py:4227`, `agent.py::_tool_refresh_ingest` (`agent.py:1280`) — but as of commit `3b1b7ef`, all three now correctly compute the same digest `AxonBrain.ingest()` actually writes to `_doc_versions` (raw MD5, no stripping). Previously `cli.py`'s copy used `api_schemas._compute_content_hash()` (SHA-256 of stripped text), which never matched, so "skip unchanged" silently never fired there — a real bug, now fixed, though the duplication itself (three independent implementations) persists.
   - `/governance` proxy: no longer raw `urllib.request` (fixed in `3b1b7ef` — both now route through `server_client._request()`/`_headers()`, fixing a real `X-API-Key` auth bug), but `cli.py:2075-2105` and `repl.py:4583-4658` still independently maintain their own subcommand-to-route table and output formatting (REPL pretty-prints `audit`/`projects` as tables; `cli.py` just JSON-dumps every subcommand's response).
   These remain strong candidates for a shared `axon/cli_shared.py`-style module of pure functions (`do_project_new`, `do_share_generate`, `do_refresh`, `do_governance(sub, base)`, etc.) that both `cli.py` flag handlers and `repl.py` slash-command handlers call. The listing-render extractions (`_print_project_tree`, `_print_knowledge_base_listing`, `_print_shares_listing`, all living in `cli.py` and imported by `repl.py`) show the pattern already works for the read-only half of each feature; the mutating half (generate/redeem/revoke/delete/etc.) hasn't had the same treatment yet.

5. **`webapp.py`'s separate session-persistence mechanism — fixed (file deleted).** No longer applicable; `axon.sessions` (`~/.axon/sessions/`) is now the only session store, used consistently by `cli.py`, `repl.py`, and MCP.

---

## 10. Infra, Rust Bridge & Integrations

### `src/axon/_atomic_persist.py`
**Role:** Shared atomic-write-if-changed helpers for persistence, extracted from `GraphRagMixin._gr_write_json_if_changed` so `CodeGraphMixin._save_code_graph` (and any future persistence code) share one digest-cached, crash-safe write path instead of maintaining divergent implementations. As of commit `2427be1`, grew two non-JSON siblings so `config.py`'s `AxonConfig.save()`, the first-run config scaffold write, and all three independent "reset config.yaml to defaults" call sites (`cli.py --config-reset`, `api_routes/config_routes.py POST /config/reset`, `repl.py /config reset`) go through the same atomicity contract instead of each doing a plain `open(...).write()`.

- `write_json_if_changed(path, payload, cache, *, sort_keys=False)` — atomically writes JSON to `path` via SHA-1-digest-gated skip-if-unchanged, using `axon.version_marker._atomic_replace` for a crash/OneDrive-safe rename; returns `True` if it wrote, `False` if skipped — `_atomic_persist.py:19`
- `write_bytes_if_changed(path, payload, cache)` — **new** (commit `2427be1`): same digest-cache/skip-if-unchanged/Windows-safe-replace contract as `write_json_if_changed`, for raw `bytes` payloads (msgpack, pre-encoded YAML/text, key material) that aren't JSON. Pass a throwaway `{}` for `cache` on one-shot writers with no cross-call digest reuse need — `_atomic_persist.py:62`
- `write_text_if_changed(path, text, cache, *, encoding="utf-8")` — **new** (commit `2427be1`): thin wrapper over `write_bytes_if_changed()` for plain-text content (YAML, `.env`-style key=value files, newline-joined lists) — `_atomic_persist.py:96`

### `src/axon/_lru_ttl_cache.py`
**Role:** **New module** (commit `8eb1d7f`) — shared LRU+TTL cache algorithm extracted from two independently-written, structurally-identical implementations: `query_router.py`'s `_query_cache` (response cache) and `graph_rag.py`/`graphrag_engine.py`'s `_traversal_cache` (BFS entity-expansion cache). Both had hand-rolled the same `OrderedDict` + monotonic-time-in-tuple + TTL-check + move-to-end-on-hit + popitem-on-overflow dance. Deliberately **not** a wrapping class: callers keep their own plain `OrderedDict`/`threading.Lock` attributes (several tests construct/inject into those directly), and these functions just operate on them; the stored tuple shape is caller-defined and preserved verbatim — the functions only own the leading `time.monotonic()` slot and the LRU/TTL bookkeeping around it.

- `lru_ttl_get(store, lock, key, ttl)` — returns the cached tuple for `key` if present and not expired (`ttl <= 0` means never expires), else `None`; moves the entry to most-recently-used on a hit, deletes it on a TTL miss — `_lru_ttl_cache.py:25`
- `lru_ttl_put(store, lock, key, *value_parts, maxsize)` — stores `(time.monotonic(), *value_parts)` under `key`, evicting the LRU entry first only when `key` is genuinely new and the store is already at `maxsize` (an update to an existing key never evicts) — `_lru_ttl_cache.py:48`

### `src/axon/_pid_check.py`
**Role:** **New module** (commit `8d3e933`-adjacent consolidation work) — shared cross-platform PID-liveness check, extracted from `axon.security.cache._pid_alive` so `server_client.py`'s single-instance store lock (see `api.py`'s `lifespan()` in section 8) can reuse the same already-verified logic instead of a second hand-rolled implementation.

- `pid_alive(pid)` — best-effort cross-platform liveness check via `os.kill(pid, 0)`; on unrecognised/ambiguous errors it deliberately returns `True` (treats the PID as alive) since both callers' failure modes are worse if a live process is wrongly treated as dead (wiping an in-use cache, or letting a second server start against a store the first still owns) — `_pid_check.py:14`

### `src/axon/_rust_loader.py`
**Role:** Dev-workflow bootstrap that prefers a freshly `cargo build`-ed Rust extension over a possibly-stale bundled `.pyd`/`.so`/`.dylib`, so editable installs pick up new Rust functions without a full package rebuild.

- `bootstrap_dev_rust_module(package_name, package_dir)` — main public entry point; if a newer `target/release` artifact exists than the bundled extension, preloads it into `sys.modules[f"{package_name}.axon_rust"]` before the normal import runs, and applies API-drift shims; returns whether it substituted a module — `_rust_loader.py:82`
- `_patch_loaded_module(module)` — applies narrow Python-side shims for native API drift (currently wraps `run_louvain` to a stable call signature, preserving the raw native fn as `_native_run_louvain`) — `_rust_loader.py:61` (semi-private, but the pattern is the place to add future native-API compat shims)
- `_load_extension_module(module_name, artifact_path)` — generic loader that imports an arbitrary compiled extension file (any `.pyd`/`.so`) via `importlib.machinery.ExtensionFileLoader` and registers it in `sys.modules`; reusable beyond just Rust artifacts if another native extension needs manual loading — `_rust_loader.py:50`
- `_platform_dev_artifact_name()` — returns the per-OS dev build filename (`axon_rust.dll`/`libaxon_rust.dylib`/`libaxon_rust.so`) — `_rust_loader.py:13`

### `src/axon/_ui_state.py`
**Role:** Minimal cross-module shared-state dict (deliberately not a class, to dodge import cycles) letting background work (embedding, ingestion) publish live progress text for the REPL's status bar to read.

- `state` — module-level dict, currently `{"embed_progress": ""}`; any axon module writes `state["embed_progress"] = ...`, the REPL toolbar reads it — `_ui_state.py:7`

### `src/axon/rust_bridge.py`
**Role:** The single optional-acceleration facade over the compiled `axon_rust` extension. Lazily imports the native module once, exposes one `can_X()` capability probe + one `X()` call method per native function, and — for the hot/critical paths — a matching pure-Python fallback so callers never need to branch on whether Rust is present. This is the canonical place to add bindings for any new Rust-accelerated routine (BM25, score fusion, hashing, msgpack codecs, graph/community-detection ops).

- `get_rust_bridge()` — process-wide singleton accessor for `RustBridge` — `rust_bridge.py:585`
- `RustBridge.py()` — returns the raw loaded native module (or `None`) for callers that need functions not yet wrapped — `rust_bridge.py:63`
- `RustBridge.is_available()` — whether the native module loaded at all — `rust_bridge.py:66`
- `can_bm25()` / `build_bm25_index(corpus)` / `search_bm25(index, query, top_k)` — native BM25 index build + search, normalizes result rows to `{index, score}` — `rust_bridge.py:69,72,75`
- `can_ingest_preprocess()` / `preprocess_documents(documents, batch_size)` — batched ingest-time document preprocessing — `rust_bridge.py:95,98`
- `can_symbol_search()` / `symbol_channel_search(corpora, query_tokens, top_k, filters)` — multi-corpus code-symbol channel search — `rust_bridge.py:108,111`
- `can_symbol_index()` / `build_symbol_index(corpora)` / `search_symbol_index(index, query_tokens, top_k)` — persistent symbol index build/search, normalizes rows to `{index, score, channel}` — `rust_bridge.py:131,134,137`
- `can_doc_hash()` / `compute_doc_hash(text)` — content hash used for ingest dedup. **Fallback fixed in commit `df49862`**: now `hashlib.md5(text.encode("utf-8")).hexdigest()` (no stripping) to match both the native implementation and `AxonBrain.ingest()`'s own hashing — previously the Python fallback computed `sha256(text.strip())`, silently disagreeing with the native path and producing different dedup hashes depending on whether the Rust extension loaded — `rust_bridge.py:171,174`
- `can_extract_code_tokens()` / `extract_code_query_tokens(query)` — tokenizes a code-search query into a `frozenset`, falling back to `axon.code_retrieval._extract_code_query_tokens` — `rust_bridge.py:184,187`
- `can_code_lexical_scores()` / `code_lexical_scores(results, query_tokens)` — per-result lexical relevance scoring for code chunks (symbol name/basename/qualified-name/text-hit heuristics), with a full Python fallback implementation — `rust_bridge.py:198,201`
- `can_decode_corpus_json()` / `decode_corpus_json(raw)` — fast corpus JSON decode (Rust-only, no Python fallback) — `rust_bridge.py:255,258`
- `can_corpus_msgpack()` / `encode_corpus_msgpack(texts, docs)` / `decode_corpus_msgpack(raw)` — msgpack corpus I/O codec — `rust_bridge.py:261,264,267`
- `can_sha256()` / `compute_sha256(text)` — generic SHA-256 hex digest, fallback `hashlib.sha256(text.strip().encode("utf-8")).hexdigest()` — **no longer identical to `compute_doc_hash`'s fallback** (different algorithm, different normalization) now that the latter was fixed to MD5-of-raw-text; the two serve genuinely different purposes (ingest dedup vs. a generic content hash) — `rust_bridge.py:270,273`
- `can_hash_store_binary()` / `save_hash_store_binary(path, hashes)` / `load_hash_store_binary(path)` — persists/loads the ingest dedup hash-set, Rust binary format with a plain-text (`\n`-joined sorted set) Python fallback for both directions — `rust_bridge.py:279,282,297`
- `can_sentence_codec()` / `encode_sentence_index(records, chunk_to_sentences)` / `decode_sentence_index(raw)` / `encode_sentence_meta(ids, meta)` / `decode_sentence_meta(raw)` — sentence-window index codec (for sentence-level retrieval) — `rust_bridge.py:315,318,321,324,327`
- `can_segment_text()` / `segment_text(text, max_tokens)` — native text segmentation into token-bounded chunks (Rust-only) — `rust_bridge.py:330,333`
- `can_cosine_similarity()` / `cosine_similarity(a, b)` — vector cosine similarity (Rust-only) — `rust_bridge.py:336,339`
- `can_score_fusion()` / `score_fusion_weighted(vector_scores, bm25_scores, alpha)` / `score_fusion_rrf(vector_ranks, bm25_ranks, k)` — hybrid retrieval score fusion (weighted-sum and reciprocal-rank-fusion variants) — `rust_bridge.py:343,346,359`
- `can_mmr_rerank()` / `mmr_rerank(results, lambda_mult, diversity_bias)` — Maximal Marginal Relevance diversity reranking — `rust_bridge.py:370,373`
- `can_result_postprocess()` / `dedupe_best_by_id(results)` / `filter_results_by_threshold(results, threshold, score_field="vector_score")` — post-retrieval result cleanup: best-scoring dedup by id, score-threshold filtering — `rust_bridge.py:384,387,391`
- `can_code_doc_bridge()` / `build_code_doc_bridge_edges(symbol_lookup, chunks, relations)` — builds graph edges linking code symbols to referencing doc chunks — `rust_bridge.py:408,411`
- `can_entity_graph_codec()` / `encode_entity_graph(graph)` / `decode_entity_graph(raw)` — GraphRAG entity-graph binary codec — `rust_bridge.py:424,427,430`
- `can_entity_embeddings_codec()` / `encode_entity_embeddings(embeddings)` / `decode_entity_embeddings(raw)` — entity-embedding binary codec — `rust_bridge.py:433,436,439`
- `can_relation_graph_codec()` / `encode_relation_graph(graph)` / `decode_relation_graph(raw)` — GraphRAG relation-graph binary codec — `rust_bridge.py:442,445,448`
- `can_dedup_corpus_payload()` / `build_dedup_corpus_payload(corpus)` — builds a dedup-ready corpus payload — `rust_bridge.py:451,454`
- `can_build_graph_edges()` / `build_graph_edges(entity_graph, relation_graph)` — derives `(nodes, edges)` tuple from entity+relation graphs for community detection — `rust_bridge.py:457,460`
- `can_run_louvain()` / `run_louvain(nodes, edges, resolution=1.0)` — Louvain community detection over the graph, returns `{node: community_id}` — `rust_bridge.py:476,479`
- `can_merge_entities_into_graph()` / `merge_entities_into_graph(entity_graph, results)` — merges freshly-extracted entities into the running entity graph (creates/updates nodes, tracks `chunk_ids`/`frequency`); has a complete Python fallback implementation, not just a stub — `rust_bridge.py:498,501`
- `can_entity_merge()` — **fixed (commit `df49862`)**: no longer a byte-identical duplicate of `can_merge_entities_into_graph()`'s body. Now a one-line, explicitly-documented alias (`return self.can_merge_entities_into_graph()`, with a docstring noting it's "kept for callers/tests using this name") — `rust_bridge.py:540`
- `can_relation_merge()` / `merge_relations_into_graph(relation_graph, results)` — merges extracted relations into the relation graph — `rust_bridge.py:545,548`
- `can_resolve_entity_alias_groups()` / `resolve_entity_alias_groups(embeddings, threshold)` — clusters entity-name embeddings into alias groups (index lists) by similarity threshold — `rust_bridge.py:558,561`

### `src/axon/maintenance.py`
**Role:** Operator-facing maintenance-mode orchestration that couples project-metadata state (`projects.py`) with the in-flight-write lease registry (`runtime.py`) into single atomic operations, so API handlers don't have to hand-orchestrate both systems themselves. Unchanged since the last audit.

- `apply_maintenance_state(name, state)` — sets a project's maintenance state (`normal`/`draining`/`readonly`/`offline`) and synchronizes the lease-registry drain (starts drain for non-normal states, stops it for `normal`); returns `{status, project, maintenance_state, active_leases, epoch}`; raises `ValueError` on invalid state/name — `maintenance.py:13`
- `get_maintenance_status(name)` — read-only snapshot of a project's current maintenance state + lease registry (`{project, maintenance_state, active_leases, epoch, draining}`) — `maintenance.py:46`

### `src/axon/extensions/__init__.py`
**Role:** Namespace-only package (no code, a single comment) holding the bundled VS Code extension VSIX artifact (`axon-copilot-0.4.6.vsix`) as package data, so `axon-ext`'s `_find_vsix()` (`ext_install.py:19`, section 9) can locate and install it without the user downloading anything separately. Not previously cataloged; added here since it's an explicit scope target.

### `src/axon/integrations/__init__.py`
**Role:** Namespace docstring only (no code) — documents that `langchain` and `llama_index` submodules are optional-extra-gated adapters (`axon-rag[langchain]`, `axon-rag[llama-index]`), both wrapping `AxonBrain.search_raw` so they inherit reranking/hybrid/HyDE/multi-query/GraphRAG budget identically to REST/REPL.

### `src/axon/integrations/langchain.py`
**Role:** LangChain `BaseRetriever` adapter — drop-in retriever for any LangChain/LCEL chain, backed entirely by `AxonBrain.search_raw`. Gated behind the `axon-rag[langchain]` extra; degrades to a raising stub when `langchain-core` isn't installed. Unchanged since the last audit.

- `AxonRetriever(brain, top_k=None, filters=None, overrides=None)` — LangChain `BaseRetriever` subclass; supports standard `invoke`/`ainvoke`/LCEL `|` via inherited `_get_relevant_documents`/`_aget_relevant_documents`, which call `brain.search_raw` and map rows to `Document` — `langchain.py:98`
- `AxonRetriever.with_overrides(overrides)` — returns a new retriever instance with merged RAG-flag overrides, leaving the original untouched (immutable-style variant builder) — `langchain.py:124`
- `AxonRetriever.aretrieve(query, *, filters=None, **overrides)` — async-only retrieval accepting per-call override kwargs directly (top_k, rerank, sentence_window, hyde, multi_query, hybrid_search, graph_rag, etc.) without needing `with_overrides` + a new instance; runs `search_raw` in a thread — `langchain.py:180`
- `_result_to_document(result)` — module-level helper converting one `brain.search_raw` result row into a LangChain `Document`, normalizing `id`/`score`/web-source metadata — `langchain.py:76`

### `src/axon/integrations/llama_index.py`
**Role:** LlamaIndex `BaseRetriever` adapter — same purpose and pattern as the LangChain adapter, backed by `AxonBrain.search_raw`. Gated behind `axon-rag[llama-index]`; degrades to a raising stub otherwise. Unchanged since the last audit.

- `AxonLlamaRetriever(brain, top_k=None, filters=None, overrides=None)` — LlamaIndex `BaseRetriever` subclass; `_retrieve`/`_aretrieve` call `brain.search_raw` and map rows to `NodeWithScore` — `llama_index.py:70`
- `AxonLlamaRetriever._build_overrides()` — merges constructor `top_k` into the overrides dict passed to `search_raw` — `llama_index.py:94`
- `_result_to_node_with_score(result)` — module-level helper converting one `brain.search_raw` result row into a LlamaIndex `TextNode` + `NodeWithScore` — `llama_index.py:55`

**Possible internal overlap**

- **`can_entity_merge()`/`can_merge_entities_into_graph()` byte-identical bodies — fixed (commit `df49862`).** `can_entity_merge()` (`rust_bridge.py:540`) is now a one-line, explicitly-documented delegate to `can_merge_entities_into_graph()` (`rust_bridge.py:498`) rather than a second copy of the same body. Note: an earlier commit in this window (`#157`) considered removing `can_entity_merge()` entirely and deliberately did **not**, because it's a documented alias with its own test — that's consistent with, not a regression of, this fix.
- **`compute_doc_hash(text)`/`compute_sha256(text)` identical Python fallbacks — fixed (commit `df49862`), now intentionally different.** `compute_doc_hash`'s fallback changed from `sha256(text.strip())` to `md5(text)` (no stripping) to match the native implementation and `AxonBrain.ingest()`'s own dedup hash; `compute_sha256`'s fallback is unchanged (`sha256(text.strip())`). The two functions now serve genuinely different purposes (ingest-dedup hash vs. a generic content hash) rather than being accidental duplicates of the same algorithm probing different native function names.
- **`_result_to_document` (`langchain.py:76`) and `_result_to_node_with_score` (`llama_index.py:55`) do the same extraction — still present, unchanged.** Both normalize `text`/`metadata`/`id`/`score` and tag `is_web` → `source_kind` from a `search_raw` row, just wrapping the result in a different framework's document type. Neither file changed in this window. A shared "normalize a search_raw row" helper in a common module could back both, reducing drift risk if the `search_raw` row shape changes. `AxonRetriever` and `AxonLlamaRetriever` themselves are intentionally parallel (one per framework), not redundant.
