# Graph Backend — Remaining Work

Handoff task list for continuing the Dynamic Graph roadmap
(`docs/architecture/DYNAMIC_GRAPH_ROADMAP.md`) after the wiring PR. Written
so a fresh Claude Code conversation can pick this up with zero prior context
— read this whole file before starting.

## Where things stand

Branch `feat/wire-graph-backend-to-production` (off `main`, currently at
`main`'s `d08fb1a`) has the wiring commit (`0f8c9c1`) plus the M2 Phase 1
commit(s) described below — **not yet pushed, no PR opened** — waiting on
explicit user approval to push per this repo's standing workflow rule (see
`CLAUDE.md` "Branch Workflow" and the user's saved feedback memory on PR
approval).

The wiring commit (`0f8c9c1`) closed the headline finding from a 4-agent
codebase audit: the `GraphBackend` abstraction (`src/axon/graph_backends/`)
was real, well-tested code that was never reachable — `get_graph_backend()`
was never called from `main.py`, so `brain._graph_backend` was always `None`
and every `/graph/*` surface silently reported `"none"`. That commit:

- Added `NoneGraphBackend`
- Attached + resynced `brain._graph_backend` in `__init__`/`switch_project()`/
  `_switch_to_scope()`, with proper `close()` cleanup on every transition
- Made `graph_backend` a real, validated, choosable field on every
  project-creation surface (REST, CLI, REPL, MCP, agent tool)
- Fixed the 3 visualize endpoints that bypassed `backend.graph_data()`
- Added ~30 tests; full suite green (5554 passed)
- Fixed doc claims in `ADMIN_REFERENCE.md`, `MCP_TOOLS.md`, `OFFLINE_GUIDE.md`,
  `QUICKREF.md`, and this roadmap doc

**Deliberately not in scope for that commit** (see "Priority 1" below): the
core `/query` pipeline still answers through the legacy `GraphRagMixin`
inheritance, not through `_graph_backend.retrieve()`. Only the standalone
`POST /graph/retrieve` route (and REPL/CLI/MCP equivalents) uses the new
backend abstraction directly. This was a deliberate scope boundary, not an
oversight — see the roadmap doc's M2 milestone.

A second, smaller finding from the same audit — fresh installs getting
`max_tokens=2048` instead of the dataclass default `8192` — was fixed
separately and already merged (PR #138, `d08fb1a`).

**M2 Phase 1 (mechanical redirects + backend fixes) has since landed** on
this same branch — see the updated "Priority 1" section below for what
shipped and what's still open (Phases 2-4).

## Priority 1 — M2: Backend Boundary Refactor (the big one)

**Goal:** `AxonBrain` stops inheriting `GraphRagMixin`; every graph
operation is routed through `self._graph_backend.*` instead. This is the
prerequisite for the rest of v0.4 (routing the main `/query` pipeline
through the backend abstraction) and for v1.0 hardening.

The M2 exit criteria is "no direct access of `_entity_graph` /
`_relation_graph` / `_community_summaries` / `_claims_graph` outside the
adapter [`GraphRagBackend`]." A pre-work audit (3 Explore agents mapping
every occurrence of those four attribute names outside
`src/axon/graph_rag.py`/`src/axon/graph_backends/graphrag_backend.py`, plus
reading the `GraphBackend` Protocol, all four concrete backends, and the
test infra) found ~176 occurrences across 12 files, of very different risk —
so M2 is being landed in phases rather than one sweep. **`AxonBrain` still
inherits `GraphRagMixin`** (`src/axon/main.py` class declaration) and the
`test_graph_backend_base.py::TestPhase2ShimRemoval::test_axon_brain_does_not_inherit_graphragmixin`
canary stays `@pytest.mark.xfail(strict=False)` until Phase 4 below lands —
don't flip it early.

### Phase 1 — SHIPPED (this branch)

Mechanical redirects + two real correctness bugs fixed along the way:

- **`GraphRagBackend.delete_documents()`** (`graph_backends/graphrag_backend.py`)
  now delegates to `AxonBrain._prune_entity_graph()` instead of a
  reimplementation that silently skipped relation-graph pruning, claims-graph
  pruning, token-index cleanup, frequency recompute, and disk persistence.
- **`GraphRagBackend.clear()`** now delegates to a new
  `GraphRagMixin._reset_graph_state()` (`graph_rag.py`) — the single source
  of truth for "reset all in-memory graph state," covering all 14 fields
  instead of the 4 it used to hand-clear. Memory-only by design (no
  persistence) — it's also used by read-only scope switching, which must
  never write project data to disk.
- **`main.py::_switch_to_scope`** now constructs/forces the right backend
  *before* clearing (previously cleared 14 attributes directly, then
  reconstructed the backend after — meaning a stale non-GraphRAG backend,
  e.g. `dynamic_graph`, was never actually asked to clear anything). Fixed
  as part of the redirect; see `TestGraphBackendWiring::test_switch_to_scope_clears_stale_graph_state_from_previous_project`
  in `tests/test_main.py` for the regression test.
- **`collection_ops.py::clear_active_project`** now calls
  `brain._graph_backend.clear()` for the in-memory reset (correctly clears
  non-GraphRAG backends too — a gap before) while keeping its own
  `_save_*()` calls for on-disk persistence.
- **Mechanical consumer redirects** to `brain._graph_backend.status()` /
  `.delete_documents()` / `FinalizationResult.communities_built`, fully
  cleaning `agent.py`, `cli.py`, `collection_ops.py`, `api_routes/ingest.py`.
  `repl.py`, `api_routes/graph.py`, `api_routes/governance.py` each keep
  exactly one documented, intentional direct-read fallback (used only when
  no backend is attached at all, or `backend.status()` itself raises) —
  matching the graceful-degradation pattern already established in
  `repl.py`'s `/graph status` before this phase.
- **`api_routes/governance.py::governance_graph_rebuild`** also had an
  unguarded `len(brain._community_summaries)` (no `getattr` fallback) — a
  real crash-on-refactor risk, fixed by routing through
  `brain._graph_backend.finalize()` like `/graph/finalize` already does.
- **`tests/test_architecture.py`** written (was named but never created by
  M1) — AST-based (not grep), three tiers: `TestPhase1FilesHaveNoDirectAccess`
  (strict, zero violations, the regression gate for this phase's cleaned
  files), `TestKnownFallbackFilesHaveNotGrown` (pinned count for the
  documented fallback sites above), `TestFullComplianceTarget` (xfail,
  tracks `main.py`/`query_router.py` — the remaining work below).

**Explicitly out of scope for Phase 1** (left for later phases, see below):
`main.py`'s `__init__`/load paths, the descendant-project graph-merge block
in `switch_project`, and all of `main.py::ingest()`; all of `query_router.py`;
`graph_render.py::build_graph_payload()`.

### Phase 2 — SHIPPED

- Moved `graph_render.py::build_graph_payload()`'s body verbatim into
  `graph_rag.py` (`GraphRagMixin`), same method name — `GraphRagBackend
  .graph_data()`'s existing `self._brain.build_graph_payload()` call and
  every other production caller keep working unchanged via MRO, now
  resolving from a whitelisted file. `tests/test_graph_rag.py`'s
  `MockBrain` (the only other `GraphRenderMixin` consumer besides
  `AxonBrain`) updated to `class MockBrain(GraphRenderMixin,
  GraphRagMixin)` to keep resolving the method after the move.
- Wired the real fixtures at `tests/fixtures/graphrag_parity/*` (6
  scenarios: `codebase`, `issue_thread`, `paper_abstract`, `project_doc`,
  `software_guide`, `stdlib_docs`) into `tests/test_graphrag_parity.py`'s
  new `TestGraphRagParityFixtures` — exercises real ingest → extraction →
  community-build → render end to end (canned-LLM `AxonBrain`, no mocked
  mixin methods) alongside the pre-existing pure-`MagicMock`
  adapter-contract tests, which stay unchanged.
- **Found and fixed a real production deadlock while wiring the fixtures**:
  `_rebuild_communities` held `_graph_lock` (an `RLock`) across its whole
  body while dispatching community-summary generation onto a real
  `ThreadPoolExecutor`; worker threads re-entered the same lock via
  `_gr_cache_get` from a different OS thread than the one holding it —
  `RLock` reentry is same-thread-only, so every worker deadlocked forever.
  Reachable via `graph_rag_community_lazy=False`, via `main.py::ingest()`'s
  synchronous post-ingest rebuild path, and via `graph_rag_claims`/
  `graph_rag_canonicalize(_relations)` (the latter two currently default
  off, so dormant but real). Fixed by giving the GraphRAG LLM/extraction
  cache its own dedicated leaf lock (`_gr_cache_lock`) instead of sharing
  `_graph_lock` — see the comment on `_gr_cache_lock_internal`'s init in
  `main.py` for the full reasoning. Regression test:
  `TestCommunitySummarizationDoesNotDeadlock` (daemon thread + 30s join
  timeout, so a regression fails cleanly instead of hanging CI).

### Phase 3 — `query_router.py`'s retrieval path (20 occurrences)

- `_expand_with_entity_graph()` (`query_router.py:251`) is the *real
  implementation* `GraphRagBackend.retrieve()` already delegates to, but it
  physically lives outside the whitelist. Decide: move its body into
  `graphrag_backend.py`/`graph_rag.py`, or treat it as an explicit,
  documented whitelist exception.
- `query()`/`query_stream()`/`_execute_retrieval_body()` use
  `_entity_graph`/`_community_summaries` as truthy guards before calling
  `_local_search_context()`/`_generate_community_summaries()`/
  `_global_search_map_reduce()`/`_expand_with_entity_graph()` — these are
  the most mechanically-redirectable sites (e.g. a new
  `has_local_context()`/`has_community_summaries()` Protocol method), but
  the guarded bodies still call mixin methods directly until those, too,
  get pulled behind the backend.
- This is also where the other half of v0.4 unblocks: plumb
  `_graph_backend.retrieve()` into the main `/query` pipeline for real
  (today only `POST /graph/retrieve` uses the backend directly — verified
  zero production call sites for `.retrieve()` otherwise). See the "v0.4"
  section of `docs/architecture/DYNAMIC_GRAPH_ROADMAP.md`.

### Phase 4 — `main.py`'s load/init/switch paths + `ingest()` (biggest, riskiest)

- `__init__`'s load block, `switch_project`'s reload-from-disk block, and
  the ~140-line descendant-project graph-merge block in `switch_project`
  (reads raw msgpack/JSON off disk and hand-merges into all four graph
  dicts) — none of these have a backend method to redirect to yet; need new
  `GraphBackend` Protocol methods (e.g. `load()`, `merge_descendants()`).
- `main.py::ingest()`'s ~370-line entity/relation/claims extraction and
  merge pipeline — `GraphRagBackend.ingest()` today is a confirmed no-op
  (bookkeeping only; real extraction happens entirely inside
  `AxonBrain.ingest()`). Closing this out means designing new Protocol
  methods (e.g. `add_entities`/`add_relations`/`add_claims`) *and* moving
  the extraction logic behind them — not a mechanical redirect.
- Only once this phase lands: flip
  `test_axon_brain_does_not_inherit_graphragmixin` to `strict=True` (or
  remove the xfail decorator) and actually drop `GraphRagMixin` from
  `AxonBrain`'s base classes.

## Priority 2 — v1.0 hardening test debt

The roadmap's v1.0 milestone requires 8 stress tests before a `1.0.0` tag;
per the audit, **0 are fully implemented** (6 entirely absent, 2 partial).
Full list and rationale in `docs/architecture/DYNAMIC_GRAPH_ROADMAP.md`
under "v1.0 — Hardening". The 8:

1. 50 concurrent read-only queries against a large GraphRAG project
2. 20 concurrent Dynamic Graph queries + 5 concurrent ingests (one project)
3. 10k sequential episode ingests — confirm no `O(total dataset)` rebuild
4. Contradiction storm on one exclusive relation family
5. Mixed-backend federated query under concurrent load
6. Repeated project switching across many projects — backend lifecycle leak
   check. Note: this session's `close()` wiring (task #5 of the just-shipped
   PR) is directly foundational to this test actually being meaningful —
   before that PR, `DynamicGraphBackend`'s SQLite connections leaked on every
   switch-away since nothing ever closed them.
7. Offline boot and query for both backends with no network access
8. Entity canonical name drift: 3 name variants for the same entity produce
   3 nodes (documented behavior, not a bug — just needs a test asserting it)

No performance baselines have ever been measured against the P95 targets
documented in the roadmap's "Performance Targets" table either — that
table is aspirational until someone actually runs the numbers.

## Priority 3 — Docs and polish (also v1.0-scoped)

- A dedicated "Dynamic Graph vs GraphRAG" user guide doc — the roadmap
  itself assigns this to v1.0, after both backends are stress-tested (see
  Priority 2). Don't write this prematurely; it'll need real usage/edge
  cases from the stress-test work to be accurate.
- `integrations/vscode-axon/src/tools/projects.ts` + `package.json`'s LM
  tool schema — `create_project` never got the `graph_backend` param.
  Deliberately deferred in the just-shipped PR (zero regression since the
  server defaults to `"graphrag"` when the field is absent). Bundling this
  in means paying the VSIX-rebuild/version-bump cost from `CLAUDE.md`'s
  Versions section — batch it with other VS Code changes if possible.
- GraphRAG prompt text in `graph_rag.py` was never rewritten to reduce
  "provenance risk" (the roadmap's own v1.0 phrasing — check the roadmap
  doc's v1.0 section for what this means; wasn't spelled out further by the
  audit).

## Priority 4 — Housekeeping (small, independent, any order)

These are unrelated to each other and to the graph-backend work above —
pick any subset, they don't block anything.

- **GitHub Project #4** (https://github.com/users/jyunming/projects/4):
  14 of 18 items are stale-but-already-shipped and just need closing. Items
  16-18 (a "LadybugDB backend" track) look abandoned/superseded by the
  roadmap's "SQLite only, no external graph DB" lock (see "V1 constraints"
  table in the roadmap doc) — confirm with the user before closing those
  specifically, since abandoning a tracked initiative is a judgment call,
  not pure cleanup. Item #15 on the board duplicated the `test_architecture.py`
  gap — that file now exists (Priority 1, Phase 1) and can be closed.
- `pyproject.toml:43-44` — comment says "the 48-tool MCP server," actual
  count is 55 (confirmed via `grep -c "@mcp.tool()" src/axon/mcp_server.py`,
  matches `CLAUDE.md`'s stated count). One-line comment fix, no code change.
- `tests/test_api_e2e.py:42` — stale skip reason references a "Phase 5
  refactor" that's already done; the audit found the test actually passes
  now. Verify and remove the skip.
- `tests/test_sparse_retrieval.py:461` — uses `skipif(True, ...)`, meaning
  it has never actually run in CI since it was written. Either env-gate it
  properly (matching how other slow/optional tests are marked, e.g.
  `@pytest.mark.slow`) or delete it if it's no longer relevant.
- CI: the mypy and Rust-extension-import steps run with
  `continue-on-error: true`, and bandit findings currently can't fail CI
  (only `pip-audit` can). Not necessarily bugs — flagged by the audit as
  worth a deliberate decision either way, not silently drifting.

## Safety notes learned the hard way this session

Worth reading before writing any test or doing any manual verification
against project creation / store paths — these cost real (recovered)
mistakes during the just-shipped PR's testing:

- **`~/.axon/.active_project` is a hardcoded global path**
  (`Path.home() / ".axon" / ".active_project"` in `projects.py`), completely
  independent of `AXON_STORE_BASE`, `--config`, or `AXON_CONFIG_PATH`. Any
  manual CLI/REPL smoke test that switches projects — even against a fully
  isolated store — will overwrite this **real, global** file. Restore it
  to `"default"` afterward (that's the function's own documented fallback
  when the file is absent, so it's always a safe value to restore to) and
  confirm via `axon --project-list` on the real store afterward.
- **`AXON_STORE_BASE` env var is silently ignored** if the real user config
  at `~/.config/axon/config.yaml` already sets `store.base` — config.yaml
  wins because of the precedence check in `AxonConfig.__post_init__`
  (`if env_store_base and not self.axon_store_base`). Don't rely on this
  env var alone for isolating a real `axon`/`axon-api` process; check
  `~/.config/axon/config.yaml` for a `store:` section first, or better,
  use one of the two mechanisms below instead.
- **What actually works for isolation:**
  - Direct Python testing: `axon.projects.set_projects_root(<tmp path>)` is
    the module's own designed override — safe, doesn't touch the real
    store.
  - Full `AxonBrain`/CLI/REPL testing: construct `AxonConfig(axon_store_base=<tmp path>, ...)`
    directly (bypasses `from_yaml()`/the real config.yaml entirely), or for
    the `axon`/`axon-api` binaries, pass an explicit `--config <path>` /
    `AXON_CONFIG_PATH=<path>` pointing at a throwaway `config.yaml` with its
    own `store: base: <tmp path>`. Always assert the resolved
    `projects_root` contains your tmp-dir marker string before doing
    anything else.
- **The real global config file
  (`~/.config/axon/config.yaml` on this machine) has plaintext secrets in it**
  (a GitHub PAT, a Gemini API key, an Ollama Cloud key). Never copy its
  `llm:` section into an isolated test config — reconstruct only the
  non-sensitive provider/model fields by hand.
- The `axon` REPL always uses `prompt_toolkit`'s interactive UI when the
  library is installed (no `sys.stdin.isatty()` fallback check) — piped
  non-interactive stdin does not reliably drive it command-by-command for
  scripted smoke tests. Its own unit tests
  (`tests/test_repl_commands.py`, using the existing `_run_repl_with_commands`
  / `_make_mock_brain` harness) are the right tool for verifying REPL
  command logic; don't fight the TTY requirement for a live smoke test —
  prefer testing the REPL's underlying shared functions (`ensure_project()`,
  `switch_project()`) via REST/MCP instead, which exercise the identical
  code path.
