# Graph Backend — Remaining Work

Handoff task list for continuing the Dynamic Graph roadmap
(`docs/architecture/DYNAMIC_GRAPH_ROADMAP.md`) after the wiring PR. Written
so a fresh Claude Code conversation can pick this up with zero prior context
— read this whole file before starting.

## Where things stand

Branch `feat/wire-graph-backend-to-production` (off `main`, currently at
`main`'s `d08fb1a`) has one commit, `0f8c9c1`, **not yet pushed, no PR
opened** — waiting on explicit user approval to push per this repo's
standing workflow rule (see `CLAUDE.md` "Branch Workflow" and the user's
saved feedback memory on PR approval).

That commit closed the headline finding from a 4-agent codebase audit: the
`GraphBackend` abstraction (`src/axon/graph_backends/`) was real,
well-tested code that was never reachable — `get_graph_backend()` was never
called from `main.py`, so `brain._graph_backend` was always `None` and every
`/graph/*` surface silently reported `"none"`. That commit:

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

## Priority 1 — M2: Backend Boundary Refactor (the big one)

**Goal:** `AxonBrain` stops inheriting `GraphRagMixin`; every graph
operation is routed through `self._graph_backend.*` instead. This is the
prerequisite for the rest of v0.4 (routing the main `/query` pipeline
through the backend abstraction) and for v1.0 hardening.

**Current state, verified:**
- `src/axon/main.py:170-171` — `class AxonBrain(..., GraphRagMixin, ...)`
  still directly inherits the mixin.
- `tests/test_graph_backend_base.py::TestPhase2ShimRemoval::test_axon_brain_does_not_inherit_graphragmixin`
  is the canary — currently `@pytest.mark.xfail(strict=False)`. When M2 is
  done, flip it to `strict=True` (or just remove the xfail decorator) so it
  becomes a hard gate against regression.

**How to scope the work:** the roadmap's own M2 exit criteria is "no direct
access of `_entity_graph` / `_relation_graph` / `_community_summaries` /
`_claims_graph` outside the adapter [`GraphRagBackend`]." Start by grepping
for those four attribute names across `src/axon/` outside
`src/axon/graph_rag.py` and `src/axon/graph_backends/graphrag_backend.py` —
every hit outside those two files is a call site that needs to route through
`brain._graph_backend` instead. Likely touches `query_router.py` (core
retrieval), `api_routes/graph.py`, `graph_render.py` (partially already
fixed — `_resolve_graph_payload()` now prefers `backend.graph_data()`, but
check for other direct reads), `repl.py`, `mcp_server.py`, `agent.py`.

**Also required for the M2 exit criteria (already-written, not-yet-passing test):**
`tests/test_architecture.py` is described in the roadmap doc (M1 section) as
using Python's `ast` module to enforce zero direct attribute access, but
**this file does not exist yet** — it needs to be written as part of this
work, not just discovered failing.

**Watch out for:** `tests/test_graphrag_parity.py` (the M1 regression
harness) must stay green throughout — it's the thing that proves GraphRAG's
externally-visible behavior doesn't drift during the refactor. Per the
audit: this suite currently uses `MagicMock` for its LLM boundary and
**never actually consumes** the real fixture corpus at
`tests/fixtures/graphrag_parity/*` (6 scenarios exist on disk: basic_entity,
multi_entity, relations, community, empty_doc, unicode_stress) — worth
wiring the real fixtures in before or during M2 so the parity check is
actually meaningful, not just decorative.

**Once M2 lands**, the other half of v0.4 becomes unblocked: plumb
`_graph_backend.retrieve()` into the main `/query` pipeline in
`query_router.py` (today only `POST /graph/retrieve` uses the backend
directly; the main query path doesn't). See the "v0.4" section of
`docs/architecture/DYNAMIC_GRAPH_ROADMAP.md` for the full deliverable list.

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
  not pure cleanup. Item #15 on the board duplicates the missing
  `test_architecture.py` file noted in Priority 1.
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
