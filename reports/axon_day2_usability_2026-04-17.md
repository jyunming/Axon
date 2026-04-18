# Axon Day‑2 Usability / Robustness Report (2026‑04‑17)

Scope: “Day‑2” end‑user workflows (repeat usage after initial setup) across **CLI**, **REPL**, **VS Code extension**, and **share/projects** flows. Goal: identify robustness gaps + usability friction with reproducible steps for R&D.

## Test environment

- OS: Windows (PowerShell)
- Repo: `C:\dev\studio_brain_open`
- Date: 2026‑04‑17
- Python: 3.11.9
- Node: v24.12.0
- Commands executed primarily via `python -c "from axon.cli import main; main()" ...` and `pytest`.
- Note: working tree is **dirty** (many local modifications). Findings below describe *observed behavior in this workspace state*; several failures are hard blockers (syntax errors) and should be validated against a clean checkout / release build after fixes.

## High‑level outcome

- **Core API e2e (non‑VSCode) passed**, but:
- **VS Code extension e2e is hard‑failing** due to a syntax error in the compiled `out/` JS artifact.
- **CLI/REPL have multiple hard blockers** (Python `SyntaxError`, `UnboundLocalError`/`NameError` for `os`) that prevent basic workflows like ingest/list/query.
- **Non‑interactive output handling is currently hostile to usability** (prints are redirected to logs, leaving users thinking commands did nothing).
- **Project/store isolation is inconsistent**: several “fast path” CLI commands ignore `--config` / store base and instead use global home state.

## Artifacts captured

- CLI log capture (due to stdout/stderr redirection): `tmp_axontest_store/logs/axon-20260417.log`
- Ad‑hoc Node runner used to get a real stack trace for the VSCode extension parse error: `tmp_vscode_runner.js`

## Test matrix (what was exercised)

- `pytest -q tests/e2e` (full e2e suite)
- `pytest -q --no-cov tests/test_repl_e2e.py`
- `pytest -q --no-cov tests/test_repl_commands.py`
- CLI “day‑2” flows with isolated config:
  - create project, ingest file, list, dry‑run query
  - project listing
  - share list / generate / redeem (same user)
  - config validation
- Additional regression spot-checks (passed):
  - `pytest -q --no-cov tests/test_query_router_robustness.py` (119 passed)
  - `pytest -q --no-cov tests/test_projects.py` (92 passed, 1 skipped)

---

## Findings (bugs / robustness gaps)

### AXON‑B001 — VS Code extension does not activate (compiled JS syntax error)

- Area: VS Code extension (`integrations/vscode-axon/out/...`)
- Severity: **Blocker**
- Evidence: `pytest -q tests/e2e` → 80 failures, all VSCode extension tests.
- Root cause (observed): syntax error in compiled output.
  - File: `integrations/vscode-axon/out/tools/graph.js`
  - Around line ~180:
    - `return new vscode.LanguageModelToolResult([new vscode.LanguageModelTextPart(lines.join(', '))]);))]);`
- Repro (fast):
  1. Run: `node tmp_vscode_runner.js C:\dev\studio_brain_open\integrations\vscode-axon\out\extension.js C:\dev\studio_brain_open\integrations\vscode-axon http://127.0.0.1:1 '' '{}' '{}'`
  2. Observe: `SyntaxError: Unexpected token ')'` with stack pointing at `out/tools/graph.js:180`.
- Expected:
  - Extension activates; tools/participants register; e2e harness can invoke tools.
- Actual:
  - Node cannot parse `graph.js`; activation fails; every VSCode extension e2e test fails.
- Notes / recommendations:
  - `out/` appears **stale or corrupted** relative to `src/` (e.g., `src/tools/graph.ts` uses `lines.join('\n')`, but `out/tools/graph.js` uses `join(', ')` and contains extra tokens).
  - Fix options:
    - Rebuild `out/` from TypeScript (`npm run compile`) and ensure packaging/tests use the rebuilt artifact.
    - Or stop committing `out/` and make CI/tests compile before running extension e2e.
    - Add a CI sanity check: “`node -c` equivalent” (or require() smoke test) on `out/extension.js`.

### AXON‑B002 — CLI has a Python `SyntaxError` in logging/print‑redirect block

Status update:
- This was observed earlier during the session, but **was not reproducible** after subsequent runs (`python -m compileall -q src/axon` succeeded and `--config-validate` worked).
- If it reappears on other branches/commits, re-check indentation around the logging/print-redirect region in `src/axon/cli.py` (~740–790).

### AXON‑B003 — CLI `os` becomes an unbound local (UnboundLocalError) due to conditional `import os`

- Area: CLI (`src/axon/cli.py`)
- Severity: **Blocker** (breaks ingest + exit path; breaks tests)
- Evidence:
  - `pytest -q --no-cov tests/test_repl_e2e.py` fails with:
    - `UnboundLocalError: cannot access local variable 'os' ...` at `src/axon/cli.py:2116` (`os._exit(0)`).
  - Also observed in actual dry‑run CLI usage when trying to ingest a file:
    - `UnboundLocalError` at `src/axon/cli.py:1550` (`if os.path.isdir(args.ingest):`).
- Likely cause:
  - `import os` inside `main()` under `if args.list_models:` / `if args.pull:` makes `os` a *local* variable for the whole function. If those branches aren’t taken, later uses of `os` crash.
- Fix recommendation:
  - Remove `import os` inside `main()` (use the module‑level `import os`), or rename local imports (`import os as _os`) consistently.

### AXON‑B004 — Non‑interactive CLI runs produce no console output (prints redirected to log)

- Area: CLI output / UX
- Severity: **High** (makes CLI feel broken in scripts/CI; hides errors)
- Repro:
  - With a config like `tmp_axontest_config.yaml`, run:
    - `... --project-list --dry-run --non-interactive`
    - `... --list --dry-run --non-interactive`
    - `... "What is Axon?" --dry-run --non-interactive`
  - Console shows **nothing**, but `tmp_axontest_store/logs/axon-20260417.log` contains the output.
- Expected:
  - CLI should print user‑facing results to stdout/stderr in non‑interactive mode.
  - Debug logs can go to a file.
- Actual:
  - `builtins.print` is replaced (non‑TTY / `--quiet`) and later `sys.stdout`/`sys.stderr` are redirected to a daily log file, so command output disappears from the terminal.
- Recommendation:
  - Do not monkey‑patch `print()` for UX output; keep stdout for user output.
  - If suppressing 3rd‑party noise is desired, route *only those loggers* to a file handler, not `sys.stdout`.
  - Add `--log-file` / `--verbose` / `--quiet` semantics that match common CLI expectations.

### AXON‑B005 — CLI fast‑path `--project-list` ignores `--config` store root (lists home projects)

- Area: Projects / config isolation
- Severity: **High**
- Repro:
  1. Create isolated store via `--config tmp_axontest_config.yaml` with `store.base: ./tmp_axontest_store`.
  2. Run: `... --project-list ...`
  3. Inspect `tmp_axontest_store/logs/axon-20260417.log`: output lists projects that **do not exist** in `tmp_axontest_store\AxonStore\jyunm`.
- Root cause (observed by code inspection):
  - `src/axon/projects.py` sets `PROJECTS_ROOT` at import time to `~/.axon/projects` unless `AXON_PROJECTS_ROOT` is set.
  - CLI “fast path” commands call `axon.projects.list_projects()` without first calling `axon.projects.set_projects_root(config.projects_root)`.
  - Also `_ACTIVE_FILE` is hard-coded to `Path.home() / ".axon" / ".active_project"`, so “active project” state leaks across stores.
- Recommendation:
  - On CLI startup (before any project ops), set project root from loaded config.
  - Remove import-time globals (`PROJECTS_ROOT`, `_ACTIVE_FILE`) or make them derived from config/store base.

### AXON‑B006 — REPL commands crash due to `os` scoping bugs (shell passthrough, /keys)

- Area: REPL (`src/axon/repl.py`)
- Severity: **High** (if present; breaks common REPL workflows)
- Status update:
  - This was observed earlier in the session, but **is not reproducible now**:
    - `pytest -q --no-cov tests/test_repl_commands.py` → `177 passed, 7 skipped`
    - `pytest -q --no-cov tests/test_repl_e2e.py` → `1 passed`
- If this reappears:
  - Watch for local `import os` inside nested scopes changing Python’s name resolution.
  - Keep CI gates on `tests/test_repl_commands.py` and `tests/test_repl_e2e.py`.

### AXON‑B007 — `python -m axon.cli` prints nothing (module not runnable as documented pattern)

- Area: CLI entry point ergonomics
- Severity: **Medium**
- Repro:
  - `python -m axon.cli --help` → no output because `src/axon/cli.py` lacks `if __name__ == "__main__": main()`.
- Expected:
  - Common Python ergonomics: `python -m axon.cli --help` should work (or at least error clearly).
- Recommendation:
  - Add a `__main__` guard, or provide `axon/__main__.py` so `python -m axon` works.

### AXON‑B008 — Share redeem UX: “Invalid share_string format” is good, but grantee-mismatch is unclear

- Area: Shares (`src/axon/shares.py`)
- Severity: **Low** (mostly messaging / user guidance)
- Observation:
  - Invalid base64 share strings correctly raise: `ValueError: Invalid share_string format.`
  - Redeeming as the *wrong OS user* fails with: `Share key HMAC verification failed.`
- Recommendation:
  - Include `grantee` in the decoded payload (or at least in the error), e.g.:
    - “This share is for grantee ‘testgrantee’ but you are ‘jyunm’.”
- Area: Shares (`src/axon/shares.py`)

### AXON‑B009 — `axon --setup` can crash with `PermissionError` writing config, no recovery guidance

- Area: CLI setup wizard / config persistence (`src/axon/cli.py`, `src/axon/config.py`)
- Severity: **High**
- Evidence:
  - `pytest -q --no-cov tests/test_cli_advanced.py` fails:
    - `PermissionError: [Errno 13] Permission denied: 'C:\\Users\\jyunm\\.config\\axon\\config.yaml'`
- Expected:
  - Setup either writes to a user-writable location, or prints a clear error + remediation (e.g., “pass --config ...”).
- Actual:
  - Unhandled exception (crash) when the default path is not writable (common in locked-down environments / CI / corporate machines).
- Recommendation:
  - Always honor an explicit `--config` path during `--setup`.
  - Catch `OSError` on save and print an actionable message (target path, how to override).

### AXON‑B010 — “Metadata-only” CLI commands can trigger Ollama auto-pull + network noise

- Area: CLI control flow / side effects (`src/axon/cli.py`)
- Severity: **High**
- Evidence:
  - `pytest -q --no-cov tests/test_cli_extra.py` shows many failures where expected output is replaced by:
    - `Model 'gemma:2b' not found locally — pulling from Ollama...`
  - Captured logs include `httpcore` DEBUG traces, implying noisy logging configuration leaks into tests/UX.
- Expected:
  - Commands like `--project-list`, `--list`, `--config-validate` should not attempt LLM/ollama operations and should be deterministic/offline-safe.
- Actual:
  - Auto-pull side effects run early enough to pollute UX and break unit tests.
- Recommendation:
  - Gate auto-pull behind “need_brain” / interactive REPL entry / explicit query/stream paths, not global CLI startup.
  - Default to *no network* unless user requested a network-required command.

### AXON‑B011 — Graph distance / hop-count feature appears broken or removed (23 test failures)

- Area: GraphRAG distance + dynamic graph retrieval (`src/axon/graph_rag.py`, `src/axon/config.py`)
- Severity: **High** (breaks graph distance tooling + retrieval hop diagnostics)
- Evidence:
  - `pytest -q --no-cov tests/test_graph_distance.py` → `23 failed, 6 passed`
  - Representative failures:
    - `AttributeError: 'GraphRagMixin' object has no attribute '_build_nx_graph'`
    - `_save_entity_graph` / `_save_relation_graph` not marking `_nx_graph_dirty`
    - YAML config fields missing:
      - `graph_rag_max_hops`, `graph_rag_hop_decay`, `graph_rag_distance_weighted`
    - Relation expansion/hop-threshold expectations not met (2-hop chunks not reachable under threshold).
- Expected:
  - Internal “nx graph” cache builder exists and supports weighted hop distances, TTL cache behavior, and hop count / path metadata on retrieved chunks.
  - Config knobs load from YAML and affect traversal.
- Actual:
  - Methods/fields absent and multiple behavioral invariants fail.
- Recommendation:
  - Decide whether these features are intentionally removed; if so, delete/update the tests and CLI flags (e.g., `--graph-rag-max-hops`, `--graph-rag-hop-decay`, `--no-graph-rag-weighted`).
  - If features are intended to exist, reintroduce:
    - `_build_nx_graph` (or update tests to new name),
    - `_nx_graph_dirty` invalidation on graph saves,
    - config fields + YAML mapping for the traversal parameters,
    - consistent scoring decay and hop metadata propagation.

---

## Additional bugs found in extended sweeps

### AXON‑B012 — Dry-run mode does not prevent embedding provider calls (can hit Ollama/network)

- Area: Embeddings (`src/axon/embeddings.py`)
- Severity: **High** (breaks `--dry-run` guarantees; causes unintended external calls)
- Evidence:
  - `pytest -q --no-cov tests/test_dry_run_embedding.py` fails:
    - `ollama._types.ResponseError: model "all-MiniLM-L6-v2" not found ...`
  - The test sets `AXON_DRY_RUN=1` and expects zero vectors, but `OpenEmbedding.embed()` still calls `ollama.Client().embeddings(...)`.
- Expected:
  - When `AXON_DRY_RUN` is set, embeddings should be deterministic/no-op (e.g., return zero vectors with the configured dimension).
- Actual:
  - Provider SDK is invoked.
- Recommendation:
  - In `OpenEmbedding.embed()` (and possibly `_load_model()`), short-circuit on `os.getenv("AXON_DRY_RUN")` and return zeros.

### AXON‑B013 — Dry-run mode does not prevent ImageLoader VLM calls

- Area: Loaders (`src/axon/loaders.py` → `ImageLoader`)
- Severity: **High** (breaks `--dry-run` guarantees)
- Evidence:
  - `pytest -q --no-cov tests/test_dry_run_image.py` fails because `ImageLoader.load()` calls `ollama.generate()` even with `AXON_DRY_RUN=1`.
- Recommendation:
  - Add a dry-run guard in `ImageLoader.load()` returning a placeholder document (or skipping image captioning) without calling external VLMs.

### AXON‑B014 — Windows: `AxonConfig` can crash if user env vars are cleared (`getpass.getuser()` imports `pwd`)

- Area: Config init (`src/axon/config.py` `__post_init__`)
- Severity: **High** (config validation and other paths can crash in sandbox/CI/corporate shells)
- Evidence:
  - `pytest -q --no-cov tests/test_config_validate.py` fails with:
    - `ModuleNotFoundError: No module named 'pwd'`
  - Trigger: tests clear `os.environ` (`patch.dict(..., clear=True)`), and `getpass.getuser()` falls back to importing `pwd` (not available on Windows).
- Recommendation:
  - Wrap `getpass.getuser()` in a try/except and fall back to a safe default (e.g. `os.environ.get("USERNAME")` or `"unknown"`).

### AXON‑B015 — API ingest refresh does not report `reingested` / `missing` / `errors` correctly

- Area: API ingest refresh job accounting (`src/axon/api_routes/ingest.py` and/or refresh logic)
- Severity: **High** (day‑2 workflow: refreshing modified docs silently does nothing)
- Evidence:
  - `pytest -q --no-cov tests/test_api.py --maxfail=5` shows failures:
    - `test_refresh_mounted_share_ingest_error_recorded_in_job` (`job["errors"]` empty)
    - `test_ingest_refresh_reingest_needed` (`reingested` empty)
    - `test_ingest_refresh_missing_file` and `TestIngestRefresh.test_missing_source_reported` (`missing` empty)
    - `TestIngestRefresh.test_no_loader_for_extension_reported_as_error` (`errors` empty)
- Expected:
  - Refresh returns accurate accounting for re-ingested/unchanged/missing/errors.
- Actual:
  - Lists are empty even for clearly missing/changed/no-loader cases.

### AXON‑B016 — CRAG eval writes results into a repo path that may be non-writable

- Area: Evaluation harness (`tests/test_crag_eval.py`)
- Severity: **Medium** (breaks test runs; confusing for contributors)
- Evidence:
  - `pytest -q --no-cov tests/test_crag_eval.py` fails:
    - `PermissionError: [Errno 13] Permission denied: 'Qualification\\GeminiQual\\crag_eval_results.json'`
  - Manual repro: writing to that file fails with “Access denied” in this environment.
- Recommendation:
  - Write outputs to `tmp_path` (pytest temp) or guard the write behind an env var / explicit `--write-results` flag.

### AXON‑B017 — Black formatting check can take >10 minutes (likely due to very large files)

- Area: Dev/test ergonomics (`tests/test_lint.py`)
- Severity: **Medium**
- Evidence:
  - `pytest ... tests/test_lint.py::test_black_formatting` consistently times out after 10 minutes here.
  - Likely culprit: very large Python files (e.g., `tests/test_main.py` is unusually large) causing black to be slow on Windows.
- Recommendation:
  - Exclude known huge/generated files from black in CI, or split them, or run black with caching/parallelism tuned for Windows.


## Usability / DX improvement opportunities (non‑bugs)

1. **Share strings are long and opaque**
   - Suggest: always copy to clipboard + print a short prefix; add `--share-string-file`.
   - Include grantee name in the decoded payload (currently redeem failures are non-obvious if run as wrong OS user).

2. **Default logging verbosity**
   - Root logger set to DEBUG and “noisy” libs sometimes leak debug lines (e.g., `asyncio ... DEBUG Using proactor`).
   - Suggest: default console level INFO with minimal noise; file handler DEBUG behind `--debug`.

3. **Non‑TTY behavior**
   - CI/scripting mode should be *more* explicit, not silent. Prefer `--json` outputs for list/query/dry-run.

4. **Project/store isolation**
   - Avoid global files under `~/.axon` for active project when `store.base` is configured elsewhere.

---

## Proposed next steps for R&D

1. Fix the **hard blockers** first (AXON‑B001/B002/B003/B006).
2. Add CI gates:
   - `pytest -q --no-cov tests/test_repl_commands.py`
   - `pytest -q --no-cov tests/test_repl_e2e.py`
   - VSCode extension: a minimal `node -e "require('./out/extension.js')"` smoke test (or rebuild step + smoke).
3. Rework CLI output/logging:
   - Keep user output on stdout/stderr.
   - Route debug logs to file with opt-in `--debug` and a clear log path message on error.
4. Make projects root and active-project file location **config-derived**, not import-time globals.
