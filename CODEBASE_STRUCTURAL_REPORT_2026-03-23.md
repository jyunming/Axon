# Codebase Structural Report (2026-03-23)

## 1) Scope
- Reviewed repository structure, core runtime/config modules, and user-facing docs consistency.
- Executed high-signal automated validation on the current branch state (`fix/codebase-bug-audit`).
- Focused on real regressions with direct user/runtime impact.

## 2) Method
- Static scan: repository map, config/runtime paths, doc statements vs implementation.
- Dynamic validation:
  - Targeted suites first for fast signal.
  - Broad suite: `pytest -q tests -m "not demo and not perf and not stress and not e2e and not eval" --tb=short --no-cov`.
- Applied minimal, isolated fixes only in affected files.

## 3) Findings (Ordered by Severity)

### High
1. **Legacy relative storage paths regressed to CWD resolution**
- Impact: `vector_store_path` / `bm25_path` values like `chroma_data` or `./chroma_data` incorrectly resolved to the current working directory on Windows, rather than the canonical `~/.axon/projects/default/...` location.
- Risk: silent data placement drift, mixed project state, hard-to-debug ingest/query behavior.
- Status: **Fixed**.

2. **Mounted-share write-guard fallback regression**
- Impact: `_is_mounted_share()` no longer honored legacy `_mounted_share` flag when `_active_project_kind` was absent.
- Risk: compatibility break for older objects/tests and potential write-protection misclassification.
- Status: **Fixed**.

### Medium
3. **README default behavior inconsistency (GraphRAG)**
- Impact: README claimed Knowledge Graph is “on by default,” while shipped `config.yaml` keeps `graph_rag` disabled by default.
- Risk: onboarding confusion and incorrect operator expectations.
- Status: **Fixed**.

## 4) Fixes Applied
- `src/axon/config.py`
  - Restored legacy-relative path normalization for `chroma_data` / `bm25_index` (`sub` and `./sub`) to resolve under `projects_root/default`.
- `src/axon/main.py`
  - Updated `_is_mounted_share()` to fallback to legacy `_mounted_share` when `_active_project_kind` is unavailable.
- `README.md`
  - Corrected Knowledge Graph default statement to match shipped config behavior.

## 5) Validation Evidence
- Targeted failing tests now pass:
  - `tests/test_config_extra.py::TestPostInit::test_resolve_safe_legacy_relative_path_redirected`
  - `tests/test_project_prompt.py::test_wsl_path_resolution_legacy_defaults`
  - `tests/test_main_extra.py::TestWriteGuards::test_is_mounted_share_fallback_to_legacy_flag`
- Broad validation result:
  - **3117 passed, 4 skipped, 187 deselected, 0 failed**
  - Command: `pytest -q tests -m "not demo and not perf and not stress and not e2e and not eval" --tb=short --no-cov`

## 6) Residual Risks / Next Angles
1. ~~Provider deprecation warning surfaced in tests (`google.generativeai` -> `google.genai` migration path).~~ **Resolved** — dependency updated to `google-genai>=1.4.0`; all `google.generativeai` references removed from source and tests.
2. Torch JIT deprecation warnings are non-blocking but indicate future compatibility debt in dependency stack.
3. Execute a separate e2e/perf/stress qualification gate before release if this branch is promotion-candidate.
