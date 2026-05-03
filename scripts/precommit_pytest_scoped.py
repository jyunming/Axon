#!/usr/bin/env python
"""Scope-aware pre-commit pytest selector.

Replacement for the testmon-based hook. testmon's transitive coverage
analysis is correct but slow when foundational modules change — a
single edit to ``axon/security/share.py`` invalidates ~all 4500
tests because share.py is imported by package init, projects.py,
main.py, and most route files.

This hook uses a simple **path-mapping heuristic** instead. Look at
the staged file paths, match them against the rules below, and run
*only* the matching test files. Predictable runtime; falls back to
``tests/test_lint.py`` (which is fast) when no rule matches.

Mappings (first match wins):

  src/axon/security/*           → tests/test_sealed_*.py
                                  tests/test_security*.py
                                  tests/test_project_seal.py
                                  tests/test_lint.py

  src/axon/api_routes/*         → tests/test_api*.py
                                  tests/test_query*.py
                                  tests/test_ingest*.py
                                  tests/test_lint.py

  src/axon/graph_backends/*     → tests/test_graph*.py
                                  tests/test_lint.py

  src/axon/dynamic_graph/*      → tests/test_dynamic*.py
                                  tests/test_graph*.py
                                  tests/test_lint.py

  src/axon/integrations/*       → tests/test_integrations*.py
                                  tests/test_v032*.py
                                  tests/test_lint.py

  src/axon/mcp_server.py        → tests/test_mcp*.py
                                  tests/test_surface_parity*.py
                                  tests/test_lint.py

  src/axon/cli.py               → tests/test_cli*.py
                                  tests/test_lint.py

  src/axon/repl.py              → tests/test_repl*.py
                                  tests/test_lint.py

  tests/<file>                  → that exact test file

  pyproject.toml                → tests/test_lint.py
                                  tests/test_packaging*.py

  src/axon/<other>.py           → tests/test_lint.py + a smoke set
                                  (covers main.py, retrievers.py, etc)

CI runs the FULL suite as the safety net. This hook only optimises
the local pre-commit experience. If you suspect the hook missed a
test that should have caught a bug, run ``python -m pytest tests/``
manually before pushing.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

# Each rule = (path-prefix-or-glob, list of test-glob patterns to add).
# First matching rule wins. ``test_lint.py`` is appended to every
# Python-source rule so type / lint regressions surface fast.
_RULES: list[tuple[str, list[str]]] = [
    # Security stack — the v0.4.0 hot path
    (
        "src/axon/security/",
        [
            "tests/test_sealed*.py",
            "tests/test_security*.py",
            "tests/test_project_seal.py",
            "tests/test_master_*.py",
            "tests/test_grantee*.py",
            "tests/test_mount_*.py",
            "tests/test_lint.py",
        ],
    ),
    # API routes
    (
        "src/axon/api_routes/",
        [
            "tests/test_api*.py",
            "tests/test_query*.py",
            "tests/test_ingest*.py",
            "tests/test_lint.py",
        ],
    ),
    # Graph backends
    (
        "src/axon/graph_backends/",
        [
            "tests/test_graph*.py",
            "tests/test_lint.py",
        ],
    ),
    # Dynamic graph
    (
        "src/axon/dynamic_graph/",
        [
            "tests/test_dynamic*.py",
            "tests/test_graph*.py",
            "tests/test_lint.py",
        ],
    ),
    # Integrations (langchain / llama_index)
    (
        "src/axon/integrations/",
        [
            "tests/test_integrations*.py",
            "tests/test_v032*.py",
            "tests/test_lint.py",
        ],
    ),
    # Specific top-level modules
    (
        "src/axon/mcp_server.py",
        [
            "tests/test_mcp*.py",
            "tests/test_surface_parity*.py",
            "tests/test_lint.py",
        ],
    ),
    (
        "src/axon/cli.py",
        [
            "tests/test_cli*.py",
            "tests/test_lint.py",
        ],
    ),
    (
        "src/axon/repl.py",
        [
            "tests/test_repl*.py",
            "tests/test_lint.py",
        ],
    ),
    # Dependency manifests — verify packaging still passes
    (
        "pyproject.toml",
        [
            "tests/test_lint.py",
            "tests/test_version_marker.py",
        ],
    ),
    (
        "src/axon/Cargo.toml",
        [
            "tests/test_lint.py",
            "tests/test_version_marker.py",
        ],
    ),
    (
        "config.yaml",
        [
            "tests/test_lint.py",
            "tests/test_api.py",
        ],
    ),
]

# Default fallback for any other src/axon/*.py file: lint + a small
# smoke set covering the foundational modules.
_FALLBACK = [
    "tests/test_lint.py",
    "tests/test_version_marker.py",
]

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _resolve_globs(patterns: list[str]) -> set[str]:
    """Resolve glob patterns relative to repo root; return existing files."""
    resolved: set[str] = set()
    for pat in patterns:
        for match in _REPO_ROOT.glob(pat):
            if match.is_file():
                resolved.add(str(match.relative_to(_REPO_ROOT)).replace("\\", "/"))
    return resolved


def _select_tests(staged_files: list[str]) -> set[str]:
    """Map staged file paths to a set of test file paths to run."""
    targets: set[str] = set()
    for f in staged_files:
        # Tests changing → run that test directly
        if f.startswith("tests/") and f.endswith(".py"):
            targets.add(f)
            continue
        # Match against rules; first rule wins
        matched = False
        for prefix, patterns in _RULES:
            if f.startswith(prefix):
                targets |= _resolve_globs(patterns)
                matched = True
                break
        if not matched:
            # Fallback for any other src/axon/*.py file
            if f.startswith("src/axon/") and f.endswith(".py"):
                targets |= _resolve_globs(_FALLBACK)
    return targets


def main() -> int:
    staged = [a for a in sys.argv[1:] if a]
    if not staged:
        return 0  # nothing staged
    targets = _select_tests(staged)
    if not targets:
        # No mapping — hook runs nothing. Trust CI.
        print("[pytest-scoped] no test mapping for staged files; skipping pytest")
        return 0
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *sorted(targets),
        "--tb=short",
        "--no-cov",
        "-q",
        "-m",
        "not extension",
    ]
    print(f"[pytest-scoped] running {len(targets)} test file(s):")
    for t in sorted(targets):
        print(f"  {t}")
    return subprocess.call(cmd, cwd=_REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
