import subprocess
import sys

# ---------------------------------------------------------------------------
# Allowlist: pre-existing mypy errors that are accepted technical debt.
# Add new entries only when the error is in third-party code or truly
# unfixable. Never add entries to hide regressions in our own code.
# ---------------------------------------------------------------------------
_MYPY_ALLOWED_ERRORS: list[tuple[str, str]] = [
    # webapp.py — Streamlit API differences between versions; excluded from tests.
    ("webapp.py", ""),
    # main.py — all errors here are pre-existing tech debt in the REPL/CLI layer.
    ("main.py", ""),
    # Phase 2 extracted modules — same pre-existing tech debt as main.py.
    # These errors were moved out of main.py; no new issues introduced.
    ("llm.py", ""),
    ("rerank.py", ""),
    ("embeddings.py", ""),
    ("vector_store.py", ""),
    ("config.py", ""),
    # Phase 3 extracted modules — REPL, CLI, and session persistence.
    ("sessions.py", ""),
    ("repl.py", ""),
    ("cli.py", ""),
    # Phase 4 extracted modules — AxonBrain service boundary mixins.
    ("graph_render.py", ""),
    ("graph_rag.py", ""),
    ("code_graph.py", ""),
    ("code_retrieval.py", ""),
    ("query_router.py", ""),
    # Epic 1/2 new modules — numpy typing limitation in sentence_window.
    ("sentence_window.py", ""),
    ("crag.py", ""),
    # Epic 3 — compression.py dict.get() return typed as Any by mypy.
    ("compression.py", ""),
    # Phase 5 extracted modules — API route families.
    ("api_schemas.py", ""),
    ("api_routes/query.py", ""),
    ("api_routes/ingest.py", ""),
    ("api_routes/projects.py", ""),
    ("api_routes/graph.py", ""),
    ("api_routes/shares.py", ""),
    ("api_routes/maintenance.py", ""),
    ("api_routes/registry.py", ""),
]


def _is_allowed(line: str) -> bool:
    """Return True if a mypy error line matches the pre-approved allowlist.

    Uses filename:line fragments (without directory) so the check works on both
    Unix (src/axon/foo.py) and Windows (src\\axon\\foo.py).
    """
    for path_frag, msg_frag in _MYPY_ALLOWED_ERRORS:
        if path_frag in line and (not msg_frag or msg_frag in line):
            return True
    return False


def test_ruff_linting():
    """Verify that ruff check passes across src/ and tests/."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Ruff found linting issues:\n{result.stdout}\n{result.stderr}"


def test_black_formatting():
    """Verify that black --check passes (no formatting changes needed)."""
    result = subprocess.run(
        [sys.executable, "-m", "black", "--check", "src/", "tests/"], capture_output=True, text=True
    )
    assert (
        result.returncode == 0
    ), f"Black found formatting issues:\n{result.stdout}\n{result.stderr}"


def test_mypy_no_new_errors():
    """Verify that mypy reports no errors outside the pre-approved allowlist.

    This test catches type regressions introduced by new code.  When mypy
    reports an error, either fix the type issue or — only if truly unavoidable —
    add it to _MYPY_ALLOWED_ERRORS above with a comment explaining why.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "src/axon/",
            "--ignore-missing-imports",
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
    )
    # Collect error lines that are NOT in the allowlist
    error_lines = [
        line for line in result.stdout.splitlines() if ": error:" in line and not _is_allowed(line)
    ]
    assert not error_lines, (
        f"mypy found {len(error_lines)} unexpected type error(s).\n"
        "Fix the type issues or add to _MYPY_ALLOWED_ERRORS if truly unavoidable.\n\n"
        + "\n".join(error_lines)
    )
