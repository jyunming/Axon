"""
tests/conftest.py

Project-level pytest fixtures for the Axon test suite.
"""

import importlib
import logging
import os
import shutil
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip test files that depend on optional packages not installed in this env.
# collect_ignore_glob would hide the files unconditionally; instead we use
# pytest's collect_ignore list so CI with full deps still runs them.
# ---------------------------------------------------------------------------
collect_ignore: list[str] = []

_OPTIONAL_TEST_FILES = {
    "e2e/test_mcp_bridge_e2e.py": "mcp",
}

for _rel, _pkg in _OPTIONAL_TEST_FILES.items():
    if importlib.util.find_spec(_pkg) is None:
        collect_ignore.append(str(Path(__file__).parent / _rel))

# ---------------------------------------------------------------------------
# tmp_path override — Windows: the system pytest temp dir
# (C:\Users\...\AppData\Local\Temp\pytest-of-<user>) is inaccessible when
# prior test runs hold open handles.  Using a project-local directory avoids
# that.  black/ruff exclude .test_tmp (see pyproject.toml) so locked
# directories from prior runs do not break linting.
# ---------------------------------------------------------------------------
BASE_TEST_TMP = Path(__file__).parent / ".test_tmp"


@pytest.fixture
def tmp_path():
    """Provide a temporary directory under the project tree."""
    base = Path(os.environ.get("AXON_TEST_TMP", BASE_TEST_TMP))
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"tmp{uuid.uuid4().hex}"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Axon logger state isolation
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clear_dry_run_env():
    """Remove AXON_DRY_RUN between tests.

    cli.py sets os.environ["AXON_DRY_RUN"] = "1" unconditionally when --dry-run
    is passed.  Tests that call the CLI with --dry-run would otherwise poison
    every subsequent test that checks os.getenv("AXON_DRY_RUN").
    """
    original = os.environ.pop("AXON_DRY_RUN", None)
    yield
    if "AXON_DRY_RUN" in os.environ:
        del os.environ["AXON_DRY_RUN"]
    if original is not None:
        os.environ["AXON_DRY_RUN"] = original


@pytest.fixture(autouse=True)
def _stub_pypi_update_check(request, monkeypatch):
    """No test should make a real network call to PyPI.

    axon.update_check.check_for_update() is called from several startup
    hooks (api.py's lifespan background task, repl.py's background thread,
    cli.py's `axon update`, doctor.py's `--doctor` check) that individual
    test files for those surfaces don't all know to mock. Stub it globally
    at the module-attribute level — every caller does a lazy
    ``from axon.update_check import check_for_update`` inside its own
    function body, so patching the attribute here is visible to all of
    them. Skipped for test_update_check.py itself, which exercises the
    real function against a mocked httpx.get.
    """
    if request.module.__name__.rsplit(".", 1)[-1] == "test_update_check":
        yield
        return
    from axon.update_check import UpdateCheckResult

    monkeypatch.setattr(
        "axon.update_check.check_for_update",
        lambda **kwargs: UpdateCheckResult(
            current="0.0.0+test", latest=None, update_available=False, skipped_reason="error"
        ),
    )
    yield


@pytest.fixture(autouse=True)
def reset_axon_logger():
    """Restore Axon logger state after each test.

    _interactive_repl() sets the 'Axon' logger to WARNING with propagate=False
    to silence noisy output during interactive sessions.  When REPL e2e tests
    call _interactive_repl(), that state persists for the rest of the pytest
    session and breaks any subsequent test that relies on caplog capturing INFO
    messages from the 'Axon' logger.  This fixture saves and restores the level
    and propagate flag so every test starts with a clean slate.
    """
    axon_logger = logging.getLogger("Axon")
    original_level = axon_logger.level
    original_propagate = axon_logger.propagate
    yield
    axon_logger.setLevel(original_level)
    axon_logger.propagate = original_propagate
