"""
tests/conftest.py

Project-level pytest fixtures for the Axon test suite.
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# tmp_path override — Windows: the system pytest temp dir
# (C:\Users\...\AppData\Local\Temp\pytest-of-<user>) is inaccessible when
# prior test runs hold open handles.  Using a project-local directory avoids
# that.  black/ruff exclude .test_tmp (see pyproject.toml) so locked
# directories from prior runs do not break linting.
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_path():
    """Provide a temporary directory under the project tree."""
    base = Path(__file__).parent / ".test_tmp"
    base.mkdir(exist_ok=True)
    d = tempfile.mkdtemp(dir=base)
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Axon logger state isolation
# ---------------------------------------------------------------------------
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
