"""
tests/conftest.py

Project-level pytest fixtures for the Axon test suite.
"""
import shutil
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# tmp_path override — avoids a Windows PermissionError on the global pytest
# temp directory (C:\Users\...\AppData\Local\Temp\pytest-of-<user>).
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_path():
    """Provide a temporary directory that lives under the project tree."""
    base = Path(__file__).parent / ".test_tmp"
    base.mkdir(exist_ok=True)
    d = tempfile.mkdtemp(dir=base)
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)
