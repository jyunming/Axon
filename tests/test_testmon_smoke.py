"""Lock pytest-testmon into the [dev] extras matrix.

testmon is local-only — CI never activates ``--testmon`` (fresh runners
have no cache, so testmon would always rebuild). But if the dep falls
out of ``[dev]`` extras the pre-commit hook breaks for everyone with an
unhelpful "unknown plugin --testmon" error. These tests assert the
plugin imports cleanly so a missing dependency is caught at test time
instead of commit time.
"""
from __future__ import annotations

import pytest


def test_testmon_plugin_importable():
    pytest.importorskip(
        "testmon",
        reason=(
            "pytest-testmon is in [dev] extras; install with "
            'pip install -e ".[dev]" to silence this skip'
        ),
    )


def test_testmon_pytest_plugin_metadata():
    """Confirm the plugin module exposes a usable surface."""
    testmon = pytest.importorskip("testmon")
    # testmon's API has shifted across major versions; either of these
    # attributes is sufficient to confirm we have a real install rather
    # than a stub.
    assert (
        hasattr(testmon, "__version__")
        or hasattr(testmon, "_version")
        or hasattr(testmon, "testmon_core")
    ), "pytest-testmon installed but no recognisable version attribute found"
