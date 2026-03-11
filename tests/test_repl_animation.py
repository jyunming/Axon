from axon.main import _AXON_BLUE, _BRAIN_ART, _get_brain_anim_row


def test_get_brain_anim_row_width():
    # Test that the output width is correct
    width = 40
    row = _get_brain_anim_row(0, 0, width)
    # The output contains ANSI codes, so we strip them to check visible width
    import re

    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    plain_row = ansi_escape.sub("", row)
    assert len(plain_row) == width


def test_get_brain_anim_row_small_width():
    # Test fallback for small width
    width = 20
    row = _get_brain_anim_row(0, 0, width)
    assert row == " " * width


def test_get_brain_anim_row_fully_connected():
    # Test the fully connected state (frame=-1)
    width = 25
    row = _get_brain_anim_row(0, -1, width)
    # Ensure it's not just spaces or the base design without color
    assert "\x1b[38;2;" in row
    # The first cell '(O)' should be in GLOW color (phase 3)
    # Row 0, cols 0-3 are L cells. Phase 3 in fully connected is GLOW.
    # GLOW is "\x1b[38;2;255;255;255m"
    assert "\x1b[38;2;255;255;255m(O)" in row


def test_axon_blue_exists():
    # Ensure we didn't accidentally delete the original blue gradient
    assert len(_AXON_BLUE) == 6
    assert "\x1b[38;2;173;216;230m" in _AXON_BLUE


def test_brain_art_content():
    assert len(_BRAIN_ART) == 6
    assert "(O)~~." in _BRAIN_ART[0]


def test_open_studio_config_path_defaults():
    import os

    from axon.main import AxonConfig

    # Mock or check defaults
    config = AxonConfig()
    home = os.path.join(os.path.expanduser("~"), ".axon")

    # Verify that the paths default to the user's home directory subdirectory
    # (unless overridden by env vars, but in a clean test env they should be default)
    assert config.vector_store_path == os.path.join(home, "projects", "default", "chroma_data")
    assert config.bm25_path == os.path.join(home, "projects", "default", "bm25_index")
    assert config.projects_root == os.path.join(home, "projects")
