"""Tests for the ``axon-ui`` console-script entry point.

Deliberately in its own module with **no** ``importorskip("streamlit")``:
``axon.webapp_launcher`` must be importable and fully exercisable without
Streamlit installed, which is the case in CI (``.[dev]`` does not pull it in)
and — since Streamlit was dropped from ``[starter]`` and ``[all]`` — the common
case for users too.
"""

import subprocess
import sys
from unittest.mock import patch

import pytest

from axon import webapp_launcher


class TestDeprecationNotice:
    def test_notice_names_the_replacement(self):
        notice = webapp_launcher.DEPRECATION_NOTICE
        assert "DEPRECATED" in notice
        assert "axon-api" in notice
        assert "http://localhost:8420/gui/" in notice

    def test_missing_streamlit_notice_points_at_the_gui_first(self):
        """The hint must lead with the no-dependency option, not pip install."""
        notice = webapp_launcher.MISSING_STREAMLIT_NOTICE
        assert "axon-api" in notice
        assert 'pip install "axon-rag[ui]"' in notice
        assert notice.index("axon-api") < notice.index("pip install")

    def test_webapp_script_path_points_at_the_streamlit_script(self):
        path = webapp_launcher.webapp_script_path()
        assert path.endswith("webapp.py")


class TestMainUi:
    """`axon-ui` still launches, but must announce its deprecation."""

    def test_emits_deprecation_warning_and_still_launches(self, capsys):
        with (
            patch.object(webapp_launcher, "_streamlit_available", return_value=True),
            patch.object(subprocess, "run") as mock_run,
        ):
            with pytest.warns(DeprecationWarning, match="axon-ui"):
                webapp_launcher.main_ui()

        # Deprecation is a nudge, not a removal — it still starts Streamlit.
        mock_run.assert_called_once()
        argv = mock_run.call_args[0][0]
        assert argv[:2] == ["streamlit", "run"]
        assert argv[2].endswith("webapp.py")
        assert "DEPRECATED" in capsys.readouterr().err

    def test_exits_with_hint_when_streamlit_missing(self, capsys):
        """Regression: this path used to be unreachable.

        `axon.webapp` renders Streamlit at module scope, so importing it without
        Streamlit raised `NameError: name 'st' is not defined` long before the
        friendly message could print. Users landed on a raw traceback instead of
        an install hint.
        """
        with (
            patch.object(webapp_launcher, "_streamlit_available", return_value=False),
            patch.object(subprocess, "run") as mock_run,
        ):
            with pytest.warns(DeprecationWarning):
                with pytest.raises(SystemExit) as exc:
                    webapp_launcher.main_ui()

        assert exc.value.code == 1
        mock_run.assert_not_called()
        err = capsys.readouterr().err
        assert "streamlit is not installed" in err
        assert "axon-rag[ui]" in err

    def test_launcher_imports_without_streamlit(self, monkeypatch):
        """The whole point of the split: no Streamlit import at module scope."""
        monkeypatch.setitem(sys.modules, "streamlit", None)
        import importlib

        reloaded = importlib.reload(webapp_launcher)
        assert reloaded._streamlit_available() is False


class TestStreamlitAvailable:
    def test_true_when_importable(self):
        with patch("importlib.util.find_spec", return_value=object()):
            assert webapp_launcher._streamlit_available() is True

    def test_false_when_absent(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert webapp_launcher._streamlit_available() is False

    def test_false_when_find_spec_raises(self):
        """sys.modules['streamlit'] = None makes find_spec raise ValueError."""
        with patch("importlib.util.find_spec", side_effect=ValueError):
            assert webapp_launcher._streamlit_available() is False
