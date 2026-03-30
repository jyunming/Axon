from __future__ import annotations
"""Tests for axon.sessions — covering uncovered lines."""
import os
from unittest.mock import MagicMock

import pytest


def _make_brain(provider="ollama", model="llama3", project="default"):
    brain = MagicMock()
    brain.config.llm_provider = provider
    brain.config.llm_model = model
    brain._active_project = project
    return brain


class TestSessionsBasic:
    def test_new_session_structure(self, tmp_path):
        from axon.sessions import _new_session

        brain = _make_brain()
        s = _new_session(brain)
        assert "id" in s
        assert "started_at" in s
        assert s["provider"] == "ollama"
        assert s["model"] == "llama3"
        assert s["project"] == "default"
        assert s["history"] == []

    def test_save_and_load_session(self, tmp_path):
        from axon.sessions import _load_session, _save_session

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("HOME", str(tmp_path))
            import axon.sessions as _s

            orig = _s._SESSIONS_DIR
            _s._SESSIONS_DIR = str(tmp_path / "sessions")
            os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

            session = {
                "id": "20240101T120000000",
                "started_at": "2024-01-01T12:00:00Z",
                "provider": "ollama",
                "model": "llama3",
                "project": "default",
                "history": [{"role": "user", "content": "hello"}],
            }
            _save_session(session)
            loaded = _load_session("20240101T120000000")
            assert loaded is not None
            assert loaded["id"] == "20240101T120000000"
            assert len(loaded["history"]) == 1

            _s._SESSIONS_DIR = orig

    def test_load_session_nonexistent_returns_none(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _load_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        result = _load_session("nonexistent_id")
        assert result is None

        _s._SESSIONS_DIR = orig

    def test_list_sessions_empty(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _list_sessions

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        sessions = _list_sessions()
        assert sessions == []

        _s._SESSIONS_DIR = orig

    def test_list_sessions_returns_saved(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _list_sessions, _save_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        for i in range(3):
            _save_session(
                {
                    "id": f"2024010{i}T120000000",
                    "started_at": f"2024-01-0{i}T12:00:00Z",
                    "provider": "ollama",
                    "model": "llama3",
                    "project": "default",
                    "history": [],
                }
            )

        sessions = _list_sessions(limit=2)
        assert len(sessions) == 2

        _s._SESSIONS_DIR = orig

    def test_print_sessions_empty(self, capsys):
        from axon.sessions import _print_sessions

        _print_sessions([])
        captured = capsys.readouterr()
        assert "no saved sessions" in captured.out

    def test_print_sessions_with_data(self, capsys):
        from axon.sessions import _print_sessions

        sessions = [
            {
                "id": "20240101T120000000",
                "started_at": "2024-01-01T12:00:00Z",
                "provider": "ollama",
                "model": "llama3",
                "history": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
            }
        ]
        _print_sessions(sessions)
        captured = capsys.readouterr()
        assert "ollama" in captured.out
        assert "llama3" in captured.out

    def test_save_session_bad_path_no_crash(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _save_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = "/nonexistent/path/that/does/not/exist"

        session = {
            "id": "test",
            "project": "default",
            "history": [],
        }
        # Should not raise
        _save_session(session)

        _s._SESSIONS_DIR = orig

    def test_sessions_dir_default_project(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _sessions_dir

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        d = _sessions_dir()
        assert os.path.exists(d)
        _s._SESSIONS_DIR = orig

    def test_load_session_corrupt_json(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _load_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        # Write corrupt JSON
        corrupt_path = tmp_path / "sessions" / "session_badid.json"
        corrupt_path.write_text("not valid json{{{", encoding="utf-8")

        result = _load_session("badid")
        assert result is None

        _s._SESSIONS_DIR = orig

    def test_sessions_dir_project(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _sessions_dir

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "axon.projects.project_sessions_path", lambda x: str(tmp_path / "project_sessions")
            )
            d = _sessions_dir(project="myproject")
            assert "project_sessions" in d
            assert os.path.exists(d)
        _s._SESSIONS_DIR = orig

    def test_load_session_not_dict(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _load_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        # Write valid JSON but not a dict
        not_dict_path = tmp_path / "sessions" / "session_list.json"
        not_dict_path.write_text("[]", encoding="utf-8")

        result = _load_session("list")
        assert result is None

        _s._SESSIONS_DIR = orig
"""Tests for axon.sessions — covering uncovered lines."""
import os
from unittest.mock import MagicMock

import pytest


def _make_brain(provider="ollama", model="llama3", project="default"):
    brain = MagicMock()
    brain.config.llm_provider = provider
    brain.config.llm_model = model
    brain._active_project = project
    return brain


class TestSessionsBasic:
    def test_new_session_structure(self, tmp_path):
        from axon.sessions import _new_session

        brain = _make_brain()
        s = _new_session(brain)
        assert "id" in s
        assert "started_at" in s
        assert s["provider"] == "ollama"
        assert s["model"] == "llama3"
        assert s["project"] == "default"
        assert s["history"] == []

    def test_save_and_load_session(self, tmp_path):
        from axon.sessions import _load_session, _save_session

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("HOME", str(tmp_path))
            import axon.sessions as _s

            orig = _s._SESSIONS_DIR
            _s._SESSIONS_DIR = str(tmp_path / "sessions")
            os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

            session = {
                "id": "20240101T120000000",
                "started_at": "2024-01-01T12:00:00Z",
                "provider": "ollama",
                "model": "llama3",
                "project": "default",
                "history": [{"role": "user", "content": "hello"}],
            }
            _save_session(session)
            loaded = _load_session("20240101T120000000")
            assert loaded is not None
            assert loaded["id"] == "20240101T120000000"
            assert len(loaded["history"]) == 1

            _s._SESSIONS_DIR = orig

    def test_load_session_nonexistent_returns_none(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _load_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        result = _load_session("nonexistent_id")
        assert result is None

        _s._SESSIONS_DIR = orig

    def test_list_sessions_empty(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _list_sessions

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        sessions = _list_sessions()
        assert sessions == []

        _s._SESSIONS_DIR = orig

    def test_list_sessions_returns_saved(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _list_sessions, _save_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        for i in range(3):
            _save_session(
                {
                    "id": f"2024010{i}T120000000",
                    "started_at": f"2024-01-0{i}T12:00:00Z",
                    "provider": "ollama",
                    "model": "llama3",
                    "project": "default",
                    "history": [],
                }
            )

        sessions = _list_sessions(limit=2)
        assert len(sessions) == 2

        _s._SESSIONS_DIR = orig

    def test_print_sessions_empty(self, capsys):
        from axon.sessions import _print_sessions

        _print_sessions([])
        captured = capsys.readouterr()
        assert "no saved sessions" in captured.out

    def test_print_sessions_with_data(self, capsys):
        from axon.sessions import _print_sessions

        sessions = [
            {
                "id": "20240101T120000000",
                "started_at": "2024-01-01T12:00:00Z",
                "provider": "ollama",
                "model": "llama3",
                "history": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
            }
        ]
        _print_sessions(sessions)
        captured = capsys.readouterr()
        assert "ollama" in captured.out
        assert "llama3" in captured.out

    def test_save_session_bad_path_no_crash(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _save_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = "/nonexistent/path/that/does/not/exist"

        session = {
            "id": "test",
            "project": "default",
            "history": [],
        }
        # Should not raise
        _save_session(session)

        _s._SESSIONS_DIR = orig

    def test_sessions_dir_default_project(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _sessions_dir

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        d = _sessions_dir()
        assert os.path.exists(d)
        _s._SESSIONS_DIR = orig

    def test_load_session_corrupt_json(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _load_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        # Write corrupt JSON
        corrupt_path = tmp_path / "sessions" / "session_badid.json"
        corrupt_path.write_text("not valid json{{{", encoding="utf-8")

        result = _load_session("badid")
        assert result is None

        _s._SESSIONS_DIR = orig

    def test_sessions_dir_project(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _sessions_dir

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "axon.projects.project_sessions_path", lambda x: str(tmp_path / "project_sessions")
            )
            d = _sessions_dir(project="myproject")
            assert "project_sessions" in d
            assert os.path.exists(d)
        _s._SESSIONS_DIR = orig

    def test_load_session_not_dict(self, tmp_path):
        import axon.sessions as _s
        from axon.sessions import _load_session

        orig = _s._SESSIONS_DIR
        _s._SESSIONS_DIR = str(tmp_path / "sessions")
        os.makedirs(_s._SESSIONS_DIR, exist_ok=True)

        # Write valid JSON but not a dict
        not_dict_path = tmp_path / "sessions" / "session_list.json"
        not_dict_path.write_text("[]", encoding="utf-8")

        result = _load_session("list")
        assert result is None

        _s._SESSIONS_DIR = orig
