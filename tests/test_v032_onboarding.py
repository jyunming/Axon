"""Tests for v0.3.2 item (23): onboarding simplification.

Covers:
- ``axon.doctor`` checks (Python, Ollama, model, store, extras).
- Aggregate ``run_doctor`` overall-status logic.
- ``cli._is_first_run`` detection: config absent + store empty → True.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

from axon.doctor import (
    Check,
    DoctorReport,
    check_default_model,
    check_ollama_reachable,
    check_optional_extras,
    check_python_version,
    check_store_writable,
    render_report,
    run_doctor,
)

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestPythonVersionCheck:
    def test_passes_on_supported_python(self):
        # The harness running these tests is itself ≥3.10, so this should pass.
        result = check_python_version()
        assert result.status == "ok"
        assert sys.version.split()[0] in result.detail

    def test_fails_on_unsupported_python(self, monkeypatch):
        monkeypatch.setattr("axon.doctor._MIN_PYTHON", (99, 0))
        result = check_python_version()
        assert result.status == "error"
        assert "99.0" in result.hint


class TestOllamaReachable:
    def test_warning_when_unreachable(self):
        # Use a port nothing is listening on.
        result = check_ollama_reachable(base_url="http://127.0.0.1:1")
        assert result.status == "warning"
        assert "127.0.0.1:1" in result.detail
        assert "OLLAMA_HOST" in result.hint or "ollama" in result.hint.lower()

    def test_ok_when_reachable(self, monkeypatch):
        # Stub urlopen to return a 200 response object.
        from contextlib import contextmanager

        @contextmanager
        def _fake_urlopen(*_args, **_kwargs):
            yield SimpleNamespace(status=200)

        import axon.doctor as mod

        with patch.object(
            __import__("urllib.request", fromlist=["urlopen"]),
            "urlopen",
            _fake_urlopen,
        ):
            result = mod.check_ollama_reachable(base_url="http://localhost:11434")
        assert result.status == "ok"
        assert "11434" in result.detail


class TestModelPresence:
    def test_warning_when_model_missing_from_tags_response(self):
        from contextlib import contextmanager

        @contextmanager
        def _fake_urlopen(*_args, **_kwargs):
            class _R:
                def read(self):
                    return b'{"models":[{"name":"qwen3:4b"}]}'

            yield _R()

        with patch.object(
            __import__("urllib.request", fromlist=["urlopen"]),
            "urlopen",
            _fake_urlopen,
        ):
            result = check_default_model(model_name="llama3.1:8b")
        assert result.status == "warning"
        assert "llama3.1:8b" in result.detail
        assert "ollama pull llama3.1:8b" in result.hint

    def test_ok_when_model_in_tags_response(self):
        from contextlib import contextmanager

        @contextmanager
        def _fake_urlopen(*_args, **_kwargs):
            class _R:
                def read(self):
                    return b'{"models":[{"name":"llama3.1:8b"}]}'

            yield _R()

        with patch.object(
            __import__("urllib.request", fromlist=["urlopen"]),
            "urlopen",
            _fake_urlopen,
        ):
            result = check_default_model(model_name="llama3.1:8b")
        assert result.status == "ok"
        assert result.detail == "llama3.1:8b"


class TestStoreWritable:
    def test_ok_when_dir_writable(self, tmp_path):
        result = check_store_writable(str(tmp_path / "store"))
        assert result.status == "ok"
        # No probe file leaks behind
        assert not any(p.name.startswith(".doctor_write_probe") for p in tmp_path.rglob("*"))

    def test_error_when_path_unwritable(self, tmp_path, monkeypatch):
        # Force write to fail by patching Path.write_text on the probe path.
        target = tmp_path / "store"
        target.mkdir()

        original_write_text = type(target).write_text

        def _fail_write(self, *_a, **_kw):
            if self.name == ".doctor_write_probe":
                raise OSError("read-only filesystem")
            return original_write_text(self, *_a, **_kw)

        monkeypatch.setattr(type(target), "write_text", _fail_write)
        result = check_store_writable(str(target))
        assert result.status == "error"
        assert "AXON_STORE_BASE" in result.hint


class TestOptionalExtras:
    def test_ok_when_streamlit_and_keyring_present(self, monkeypatch):
        # Stub both imports so the check sees them as available.
        fake = SimpleNamespace()
        monkeypatch.setitem(sys.modules, "streamlit", fake)
        monkeypatch.setitem(sys.modules, "cryptography", fake)
        monkeypatch.setitem(sys.modules, "keyring", fake)
        result = check_optional_extras()
        assert result.status == "ok"

    def test_warning_when_streamlit_missing(self, monkeypatch):
        # Force the streamlit import to fail by removing it from sys.modules
        # and inserting a None entry so import raises ImportError.
        monkeypatch.setitem(sys.modules, "streamlit", None)
        result = check_optional_extras()
        assert result.status == "warning"
        assert "axon-rag[starter]" in result.hint


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class TestRunDoctor:
    def test_overall_error_when_any_check_errors(self, monkeypatch):
        # Force the python check to fail.
        monkeypatch.setattr("axon.doctor._MIN_PYTHON", (99, 0))
        report = run_doctor()
        assert report.overall == "error"
        names = [c.name for c in report.checks]
        assert "Python version" in names

    def test_overall_warning_when_no_errors_only_warnings(self, monkeypatch):
        # Patch out individual checks to produce a no-error / has-warning report.
        from axon import doctor as mod

        def _ok(*a, **kw):
            return Check("ok-thing", "ok")

        def _warn(*a, **kw):
            return Check("warn-thing", "warning")

        monkeypatch.setattr(mod, "_CHECK_FUNCS", (_ok, _warn))
        report = run_doctor()
        assert report.overall == "warning"

    def test_overall_ok_when_all_pass(self, monkeypatch):
        from axon import doctor as mod

        def _ok(*a, **kw):
            return Check("ok-thing", "ok")

        monkeypatch.setattr(mod, "_CHECK_FUNCS", (_ok, _ok))
        report = run_doctor()
        assert report.overall == "ok"

    def test_render_report_has_summary(self):
        report = DoctorReport(
            overall="warning",
            checks=[Check("Test", "warning", detail="…", hint="do X")],
        )
        out = render_report(report, use_color=False)
        assert "Test" in out
        assert "do X" in out
        assert "Some optional checks" in out


# ---------------------------------------------------------------------------
# CLI first-run detection
# ---------------------------------------------------------------------------


class TestIsFirstRun:
    def test_explicit_config_disables_first_run(self, tmp_path):
        from axon.cli import _is_first_run

        args = SimpleNamespace(config=str(tmp_path / "explicit.yaml"))
        assert _is_first_run(args) is False

    def test_existing_default_config_disables_first_run(self, tmp_path, monkeypatch):
        from axon import cli

        monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
        cfg = tmp_path / ".config" / "axon" / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("foo: bar", encoding="utf-8")
        args = SimpleNamespace(config=None)
        assert cli._is_first_run(args) is False

    def test_no_config_no_store_returns_true(self, tmp_path, monkeypatch):
        from axon import cli

        monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
        monkeypatch.delenv("AXON_STORE_BASE", raising=False)
        args = SimpleNamespace(config=None)
        assert cli._is_first_run(args) is True

    def test_existing_project_under_store_returns_false(self, tmp_path, monkeypatch):
        from axon import cli

        monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
        store = tmp_path / ".axon" / "AxonStore" / "alice"
        (store / "default").mkdir(parents=True)
        (store / "default" / "meta.json").write_text("{}", encoding="utf-8")
        monkeypatch.delenv("AXON_STORE_BASE", raising=False)
        args = SimpleNamespace(config=None)
        assert cli._is_first_run(args) is False

    def test_axon_store_base_env_respected(self, tmp_path, monkeypatch):
        from axon import cli

        monkeypatch.setattr(cli.Path, "home", lambda: tmp_path / "elsewhere")
        custom = tmp_path / "custom_store"
        custom.mkdir()
        os.environ["AXON_STORE_BASE"] = str(custom)
        try:
            args = SimpleNamespace(config=None)
            # Empty custom store dir → first run
            assert cli._is_first_run(args) is True
        finally:
            del os.environ["AXON_STORE_BASE"]
