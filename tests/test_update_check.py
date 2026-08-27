"""Tests for src/axon/update_check.py — the shared PyPI version-check
primitive and the `axon update` upgrade sequence."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


class TestIsNewer:
    def test_newer_patch(self):
        from axon.update_check import is_newer

        assert is_newer("0.4.5", "0.4.4") is True

    def test_newer_minor(self):
        from axon.update_check import is_newer

        assert is_newer("0.5.0", "0.4.9") is True

    def test_equal_is_not_newer(self):
        from axon.update_check import is_newer

        assert is_newer("0.4.4", "0.4.4") is False

    def test_older_is_not_newer(self):
        from axon.update_check import is_newer

        assert is_newer("0.4.3", "0.4.4") is False

    def test_pre_release_suffix_treated_as_equal(self):
        """Deliberately conservative: same numeric prefix never reports an
        update, even with a differing suffix — avoids guessing PEP 440
        pre-release ordering without a real parser."""
        from axon.update_check import is_newer

        assert is_newer("0.4.4rc1", "0.4.4") is False

    def test_unparseable_falls_back_to_string_comparison(self):
        from axon.update_check import is_newer

        assert is_newer("abc", "abc") is False


# ---------------------------------------------------------------------------
# check_for_update — network, cache, offline
# ---------------------------------------------------------------------------


class TestCheckForUpdate:
    def test_offline_mode_skips_network(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        with patch("httpx.get") as mock_get:
            result = update_check.check_for_update(offline=True)
        mock_get.assert_not_called()
        assert result.skipped_reason == "offline_mode"
        assert result.update_available is False

    def test_success_update_available(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        monkeypatch.setattr(update_check, "current_version", lambda: "0.4.4")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"info": {"version": "0.5.0"}}
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            result = update_check.check_for_update(force=True)
        mock_get.assert_called_once()
        assert result.latest == "0.5.0"
        assert result.update_available is True
        assert result.skipped_reason is None

    def test_success_already_current(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        monkeypatch.setattr(update_check, "current_version", lambda: "0.4.4")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"info": {"version": "0.4.4"}}
        with patch("httpx.get", return_value=mock_resp):
            result = update_check.check_for_update(force=True)
        assert result.update_available is False

    def test_network_error_is_quiet_failure(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        with patch("httpx.get", side_effect=ConnectionError("no network")):
            result = update_check.check_for_update(force=True)
        assert result.skipped_reason == "error"
        assert result.latest is None
        assert result.update_available is False

    def test_timeout_is_quiet_failure(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        with patch("httpx.get", side_effect=TimeoutError("timed out")):
            result = update_check.check_for_update(force=True)
        assert result.skipped_reason == "error"

    def test_malformed_json_is_quiet_failure(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("bad json")
        with patch("httpx.get", return_value=mock_resp):
            result = update_check.check_for_update(force=True)
        assert result.skipped_reason == "error"

    def test_http_error_status_is_quiet_failure(self, tmp_path, monkeypatch):
        from axon import update_check

        monkeypatch.setattr(update_check, "_CACHE_PATH", tmp_path / "cache.json")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = RuntimeError("500")
        with patch("httpx.get", return_value=mock_resp):
            result = update_check.check_for_update(force=True)
        assert result.skipped_reason == "error"

    def test_cache_hit_skips_network(self, tmp_path, monkeypatch):
        from axon import update_check

        cache_path = tmp_path / "cache.json"
        cache_path.write_text(
            json.dumps({"checked_at": __import__("time").time(), "latest_version": "9.9.9"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(update_check, "_CACHE_PATH", cache_path)
        monkeypatch.setattr(update_check, "current_version", lambda: "0.4.4")
        with patch("httpx.get") as mock_get:
            result = update_check.check_for_update()
        mock_get.assert_not_called()
        assert result.from_cache is True
        assert result.latest == "9.9.9"
        assert result.update_available is True

    def test_stale_cache_triggers_network(self, tmp_path, monkeypatch):
        from axon import update_check

        cache_path = tmp_path / "cache.json"
        stale_time = __import__("time").time() - update_check._CACHE_TTL_S - 100
        cache_path.write_text(
            json.dumps({"checked_at": stale_time, "latest_version": "1.0.0"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(update_check, "_CACHE_PATH", cache_path)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"info": {"version": "2.0.0"}}
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            result = update_check.check_for_update()
        mock_get.assert_called_once()
        assert result.latest == "2.0.0"

    def test_force_bypasses_fresh_cache(self, tmp_path, monkeypatch):
        from axon import update_check

        cache_path = tmp_path / "cache.json"
        cache_path.write_text(
            json.dumps({"checked_at": __import__("time").time(), "latest_version": "1.0.0"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(update_check, "_CACHE_PATH", cache_path)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"info": {"version": "2.0.0"}}
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            result = update_check.check_for_update(force=True)
        mock_get.assert_called_once()
        assert result.latest == "2.0.0"


class TestFormatSuggestion:
    def test_none_when_up_to_date(self):
        from axon.update_check import UpdateCheckResult, format_suggestion

        r = UpdateCheckResult(current="0.4.4", latest="0.4.4", update_available=False)
        assert format_suggestion(r) is None

    def test_none_when_check_failed(self):
        from axon.update_check import UpdateCheckResult, format_suggestion

        r = UpdateCheckResult(
            current="0.4.4", latest=None, update_available=False, skipped_reason="error"
        )
        assert format_suggestion(r) is None

    def test_text_when_update_available(self):
        from axon.update_check import UpdateCheckResult, format_suggestion

        r = UpdateCheckResult(current="0.4.4", latest="0.5.0", update_available=True)
        text = format_suggestion(r)
        assert "0.4.4" in text and "0.5.0" in text and "axon update" in text


# ---------------------------------------------------------------------------
# Install-method / environment detection
# ---------------------------------------------------------------------------


class TestDetectInstallMethod:
    def test_pipx_via_env_var(self, monkeypatch):
        from axon import update_check

        monkeypatch.setenv("PIPX_HOME", "/home/user/.local/pipx")
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        assert update_check.detect_install_method() == "pipx"

    def test_conda_via_env_var(self, monkeypatch):
        from axon import update_check

        monkeypatch.delenv("PIPX_HOME", raising=False)
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/axon")
        assert update_check.detect_install_method() == "conda"

    def test_falls_back_to_pip(self, monkeypatch):
        from axon import update_check

        monkeypatch.delenv("PIPX_HOME", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
        assert update_check.detect_install_method() == "pip"

    def test_upgrade_command_pip(self):
        from axon.update_check import upgrade_command_for

        assert upgrade_command_for("pip") == ["pip", "install", "-U", "axon-rag"]

    def test_upgrade_command_pipx(self):
        from axon.update_check import upgrade_command_for

        assert upgrade_command_for("pipx") == ["pipx", "upgrade", "axon-rag"]

    def test_upgrade_command_conda(self):
        from axon.update_check import upgrade_command_for

        assert upgrade_command_for("conda") == ["conda", "update", "-y", "axon-rag"]


class TestIsRunningInDocker:
    def test_true_when_dockerenv_present(self):
        from axon import update_check

        with patch("axon.update_check.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = True
            assert update_check.is_running_in_docker() is True

    def test_false_when_no_markers(self):
        from axon import update_check

        with patch("axon.update_check.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            assert update_check.is_running_in_docker() is False


# ---------------------------------------------------------------------------
# run_update — the mutating sequence
# ---------------------------------------------------------------------------


class TestRunUpdate:
    def _patch_check(
        self, monkeypatch, *, available, current="0.4.4", latest="0.5.0", skipped=None
    ):
        from axon import update_check

        def fake_check(**kwargs):
            return update_check.UpdateCheckResult(
                current=current, latest=latest, update_available=available, skipped_reason=skipped
            )

        monkeypatch.setattr(update_check, "check_for_update", fake_check)

    def test_offline_mode_refuses(self, monkeypatch):
        from axon.update_check import run_update

        self._patch_check(monkeypatch, available=False, skipped="offline_mode")
        result = run_update(config=SimpleNamespace(offline_mode=True))
        assert result.status == "refused"
        assert "offline_mode" in result.detail

    def test_check_error_fails(self, monkeypatch):
        from axon.update_check import run_update

        self._patch_check(monkeypatch, available=False, latest=None, skipped="error")
        result = run_update(config=None)
        assert result.status == "failed"

    def test_already_current_short_circuits(self, monkeypatch):
        from axon.update_check import run_update

        self._patch_check(monkeypatch, available=False, current="0.4.4", latest="0.4.4")
        result = run_update(config=None)
        assert result.status == "already_current"

    def test_docker_bail_out(self, monkeypatch):
        from axon import update_check

        self._patch_check(monkeypatch, available=True)
        monkeypatch.setattr(update_check, "is_running_in_docker", lambda: True)
        result = update_check.run_update(config=None)
        assert result.status == "refused"
        assert "docker compose" in result.detail

    def test_refuses_when_axon_api_live(self, monkeypatch):
        from axon import update_check

        self._patch_check(monkeypatch, available=True)
        monkeypatch.setattr(update_check, "is_running_in_docker", lambda: False)
        fake_config = SimpleNamespace(offline_mode=False)
        with patch(
            "axon.server_client.find_live_server_for_store",
            return_value={"host": "127.0.0.1", "port": 8420, "pid": 1234},
        ):
            result = update_check.run_update(config=fake_config)
        assert result.status == "refused"
        assert "1234" in result.detail

    def test_successful_upgrade_reinstalls_vsix(self, monkeypatch):
        from axon import update_check

        self._patch_check(monkeypatch, available=True, current="0.4.4", latest="0.5.0")
        monkeypatch.setattr(update_check, "is_running_in_docker", lambda: False)
        monkeypatch.setattr(update_check, "current_version", lambda: "0.5.0")
        mock_subprocess_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_subprocess_result) as mock_run, patch.object(
            update_check, "_reinstall_vscode_extension", return_value="installed"
        ):
            result = update_check.run_update(config=None)
        mock_run.assert_called_once()
        assert result.status == "upgraded"
        assert result.package_before == "0.4.4"
        assert result.package_after == "0.5.0"
        assert result.vsix_status == "installed"

    def test_upgrade_command_failure_reported(self, monkeypatch):
        from axon import update_check

        self._patch_check(monkeypatch, available=True)
        monkeypatch.setattr(update_check, "is_running_in_docker", lambda: False)
        mock_subprocess_result = MagicMock(returncode=1, stdout="", stderr="permission denied")
        with patch("subprocess.run", return_value=mock_subprocess_result):
            result = update_check.run_update(config=None)
        assert result.status == "failed"
        assert "permission denied" in result.detail

    def test_missing_code_cli_degrades_to_partial_success(self, monkeypatch, tmp_path):
        """A missing `code` CLI must not fail the whole command — the
        package upgrade already succeeded."""
        from axon import update_check

        self._patch_check(monkeypatch, available=True, current="0.4.4", latest="0.5.0")
        monkeypatch.setattr(update_check, "is_running_in_docker", lambda: False)
        monkeypatch.setattr(update_check, "current_version", lambda: "0.5.0")
        mock_subprocess_result = MagicMock(returncode=0, stdout="", stderr="")
        vsix_path = tmp_path / "axon-copilot-0.5.0.vsix"
        vsix_path.write_bytes(b"fake")
        with patch("subprocess.run", return_value=mock_subprocess_result), patch(
            "axon.ext_install._find_vsix", return_value=vsix_path
        ), patch("axon.ext_install._find_code_cmd", return_value=None):
            result = update_check.run_update(config=None)
        assert result.status == "upgraded"
        assert "skipped" in result.vsix_status
        assert "code" in result.vsix_status.lower()

    def test_vsix_reinstall_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        from axon import update_check

        vsix_path = tmp_path / "axon-copilot-0.5.0.vsix"
        vsix_path.write_bytes(b"fake")
        mock_failed = MagicMock(returncode=1, stderr="install failed")
        with patch("axon.ext_install._find_vsix", return_value=vsix_path), patch(
            "axon.ext_install._find_code_cmd", return_value="code"
        ), patch("subprocess.run", return_value=mock_failed):
            status = update_check._reinstall_vscode_extension()
        assert "failed" in status

    def test_vsix_reinstall_missing_package_data(self):
        from axon import update_check

        with patch("axon.ext_install._find_vsix", side_effect=FileNotFoundError("no vsix found")):
            status = update_check._reinstall_vscode_extension()
        assert "skipped" in status


# ---------------------------------------------------------------------------
# doctor.py wiring
# ---------------------------------------------------------------------------


class TestDoctorCheckUpdateAvailable:
    def test_offline_mode(self):
        from axon.doctor import check_update_available

        result = check_update_available(offline=True)
        assert result.status == "ok"
        assert "offline" in result.detail

    def test_up_to_date(self, monkeypatch):
        from axon import update_check
        from axon.doctor import check_update_available

        monkeypatch.setattr(
            update_check,
            "check_for_update",
            lambda **kw: update_check.UpdateCheckResult("0.4.4", "0.4.4", False),
        )
        result = check_update_available(offline=False)
        assert result.status == "ok"

    def test_update_available_is_warning_with_hint(self, monkeypatch):
        from axon import update_check
        from axon.doctor import check_update_available

        monkeypatch.setattr(
            update_check,
            "check_for_update",
            lambda **kw: update_check.UpdateCheckResult("0.4.4", "0.5.0", True),
        )
        result = check_update_available(offline=False)
        assert result.status == "warning"
        assert "axon update" in result.hint

    def test_network_error_stays_quiet_ok(self, monkeypatch):
        from axon import update_check
        from axon.doctor import check_update_available

        monkeypatch.setattr(
            update_check,
            "check_for_update",
            lambda **kw: update_check.UpdateCheckResult(
                "0.4.4", None, False, skipped_reason="error"
            ),
        )
        result = check_update_available(offline=False)
        assert result.status == "ok"

    def test_included_in_check_funcs_and_run_doctor(self):
        from axon.doctor import _CHECK_FUNCS, check_update_available, run_doctor

        assert check_update_available in _CHECK_FUNCS
        report = run_doctor(None)
        names = [c.name for c in report.checks]
        assert "Update available" in names
