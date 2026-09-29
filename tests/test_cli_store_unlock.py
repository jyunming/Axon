"""One-shot sealed CLI commands prompt for the store passphrase."""

from __future__ import annotations

import getpass
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("keyring")
pytest.importorskip("cryptography")

from axon.cli import _ensure_store_unlocked  # noqa: E402
from axon.security import master as _master  # noqa: E402
from tests.test_project_seal import _InMemoryKeyring  # noqa: E402

PASSPHRASE = "correct horse battery staple"


@pytest.fixture
def store(tmp_path):
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        _master._unlocked_masters.clear()
        user_dir = tmp_path / "alice"
        user_dir.mkdir()
        _master.bootstrap_store(user_dir, PASSPHRASE)
        _master._unlocked_masters.clear()
        yield user_dir
        _master._unlocked_masters.clear()


def _tty(monkeypatch, value=True):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: value, raising=False)


def test_prompts_and_unlocks(store, monkeypatch):
    _tty(monkeypatch)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": PASSPHRASE)
    assert not _master.is_unlocked(store)
    _ensure_store_unlocked(store)
    assert _master.is_unlocked(store)


def test_wrong_passphrase_exits(store, monkeypatch, capsys):
    _tty(monkeypatch)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "not the passphrase")
    with pytest.raises(SystemExit) as exc:
        _ensure_store_unlocked(store)
    assert exc.value.code == 1
    assert "Unlock failed" in capsys.readouterr().out
    assert not _master.is_unlocked(store)


def test_no_prompt_without_terminal(store, monkeypatch):
    _tty(monkeypatch, False)

    def _boom(prompt=""):
        raise AssertionError("must not prompt without a terminal")

    monkeypatch.setattr(getpass, "getpass", _boom)
    _ensure_store_unlocked(store)
    assert not _master.is_unlocked(store)


def test_no_prompt_when_already_unlocked(store, monkeypatch):
    _master.unlock_store(store, PASSPHRASE)
    _tty(monkeypatch)

    def _boom(prompt=""):
        raise AssertionError("must not prompt when already unlocked")

    monkeypatch.setattr(getpass, "getpass", _boom)
    _ensure_store_unlocked(store)


def test_no_prompt_when_store_not_initialised(tmp_path, monkeypatch):
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        _tty(monkeypatch)

        def _boom(prompt=""):
            raise AssertionError("must not prompt for an uninitialised store")

        monkeypatch.setattr(getpass, "getpass", _boom)
        _ensure_store_unlocked(tmp_path)


class TestCliCallsUnlock:
    def test_project_seal_unlocks_first(self):
        from tests.test_cli_extra import run_cli

        order: list[str] = []
        with (
            patch("axon.cli._ensure_store_unlocked", side_effect=lambda d: order.append("unlock")),
            patch(
                "axon.security.project_seal",
                side_effect=lambda *a, **k: order.append("seal") or {"status": "sealed"},
            ),
        ):
            code = run_cli("--project-seal", "Research")
        assert code == 0
        assert order == ["unlock", "seal"]

    def test_sealed_hard_revoke_unlocks_first(self):
        from tests.test_cli_extra import run_cli

        order: list[str] = []
        with (
            patch("axon.cli._ensure_store_unlocked", side_effect=lambda d: order.append("unlock")),
            patch(
                "axon.security.revoke_sealed_share",
                side_effect=lambda **k: order.append("revoke") or {},
            ),
        ):
            code = run_cli(
                "--share-revoke", "ssk_abc", "--share-project", "research", "--share-rotate"
            )
        assert code == 0
        assert order == ["unlock", "revoke"]

    def test_sealed_soft_revoke_does_not_prompt(self):
        from tests.test_cli_extra import run_cli

        with (
            patch("axon.cli._ensure_store_unlocked") as unlock,
            patch("axon.security.revoke_sealed_share", return_value={}),
        ):
            code = run_cli("--share-revoke", "ssk_abc", "--share-project", "research")
        assert code == 0
        unlock.assert_not_called()


def test_ctrl_c_at_prompt_exits_cleanly(store, monkeypatch, capsys):
    _tty(monkeypatch)

    def _interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(getpass, "getpass", _interrupt)
    with pytest.raises(SystemExit) as exc:
        _ensure_store_unlocked(store)
    assert exc.value.code == 1
    assert "Cancelled" in capsys.readouterr().out
