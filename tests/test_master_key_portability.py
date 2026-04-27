"""Cross-platform sealed-mount key portability tests.

Verifies that master.enc is always written alongside the OS keyring so an
owner can copy the file to another machine / OS and unlock with the same
passphrase even when the destination keyring has no entry.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

import axon.security.master as _master_mod  # noqa: E402
from axon.security import SecurityError  # noqa: E402
from axon.security.master import (  # noqa: E402
    BadPassphraseError,
    bootstrap_store,
    change_passphrase,
    get_or_create_project_dek,
    is_bootstrapped,
    unlock_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASSPHRASE = "test-portability-passphrase"
_NEW_PASSPHRASE = "new-portability-passphrase"


@pytest.fixture(autouse=True)
def _clear_master_cache():
    """Prevent _unlocked_masters leaking across tests (cache is module-global)."""
    _master_mod._unlocked_masters.clear()
    yield
    _master_mod._unlocked_masters.clear()


def _fallback_path(user_dir: Path) -> Path:
    return user_dir / ".security" / "master.enc"


def _mock_keyring_available():
    """Patch that makes the keyring appear fully functional."""
    _store: dict[tuple[str, str], str] = {}

    def _store_secret(service: str, username: str, secret: str) -> None:
        _store[(service, username)] = secret

    def _get_secret(service: str, username: str) -> str | None:
        return _store.get((service, username))

    return patch.multiple(
        "axon.security.keyring",
        store_secret=_store_secret,
        get_secret=_get_secret,
    )


def _mock_keyring_absent():
    """Patch that makes the keyring raise KeyringUnavailableError on every call."""
    from axon.security.keyring import KeyringUnavailableError

    def _raise(*a, **kw):
        raise KeyringUnavailableError("mocked: no keyring")

    return patch.multiple(
        "axon.security.keyring",
        store_secret=_raise,
        get_secret=_raise,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDualWrite:
    def test_bootstrap_always_writes_fallback_file(self, tmp_path):
        """master.enc is written even when the OS keyring is available."""
        with _mock_keyring_available():
            bootstrap_store(tmp_path, _PASSPHRASE)

        assert _fallback_path(tmp_path).is_file(), "master.enc must exist after bootstrap"

    def test_fallback_file_is_valid_json(self, tmp_path):
        """The fallback file contains a well-formed JSON record."""
        with _mock_keyring_available():
            bootstrap_store(tmp_path, _PASSPHRASE)

        record = json.loads(_fallback_path(tmp_path).read_text(encoding="utf-8"))
        assert record["v"] == 1
        assert "salt" in record
        assert "wrapped" in record

    def test_bootstrap_idempotent_raises_if_already_bootstrapped(self, tmp_path):
        """Calling bootstrap_store twice raises, same as before the fix."""
        with _mock_keyring_available():
            bootstrap_store(tmp_path, _PASSPHRASE)
            with pytest.raises(SecurityError):
                bootstrap_store(tmp_path, _PASSPHRASE)


class TestFallbackUnlock:
    def test_fallback_file_unlocks_when_keyring_missing(self, tmp_path):
        """Bootstrap with keyring available; unlock with keyring absent — must succeed via file."""
        with _mock_keyring_available():
            bootstrap_store(tmp_path, _PASSPHRASE)
            assert is_bootstrapped(tmp_path)

        # Simulate "different machine" — keyring has no entry
        with _mock_keyring_absent():
            unlock_store(tmp_path, _PASSPHRASE)
            # If we reach here the unlock succeeded via master.enc
            assert True

    def test_wrong_passphrase_fails_via_file(self, tmp_path):
        """Wrong passphrase still raises even when reading from the fallback file."""
        with _mock_keyring_available():
            bootstrap_store(tmp_path, _PASSPHRASE)

        with _mock_keyring_absent():
            with pytest.raises(Exception, match="[Pp]assphrase|[Uu]nwrap|[Ii]nvalid"):
                unlock_store(tmp_path, "wrong-passphrase")


class TestChangePassphrase:
    def test_change_passphrase_updates_fallback_file(self, tmp_path):
        """After change_passphrase(), the new passphrase works via the fallback file."""
        with _mock_keyring_available():
            bootstrap_store(tmp_path, _PASSPHRASE)
            old_content = _fallback_path(tmp_path).read_bytes()

            change_passphrase(tmp_path, _PASSPHRASE, _NEW_PASSPHRASE)
            new_content = _fallback_path(tmp_path).read_bytes()

        assert old_content != new_content, "File content must change after passphrase rotation"

        with _mock_keyring_absent():
            # Old passphrase must fail
            with pytest.raises(BadPassphraseError):
                unlock_store(tmp_path, _PASSPHRASE)

        with _mock_keyring_absent():
            # New passphrase must succeed
            unlock_store(tmp_path, _NEW_PASSPHRASE)


class TestCrossPlatformCopySimulation:
    def test_copy_master_enc_to_new_user_dir(self, tmp_path):
        """Simulate copying master.enc from 'source machine' to 'destination machine'.

        Bootstrap in src_dir (keyring available).
        Copy only master.enc to dst_dir (simulating a new OS install with no keyring entry).
        Unlock from dst_dir with keyring absent — must succeed.
        """
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        with _mock_keyring_available():
            bootstrap_store(src_dir, _PASSPHRASE)

        # Copy the portable fallback file to the destination
        src_security = _fallback_path(src_dir)
        dst_security = dst_dir / ".security"
        dst_security.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_security, dst_security / "master.enc")

        # On the destination OS the keyring has no entry — reads from file
        with _mock_keyring_absent():
            unlock_store(dst_dir, _PASSPHRASE)

    def test_dek_accessible_after_cross_platform_copy(self, tmp_path):
        """After copying master.enc, the project DEK can be derived on the new OS."""
        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        project_dir = tmp_path / "project"
        src_dir.mkdir()
        dst_dir.mkdir()
        project_dir.mkdir()

        with _mock_keyring_available():
            bootstrap_store(src_dir, _PASSPHRASE)
            unlock_store(src_dir, _PASSPHRASE)
            original_dek = get_or_create_project_dek(src_dir, project_dir)

        # Copy master.enc and dek.wrapped to destination
        src_security = _fallback_path(src_dir)
        dst_security = dst_dir / ".security"
        dst_security.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_security, dst_security / "master.enc")

        # Also copy the project's dek.wrapped (this would be synced via OneDrive)
        proj_security = project_dir / ".security"
        dst_proj_security = tmp_path / "dst_project" / ".security"
        dst_proj_security.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proj_security / "dek.wrapped", dst_proj_security / "dek.wrapped")

        dst_project_dir = tmp_path / "dst_project"

        with _mock_keyring_absent():
            unlock_store(dst_dir, _PASSPHRASE)
            recovered_dek = get_or_create_project_dek(dst_dir, dst_project_dir)

        assert recovered_dek == original_dek, "DEK must be identical after cross-platform copy"
