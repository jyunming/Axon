"""Phase 6: passphrase-fallback master store for headless environments.

Verifies that ``axon.security.master`` falls back to a file at
``<user_dir>/.security/master.enc`` when the OS keyring is unavailable,
and that the SAME passphrase still unlocks the master across keyring
and file storage.

Skips when the ``cryptography`` / ``keyring`` packages aren't installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon.security import SecurityError  # noqa: E402
from axon.security import fallback_store as _fb  # noqa: E402
from axon.security import keyring as _kr  # noqa: E402
from axon.security.master import (  # noqa: E402
    BadPassphraseError,
    bootstrap_store,
    change_passphrase,
    is_bootstrapped,
    is_unlocked,
    lock_store,
    unlock_store,
)

# ---------------------------------------------------------------------------
# Fixtures: in-memory keyring (control case) + keyring-unavailable (fallback)
# ---------------------------------------------------------------------------


class _InMemoryKeyring:
    priority = 1

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, secret):
        self._store[(service, username)] = secret

    def get_password(self, service, username):
        return self._store.get((service, username))

    def delete_password(self, service, username):
        import keyring.errors

        if (service, username) not in self._store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


@pytest.fixture
def kr_unavailable():
    """Patch the keyring module so every call raises KeyringUnavailableError."""

    def _raise(*args, **kwargs):
        raise _kr.KeyringUnavailableError("no keyring backend (headless test)")

    with patch.object(_kr, "store_secret", side_effect=_raise), patch.object(
        _kr, "get_secret", side_effect=_raise
    ), patch.object(_kr, "delete_secret", side_effect=_raise), patch.object(
        _kr, "is_available", return_value=False
    ):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield
        _master_mod._unlocked_masters.clear()


@pytest.fixture
def kr_available():
    """Healthy in-memory keyring fixture for control comparisons."""
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


@pytest.fixture
def user_dir(tmp_path):
    ud = tmp_path / "alice"
    ud.mkdir()
    return ud


# ---------------------------------------------------------------------------
# fallback_store unit tests
# ---------------------------------------------------------------------------


class TestFallbackStorePrimitives:
    def test_path_lives_under_security_subdir(self, user_dir):
        path = _fb.fallback_master_path(user_dir)
        assert path.parent.name == ".security"
        assert path.name == "master.enc"

    def test_is_present_false_when_absent(self, user_dir):
        assert _fb.is_present(user_dir) is False

    def test_write_then_read_roundtrip(self, user_dir):
        payload = json.dumps({"v": 1, "salt": "AAAA", "wrapped": "BBBB"})
        _fb.write_master_record(user_dir, payload)
        assert _fb.is_present(user_dir) is True
        assert _fb.read_master_record(user_dir) == payload

    def test_read_returns_none_when_absent(self, user_dir):
        assert _fb.read_master_record(user_dir) is None

    def test_write_rejects_invalid_json(self, user_dir):
        with pytest.raises(ValueError, match="valid JSON"):
            _fb.write_master_record(user_dir, "not-json")

    def test_write_rejects_empty(self, user_dir):
        with pytest.raises(ValueError, match="non-empty"):
            _fb.write_master_record(user_dir, "")

    def test_delete_returns_true_when_present(self, user_dir):
        _fb.write_master_record(user_dir, '{"v": 1}')
        assert _fb.delete_master_record(user_dir) is True
        assert _fb.is_present(user_dir) is False

    def test_delete_returns_false_when_absent(self, user_dir):
        assert _fb.delete_master_record(user_dir) is False


# ---------------------------------------------------------------------------
# Bootstrap → unlock round-trip with keyring unavailable
# ---------------------------------------------------------------------------


class TestBootstrapUnlockOnHeadless:
    def test_bootstrap_writes_to_fallback_file(self, kr_unavailable, user_dir):
        result = bootstrap_store(user_dir, "headless-pp")
        assert result["initialized"] is True
        # File exists on disk; keyring was untouched.
        assert _fb.is_present(user_dir) is True

    def test_bootstrap_caches_unlock_immediately(self, kr_unavailable, user_dir):
        bootstrap_store(user_dir, "headless-pp")
        assert is_unlocked(user_dir) is True

    def test_unlock_after_lock_uses_fallback_file(self, kr_unavailable, user_dir):
        bootstrap_store(user_dir, "headless-pp")
        lock_store(user_dir)
        assert is_unlocked(user_dir) is False
        unlock_store(user_dir, "headless-pp")
        assert is_unlocked(user_dir) is True

    def test_unlock_with_wrong_passphrase_raises(self, kr_unavailable, user_dir):
        bootstrap_store(user_dir, "headless-pp")
        lock_store(user_dir)
        with pytest.raises(BadPassphraseError):
            unlock_store(user_dir, "wrong-pp")

    def test_change_passphrase_rewrites_fallback(self, kr_unavailable, user_dir):
        bootstrap_store(user_dir, "headless-pp")
        change_passphrase(user_dir, "headless-pp", "new-headless-pp")
        # File still present, but new passphrase unlocks.
        assert _fb.is_present(user_dir) is True
        lock_store(user_dir)
        unlock_store(user_dir, "new-headless-pp")
        assert is_unlocked(user_dir) is True
        # Old passphrase no longer works.
        lock_store(user_dir)
        with pytest.raises(BadPassphraseError):
            unlock_store(user_dir, "headless-pp")

    def test_is_bootstrapped_detects_fallback_file(self, kr_unavailable, user_dir):
        assert is_bootstrapped(user_dir) is False
        bootstrap_store(user_dir, "headless-pp")
        assert is_bootstrapped(user_dir) is True


# ---------------------------------------------------------------------------
# File-fallback content matches the keyring-backed format
# ---------------------------------------------------------------------------


class TestFallbackFormatParity:
    def test_fallback_record_has_keyring_shape(self, kr_unavailable, user_dir):
        bootstrap_store(user_dir, "headless-pp")
        raw = _fb.read_master_record(user_dir)
        record = json.loads(raw)
        assert set(record.keys()) >= {"v", "salt", "wrapped"}
        assert record["v"] == 1
        # Wrapped master is exactly 40 bytes (AES-KW of 32-byte master).
        # Decoded length: AES-KW output = input + 8 = 40.
        import base64

        assert len(base64.b64decode(record["wrapped"])) == 40
        # Salt is 32 bytes per scrypt config.
        assert len(base64.b64decode(record["salt"])) == 32

    def test_corrupted_fallback_raises_security_error(self, kr_unavailable, user_dir):
        _fb.write_master_record(user_dir, '{"v": 999, "salt": "AAAA", "wrapped": "BBBB"}')
        with pytest.raises(SecurityError, match="schema_version mismatch"):
            unlock_store(user_dir, "anything")


# ---------------------------------------------------------------------------
# Keyring-available path is unchanged (regression coverage)
# ---------------------------------------------------------------------------


class TestKeyringStillPreferredWhenAvailable:
    def test_bootstrap_with_keyring_also_writes_fallback_file(self, kr_available, user_dir):
        bootstrap_store(user_dir, "with-keyring-pp")
        # Master goes to keyring AND the portable fallback file for cross-platform use.
        assert _fb.is_present(user_dir) is True


# ---------------------------------------------------------------------------
# Defensive error handling — file unreadable, error sourcing
# ---------------------------------------------------------------------------


class TestFallbackErrorSurfacing:
    def test_unreadable_file_raises_security_error_not_silent_none(self, kr_unavailable, user_dir):
        """If the fallback file exists but read_text raises OSError,
        master must surface as SecurityError — NOT return None which
        would let the user think 'not bootstrapped' and re-bootstrap
        (overwriting the existing wrap)."""
        # Bootstrap so the file exists.
        bootstrap_store(user_dir, "headless-pp")
        lock_store(user_dir)
        # Patch read_text on the fallback file to raise.
        original_read_text = Path.read_text

        def boom_on_master(self, *args, **kwargs):
            if self.name == "master.enc":
                raise PermissionError("simulated EACCES")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", boom_on_master):
            with pytest.raises(SecurityError, match="unreadable"):
                unlock_store(user_dir, "headless-pp")

    def test_corrupted_fallback_error_message_cites_file_not_keyring(
        self, kr_unavailable, user_dir
    ):
        """When the fallback file holds bad JSON, the SecurityError
        message must point at the FILE path (not the keyring service
        name) so headless-server users can find what to fix."""
        # Write garbage directly to the fallback path.
        _fb.write_master_record(user_dir, '{"not": "the right shape"}')
        with pytest.raises(SecurityError) as excinfo:
            unlock_store(user_dir, "anything")
        msg = str(excinfo.value)
        assert "master.enc" in msg or "fallback file" in msg
        # Should NOT be quoting only the keyring service name as the source.
        assert "keyring service" not in msg

    def test_corrupted_keyring_error_message_cites_keyring(self, kr_available, user_dir):
        """Symmetric: when the bytes came from the keyring, the
        message points at the keyring service name."""
        bootstrap_store(user_dir, "with-keyring-pp")
        # Stomp the keyring entry with garbage.
        kr_available.set_password(f"axon.master.{user_dir.name}", "master", '{"not": "right"}')
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        with pytest.raises(SecurityError) as excinfo:
            unlock_store(user_dir, "with-keyring-pp")
        msg = str(excinfo.value)
        assert "keyring service" in msg
