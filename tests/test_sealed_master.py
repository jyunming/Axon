"""Phase 2: master-key + project-DEK lifecycle tests.

Covers ``axon.security.master`` end-to-end via a mocked keyring backend
(no touching the developer's real OS keyring during the test run).

Skips when the ``cryptography`` or ``keyring`` packages are missing
(the ``sealed`` extra hasn't been installed).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon.security import SecurityError  # noqa: E402
from axon.security.master import (  # noqa: E402
    DEK_LEN,
    MASTER_USERNAME,
    BadPassphraseError,
    bootstrap_store,
    change_passphrase,
    get_master_key,
    get_or_create_project_dek,
    get_project_dek,
    is_bootstrapped,
    is_unlocked,
    lock_store,
    unlock_store,
)

# ---------------------------------------------------------------------------
# In-memory keyring backend (used by every test — never touches the OS keyring)
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
def kr_backend():
    """Each test gets a fresh in-memory keyring + a clean unlock cache."""
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        # Also clear any cached unlocked masters from prior tests.
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


@pytest.fixture
def user_dir(tmp_path):
    """A tmp_path-rooted "AxonStore user dir" with a stable basename."""
    ud = tmp_path / "alice"
    ud.mkdir()
    return ud


# ---------------------------------------------------------------------------
# bootstrap_store
# ---------------------------------------------------------------------------


class TestBootstrapStore:
    def test_first_bootstrap_marks_initialized(self, kr_backend, user_dir):
        result = bootstrap_store(user_dir, "correct-horse-battery-staple")
        assert result["initialized"] is True
        assert result["owner"] == "alice"
        assert is_bootstrapped(user_dir)

    def test_bootstrap_caches_master_so_user_is_unlocked_immediately(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        assert is_unlocked(user_dir)

    def test_bootstrap_again_raises_security_error(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        with pytest.raises(SecurityError, match="already bootstrapped"):
            bootstrap_store(user_dir, "passw0rd-2")

    def test_empty_passphrase_rejected(self, kr_backend, user_dir):
        with pytest.raises(BadPassphraseError, match="non-empty"):
            bootstrap_store(user_dir, "")

    def test_bootstrap_writes_keyring_record_with_schema_version(self, kr_backend, user_dir):
        import json

        bootstrap_store(user_dir, "passw0rd")
        record = json.loads(kr_backend.get_password("axon.master.alice", MASTER_USERNAME))
        assert record["v"] == 1
        assert "salt" in record and "wrapped" in record


# ---------------------------------------------------------------------------
# unlock_store / lock_store / is_unlocked
# ---------------------------------------------------------------------------


class TestUnlockLock:
    def test_unlock_after_bootstrap_with_correct_passphrase(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        lock_store(user_dir)
        assert not is_unlocked(user_dir)

        result = unlock_store(user_dir, "passw0rd")
        assert result["unlocked"] is True
        assert is_unlocked(user_dir)

    def test_unlock_with_wrong_passphrase_raises(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "right-pw!")
        lock_store(user_dir)
        with pytest.raises(BadPassphraseError, match="Wrong passphrase"):
            unlock_store(user_dir, "wrong-pw!")
        assert not is_unlocked(user_dir)

    def test_unlock_before_bootstrap_raises(self, kr_backend, user_dir):
        with pytest.raises(SecurityError, match="not bootstrapped"):
            unlock_store(user_dir, "passw0rd")

    def test_lock_clears_master_cache(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        assert is_unlocked(user_dir)
        result = lock_store(user_dir)
        assert result["locked"] is True
        assert result["was_unlocked"] is True
        assert not is_unlocked(user_dir)

    def test_lock_idempotent_when_not_unlocked(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        lock_store(user_dir)
        result = lock_store(user_dir)  # again
        assert result["locked"] is True
        assert result["was_unlocked"] is False


# ---------------------------------------------------------------------------
# change_passphrase
# ---------------------------------------------------------------------------


class TestChangePassphrase:
    def test_rotate_with_correct_old_passphrase(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "old-passw")
        master_before = get_master_key(user_dir)
        result = change_passphrase(user_dir, "old-passw", "new-passw")
        assert result["rotated"] is True
        # Master is UNCHANGED across passphrase rotation.
        assert get_master_key(user_dir) == master_before

    def test_rotate_with_wrong_old_raises(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "old-passw")
        with pytest.raises(BadPassphraseError, match="Old passphrase"):
            change_passphrase(user_dir, "wrong-old", "new-passw")

    def test_after_rotate_old_passphrase_no_longer_works(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "old-passw")
        change_passphrase(user_dir, "old-passw", "new-passw")
        lock_store(user_dir)
        with pytest.raises(BadPassphraseError):
            unlock_store(user_dir, "old-passw")
        # New one works.
        unlock_store(user_dir, "new-passw")
        assert is_unlocked(user_dir)

    def test_empty_new_passphrase_rejected(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "old-passw")
        with pytest.raises(BadPassphraseError):
            change_passphrase(user_dir, "old-passw", "")


# ---------------------------------------------------------------------------
# get_master_key
# ---------------------------------------------------------------------------


class TestGetMasterKey:
    def test_returns_32_bytes_when_unlocked(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        master = get_master_key(user_dir)
        assert isinstance(master, bytes)
        assert len(master) == 32

    def test_raises_when_locked(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        lock_store(user_dir)
        with pytest.raises(SecurityError, match="locked"):
            get_master_key(user_dir)

    def test_master_stable_across_lock_unlock(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        m1 = get_master_key(user_dir)
        lock_store(user_dir)
        unlock_store(user_dir, "passw0rd")
        m2 = get_master_key(user_dir)
        assert m1 == m2


# ---------------------------------------------------------------------------
# get_or_create_project_dek
# ---------------------------------------------------------------------------


class TestProjectDek:
    def test_first_call_creates_and_persists_wrapped_dek(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        dek = get_or_create_project_dek(user_dir, proj)
        assert isinstance(dek, bytes)
        assert len(dek) == DEK_LEN
        # On disk: 40 bytes (AES-KW wraps 32 → 40).
        wrap_path = proj / ".security" / "dek.wrapped"
        assert wrap_path.is_file()
        assert wrap_path.stat().st_size == 40

    def test_idempotent_returns_same_dek(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        d1 = get_or_create_project_dek(user_dir, proj)
        d2 = get_or_create_project_dek(user_dir, proj)
        assert d1 == d2

    def test_distinct_projects_get_distinct_deks(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        p1 = user_dir / "p1"
        p2 = user_dir / "p2"
        p1.mkdir()
        p2.mkdir()
        d1 = get_or_create_project_dek(user_dir, p1)
        d2 = get_or_create_project_dek(user_dir, p2)
        assert d1 != d2

    def test_locked_store_blocks_dek_access(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        lock_store(user_dir)
        proj = user_dir / "research"
        proj.mkdir()
        with pytest.raises(SecurityError, match="locked"):
            get_or_create_project_dek(user_dir, proj)

    def test_dek_survives_passphrase_rotation(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "old-passw")
        proj = user_dir / "research"
        proj.mkdir()
        dek_before = get_or_create_project_dek(user_dir, proj)
        change_passphrase(user_dir, "old-passw", "new-passw")
        # DEK should be retrievable AND identical — passphrase rotation
        # doesn't touch project DEKs.
        dek_after = get_or_create_project_dek(user_dir, proj)
        assert dek_before == dek_after

    def test_dek_survives_lock_unlock_cycle(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        dek1 = get_or_create_project_dek(user_dir, proj)
        lock_store(user_dir)
        unlock_store(user_dir, "passw0rd")
        dek2 = get_or_create_project_dek(user_dir, proj)
        assert dek1 == dek2

    def test_no_partial_wrap_left_after_failed_write(self, kr_backend, user_dir):
        """Atomic per-file write — the .sealing tempfile must not survive."""
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        get_or_create_project_dek(user_dir, proj)
        sec_dir = proj / ".security"
        leftover = list(sec_dir.glob("*.sealing"))
        assert leftover == []

    def test_corrupt_wrapped_dek_raises_security_error(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        get_or_create_project_dek(user_dir, proj)
        # Tamper.
        wrap_path = proj / ".security" / "dek.wrapped"
        wrap_path.write_bytes(b"\xff" * 40)
        with pytest.raises(SecurityError, match="won't unwrap"):
            get_or_create_project_dek(user_dir, proj)


# ---------------------------------------------------------------------------
# get_project_dek (read-only variant)
# ---------------------------------------------------------------------------


class TestGetProjectDekReadOnly:
    def test_raises_when_no_dek_file(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        # No DEK created yet.
        with pytest.raises(SecurityError, match="missing"):
            get_project_dek(user_dir, proj)

    def test_returns_existing_dek_unchanged(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "passw0rd")
        proj = user_dir / "research"
        proj.mkdir()
        original = get_or_create_project_dek(user_dir, proj)
        recovered = get_project_dek(user_dir, proj)
        assert original == recovered


# ---------------------------------------------------------------------------
# Keyring record validation
# ---------------------------------------------------------------------------


class TestKeyringRecordValidation:
    def test_malformed_json_raises_security_error(self, kr_backend, user_dir):
        kr_backend.set_password("axon.master.alice", MASTER_USERNAME, "{not json")
        with pytest.raises(SecurityError, match="unparseable JSON"):
            unlock_store(user_dir, "pw")

    def test_wrong_schema_version_raises_security_error(self, kr_backend, user_dir):
        import json

        kr_backend.set_password(
            "axon.master.alice",
            MASTER_USERNAME,
            json.dumps({"v": 99, "salt": "x", "wrapped": "x"}),
        )
        with pytest.raises(SecurityError, match="schema_version mismatch"):
            unlock_store(user_dir, "pw")


# ---------------------------------------------------------------------------
# Passphrase minimum-length enforcement
# ---------------------------------------------------------------------------


class TestPassphraseMinimumLength:
    """bootstrap_store and change_passphrase enforce a minimum passphrase
    length; unlock_store deliberately does NOT so it returns a clear
    "wrong passphrase" error rather than a misleading "too short" one.
    """

    def test_bootstrap_rejects_short_passphrase(self, kr_backend, user_dir):
        """A passphrase shorter than 8 characters is rejected at bootstrap."""
        with pytest.raises(BadPassphraseError, match="at least 8"):
            bootstrap_store(user_dir, "short")

    def test_bootstrap_rejects_exactly_seven_chars(self, kr_backend, user_dir):
        with pytest.raises(BadPassphraseError, match="at least 8"):
            bootstrap_store(user_dir, "1234567")

    def test_bootstrap_accepts_exactly_eight_chars(self, kr_backend, user_dir):
        result = bootstrap_store(user_dir, "12345678")
        assert result["initialized"] is True

    def test_change_passphrase_rejects_short_new_passphrase(self, kr_backend, user_dir):
        """change_passphrase rejects a new passphrase that is too short."""
        bootstrap_store(user_dir, "long-enough-old")
        with pytest.raises(BadPassphraseError, match="at least 8"):
            change_passphrase(user_dir, "long-enough-old", "short")

    def test_change_passphrase_accepts_eight_char_new_passphrase(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "long-enough-old")
        result = change_passphrase(user_dir, "long-enough-old", "new12345")
        assert result["rotated"] is True

    def test_unlock_accepts_short_passphrase_and_raises_bad_passphrase(self, kr_backend, user_dir):
        """unlock_store does NOT enforce minimum length — it lets the
        wrong-passphrase path fire normally so the error message is clear.
        Passing a short passphrase should raise BadPassphraseError (wrong
        passphrase), NOT BadPassphraseError (too short).
        """
        bootstrap_store(user_dir, "correct-long-pass")
        lock_store(user_dir)
        # A short passphrase is just wrong — no "too short" error raised.
        with pytest.raises(BadPassphraseError, match="Wrong passphrase"):
            unlock_store(user_dir, "tiny")
