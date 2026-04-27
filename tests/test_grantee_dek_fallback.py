"""Grantee DEK file fallback for headless Linux / Docker / CI.

Verifies that ``axon.security.share`` falls back to a file at
``<user_dir>/.security/shares/<key_id>.dek.wrapped`` when the OS keyring
is unavailable, mirroring the master.enc dual-write pattern.

Skips when the ``cryptography`` / ``keyring`` packages aren't installed.
"""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon.security import SecurityError  # noqa: E402
from axon.security import keyring as _kr  # noqa: E402
from axon.security.share import (  # noqa: E402
    _grantee_dek_fallback_path,
    _read_grantee_dek_fallback,
    _write_grantee_dek_fallback,
    delete_grantee_dek,
    get_grantee_dek,
    redeem_sealed_share,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_TEST_MASTER = b"\xab" * 32  # 32-byte fake master key
_TEST_DEK = b"\xcd" * 32  # 32-byte fake DEK
_KEY_ID = "testkey01"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_dir(tmp_path):
    ud = tmp_path / "bob"
    ud.mkdir()
    return ud


@pytest.fixture
def kr_unavailable():
    """Patch keyring so every call raises KeyringUnavailableError."""

    def _raise(*args, **kwargs):
        raise _kr.KeyringUnavailableError("no keyring backend (headless test)")

    with (
        patch.object(_kr, "store_secret", side_effect=_raise),
        patch.object(_kr, "get_secret", side_effect=_raise),
        patch.object(_kr, "delete_secret", side_effect=_raise),
        patch.object(_kr, "is_available", return_value=False),
    ):
        yield


@pytest.fixture
def master_unlocked(user_dir):
    """Patch get_master_key to return _TEST_MASTER without a real store."""
    with patch("axon.security.share.get_master_key", return_value=_TEST_MASTER):
        yield


# ---------------------------------------------------------------------------
# Unit tests for the private helpers
# ---------------------------------------------------------------------------


class TestFallbackPathHelper:
    def test_path_format(self, user_dir):
        p = _grantee_dek_fallback_path(user_dir, _KEY_ID)
        assert p.parent.name == "shares"
        assert p.parent.parent.name == ".security"
        assert p.name == f"{_KEY_ID}.dek.wrapped"


class TestWriteAndReadFallback:
    def test_write_then_read_roundtrip(self, user_dir, master_unlocked):
        _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)
        recovered = _read_grantee_dek_fallback(user_dir, _KEY_ID)
        assert recovered == _TEST_DEK

    def test_read_returns_none_when_absent(self, user_dir, master_unlocked):
        assert _read_grantee_dek_fallback(user_dir, _KEY_ID) is None

    def test_write_creates_parent_dirs(self, user_dir, master_unlocked):
        _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)
        assert _grantee_dek_fallback_path(user_dir, _KEY_ID).is_file()

    def test_write_is_atomic_via_tmp(self, user_dir, master_unlocked):
        """Verifies that a .tmp file is not left behind after a write."""
        _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)
        path = _grantee_dek_fallback_path(user_dir, _KEY_ID)
        tmp = path.with_suffix(path.suffix + ".tmp")
        assert not tmp.exists()

    def test_read_raises_security_error_on_corrupt_file(self, user_dir, master_unlocked):
        path = _grantee_dek_fallback_path(user_dir, _KEY_ID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-valid-aes-kw-data")
        with pytest.raises(SecurityError, match="unreadable"):
            _read_grantee_dek_fallback(user_dir, _KEY_ID)

    def test_store_locked_raises_security_error(self, user_dir):
        """If get_master_key raises (store locked), write should surface it."""
        with patch("axon.security.share.get_master_key", side_effect=SecurityError("locked")):
            with pytest.raises(SecurityError, match="locked"):
                _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)


# ---------------------------------------------------------------------------
# get_grantee_dek: file fallback path
# ---------------------------------------------------------------------------


class TestGetGranteeDek:
    def test_returns_dek_from_file_when_keyring_absent(
        self, user_dir, kr_unavailable, master_unlocked
    ):
        _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)
        dek = get_grantee_dek(_KEY_ID, user_dir=user_dir)
        assert dek == _TEST_DEK

    def test_raises_if_keyring_absent_and_no_user_dir(self, kr_unavailable):
        with pytest.raises(SecurityError, match="Pass user_dir"):
            get_grantee_dek(_KEY_ID)

    def test_raises_if_keyring_absent_and_file_missing(
        self, user_dir, kr_unavailable, master_unlocked
    ):
        with pytest.raises(SecurityError, match="not found in keyring or file fallback"):
            get_grantee_dek(_KEY_ID, user_dir=user_dir)

    def test_returns_dek_from_keyring_when_available(self, user_dir):
        """Keyring path is unaffected — returns base64-decoded secret."""
        encoded = _b64(_TEST_DEK)
        with patch.object(_kr, "get_secret", return_value=encoded):
            dek = get_grantee_dek(_KEY_ID, user_dir=user_dir)
        assert dek == _TEST_DEK

    def test_raises_on_missing_keyring_entry(self, user_dir):
        with patch.object(_kr, "get_secret", return_value=None):
            with pytest.raises(SecurityError, match="No sealed-share DEK"):
                get_grantee_dek(_KEY_ID, user_dir=user_dir)


# ---------------------------------------------------------------------------
# delete_grantee_dek: file cleanup
# ---------------------------------------------------------------------------


class TestDeleteGranteeDek:
    def test_delete_removes_fallback_file(self, user_dir, master_unlocked):
        _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)
        path = _grantee_dek_fallback_path(user_dir, _KEY_ID)
        assert path.is_file()
        with patch.object(_kr, "get_secret", return_value=None):
            result = delete_grantee_dek(_KEY_ID, user_dir=user_dir)
        assert result is True
        assert not path.is_file()

    def test_delete_returns_false_when_nothing_present(self, user_dir):
        with patch.object(_kr, "get_secret", return_value=None):
            result = delete_grantee_dek(_KEY_ID, user_dir=user_dir)
        assert result is False

    def test_delete_without_user_dir_does_not_raise(self):
        """Caller may omit user_dir; keyring-only cleanup should still work."""
        with patch.object(_kr, "get_secret", return_value=_b64(_TEST_DEK)):
            with patch.object(_kr, "delete_secret"):
                result = delete_grantee_dek(_KEY_ID)
        assert result is True

    def test_delete_cleans_file_even_when_keyring_unavailable(
        self, user_dir, kr_unavailable, master_unlocked
    ):
        _write_grantee_dek_fallback(user_dir, _KEY_ID, _TEST_DEK)
        path = _grantee_dek_fallback_path(user_dir, _KEY_ID)
        assert path.is_file()
        result = delete_grantee_dek(_KEY_ID, user_dir=user_dir)
        assert result is True
        assert not path.is_file()


# ---------------------------------------------------------------------------
# redeem_sealed_share: file fallback integration
# ---------------------------------------------------------------------------


def _make_share_string(
    owner_user_dir: Path,
    project_dir: Path,
    key_id: str,
) -> str:
    """Build a minimal sealed share_string pointing at a pre-created wrap file."""
    import secrets as _secrets

    from cryptography.hazmat.primitives.keywrap import aes_key_wrap

    from axon.security.share import (
        SEALED_SHARE_PREFIX,
        _derive_share_kek,
    )

    token_bytes = _secrets.token_bytes(32)
    kek = _derive_share_kek(token_bytes, key_id)
    wrapped_dek = aes_key_wrap(kek, _TEST_DEK)
    wrap_path = project_dir / ".security" / "shares" / f"{key_id}.wrapped"
    wrap_path.parent.mkdir(parents=True, exist_ok=True)
    wrap_path.write_bytes(wrapped_dek)

    owner_name = owner_user_dir.name
    owner_store_path = str(owner_user_dir.parent)
    raw = f"{SEALED_SHARE_PREFIX}:{key_id}:{token_bytes.hex()}:{owner_name}:myproject:{owner_store_path}"
    return base64.urlsafe_b64encode(raw.encode()).decode("ascii")


class TestRedeemWritesFallback:
    def test_redeem_writes_dek_to_file_when_keyring_unavailable(
        self, user_dir, kr_unavailable, master_unlocked
    ):
        # owner lives alongside grantee (user_dir = bob) in the same tmp_path
        owner_dir = user_dir.parent / "owner"
        owner_dir.mkdir()
        project_dir = owner_dir / "myproject"
        project_dir.mkdir(parents=True)
        grantee_dir = user_dir

        share_string = _make_share_string(owner_dir, project_dir, _KEY_ID)

        # Mock mounts module so the descriptor write doesn't need full store.
        mock_descriptor = {
            "mount_name": "owner_myproject",
            "mount_type": "sealed",
            "state": "active",
        }
        with patch(
            "axon.security.share._create_sealed_mount_descriptor",
            return_value=mock_descriptor,
        ):
            result = redeem_sealed_share(grantee_dir, share_string)

        assert result["key_id"] == _KEY_ID
        fb_path = _grantee_dek_fallback_path(grantee_dir, _KEY_ID)
        assert fb_path.is_file(), "Fallback file should be created when keyring unavailable"

    def test_redeem_raises_if_store_not_bootstrapped_and_keyring_unavailable(
        self, user_dir, kr_unavailable
    ):
        """If the grantee's master store is locked/not-bootstrapped, a clear
        SecurityError should surface (not a silent failure)."""
        owner_dir = user_dir.parent / "owner"
        owner_dir.mkdir()
        project_dir = owner_dir / "myproject"
        project_dir.mkdir(parents=True)
        grantee_dir = user_dir

        share_string = _make_share_string(owner_dir, project_dir, _KEY_ID)

        with (
            patch("axon.security.share.get_master_key", side_effect=SecurityError("locked")),
            patch("axon.security.share._create_sealed_mount_descriptor", return_value={}),
        ):
            with pytest.raises(SecurityError, match="locked"):
                redeem_sealed_share(grantee_dir, share_string)

    def test_redeem_dual_writes_when_keyring_available(self, user_dir, master_unlocked):
        """When keyring IS available, file fallback should ALSO be written."""
        owner_dir = user_dir.parent / "owner"
        owner_dir.mkdir()
        project_dir = owner_dir / "myproject"
        project_dir.mkdir(parents=True)
        grantee_dir = user_dir

        share_string = _make_share_string(owner_dir, project_dir, _KEY_ID)

        with (
            patch.object(_kr, "store_secret"),  # keyring succeeds
            patch("axon.security.share._create_sealed_mount_descriptor", return_value={}),
        ):
            redeem_sealed_share(grantee_dir, share_string)

        fb_path = _grantee_dek_fallback_path(grantee_dir, _KEY_ID)
        assert fb_path.is_file(), "File fallback should be written even when keyring succeeds"


# ---------------------------------------------------------------------------
# Full headless round-trip
# ---------------------------------------------------------------------------


class TestHeadlessRoundTrip:
    def test_dek_round_trip_headless(self, user_dir, kr_unavailable, master_unlocked):
        """Redeem with keyring absent → get_grantee_dek with keyring absent."""
        owner_dir = user_dir.parent / "owner"
        owner_dir.mkdir()
        project_dir = owner_dir / "myproject"
        project_dir.mkdir(parents=True)
        grantee_dir = user_dir

        share_string = _make_share_string(owner_dir, project_dir, _KEY_ID)

        with patch("axon.security.share._create_sealed_mount_descriptor", return_value={}):
            redeem_sealed_share(grantee_dir, share_string)

        # DEK should be retrievable via file fallback.
        recovered_dek = get_grantee_dek(_KEY_ID, user_dir=grantee_dir)
        assert recovered_dek == _TEST_DEK
