"""Phase 1 of the sealed-mount design: keyring helper tests.

Covers ``axon.security.keyring``:
- service-name conventions
- store / get / delete round-trip with a mocked backend
- KeyringUnavailableError when the active backend is the no-op fail
  stub
- delete_secret silently no-ops on "not found"

Skips the entire module when the optional ``keyring`` package isn't
installed (the ``sealed`` extra hasn't been pulled in).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("keyring")

from axon.security.keyring import (  # noqa: E402
    MASTER_SERVICE_PREFIX,
    SHARE_SERVICE_PREFIX,
    KeyringUnavailableError,
    delete_secret,
    get_secret,
    is_available,
    master_service,
    share_service,
    store_secret,
)

# ---------------------------------------------------------------------------
# Fake in-memory backend used to exercise store/get/delete without
# touching the real OS keyring (which would mutate the test machine).
# ---------------------------------------------------------------------------


class _InMemoryKeyring:
    """Minimal keyring backend for tests — exposes the same API surface
    that ``axon.security.keyring`` uses (``set_password`` /
    ``get_password`` / ``delete_password``)."""

    priority = 1  # required by the keyring backend protocol

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, secret: str) -> None:
        self._store[(service, username)] = secret

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        import keyring.errors

        if (service, username) not in self._store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


# ---------------------------------------------------------------------------
# Service-name conventions
# ---------------------------------------------------------------------------


class TestServiceNames:
    def test_master_service_uses_prefix(self):
        s = master_service("alice")
        assert s.startswith(MASTER_SERVICE_PREFIX)
        assert s == "axon.master.alice"

    def test_share_service_uses_prefix(self):
        s = share_service("sk_a1b2c3d4")
        assert s.startswith(SHARE_SERVICE_PREFIX)
        assert s == "axon.share.sk_a1b2c3d4"

    def test_master_service_rejects_empty_owner(self):
        with pytest.raises(ValueError):
            master_service("")

    def test_share_service_rejects_empty_key_id(self):
        with pytest.raises(ValueError):
            share_service("")


# ---------------------------------------------------------------------------
# Store / get / delete round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """All wrappers route through _active_backend() → backend.{set,get,delete}_password,
    so patching get_keyring alone is enough to substitute the in-memory backend."""

    def test_store_then_get_returns_secret(self):
        backend = _InMemoryKeyring()
        with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
            store_secret("axon.master.alice", "default", "s3cret")
            assert get_secret("axon.master.alice", "default") == "s3cret"

    def test_get_returns_none_for_missing(self):
        backend = _InMemoryKeyring()
        with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
            assert get_secret("axon.master.alice", "default") is None

    def test_delete_then_get_returns_none(self):
        backend = _InMemoryKeyring()
        with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
            store_secret("svc", "u", "v")
            delete_secret("svc", "u")
            assert get_secret("svc", "u") is None

    def test_delete_missing_key_is_silent(self):
        """delete_secret on a key that doesn't exist must NOT raise —
        the desired post-condition is "secret is absent", which is
        already satisfied. Callers should be able to call delete
        defensively."""
        backend = _InMemoryKeyring()
        with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
            # Should NOT raise.
            delete_secret("svc", "absent")


# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_no_op_fail_backend_raises_unavailable(self):
        """When the active backend is the keyring 'fail' stub (i.e. the
        user has no real backend installed), every operation must raise
        KeyringUnavailableError instead of a confusing low-level error."""
        from keyring.backends.fail import Keyring as FailKeyring

        with patch("axon.security.keyring._keyring.get_keyring", return_value=FailKeyring()):
            with pytest.raises(KeyringUnavailableError):
                store_secret("svc", "u", "x")
            with pytest.raises(KeyringUnavailableError):
                get_secret("svc", "u")
            with pytest.raises(KeyringUnavailableError):
                delete_secret("svc", "u")

    def test_is_available_true_when_round_trip_works(self):
        backend = _InMemoryKeyring()
        with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
            assert is_available() is True

    def test_is_available_false_when_backend_unavailable(self):
        from keyring.backends.fail import Keyring as FailKeyring

        with patch("axon.security.keyring._keyring.get_keyring", return_value=FailKeyring()):
            assert is_available() is False
