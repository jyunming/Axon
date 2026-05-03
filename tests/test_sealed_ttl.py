"""Tests for v0.4.0 TTL-gated sealed-share — PR B (expiry sidecar +
TTL check + auto-destroy plumbing in :mod:`axon.security.share`).

Covers:

1. Sidecar write/read/sign/verify round-trip — owner side
2. ``generate_sealed_share(expires_at=...)`` writes a valid sidecar
3. ``_check_expiry_or_raise`` raises ShareExpiredError on:
   - Expired ``expires_at``
   - Tampered ``expires_at`` (signature mismatch)
   - Tampered embedded ``key_id`` (rename attack)
   - Sidecar present but pubkey_hex empty (TTL unenforceable)
   - Truncated/garbage sidecar JSON
4. No-sidecar = no-op (TTL-less shares unaffected)
5. SEALED1 backward compatibility — TTL is unenforceable, code path
   doesn't crash if a sidecar somehow exists (defensive)

The auto-destroy half (in ``axon.main._auto_destroy_expired_share``)
is tested separately — needs a brain instance.

Run with:
    PYTHONPATH=src python -m pytest tests/test_sealed_ttl.py -v --no-cov
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="requires axon-rag[sealed] / cryptography")

from axon.security import SecurityError, ShareExpiredError  # noqa: E402
from axon.security.share import (  # noqa: E402
    _check_expiry_or_raise,
    _expiry_signing_message,
    _write_expiry_sidecar,
    share_expiry_path,
)
from axon.security.signing import (  # noqa: E402
    derive_signing_keypair,
    pubkey_to_hex,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keypair():
    """Return a fresh (privkey, pubkey, pubkey_hex) triple."""
    master = os.urandom(32)
    priv, pub = derive_signing_keypair(master)
    return priv, pub, pubkey_to_hex(pub)


def _make_project_dir(tmp_path: Path) -> Path:
    """Make a minimal project layout that supports share_expiry_path."""
    proj = tmp_path / "research"
    (proj / ".security" / "shares").mkdir(parents=True)
    return proj


# ---------------------------------------------------------------------------
# Sidecar write / verify round-trip
# ---------------------------------------------------------------------------


class TestWriteExpirySidecar:
    """``_write_expiry_sidecar`` — owner-side persistence."""

    def test_round_trip_with_future_expiry_passes(self, tmp_path):
        priv, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        future = datetime.now(timezone.utc) + timedelta(days=30)
        sidecar_path = _write_expiry_sidecar(proj, "ssk_round01", future, priv)
        assert sidecar_path.is_file()
        # No exception → sidecar verifies cleanly
        _check_expiry_or_raise(proj, "ssk_round01", pubkey_hex)

    def test_naive_datetime_rejected(self, tmp_path):
        priv, *_ = _make_keypair()
        proj = _make_project_dir(tmp_path)
        naive = datetime(2099, 1, 1)  # no tzinfo
        with pytest.raises(SecurityError, match="timezone-aware"):
            _write_expiry_sidecar(proj, "ssk_naive", naive, priv)

    def test_non_datetime_rejected(self, tmp_path):
        priv, *_ = _make_keypair()
        proj = _make_project_dir(tmp_path)
        with pytest.raises(SecurityError, match="must be a datetime"):
            _write_expiry_sidecar(proj, "ssk_str", "2099-01-01", priv)  # type: ignore[arg-type]

    def test_sidecar_contents_format(self, tmp_path):
        priv, _pub, _pkhex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        future = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        path = _write_expiry_sidecar(proj, "ssk_format", future, priv)
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        assert sidecar["key_id"] == "ssk_format"
        assert sidecar["expires_at"] == "2099-12-31T23:59:59Z"  # canonical Z form
        assert "sig" in sidecar
        # base64url-encoded 64-byte Ed25519 signature decodes to 64 bytes
        pad = "=" * (-len(sidecar["sig"]) % 4)
        sig_bytes = base64.urlsafe_b64decode(sidecar["sig"] + pad)
        assert len(sig_bytes) == 64


# ---------------------------------------------------------------------------
# _check_expiry_or_raise — grantee-side TTL enforcement
# ---------------------------------------------------------------------------


class TestCheckExpiryOrRaise:
    """Verify the 7 failure modes plus the 2 happy paths."""

    def test_no_sidecar_is_noop(self, tmp_path):
        """A share with no expiry sidecar must not raise — unchanged
        behavior for TTL-less shares (which is the vast majority)."""
        proj = _make_project_dir(tmp_path)
        _, _pub, pubkey_hex = _make_keypair()
        # No sidecar written. Must return without raising.
        _check_expiry_or_raise(proj, "ssk_no_sidecar", pubkey_hex)

    def test_future_expiry_passes(self, tmp_path):
        priv, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        _write_expiry_sidecar(proj, "ssk_future", future, priv)
        _check_expiry_or_raise(proj, "ssk_future", pubkey_hex)  # no raise

    def test_expired_raises(self, tmp_path):
        priv, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        _write_expiry_sidecar(proj, "ssk_expired", past, priv)
        with pytest.raises(ShareExpiredError, match="expired at"):
            _check_expiry_or_raise(proj, "ssk_expired", pubkey_hex)

    def test_tampered_expires_at_fails_signature(self, tmp_path):
        """Edit the expires_at in the sidecar JSON without re-signing.
        Verify must reject — otherwise a grantee can extend their own
        access by bumping the date."""
        priv, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        past = datetime.now(timezone.utc) - timedelta(days=1)
        path = _write_expiry_sidecar(proj, "ssk_tamper", past, priv)
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        sidecar["expires_at"] = "2099-12-31T23:59:59Z"  # forged future
        path.write_text(json.dumps(sidecar), encoding="utf-8")
        with pytest.raises(ShareExpiredError, match="signature"):
            _check_expiry_or_raise(proj, "ssk_tamper", pubkey_hex)

    def test_tampered_embedded_key_id_fails(self, tmp_path):
        """Rename attack — copy alice's longer-lived sidecar onto
        bob's key_id. The filename matches bob, but the embedded
        key_id field still says alice. Must fail."""
        priv, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        future = datetime.now(timezone.utc) + timedelta(days=365)
        # Write sidecar for alice
        alice_path = _write_expiry_sidecar(proj, "ssk_alice", future, priv)
        # Copy onto bob's filename without updating contents
        bob_path = share_expiry_path(proj, "ssk_bob")
        bob_path.write_bytes(alice_path.read_bytes())
        with pytest.raises(ShareExpiredError, match="key_id mismatch"):
            _check_expiry_or_raise(proj, "ssk_bob", pubkey_hex)

    def test_missing_pubkey_hex_treated_as_expired(self, tmp_path):
        """SEALED1 mount with a sidecar somehow present — TTL is
        unenforceable so we treat as expired (fail closed)."""
        priv, *_ = _make_keypair()
        proj = _make_project_dir(tmp_path)
        future = datetime.now(timezone.utc) + timedelta(days=30)
        _write_expiry_sidecar(proj, "ssk_no_pk", future, priv)
        with pytest.raises(ShareExpiredError, match="no signing pubkey"):
            _check_expiry_or_raise(proj, "ssk_no_pk", "")

    def test_malformed_json_fails(self, tmp_path):
        _, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        path = share_expiry_path(proj, "ssk_garbage")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not-json{")
        with pytest.raises(ShareExpiredError, match="malformed"):
            _check_expiry_or_raise(proj, "ssk_garbage", pubkey_hex)

    def test_missing_required_fields_fails(self, tmp_path):
        _, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        path = share_expiry_path(proj, "ssk_partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Missing 'sig' field
        path.write_text(json.dumps({"key_id": "ssk_partial", "expires_at": "2099-01-01T00:00:00Z"}))
        with pytest.raises(ShareExpiredError, match="missing required"):
            _check_expiry_or_raise(proj, "ssk_partial", pubkey_hex)

    @pytest.mark.parametrize(
        "non_dict_payload",
        [
            "[]",  # array
            "null",
            "42",
            '"a string"',
            "true",
        ],
    )
    def test_non_dict_json_treated_as_malformed(self, tmp_path, non_dict_payload):
        """JSON ``[]``, ``null``, etc. parse cleanly but aren't dicts.
        Without an isinstance() check, ``.get()`` / ``.replace()`` would
        raise AttributeError and bypass the ShareExpiredError contract.
        """
        _, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        path = share_expiry_path(proj, "ssk_nondict")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(non_dict_payload)
        with pytest.raises(ShareExpiredError, match="not a JSON object"):
            _check_expiry_or_raise(proj, "ssk_nondict", pubkey_hex)

    def test_non_string_field_values_treated_as_malformed(self, tmp_path):
        """Sidecar with valid object shape but a non-string field
        (e.g. ``"expires_at": 42``) must fail with the missing-fields
        error rather than crash on ``.replace()`` later."""
        _, _pub, pubkey_hex = _make_keypair()
        proj = _make_project_dir(tmp_path)
        path = share_expiry_path(proj, "ssk_typebad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key_id": "ssk_typebad", "expires_at": 42, "sig": "abc"}))
        with pytest.raises(ShareExpiredError, match="non-empty strings"):
            _check_expiry_or_raise(proj, "ssk_typebad", pubkey_hex)


# ---------------------------------------------------------------------------
# Wire-format invariants
# ---------------------------------------------------------------------------


class TestSigningMessageFormat:
    """The signed message MUST be ``"key_id:expires_at_iso".encode()``.
    Don't change the format without bumping the envelope version."""

    def test_signing_message_is_concatenation(self):
        msg = _expiry_signing_message("ssk_format", "2099-12-31T23:59:59Z")
        assert msg == b"ssk_format:2099-12-31T23:59:59Z"

    def test_auto_destroy_strips_mounts_prefix(self, tmp_path, monkeypatch):
        """Regression: ``_mount_sealed_project`` calls auto_destroy with the
        full project identifier (e.g. ``"mounts/alice_research"``), but
        ``remove_mount_descriptor`` expects the bare mount name. The
        helper must strip the prefix or the descriptor stays orphaned.
        """
        from unittest.mock import MagicMock

        from axon.main import AxonBrain

        captured: dict[str, str] = {}

        def fake_remove_mount_descriptor(user_dir, name):
            captured["name"] = name
            return True

        # Patch where remove_mount_descriptor is imported (lazy in the helper)
        import axon.mounts as _mounts

        monkeypatch.setattr(_mounts, "remove_mount_descriptor", fake_remove_mount_descriptor)
        monkeypatch.setattr("axon.security.share.delete_grantee_dek", lambda *a, **k: True)

        brain = MagicMock()
        brain.config = MagicMock()
        brain.config.projects_root = str(tmp_path)
        brain._sealed_cache = None
        AxonBrain._auto_destroy_expired_share(
            brain, "mounts/alice_research", "ssk_test", Exception("expired")
        )
        # Must have stripped the ``mounts/`` prefix
        assert captured["name"] == "alice_research"

    def test_iso_form_uses_z_suffix(self, tmp_path):
        """The sidecar must always emit the ``Z`` form, never
        ``+00:00``. Otherwise a grantee parsing in a stricter ISO
        library would reject the file."""
        priv, *_ = _make_keypair()
        proj = _make_project_dir(tmp_path)
        # datetime with explicit +00:00 — write_expiry_sidecar must
        # canonicalise to Z
        future = datetime(2099, 1, 1, tzinfo=timezone(timedelta(0)))
        path = _write_expiry_sidecar(proj, "ssk_iso", future, priv)
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        assert sidecar["expires_at"].endswith("Z")
        assert "+00:00" not in sidecar["expires_at"]
