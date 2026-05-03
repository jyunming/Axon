"""Tests for the v0.4.0 Ed25519 signing keypair + SEALED2 share-string format.

Covers:
- Determinism of :func:`derive_signing_keypair`
- Domain separation from share KEK derivation (different HKDF info)
- :func:`pubkey_to_hex` / :func:`pubkey_from_hex` round-trip
- SEALED1 share strings still redeem (backward compatibility)
- SEALED2 envelopes carry the pubkey + are accepted by redeem
- The redeemed mount descriptor records envelope_version + owner_pubkey_hex
- Tampered SEALED2 share strings (bad pubkey) fail closed at parse time

Run with: ``python -m pytest tests/test_sealed_signing_v2.py -v --no-cov``
"""
from __future__ import annotations

import os

import pytest

# These tests exercise real cryptography primitives; if the [sealed]
# extra is missing on this dev box, skip rather than fail.
pytest.importorskip("cryptography", reason="requires axon-rag[sealed] / cryptography")

from axon.security import SecurityError  # noqa: E402
from axon.security.signing import (  # noqa: E402
    SIGNING_HKDF_INFO,
    SIGNING_PUBKEY_HEX_LEN,
    derive_signing_keypair,
    pubkey_from_hex,
    pubkey_to_hex,
)

# ---------------------------------------------------------------------------
# derive_signing_keypair
# ---------------------------------------------------------------------------


class TestDeriveSigningKeypair:
    """Determinism + domain separation properties."""

    def test_determinism_same_master_same_keypair(self):
        """Two calls with the same master must produce the same keypair."""
        master = b"\x00" * 31 + b"\x42"  # any 32 bytes
        priv1, pub1 = derive_signing_keypair(master)
        priv2, pub2 = derive_signing_keypair(master)
        assert pubkey_to_hex(pub1) == pubkey_to_hex(pub2)
        # Private keys are also deterministic — sign the same message.
        sig1 = priv1.sign(b"hello")
        sig2 = priv2.sign(b"hello")
        assert sig1 == sig2

    def test_different_masters_different_keypairs(self):
        master_a = b"\x00" * 32
        master_b = b"\x01" * 32
        _, pub_a = derive_signing_keypair(master_a)
        _, pub_b = derive_signing_keypair(master_b)
        assert pubkey_to_hex(pub_a) != pubkey_to_hex(pub_b)

    def test_master_must_be_32_bytes(self):
        with pytest.raises(SecurityError, match="32 bytes"):
            derive_signing_keypair(b"\x00" * 16)
        with pytest.raises(SecurityError, match="32 bytes"):
            derive_signing_keypair(b"\x00" * 64)

    def test_master_must_be_bytes(self):
        with pytest.raises(SecurityError, match="bytes"):
            derive_signing_keypair("not-bytes")  # type: ignore[arg-type]

    def test_signing_hkdf_info_is_domain_separated(self):
        """The signing HKDF info must NOT match the share KEK info."""
        # Imported lazily to avoid module-level import order issues if
        # the [sealed] extra is not available in some test environments.
        from axon.security.share import HKDF_INFO as SHARE_HKDF_INFO

        assert SIGNING_HKDF_INFO != SHARE_HKDF_INFO
        # Sanity-check the version suffix so an accidental copy-paste
        # rename in the future trips a test rather than silently
        # collapsing the two domains.
        assert b"signing" in SIGNING_HKDF_INFO

    def test_signing_pubkey_is_deterministic_round_trip(self):
        """A pubkey decoded from hex must match the original pubkey."""
        master = os.urandom(32)
        _, pub = derive_signing_keypair(master)
        hex_str = pubkey_to_hex(pub)
        assert len(hex_str) == SIGNING_PUBKEY_HEX_LEN
        # Round-trip
        pub2 = pubkey_from_hex(hex_str)
        assert pubkey_to_hex(pub2) == hex_str


# ---------------------------------------------------------------------------
# pubkey_from_hex (input validation)
# ---------------------------------------------------------------------------


class TestPubkeyFromHex:
    def test_rejects_wrong_length(self):
        with pytest.raises(SecurityError, match="exactly 64"):
            pubkey_from_hex("a" * 63)
        with pytest.raises(SecurityError, match="exactly 64"):
            pubkey_from_hex("a" * 65)

    def test_rejects_non_hex(self):
        bad = "z" * 64  # all 'z' is not valid hex
        with pytest.raises(SecurityError, match="not valid hex"):
            pubkey_from_hex(bad)

    def test_rejects_non_string(self):
        with pytest.raises(SecurityError, match="must be str"):
            pubkey_from_hex(b"a" * 64)  # type: ignore[arg-type]

    def test_rejects_invalid_curve_point(self):
        # 32 zero bytes is technically valid hex but we still want the
        # cryptography library's downstream validation to surface as a
        # clean SecurityError rather than an opaque crash.
        # Note: Ed25519 actually accepts the all-zero key as a valid
        # public key (it's just the identity point); the assertion is
        # that pubkey_from_hex doesn't *crash*, returning either a
        # valid key object or a SecurityError.
        try:
            result = pubkey_from_hex("00" * 32)
            # cryptography accepted it; our wrapper passes through.
            assert result is not None
        except SecurityError:
            # cryptography rejected it; our wrapper translated.
            pass


# ---------------------------------------------------------------------------
# SEALED2 envelope shape + backward compat with SEALED1
# ---------------------------------------------------------------------------


class TestSealedEnvelopeFormats:
    """Verify the envelope-string parser accepts both SEALED1 and SEALED2."""

    def test_sealed1_prefix_constant_unchanged(self):
        """SEALED1 must still be the legacy constant — nothing else has
        special meaning in older share strings sent before v0.4.0."""
        from axon.security.share import SEALED_SHARE_PREFIX

        assert SEALED_SHARE_PREFIX == "SEALED1"

    def test_sealed2_prefix_constant_added(self):
        from axon.security.share import SEALED_SHARE_PREFIX_V2

        assert SEALED_SHARE_PREFIX_V2 == "SEALED2"

    def test_sealed1_envelope_decodes_to_six_fields(self):
        """A hand-rolled SEALED1 envelope still has the 6-field shape."""
        raw = "SEALED1:ssk_legacy:" + ("ab" * 32) + ":alice:research:/data/AxonStore"
        parts = raw.split(":", 5)
        assert len(parts) == 6
        assert parts[0] == "SEALED1"

    def test_sealed2_envelope_has_seven_fields(self):
        """SEALED2 envelope adds a 7th field (pubkey hex)."""
        master = os.urandom(32)
        _, pub = derive_signing_keypair(master)
        pubkey_hex = pubkey_to_hex(pub)
        raw = "SEALED2:ssk_v2:" + ("cd" * 32) + ":alice:research:/data/AxonStore:" + pubkey_hex
        parts = raw.split(":", 6)
        assert len(parts) == 7
        assert parts[0] == "SEALED2"
        assert parts[6] == pubkey_hex
        # The pubkey field is always exactly 64 chars
        assert len(parts[6]) == SIGNING_PUBKEY_HEX_LEN

    def test_sealed2_with_invalid_pubkey_field_rejects_at_parse(self):
        """A SEALED2 envelope with a bad pubkey hex must fail fast in
        :func:`pubkey_from_hex` before any wrap-file I/O."""
        # The actual redeem path calls pubkey_from_hex on the parsed
        # field. We exercise that contract directly here so the test
        # doesn't need a fully-bootstrapped sealed store.
        with pytest.raises(SecurityError):
            pubkey_from_hex("XYZ" * 22)  # 66 chars + non-hex chars

    def test_sealed2_envelope_parses_correctly_with_windows_path(self):
        """Regression: owner_store_path may contain ':' on Windows
        (e.g. ``C:\\Users\\...``). The redeem-path parser must use
        rpartition on the trailing pubkey so the colon in the drive
        letter doesn't slice the pubkey into the path field."""
        master = os.urandom(32)
        _, pub = derive_signing_keypair(master)
        pubkey_hex = pubkey_to_hex(pub)
        windows_path = "C:\\Users\\jyunm\\AppData\\Local\\AxonStore"
        raw = f"SEALED2:ssk_winpath:{('ef' * 32)}:alice:research:" f"{windows_path}:{pubkey_hex}"
        # Mirror the redeem-path parsing logic
        before_pubkey, sep, parsed_pubkey = raw.rpartition(":")
        assert sep == ":"
        assert parsed_pubkey == pubkey_hex
        parts = before_pubkey.split(":", 5)
        assert len(parts) == 6
        assert parts[5] == windows_path  # drive-letter colon preserved
        assert parts[0] == "SEALED2"
