"""Phase 1 of the sealed-mount design: crypto primitive tests.

Covers ``axon.security.crypto`` end-to-end:
- :class:`SealedFile` round-trip, atomic write, header parsing,
  tamper / wrong-key / wrong-AAD detection
- DEK generation
- HKDF-SHA256 KEK derivation determinism + per-key_id divergence
- AES-KW (RFC 3394) wrap/unwrap, wrong-KEK rejection
- Built-in self-check

Skips the entire module when the optional ``cryptography`` package
isn't installed (the ``sealed`` extra hasn't been pulled in).
"""
from __future__ import annotations

import os

import pytest

# Skip the whole module if the sealed extra isn't installed — keeps the
# core test suite green on minimal installs.
pytest.importorskip("cryptography")

from axon.security.crypto import (  # noqa: E402
    CIPHER_AES_256_GCM,
    DEK_LEN,
    HEADER_LEN,
    MAGIC,
    NONCE_LEN,
    SCHEMA_VERSION,
    TAG_LEN,
    SealedFile,
    SealedFormatError,
    _self_check,
    derive_kek,
    generate_dek,
    make_aad,
    unwrap_key,
    wrap_key,
)

# ---------------------------------------------------------------------------
# generate_dek
# ---------------------------------------------------------------------------


class TestGenerateDek:
    def test_returns_32_bytes(self):
        dek = generate_dek()
        assert isinstance(dek, bytes)
        assert len(dek) == DEK_LEN == 32

    def test_distinct_calls_produce_distinct_keys(self):
        # 1000 fresh DEKs — collision probability is ~2^-256 * 1000,
        # i.e. astronomically zero. Catches a stuck PRNG.
        keys = {generate_dek() for _ in range(1000)}
        assert len(keys) == 1000


# ---------------------------------------------------------------------------
# derive_kek
# ---------------------------------------------------------------------------


class TestDeriveKek:
    def test_deterministic_for_same_inputs(self):
        token = b"\x01" * 32
        a = derive_kek(token, "sk_abc")
        b = derive_kek(token, "sk_abc")
        assert a == b
        assert len(a) == 32

    def test_different_key_id_yields_different_kek(self):
        token = b"\x01" * 32
        assert derive_kek(token, "sk_aaa") != derive_kek(token, "sk_bbb")

    def test_different_token_yields_different_kek(self):
        assert derive_kek(b"\x01" * 32, "sk_abc") != derive_kek(b"\x02" * 32, "sk_abc")

    def test_different_info_yields_different_kek(self):
        token = b"\x01" * 32
        v1 = derive_kek(token, "sk_abc", info=b"axon-share-v1")
        v2 = derive_kek(token, "sk_abc", info=b"axon-share-v2")
        assert v1 != v2

    def test_token_must_be_bytes(self):
        with pytest.raises(TypeError):
            derive_kek("not-bytes", "sk_abc")  # type: ignore[arg-type]

    def test_key_id_must_be_non_empty(self):
        with pytest.raises(ValueError):
            derive_kek(b"\x01" * 32, "")


# ---------------------------------------------------------------------------
# wrap_key / unwrap_key
# ---------------------------------------------------------------------------


class TestKeyWrap:
    def test_round_trip(self):
        dek = generate_dek()
        kek = generate_dek()
        wrapped = wrap_key(dek, kek)
        # AES-KW wraps a 32-byte payload into 32 + 8 = 40 bytes.
        assert len(wrapped) == 40
        assert unwrap_key(wrapped, kek) == dek

    def test_wrong_kek_raises(self):
        from cryptography.hazmat.primitives.keywrap import InvalidUnwrap

        dek = generate_dek()
        kek = generate_dek()
        wrong_kek = generate_dek()
        wrapped = wrap_key(dek, kek)
        with pytest.raises(InvalidUnwrap):
            unwrap_key(wrapped, wrong_kek)

    def test_tampered_wrap_raises(self):
        from cryptography.hazmat.primitives.keywrap import InvalidUnwrap

        dek = generate_dek()
        kek = generate_dek()
        wrapped = bytearray(wrap_key(dek, kek))
        wrapped[0] ^= 0xFF  # flip a bit in the integrity check block
        with pytest.raises(InvalidUnwrap):
            unwrap_key(bytes(wrapped), kek)

    def test_wrong_size_key_rejected(self):
        with pytest.raises(ValueError):
            wrap_key(b"short", generate_dek())
        with pytest.raises(ValueError):
            wrap_key(generate_dek(), b"short")


# ---------------------------------------------------------------------------
# SealedFile — write / read happy paths
# ---------------------------------------------------------------------------


class TestSealedFileRoundTrip:
    def test_basic_round_trip(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "secret.bin"
        payload = b"hello sealed axon"
        SealedFile.write(path, payload, key)
        assert SealedFile.read(path, key) == payload

    def test_round_trip_with_aad(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "secret.bin"
        payload = b"with aad"
        aad = make_aad("sk_abc", "vector_store_data/manifest.json")
        SealedFile.write(path, payload, key, aad=aad)
        assert SealedFile.read(path, key, aad=aad) == payload

    def test_round_trip_empty_plaintext(self, tmp_path):
        # 0-byte plaintext is legal — file still has header + nonce + tag.
        key = generate_dek()
        path = tmp_path / "empty.bin"
        SealedFile.write(path, b"", key)
        assert SealedFile.read(path, key) == b""
        # Sanity: file is exactly header + nonce + tag bytes long.
        assert path.stat().st_size == HEADER_LEN + NONCE_LEN + TAG_LEN

    def test_round_trip_large_payload(self, tmp_path):
        # 1 MiB payload — exercises the AESGCM encrypt path on a
        # non-trivial buffer without taking real time.
        key = generate_dek()
        path = tmp_path / "big.bin"
        payload = os.urandom(1024 * 1024)
        SealedFile.write(path, payload, key)
        assert SealedFile.read(path, key) == payload

    def test_creates_parent_dirs(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "nested" / "deep" / "secret.bin"
        SealedFile.write(path, b"x", key)
        assert path.is_file()


# ---------------------------------------------------------------------------
# SealedFile — atomic write
# ---------------------------------------------------------------------------


class TestSealedFileAtomicWrite:
    def test_no_sealing_tempfile_left_after_success(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"x", key)
        siblings = sorted(p.name for p in tmp_path.iterdir())
        # The .sealing temp file must be replaced/removed by os.replace.
        assert "secret.bin.sealing" not in siblings
        assert "secret.bin" in siblings

    def test_overwrites_existing_file(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"first", key)
        SealedFile.write(path, b"second", key)
        assert SealedFile.read(path, key) == b"second"


# ---------------------------------------------------------------------------
# SealedFile — header parsing
# ---------------------------------------------------------------------------


class TestSealedFileHeader:
    def test_header_magic_present_in_written_file(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"x", key)
        body = path.read_bytes()
        assert body[:4] == MAGIC
        assert body[4] == SCHEMA_VERSION
        assert body[5] == CIPHER_AES_256_GCM

    def test_bad_magic_raises_format_error(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "bad.bin"
        # Write a file with the wrong magic but otherwise plausible shape.
        body = (
            b"NOPE" + b"\x01" + b"\x00" + (b"\x00" * 10) + b"\x00" * NONCE_LEN + b"\x00" * TAG_LEN
        )
        path.write_bytes(body)
        with pytest.raises(SealedFormatError, match="magic"):
            SealedFile.read(path, key)

    def test_unsupported_schema_version_raises_format_error(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "future.bin"
        body = MAGIC + b"\x99" + b"\x00" + (b"\x00" * 10) + b"\x00" * NONCE_LEN + b"\x00" * TAG_LEN
        path.write_bytes(body)
        with pytest.raises(SealedFormatError, match="version"):
            SealedFile.read(path, key)

    def test_unknown_cipher_id_raises_format_error(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "wrong-cipher.bin"
        body = (
            MAGIC
            + bytes([SCHEMA_VERSION])
            + b"\x99"
            + (b"\x00" * 10)
            + b"\x00" * NONCE_LEN
            + b"\x00" * TAG_LEN
        )
        path.write_bytes(body)
        with pytest.raises(SealedFormatError, match="cipher_id"):
            SealedFile.read(path, key)

    def test_short_file_raises_format_error(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "tiny.bin"
        path.write_bytes(b"AXSL")  # missing everything else
        with pytest.raises(SealedFormatError, match="too short"):
            SealedFile.read(path, key)


# ---------------------------------------------------------------------------
# SealedFile — tamper / wrong key / wrong AAD detection
# ---------------------------------------------------------------------------


class TestSealedFileIntegrity:
    def test_wrong_key_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        wrong_key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"hello", key)
        with pytest.raises(InvalidTag):
            SealedFile.read(path, wrong_key)

    def test_tampered_ciphertext_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"hello world this is the secret payload", key)
        body = bytearray(path.read_bytes())
        # Flip a bit in the ciphertext region (right after the nonce).
        body[HEADER_LEN + NONCE_LEN + 1] ^= 0xFF
        path.write_bytes(bytes(body))
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key)

    def test_tampered_tag_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"hello", key)
        body = bytearray(path.read_bytes())
        body[-1] ^= 0xFF  # flip a bit in the GCM tag
        path.write_bytes(bytes(body))
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key)

    def test_wrong_aad_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"x", key, aad=b"correct-aad")
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key, aad=b"wrong-aad")

    def test_aad_omitted_on_read_when_written_with_aad_raises(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "secret.bin"
        SealedFile.write(path, b"x", key, aad=b"present")
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key)  # no AAD → mismatch


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_write_rejects_wrong_size_key(self, tmp_path):
        with pytest.raises(ValueError, match="32 bytes"):
            SealedFile.write(tmp_path / "x", b"data", b"short")

    def test_read_rejects_wrong_size_key(self, tmp_path):
        path = tmp_path / "x"
        SealedFile.write(path, b"data", generate_dek())
        with pytest.raises(ValueError, match="32 bytes"):
            SealedFile.read(path, b"short")


# ---------------------------------------------------------------------------
# AAD helper
# ---------------------------------------------------------------------------


class TestMakeAad:
    def test_combines_key_id_and_relpath(self):
        aad = make_aad("sk_a", "meta.json")
        assert b"sk_a" in aad
        assert b"meta.json" in aad
        # NUL separator means key_id "sk" + relpath "meta.json" doesn't
        # collide with key_id "s" + relpath "kmeta.json".
        assert b"\x00" in aad

    def test_distinct_inputs_produce_distinct_aads(self):
        a = make_aad("sk_a", "meta.json")
        b = make_aad("sk_a", "vector_store_data/manifest.json")
        c = make_aad("sk_b", "meta.json")
        assert a != b
        assert a != c
        assert b != c


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


class TestSelfCheck:
    def test_self_check_ok_on_healthy_install(self):
        result = _self_check()
        assert result["ok"] is True
        assert "OK" in result["details"]
