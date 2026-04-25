"""Cryptographic primitives for sealed Axon projects (Phase 1 of #SEALED).

Two primitive layers:

1. :class:`SealedFile` — wraps any byte payload with AES-256-GCM. File
   format: 16-byte ``AXSL`` header + 12-byte nonce + ciphertext +
   16-byte authentication tag. Tampering (header, nonce, ciphertext, or
   tag) raises :class:`cryptography.exceptions.InvalidTag` — never
   silently succeeds.

2. **Envelope keys** — :func:`derive_kek` (HKDF-SHA256) derives a Key
   Encryption Key from a share token; :func:`wrap_key` /
   :func:`unwrap_key` (AES-256-KW, RFC 3394) wrap/unwrap a Data
   Encryption Key for transport. :func:`generate_dek` produces a fresh
   256-bit DEK from the OS CSPRNG.

This module **only** provides the primitives — no code outside the
package consumes them yet. Wiring into project-seal / generate-sealed-
share / redeem-sealed-share lands in Phases 2–4 per
``docs/SHARE_MOUNT_SEALED.md``.

Dependencies: ``cryptography`` (PyPI) — installed via the ``sealed``
extra (``pip install axon-rag[sealed]``). Importing this module on a
minimal install raises a friendly ``ImportError`` with the install
hint.
"""
from __future__ import annotations

import os
import secrets
import struct
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.keywrap import (
        aes_key_unwrap,
        aes_key_wrap,
    )
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.crypto requires the 'cryptography' package. "
        "Install with: pip install axon-rag[sealed]"
    ) from exc

__all__ = [
    "MAGIC",
    "SCHEMA_VERSION",
    "CIPHER_AES_256_GCM",
    "HEADER_LEN",
    "NONCE_LEN",
    "TAG_LEN",
    "DEK_LEN",
    "SealedFormatError",
    "SealedFile",
    "generate_dek",
    "derive_kek",
    "wrap_key",
    "unwrap_key",
]

# ---------------------------------------------------------------------------
# File-format constants
# ---------------------------------------------------------------------------

MAGIC: bytes = b"AXSL"
SCHEMA_VERSION: int = 1
CIPHER_AES_256_GCM: int = 0

HEADER_LEN: int = 16  # 4 magic + 1 version + 1 cipher_id + 10 reserved/zero
NONCE_LEN: int = 12  # AES-GCM standard nonce length
TAG_LEN: int = 16  # AES-GCM authentication tag length
DEK_LEN: int = 32  # 256-bit Data Encryption Key

# 4-byte magic | 1-byte version | 1-byte cipher_id | 10-byte reserved (zero)
_HEADER_STRUCT = struct.Struct(">4sBB10x")


class SealedFormatError(Exception):
    """Raised when a sealed file's header is malformed, has the wrong
    magic, an unsupported schema version, or an unknown cipher ID.
    Distinct from :class:`cryptography.exceptions.InvalidTag` — that
    indicates the bytes were AES-GCM-validated against the wrong key
    (tamper / wrong-key), whereas ``SealedFormatError`` indicates the
    file isn't a sealed Axon file at all (or is from a future schema we
    don't support).
    """


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------


def _pack_header(*, version: int = SCHEMA_VERSION, cipher_id: int = CIPHER_AES_256_GCM) -> bytes:
    return _HEADER_STRUCT.pack(MAGIC, version, cipher_id)


def _unpack_header(header: bytes) -> tuple[int, int]:
    """Return ``(version, cipher_id)``; raise SealedFormatError on mismatch."""
    if len(header) != HEADER_LEN:
        raise SealedFormatError(f"Header must be {HEADER_LEN} bytes, got {len(header)}")
    try:
        magic, version, cipher_id = _HEADER_STRUCT.unpack(header)
    except struct.error as exc:
        raise SealedFormatError(f"Could not unpack header: {exc}") from exc
    if magic != MAGIC:
        raise SealedFormatError(
            f"Bad magic: expected {MAGIC!r}, got {magic!r} — not an AXSL sealed file"
        )
    if version != SCHEMA_VERSION:
        # Forward-compat policy: refuse to read newer schemas rather than
        # silently misinterpret. v1 readers don't know what v2 will do.
        raise SealedFormatError(
            f"Unsupported schema version {version} (this build understands {SCHEMA_VERSION})"
        )
    if cipher_id != CIPHER_AES_256_GCM:
        raise SealedFormatError(
            f"Unsupported cipher_id {cipher_id} (only AES-256-GCM = {CIPHER_AES_256_GCM})"
        )
    return version, cipher_id


# ---------------------------------------------------------------------------
# SealedFile — one-call seal / unseal
# ---------------------------------------------------------------------------


class SealedFile:
    """Atomic AES-256-GCM file wrapper.
    Both :meth:`write` and :meth:`read` operate on whole-file
    plaintext: there is no streaming API in v1 (decrypt-into-memory
    policy per the plan doc). Adding streaming is straightforward later
    — the on-disk format already records everything needed.
    """

    @staticmethod
    def write(
        path: Path | str,
        plaintext: bytes,
        key: bytes,
        *,
        aad: bytes = b"",
    ) -> None:
        """Encrypt *plaintext* with *key* and write to *path* atomically.
        Args:
            path: Destination file path.
            plaintext: Bytes to encrypt.
            key: 32-byte AES-256 key.
            aad: Additional Authenticated Data — bound into the GCM tag
                but NOT stored in the file. Callers must supply the
                same AAD on read or :class:`InvalidTag` will be raised.
                Recommended: ``key_id || file_relpath`` to prevent files
                from being swapped between projects.
        The write is atomic — encrypted bytes are written to a sibling
        ``<name>.sealing`` file then renamed via :func:`os.replace` so a
        crash mid-write never leaves a half-sealed file at the live
        path.
        """
        if len(key) != 32:
            raise ValueError(f"key must be 32 bytes (AES-256), got {len(key)}")
        path = Path(path)
        nonce = os.urandom(NONCE_LEN)
        # AESGCM.encrypt returns ciphertext || tag concatenated.
        ct_and_tag = AESGCM(key).encrypt(nonce, plaintext, aad if aad else None)
        body = _pack_header() + nonce + ct_and_tag
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".sealing")
        tmp.write_bytes(body)
        os.replace(tmp, path)

    @staticmethod
    def read(
        path: Path | str,
        key: bytes,
        *,
        aad: bytes = b"",
    ) -> bytes:
        """Decrypt and return the plaintext content of *path*.
        Raises:
            SealedFormatError: file doesn't have an AXSL header, or has
                an unsupported version / cipher.
            cryptography.exceptions.InvalidTag: bytes were tampered
                with, the wrong key was supplied, or the AAD doesn't
                match the value used at write time.
            FileNotFoundError: file does not exist.
        """
        if len(key) != 32:
            raise ValueError(f"key must be 32 bytes (AES-256), got {len(key)}")
        path = Path(path)
        body = path.read_bytes()
        if len(body) < HEADER_LEN + NONCE_LEN + TAG_LEN:
            raise SealedFormatError(f"File too short to be a sealed AXSL file ({len(body)} bytes)")
        _unpack_header(body[:HEADER_LEN])
        nonce = body[HEADER_LEN : HEADER_LEN + NONCE_LEN]
        ct_and_tag = body[HEADER_LEN + NONCE_LEN :]
        return AESGCM(key).decrypt(nonce, ct_and_tag, aad if aad else None)


# ---------------------------------------------------------------------------
# Key generation + envelope wrap/unwrap
# ---------------------------------------------------------------------------


def generate_dek() -> bytes:
    """Return a fresh 256-bit Data Encryption Key from the OS CSPRNG."""
    return secrets.token_bytes(DEK_LEN)


def derive_kek(token: bytes, key_id: str, *, info: bytes = b"axon-share-v1") -> bytes:
    """Derive a 256-bit Key Encryption Key from a share token via HKDF-SHA256.
    Args:
        token: The raw share token (32 random bytes from ``secrets``).
            Same token Axon already issues today via
            ``shares.generate_share_key``.
        key_id: The share's ``key_id`` (e.g. ``sk_a1b2c3d4``). Used as
            the HKDF salt so two shares with the same token (impossible
            in practice but defensive) derive distinct KEKs.
        info: HKDF info parameter — versions the KEK derivation. Bumping
            this string breaks backwards compatibility, so it should
            change only when the wrapping scheme changes (e.g. v2
            switches to AES-XTS).
    Returns:
        32-byte KEK suitable for :func:`wrap_key`/:func:`unwrap_key`.
    """
    if not isinstance(token, bytes | bytearray):
        raise TypeError("token must be bytes")
    if not key_id:
        raise ValueError("key_id must be a non-empty string")
    salt = key_id.encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=DEK_LEN,
        salt=salt,
        info=info,
    ).derive(bytes(token))


def wrap_key(key: bytes, kek: bytes) -> bytes:
    """AES-256 Key Wrap (RFC 3394) of *key* under *kek*.
    Used to encrypt a 32-byte DEK with a 32-byte KEK; produces a
    40-byte wrapped output (8-byte integrity check + 32-byte ciphertext).
    """
    if len(key) != 32 or len(kek) != 32:
        raise ValueError(f"both key and kek must be 32 bytes; got key={len(key)}, kek={len(kek)}")
    return aes_key_wrap(kek, key)


def unwrap_key(wrapped: bytes, kek: bytes) -> bytes:
    """Reverse of :func:`wrap_key`. Raises ``cryptography.exceptions.InvalidUnwrap``
    if the wrap is invalid (wrong KEK, tampered bytes)."""
    if len(kek) != 32:
        raise ValueError(f"kek must be 32 bytes; got {len(kek)}")
    return aes_key_unwrap(kek, wrapped)


# ---------------------------------------------------------------------------
# Convenience: build an AAD value
# ---------------------------------------------------------------------------


def make_aad(key_id: str, relpath: str) -> bytes:
    """Recommended AAD for project files: ``key_id || NUL || relpath``.
    Binding both the share key and the in-project relative path into
    the GCM tag prevents a same-key attacker from swapping files
    between projects (e.g. moving an encrypted ``meta.json`` from
    project A onto project B and having it decrypt cleanly).
    """
    return key_id.encode("utf-8") + b"\x00" + relpath.encode("utf-8")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check() -> dict[str, Any]:
    """Round-trip a tiny payload to confirm the build is wired correctly.
    Returned as a dict so callers (e.g. ``axon doctor`` in a later
    phase) can include it in diagnostics output. Never raises — failures
    are reported in the dict.
    """
    out: dict[str, Any] = {"ok": False, "details": ""}
    try:
        dek = generate_dek()
        token = secrets.token_bytes(32)
        kek = derive_kek(token, "sk_selfcheck")
        wrapped = wrap_key(dek, kek)
        unwrapped = unwrap_key(wrapped, kek)
        if unwrapped != dek:
            out["details"] = "wrap/unwrap roundtrip mismatch"
            return out
        # File round-trip in /tmp
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "selfcheck.bin"
            payload = b"axon-sealed-selfcheck"
            SealedFile.write(p, payload, dek, aad=b"selfcheck-aad")
            recovered = SealedFile.read(p, dek, aad=b"selfcheck-aad")
            if recovered != payload:
                out["details"] = "SealedFile roundtrip mismatch"
                return out
        out["ok"] = True
        out["details"] = "AES-256-GCM + AES-KW + HKDF-SHA256 OK"
    except Exception as exc:  # pragma: no cover — self-check is defensive
        out["details"] = f"self-check raised: {type(exc).__name__}: {exc}"
    return out
