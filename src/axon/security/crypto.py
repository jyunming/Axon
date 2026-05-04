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
``docs/architecture/SEALED_SHARING_DESIGN.md``.

Dependencies: ``cryptography`` (PyPI) — installed via the ``sealed``
extra (``pip install axon-rag[sealed]``). Importing this module on a
minimal install raises a friendly ``ImportError`` with the install
hint.
"""
from __future__ import annotations

import os
import secrets
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
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
    "STREAMING_CHUNK_SIZE",
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

# Default chunk size for streaming encryption — 1 MiB strikes a practical
# balance between per-call overhead (smaller chunks = more update() calls)
# and memory residency (larger chunks = more bytes pinned per worker).
# Override via the ``chunk_size`` kwarg on :meth:`SealedFile.write_stream`.
STREAMING_CHUNK_SIZE: int = 1024 * 1024

# 4-byte magic | 1-byte version | 1-byte cipher_id | 10-byte reserved (zero)
# Header layout (16 bytes, big-endian):
#   4s  — MAGIC ("AXSL")
#   B   — schema_version (currently 1)
#   B   — cipher_id (0 = AES-256-GCM)
#   I   — padding_length (v0.4.0 Item 4c — uint32 trailing random bytes
#         appended AFTER the GCM tag; readers slice these off before
#         calling AESGCM.decrypt). v0.3.x writers leave this 4-byte
#         region zero — backward-compatible.
#   6x  — reserved (must be zero)
_HEADER_STRUCT = struct.Struct(">4sBBI6x")


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


def _pack_header(
    *,
    version: int = SCHEMA_VERSION,
    cipher_id: int = CIPHER_AES_256_GCM,
    padding_length: int = 0,
) -> bytes:
    return _HEADER_STRUCT.pack(MAGIC, version, cipher_id, padding_length)


def _unpack_header(header: bytes) -> tuple[int, int, int]:
    """Return ``(version, cipher_id, padding_length)``.

    v0.4.0 Item 4c: ``padding_length`` is the count of random bytes
    appended after the GCM tag. v0.3.x writers leave the 4-byte region
    zero — old files therefore unpack with ``padding_length=0`` and
    decrypt unchanged.
    """
    if len(header) != HEADER_LEN:
        raise SealedFormatError(f"Header must be {HEADER_LEN} bytes, got {len(header)}")
    try:
        magic, version, cipher_id, padding_length = _HEADER_STRUCT.unpack(header)
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
    if padding_length < 0 or padding_length > 1024 * 1024:
        # Sanity bound: 1 MiB cap on padding length keeps a corrupt
        # header from making us slice past the file's end with nothing
        # actionable to do.
        raise SealedFormatError(f"Implausible padding_length in header: {padding_length}")
    return version, cipher_id, padding_length


# ---------------------------------------------------------------------------
# SealedFile — one-call seal / unseal
# ---------------------------------------------------------------------------


class SealedFile:
    """Atomic AES-256-GCM file wrapper.

    Three writers, one reader. All three writers produce byte-for-byte
    compatible output (``MAGIC + header + nonce + ciphertext + tag``)
    and any of them can be read back with :meth:`read` regardless of
    which one wrote the file:

    - :meth:`write` — buffers the whole plaintext in memory; simplest
      and fastest for small payloads (config, metadata, share wraps).
    - :meth:`write_stream` — encrypts an iterator of plaintext chunks
      incrementally via the lower-level ``Cipher`` API, never holding
      more than ``chunk_size`` bytes of plaintext + ciphertext in RAM
      at once. Use for large content files (vector segments, BM25
      indexes, raw documents) that would OOM on a buffered write.
    - :meth:`write_stream_from_path` — convenience wrapper that opens
      *src_path* for reading and pipes ``chunk_size``-byte reads into
      ``write_stream``.

    The reader (:meth:`read`) currently still loads the whole file —
    streaming reads are tracked separately. The point of the streaming
    writer is removing the 1 GiB ceiling enforced by the seal/revoke
    paths because *those* paths buffer the entire plaintext.
    """

    @staticmethod
    def write(
        path: Path | str,
        plaintext: bytes,
        key: bytes,
        *,
        aad: bytes = b"",
        padding_bytes: int = 0,
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
            padding_bytes: v0.4.0 Item 4c — when ``> 0``, append a random
                number of random bytes between ``0`` and ``padding_bytes``
                (inclusive) AFTER the GCM tag. The exact length is
                recorded in the file header so :meth:`read` can slice it
                off before calling AESGCM.decrypt. Defeats the trivial
                file-size leak (an observer otherwise infers plaintext
                length within ±28 bytes from the on-disk size). Cost:
                up to ``padding_bytes`` extra storage per file.

        The write is atomic — encrypted bytes are written to a sibling
        ``<name>.sealing`` file then renamed via :func:`os.replace` so a
        crash mid-write never leaves a half-sealed file at the live
        path.

        For payloads bigger than a few hundred MB, prefer
        :meth:`write_stream` to avoid pinning the entire plaintext in
        memory.
        """
        if len(key) != 32:
            raise ValueError(f"key must be 32 bytes (AES-256), got {len(key)}")
        if padding_bytes < 0:
            raise ValueError("padding_bytes must be >= 0")
        path = Path(path)
        nonce = os.urandom(NONCE_LEN)
        # AESGCM.encrypt returns ciphertext || tag concatenated.
        ct_and_tag = AESGCM(key).encrypt(nonce, plaintext, aad if aad else None)
        # v0.4.0 Item 4c: random-length padding (0..padding_bytes inclusive).
        # Length is the value that goes into the header; ``read`` uses it to
        # know how many trailing bytes to discard.
        pad_len = secrets.randbelow(padding_bytes + 1) if padding_bytes > 0 else 0
        padding = secrets.token_bytes(pad_len) if pad_len else b""
        body = _pack_header(padding_length=pad_len) + nonce + ct_and_tag + padding
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".sealing")
        tmp.write_bytes(body)
        os.replace(tmp, path)

    @staticmethod
    def write_stream(
        path: Path | str,
        plaintext_iter: Iterable[bytes],
        key: bytes,
        *,
        aad: bytes = b"",
        chunk_size: int = STREAMING_CHUNK_SIZE,
        padding_bytes: int = 0,
    ) -> None:
        """Encrypt *plaintext_iter* with *key* and write to *path* atomically.

        Streams the encryption with the lower-level
        :class:`cryptography.hazmat.primitives.ciphers.Cipher` API so
        only one caller-supplied chunk of plaintext + ciphertext is resident
        in memory at a time. Memory use is bounded by the caller's chunk size;
        :meth:`write_stream_from_path` reads in ``chunk_size``-byte blocks. The on-disk layout is identical to :meth:`write`
        (``MAGIC + header + nonce + ciphertext + tag``); files written
        by either method are interchangeable on read.

        AES-GCM produces a single authentication tag covering the entire
        ciphertext, so the writer must hold the destination tempfile
        open until the iterator is exhausted and ``finalize()`` returns
        — only then is the trailing 16-byte tag known. The atomic
        ``tempfile + os.replace`` discipline still applies: a crash
        mid-write leaves the original ``path`` intact (or absent) but
        never partially overwritten.

        Args:
            path: Destination file path.
            plaintext_iter: Any iterable of ``bytes`` chunks. Empty
                chunks are skipped silently. Whole-payload zero-byte
                files (an empty iterator) ARE valid — they produce a
                sealed file whose ciphertext section is zero bytes long.
            key: 32-byte AES-256 key.
            aad: Additional Authenticated Data — bound into the GCM
                tag but NOT stored in the file. Same semantics as
                :meth:`write`.
            chunk_size: Bytes pulled per ``encryptor.update()`` call.
                Iterator chunks larger than ``chunk_size`` are passed
                through whole; smaller iterator chunks are passed
                through whole. The parameter is mainly a hint to
                :meth:`write_stream_from_path` controlling its read
                size — for direct callers who provide their own iterator
                it has no effect on per-chunk processing because
                ``encryptor.update`` accepts arbitrarily-sized inputs.
        """
        if len(key) != 32:
            raise ValueError(f"key must be 32 bytes (AES-256), got {len(key)}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if padding_bytes < 0:
            raise ValueError("padding_bytes must be >= 0")
        path = Path(path)
        nonce = os.urandom(NONCE_LEN)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        if aad:
            # Must be supplied BEFORE the first update() call. AESGCM in
            # the high-level AEAD API does this implicitly; the low-level
            # Cipher API surfaces the ordering requirement to the caller.
            encryptor.authenticate_additional_data(aad)
        # v0.4.0 Item 4c: pre-roll the padding length so it can go into
        # the header BEFORE we start streaming ciphertext. The header is
        # written first because the file is built linearly.
        pad_len = secrets.randbelow(padding_bytes + 1) if padding_bytes > 0 else 0
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".sealing")
        # Open BEFORE the loop so we can stream-write ciphertext chunks
        # as they're produced. Atomic os.replace at the end.
        try:
            with tmp.open("wb") as fh:
                fh.write(_pack_header(padding_length=pad_len))
                fh.write(nonce)
                for chunk in plaintext_iter:
                    if not chunk:
                        continue
                    ct = encryptor.update(chunk)
                    if ct:
                        fh.write(ct)
                # finalize() returns any remaining ciphertext (typically
                # empty for GCM) and computes the auth tag. The tag is
                # then available on encryptor.tag — append it to the
                # file so the existing read() path can pick it up.
                tail = encryptor.finalize()
                if tail:
                    fh.write(tail)
                fh.write(encryptor.tag)
                if pad_len:
                    fh.write(secrets.token_bytes(pad_len))
            os.replace(tmp, path)
        except BaseException:
            # Best-effort cleanup on any failure — never leave a partial
            # ``.sealing`` tempfile behind.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def write_stream_from_path(
        src_path: Path | str,
        dst_path: Path | str,
        key: bytes,
        *,
        aad: bytes = b"",
        chunk_size: int = STREAMING_CHUNK_SIZE,
        padding_bytes: int = 0,
    ) -> None:
        """Convenience wrapper: read *src_path* in chunks, seal to *dst_path*.

        Equivalent to opening *src_path* and feeding fixed-size reads
        into :meth:`write_stream`. Used by the project-seal and
        hard-revoke paths so a multi-GB content file can be re-encrypted
        without ever materialising the whole plaintext.

        ``padding_bytes`` is forwarded to :meth:`write_stream` — see
        Item 4c semantics there.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        src_path = Path(src_path)

        def _reader() -> Iterator[bytes]:
            with src_path.open("rb") as fh:
                while True:
                    buf = fh.read(chunk_size)
                    if not buf:
                        break
                    yield buf

        SealedFile.write_stream(
            dst_path,
            _reader(),
            key,
            aad=aad,
            chunk_size=chunk_size,
            padding_bytes=padding_bytes,
        )

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
        _, _, padding_length = _unpack_header(body[:HEADER_LEN])
        if padding_length:
            # v0.4.0 Item 4c — slice off trailing random padding before
            # AESGCM.decrypt. Without this, the GCM tag bytes are in the
            # wrong place and decrypt raises InvalidTag.
            if len(body) < HEADER_LEN + NONCE_LEN + TAG_LEN + padding_length:
                raise SealedFormatError(
                    f"Truncated sealed file: header claims {padding_length} bytes "
                    f"of padding but body is only {len(body)} bytes"
                )
            body = body[: len(body) - padding_length]
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
