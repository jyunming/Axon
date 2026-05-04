"""Sealed-share generation + redemption (Phase 3 of #SEALED).

A *sealed share* lets an owner grant a grantee read access to a sealed
project without the grantee ever seeing the owner's master key. The
trick is **per-share key wrapping**:

1. Owner has a project DEK, wrapped under their master and stored at
   ``<project>/.security/dek.wrapped`` (Phase 2).
2. To share with Bob, the owner mints a 32-byte random *share token*,
   derives a per-share KEK via HKDF-SHA256, and uses AES-KW to wrap a
   COPY of the DEK under that KEK. The wrapped copy lands at
   ``<project>/.security/shares/<key_id>.wrapped`` — synced via
   OneDrive along with the rest of the project.
3. The owner sends Bob a ``share_string`` carrying ``key_id`` + the
   raw share token (out-of-band — Slack, email, paper note).
4. Bob runs ``axon share redeem <share_string>``. His machine:
   - re-derives the KEK from the token,
   - unwraps the DEK,
   - stores the unwrapped DEK in his OS keyring at
     ``axon.share.<key_id>`` (so the token is wiped from memory and
     never written to disk),
   - writes a ``mount.json`` with ``mount_type="sealed"`` so the brain
     knows to fetch the DEK from the keyring at switch-project time.

The owner's master never leaves the owner's machine. Each share gets
its own KEK, so revocation = delete that wrap file (soft) or rotate
the project DEK and re-wrap for everyone except the revoked grantee
(hard — Phase 4).

Wire-level:

- Share token: 32 random bytes, hex-encoded inside ``share_string``.
- KEK derivation: ``HKDF(token, salt=key_id.encode(), info=b"axon-share-v1", length=32)``.
- DEK wrap: AES-KW (RFC 3394, no padding — DEK is exactly 32 bytes
  which is a multiple of 8). Wrapped output = 40 bytes.
- ``share_string`` envelope:
  ``base64.urlsafe_b64encode(b"SEALED1:" + key_id + b":" + token_hex + b":" + owner + b":" + project + b":" + owner_store_path)``
  The ``SEALED1`` prefix lets the redeem path tell sealed shares apart
  from the legacy plaintext-mount shares (which lack the prefix).

Phase 3 deliverable. Replaces the ``generate_sealed_share`` /
``redeem_sealed_share`` stubs in ``axon/security/__init__.py``.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.keywrap import (
        InvalidUnwrap,
        aes_key_unwrap,
        aes_key_wrap,
    )
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.share requires the 'cryptography' package. "
        "Install with: pip install axon-rag[sealed]"
    ) from exc

# Strict filename-safe pattern for key_id. Used both for the on-disk
# wrap filename and the colon-delimited share_string envelope, so we
# refuse path separators (``/``, ``\\``, ``..``), the colon delimiter
# itself, and anything else outside ``[A-Za-z0-9_-]``. Length capped
# to mitigate pathological filename lengths on FAT-style filesystems.
import re as _re

from . import SecurityError
from . import keyring as _kr
from .crypto import DEK_LEN, unwrap_key, wrap_key
from .master import get_master_key, get_project_dek
from .seal import is_project_sealed

_KEY_ID_PATTERN = _re.compile(r"^[A-Za-z0-9_-]{1,64}$")

logger = logging.getLogger("Axon")

__all__ = [
    "SEALED_SHARE_PREFIX",
    "SEALED_SHARE_PREFIX_V2",
    "SHARE_KEYRING_PREFIX",
    "SHARE_DIR_NAME",
    "WRAP_FILENAME_FORMAT",
    "KEK_FILENAME_FORMAT",
    "EXPIRY_FILENAME_FORMAT",
    "generate_sealed_share",
    "redeem_sealed_share",
    "is_sealed_share_envelope",
    "get_grantee_dek",
    "delete_grantee_dek",
    "share_wrap_path",
    "share_kek_path",
    "share_expiry_path",
    "list_sealed_share_key_ids",
    "revoke_sealed_share",
]


def is_sealed_share_envelope(decoded: str) -> bool:
    """Return True if *decoded* is a sealed-share envelope (any version).

    Centralizes the prefix check so callers don't hard-code ``"SEALED1:"``
    or ``"SEALED2:"`` and silently mis-route SEALED2 strings to the
    plaintext path. *decoded* should be the result of base64-decoding
    the user-supplied ``share_string``.
    """
    return decoded.startswith(f"{SEALED_SHARE_PREFIX}:") or decoded.startswith(
        f"{SEALED_SHARE_PREFIX_V2}:"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Legacy 6-field envelope (no signing pubkey). Continues to be accepted
# on redeem indefinitely — older share strings sent before v0.4.0 must
# keep working.
SEALED_SHARE_PREFIX: str = "SEALED1"

# v0.4.0+ envelope with a 7th field carrying the owner's Ed25519 signing
# public key (64 hex chars). The grantee uses this pubkey to verify
# signed metadata sidecars (currently the expiry sidecar in PR B).
# Generated by default whenever the owner's signing key is derivable
# (i.e. store is unlocked).
SEALED_SHARE_PREFIX_V2: str = "SEALED2"
SHARE_KEYRING_PREFIX: str = "axon.share"  # service = "axon.share.<key_id>"
SHARE_DIR_NAME: str = ".security/shares"
WRAP_FILENAME_FORMAT: str = "{key_id}.wrapped"
KEK_FILENAME_FORMAT: str = "{key_id}.kek"
# v0.4.0+: signed expiry sidecar — JSON document at
# ``<project>/.security/shares/<key_id>.expiry`` carrying
# ``{key_id, expires_at (ISO 8601 UTC), sig (Ed25519 over
# "<key_id>:<expires_at_iso>")}``. Generators set this when
# expires_at is supplied; redeem path verifies it on every mount
# and raises ShareExpiredError if elapsed (or sig mismatches).
EXPIRY_FILENAME_FORMAT: str = "{key_id}.expiry"
HKDF_INFO: bytes = b"axon-share-v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_key_id(key_id: str) -> None:
    """Refuse key_ids that could escape paths or break the envelope.
    key_id appears in two places where untrusted input would be
    dangerous:
    1. The on-disk wrap filename ``<project>/.security/shares/<key_id>.wrapped``.
       A crafted ``../foo`` would escape the shares/ directory and let
       a tampered share_string write to / read from arbitrary paths
       under the project root.
    2. The colon-delimited share_string envelope. A key_id containing
       ``:`` would break the parser and let an attacker smuggle
       extra fields.
    Strict pattern: ``[A-Za-z0-9_-]{1,64}``. Mirrors the conventional
    filename-safe charset and bounds length to keep the on-disk
    representation finite.
    """
    if not isinstance(key_id, str) or not key_id:
        raise SecurityError("key_id must be a non-empty string")
    if not _KEY_ID_PATTERN.match(key_id):
        raise SecurityError(
            f"Invalid key_id {key_id!r}: must match [A-Za-z0-9_-]{{1,64}} "
            "(no path separators, no colons, no whitespace)."
        )


def share_wrap_path(project_dir: Path, key_id: str) -> Path:
    """Return the on-disk path of the wrapped DEK for *key_id*."""
    _validate_key_id(key_id)
    return Path(project_dir) / SHARE_DIR_NAME / WRAP_FILENAME_FORMAT.format(key_id=key_id)


def share_expiry_path(project_dir: Path, key_id: str) -> Path:
    """Return the on-disk path of the signed expiry sidecar for *key_id*.

    Layout: ``<project>/.security/shares/<key_id>.expiry`` — a JSON
    file with three fields:

    - ``key_id`` (str)            — same as the filename stem; defends
                                    against rename-attack (an attacker
                                    moving Alice's longer-lived sidecar
                                    onto Bob's key_id would change
                                    neither key_id nor signature, and
                                    the verify step would fail).
    - ``expires_at`` (ISO 8601)   — UTC timestamp, ``Z`` suffix.
    - ``sig`` (base64url)         — Ed25519 signature over the bytes
                                    ``f"{key_id}:{expires_at}".encode()``
                                    using the owner's signing key
                                    (derived from master via HKDF —
                                    see ``axon.security.signing``).

    No sidecar at this path = the share has no TTL; redeem proceeds
    without an expiry check. Once a sidecar exists, the share is
    irrevocably TTL-gated for the rest of its life.
    """
    _validate_key_id(key_id)
    return Path(project_dir) / SHARE_DIR_NAME / EXPIRY_FILENAME_FORMAT.format(key_id=key_id)


def _expiry_signing_message(key_id: str, expires_at_iso: str) -> bytes:
    """Canonical signed payload — must match owner-side write and
    grantee-side verify exactly. Don't include any other fields here
    or older clients will reject newer sidecars."""
    return f"{key_id}:{expires_at_iso}".encode()


def _write_expiry_sidecar(
    project_dir: Path,
    key_id: str,
    expires_at: Any,
    privkey: Any,
) -> Path:
    """Sign + persist an expiry sidecar atomically.

    Args:
        project_dir: Owner's project root (NOT user_dir).
        key_id: Share key identifier.
        expires_at: A timezone-aware ``datetime`` in UTC (or
            convertible). Naive datetimes are rejected.
        privkey: Ed25519PrivateKey from
            :func:`axon.security.signing.derive_signing_keypair`.

    Returns:
        The path the sidecar was written to.

    Raises:
        SecurityError: ``expires_at`` is naive / not a datetime, or
            an I/O error occurred during atomic write.
    """
    from datetime import datetime, timezone

    if not isinstance(expires_at, datetime):
        raise SecurityError(f"expires_at must be a datetime, got {type(expires_at).__name__}")
    if expires_at.tzinfo is None:
        raise SecurityError(
            "expires_at must be timezone-aware (UTC). Naive datetimes "
            "lead to silent off-by-N-hour bugs in cross-timezone shares."
        )
    expires_at_utc = expires_at.astimezone(timezone.utc)
    # Format with explicit ``Z`` suffix so it survives a round-trip
    # through any JSON parser that doesn't preserve ``+00:00``.
    iso = expires_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    sig = privkey.sign(_expiry_signing_message(key_id, iso))
    sidecar = {
        "key_id": key_id,
        "expires_at": iso,
        "sig": base64.urlsafe_b64encode(sig).decode("ascii").rstrip("="),
    }
    target = share_expiry_path(project_dir, key_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".sealing")
    try:
        tmp.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise SecurityError(
            f"Failed to write expiry sidecar for {key_id} at {target}: {exc}"
        ) from exc
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def _check_expiry_or_raise(
    project_dir: Path,
    key_id: str,
    pubkey_hex: str,
) -> None:
    """Verify + check the expiry sidecar for *key_id*. No sidecar = no-op.

    Raises:
        ShareExpiredError: signature failed verification, sidecar is
            malformed, key_id mismatch, or ``now > expires_at``.
            All three failure modes mean the same thing to the caller:
            this share is no longer valid; auto-destroy.
    """
    from datetime import datetime, timezone

    from . import ShareExpiredError as _Expired
    from .signing import pubkey_from_hex

    expiry_path = share_expiry_path(project_dir, key_id)
    if not expiry_path.is_file():
        return  # no TTL on this share
    try:
        sidecar = json.loads(expiry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _Expired(f"Expiry sidecar for {key_id} at {expiry_path} is malformed: {exc}") from exc
    # JSON ``[]``, ``null``, ``42``, etc. all parse as valid JSON but
    # would crash ``.get()`` / ``.replace()`` later. Reject anything
    # that isn't a JSON object up front so the contract "all failures
    # → ShareExpiredError" holds.
    if not isinstance(sidecar, dict):
        raise _Expired(
            f"Expiry sidecar for {key_id} is not a JSON object " f"(got {type(sidecar).__name__})."
        )
    sidecar_key_id = sidecar.get("key_id", "")
    expires_at_iso = sidecar.get("expires_at", "")
    sig_b64url = sidecar.get("sig", "")
    # Each field must be a non-empty string. Non-string types (e.g.
    # ``"expires_at": 42``) would crash later .replace()/.encode()
    # calls; treat as malformed.
    if not (
        isinstance(sidecar_key_id, str)
        and isinstance(expires_at_iso, str)
        and isinstance(sig_b64url, str)
        and sidecar_key_id
        and expires_at_iso
        and sig_b64url
    ):
        raise _Expired(
            f"Expiry sidecar for {key_id} is missing required fields "
            "(key_id, expires_at, sig must all be non-empty strings)."
        )
    # Defend against rename attack: refuse if the embedded key_id
    # doesn't match the filename. An attacker who copied alice's
    # longer-lived sidecar onto bob's key_id would still trip this.
    if sidecar_key_id != key_id:
        raise _Expired(
            f"Expiry sidecar key_id mismatch (file={key_id!r}, "
            f"contents={sidecar_key_id!r}). Possible tamper attempt."
        )
    if not pubkey_hex:
        # SEALED1 mount (no pubkey recorded) — TTL is unenforceable.
        # Treat as expired since we can't verify the sidecar.
        raise _Expired(
            f"Cannot verify expiry sidecar for {key_id}: no signing "
            "pubkey recorded for this mount. Re-redeem the share with a "
            "v0.4.0+ generator (SEALED2 envelope) to enable TTL gating."
        )
    try:
        pubkey = pubkey_from_hex(pubkey_hex)
    except SecurityError as exc:
        raise _Expired(
            f"Mount descriptor for {key_id} carries an invalid signing " f"pubkey: {exc}"
        ) from exc
    # Pad b64url back to a multiple of 4 (we strip ``=`` on write).
    pad = "=" * (-len(sig_b64url) % 4)
    try:
        sig = base64.urlsafe_b64decode(sig_b64url.encode("ascii") + pad.encode())
    except Exception as exc:
        raise _Expired(f"Expiry sidecar for {key_id} has malformed signature: {exc}") from exc
    msg = _expiry_signing_message(key_id, expires_at_iso)
    try:
        pubkey.verify(sig, msg)
    except Exception as exc:
        # cryptography raises InvalidSignature; we treat any verify
        # failure as expired so a tampered sidecar can't extend access.
        raise _Expired(
            f"Expiry sidecar signature for {key_id} failed verification: "
            "the file may have been tampered with, or the share was "
            "minted by a different owner key. Treating as expired."
        ) from exc
    # Parse the timestamp — accept both ``Z`` and ``+00:00`` forms.
    try:
        expires_at = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _Expired(
            f"Expiry sidecar expires_at {expires_at_iso!r} is not valid " f"ISO 8601: {exc}"
        ) from exc
    if expires_at.tzinfo is None:
        # Defend against a sidecar that somehow shed its timezone.
        raise _Expired(
            f"Expiry sidecar expires_at {expires_at_iso!r} is naive (no "
            "timezone). Treating as expired to avoid silent off-by-hour bugs."
        )
    if datetime.now(timezone.utc) > expires_at:
        raise _Expired(
            f"Sealed share {key_id} expired at {expires_at_iso}. "
            "Request a fresh share from the owner."
        )


def share_kek_path(project_dir: Path, key_id: str) -> Path:
    """Return the on-disk path of the master-wrapped KEK for *key_id*.
    Owner-only: this file holds the per-share KEK encrypted under the
    OWNER'S MASTER (NOT the share token), so the owner can re-derive
    the KEK at hard-revoke time without needing the share token (which
    only the grantee has). Enables selective re-wrap on hard revoke
    instead of forcing every surviving grantee to re-redeem.
    """
    _validate_key_id(key_id)
    return Path(project_dir) / SHARE_DIR_NAME / KEK_FILENAME_FORMAT.format(key_id=key_id)


def _persist_share_kek(project_dir: Path, key_id: str, kek: bytes, master: bytes) -> Path:
    """Wrap *kek* under *master* (AES-KW) and persist atomically.
    Mirrors the same envelope pattern used for the project DEK in
    ``master.py`` (40-byte AES-KW output for a 32-byte plaintext key).
    Used by :func:`generate_sealed_share` so the owner can re-wrap the
    DEK for surviving grantees during a hard revoke without ever
    needing the share token back.
    """
    if len(kek) != 32:
        raise SecurityError(f"share KEK must be 32 bytes, got {len(kek)}")
    if len(master) != 32:
        raise SecurityError(f"master key must be 32 bytes, got {len(master)}")
    path = share_kek_path(project_dir, key_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = aes_key_wrap(master, kek)
    tmp = path.with_suffix(path.suffix + ".sealing")
    try:
        tmp.write_bytes(wrapped)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise SecurityError(f"Failed to persist share KEK at {path}: {exc}") from exc
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _load_share_kek(project_dir: Path, key_id: str, master: bytes) -> bytes | None:
    """Load + unwrap the persisted KEK for *key_id*; None if absent.
    Returns ``None`` (not raises) when the ``.kek`` file is missing,
    so callers can fall back to the legacy "all-shares-invalidated"
    behaviour for projects created before per-share KEK persistence
    was added. Raises :class:`SecurityError` only when the file exists
    but is unreadable / malformed / won't unwrap.
    """
    if len(master) != 32:
        raise SecurityError(f"master key must be 32 bytes, got {len(master)}")
    path = share_kek_path(project_dir, key_id)
    if not path.is_file():
        return None
    try:
        wrapped = path.read_bytes()
    except OSError as exc:
        raise SecurityError(f"Failed to read persisted share KEK at {path}: {exc}") from exc
    try:
        kek = aes_key_unwrap(master, wrapped)
    except InvalidUnwrap as exc:
        raise SecurityError(
            f"Persisted share KEK at {path} won't unwrap with the current "
            "master. Project may have been sealed under a different owner."
        ) from exc
    except ValueError as exc:
        raise SecurityError(
            f"Persisted share KEK at {path} is malformed ({exc}); "
            "the file may not have finished syncing."
        ) from exc
    if len(kek) != 32:
        raise SecurityError(
            f"Persisted share KEK at {path} unwrapped to {len(kek)} bytes "
            "(expected 32); KEK file may be corrupted."
        )
    return kek


def _derive_share_kek(token_bytes: bytes, key_id: str) -> bytes:
    """HKDF-SHA256 → 32-byte KEK from share token + key_id."""
    if len(token_bytes) != 32:
        raise ValueError(f"share token must be 32 bytes, got {len(token_bytes)}")
    if not key_id:
        raise ValueError("key_id must be a non-empty string")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=key_id.encode("utf-8"),
        info=HKDF_INFO,
    ).derive(token_bytes)


def _resolve_owned_project_dir(project_name: str, user_dir: Path) -> Path:
    """Mirror the AxonStore ``subs/`` layout for nested projects."""
    segments = project_name.split("/")
    project_dir = Path(user_dir) / segments[0]
    for seg in segments[1:]:
        project_dir = project_dir / "subs" / seg
    return project_dir


def _share_keyring_service(key_id: str) -> str:
    """Per-share keyring service name for the grantee's unwrapped DEK."""
    return f"{SHARE_KEYRING_PREFIX}.{key_id}"


# ---------------------------------------------------------------------------
# Grantee DEK file fallback helpers (headless Linux / Docker / CI)
# ---------------------------------------------------------------------------


def _grantee_dek_fallback_path(user_dir: Path, key_id: str) -> Path:
    """``<user_dir>/.security/shares/<key_id>.dek.wrapped`` — per-share fallback."""
    _validate_key_id(key_id)
    return Path(user_dir) / ".security" / "shares" / f"{key_id}.dek.wrapped"


def _write_grantee_dek_fallback(user_dir: Path, key_id: str, dek: bytes) -> None:
    """Wrap *dek* under the grantee's master key and persist to file atomically.

    Mirrors the pattern used for project DEKs in ``master.py``:
    ``<project>/.security/dek.wrapped`` (40-byte AES-KW output).  The
    grantee must have their store unlocked (``axon --store-unlock``) before
    this is called, because :func:`get_master_key` will raise
    ``SecurityError`` otherwise.
    """
    master = get_master_key(user_dir)
    wrapped = wrap_key(dek, master)
    path = _grantee_dek_fallback_path(user_dir, key_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(wrapped)
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    try:
        import os as _os

        _os.chmod(path, 0o600)
    except OSError:
        pass


def _read_grantee_dek_fallback(user_dir: Path, key_id: str) -> bytes | None:
    """Unwrap the DEK from the file fallback using the grantee's master key.

    Returns ``None`` when the fallback file does not exist (analogous to
    :func:`axon.security.keyring.get_secret` returning None).  Raises
    :class:`SecurityError` when the file exists but can't be decrypted.
    """
    path = _grantee_dek_fallback_path(user_dir, key_id)
    if not path.is_file():
        return None
    master = get_master_key(user_dir)
    try:
        return unwrap_key(path.read_bytes(), master)
    except Exception as exc:
        raise SecurityError(f"Grantee DEK fallback file is unreadable: {exc}") from exc


# ---------------------------------------------------------------------------
# generate_sealed_share (owner side)
# ---------------------------------------------------------------------------


def generate_sealed_share(
    owner_user_dir: Path,
    project: str,
    grantee: str,
    key_id: str,
    *,
    expires_at: Any = None,
) -> dict[str, Any]:
    """Mint a per-share KEK, wrap the DEK under it, return a share envelope.
    Args:
        owner_user_dir: Owner's AxonStore user directory.
        project: Name of the sealed project to share (must be sealed).
        grantee: OS username of the recipient (recorded for audit; not
            cryptographically bound — share_strings are bearer tokens).
        key_id: Caller-supplied key identifier (e.g. ``ssk_a1b2c3d4``).
            Used as the HKDF salt + filename of the wrap file. Must be
            unique per share — re-using a key_id collides on disk.
        expires_at: Optional timezone-aware UTC datetime. When set,
            writes a signed expiry sidecar at
            ``<project>/.security/shares/<key_id>.expiry``. Grantees
            running v0.4.0+ verify the signature on every mount and
            auto-destroy the share once the timestamp elapses. Naive
            datetimes are rejected to prevent silent off-by-N-hour
            bugs across timezones.
    Returns:
        ``{"key_id": ..., "share_string": ..., "wrapped_path": ...,
        "owner": ..., "project": ..., "grantee": ...}``
    Raises:
        SecurityError: project is not sealed (run ``project_seal`` first),
            store is locked, wrap file already exists for *key_id*, or
            any I/O error occurred.
    """
    _validate_key_id(key_id)
    owner_user_dir = Path(owner_user_dir).resolve()
    project_dir = _resolve_owned_project_dir(project, owner_user_dir)
    if not project_dir.is_dir():
        raise SecurityError(
            f"Project '{project}' does not exist at {project_dir}. "
            "Create + seal it before generating a sealed share."
        )
    if not is_project_sealed(project_dir):
        raise SecurityError(
            f"Project '{project}' is not sealed. Run "
            f"``axon --project-seal {project}`` first to encrypt at rest."
        )
    wrap_path = share_wrap_path(project_dir, key_id)
    if wrap_path.exists():
        raise SecurityError(
            f"Share wrap already exists at {wrap_path}. "
            f"Pick a different key_id or revoke the existing share first."
        )
    # Read-only DEK lookup. Raises SecurityError if store is locked or
    # if dek.wrapped is missing (sealed marker without DEK = corrupted
    # state worth surfacing rather than silently minting).
    dek = get_project_dek(owner_user_dir, project_dir)
    if len(dek) != DEK_LEN:
        raise SecurityError(
            f"Project DEK is {len(dek)} bytes (expected {DEK_LEN}); "
            "the wrapped DEK file may be corrupted."
        )
    # Mint the share token + derive the per-share KEK.
    token_bytes = secrets.token_bytes(32)
    kek = _derive_share_kek(token_bytes, key_id)
    wrapped_dek = aes_key_wrap(kek, dek)  # 40 bytes for a 32-byte DEK
    # Persist the KEK encrypted under the OWNER'S MASTER so the owner
    # can re-wrap the DEK for surviving grantees during a hard revoke
    # without ever needing the share token back. Done BEFORE writing
    # the wrap file so a crash here doesn't strand a wrap without its
    # KEK sidecar (a wrap without a KEK falls back to "all
    # invalidated" on hard revoke; a KEK without a wrap is harmless).
    master = get_master_key(owner_user_dir)
    _persist_share_kek(project_dir, key_id, kek, master)
    # Atomic write of the wrap file.
    wrap_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wrap_path.with_suffix(wrap_path.suffix + ".sealing")
    try:
        tmp.write_bytes(wrapped_dek)
        os.replace(tmp, wrap_path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise SecurityError(f"Failed to write sealed-share wrap to {wrap_path}: {exc}") from exc
    # Lock down (no-op on Windows; defence-in-depth on POSIX).
    try:
        os.chmod(wrap_path, 0o600)
    except OSError:
        pass
    owner_name = owner_user_dir.name
    owner_store_path = str(owner_user_dir.parent)
    token_hex = token_bytes.hex()
    # Embed the owner's Ed25519 signing pubkey so the grantee can verify
    # signed metadata sidecars (expiry, future fields) without trusting
    # the sync path. Master is already in memory (we just used it to
    # wrap the KEK above), so deriving the keypair adds no I/O.
    from .signing import derive_signing_keypair, pubkey_to_hex

    privkey, pubkey = derive_signing_keypair(master)
    pubkey_hex = pubkey_to_hex(pubkey)
    raw = (
        f"{SEALED_SHARE_PREFIX_V2}:{key_id}:{token_hex}:"
        f"{owner_name}:{project}:{owner_store_path}:{pubkey_hex}"
    )
    share_string = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    # Write the signed expiry sidecar AFTER the wrap file is in place
    # but BEFORE we return the share_string. If sidecar write fails, the
    # whole generate_sealed_share call fails — better to surface the
    # error than to mint a TTL-less share when one was requested.
    expiry_path: Path | None = None
    expires_at_iso: str | None = None
    if expires_at is not None:
        expiry_path = _write_expiry_sidecar(project_dir, key_id, expires_at, privkey)
        # Re-serialize the same way as the sidecar so the audit log
        # uses the canonical form (ISO 8601 UTC with ``Z``).
        from datetime import datetime, timezone

        if isinstance(expires_at, datetime):
            expires_at_iso = expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(
        "Generated sealed share key_id=%s project=%s grantee=%s wrapped_path=%s envelope=SEALED2 expires_at=%s",
        key_id,
        project,
        grantee,
        wrap_path,
        expires_at_iso or "never",
    )
    result: dict[str, Any] = {
        "key_id": key_id,
        "share_string": share_string,
        "wrapped_path": str(wrap_path),
        "owner": owner_name,
        "project": project,
        "grantee": grantee,
        "sealed": True,
    }
    if expires_at_iso is not None:
        result["expires_at"] = expires_at_iso
        result["expiry_sidecar_path"] = str(expiry_path)
    return result


# ---------------------------------------------------------------------------
# redeem_sealed_share (grantee side)
# ---------------------------------------------------------------------------


def redeem_sealed_share(
    grantee_user_dir: Path,
    share_string: str,
) -> dict[str, Any]:
    """Parse a sealed share_string, unwrap the DEK, persist it in the keyring.
    Args:
        grantee_user_dir: Grantee's AxonStore user directory.
        share_string: The base64 envelope generated by
            :func:`generate_sealed_share`. Both ``SEALED1:`` (legacy,
            6 fields) and ``SEALED2:`` (v0.4.0+, 7 fields with embedded
            owner signing pubkey) are accepted after base64 decode.
            Generators emit SEALED2 by default; SEALED1 strings sent
            before the upgrade keep redeeming.
    Returns:
        ``{"key_id": ..., "mount_name": ..., "owner": ..., "project": ...,
        "descriptor": <mount_descriptor>, "envelope_version": 1|2}``
    Raises:
        SecurityError: share_string malformed, wrap file missing, KEK
            derivation produces an invalid DEK (token corruption), or
            I/O error writing the mount descriptor. SEALED2 strings
            with a malformed pubkey field also raise here.
        ValueError: share_string is a non-sealed legacy share (let the
            caller route to ``axon.shares.redeem_share_key`` instead).
    """
    grantee_user_dir = Path(grantee_user_dir).resolve()
    try:
        raw = base64.urlsafe_b64decode(share_string.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise SecurityError(f"Invalid sealed share_string format: {exc}") from exc
    # Auto-detect SEALED1 (legacy, 6 fields) vs SEALED2 (v0.4.0+, 7
    # fields with signing pubkey). Both formats remain accepted; new
    # generators emit SEALED2 by default but old share strings sent
    # before v0.4.0 must keep redeeming.
    if raw.startswith(f"{SEALED_SHARE_PREFIX_V2}:"):
        envelope_version = 2
        # Split off the trailing pubkey FIRST. The owner_store_path can
        # legitimately contain a colon on Windows (e.g. "C:\Users\...")
        # so a left-side split with a fixed count would slice the pubkey
        # into the path. The pubkey is always the trailing colon-segment.
        before_pubkey, _sep, pubkey_hex = raw.rpartition(":")
        if not _sep:
            raise SecurityError("SEALED2 share_string has no trailing pubkey field")
        parts = before_pubkey.split(":", 5)
        if len(parts) != 6:
            raise SecurityError(
                f"SEALED2 share_string has wrong shape (expected 6 colon-separated "
                f"fields before the pubkey, got {len(parts)})"
            )
        _prefix, key_id, token_hex, owner, project, owner_store_path = parts
        # Validate the pubkey early — a malformed SEALED2 string should
        # fail before we touch the wrap file. We don't *use* the pubkey
        # in PR A; PR B will use it for expiry sidecar verification.
        from .signing import pubkey_from_hex

        _ = pubkey_from_hex(pubkey_hex)
    elif raw.startswith(f"{SEALED_SHARE_PREFIX}:"):
        envelope_version = 1
        pubkey_hex = ""  # SEALED1 has no pubkey
        parts = raw.split(":", 5)
        if len(parts) != 6:
            raise SecurityError(
                f"SEALED1 share_string has wrong shape (expected 6 colon-separated "
                f"fields after the prefix, got {len(parts)})"
            )
        _prefix, key_id, token_hex, owner, project, owner_store_path = parts
    else:
        # Legacy plaintext-mount share — caller should route to the
        # non-sealed redeem path instead.
        raise ValueError(
            "share_string is not a sealed share. "
            "Route to axon.shares.redeem_share_key() for legacy shares."
        )
    if not token_hex or not owner or not project:
        raise SecurityError("Sealed share_string has empty token / owner / project")
    # key_id from a share_string is bearer-token data — validate before
    # we use it to build paths or keyring service names.
    _validate_key_id(key_id)
    try:
        token_bytes = bytes.fromhex(token_hex)
    except ValueError as exc:
        raise SecurityError(f"Sealed share_string token is not valid hex: {exc}") from exc
    if len(token_bytes) != 32:
        raise SecurityError(f"Sealed share token must be 32 bytes, got {len(token_bytes)}")
    owner_user_dir = Path(owner_store_path) / owner
    owner_project_dir = _resolve_owned_project_dir(project, owner_user_dir)
    if not owner_project_dir.is_dir():
        raise SecurityError(
            f"Owner's project directory does not exist at {owner_project_dir}. "
            "The owner's AxonStore folder may not be synced yet — wait for "
            "OneDrive/Dropbox to finish, then retry."
        )
    wrap_path = share_wrap_path(owner_project_dir, key_id)
    if not wrap_path.is_file():
        raise SecurityError(
            f"Sealed-share wrap file missing at {wrap_path}. "
            "Either the owner has revoked this share, or the file has not "
            "yet synced to your machine."
        )
    # Derive the KEK and unwrap the DEK. Wipe the token from local
    # variables ASAP — once the DEK is in the keyring there's no need
    # to keep the token in memory.
    try:
        kek = _derive_share_kek(token_bytes, key_id)
        dek = aes_key_unwrap(kek, wrap_path.read_bytes())
    except InvalidUnwrap as exc:
        raise SecurityError(
            f"Wrap file at {wrap_path} won't unwrap with the share token. "
            "Either the share_string is for a different key_id, or the "
            "owner has rotated the project DEK (hard revocation)."
        ) from exc
    except ValueError as exc:
        # aes_key_unwrap raises ValueError on truncated / wrong-length
        # ciphertext (e.g. partial OneDrive sync delivered a half-file).
        raise SecurityError(
            f"Wrap file at {wrap_path} is malformed ({exc}); the file "
            "may not have finished syncing. Wait for the sync to complete "
            "and retry."
        ) from exc
    except OSError as exc:
        raise SecurityError(f"Failed to read wrap file at {wrap_path}: {exc}") from exc
    finally:
        # Best-effort wipe — Python doesn't guarantee these are zeroed
        # before GC, but at least the names no longer reference them.
        token_bytes = b""  # noqa: F841
        kek = b""  # noqa: F841
    if len(dek) != DEK_LEN:
        raise SecurityError(
            f"Unwrapped DEK is {len(dek)} bytes (expected {DEK_LEN}); "
            "share material may be corrupted."
        )
    # Persist the DEK in the grantee's OS keyring. The wrap file on
    # the synced disk is now sufficient to re-derive on another
    # grantee machine, but on this machine we cache the plaintext DEK
    # so query latency isn't gated on a keyring round-trip per file.
    service = _share_keyring_service(key_id)
    # v0.4.0 Item 2: respect security.keyring_mode. In session/never
    # modes the user has explicitly opted out of persistent DEK storage,
    # so the file fallback (which IS persistent on disk) must NOT fire.
    # ``store_secret`` already routes session→memory and never→drop;
    # the only mode where a real OS keyring write happens — and where
    # KeyringUnavailableError is the legitimate signal to engage the
    # encrypted-on-disk fallback — is "persistent".
    _km = _kr.get_keyring_mode()
    try:
        _kr.store_secret(service, "dek", base64.b64encode(dek).decode("ascii"))
    except _kr.KeyringUnavailableError:
        # Keyring unavailable (headless Linux / Docker / CI) — fall back to
        # a file wrapped under the grantee's own master key, mirroring the
        # master.enc dual-write pattern in master.py. Only triggered in
        # persistent mode (session/never don't raise this).
        try:
            _write_grantee_dek_fallback(grantee_user_dir, key_id, dek)
            logger.info(
                "Grantee DEK for %s persisted to file fallback (keyring unavailable)",
                key_id,
            )
        except SecurityError:
            raise  # store not bootstrapped or locked — surface as-is
        except OSError as exc:
            raise SecurityError(
                f"Cannot persist grantee DEK for {key_id}: keyring unavailable and "
                f"file fallback failed: {exc}"
            ) from exc
    else:
        if _km != "persistent":
            # session / never: store_secret succeeded by routing to the
            # in-memory cache or by silently dropping the secret. Writing
            # the disk fallback here would defeat the point of the mode.
            # Skip the fallback but still build the mount descriptor below.
            logger.debug(
                "keyring_mode=%s: skipping grantee DEK file fallback for %s",
                _km,
                key_id,
            )
        else:
            # Persistent + keyring available — write the file fallback so
            # the DEK survives a cross-platform copy (mirrors the
            # master.enc dual-write).
            try:
                _write_grantee_dek_fallback(grantee_user_dir, key_id, dek)
            except Exception as _fb_exc:
                # File fallback is best-effort when keyring is available;
                # a failure here should not abort a successful keyring write.
                logger.debug(
                    "Grantee DEK file fallback write failed (keyring write succeeded): %s",
                    _fb_exc,
                )
    # Build the mount descriptor — same path format as the legacy
    # share path so the rest of the brain doesn't need a special case
    # at list-mounts / list-projects time.
    mount_name = f"{owner}_{project.replace('/', '_')}"
    descriptor = _create_sealed_mount_descriptor(
        grantee_user_dir=grantee_user_dir,
        mount_name=mount_name,
        owner=owner,
        project=project,
        owner_user_dir=owner_user_dir,
        target_project_dir=owner_project_dir,
        share_key_id=key_id,
        envelope_version=envelope_version,
        owner_pubkey_hex=pubkey_hex,
    )
    logger.info(
        "Redeemed sealed share key_id=%s owner=%s project=%s mount=%s envelope=SEALED%d",
        key_id,
        owner,
        project,
        mount_name,
        envelope_version,
    )
    return {
        "key_id": key_id,
        "mount_name": mount_name,
        "owner": owner,
        "project": project,
        "descriptor": descriptor,
        "sealed": True,
        "envelope_version": envelope_version,
    }


def _create_sealed_mount_descriptor(
    *,
    grantee_user_dir: Path,
    mount_name: str,
    owner: str,
    project: str,
    owner_user_dir: Path,
    target_project_dir: Path,
    share_key_id: str,
    envelope_version: int = 1,
    owner_pubkey_hex: str = "",
) -> dict[str, Any]:
    """Write a mount.json carrying the ``mount_type="sealed"`` flag.
    Builds the descriptor via :func:`axon.mounts.create_mount_descriptor`
    so the canonical schema (state, revoked, descriptor_version,
    project_id, store_id, redeemed_at) is preserved — without that,
    :func:`axon.mounts.validate_mount_descriptor` rejects the mount and
    ``axon.mounts.list_mount_descriptors`` filters it out. The sealed
    flag and pointer to the share key are added afterward + persisted
    atomically (the mounts module's plain write isn't atomic).

    Args:
        envelope_version: ``1`` for legacy SEALED1 share strings, ``2``
            for v0.4.0+ SEALED2. Recorded so future TTL checks know
            whether to expect a signed expiry sidecar (only SEALED2).
        owner_pubkey_hex: 64-char hex Ed25519 public key from the
            SEALED2 envelope, ``""`` for SEALED1. Persisted in the
            descriptor so the grantee can verify expiry sidecars
            without re-parsing the original share string.
    """
    from axon.mounts import create_mount_descriptor, mount_descriptor_path

    try:
        descriptor = create_mount_descriptor(
            grantee_user_dir=grantee_user_dir,
            mount_name=mount_name,
            owner=owner,
            project=project,
            owner_user_dir=owner_user_dir,
            target_project_dir=target_project_dir,
            share_key_id=share_key_id,
        )
    except OSError as exc:
        raise SecurityError(
            f"Failed to create mount descriptor for sealed mount " f"{mount_name!r}: {exc}"
        ) from exc
    # Layer in the sealed-specific fields and persist atomically.
    descriptor["mount_type"] = "sealed"
    descriptor["envelope_version"] = envelope_version
    if owner_pubkey_hex:
        descriptor["owner_pubkey_hex"] = owner_pubkey_hex
    target = mount_descriptor_path(grantee_user_dir, mount_name)
    tmp = target.with_suffix(target.suffix + ".sealing")
    try:
        tmp.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise SecurityError(
            f"Failed to persist sealed mount descriptor at {target}: {exc}"
        ) from exc
    return descriptor


# ---------------------------------------------------------------------------
# Grantee DEK lookup (used at switch_project time)
# ---------------------------------------------------------------------------


def _find_sealed_mount(user_dir: Path, key_id: str) -> dict[str, Any] | None:
    """Return the sealed-mount descriptor matching *key_id*, or ``None``."""
    try:
        from axon.mounts import list_mount_descriptors
    except ImportError:
        return None
    for desc in list_mount_descriptors(user_dir):
        if desc.get("mount_type") == "sealed" and desc.get("share_key_id") == key_id:
            return desc
    return None


def get_grantee_dek(key_id: str, user_dir: Path | None = None) -> bytes:
    """Fetch the grantee's cached DEK for *key_id*, after TTL check.

    v0.4.0+: when *user_dir* is provided AND a sealed-mount descriptor
    exists for *key_id* AND that descriptor records the owner's signing
    pubkey (i.e. the share was redeemed from a SEALED2 envelope), this
    function consults the signed expiry sidecar at
    ``<owner-project>/.security/shares/<key_id>.expiry`` BEFORE returning
    the DEK. Tampered sidecars, malformed signatures, key_id mismatch,
    or ``now > expires_at`` all raise :class:`ShareExpiredError`. Callers
    (notably :meth:`AxonBrain._mount_sealed_project`) catch the exception
    and trigger the auto-destroy flow.

    Resolution order (after expiry check):
    1. OS keyring (``axon.share.<key_id>`` / ``dek``).
    2. File fallback at ``<user_dir>/.security/shares/<key_id>.dek.wrapped``
       (used when keyring is unavailable — headless Linux / Docker / CI).

    Args:
        key_id: Share key identifier.
        user_dir: Grantee's AxonStore user directory.  Required when the
            keyring is unavailable (supplies the master key path for the
            file fallback). Also required for TTL enforcement — without
            it we can't locate the mount descriptor or the synced
            expiry sidecar. When ``None`` and the keyring is available,
            the DEK is returned without an expiry check (callers that
            care about TTL must pass *user_dir*).

    Raises:
        ShareExpiredError: signed expiry sidecar shows the share is
            elapsed, the signature failed verification, the embedded
            key_id doesn't match the filename, or the mount records no
            signing pubkey (TTL unenforceable). Subclass of SecurityError.
        SecurityError: keyring unavailable and *user_dir* not given (or
            file fallback also absent), no DEK stored for *key_id*, or
            the stored value is malformed.
    """
    _validate_key_id(key_id)
    # TTL gate. The check is best-effort — if user_dir is None, no
    # mount exists for this key_id, or the mount predates SEALED2,
    # the gate falls open (and we surface a SecurityError later if
    # the sidecar exists but pubkey is missing — see
    # ``_check_expiry_or_raise``). All actual decisions live there.
    if user_dir is not None:
        mount = _find_sealed_mount(Path(user_dir), key_id)
        if mount is not None:
            target = mount.get("target_project_dir", "")
            pubkey_hex = mount.get("owner_pubkey_hex", "")
            if target:
                # Note: ``_check_expiry_or_raise`` is a no-op when the
                # expiry sidecar doesn't exist — most shares (TTL-less)
                # pay zero cost on this path.
                _check_expiry_or_raise(Path(target), key_id, pubkey_hex)
    service = _share_keyring_service(key_id)
    try:
        secret = _kr.get_secret(service, "dek")
    except _kr.KeyringUnavailableError:
        # Keyring unavailable — try the file fallback.
        if user_dir is None:
            raise SecurityError(
                f"OS keyring unavailable; cannot fetch sealed-share DEK for "
                f"{key_id}. Pass user_dir to use the file fallback."
            )
        dek = _read_grantee_dek_fallback(Path(user_dir), key_id)
        if dek is None:
            raise SecurityError(
                f"Share DEK for {key_id!r} not found in keyring or file fallback. "
                "Redeem the share again, or ensure the grantee store is bootstrapped "
                "and unlocked (axon --store-unlock) before querying."
            )
        if len(dek) != DEK_LEN:
            raise SecurityError(
                f"Grantee DEK fallback for {key_id} is {len(dek)} bytes "
                f"(expected {DEK_LEN}); file may be corrupted."
            )
        return dek
    if secret is None:
        raise SecurityError(
            f"No sealed-share DEK in keyring for {key_id}. "
            "Did you run `axon share redeem ...` on this machine?"
        )
    try:
        dek = base64.b64decode(secret.encode("ascii"))
    except Exception as exc:
        raise SecurityError(
            f"Sealed-share DEK in keyring for {key_id} is malformed: {exc}"
        ) from exc
    if len(dek) != DEK_LEN:
        raise SecurityError(
            f"Sealed-share DEK in keyring for {key_id} is {len(dek)} bytes "
            f"(expected {DEK_LEN}); keyring entry may be corrupted."
        )
    return dek


# ---------------------------------------------------------------------------
# Soft + hard revocation (Phase 4)
# ---------------------------------------------------------------------------


def list_sealed_share_key_ids(project_dir: Path | str) -> list[str]:
    """Return key_ids of every active sealed-share wrap under *project_dir*.
    Walks ``<project>/.security/shares/`` and parses ``.wrapped`` filenames.
    Filters out anything whose stem doesn't match the strict key_id
    pattern (``[A-Za-z0-9_-]{1,64}``) so sync-engine artifacts like
    ``<wrap>-OneDrive-MachineB.conflict.wrapped`` (OneDrive),
    ``<wrap> (1).wrapped`` (Dropbox), or ``<wrap>.tmp.drivedownload``
    (Google Drive) don't pollute the listing.
    Used by ``hard_revoke`` to enumerate which shares need to be re-issued
    after a DEK rotation, and by ``list_sealed_shares`` for owner-side
    audit output.
    """
    shares_dir = Path(project_dir) / SHARE_DIR_NAME
    if not shares_dir.is_dir():
        return []
    suffix = ".wrapped"
    out: list[str] = []
    for p in shares_dir.iterdir():
        if not p.is_file() or not p.name.endswith(suffix):
            continue
        stem = p.stem
        if _KEY_ID_PATTERN.match(stem):
            out.append(stem)
        else:
            logger.debug(
                "Ignoring non-key_id wrap-shaped file under %s: %s",
                shares_dir,
                p.name,
            )
    return sorted(out)


def revoke_sealed_share(
    owner_user_dir: Path,
    project: str,
    key_id: str,
    *,
    rotate: bool = False,
) -> dict[str, Any]:
    """Revoke a sealed share — soft (default) or hard (``rotate=True``).
    **Soft (``rotate=False``, default)** — fast, surface-only:
    - Delete ``<project>/.security/shares/<key_id>.wrapped`` so a fresh
      ``redeem_sealed_share`` for this key_id fails.
    - Caveat: a grantee who already redeemed and cached the DEK in
      their OS keyring CAN still decrypt files synced before the
      revocation. Cached bytes stay decryptable; this is documented
      and covered by the "soft vs hard" trade-off.
    **Hard (``rotate=True``)** — slow, breaks the revoked grantee's cached DEK:
    - Generate a fresh project DEK + a fresh ``seal_id``.
    - Re-encrypt every content file under the project with the new
      DEK + AAD (``make_aad(new_seal_id, rel)``).
    - Update ``.security/dek.wrapped`` (master-wrapped form).
    - Update the sealed marker with the new ``seal_id``.
    - **Selective re-wrap** for surviving shares: for each other
      active share, load the per-share KEK persisted at
      ``.security/shares/<key_id>.kek``, re-wrap the new DEK under
      that KEK, and atomically replace ``.security/shares/<key_id>.wrapped``.
      Surviving grantees can redeem the new wrap without re-issuing.
    - Legacy fallback: if a surviving share has no ``.kek`` file
      (project predates per-share KEK persistence), that share is
      invalidated and listed in ``invalidated_share_key_ids``.
    - Returns the list of key_ids in ``invalidated_share_key_ids``
      so the caller can re-issue them (only the revoked share +
      any legacy-fallback shares — NOT surviving shares that were
      re-wrapped successfully).
    Args:
        owner_user_dir: Owner's AxonStore user directory.
        project: Sealed project the share belongs to.
        key_id: Share key_id to revoke. For soft revoke this names the
            specific wrap to delete; for hard revoke it's recorded in
            the result for audit but every share is invalidated anyway.
        rotate: When True, do the hard rotation. Default False (soft).
    Returns:
        ``{"status": "soft_revoked" | "hard_revoked", "key_id": ...,
        "rotate": bool, "wrap_deleted": bool,
        "invalidated_share_key_ids": list[str] (hard only),
        "files_resealed": int (hard only),
        "new_seal_id": str (hard only)}``
    Raises:
        SecurityError: project not sealed, store locked (hard rotate
            needs to read + re-write the DEK), wrap missing (soft when
            ``key_id`` doesn't exist), or any I/O / decryption error
            during the rotate.
    """
    _validate_key_id(key_id)
    owner_user_dir = Path(owner_user_dir).resolve()
    project_dir = _resolve_owned_project_dir(project, owner_user_dir)
    if not project_dir.is_dir():
        raise SecurityError(f"Project '{project}' does not exist at {project_dir}; cannot revoke.")
    if not is_project_sealed(project_dir):
        raise SecurityError(f"Project '{project}' is not sealed; nothing to revoke.")
    if not rotate:
        return _soft_revoke(project_dir, key_id)
    return _hard_revoke(owner_user_dir, project, project_dir, key_id)


def _soft_revoke(project_dir: Path, key_id: str) -> dict[str, Any]:
    """Delete the wrap file for *key_id* — fresh redeem will fail.
    Also deletes the persisted ``.kek`` sidecar (if present) so the
    revoked share's KEK isn't kept around indefinitely. Best-effort:
    a missing KEK file is fine.
    """
    wrap = share_wrap_path(project_dir, key_id)
    if not wrap.is_file():
        raise SecurityError(
            f"No sealed-share wrap exists for key_id={key_id!r} at {wrap}. "
            "Either it was already revoked or never generated."
        )
    try:
        wrap.unlink()
    except OSError as exc:
        raise SecurityError(f"Failed to delete sealed-share wrap at {wrap}: {exc}") from exc
    # Best-effort: also drop the persisted KEK sidecar.
    kek_path = share_kek_path(project_dir, key_id)
    if kek_path.is_file():
        try:
            kek_path.unlink()
        except OSError as exc:
            logger.debug("Could not delete share KEK sidecar %s: %s", kek_path, exc)
    logger.info("Soft-revoked sealed share key_id=%s under %s", key_id, project_dir)
    return {
        "status": "soft_revoked",
        "key_id": key_id,
        "rotate": False,
        "wrap_deleted": True,
    }


# ---------------------------------------------------------------------------
# Crash-safe hard-revoke staging
# ---------------------------------------------------------------------------
#
# Hard revoke is a multi-step operation: generate new DEK → re-encrypt every
# content file → swap in the new master-wrapped DEK → write new sealed
# marker → delete share wraps. A crash anywhere in the middle without staged
# state would leave files encrypted under a key that was never persisted —
# permanently undecryptable.
#
# Strategy: persist the new master-wrapped DEK + a rotation marker BEFORE
# touching any content file. Each file is rotated atomically (tempfile +
# os.replace), and reads during rotation may transiently see a mix of
# old/new ciphertext (the resume helper handles both). On completion, the
# staged DEK is promoted over the live one, the sealed marker is updated,
# and the rotation context is cleared.
#
# Resume helper (``_resume_rotation_if_needed``): if a prior rotation
# crashed, the rotation marker survives. The next ``revoke_sealed_share(...,
# rotate=True)`` call detects it, recovers the new DEK + new seal_id, and
# resumes the per-file loop. Each file is decrypted with whichever (DEK,
# seal_id) pair works (old or new), then re-encrypted with the new pair.

_ROTATION_DEK_FILENAME = "dek.wrapped.rotating"
_ROTATION_MARKER_FILENAME = ".sealing.rotation"


def _rotation_dek_path(project_dir: Path) -> Path:
    return project_dir / ".security" / _ROTATION_DEK_FILENAME


def _rotation_marker_path(project_dir: Path) -> Path:
    return project_dir / ".security" / _ROTATION_MARKER_FILENAME


def _stage_rotation(
    project_dir: Path,
    *,
    master: bytes,
    new_dek: bytes,
    old_seal_id: str,
    new_seal_id: str,
) -> None:
    """Persist new DEK + rotation marker BEFORE any file is rotated.
    Atomic per-file (tempfile + os.replace). After both files land, a
    crash anywhere in the rotation loop is recoverable: the marker
    plus the staged DEK contain everything needed to resume.
    """
    from cryptography.hazmat.primitives.keywrap import aes_key_wrap as _wrap

    sec_dir = project_dir / ".security"
    sec_dir.mkdir(parents=True, exist_ok=True)
    dek_path = _rotation_dek_path(project_dir)
    marker_path = _rotation_marker_path(project_dir)
    dek_tmp = dek_path.with_suffix(dek_path.suffix + ".write")
    marker_tmp = marker_path.with_suffix(marker_path.suffix + ".write")
    try:
        dek_tmp.write_bytes(_wrap(master, new_dek))
        os.replace(dek_tmp, dek_path)
    except OSError as exc:
        try:
            dek_tmp.unlink()
        except OSError:
            pass
        raise SecurityError(f"hard_revoke failed to stage new DEK at {dek_path}: {exc}") from exc
    payload = json.dumps(
        {"v": 1, "old_seal_id": old_seal_id, "new_seal_id": new_seal_id},
        sort_keys=True,
    )
    try:
        marker_tmp.write_text(payload, encoding="utf-8")
        os.replace(marker_tmp, marker_path)
    except OSError as exc:
        try:
            marker_tmp.unlink()
        except OSError:
            pass
        raise SecurityError(
            f"hard_revoke failed to stage rotation marker at {marker_path}: {exc}"
        ) from exc


def _read_rotation_marker(project_dir: Path) -> dict[str, str] | None:
    """Return ``{old_seal_id, new_seal_id}`` if a rotation is in progress."""
    path = _rotation_marker_path(project_dir)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SecurityError(
            f"Rotation marker at {path} is unreadable / malformed ({exc}). "
            "Manual recovery may be needed."
        ) from exc
    if (
        not isinstance(record, dict)
        or record.get("v") != 1
        or not isinstance(record.get("old_seal_id"), str)
        or not isinstance(record.get("new_seal_id"), str)
        or not record["old_seal_id"]
        or not record["new_seal_id"]
    ):
        raise SecurityError(f"Rotation marker at {path} has unexpected schema; refusing to resume.")
    return {
        "old_seal_id": record["old_seal_id"],
        "new_seal_id": record["new_seal_id"],
    }


def _read_staged_dek(project_dir: Path, master: bytes) -> bytes:
    """Unwrap the staged dek.wrapped.rotating with *master*."""
    from cryptography.hazmat.primitives.keywrap import (
        InvalidUnwrap,
        aes_key_unwrap,
    )

    path = _rotation_dek_path(project_dir)
    if not path.is_file():
        raise SecurityError(
            f"Rotation marker exists but staged DEK at {path} is missing; "
            "cannot resume — manual recovery needed."
        )
    try:
        wrapped = path.read_bytes()
        dek: bytes = aes_key_unwrap(master, wrapped)
    except InvalidUnwrap as exc:
        raise SecurityError(f"Staged DEK at {path} won't unwrap with the current master.") from exc
    except OSError as exc:
        raise SecurityError(f"Could not read staged DEK at {path}: {exc}") from exc
    if len(dek) != DEK_LEN:
        raise SecurityError(f"Staged DEK at {path} is {len(dek)} bytes (expected {DEK_LEN}).")
    return dek


def _clear_rotation_context(project_dir: Path) -> None:
    """Best-effort cleanup of the staging files after a successful promote."""
    for path in (_rotation_dek_path(project_dir), _rotation_marker_path(project_dir)):
        try:
            path.unlink()
        except OSError as exc:
            logger.debug("Could not clear rotation context file %s: %s", path, exc)


def _hard_revoke(
    owner_user_dir: Path,
    project: str,
    project_dir: Path,
    key_id: str,
) -> dict[str, Any]:
    """Rotate the project DEK; re-encrypt every file; nuke every share wrap.
    Crash-safe via staged ``dek.wrapped.rotating`` + ``.sealing.rotation``
    sidecars: if a prior rotation crashed, the next call detects the
    rotation marker, recovers the staged DEK + seal_ids, and resumes
    the per-file loop. Each file is decrypted with whichever (DEK,
    seal_id) pair works (old or new) and re-encrypted with the new pair.
    """
    # Defer imports — these pull in sealed-only modules that may not be
    # installed (the Phase 2 stack guards on ImportError already).
    from cryptography.exceptions import InvalidTag

    from .crypto import SealedFile, generate_dek, make_aad
    from .master import get_master_key, get_project_dek
    from .seal import (
        _SEAL_MAX_FILE_BYTES,
        _should_seal,
        _write_sealed_marker,
        read_sealed_marker,
    )

    marker = read_sealed_marker(project_dir)
    if marker is None:
        raise SecurityError(f"Sealed marker missing at {project_dir}; cannot hard-revoke.")
    old_seal_id_marker = marker.get("seal_id", "")
    if not isinstance(old_seal_id_marker, str) or not old_seal_id_marker:
        raise SecurityError(f"Sealed marker at {project_dir} has no seal_id; refusing to rotate.")
    # The master + the live DEK are required either way; surface clear
    # errors before touching any state.
    master = get_master_key(owner_user_dir)
    old_dek = get_project_dek(owner_user_dir, project_dir)
    # Resume detection: if a prior rotation crashed, recover its
    # new_seal_id + staged DEK rather than minting fresh ones (otherwise
    # any files already rotated under the old new_seal_id would be
    # stranded).
    rotation = _read_rotation_marker(project_dir)
    if rotation is not None:
        new_seal_id = rotation["new_seal_id"]
        new_dek = _read_staged_dek(project_dir, master)
        # The persisted old_seal_id from the marker is the source of
        # truth for the partially-rotated state. If it doesn't match
        # the live marker, the live marker may have been promoted
        # already — refuse rather than corrupt.
        if rotation["old_seal_id"] != old_seal_id_marker:
            raise SecurityError(
                "Rotation marker's old_seal_id does not match the live "
                "sealed marker; manual recovery needed."
            )
        old_seal_id = old_seal_id_marker
        logger.info(
            "Resuming crashed hard_revoke for '%s' (new_seal_id=%s)",
            project,
            new_seal_id,
        )
    else:
        new_dek = generate_dek()
        new_seal_id = "seal_" + secrets.token_hex(8)
        old_seal_id = old_seal_id_marker
        # Persist new DEK + rotation marker BEFORE rotating any file.
        _stage_rotation(
            project_dir,
            master=master,
            new_dek=new_dek,
            old_seal_id=old_seal_id,
            new_seal_id=new_seal_id,
        )
    files_resealed = 0
    files_already_rotated = 0
    # Re-encrypt every content file. On resume, files already rotated
    # under new_dek/new_seal_id will fail the old-pair decrypt and
    # succeed under the new pair — those are skipped + counted.
    for src in project_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(project_dir)
        if not _should_seal(rel):
            continue
        try:
            size = src.stat().st_size
        except OSError:
            size = 0
        if size > _SEAL_MAX_FILE_BYTES:
            raise SecurityError(
                f"hard_revoke refusing to rotate {src} ({size:,} bytes); "
                f"per-file limit is {_SEAL_MAX_FILE_BYTES:,} bytes."
            )
        rel_unix = str(rel).replace("\\", "/")
        old_aad = make_aad(old_seal_id, rel_unix)
        new_aad = make_aad(new_seal_id, rel_unix)
        # Try old pair first (the common case). If the file was already
        # rotated by a crashed prior pass, old decrypt fails with
        # InvalidTag and we try the new pair to confirm — then skip.
        plaintext: bytes | None = None
        try:
            plaintext = SealedFile.read(src, old_dek, aad=old_aad)
        except InvalidTag:
            try:
                SealedFile.read(src, new_dek, aad=new_aad)
                files_already_rotated += 1
                continue  # already rotated by a crashed prior pass
            except Exception as exc:
                raise SecurityError(
                    f"hard_revoke: {src} won't decrypt with old OR new DEK; "
                    f"file may be corrupted ({type(exc).__name__}: {exc})."
                ) from exc
        except Exception as exc:
            raise SecurityError(
                f"hard_revoke failed to decrypt {src} during rotation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        tmp = src.with_suffix(src.suffix + ".sealing")
        try:
            SealedFile.write(tmp, plaintext, new_dek, aad=new_aad)
            os.replace(tmp, src)
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise SecurityError(f"hard_revoke failed to atomically replace {src}: {exc}") from exc
        files_resealed += 1
    # Track which specific key_id was wrap-deleted before any cleanup.
    wrap_existed = share_wrap_path(project_dir, key_id).is_file() if key_id else False
    # Selective re-wrap (audit C1):
    #   1. Enumerate every existing share key_id (the union of "to be
    #      revoked" and "to survive").
    #   2. For each SURVIVING key_id, try to load its persisted KEK.
    #      - On hit  → AES-KW wrap the new DEK under that KEK and stage
    #        the new wrap to ``<key_id>.wrapped.new``.
    #      - On miss → log a one-shot warning and treat the share as
    #        invalidated (legacy fallback for projects created before
    #        per-share KEK persistence shipped).
    #   3. Promote the staged DEK over the live one.
    #   4. For each surviving key_id with a staged ``.wrapped.new``,
    #      atomically rename it over ``<key_id>.wrapped``. For the
    #      revoked key_id (and any survivor without a KEK), delete the
    #      old ``.wrapped``. Always delete the revoked key_id's ``.kek``.
    # Crash window analysis: if we crash AFTER promoting the staged DEK
    # but BEFORE renaming all the ``.wrapped.new`` files, the project's
    # live state is "new DEK on disk + a mix of old/new wraps". A
    # subsequent successful revoke (or manual fixup) will detect the
    # mismatch — the next call's stage_rotation will see no rotation
    # marker (we cleared it on success), so it would mint a fresh DEK
    # and re-rotate. The .wrapped.new files would still be cleanable by
    # ops. We surface the partial state via a clear error path on read.
    all_key_ids: list[str] = list_sealed_share_key_ids(project_dir)
    survivor_ids = [kid for kid in all_key_ids if kid != key_id]
    rewrapped_survivors: list[str] = []
    invalidated_survivors: list[str] = []
    legacy_warned = False
    for survivor_kid in survivor_ids:
        try:
            survivor_kek = _load_share_kek(project_dir, survivor_kid, master)
        except SecurityError as exc:
            # KEK file present but unwrap/read failed — treat as
            # invalidated rather than aborting the whole revoke (an
            # unrelated grantee shouldn't block the revocation).
            logger.warning(
                "hard_revoke: failed to load KEK for surviving share %s "
                "(%s); will invalidate this share. Re-issue required.",
                survivor_kid,
                exc,
            )
            invalidated_survivors.append(survivor_kid)
            continue
        if survivor_kek is None:
            # Legacy project — per-share KEK was never persisted. Fall
            # back to "all surviving shares invalidated" with a single
            # consolidated warning so the log isn't spammed once per
            # share.
            if not legacy_warned:
                logger.warning(
                    "hard_revoke: surviving share(s) in project at %s are "
                    "missing per-share KEK files (.security/shares/<key_id>.kek). "
                    "These shares cannot be selectively re-wrapped and will be "
                    "invalidated. Grantees for those shares must re-redeem. "
                    "New shares generated after this revoke will persist KEK files.",
                    project_dir,
                )
                legacy_warned = True
            invalidated_survivors.append(survivor_kid)
            continue
        new_wrapped_dek = aes_key_wrap(survivor_kek, new_dek)
        # Stage the new wrap to ``.wrapped.new``. Promote AFTER the
        # staged DEK is promoted so a crash mid-loop never points
        # surviving grantees at a wrap whose DEK isn't yet live.
        live_wrap = share_wrap_path(project_dir, survivor_kid)
        staged_wrap = live_wrap.with_suffix(live_wrap.suffix + ".new")
        tmp_wrap = live_wrap.with_suffix(live_wrap.suffix + ".tmp")
        try:
            # Write to a sibling temp file then rename so a crash mid-write
            # never leaves a partially-written .wrapped.new on disk.
            tmp_wrap.write_bytes(new_wrapped_dek)
            tmp_wrap.replace(staged_wrap)
        except OSError as exc:
            tmp_wrap.unlink(missing_ok=True)
            raise SecurityError(
                f"hard_revoke failed to stage re-wrap for share {survivor_kid} "
                f"at {staged_wrap}: {exc}"
            ) from exc
        rewrapped_survivors.append(survivor_kid)
    # Promote staged DEK over the live one BEFORE renaming the staged
    # share wraps. The staged share wraps already encode the new DEK,
    # so any ordering where survivors go live before the project DEK is
    # promoted would be inconsistent — survivors would unwrap to a DEK
    # that isn't yet the live DEK on the on-disk files. Promote DEK
    # first, then atomically rename staged wraps. A crash between these
    # two steps leaves the new DEK live + ``.wrapped.new`` sidecars on
    # disk; the recovery path below handles them on the next call.
    dek_wrap_path = project_dir / ".security" / "dek.wrapped"
    try:
        os.replace(_rotation_dek_path(project_dir), dek_wrap_path)
    except OSError as exc:
        raise SecurityError(
            f"hard_revoke failed to promote staged DEK at {dek_wrap_path}: {exc}"
        ) from exc
    # Atomically rename each staged ``.wrapped.new`` over the live
    # ``.wrapped``. Any failure here leaves a recoverable state (the
    # next call's read of the staged file lets ops complete by hand).
    shares_dir = project_dir / SHARE_DIR_NAME
    for survivor_kid in rewrapped_survivors:
        live_wrap = share_wrap_path(project_dir, survivor_kid)
        staged_wrap = live_wrap.with_suffix(live_wrap.suffix + ".new")
        try:
            os.replace(staged_wrap, live_wrap)
        except OSError as exc:
            raise SecurityError(
                f"hard_revoke: staged DEK promoted but failed to atomically "
                f"rename re-wrap {staged_wrap} → {live_wrap} ({exc}). "
                "Manual recovery: rename the .wrapped.new files into place."
            ) from exc
    # Delete the wraps + KEKs of the revoked + invalidated key_ids.
    # The revoked key_id is always cleaned up (wrap + KEK deleted);
    # surviving shares without a usable KEK fall through to the legacy
    # "must re-redeem" path and also get their files deleted.
    # ``delete_keyids`` is used ONLY for on-disk cleanup (best-effort:
    # the files may not exist). ``invalidated_share_key_ids`` in the
    # return dict lists only key_ids that were REAL shares and now need
    # re-issuing — phantom key_ids (trigger key_id that never had a
    # wrap) are excluded from the audit output.
    delete_keyids = set(invalidated_survivors)
    if key_id:
        delete_keyids.add(key_id)
    if shares_dir.is_dir():
        for delete_kid in delete_keyids:
            for path in (
                share_wrap_path(project_dir, delete_kid),
                share_kek_path(project_dir, delete_kid),
            ):
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError as exc:
                        logger.debug("Could not delete %s: %s", path, exc)
        # Also sweep any orphaned ``.wrapped.new`` left from earlier
        # crashed runs (not from this run — those have all been
        # promoted by now).
        for entry in shares_dir.iterdir():
            if entry.is_file() and entry.name.endswith(".wrapped.new"):
                try:
                    entry.unlink()
                except OSError as exc:
                    logger.debug("Could not delete orphan stage %s: %s", entry, exc)
    # ``invalidated_share_key_ids`` lists only key_ids that were actual
    # shares and now need re-issuing (the revoked one if it existed +
    # any survivors that fell back to the legacy/error path). Phantom
    # key_ids (the trigger key_id was never a real share) are excluded.
    invalidated_set = set(invalidated_survivors)
    if key_id and key_id in all_key_ids:
        invalidated_set.add(key_id)
    invalidated: list[str] = sorted(invalidated_set)
    # Refresh the sealed marker to carry the new seal_id.
    _write_sealed_marker(
        project_dir,
        seal_id=new_seal_id,
        files_sealed=files_resealed + files_already_rotated,
        files_skipped=0,
    )
    # Rotation complete — clear the staging context.
    _clear_rotation_context(project_dir)
    # Bump the cross-machine staleness marker so any grantee still
    # holding open file handles to the old DEK detects the rotation
    # on next mount-switch and refreshes (audit P1: previously, the
    # marker was only bumped by ingest, so a quiet rotation could go
    # undetected by mounted grantees).
    try:
        from axon.projects import get_or_create_node_id
        from axon.version_marker import bump as _vm_bump

        # v0.4.0 Item 4a: stamp the marker with the store-scoped UUID
        # rather than the hostname (the synced volume must not leak the
        # owner's machine identity).
        _node_id = ""
        try:
            _node_id = get_or_create_node_id(owner_user_dir)
        except Exception:
            pass
        _vm_bump(project_dir, node_id=_node_id)
    except Exception as _vm_exc:  # pragma: no cover — defensive
        logger.debug("hard_revoke: version_marker bump raised: %s", _vm_exc)
    logger.info(
        "Hard-revoked sealed project '%s': rotated DEK, resealed %d files "
        "(%d already rotated by prior crashed run), invalidated %d share(s) "
        "(triggering key_id=%s)",
        project,
        files_resealed,
        files_already_rotated,
        len(invalidated),
        key_id,
    )
    return {
        "status": "hard_revoked",
        "key_id": key_id,
        "rotate": True,
        "wrap_deleted": wrap_existed,
        "invalidated_share_key_ids": invalidated,
        "files_resealed": files_resealed,
        "files_already_rotated": files_already_rotated,
        "new_seal_id": new_seal_id,
    }


def delete_grantee_dek(key_id: str, user_dir: Path | None = None) -> bool:
    """Remove the grantee's cached DEK for *key_id*; True iff one was present.

    Deletes both the OS keyring entry and the file fallback (if present) so
    cleanup is complete regardless of how the DEK was originally persisted.

    Used by Phase 4 revocation cleanup. Best-effort — a missing
    entry returns False without raising, so callers can use this as
    an idempotent cleanup hook. The keyring layer itself silently
    no-ops on missing-key delete, so we have to probe first to
    distinguish "deleted something" from "nothing was there".

    Args:
        key_id: Share key identifier.
        user_dir: Grantee's AxonStore user directory.  When provided, the
            file fallback at ``<user_dir>/.security/shares/<key_id>.dek.wrapped``
            is also deleted (best-effort).
    """
    if not key_id:
        return False
    # Validate key_id without raising — invalid IDs mean nothing was stored.
    try:
        _validate_key_id(key_id)
    except SecurityError:
        return False
    service = _share_keyring_service(key_id)
    had_secret = False
    try:
        had_secret = _kr.get_secret(service, "dek") is not None
        if had_secret:
            _kr.delete_secret(service, "dek")
    except Exception as exc:
        logger.debug("delete_grantee_dek(%s) keyring op failed: %s", key_id, exc)
    # Also clean up the file fallback, regardless of keyring availability.
    had_file = False
    if user_dir is not None:
        fb_path = _grantee_dek_fallback_path(Path(user_dir), key_id)
        had_file = fb_path.is_file()
        try:
            fb_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("delete_grantee_dek(%s) file unlink failed: %s", key_id, exc)
    return had_secret or had_file
