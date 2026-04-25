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
from .crypto import DEK_LEN
from .master import get_project_dek
from .seal import is_project_sealed

_KEY_ID_PATTERN = _re.compile(r"^[A-Za-z0-9_-]{1,64}$")

logger = logging.getLogger("Axon")

__all__ = [
    "SEALED_SHARE_PREFIX",
    "SHARE_KEYRING_PREFIX",
    "SHARE_DIR_NAME",
    "WRAP_FILENAME_FORMAT",
    "generate_sealed_share",
    "redeem_sealed_share",
    "get_grantee_dek",
    "delete_grantee_dek",
    "share_wrap_path",
    "list_sealed_share_key_ids",
    "revoke_sealed_share",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEALED_SHARE_PREFIX: str = "SEALED1"
SHARE_KEYRING_PREFIX: str = "axon.share"  # service = "axon.share.<key_id>"
SHARE_DIR_NAME: str = ".security/shares"
WRAP_FILENAME_FORMAT: str = "{key_id}.wrapped"
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
# generate_sealed_share (owner side)
# ---------------------------------------------------------------------------


def generate_sealed_share(
    owner_user_dir: Path,
    project: str,
    grantee: str,
    key_id: str,
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
    raw = (
        f"{SEALED_SHARE_PREFIX}:{key_id}:{token_hex}:" f"{owner_name}:{project}:{owner_store_path}"
    )
    share_string = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")

    logger.info(
        "Generated sealed share key_id=%s project=%s grantee=%s wrapped_path=%s",
        key_id,
        project,
        grantee,
        wrap_path,
    )
    return {
        "key_id": key_id,
        "share_string": share_string,
        "wrapped_path": str(wrap_path),
        "owner": owner_name,
        "project": project,
        "grantee": grantee,
        "sealed": True,
    }


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
            :func:`generate_sealed_share` (must start with the
            ``SEALED1:`` prefix after base64 decode).

    Returns:
        ``{"key_id": ..., "mount_name": ..., "owner": ..., "project": ...,
        "descriptor": <mount_descriptor>}``

    Raises:
        SecurityError: share_string malformed, wrap file missing, KEK
            derivation produces an invalid DEK (token corruption),
            or I/O error writing the mount descriptor.
        ValueError: share_string is a non-sealed legacy share (let the
            caller route to ``axon.shares.redeem_share_key`` instead).
    """
    grantee_user_dir = Path(grantee_user_dir).resolve()

    try:
        raw = base64.urlsafe_b64decode(share_string.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise SecurityError(f"Invalid sealed share_string format: {exc}") from exc

    if not raw.startswith(f"{SEALED_SHARE_PREFIX}:"):
        # Legacy plaintext-mount share — caller should route to the
        # non-sealed redeem path instead.
        raise ValueError(
            "share_string is not a sealed share. "
            "Route to axon.shares.redeem_share_key() for legacy shares."
        )

    parts = raw.split(":", 5)
    if len(parts) != 6:
        raise SecurityError(
            f"Sealed share_string has wrong shape (expected 6 colon-separated "
            f"fields after the prefix, got {len(parts)})"
        )
    _prefix, key_id, token_hex, owner, project, owner_store_path = parts
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
    try:
        _kr.store_secret(service, "dek", base64.b64encode(dek).decode("ascii"))
    except _kr.KeyringUnavailableError as exc:
        raise SecurityError(
            f"Cannot persist sealed-share DEK to OS keyring: {exc}. "
            "On a headless Linux server, install gnome-keyring or use "
            "the passphrase fallback (Phase 6)."
        ) from exc

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
    )

    logger.info(
        "Redeemed sealed share key_id=%s owner=%s project=%s mount=%s",
        key_id,
        owner,
        project,
        mount_name,
    )
    return {
        "key_id": key_id,
        "mount_name": mount_name,
        "owner": owner,
        "project": project,
        "descriptor": descriptor,
        "sealed": True,
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
) -> dict[str, Any]:
    """Write a mount.json carrying the ``mount_type="sealed"`` flag.

    Builds the descriptor via :func:`axon.mounts.create_mount_descriptor`
    so the canonical schema (state, revoked, descriptor_version,
    project_id, store_id, redeemed_at) is preserved — without that,
    :func:`axon.mounts.validate_mount_descriptor` rejects the mount and
    ``axon.mounts.list_mount_descriptors`` filters it out. The sealed
    flag and pointer to the share key are added afterward + persisted
    atomically (the mounts module's plain write isn't atomic).
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


def get_grantee_dek(key_id: str) -> bytes:
    """Fetch the grantee's cached DEK for *key_id* from the OS keyring.

    Raises:
        SecurityError: keyring unavailable, no DEK stored under
            ``axon.share.<key_id>`` (grantee never redeemed), or the
            stored value is malformed base64.
    """
    _validate_key_id(key_id)
    service = _share_keyring_service(key_id)
    try:
        secret = _kr.get_secret(service, "dek")
    except _kr.KeyringUnavailableError as exc:
        raise SecurityError(
            f"OS keyring unavailable; cannot fetch sealed-share DEK for " f"{key_id}: {exc}"
        ) from exc
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

    **Hard (``rotate=True``)** — slow, breaks all cached DEKs:

    - Generate a fresh project DEK + a fresh ``seal_id``.
    - Re-encrypt every content file under the project with the new
      DEK + AAD (``make_aad(new_seal_id, rel)``).
    - Update ``.security/dek.wrapped`` (master-wrapped form).
    - Update the sealed marker with the new ``seal_id``.
    - **Delete every share wrap** in ``.security/shares/`` (including
      the one being revoked AND every still-valid share's wrap).
    - Returns the list of invalidated key_ids in
      ``invalidated_share_key_ids`` so the caller can re-issue them.

    Why hard revoke kills ALL share wraps in v1: persisting per-share
    KEKs encrypted-under-master so the owner can re-wrap surviving
    shares without the original token is a meaningful design surface
    (more files, more recovery edge-cases) tracked as a follow-up.
    The simpler "rotate invalidates all" model matches the same UX as
    "compromised master" — rare, well-understood, all-affected-grantees
    re-redeem.

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
    """Delete the wrap file for *key_id* — fresh redeem will fail."""
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

    # Promote staged DEK over the live one. The staged DEK is identical
    # to what we'd compute now; using it ensures a crash-safe ordering.
    dek_wrap_path = project_dir / ".security" / "dek.wrapped"
    try:
        os.replace(_rotation_dek_path(project_dir), dek_wrap_path)
    except OSError as exc:
        raise SecurityError(
            f"hard_revoke failed to promote staged DEK at {dek_wrap_path}: {exc}"
        ) from exc

    # Track which specific key_id was wrap-deleted before the bulk-delete.
    wrap_existed = share_wrap_path(project_dir, key_id).is_file() if key_id else False

    # Nuke every share wrap — surviving grantees re-redeem to pick up
    # the new DEK. (Persisting per-share KEKs encrypted under the
    # master so we can re-wrap surviving shares without the original
    # token is a future enhancement; for v1, all-shares-invalidated
    # matches the "compromised master" UX.)
    invalidated: list[str] = list_sealed_share_key_ids(project_dir)
    shares_dir = project_dir / SHARE_DIR_NAME
    if shares_dir.is_dir():
        for entry in shares_dir.iterdir():
            if entry.is_file() and entry.name.endswith(".wrapped"):
                try:
                    entry.unlink()
                except OSError as exc:
                    logger.debug("Could not delete share wrap %s: %s", entry, exc)

    # Refresh the sealed marker to carry the new seal_id.
    _write_sealed_marker(
        project_dir,
        seal_id=new_seal_id,
        files_sealed=files_resealed + files_already_rotated,
        files_skipped=0,
    )

    # Rotation complete — clear the staging context.
    _clear_rotation_context(project_dir)

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


def delete_grantee_dek(key_id: str) -> bool:
    """Remove the grantee's cached DEK for *key_id*; True iff one was present.

    Used by Phase 4 revocation cleanup. Best-effort — a missing
    entry returns False without raising, so callers can use this as
    an idempotent cleanup hook. The keyring layer itself silently
    no-ops on missing-key delete, so we have to probe first to
    distinguish "deleted something" from "nothing was there".
    """
    if not key_id:
        return False
    service = _share_keyring_service(key_id)
    try:
        had_secret = _kr.get_secret(service, "dek") is not None
        if not had_secret:
            return False
        _kr.delete_secret(service, "dek")
        return True
    except Exception as exc:
        logger.debug("delete_grantee_dek(%s) failed: %s", key_id, exc)
        return False
