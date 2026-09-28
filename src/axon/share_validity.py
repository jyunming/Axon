"""One answer to "is this received share still valid?".

Before this module, six call sites each decided share validity on their
own (REST ``/share/list``, REST ``/project/switch``, CLI/REPL share list,
``AxonBrain.switch_project``, the per-query ``_check_mount_revocation``
and ``list_share_mounts``) and they disagreed — a revoked plain share was
refused by REST but still mountable from the CLI, a soft-revoked sealed
share kept working everywhere except ``/share/list``, and so on.

Authority (no new record is introduced — these files already exist):

* **plain share** — the owner's ``<owner>/.shares/.share_manifest.json``
  entry for the key_id (``revoked`` / ``expires_at``).
* **sealed share** — presence of ``<project>/.security/shares/<kid>.wrapped``
  (deleted on soft revoke) plus the owner-signed ``<kid>.expiry`` sidecar.

The grantee's ``mounts/<name>/mount.json`` descriptor and the ``received``
records in ``.share_keys.json`` are only *pointers* to those records.

Everything here is read-only except :func:`reconcile_received_mounts`,
which is the single place that removes stale grantee-side state.

States
------
``VALID``         allow; keep the mount.
``REVOKED``       deny; remove the descriptor (sealed: the cached DEK is
                  kept — a soft revoke never promised to wipe it).
``EXPIRED``       deny; remove the descriptor (sealed: also delete the
                  cached DEK — same as the existing auto-destroy flow).
``UNVERIFIABLE``  deny; KEEP the mount — the authoritative record could
                  not be read (owner offline, sync incomplete, cloud
                  placeholder), so nothing destructive happens.
``INVALID``       deny; keep — the local descriptor itself is malformed.

``ShareStatus.detail`` is built only from the key_id, project/owner
names, paths, timestamps and the reason code. It never contains share
tokens, share strings, HMACs or DEKs, and it never echoes the text of an
underlying exception.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("AxonShares")

__all__ = [
    "ShareState",
    "ShareStatus",
    "ShareInvalidError",
    "share_status",
    "require_valid",
    "owner_share_status",
    "load_owner_manifest",
    "reconcile_received_mounts",
]


class ShareState(str, Enum):
    VALID = "valid"
    REVOKED = "revoked"
    EXPIRED = "expired"
    UNVERIFIABLE = "unverifiable"
    INVALID = "invalid"


@dataclass(frozen=True)
class ShareStatus:
    """Outcome of a validity check. ``detail`` is safe to show and log."""

    state: ShareState
    reason: str
    detail: str
    kind: str  # "plain" | "sealed"
    key_id: str = ""
    authority: str = ""  # path of the authoritative owner-side record
    expires_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.state is ShareState.VALID

    @property
    def terminal(self) -> bool:
        """True when the owner has definitively ended the share."""
        return self.state in (ShareState.REVOKED, ShareState.EXPIRED)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "reason": self.reason}


class ShareInvalidError(PermissionError):
    """Raised by :func:`require_valid`; carries the :class:`ShareStatus`."""

    def __init__(self, status: ShareStatus) -> None:
        super().__init__(status.detail)
        self.status = status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_file(owner_user_dir: Path | str) -> Path:
    # Deliberately NOT shares._manifest_path(): that helper mkdirs, and a
    # read-only check from the grantee side must never create directories
    # inside the owner's (possibly synced) store.
    return Path(owner_user_dir) / ".shares" / ".share_manifest.json"


def _label(project: str, key_id: str) -> str:
    return f"Share '{project}' (key {key_id})" if project else f"Share {key_id}"


def load_owner_manifest(owner_user_dir: Path | str) -> dict[str, Any] | None:
    """Strictly read an owner's public share manifest.

    Returns ``None`` when the manifest is missing, unreadable, not a JSON
    object, or its ``issued`` field is not a list — i.e. whenever it cannot
    be used as an authority. (The lenient ``shares._read_json`` maps all of
    those to ``{}``, which would be indistinguishable from "no shares".)
    """
    path = _manifest_file(owner_user_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("issued", []), list):
        return None
    return data


def _plain_record_status(
    manifest: dict[str, Any] | None,
    manifest_path: Path,
    key_id: str,
    project: str,
    now: datetime | None,
) -> ShareStatus:
    from axon.shares import _is_expired

    authority = str(manifest_path)
    label = _label(project, key_id)
    if manifest is None:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "manifest_unreadable",
            f"{label}: the owner's share manifest at {manifest_path} is missing or "
            "unreadable, so the share cannot be verified (owner offline or sync incomplete?).",
            "plain",
            key_id,
            authority,
        )
    records = [
        r for r in manifest.get("issued", []) if isinstance(r, dict) and r.get("key_id") == key_id
    ]
    if not records:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "record_absent",
            f"{label}: no record for this key in the owner's share manifest at "
            f"{manifest_path}; the share cannot be verified.",
            "plain",
            key_id,
            authority,
        )
    if any(r.get("revoked") for r in records):
        return ShareStatus(
            ShareState.REVOKED,
            "revoked",
            f"{label} has been revoked by the owner. "
            "Run `/project switch default` to continue with your own projects.",
            "plain",
            key_id,
            authority,
        )
    for r in records:
        exp = r.get("expires_at")
        if _is_expired(exp, now=now):
            exp_s = exp if isinstance(exp, str) else repr(exp)
            return ShareStatus(
                ShareState.EXPIRED,
                "expired",
                f"{label} expired at {exp_s}. Ask the owner to extend "
                "(`/share extend`) and re-redeem.",
                "plain",
                key_id,
                authority,
                exp_s,
            )
    exp0 = records[0].get("expires_at")
    return ShareStatus(
        ShareState.VALID,
        "ok",
        f"{label} is valid.",
        "plain",
        key_id,
        authority,
        exp0 if isinstance(exp0, str) else None,
    )


def _parse_sidecar_expiry(raw: bytes) -> tuple[str | None, datetime | None]:
    """Best-effort parse of a sealed expiry sidecar's ``expires_at``.

    Informational only — the signature is verified separately by
    ``security.share._check_expiry_or_raise``.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    iso = data.get("expires_at")
    if not isinstance(iso, str) or not iso:
        return None, None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso, None
    if dt.tzinfo is None:
        return iso, None
    return iso, dt


def _owner_expiry_signature_ok(raw: bytes, key_id: str, owner_user_dir: Path) -> bool:
    """True iff the sidecar's signature verifies with the owner's signing key.

    Needs the owner's store unlocked (the key is derived from the master);
    any failure — locked store, malformed JSON, wrong key_id, bad signature —
    returns False. Never raises.
    """
    try:
        import base64

        from axon.security.share import _expiry_signing_message
        from axon.security.signing import get_signing_pubkey_hex, pubkey_from_hex

        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict) or data.get("key_id") != key_id:
            return False
        iso, sig_b64 = data.get("expires_at"), data.get("sig")
        if not (isinstance(iso, str) and iso and isinstance(sig_b64, str) and sig_b64):
            return False
        sig = base64.urlsafe_b64decode(sig_b64.encode("ascii") + b"=" * (-len(sig_b64) % 4))
        pubkey = pubkey_from_hex(get_signing_pubkey_hex(owner_user_dir))
        pubkey.verify(sig, _expiry_signing_message(key_id, iso))
        return True
    except Exception:
        return False


def _sealed_status(
    target: Path,
    key_id: str,
    project: str,
    pubkey_hex: str,
    now: datetime | None,
) -> ShareStatus:
    """Grantee-side evaluation of a sealed share (signature-verified expiry)."""
    label = _label(project, key_id)
    try:
        from axon.security import SecurityError, ShareExpiredError
        from axon.security.share import (
            SHARE_DIR_NAME,
            _check_expiry_or_raise,
            share_expiry_path,
            share_wrap_path,
        )
    except ImportError:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "sealed_support_missing",
            f"{label}: sealed-share support is not installed "
            "(pip install axon-rag[sealed]); the share cannot be verified.",
            "sealed",
            key_id,
        )
    try:
        wrap = share_wrap_path(target, key_id)
        expiry = share_expiry_path(target, key_id)
    except SecurityError:
        return ShareStatus(
            ShareState.INVALID,
            "key_id_invalid",
            "Mount descriptor carries a malformed sealed share key id.",
            "sealed",
            "",
        )
    authority = str(wrap)
    shares_dir = target / SHARE_DIR_NAME
    marker = target / ".security" / ".sealed"
    try:
        if not marker.is_file():
            return ShareStatus(
                ShareState.UNVERIFIABLE,
                "sealed_marker_missing",
                f"{label}: the owner's sealed marker at {marker} is not present "
                "(sync incomplete?); the share cannot be verified.",
                "sealed",
                key_id,
                authority,
            )
        if not shares_dir.is_dir():
            return ShareStatus(
                ShareState.UNVERIFIABLE,
                "shares_dir_missing",
                f"{label}: the owner's share directory {shares_dir} is not present "
                "(sync incomplete?); the share cannot be verified.",
                "sealed",
                key_id,
                authority,
            )
        if not wrap.is_file():
            return ShareStatus(
                ShareState.REVOKED,
                "wrap_absent",
                f"{label} has been revoked by the owner (share wrap removed). "
                "Run `/project switch default` to continue with your own projects.",
                "sealed",
                key_id,
                authority,
            )
        has_expiry = expiry.is_file()
    except OSError:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "owner_files_unreadable",
            f"{label}: the owner's share files under {shares_dir} could not be read.",
            "sealed",
            key_id,
            authority,
        )
    raw: bytes | None = None
    if has_expiry:
        # Read the sidecar ourselves first: _check_expiry_or_raise maps an
        # OSError on read to ShareExpiredError, which would send an
        # unreadable (e.g. cloud-placeholder) sidecar down the
        # expired → auto-destroy path. An unreadable file is UNVERIFIABLE.
        try:
            raw = expiry.read_bytes()
        except OSError:
            return ShareStatus(
                ShareState.UNVERIFIABLE,
                "expiry_unreadable",
                f"{label}: the signed expiry file {expiry} could not be read "
                "(sync incomplete?); the share cannot be verified.",
                "sealed",
                key_id,
                authority,
            )
    exp_iso, exp_dt = _parse_sidecar_expiry(raw) if raw is not None else (None, None)
    try:
        _check_expiry_or_raise(target, key_id, pubkey_hex)
    except ShareExpiredError as exc:
        if isinstance(exc.__cause__, OSError):
            # Became unreadable between our read and its read.
            return ShareStatus(
                ShareState.UNVERIFIABLE,
                "expiry_unreadable",
                f"{label}: the signed expiry file {expiry} could not be read "
                "(sync incomplete?); the share cannot be verified.",
                "sealed",
                key_id,
                authority,
            )
        # Reason is classified from the message only; the message itself is
        # never copied into ``detail``.
        if " expired at " in str(exc) and exp_iso:
            return ShareStatus(
                ShareState.EXPIRED,
                "expired",
                f"{label} expired at {exp_iso}. Request a fresh share from the owner.",
                "sealed",
                key_id,
                authority,
                exp_iso,
            )
        return ShareStatus(
            ShareState.EXPIRED,
            "expiry_unverified",
            f"{label}: the expiry file {expiry} failed signature or format "
            "verification; treating the share as expired. Request a fresh share "
            "from the owner.",
            "sealed",
            key_id,
            authority,
            exp_iso,
        )
    except ImportError:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "sealed_support_missing",
            f"{label}: sealed-share support is not installed "
            "(pip install axon-rag[sealed]); the share cannot be verified.",
            "sealed",
            key_id,
            authority,
        )
    except SecurityError:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "expiry_check_failed",
            f"{label}: the expiry check could not be completed.",
            "sealed",
            key_id,
            authority,
        )
    if exp_dt is not None and now is not None:
        ref = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        if ref > exp_dt:
            return ShareStatus(
                ShareState.EXPIRED,
                "expired",
                f"{label} expired at {exp_iso}. Request a fresh share from the owner.",
                "sealed",
                key_id,
                authority,
                exp_iso,
            )
    return ShareStatus(
        ShareState.VALID,
        "ok",
        f"{label} is valid.",
        "sealed",
        key_id,
        authority,
        exp_iso,
    )


# ---------------------------------------------------------------------------
# Public API — read-only
# ---------------------------------------------------------------------------


def share_status(descriptor: Any, now: datetime | None = None) -> ShareStatus:
    """Decide the validity of a received share from its mount descriptor.

    Pure and read-only: never writes, never deletes, never creates
    directories. Evaluation order: descriptor sanity (INVALID) → target
    directory present (else UNVERIFIABLE) → plain: owner manifest readable,
    record present, revoked, expired → sealed: sealed marker and share dir
    present, wrap present (else REVOKED), expiry sidecar readable, signature
    and timestamp (else EXPIRED).

    *now* overrides the clock for the plain ``expires_at`` check (which
    keeps its 5-minute clock-skew leeway). For sealed shares the signed
    expiry is always checked against the real clock (strict, no leeway);
    *now* can only additionally tighten it.
    """
    if not isinstance(descriptor, dict):
        return ShareStatus(
            ShareState.INVALID,
            "descriptor_malformed",
            "Mount descriptor is not a JSON object.",
            "plain",
        )
    kind = "sealed" if descriptor.get("mount_type") == "sealed" else "plain"
    key_id = descriptor.get("share_key_id")
    raw_project = descriptor.get("project")
    project: str = raw_project if isinstance(raw_project, str) else ""
    if not isinstance(key_id, str) or not key_id:
        return ShareStatus(
            ShareState.INVALID,
            "key_id_missing",
            "Mount descriptor has no share key id; re-redeem the share.",
            kind,
        )
    label = _label(project, key_id)
    if descriptor.get("revoked") or descriptor.get("state", "active") != "active":
        return ShareStatus(
            ShareState.INVALID,
            "descriptor_inactive",
            f"{label}: the local mount descriptor is not active.",
            kind,
            key_id,
        )
    target_s = descriptor.get("target_project_dir")
    if not isinstance(target_s, str) or not target_s:
        return ShareStatus(
            ShareState.INVALID,
            "target_unset",
            f"{label}: the mount descriptor has no target project directory.",
            kind,
            key_id,
        )
    target = Path(target_s)
    try:
        target_ok = target.is_dir()
    except OSError:
        target_ok = False
    if not target_ok:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "target_missing",
            f"{label}: target project directory does not exist: {target} "
            "(owner offline, sync incomplete, or project deleted).",
            kind,
            key_id,
        )
    if kind == "sealed":
        pubkey = descriptor.get("owner_pubkey_hex")
        return _sealed_status(
            target, key_id, project, pubkey if isinstance(pubkey, str) else "", now
        )
    owner_dir = descriptor.get("owner_user_dir")
    if not isinstance(owner_dir, str) or not owner_dir:
        return ShareStatus(
            ShareState.INVALID,
            "owner_unset",
            f"{label}: the mount descriptor has no owner store directory.",
            kind,
            key_id,
        )
    return _plain_record_status(
        load_owner_manifest(owner_dir), _manifest_file(owner_dir), key_id, project, now
    )


def require_valid(descriptor: Any, now: datetime | None = None) -> ShareStatus:
    """Return the VALID status for *descriptor* or raise :class:`ShareInvalidError`."""
    status = share_status(descriptor, now=now)
    if not status.ok:
        raise ShareInvalidError(status)
    return status


def owner_share_status(
    owner_user_dir: Path | str,
    project: str,
    key_id: str,
    kind: str,
    now: datetime | None = None,
) -> ShareStatus:
    """Owner-side view of one issued share, from the same authoritative records.

    * plain — the owner's own manifest entry (same rules as the grantee).
    * sealed — wrap present = not revoked. The ``.expiry`` sidecar lives
      in the shared project folder, which grantees can usually write to, so
      its ``expires_at`` only counts once its signature verifies against the
      owner's own signing key. This feeds owner-side decisions such as
      delete gating, where the safe answer is "still active": a forged,
      malformed or (store locked) unverifiable sidecar leaves the share
      VALID rather than letting a grantee unblock deletion of a project
      they still hold a real share on.
    """
    owner_user_dir = Path(owner_user_dir)
    if kind != "sealed":
        return _plain_record_status(
            load_owner_manifest(owner_user_dir),
            _manifest_file(owner_user_dir),
            key_id,
            project,
            now,
        )
    label = _label(project, key_id)
    try:
        from axon.security import SecurityError
        from axon.security.share import (
            SHARE_DIR_NAME,
            _resolve_owned_project_dir,
            share_expiry_path,
            share_wrap_path,
        )
    except ImportError:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "sealed_support_missing",
            f"{label}: sealed-share support is not installed.",
            "sealed",
            key_id,
        )
    pdir = _resolve_owned_project_dir(project, owner_user_dir)
    try:
        wrap = share_wrap_path(pdir, key_id)
        expiry = share_expiry_path(pdir, key_id)
    except SecurityError:
        return ShareStatus(
            ShareState.INVALID, "key_id_invalid", "Malformed sealed share key id.", "sealed"
        )
    authority = str(wrap)
    try:
        if not (pdir / ".security" / ".sealed").is_file() or not (pdir / SHARE_DIR_NAME).is_dir():
            return ShareStatus(
                ShareState.UNVERIFIABLE,
                "sealed_marker_missing",
                f"{label}: {pdir} is not a readable sealed project.",
                "sealed",
                key_id,
                authority,
            )
        if not wrap.is_file():
            return ShareStatus(
                ShareState.REVOKED,
                "wrap_absent",
                f"{label} is revoked.",
                "sealed",
                key_id,
                authority,
            )
        raw = expiry.read_bytes() if expiry.is_file() else None
    except OSError:
        return ShareStatus(
            ShareState.UNVERIFIABLE,
            "owner_files_unreadable",
            f"{label}: share files under {pdir} could not be read.",
            "sealed",
            key_id,
            authority,
        )
    if raw is None:
        return ShareStatus(
            ShareState.VALID, "ok", f"{label} is valid.", "sealed", key_id, authority
        )
    exp_iso, exp_dt = _parse_sidecar_expiry(raw)
    if exp_dt is None or not _owner_expiry_signature_ok(raw, key_id, owner_user_dir):
        return ShareStatus(
            ShareState.VALID,
            "expiry_unverified",
            f"{label}: its expiry file {expiry} could not be verified against the "
            "owner's signing key (malformed, tampered, or the store is locked); "
            "treated as still active.",
            "sealed",
            key_id,
            authority,
            exp_iso,
        )
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    if ref > exp_dt:
        return ShareStatus(
            ShareState.EXPIRED,
            "expired",
            f"{label} expired at {exp_iso}.",
            "sealed",
            key_id,
            authority,
            exp_iso,
        )
    return ShareStatus(
        ShareState.VALID, "ok", f"{label} is valid.", "sealed", key_id, authority, exp_iso
    )


# ---------------------------------------------------------------------------
# The one function with side effects
# ---------------------------------------------------------------------------


def reconcile_received_mounts(user_dir: Path | str, now: datetime | None = None) -> list[str]:
    """Bring a grantee's local share state in line with the owners' records.

    Descriptor-driven: every active ``mounts/*/mount.json`` (plain and
    sealed) is evaluated with :func:`share_status`, then

    * REVOKED — the descriptor is removed (sealed: cached DEK kept);
    * EXPIRED — the descriptor is removed; for sealed shares the cached
      DEK is also deleted (keyring + file fallback);
    * VALID / UNVERIFIABLE / INVALID — nothing is touched.

    Afterwards the ``received`` records in the grantee's ``.share_keys.json``
    are pruned when their descriptor is gone or now carries a different
    ``share_key_id`` (e.g. the mount name was re-used by a newer share);
    the newer descriptor itself is never touched by that pruning.

    ``shares._lock`` is held throughout so the keys-file read-modify-write
    cannot clobber a concurrent ``redeem_share_key``.

    Returns:
        Names of the mount descriptors that were removed.
    """
    from axon import shares as _shares
    from axon.mounts import (
        list_mount_descriptors,
        load_mount_descriptor,
        remove_mount_descriptor,
    )

    user_dir = Path(user_dir)
    removed: list[str] = []
    with _shares._lock:
        for desc in list_mount_descriptors(user_dir):
            mount_name = desc.get("mount_name")
            if not isinstance(mount_name, str) or not mount_name:
                continue
            status = share_status(desc, now=now)
            if not status.terminal:
                continue
            # Re-read before deleting: a concurrent redeem may have replaced
            # this mount with a newer share under the same name.
            try:
                current = load_mount_descriptor(user_dir, mount_name)
            except ValueError:
                continue
            if not isinstance(current, dict) or current.get("share_key_id") != status.key_id:
                continue
            try:
                remove_mount_descriptor(user_dir, mount_name)
            except OSError as exc:
                logger.warning(
                    "Could not remove %s share mount '%s' (key %s): %s",
                    status.state.value,
                    mount_name,
                    status.key_id,
                    type(exc).__name__,
                )
                continue
            removed.append(mount_name)
            logger.info(
                "Removed share mount '%s' (key %s): %s/%s",
                mount_name,
                status.key_id,
                status.state.value,
                status.reason,
            )
            if status.kind == "sealed" and status.state is ShareState.EXPIRED:
                try:
                    from axon.security.share import delete_grantee_dek

                    delete_grantee_dek(status.key_id, user_dir=user_dir)
                except Exception as exc:  # best-effort, like auto-destroy
                    logger.warning(
                        "Could not delete cached DEK for expired sealed share %s: %s",
                        status.key_id,
                        type(exc).__name__,
                    )
        keys_path = user_dir / ".shares" / ".share_keys.json"
        if keys_path.is_file():
            keys = _shares._read_json(keys_path)
            received = keys.get("received") if isinstance(keys, dict) else None
            if isinstance(received, list):
                kept: list[Any] = []
                for record in received:
                    if not isinstance(record, dict):
                        continue
                    mount_name = record.get("mount_name")
                    if not isinstance(mount_name, str) or not mount_name:
                        continue
                    try:
                        current = load_mount_descriptor(user_dir, mount_name)
                    except ValueError:
                        current = None
                    if not isinstance(current, dict):
                        continue
                    if current.get("share_key_id") != record.get("key_id"):
                        continue
                    kept.append(record)
                if len(kept) != len(received):
                    keys["received"] = kept
                    _shares._write_json(keys_path, keys)
    return removed
