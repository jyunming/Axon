"""
AxonStore share key management.

Handles generation, redemption, revocation, and validation of share keys
that allow one user to grant another read or read/write access to a project.

Two files under {user_dir}/.shares/:
  .share_manifest.json  — world-readable (644): revocation status only, no tokens
  .share_keys.json      — private (600): full token records for issued + received keys

Share flow:
  1. Owner calls generate_share_key()  -> gets a share_string to send out-of-band
  2. Grantee calls redeem_share_key()  -> descriptor created in grantee's mounts/
  3. Owner calls revoke_share_key()    -> marks revoked in manifest (lazy invalidate)
  4. Owner calls extend_share_key()    -> bumps expires_at when the share is still in use
  5. Grantee's Axon calls validate_received_shares() on next access -> removes stale descriptors

Expiry (issue #54)
------------------
Each share record carries an optional ``expires_at`` ISO timestamp. When set,
``redeem_share_key`` and ``validate_received_shares`` treat past-expiry the
same as revoked: the mount descriptor is removed and the grantee loses
access. Owners can renew via :func:`extend_share_key`. ``ttl_days=None``
preserves the original "never expires" behaviour for backward compatibility.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


_EXPIRY_CLOCK_SKEW_SECONDS = 300  # 5-minute leeway for cross-machine clock drift


def _is_expired(expires_at: str | None, *, now: datetime | None = None) -> bool:
    """Return True if *expires_at* is set and lies in the past (modulo clock skew).
    - Missing / empty value → no expiry → False.
    - **Unparseable timestamp → expired (fail-closed).** A corrupted or
      hand-edited manifest record should NOT silently grant access; we
      log a warning and deny instead. This is a defence-in-depth choice
      for #54 — fail-open here would weaken TTL enforcement.
    - Valid timestamp → expired iff ``exp + skew_leeway <= now`` so a
      grantee with a slightly slow clock isn't booted off the moment
      the owner's clock crosses the expiry instant.
    """
    if not expires_at:
        return False
    try:
        exp = datetime.fromisoformat(expires_at)
    except (TypeError, ValueError) as exc:
        import logging as _logging

        _logging.getLogger("AxonShares").warning(
            "share-key expires_at value %r is unparseable (%s); treating as expired "
            "(fail-closed). Fix the manifest or generate a new share.",
            expires_at,
            exc,
        )
        return True
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    ref = (now or _utcnow()).astimezone(timezone.utc)
    # Leeway: token is still valid for _EXPIRY_CLOCK_SKEW_SECONDS past
    # nominal expiry — protects against minor clock drift between the
    # owner who issued the share and the grantee who's reading it.
    return exp + timedelta(seconds=_EXPIRY_CLOCK_SKEW_SECONDS) <= ref


_lock = threading.Lock()  # process-wide lock for share key file I/O


def _shares_dir(user_dir: Path) -> Path:
    d = user_dir / ".shares"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(user_dir: Path) -> Path:
    return _shares_dir(user_dir) / ".share_manifest.json"


def _keys_path(user_dir: Path) -> Path:
    return _shares_dir(user_dir) / ".share_keys.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Set permissions: manifest is 644 (world-readable), keys is 600 (owner only)
    if path.name == ".share_manifest.json":
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
    elif path.name == ".share_keys.json":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _compute_hmac(
    token: str, key_id: str, project: str, grantee: str, owner_store_path: str
) -> str:
    """HMAC-SHA256 binding the token to (key_id, project, grantee, owner_store_path).
    Including owner_store_path prevents a recipient from swapping the store path
    in the share string to bypass revocation — the HMAC would no longer verify.
    """
    message = f"{key_id}:{project}:{grantee}:{owner_store_path}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


def generate_share_key(
    owner_user_dir: Path,
    project: str,
    grantee: str,
    *,
    ttl_days: int | None = None,
) -> dict[str, Any]:
    """Generate a share key allowing grantee to access owner's project.
    All shares are read-only. The grantee can query but not ingest or delete.
    Args:
        owner_user_dir: Path to the owner's user directory under AxonStore.
        project: Name of the project to share (must exist).
        grantee: OS username of the recipient.
        ttl_days: Optional time-to-live in days. When set, the share key
            (and the grantee's mount descriptor) automatically expire
            ``ttl_days`` after creation; grantees can no longer redeem
            or query the share. Owners can renew via
            :func:`extend_share_key`. ``None`` (default) preserves the
            original "never expires" behaviour.
    Returns:
        Dict with: key_id, share_string, project, grantee, owner,
        expires_at (or ``None`` when ``ttl_days`` was ``None``).
        The share_string should be transmitted to the grantee out-of-band.
    """
    key_id = "sk_" + secrets.token_hex(4)
    token = secrets.token_hex(32)
    owner_name = owner_user_dir.name
    owner_store_path = str(owner_user_dir.parent)
    token_hmac = _compute_hmac(token, key_id, project, grantee, owner_store_path)
    now_dt = _utcnow()
    now = _iso(now_dt)
    expires_at: str | None = None
    if ttl_days is not None:
        if ttl_days <= 0:
            raise ValueError("ttl_days must be a positive integer (or None for no expiry).")
        expires_at = _iso(now_dt + timedelta(days=int(ttl_days)))
    issued_record = {
        "key_id": key_id,
        "project": project,
        "grantee": grantee,
        "token": token,
        "token_hmac": token_hmac,
        "created_at": now,
        "expires_at": expires_at,
        "revoked": False,
        "revoked_at": None,
    }
    manifest_record = {
        "key_id": key_id,
        "project": project,
        "grantee": grantee,
        "created_at": now,
        "expires_at": expires_at,
        "revoked": False,
        "revoked_at": None,
    }
    with _lock:
        # Update private keys file
        keys = _read_json(_keys_path(owner_user_dir))
        keys.setdefault("issued", [])
        keys["issued"].append(issued_record)
        _write_json(_keys_path(owner_user_dir), keys)
        # Update public manifest
        manifest = _read_json(_manifest_path(owner_user_dir))
        manifest.setdefault("issued", [])
        manifest["issued"].append(manifest_record)
        _write_json(_manifest_path(owner_user_dir), manifest)
    # Build share_string: base64(key_id:token:owner:project:owner_store_path)
    # owner_store_path is the AxonStore root (parent of owner_user_dir)
    raw = f"{key_id}:{token}:{owner_name}:{project}:{owner_store_path}"
    share_string = base64.urlsafe_b64encode(raw.encode()).decode()
    return {
        "key_id": key_id,
        "share_string": share_string,
        "project": project,
        "grantee": grantee,
        "owner": owner_name,
        "expires_at": expires_at,
    }


def redeem_share_key(
    grantee_user_dir: Path,
    share_string: str,
) -> dict[str, Any]:
    """Redeem a share_string, creating a mount descriptor in grantee's mounts/.
    A ``mount.json`` descriptor is created under ``mounts/{owner}_{project}/``
    as the canonical, platform-independent record of the mounted project.
    Args:
        grantee_user_dir: Path to the grantee's user directory under AxonStore.
        share_string: The base64 string generated by generate_share_key().
    Returns:
        Dict with: key_id, mount_name, owner, project, descriptor.
    Raises:
        ValueError: If the share_string is invalid, the key is revoked, or
                    the project directory does not exist.
    """
    try:
        raw = base64.urlsafe_b64decode(share_string.encode()).decode()
        key_id, token, owner, project, owner_store_path = raw.split(":", 4)
    except Exception:
        raise ValueError("Invalid share_string format.")
    owner_user_dir = Path(owner_store_path) / owner
    # Resolve nested projects via subs/ layout (e.g. research/papers → research/subs/papers)
    _segments = project.split("/")
    owner_project_dir = owner_user_dir / _segments[0]
    for _seg in _segments[1:]:
        owner_project_dir = owner_project_dir / "subs" / _seg
    if not owner_project_dir.exists():
        raise ValueError(f"Owner project directory does not exist: {owner_project_dir}")
    # Check owner manifest for revocation
    manifest = _read_json(_manifest_path(owner_user_dir))
    issued = manifest.get("issued", [])
    manifest_record = next((r for r in issued if r["key_id"] == key_id), None)
    if manifest_record is None:
        raise ValueError(f"Key '{key_id}' not found in owner's share manifest.")
    if manifest_record.get("revoked"):
        raise ValueError(f"Key '{key_id}' has been revoked by the owner.")
    if _is_expired(manifest_record.get("expires_at")):
        raise ValueError(
            f"Key '{key_id}' expired at {manifest_record.get('expires_at')}. "
            "Ask the owner to extend the share (`/share extend`)."
        )
    # Verify HMAC against owner's private key store
    owner_keys = _read_json(_keys_path(owner_user_dir))
    issued_records = owner_keys.get("issued", [])
    key_record = next((r for r in issued_records if r["key_id"] == key_id), None)
    if key_record is None:
        raise ValueError(f"Key '{key_id}' not found in owner's key store.")
    expected_hmac = _compute_hmac(token, key_id, project, grantee_user_dir.name, owner_store_path)
    if not hmac.compare_digest(key_record["token_hmac"], expected_hmac):
        raise ValueError("Share key HMAC verification failed.")
    mount_name = f"{owner}_{project.replace('/', '_')}"
    # Create descriptor in mounts/ (canonical, platform-independent)
    from axon.mounts import create_mount_descriptor

    descriptor = create_mount_descriptor(
        grantee_user_dir=grantee_user_dir,
        mount_name=mount_name,
        owner=owner,
        project=project,
        owner_user_dir=owner_user_dir,
        target_project_dir=owner_project_dir,
        share_key_id=key_id,
    )
    # Record in grantee's received keys
    received_record = {
        "key_id": key_id,
        "owner": owner,
        "owner_manifest_path": str(_manifest_path(owner_user_dir)),
        "project": project,
        "mount_name": mount_name,
        "redeemed_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        grantee_keys = _read_json(_keys_path(grantee_user_dir))
        grantee_keys.setdefault("received", [])
        # Remove any existing record for the same mount
        grantee_keys["received"] = [
            r for r in grantee_keys["received"] if r.get("mount_name") != mount_name
        ]
        grantee_keys["received"].append(received_record)
        _write_json(_keys_path(grantee_user_dir), grantee_keys)
    return {
        "key_id": key_id,
        "mount_name": mount_name,
        "owner": owner,
        "project": project,
        "descriptor": descriptor,
    }


def revoke_share_key(owner_user_dir: Path, key_id: str) -> dict[str, Any]:
    """Revoke a share key (lazy — does not remove grantee's descriptor immediately).
    The grantee's mount descriptor will be removed the next time they call
    validate_received_shares() or attempt to list/access the shared project.
    Args:
        owner_user_dir: Path to the owner's user directory.
        key_id: The key_id to revoke (e.g. 'sk_a1b2c3d4').
    Returns:
        Dict with: key_id, grantee, project, revoked_at.
    Raises:
        ValueError: If the key is not found or already revoked.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        # Update private keys file
        keys = _read_json(_keys_path(owner_user_dir))
        issued = keys.get("issued", [])
        record = next((r for r in issued if r["key_id"] == key_id), None)
        if record is None:
            raise ValueError(f"Key '{key_id}' not found.")
        if record.get("revoked"):
            raise ValueError(f"Key '{key_id}' is already revoked.")
        record["revoked"] = True
        record["revoked_at"] = now
        _write_json(_keys_path(owner_user_dir), keys)
        # Update public manifest — upsert so grantees always see the revocation
        manifest = _read_json(_manifest_path(owner_user_dir))
        issued = manifest.setdefault("issued", [])
        manifest_record = next((r for r in issued if r["key_id"] == key_id), None)
        if manifest_record is not None:
            manifest_record["revoked"] = True
            manifest_record["revoked_at"] = now
        else:
            # Manifest was out of sync; add a minimal revocation tombstone so
            # redeem_share_key() will correctly reject the key on the next attempt.
            issued.append(
                {
                    "key_id": key_id,
                    "project": record.get("project", ""),
                    "grantee": record.get("grantee", ""),
                    "revoked": True,
                    "revoked_at": now,
                }
            )
            import logging as _logging

            _logging.getLogger("AxonShares").warning(
                "revoke_share_key: key %s absent from manifest; tombstone added", key_id
            )
        _write_json(_manifest_path(owner_user_dir), manifest)
    return {
        "key_id": key_id,
        "grantee": record.get("grantee"),
        "project": record.get("project"),
        "revoked_at": now,
    }


def list_shares(user_dir: Path) -> dict[str, list]:
    """Return all active shares for a user — both issued (sharing) and received (shared).
    Args:
        user_dir: Path to the user's directory under AxonStore.
    Returns:
        Dict with 'sharing' (keys this user has issued) and 'shared' (keys received).
    """
    keys = _read_json(_keys_path(user_dir))
    sharing = [
        {
            "key_id": r["key_id"],
            "project": r["project"],
            "grantee": r["grantee"],
            "revoked": r.get("revoked", False),
            "created_at": r.get("created_at"),
            "expires_at": r.get("expires_at"),
            "expired": _is_expired(r.get("expires_at")),
        }
        for r in keys.get("issued", [])
    ]
    shared = [
        {
            "key_id": r["key_id"],
            "owner": r["owner"],
            "project": r["project"],
            "mount": r.get("mount_name"),
            "redeemed_at": r.get("redeemed_at"),
        }
        for r in keys.get("received", [])
    ]
    return {"sharing": sharing, "shared": shared}


def validate_received_shares(user_dir: Path) -> list[str]:
    """Check all received shares for revocation; update descriptors and remove stale symlinks.
    Scans received share records against the owner's manifest.  When a share has
    been revoked, the ``mounts/`` descriptor is removed (primary).
    Args:
        user_dir: Path to the grantee's user directory.
    Returns:
        List of mount names that were removed due to revocation.
    """
    from axon.mounts import remove_mount_descriptor

    keys = _read_json(_keys_path(user_dir))
    received = keys.get("received", [])
    removed = []
    updated = False
    for record in list(received):
        manifest_path = Path(record.get("owner_manifest_path", ""))
        if not manifest_path.exists():
            continue  # Can't check — leave descriptor in place
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key_id = record.get("key_id")
        if not key_id:
            continue  # skip malformed received record
        manifest_record = next(
            (r for r in manifest.get("issued", []) if r["key_id"] == key_id), None
        )
        if manifest_record and (
            manifest_record.get("revoked") or _is_expired(manifest_record.get("expires_at"))
        ):
            mount_name = record["mount_name"]
            # Primary: remove descriptor from mounts/
            remove_mount_descriptor(user_dir, mount_name)
            removed.append(mount_name)
            received.remove(record)
            updated = True
    if updated:
        with _lock:
            keys["received"] = received
            _write_json(_keys_path(user_dir), keys)
    return removed


def extend_share_key(
    owner_user_dir: Path,
    key_id: str,
    *,
    ttl_days: int | None,
) -> dict[str, Any]:
    """Renew (or remove) a share key's expiry.
    Args:
        owner_user_dir: Path to the owner's user directory.
        key_id: The key_id to extend (e.g. ``'sk_a1b2c3d4'``).
        ttl_days: New time-to-live in days, measured from *now*.
            ``None`` clears the expiry entirely (key never expires until
            revoked).
    Returns:
        Dict with: key_id, project, grantee, expires_at (the new value,
        or ``None`` when cleared).
    Raises:
        ValueError: If the key is not found or has been revoked.
    """
    if ttl_days is not None and ttl_days <= 0:
        raise ValueError("ttl_days must be a positive integer (or None to clear expiry).")
    now_dt = _utcnow()
    new_expires_at: str | None = None
    if ttl_days is not None:
        new_expires_at = _iso(now_dt + timedelta(days=int(ttl_days)))
    with _lock:
        # Update private keys file
        keys = _read_json(_keys_path(owner_user_dir))
        issued = keys.get("issued", [])
        record = next((r for r in issued if r["key_id"] == key_id), None)
        if record is None:
            raise ValueError(f"Key '{key_id}' not found.")
        if record.get("revoked"):
            raise ValueError(
                f"Key '{key_id}' is revoked; revoked keys cannot be extended. "
                "Generate a new share key instead."
            )
        record["expires_at"] = new_expires_at
        _write_json(_keys_path(owner_user_dir), keys)
        # Mirror to the public manifest so grantees see the new expiry.
        manifest = _read_json(_manifest_path(owner_user_dir))
        issued_m = manifest.setdefault("issued", [])
        manifest_record = next((r for r in issued_m if r["key_id"] == key_id), None)
        if manifest_record is not None:
            manifest_record["expires_at"] = new_expires_at
        else:
            issued_m.append(
                {
                    "key_id": key_id,
                    "project": record.get("project", ""),
                    "grantee": record.get("grantee", ""),
                    "created_at": record.get("created_at"),
                    "expires_at": new_expires_at,
                    "revoked": False,
                    "revoked_at": None,
                }
            )
        _write_json(_manifest_path(owner_user_dir), manifest)
    return {
        "key_id": key_id,
        "project": record.get("project"),
        "grantee": record.get("grantee"),
        "expires_at": new_expires_at,
    }
