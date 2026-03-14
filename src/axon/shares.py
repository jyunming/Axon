"""
AxonStore share key management.

Handles generation, redemption, revocation, and validation of share keys
that allow one user to grant another read or read/write access to a project.

Two files under {user_dir}/.shares/:
  .share_manifest.json  — world-readable (644): revocation status only, no tokens
  .share_keys.json      — private (600): full token records for issued + received keys

Share flow:
  1. Owner calls generate_share_key()  -> gets a share_string to send out-of-band
  2. Grantee calls redeem_share_key()  -> symlink created in grantee's ShareMount/
  3. Owner calls revoke_share_key()    -> marks revoked in manifest (lazy unlink)
  4. Grantee's Axon calls validate_received_shares() on next access -> removes stale links
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _compute_hmac(token: str, key_id: str, project: str, grantee: str) -> str:
    """HMAC-SHA256 binding the token to (key_id, project, grantee).

    Prevents a leaked token from being used for a different project or grantee.
    """
    message = f"{key_id}:{project}:{grantee}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


def generate_share_key(
    owner_user_dir: Path,
    project: str,
    grantee: str,
    write_access: bool = False,
) -> dict[str, Any]:
    """Generate a share key allowing grantee to access owner's project.

    Args:
        owner_user_dir: Path to the owner's user directory under AxonStore.
        project: Name of the project to share (must exist).
        grantee: OS username of the recipient.
        write_access: If True, grantee may ingest into the shared project.

    Returns:
        Dict with: key_id, share_string, project, grantee, write_access.
        The share_string should be transmitted to the grantee out-of-band.
    """
    key_id = "sk_" + secrets.token_hex(4)
    token = secrets.token_hex(32)
    token_hmac = _compute_hmac(token, key_id, project, grantee)
    owner_name = owner_user_dir.name
    now = datetime.now(timezone.utc).isoformat()

    issued_record = {
        "key_id": key_id,
        "project": project,
        "grantee": grantee,
        "write_access": write_access,
        "token": token,
        "token_hmac": token_hmac,
        "created_at": now,
        "revoked": False,
        "revoked_at": None,
    }

    manifest_record = {
        "key_id": key_id,
        "project": project,
        "grantee": grantee,
        "write_access": write_access,
        "created_at": now,
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
    owner_store_path = str(owner_user_dir.parent)
    raw = f"{key_id}:{token}:{owner_name}:{project}:{owner_store_path}"
    share_string = base64.urlsafe_b64encode(raw.encode()).decode()

    return {
        "key_id": key_id,
        "share_string": share_string,
        "project": project,
        "grantee": grantee,
        "write_access": write_access,
        "owner": owner_name,
    }


def redeem_share_key(
    grantee_user_dir: Path,
    share_string: str,
) -> dict[str, Any]:
    """Redeem a share_string, creating a symlink in grantee's ShareMount/.

    Args:
        grantee_user_dir: Path to the grantee's user directory under AxonStore.
        share_string: The base64 string generated by generate_share_key().

    Returns:
        Dict with: mount_name, mount_path, owner, project, write_access.

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
    owner_project_dir = owner_user_dir / project
    # Handle sub-project paths like "research/papers" -> research/subs/papers
    # For simplicity in this implementation, project must be a top-level name
    # (no sub-project sharing for now; sub-projects are shared via parent)
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

    # Verify HMAC against owner's private key store
    owner_keys = _read_json(_keys_path(owner_user_dir))
    issued_records = owner_keys.get("issued", [])
    key_record = next((r for r in issued_records if r["key_id"] == key_id), None)
    if key_record is None:
        raise ValueError(f"Key '{key_id}' not found in owner's key store.")
    expected_hmac = _compute_hmac(token, key_id, project, grantee_user_dir.name)
    if not hmac.compare_digest(key_record["token_hmac"], expected_hmac):
        raise ValueError("Share key HMAC verification failed.")

    write_access = manifest_record.get("write_access", False)

    # Create symlink: grantee/ShareMount/{owner}_{project}
    from axon.projects import _make_share_link

    mount_name = f"{owner}_{project}"
    link_path = grantee_user_dir / "ShareMount" / mount_name
    _make_share_link(owner_project_dir, link_path)

    # Record in grantee's received keys
    received_record = {
        "key_id": key_id,
        "owner": owner,
        "owner_manifest_path": str(_manifest_path(owner_user_dir)),
        "project": project,
        "write_access": write_access,
        "symlink_path": str(link_path),
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
        "mount_name": mount_name,
        "mount_path": str(link_path),
        "owner": owner,
        "project": project,
        "write_access": write_access,
    }


def revoke_share_key(owner_user_dir: Path, key_id: str) -> dict[str, Any]:
    """Revoke a share key (lazy — does not remove grantee's symlink immediately).

    The grantee's symlink will be removed the next time they attempt to access
    the shared project and Axon detects the revoked status.

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

        # Update public manifest
        manifest = _read_json(_manifest_path(owner_user_dir))
        for r in manifest.get("issued", []):
            if r["key_id"] == key_id:
                r["revoked"] = True
                r["revoked_at"] = now
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
            "write_access": r.get("write_access", False),
            "revoked": r.get("revoked", False),
            "created_at": r.get("created_at"),
        }
        for r in keys.get("issued", [])
    ]

    shared = [
        {
            "key_id": r["key_id"],
            "owner": r["owner"],
            "project": r["project"],
            "write_access": r.get("write_access", False),
            "mount": r.get("mount_name"),
            "redeemed_at": r.get("redeemed_at"),
        }
        for r in keys.get("received", [])
    ]

    return {"sharing": sharing, "shared": shared}


def validate_received_shares(user_dir: Path) -> list[str]:
    """Check all received shares for revocation; remove stale symlinks.

    Called on each project list or access attempt. Removes symlinks for
    any share that the owner has revoked.

    Args:
        user_dir: Path to the grantee's user directory.

    Returns:
        List of mount names that were removed due to revocation.
    """
    from axon.projects import _remove_share_link

    keys = _read_json(_keys_path(user_dir))
    received = keys.get("received", [])
    removed = []
    updated = False

    for record in list(received):
        manifest_path = Path(record.get("owner_manifest_path", ""))
        if not manifest_path.exists():
            continue  # Can't check — leave symlink in place
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key_id = record["key_id"]
        manifest_record = next(
            (r for r in manifest.get("issued", []) if r["key_id"] == key_id), None
        )
        if manifest_record and manifest_record.get("revoked"):
            # Remove symlink from grantee's ShareMount
            link = Path(record["symlink_path"])
            _remove_share_link(link)
            removed.append(record["mount_name"])
            received.remove(record)
            updated = True

    if updated:
        with _lock:
            keys["received"] = received
            _write_json(_keys_path(user_dir), keys)

    return removed
