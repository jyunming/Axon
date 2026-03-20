"""Descriptor-backed mount model for Axon mounted projects.

Each received share is recorded as a ``mount.json`` descriptor under
``{user_dir}/mounts/{mount_name}/``.  This is the canonical, platform-
independent source of truth for mounted projects.

The ``mounts/`` descriptor model is the sole source of truth for received
share mounts.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "mounts_root",
    "mount_descriptor_dir",
    "mount_descriptor_path",
    "create_mount_descriptor",
    "load_mount_descriptor",
    "list_mount_descriptors",
    "remove_mount_descriptor",
    "validate_mount_descriptor",
]

DESCRIPTOR_VERSION = 1


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def mounts_root(user_dir: Path) -> Path:
    """Return the ``mounts/`` directory under *user_dir*."""
    return user_dir / "mounts"


def mount_descriptor_dir(user_dir: Path, mount_name: str) -> Path:
    """Return the per-mount subdirectory under ``mounts/``."""
    return mounts_root(user_dir) / mount_name


def mount_descriptor_path(user_dir: Path, mount_name: str) -> Path:
    """Return the ``mount.json`` path for *mount_name*."""
    return mount_descriptor_dir(user_dir, mount_name) / "mount.json"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_mount_descriptor(
    grantee_user_dir: Path,
    mount_name: str,
    owner: str,
    project: str,
    owner_user_dir: Path,
    target_project_dir: Path,
    share_key_id: str,
) -> dict[str, Any]:
    """Write a ``mount.json`` descriptor and return it.

    Reads ``project_namespace_id`` from the owner's ``meta.json`` and
    ``store_namespace_id`` from the owner's ``store_meta.json`` if available.

    Args:
        grantee_user_dir: Grantee's AxonStore user directory.
        mount_name:        Logical mount identifier (e.g. ``alice_research``).
        owner:             Owner's username.
        project:           Owner's project name.
        owner_user_dir:    Owner's AxonStore user directory.
        target_project_dir: Absolute path to owner's project data directory.
        share_key_id:      The ``key_id`` of the share that was redeemed.

    Returns:
        The descriptor dict that was written to disk.
    """
    project_namespace_id = ""
    store_namespace_id = ""
    try:
        meta = json.loads((target_project_dir / "meta.json").read_text(encoding="utf-8"))
        project_namespace_id = meta.get("project_namespace_id", "")
    except Exception:
        pass
    try:
        store_meta = json.loads((owner_user_dir / "store_meta.json").read_text(encoding="utf-8"))
        store_namespace_id = store_meta.get("store_namespace_id", "")
    except Exception:
        pass

    now = datetime.now(timezone.utc).isoformat()
    descriptor: dict[str, Any] = {
        "mount_name": mount_name,
        "owner": owner,
        "project": project,
        "owner_user_dir": str(owner_user_dir),
        "target_project_dir": str(target_project_dir),
        "project_namespace_id": project_namespace_id,
        "store_namespace_id": store_namespace_id,
        "share_key_id": share_key_id,
        "redeemed_at": now,
        "state": "active",
        "revoked": False,
        "revoked_at": None,
        "readonly": True,
        "descriptor_version": DESCRIPTOR_VERSION,
    }

    desc_dir = mount_descriptor_dir(grantee_user_dir, mount_name)
    desc_dir.mkdir(parents=True, exist_ok=True)
    mount_descriptor_path(grantee_user_dir, mount_name).write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return descriptor


def load_mount_descriptor(user_dir: Path, mount_name: str) -> dict[str, Any] | None:
    """Load and return the descriptor for *mount_name*, or ``None`` if absent/corrupt."""
    path = mount_descriptor_path(user_dir, mount_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception:
        return None


def list_mount_descriptors(user_dir: Path) -> list[dict[str, Any]]:
    """Return all valid, active (non-revoked) mount descriptors under *user_dir*."""
    root = mounts_root(user_dir)
    if not root.exists():
        return []
    results = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        path = entry / "mount.json"
        if not path.exists():
            continue
        try:
            desc = json.loads(path.read_text(encoding="utf-8"))
            if not desc.get("revoked") and desc.get("state") == "active":
                results.append(desc)
        except Exception:
            continue
    return results


def remove_mount_descriptor(user_dir: Path, mount_name: str) -> bool:
    """Remove the descriptor directory for *mount_name*.

    Returns:
        True if the descriptor existed and was removed, False otherwise.
    """
    desc_dir = mount_descriptor_dir(user_dir, mount_name)
    if not desc_dir.exists():
        return False
    shutil.rmtree(desc_dir)
    return True


def validate_mount_descriptor(descriptor: dict[str, Any]) -> tuple[bool, str]:
    """Check whether *descriptor* points to a valid, accessible project.

    Returns:
        ``(True, "")`` when valid, or ``(False, reason)`` when not.
    """
    if descriptor.get("revoked"):
        return False, "mount is revoked"
    if descriptor.get("state") != "active":
        return False, f"mount state is '{descriptor.get('state')}'"
    target = descriptor.get("target_project_dir", "")
    if not target or not Path(target).exists():
        return False, f"target project directory does not exist: {target}"
    return True, ""
