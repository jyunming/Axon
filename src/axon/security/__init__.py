"""Security primitives for Axon — sealed shares + (Phase 1) crypto foundations.

Top-level package keeps the **existing public stub interface** intact so
every caller that imports from ``axon.security`` continues to work
exactly as before. New cryptographic primitives live in
:mod:`axon.security.crypto` and are imported lazily so users on minimal
installs (no ``cryptography``/``keyring`` packages) keep the existing
behaviour — the sealed-share helpers in this file still raise
``SecurityError("not configured")``.

The plan landing this work in phases is in
``docs/SHARE_MOUNT_SEALED.md``. **Phase 1 (this commit) ships the
crypto + keyring primitives only — no code outside this package uses
them yet.** Behaviour for existing projects is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "SecurityError",
    "store_status",
    "bootstrap_store",
    "unlock_store",
    "lock_store",
    "change_passphrase",
    "is_unlocked",
    "get_sealed_project_record",
    "generate_sealed_share",
    "redeem_sealed_share",
    "validate_received_sealed_shares",
    "list_sealed_shares",
    "resolve_owned_sealed_project_path",
    "project_rotate_keys",
    "project_seal",
]


class SecurityError(Exception):
    """Raised for security operation failures."""


# ---------------------------------------------------------------------------
# Existing stub interface — preserved verbatim from the previous
# axon/security.py module so every existing caller continues to work.
# Real implementations land in later phases per docs/SHARE_MOUNT_SEALED.md.
# ---------------------------------------------------------------------------


def store_status(user_dir: Path) -> dict[str, Any]:
    """Return the current security store status."""
    return {
        "initialized": False,
        "unlocked": False,
        "sealed_hidden_count": 0,
        "public_key_fingerprint": "",
        "cipher_suite": "",
    }


def bootstrap_store(user_dir: Path, passphrase: str) -> dict[str, Any]:
    """Bootstrap the security store with a passphrase."""
    raise SecurityError("Security store not configured")


def unlock_store(user_dir: Path, passphrase: str) -> dict[str, Any]:
    """Unlock the security store."""
    raise SecurityError("unlock failed: store not initialized")


def lock_store(user_dir: Path) -> dict[str, Any]:
    """Lock the security store."""
    return {"initialized": False, "unlocked": False}


def change_passphrase(user_dir: Path, old_passphrase: str, new_passphrase: str) -> dict[str, Any]:
    """Change the store passphrase."""
    raise SecurityError("passphrase change failed: store not initialized")


def is_unlocked(user_dir: Path) -> bool:
    """Return True if the sealed store is currently unlocked."""
    return False


def get_sealed_project_record(project: str, user_dir: Path) -> dict[str, Any] | None:
    """Return the sealed project record if this project is sealed, else None."""
    return None


def generate_sealed_share(
    owner_user_dir: Path,
    project: str,
    grantee: str,
    key_id: str,
) -> dict[str, Any]:
    """Generate a sealed share envelope."""
    raise SecurityError("generate_sealed_share not configured")


def redeem_sealed_share(user_dir: Path, share_string: str) -> dict[str, Any]:
    """Redeem a sealed share string."""
    raise SecurityError("redeem_sealed_share not configured")


def validate_received_sealed_shares(user_dir: Path) -> list[str]:
    """Validate all received sealed shares and remove stale ones."""
    return []


def list_sealed_shares(user_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """List all sealed shares (sharing + shared)."""
    return {"sharing": [], "shared": []}


def resolve_owned_sealed_project_path(project_name: str, user_dir: Path) -> Path:
    """Resolve the on-disk path for a sealed project owned by this user."""
    raise SecurityError(f"Sealed project '{project_name}' not found")


def project_rotate_keys(project_root: Path) -> dict[str, Any]:
    """Rotate the keys for a sealed project."""
    raise SecurityError("project_rotate_keys not configured")


def project_seal(
    project_name: str,
    user_dir: Path,
    *,
    migration_mode: str = "snapshot",
    config: Any = None,
    embedding: Any = None,
) -> dict[str, Any]:
    """Seal an existing open project, converting it to sealed_v1 mode."""
    raise SecurityError("project_seal not configured")
