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
    """Return the current security store status.

    Phase 2 (PR #57) — wired to the on-disk master record via
    :mod:`axon.security.master`. When the optional ``[sealed]`` extra
    is not installed, falls back to the v0 stub response so callers
    on minimal installs keep working.
    """
    try:
        from . import master as _m
    except ImportError:
        return {
            "initialized": False,
            "unlocked": False,
            "sealed_hidden_count": 0,
            "public_key_fingerprint": "",
            "cipher_suite": "",
        }

    initialized = _m.is_bootstrapped(user_dir)
    return {
        "initialized": initialized,
        "unlocked": _m.is_unlocked(user_dir) if initialized else False,
        "sealed_hidden_count": 0,
        "public_key_fingerprint": "",
        "cipher_suite": "AES-256-GCM-v1" if initialized else "",
    }


def bootstrap_store(user_dir: Path, passphrase: str) -> dict[str, Any]:
    """Bootstrap the security store with a passphrase.

    Phase 2 (PR #57) — wired to :mod:`axon.security.master`.
    Requires the ``[sealed]`` extra (``pip install axon-rag[sealed]``)
    so the ``cryptography`` + ``keyring`` dependencies are present.
    """
    try:
        from . import master as _m
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _m.bootstrap_store(user_dir, passphrase)


def unlock_store(user_dir: Path, passphrase: str) -> dict[str, Any]:
    """Unlock the security store.

    Phase 2 (PR #57) — wired to :mod:`axon.security.master`. Raises
    :class:`axon.security.master.BadPassphraseError` (a SecurityError
    subclass) on a wrong passphrase.
    """
    try:
        from . import master as _m
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _m.unlock_store(user_dir, passphrase)


def lock_store(user_dir: Path) -> dict[str, Any]:
    """Lock the security store.

    Phase 2 (PR #57) — wired to :mod:`axon.security.master`. Always
    returns successfully even when nothing was unlocked, so callers
    can use this as a no-throw cleanup hook on shutdown.
    """
    try:
        from . import master as _m
    except ImportError:
        return {"initialized": False, "unlocked": False}
    return _m.lock_store(user_dir)


def change_passphrase(user_dir: Path, old_passphrase: str, new_passphrase: str) -> dict[str, Any]:
    """Change the store passphrase.

    Phase 2 (PR #57) — wired to :mod:`axon.security.master`. Project
    DEKs are not touched (they're wrapped under the master, not the
    passphrase), so this is O(1) regardless of how many sealed
    projects the owner has.
    """
    try:
        from . import master as _m
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _m.change_passphrase(user_dir, old_passphrase, new_passphrase)


def is_unlocked(user_dir: Path) -> bool:
    """Return True if the sealed store is currently unlocked.

    Phase 2 (PR #57) — wired to :mod:`axon.security.master`.
    """
    try:
        from . import master as _m
    except ImportError:
        return False
    return _m.is_unlocked(user_dir)


def get_sealed_project_record(project: str, user_dir: Path) -> dict[str, Any] | None:
    """Return the sealed project record if this project is sealed, else None.

    Phase 2 (PR #57) — wired to the on-disk ``.security/.sealed`` marker
    via :mod:`axon.security.seal`.
    """
    from .seal import get_sealed_project_record as _impl

    return _impl(project, user_dir)


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
    migration_mode: str = "in_place",
    config: Any = None,
    embedding: Any = None,
) -> dict[str, Any]:
    """Seal an existing open project, converting it to sealed_v1 mode.

    Phase 2 (PR #57) — wired to the in-place per-file AES-256-GCM
    sealer in :mod:`axon.security.seal`. ``migration_mode`` /
    ``config`` / ``embedding`` are reserved for future variants and
    have no effect in v1; the parameters are preserved so existing
    callers (api_routes/projects.py) keep working.
    """
    from .seal import project_seal as _impl

    return _impl(
        project_name,
        user_dir,
        migration_mode=migration_mode,
        config=config,
        embedding=embedding,
    )
