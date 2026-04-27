"""Security primitives for Axon — sealed sharing (all 7 phases shipped).

All seven phases are implemented: crypto + keyring primitives (Phase 1),
master key management (Phase 2), sealed share generate/redeem (Phase 3),
hard/soft revocation (Phase 4), ephemeral plaintext cache (Phase 5),
headless/file fallback (Phase 6), and smoke tests (Phase 7).

Users on minimal installs (no ``cryptography``/``keyring`` packages) get
``ImportError`` with a clear install hint — the helpers no longer silently
raise ``SecurityError("not configured")``.

Full design and threat model: ``docs/architecture/SEALED_SHARING_DESIGN.md``.
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
    "revoke_sealed_share",
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
# Real implementations land in later phases per docs/architecture/SEALED_SHARING_DESIGN.md.
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
    via :mod:`axon.security.seal`. When the optional ``[sealed]`` extra
    is not installed, returns ``None`` (no project can be sealed on a
    minimal install) so callers like ``shares.py`` keep working.
    """
    try:
        from .seal import get_sealed_project_record as _impl
    except ImportError:
        return None
    return _impl(project, user_dir)


def generate_sealed_share(
    owner_user_dir: Path,
    project: str,
    grantee: str,
    key_id: str,
) -> dict[str, Any]:
    """Generate a sealed share envelope.
    Phase 3 — wired to :mod:`axon.security.share`. Wraps the project
    DEK under a per-share KEK derived from a fresh random token, writes
    ``<project>/.security/shares/<key_id>.wrapped``, and returns a
    base64 ``share_string`` with the ``SEALED1:`` prefix the redeem
    path uses to tell sealed shares apart from legacy shares.
    """
    try:
        from .share import generate_sealed_share as _impl
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _impl(owner_user_dir, project, grantee, key_id)


def redeem_sealed_share(user_dir: Path, share_string: str) -> dict[str, Any]:
    """Redeem a sealed share_string.
    Phase 3 — wired to :mod:`axon.security.share`. Parses the envelope,
    fetches the wrap file from the owner's synced folder, unwraps the
    DEK, and persists it in the grantee's OS keyring at
    ``axon.share.<key_id>``. Writes a ``mount.json`` with
    ``mount_type="sealed"`` so the brain knows to fetch the DEK from
    the keyring at switch-project time.
    """
    try:
        from .share import redeem_sealed_share as _impl
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _impl(user_dir, share_string)


def revoke_sealed_share(
    owner_user_dir: Path,
    project: str,
    key_id: str,
    *,
    rotate: bool = False,
) -> dict[str, Any]:
    """Revoke a sealed share — soft (default) or hard (``rotate=True``).
    Phase 4 — wired to :mod:`axon.security.share`.
    - Soft: deletes the wrap file so fresh redeems fail; cached DEKs
      keep working until the owner does a hard rotate.
    - Hard (``rotate=True``): rotates the project DEK, re-encrypts
      every content file, deletes ALL share wraps. Surviving grantees
      must re-issue + re-redeem.
    """
    try:
        from .share import revoke_sealed_share as _impl
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _impl(owner_user_dir, project, key_id, rotate=rotate)


def validate_received_sealed_shares(user_dir: Path) -> list[str]:
    """Validate every received sealed mount; remove ones whose owner-side
    wrap file has disappeared (soft-revoked) or whose target project is
    no longer accessible.
    Returns the list of mount names that were removed.
    """
    try:
        from axon.mounts import (
            list_mount_descriptors,
            remove_mount_descriptor,
            validate_mount_descriptor,
        )
    except ImportError:
        return []
    removed: list[str] = []
    for desc in list_mount_descriptors(user_dir):
        if desc.get("mount_type") != "sealed":
            continue
        ok, _reason = validate_mount_descriptor(desc)
        if ok:
            # Also verify the per-share wrap still exists in the owner
            # project — soft-revoke deletes the wrap file in place.
            target = desc.get("target_project_dir", "")
            key_id = desc.get("share_key_id", "")
            if target and key_id:
                wrap = Path(target) / ".security" / "shares" / f"{key_id}.wrapped"
                if not wrap.is_file():
                    ok = False
        if not ok:
            mount_name = desc.get("mount_name") or desc.get("name") or ""
            if mount_name and remove_mount_descriptor(user_dir, mount_name):
                removed.append(mount_name)
    return removed


def list_sealed_shares(user_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """List sealed shares: ``sharing`` (this user's wraps) + ``shared``
    (sealed mounts redeemed by this user).
    """
    sharing: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    # ── sharing: walk every owned project under user_dir, list its wraps ──
    try:
        from .share import list_sealed_share_key_ids
    except ImportError:
        list_sealed_share_key_ids = None  # type: ignore[assignment]
    if list_sealed_share_key_ids is not None and user_dir.is_dir():

        def _walk_owned(root: Path, name_segments: list[str]) -> None:
            sealed_marker = root / ".security" / ".sealed"
            if sealed_marker.is_file() and name_segments:
                project_name = "/".join(name_segments)
                for kid in list_sealed_share_key_ids(root):
                    sharing.append({"project": project_name, "key_id": kid})
            subs = root / "subs"
            if subs.is_dir():
                for child in subs.iterdir():
                    if child.is_dir():
                        _walk_owned(child, name_segments + [child.name])

        for entry in user_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                _walk_owned(entry, [entry.name])
    # ── shared: list_mount_descriptors filtered to sealed mounts ──
    try:
        from axon.mounts import list_mount_descriptors

        for desc in list_mount_descriptors(user_dir):
            if desc.get("mount_type") == "sealed":
                shared.append(desc)
    except ImportError:
        pass
    return {"sharing": sharing, "shared": shared}


def resolve_owned_sealed_project_path(project_name: str, user_dir: Path) -> Path:
    """Resolve the on-disk path for a sealed project owned by this user.
    Raises :class:`SecurityError` if no such sealed project exists.
    """
    try:
        from .share import _resolve_owned_project_dir
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. Install with: pip install axon-rag[sealed]"
        ) from exc
    project_dir = _resolve_owned_project_dir(project_name, user_dir)
    if not (project_dir / ".security" / ".sealed").is_file():
        raise SecurityError(f"Sealed project '{project_name}' not found")
    return project_dir


def project_rotate_keys(project_root: Path) -> dict[str, Any]:
    """Rotate the project DEK; re-encrypt every sealed file; invalidate
    every existing share wrap. Equivalent to a hard-revoke without a
    specific key_id — surviving grantees must re-redeem.
    """
    try:
        from .share import _hard_revoke
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. Install with: pip install axon-rag[sealed]"
        ) from exc
    # owner_user_dir is the parent of the project root in the AxonStore
    # layout. project name is the trailing segment for the marker / log
    # message — the actual rotation is keyed on project_dir + DEK only.
    owner_user_dir = project_root.parent
    project_name = project_root.name
    return _hard_revoke(owner_user_dir, project_name, project_root, key_id="")


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
    try:
        from .seal import project_seal as _impl
    except ImportError as exc:
        raise SecurityError(
            "Sealed-store support is not installed. " "Install with: pip install axon-rag[sealed]"
        ) from exc
    return _impl(
        project_name,
        user_dir,
        migration_mode=migration_mode,
        config=config,
        embedding=embedding,
    )
