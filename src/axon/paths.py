"""Path classification for share-mount / cloud-sync safety.

Provides predicates used by SQLite-backed components (governance audit,
dynamic graph) to recognise paths that sit on filesystems with unreliable
locking or atomic-rename semantics: consumer cloud-sync folders (OneDrive,
Dropbox, Google Drive), Windows UNC shares, and WSL-mounted Windows drives.

Rationale: SQLite's own maintainers warn that placing a database on such a
filesystem can produce silent corruption (https://sqlite.org/useovernet.html).
Callers use :func:`is_cloud_sync_or_mount_path` to gate WAL-mode usage and
to redirect hot on-disk state to a local-only location.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "is_cloud_sync_path",
    "is_wsl_windows_mount_path",
    "is_unc_path",
    "is_cloud_sync_or_mount_path",
    "cloud_sync_path_reason",
]


# Case-insensitive exact-match path segments used by consumer cloud-sync tools.
# OneDrive for Business uses "OneDrive - <TenantName>" which is matched via the
# prefix tuple below.
_CLOUD_SYNC_SEGMENTS: frozenset[str] = frozenset(
    {
        "onedrive",
        "onedrive - personal",
        "dropbox",
        "google drive",
        "googledrive",
        "my drive",
        "icloud drive",
        "icloud",
    }
)

_CLOUD_SYNC_PREFIXES: tuple[str, ...] = ("onedrive - ",)


def _segments(p: str | os.PathLike[str]) -> list[str]:
    """Return path segments without resolving symlinks (no filesystem I/O)."""
    s = str(p).replace("\\", "/").strip("/")
    return [seg for seg in s.split("/") if seg]


def is_cloud_sync_path(p: str | os.PathLike[str] | None) -> bool:
    """Return ``True`` when *p* is under a consumer cloud-sync folder.

    Matches OneDrive (Personal and Business), Dropbox, Google Drive (Mirror
    and Stream), and iCloud Drive.  Case-insensitive.  Does not hit the
    filesystem.
    """
    if not p:
        return False
    for seg in _segments(p):
        low = seg.lower()
        if low in _CLOUD_SYNC_SEGMENTS:
            return True
        if any(low.startswith(pref) for pref in _CLOUD_SYNC_PREFIXES):
            return True
    return False


def is_unc_path(p: str | os.PathLike[str] | None) -> bool:
    """Return ``True`` for Windows UNC paths (``\\\\server\\share\\...``)."""
    if not p:
        return False
    s = str(p)
    return s.startswith("\\\\") or s.startswith("//")


def is_wsl_windows_mount_path(p: str | os.PathLike[str] | None) -> bool:
    """Return ``True`` for WSL-mounted Windows drives or Windows→WSL paths.

    Matches ``/mnt/<letter>/...`` (WSL seeing a Windows drive — POSIX locking
    broken) and ``//wsl$/<distro>/...`` / ``//wsl.localhost/<distro>/...``
    (Windows seeing a WSL distro — SMB redirector in between).
    """
    if not p:
        return False
    s = str(p).replace("\\", "/")
    if s.startswith("//wsl$/") or s.startswith("//wsl.localhost/"):
        return True
    if s.startswith("/mnt/") and len(s) >= 7 and s[6] == "/":
        return True
    return False


def cloud_sync_path_reason(p: str | os.PathLike[str] | None) -> str:
    """Return a short human-readable reason *p* is considered unsafe, or ``""``."""
    if not p:
        return ""
    if is_cloud_sync_path(p):
        return (
            "path is inside a consumer cloud-sync folder "
            "(OneDrive/Dropbox/Google Drive); SQLite WAL corrupts on cloud sync"
        )
    if is_unc_path(p):
        return "path is a Windows UNC / network share; file locking is unreliable"
    if is_wsl_windows_mount_path(p):
        return "path is a WSL-mounted Windows drive; POSIX file locking is not honoured"
    return ""


def is_cloud_sync_or_mount_path(p: str | os.PathLike[str] | None) -> bool:
    """Return ``True`` for any path unsafe for SQLite-WAL or atomic rename.

    This is the single predicate callers should use: it unions cloud-sync
    folders, UNC / network shares, and WSL Windows mounts.
    """
    return bool(cloud_sync_path_reason(p))


def safe_local_path(p: Path | str) -> Path:
    """Coerce *p* into a guaranteed-local fallback path rooted at ``~/.axon/``.

    Used by components that want to move hot SQLite state off a synced /
    network path.  Returns ``~/.axon/{basename}`` when *p* is unsafe, else
    returns ``Path(p)`` unchanged.  The caller is responsible for ``mkdir``.

    Cross-platform basename extraction: ``Path("C:/...").name`` on Linux
    returns the whole string because Linux doesn't recognise ``C:\\`` as
    a path separator.  Normalise separators first so the basename is
    correct on both OSes.
    """
    if not is_cloud_sync_or_mount_path(p):
        return Path(p)
    # Normalise both Windows and POSIX separators so the basename is
    # correct regardless of which OS produced the path string.
    normed = str(p).replace("\\", "/").rstrip("/")
    basename = normed.rsplit("/", 1)[-1] if "/" in normed else normed
    return Path.home() / ".axon" / basename
