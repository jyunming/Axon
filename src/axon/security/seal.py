"""``axon project seal <name>`` — encrypt every content file in a project (Phase 2).

Walks an existing plaintext project directory and rewrites each
content file as AXSL-sealed ciphertext using the project's DEK. The
operation is **atomic per file**: each file is written to a sibling
``<name>.sealing`` tempfile, fsynced, then ``os.replace``'d over the
live name — so a crash mid-seal leaves the original file intact (or
fully sealed) but never partially overwritten.

What gets sealed (per ``docs/SHARE_MOUNT_SEALED.md`` §4.6):
  - ``meta.json``
  - everything under ``bm25_index/``
  - everything under ``vector_store_data/``
  - the dynamic-graph snapshot (``.dynamic_graph.snapshot.json``)

What stays plaintext:
  - ``version.json`` (grantees need to detect changes without the DEK)
  - ``.security/`` (the wrap files themselves, plus the sealed marker)
  - ``store_meta.json`` (AxonStore root metadata)
  - Anything else not in a content directory (config files, logs, etc.)

After a successful seal, ``.security/.sealed`` is written as a
plaintext JSON marker so :func:`is_project_sealed` can probe the
state without any key material.

Idempotency:
  - If ``.security/.sealed`` already exists, ``project_seal`` returns
    ``{"status": "already_sealed"}`` without touching files.
  - If individual files are already AXSL-headered (mid-seal crash
    recovery), they're skipped and counted as ``files_already_sealed``.
  - Leftover ``.sealing`` orphans from a crashed prior attempt are
    removed before sealing begins.

Phase 2 part 3. Replaces the ``project_seal`` and
``get_sealed_project_record`` stubs in ``axon/security/__init__.py``.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from . import SecurityError
from .cache import is_sealed_file
from .crypto import SealedFile, make_aad
from .master import get_or_create_project_dek

logger = logging.getLogger("Axon")

__all__ = [
    "SEALED_MARKER_PATH",
    "SEAL_SCHEMA_VERSION",
    "CIPHER_SUITE_AES_256_GCM_V1",
    "is_project_sealed",
    "read_sealed_marker",
    "project_seal",
    "get_sealed_project_record",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEALED_MARKER_PATH: str = ".security/.sealed"
SEAL_SCHEMA_VERSION: int = 1
CIPHER_SUITE_AES_256_GCM_V1: str = "AES-256-GCM-v1"

# Content directories whose every file gets sealed. Anything OUTSIDE
# this set + the explicit single-file allowlist below stays plaintext.
_SEAL_CONTENT_DIRS: frozenset[str] = frozenset(
    {
        "bm25_index",
        "vector_store_data",
    }
)

# Single files at the project root that get sealed.
_SEAL_ROOT_FILES: frozenset[str] = frozenset(
    {
        "meta.json",
        ".dynamic_graph.snapshot.json",  # may live under bm25_index too — caught by dir rule
    }
)

# Explicitly NEVER seal — these need to stay plaintext for the
# refresh-marker / store-meta probes to work without unlocking.
_NEVER_SEAL_DIRS: frozenset[str] = frozenset({".security"})
_NEVER_SEAL_NAMES: frozenset[str] = frozenset(
    {
        "version.json",
        "store_meta.json",
        ".sealed",
    }
)


def _should_seal(rel: Path) -> bool:
    """Decide whether to encrypt a file at *rel* (relative to project_dir)."""
    if rel.name in _NEVER_SEAL_NAMES:
        return False
    if any(part in _NEVER_SEAL_DIRS for part in rel.parts):
        return False
    if rel.parts and rel.parts[0] in _SEAL_CONTENT_DIRS:
        return True
    if len(rel.parts) == 1 and rel.name in _SEAL_ROOT_FILES:
        return True
    return False


# ---------------------------------------------------------------------------
# Sealed marker
# ---------------------------------------------------------------------------


def is_project_sealed(project_dir: Path | str) -> bool:
    """Return True iff ``<project>/.security/.sealed`` exists.

    Cheap probe — does not require an unlocked store. Used by
    :func:`get_sealed_project_record` and by api_routes/shares to
    decide whether the sealed-share code path applies to a project.
    """
    return (Path(project_dir) / SEALED_MARKER_PATH).is_file()


def read_sealed_marker(project_dir: Path | str) -> dict[str, Any] | None:
    """Read the sealed marker JSON; return ``None`` when absent.

    Raises ``SecurityError`` if the file exists but is malformed —
    that's a defence-in-depth signal worth surfacing to the user
    (someone may have tampered with ``.security/``).
    """
    path = Path(project_dir) / SEALED_MARKER_PATH
    if not path.is_file():
        return None
    try:
        marker: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SecurityError(
            f"Sealed marker at {path} is unreadable / malformed ({exc}). "
            "Manual recovery may be needed."
        ) from exc
    if not isinstance(marker, dict) or marker.get("v") != SEAL_SCHEMA_VERSION:
        raise SecurityError(
            f"Sealed marker schema_version mismatch at {path} "
            f"(expected {SEAL_SCHEMA_VERSION}, got {marker.get('v') if isinstance(marker, dict) else type(marker).__name__})."
        )
    return marker


def _write_sealed_marker(
    project_dir: Path, *, seal_id: str, files_sealed: int, files_skipped: int
) -> None:
    """Write the plaintext sealed marker after a successful seal."""
    from datetime import datetime, timezone

    marker = {
        "v": SEAL_SCHEMA_VERSION,
        "cipher_suite": CIPHER_SUITE_AES_256_GCM_V1,
        "seal_id": seal_id,
        "sealed_at": datetime.now(tz=timezone.utc).isoformat(),
        "files_sealed": files_sealed,
        "files_skipped": files_skipped,
    }
    target = project_dir / SEALED_MARKER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".sealing")
    tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# Crash-recovery: clean up .sealing tempfiles from a prior failed attempt
# ---------------------------------------------------------------------------


def _cleanup_sealing_orphans(project_dir: Path) -> int:
    """Delete any ``*.sealing`` orphans under *project_dir*.

    A previous ``project_seal`` may have crashed between the tempfile
    write and the ``os.replace``. The original file is intact (atomic
    rename hadn't happened) so the orphan is just dead weight; remove
    it before re-sealing.

    Returns the number of orphans removed.
    """
    removed = 0
    for orphan in project_dir.rglob("*.sealing"):
        try:
            orphan.unlink()
            removed += 1
        except OSError as exc:
            logger.debug("could not remove .sealing orphan %s: %s", orphan, exc)
    return removed


# ---------------------------------------------------------------------------
# project_seal
# ---------------------------------------------------------------------------


def _resolve_project_dir(project_name: str, user_dir: Path) -> Path:
    """Mirror the AxonStore subs/ layout for nested projects.

    ``research/papers`` → ``<user_dir>/research/subs/papers``
    """
    segments = project_name.split("/")
    project_dir = Path(user_dir) / segments[0]
    for seg in segments[1:]:
        project_dir = project_dir / "subs" / seg
    return project_dir


def project_seal(
    project_name: str,
    user_dir: Path,
    *,
    migration_mode: str = "in_place",
    config: Any = None,
    embedding: Any = None,
) -> dict[str, Any]:
    """Encrypt every content file in *project_name* in place.

    Idempotent — if the project is already sealed, returns immediately
    with ``status="already_sealed"``. On a crashed prior attempt, any
    individually-sealed files are kept (skipped on re-run) and any
    ``.sealing`` orphans are removed.

    Args:
        project_name: Project name as used in the AxonStore (supports
            the ``parent/child`` nested form).
        user_dir: AxonStore user directory (``~/.axon/AxonStore/<owner>``).
        migration_mode: Reserved for future variants — only ``"in_place"``
            is implemented in v1. Other values are accepted for
            backward-compat with the existing api_routes signature but
            currently have no effect.
        config: Reserved for future variants (e.g. backend-specific
            re-build during seal).
        embedding: Reserved for future variants.

    Returns:
        ``{"status": "sealed" | "already_sealed", "project": ...,
        "files_sealed": int, "files_skipped": int, "orphans_removed": int}``

    Raises:
        SecurityError: store is locked (call ``unlock_store`` first), or
            project does not exist, or any file write failed.
    """
    project_dir = _resolve_project_dir(project_name, Path(user_dir))
    if not project_dir.is_dir():
        raise SecurityError(
            f"Project does not exist: {project_dir} "
            "(check the project name and that AxonStore is initialised)"
        )

    if is_project_sealed(project_dir):
        existing = read_sealed_marker(project_dir) or {}
        return {
            "status": "already_sealed",
            "project": project_name,
            "seal_id": existing.get("seal_id", ""),
        }

    # Get/create the per-project DEK. Raises SecurityError if the
    # store is locked — surfaces upward to the caller.
    dek = get_or_create_project_dek(Path(user_dir), project_dir)

    # Crash-recovery: drop any leftover .sealing tempfiles before we
    # start writing new ones.
    orphans_removed = _cleanup_sealing_orphans(project_dir)
    if orphans_removed:
        logger.info(
            "project_seal: removed %d leftover .sealing orphan(s) under %s",
            orphans_removed,
            project_dir,
        )

    # Stable seal_id used as the AAD's key_id position so that files
    # cannot be swapped between two projects sealed with the same
    # owner master. Persisted in the marker so the read path can
    # recompute the AAD.
    seal_id = "seal_" + secrets.token_hex(8)

    files_sealed = 0
    files_skipped = 0
    files_already_sealed = 0

    for src in project_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(project_dir)
        if not _should_seal(rel):
            files_skipped += 1
            continue
        if is_sealed_file(src):
            # Resumed after a crash — file already encrypted.
            files_already_sealed += 1
            files_sealed += 1
            continue

        plaintext = src.read_bytes()
        aad = make_aad(seal_id, str(rel).replace("\\", "/"))
        tmp = src.with_suffix(src.suffix + ".sealing")
        try:
            SealedFile.write(tmp, plaintext, dek, aad=aad)
            os.replace(tmp, src)
        except OSError as exc:
            # Clean up the failed tempfile before propagating.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise SecurityError(f"project_seal failed to atomically replace {src}: {exc}") from exc
        files_sealed += 1

    _write_sealed_marker(
        project_dir,
        seal_id=seal_id,
        files_sealed=files_sealed,
        files_skipped=files_skipped,
    )

    logger.info(
        "project_seal: %s sealed (%d files; %d skipped; %d already-sealed; "
        "%d orphans removed; seal_id=%s)",
        project_name,
        files_sealed,
        files_skipped,
        files_already_sealed,
        orphans_removed,
        seal_id,
    )
    return {
        "status": "sealed",
        "project": project_name,
        "seal_id": seal_id,
        "files_sealed": files_sealed,
        "files_already_sealed": files_already_sealed,
        "files_skipped": files_skipped,
        "orphans_removed": orphans_removed,
    }


def get_sealed_project_record(project: str, user_dir: Path) -> dict[str, Any] | None:
    """Cheap probe: return the marker dict if sealed, else ``None``.

    Used by api_routes/shares.py to decide whether a share should go
    through the sealed-share generation path. Does NOT require an
    unlocked store.
    """
    project_dir = _resolve_project_dir(project, Path(user_dir))
    if not is_project_sealed(project_dir):
        return None
    try:
        return read_sealed_marker(project_dir)
    except SecurityError:
        # Marker exists but is malformed — surface as "not sealed" for
        # the share routing decision; the actual unlock attempt will
        # surface the underlying SecurityError with full context.
        return None
