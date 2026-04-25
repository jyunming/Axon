"""Cross-machine staleness detection for shared-filesystem Axon mounts.

The owner writes a ``version.json`` marker LAST after every successful
ingest.  Grantees read it on mount switch (and, in a future iteration,
on a TTL background poll) so they can detect when the owner has re-
indexed while they were idle and avoid serving stale query results.

Marker schema (v1)::

    {
        "schema_version": 1,
        "seq": 42,                     # monotonic per-ingest counter
        "generated_at": "...ISO...",
        "owner_host": "alice-laptop",  # informational
        "hash_algo": "sha256",
        "artifacts": {
            "meta.json":                                "<hex digest>",
            "bm25_index/.bm25_log.jsonl":               "<hex digest>",
            "vector_store_data/manifest.json":          "<hex digest>",
            "bm25_index/.dynamic_graph.snapshot.json":  "<hex digest>",
        }
    }

We hash *manifest-level* files only — the small JSON / log files that
each backend updates atomically when its data changes.  This stays
sub-millisecond for any project size.

Atomicity: temp-file + ``os.replace``.  ``os.replace`` is atomic for
same-filesystem rename on both POSIX and Windows (3.3+ via
``MoveFileExW(MOVEFILE_REPLACE_EXISTING)``), so cross-machine sync
clients can never publish a half-written marker.

This module has no dependencies outside the standard library and is
safe to import everywhere — including from low-level paths that must
not pull in heavyweight optional dependencies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "VERSION_MARKER_FILENAME",
    "SCHEMA_VERSION",
    "MANIFEST_FILES",
    "MountSyncPendingError",
    "bump",
    "read",
    "is_newer_than",
    "rollup_hashes",
    "artifacts_match",
]


class MountSyncPendingError(RuntimeError):
    """Raised when the marker says the owner has advanced but the underlying
    index files haven't fully synced to this grantee yet — typically because
    the owner's ``version.json`` arrived first (it's tiny) and the larger
    index files are still in flight on the cloud-sync client.
    Callers should surface this to the user as a transient "sync in progress"
    condition (e.g. HTTP 503 with ``X-Axon-Mount-Sync-Pending: true``) rather
    than treat it as a hard failure.
    """


logger = logging.getLogger("Axon")

VERSION_MARKER_FILENAME = "version.json"
SCHEMA_VERSION = 1

# Files inside the project directory whose content is hashed into the
# marker.  Each one is the *manifest* / *log* of one backend; changes to
# the underlying data flow through these files because every backend
# writes them atomically as part of its commit. Order is significant
# only for human-readability; we hash each individually so missing files
# are tolerated.
MANIFEST_FILES: tuple[str, ...] = (
    "meta.json",
    "bm25_index/.bm25_log.jsonl",
    "vector_store_data/manifest.json",
    "bm25_index/.dynamic_graph.snapshot.json",
)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_hostname() -> str:
    try:
        return socket.gethostname() or ""
    except Exception:
        return ""


def rollup_hashes(
    project_dir: Path | str,
    *,
    files: tuple[str, ...] = MANIFEST_FILES,
    hash_algo: str = "sha256",
) -> dict[str, str]:
    """Hash every existing manifest file and return ``{relpath: hexdigest}``.
    Missing files are silently skipped — their absence is information
    too: the grantee compares the dicts directly, so a key disappearing
    counts as a change.
    """
    project_dir = Path(project_dir)
    out: dict[str, str] = {}
    for rel in files:
        p = project_dir / rel
        if not p.is_file():
            continue
        try:
            h = hashlib.new(hash_algo)
            with p.open("rb") as fh:
                # Read in 1 MiB chunks so we don't hold whole files in memory.
                for buf in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(buf)
            out[rel] = h.hexdigest()
        except OSError as exc:
            logger.debug("version_marker: could not hash %s: %s", p, exc)
    return out


def bump(
    project_dir: Path | str,
    *,
    seq: int | None = None,
    hash_algo: str = "sha256",
) -> dict[str, Any]:
    """Compute and persist a fresh version marker.
    Args:
        project_dir: The project root (one level above ``bm25_index``
            and ``vector_store_data``).
        seq: Override the auto-incremented sequence number — useful in
            tests.  Production callers should leave this ``None``.
        hash_algo: ``hashlib`` algorithm name.  Defaults to SHA-256;
            BLAKE3 (faster) is a future option once we add the dep.
    Returns:
        The marker dict that was written to disk.
    The write is atomic: we serialise to ``version.json.tmp`` first
    and then ``os.replace`` over the live name, so concurrent readers
    never observe a half-written file.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    if seq is None:
        prev = read(project_dir)
        seq = (prev.get("seq", 0) + 1) if prev else 1
    marker = {
        "schema_version": SCHEMA_VERSION,
        "seq": int(seq),
        "generated_at": _now_iso(),
        "owner_host": _safe_hostname(),
        "hash_algo": hash_algo,
        "artifacts": rollup_hashes(project_dir, hash_algo=hash_algo),
    }
    target = project_dir / VERSION_MARKER_FILENAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    _atomic_replace(tmp, target)
    return marker


def _atomic_replace(src: Path, dst: Path) -> None:
    """``os.replace`` with a copy+unlink fallback for transient locks.
    On Windows + cloud-sync paths (OneDrive, network shares), the live
    target file can be briefly held open by the sync client / file
    indexer, causing ``os.replace`` to raise ``PermissionError`` or
    ``OSError``. The fallback copies the temp file's bytes over the
    live target then unlinks the temp — slower but resilient. Same
    pattern as ``BM25Retriever.save()``.
    On the unlikely path that even the fallback fails, we re-raise the
    original ``os.replace`` error and clean up the orphan ``.tmp`` so
    we don't leave junk behind.
    """
    import shutil

    try:
        os.replace(src, dst)
        return
    except OSError as primary_exc:
        try:
            shutil.copy2(src, dst)
            try:
                src.unlink()
            except OSError:
                pass
            return
        except OSError:
            # Both replace and copy failed. Clean up the temp file
            # if possible and re-raise the original error so the
            # caller sees the most informative failure.
            try:
                src.unlink()
            except OSError:
                pass
            raise primary_exc


def read(project_dir: Path | str) -> dict[str, Any] | None:
    """Return the marker dict at ``project_dir/version.json`` or ``None``.
    Missing or unreadable files return ``None`` (callers treat this as
    "no marker present" — typical for a freshly-created project that
    has not yet been ingested).
    """
    target = Path(project_dir) / VERSION_MARKER_FILENAME
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("version_marker: unreadable %s: %s", target, exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def artifacts_match(project_dir: Path | str, marker: dict[str, Any] | None) -> bool:
    """Return True iff every artifact hash in *marker* matches the on-disk file.
    Used by the grantee-side refresh path to detect the **mid-sync** race:
    the marker says ``seq=N`` but the index files are still at ``seq=N-1``
    because they haven't fully replicated yet. Re-hashing the actual files
    catches this; ``True`` means we're safe to reopen handles, ``False``
    means we should wait + retry (or surface ``MountSyncPendingError``).
    A marker with no artifacts (e.g. very early bump on an empty project)
    trivially matches.
    """
    if not marker:
        return True
    expected = marker.get("artifacts") or {}
    if not expected:
        return True
    hash_algo = marker.get("hash_algo", "sha256")
    actual = rollup_hashes(project_dir, hash_algo=hash_algo)
    return actual == expected


def is_newer_than(current: dict[str, Any] | None, cached: dict[str, Any] | None) -> bool:
    """Return True iff ``current`` represents an ingest later than ``cached``.
    Comparison rule:
      - If we have no cached marker but a current one exists, that's newer.
      - If sequence advanced, that's newer.
      - If sequence is equal but the artifact hash dict differs, that's
        newer too (defensive: catches the unlikely case where the owner
        advanced state without bumping seq, e.g. tests poking the file).
    """
    if cached is None:
        return current is not None
    if current is None:
        return False
    cur_seq = int(current.get("seq", 0) or 0)
    cached_seq = int(cached.get("seq", 0) or 0)
    if cur_seq != cached_seq:
        return cur_seq > cached_seq
    return current.get("artifacts") != cached.get("artifacts")
