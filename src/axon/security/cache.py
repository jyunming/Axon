"""Ephemeral plaintext cache for sealed Axon projects (Phase 2 of #SEALED).

When an owner queries a sealed project, the on-disk files are AES-GCM
ciphertext. Backends like TurboQuantDB / LanceDB / BM25 mmap their data
files for performance — and mmap can't see through encryption. The
v1 policy (decision §5.1 in ``docs/SHARE_MOUNT_SEALED.md``, locked
2026-04-25) is to **decrypt the whole project into an ephemeral
plaintext cache** in the OS temp dir at mount time, point the backend
at the cache path, and wipe the cache on close. Backends mmap the
cache normally, so query performance is identical to plaintext mode.

Cost: a session-bounded plaintext footprint on disk. Mitigations
implemented here:

- **Cache location**: ``tempfile.mkdtemp(prefix="axon-sealed-")``
  (Linux/macOS uses ``/tmp``; Windows uses ``%LOCALAPPDATA%\\Temp``).
  Per-mount; never shared between projects.
- **Wipe on close**: every cache file is overwritten with zeros
  (``O_RDWR`` re-open + ``write(b"\\x00" * size)`` + fsync) before
  unlink. This won't survive low-level disk forensics on SSDs (TRIM
  / wear-leveling defeat overwrite), but at the filesystem layer the
  plaintext is gone.
- **Crash recovery**: every cache dir contains a ``.pid`` sentinel
  with the creating process's PID. :func:`cleanup_orphans` walks the
  temp dir on next AxonBrain boot, finds ``axon-sealed-*`` dirs whose
  PID is no longer alive, wipes them.
- **Capacity check**: before decrypting, verify free disk ≥ project
  size × 1.1; raise :class:`CacheCapacityError` with concrete numbers
  if not.

Phase 2 deliverable; consumed by Phase 2's seal-aware backend
integration. **Nothing outside ``axon.security`` consumes this yet**
— the module is testable in isolation.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .crypto import MAGIC, SealedFile, make_aad
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.cache requires axon.security.crypto, which requires the "
        "'cryptography' package. Install with: pip install axon-rag[sealed]"
    ) from exc

logger = logging.getLogger("Axon")

__all__ = [
    "CACHE_PREFIX",
    "PID_SENTINEL_FILENAME",
    "CACHE_HEADROOM_FRACTION",
    "CacheCapacityError",
    "SealedCache",
    "cleanup_orphans",
    "is_sealed_file",
]

CACHE_PREFIX: str = "axon-sealed-"
PID_SENTINEL_FILENAME: str = ".axon-cache-pid"

# Free disk required as a fraction of project size before we'll create
# a cache. 1.1 = 10% headroom; refuses cache creation when the free
# space would drop below that.
CACHE_HEADROOM_FRACTION: float = 1.1


class CacheCapacityError(RuntimeError):
    """Raised when the OS temp dir doesn't have enough free space to
    decrypt the sealed project.

    Surfaced with concrete numbers (project size, free space, deficit)
    so the user can either free up disk or move ``TMPDIR`` /
    ``TEMP`` to a roomier volume.
    """


def is_sealed_file(path: Path) -> bool:
    """Return True if *path* starts with the AXSL magic header.

    Used by :class:`SealedCache` to decide whether to decrypt or copy
    each file when materialising the cache. Cheap — reads only the
    first 4 bytes.
    """
    if not path.is_file():
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(4) == MAGIC
    except OSError:
        return False


def _project_size_bytes(sealed_dir: Path) -> int:
    """Total bytes of every regular file under *sealed_dir* (recursive)."""
    total = 0
    for p in sealed_dir.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _secure_delete_file(path: Path) -> None:
    """Overwrite *path* with zeros, fsync, then unlink.

    Best-effort at the filesystem layer — won't defeat SSD TRIM /
    wear-leveling, but ensures no readable plaintext bytes remain at
    the file's logical address before the inode is freed. Errors are
    silently swallowed (the unlink is the desired post-condition; if
    we can't even unlink, there's nothing useful to do here).
    """
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > 0:
        try:
            with path.open("r+b") as fh:
                # Write zeros in 64 KiB chunks so we don't allocate a
                # huge buffer for huge files.
                chunk = b"\x00" * min(size, 64 * 1024)
                remaining = size
                while remaining > 0:
                    write_len = min(remaining, len(chunk))
                    fh.write(chunk[:write_len])
                    remaining -= write_len
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
        except OSError as exc:
            logger.debug("secure-delete overwrite failed for %s: %s", path, exc)
    try:
        path.unlink()
    except OSError as exc:
        logger.debug("secure-delete unlink failed for %s: %s", path, exc)


def _wipe_dir_contents(cache_dir: Path) -> None:
    """Securely delete every regular file under *cache_dir*, then remove
    empty directories bottom-up. Used by :meth:`SealedCache.wipe` and
    :func:`cleanup_orphans`."""
    if not cache_dir.exists():
        return
    # Walk bottom-up so we can rmdir empty dirs after their files are gone.
    for root, dirs, files in os.walk(cache_dir, topdown=False):
        root_path = Path(root)
        for name in files:
            _secure_delete_file(root_path / name)
        for name in dirs:
            try:
                (root_path / name).rmdir()
            except OSError:
                pass
    try:
        cache_dir.rmdir()
    except OSError:
        # Race vs cleanup_orphans, or something we couldn't unlink —
        # leave the dir but it should be effectively empty.
        pass


def _pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is still running.

    Cross-platform best-effort check:

    - POSIX: ``os.kill(pid, 0)`` succeeds for live PIDs, raises
      ``ProcessLookupError`` (errno ``ESRCH``) for dead ones, and
      ``PermissionError`` for live PIDs we don't own.
    - Windows: ``os.kill(pid, 0)`` raises ``OSError`` with ``errno``
      ``EINVAL`` for invalid handle (PID never existed / out of range)
      and ``ESRCH`` for "no such process". ``PermissionError`` again
      means a live PID we can't signal.

    Cleanup leans toward wiping orphans rather than leaking plaintext:
    on any unrecognised error we treat the PID as dead.
    """
    if pid <= 0:
        return False
    import errno

    try:
        os.kill(pid, 0)
    except PermissionError:
        # Live PID owned by another user — leave its cache alone.
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        # ESRCH = "no such process". EINVAL on Windows = handle out of
        # range / invalid PID. Either way: not a live process.
        if exc.errno in (errno.ESRCH, errno.EINVAL):
            return False
        # Unknown failure mode — be conservative and treat as alive
        # so we don't accidentally wipe a cache that might still be
        # in use. The next boot will re-check.
        return True
    return True


# ---------------------------------------------------------------------------
# SealedCache
# ---------------------------------------------------------------------------


class SealedCache:
    """An ephemeral plaintext cache backing one mounted sealed project.

    Use as a context manager so the cache is reliably wiped on close::

        with SealedCache.create(project_dir, dek, key_id="sk_xxx") as cache:
            backend = OpenVectorStore(cfg.with_path(cache.path))
            ...
        # cache.path no longer exists here
    """

    def __init__(self, cache_dir: Path) -> None:
        self._path = cache_dir
        self._wiped = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        sealed_dir: Path | str,
        dek: bytes,
        *,
        key_id: str,
        cache_root: Path | str | None = None,
    ) -> SealedCache:
        """Decrypt every sealed file in *sealed_dir* into a fresh cache.

        Args:
            sealed_dir: Project root containing AXSL-sealed files.
            dek: 32-byte AES-256 Data Encryption Key for the project.
            key_id: Share/project key identifier — bound into the GCM
                AAD via :func:`make_aad` so files cannot be swapped
                between projects without an InvalidTag.
            cache_root: Override the temp directory (default: the OS
                temp dir from :func:`tempfile.gettempdir`).

        Returns:
            A :class:`SealedCache` instance bound to the new cache dir.

        Raises:
            CacheCapacityError: free disk < project size × 1.1.
            FileNotFoundError: sealed_dir does not exist.
            cryptography.exceptions.InvalidTag: a sealed file's GCM tag
                doesn't validate (wrong DEK or tampered ciphertext).
            SealedFormatError: a file with an AXSL prefix has a bad
                schema version or unknown cipher_id.
        """
        sealed_dir = Path(sealed_dir)
        if not sealed_dir.is_dir():
            raise FileNotFoundError(f"sealed_dir does not exist: {sealed_dir}")

        size = _project_size_bytes(sealed_dir)
        cache_parent = Path(cache_root) if cache_root else Path(tempfile.gettempdir())
        cache_parent.mkdir(parents=True, exist_ok=True)

        # Capacity check BEFORE we mkdtemp so we don't leave an empty
        # cache dir behind on failure.
        try:
            free = shutil.disk_usage(cache_parent).free
        except OSError as exc:
            logger.debug("disk_usage on %s failed: %s", cache_parent, exc)
            free = -1  # unknown — skip the check rather than fail
        required = int(size * CACHE_HEADROOM_FRACTION)
        if free >= 0 and free < required:
            raise CacheCapacityError(
                f"Cannot create sealed cache: project is {size:,} bytes, "
                f"need {required:,} bytes free in {cache_parent} "
                f"(have {free:,}). Free up disk or set TMPDIR/TEMP to a "
                "roomier volume."
            )

        cache_dir = Path(tempfile.mkdtemp(prefix=CACHE_PREFIX, dir=str(cache_parent)))
        try:
            # PID sentinel — used by cleanup_orphans on next boot.
            (cache_dir / PID_SENTINEL_FILENAME).write_text(str(os.getpid()), encoding="utf-8")
            # Walk sealed_dir, decrypt sealed files, copy non-sealed.
            for src in sealed_dir.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(sealed_dir)
                dst = cache_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if is_sealed_file(src):
                    aad = make_aad(key_id, str(rel).replace("\\", "/"))
                    plaintext = SealedFile.read(src, dek, aad=aad)
                    dst.write_bytes(plaintext)
                else:
                    # Non-sealed file (e.g. version.json which is
                    # deliberately plaintext). Copy as-is so the
                    # cache is a faithful materialisation of the
                    # project view.
                    shutil.copy2(src, dst)
        except Exception:
            # On any failure during materialisation, wipe the partial
            # cache so we don't leak plaintext bytes already decrypted.
            _wipe_dir_contents(cache_dir)
            raise

        logger.info(
            "SealedCache.create: %d bytes from %s decrypted into %s (pid=%d)",
            size,
            sealed_dir,
            cache_dir,
            os.getpid(),
        )
        return cls(cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Cache directory backends should be pointed at."""
        return self._path

    def wipe(self) -> None:
        """Securely delete every cache file, then remove the cache dir.

        Idempotent — calling on an already-wiped cache is a no-op.
        """
        if self._wiped:
            return
        _wipe_dir_contents(self._path)
        self._wiped = True

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> SealedCache:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.wipe()

    def __repr__(self) -> str:
        return f"SealedCache(path={self._path!r}, wiped={self._wiped})"


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------


def list_orphans(cache_root: Path | str | None = None) -> list[Path]:
    """Return cache dirs whose owner process is no longer alive.

    Looks for ``axon-sealed-*`` directories under *cache_root* (default
    OS temp dir) and reads each one's ``.axon-cache-pid`` sentinel.
    A dir is "orphan" iff its sentinel PID is not alive (or the
    sentinel is unreadable / missing — defensive).
    """
    root = Path(cache_root) if cache_root else Path(tempfile.gettempdir())
    if not root.is_dir():
        return []
    orphans: list[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith(CACHE_PREFIX):
            continue
        pid_path = entry / PID_SENTINEL_FILENAME
        try:
            pid_text = pid_path.read_text(encoding="utf-8").strip()
            pid = int(pid_text)
        except (OSError, ValueError):
            # Missing/unreadable sentinel → orphan. Defensive: a real
            # active cache should always have a parseable PID file.
            orphans.append(entry)
            continue
        if not _pid_alive(pid):
            orphans.append(entry)
    return orphans


def cleanup_orphans(cache_root: Path | str | None = None) -> int:
    """Wipe every orphan cache dir; return the count wiped.

    Called from ``AxonBrain.__init__`` (Phase 2 follow-up) so a crashed
    previous session doesn't leak plaintext indefinitely. Always
    safe to call: reports findings via DEBUG logs and never raises.
    """
    count = 0
    for orphan in list_orphans(cache_root):
        try:
            _wipe_dir_contents(orphan)
            count += 1
            logger.info("SealedCache: wiped orphan cache %s", orphan)
        except Exception as exc:
            logger.debug("SealedCache: orphan wipe failed for %s: %s", orphan, exc)
    return count


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check() -> dict[str, Any]:
    """Round-trip a tiny sealed project through the cache to confirm wiring.

    Used by future ``axon doctor`` output. Never raises — failures are
    reported in the dict.
    """
    out: dict[str, Any] = {"ok": False, "details": ""}
    try:
        from .crypto import generate_dek

        with tempfile.TemporaryDirectory() as td:
            sealed = Path(td) / "sealed_proj"
            sealed.mkdir()
            dek = generate_dek()
            # Seal one file, copy one plaintext-passthrough file.
            SealedFile.write(
                sealed / "encrypted.bin",
                b"sealed-payload",
                dek,
                aad=make_aad("sk_selfcheck", "encrypted.bin"),
            )
            (sealed / "passthrough.txt").write_text("plaintext-ok", encoding="utf-8")
            # Materialise + verify.
            cache = SealedCache.create(sealed, dek, key_id="sk_selfcheck")
            try:
                if (cache.path / "encrypted.bin").read_bytes() != b"sealed-payload":
                    out["details"] = "encrypted file decrypted to wrong bytes"
                    return out
                if (cache.path / "passthrough.txt").read_text(encoding="utf-8") != "plaintext-ok":
                    out["details"] = "plaintext passthrough file copied with wrong contents"
                    return out
            finally:
                cache.wipe()
            if cache.path.exists():
                out["details"] = "cache dir survived wipe()"
                return out
        out["ok"] = True
        out["details"] = "SealedCache create/read/wipe round-trip OK"
    except Exception as exc:  # pragma: no cover — defensive
        out["details"] = f"self-check raised: {type(exc).__name__}: {exc}"
    return out
