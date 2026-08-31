"""Ephemeral plaintext cache for sealed Axon projects (Phase 2 of #SEALED).

When an owner queries a sealed project, the on-disk files are AES-GCM
ciphertext. Backends like TurboQuantDB / LanceDB / BM25 mmap their data
files for performance — and mmap can't see through encryption. The
v1 policy (decision §5.1 in ``docs/architecture/SEALED_SHARING_DESIGN.md``, locked
2026-04-25) is to **decrypt the whole project into an ephemeral
plaintext cache** in the OS temp dir at mount time, point the backend
at the cache path, and wipe the cache on close. Backends mmap the
cache normally, so query performance is identical to plaintext mode.

Cost: a session-bounded plaintext footprint on disk. Mitigations
implemented here:

- **Cache location**: ``tempfile.mkdtemp(prefix="axon-sealed-")``
  (Linux/macOS uses ``/tmp``; Windows uses ``%LOCALAPPDATA%\\Temp``).
  Per-mount; never shared between projects.
- **Wipe on close**: every cache file is overwritten with random bytes
  (``O_RDWR`` re-open + ``write(os.urandom(...))`` + fsync +
  ``FlushFileBuffers`` on Windows) before unlink. This won't survive
  low-level disk forensics on SSDs (TRIM / wear-leveling defeat
  overwrite) or NTFS copy-on-write sector reuse, but at the filesystem
  layer the plaintext is gone. On Windows, enable BitLocker on the
  drive containing ``%TEMP%`` (or redirect via ``AXON_CACHE_DIR``) for
  full protection of highly sensitive data.
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
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from axon._pid_check import pid_alive

try:
    from .crypto import MAGIC, SealedFile, make_aad
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.cache requires axon.security.crypto, which requires the "
        "'cryptography' package. Install with: pip install axon-rag[sealed]"
    ) from exc

logger = logging.getLogger("Axon")

# Tracks cache dirs for which the Windows NTFS plaintext-persistence warning
# has already been logged — so we emit it at most once per cache lifetime
# rather than once per file deleted. Protected by a lock to avoid double-emit
# under concurrent wipe calls.
_windows_warned_paths: set[str] = set()
_windows_warned_lock = threading.Lock()

__all__ = [
    "CACHE_PREFIX",
    "PID_SENTINEL_FILENAME",
    "CACHE_HEADROOM_FRACTION",
    "CacheCapacityError",
    "SealedCache",
    "SealedFileTamperError",
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


class SealedFileTamperError(RuntimeError):
    """Raised when a file in the seal policy's must-seal set lacks the
    AXSL header at mount time.

    AAD binding into the GCM tag prevents an attacker from forging
    new ciphertext, but it does NOT prevent an attacker with write
    access to the synced filesystem from REPLACING an encrypted file
    with a plaintext one. Without an explicit check at materialise
    time, the cache would copy that plaintext to a path the backend
    later reads as authoritative — bypassing authenticated encryption.

    This error surfaces the tamper to the caller (``materialize_for_read``)
    which wraps it as :class:`SecurityError`. The partial cache is
    wiped by the same ``except`` block that catches every other
    materialisation failure.
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
    """Overwrite *path* with random bytes, fsync, then unlink.

    Uses ``os.urandom`` for overwrite content instead of zeros — random
    bytes prevent zero-pattern detection and make statistical recovery
    harder. On Windows, calls ``FlushFileBuffers`` via ctypes after the
    Python-level ``fsync`` to maximise the chance that the random data
    actually reaches the storage device before the file handle is closed.

    **Windows NTFS limitation**: NTFS copy-on-write and SSD TRIM /
    wear-leveling may redirect writes to new sectors, leaving the old
    sector content accessible via low-level disk forensics even after
    this function completes. This implementation is best-effort at the
    filesystem layer — it ensures no readable plaintext survives at the
    file's *logical* address. For compliance or classified use, enable
    BitLocker on the volume containing the cache (see
    ``docs/SHARING.md#security-considerations``).

    Errors are silently swallowed (the unlink is the desired
    post-condition; if we can't even unlink, there's nothing useful to
    do here).
    """
    # Emit a one-time INFO warning on Windows about the NTFS limitation.
    if sys.platform == "win32":
        parent_key = str(path.parent)
        with _windows_warned_lock:
            if parent_key not in _windows_warned_paths:
                _windows_warned_paths.add(parent_key)
                logger.info(
                    "SealedCache: Windows NTFS secure-delete is best-effort — "
                    "NTFS copy-on-write and SSD TRIM/wear-leveling may retain "
                    "plaintext in freed sectors. Enable BitLocker on %s for "
                    "full protection.",
                    path.parent,
                )

    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > 0:
        try:
            with path.open("r+b") as fh:
                # Overwrite with random bytes in 64 KiB chunks to avoid
                # allocating a large buffer for large files.
                chunk_size = min(size, 64 * 1024)
                remaining = size
                while remaining > 0:
                    write_len = min(remaining, chunk_size)
                    fh.write(os.urandom(write_len))
                    remaining -= write_len
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
                # On Windows, call FlushFileBuffers for best-effort
                # write-through to the storage device.
                if sys.platform == "win32":
                    try:
                        import ctypes
                        import msvcrt

                        handle = msvcrt.get_osfhandle(fh.fileno())
                        ctypes.windll.kernel32.FlushFileBuffers(handle)
                    except Exception as _flush_exc:  # noqa: BLE001
                        logger.debug("FlushFileBuffers failed for %s: %s", path, _flush_exc)
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


# Re-exported for this module's one caller below; the actual check now
# lives in axon._pid_check so server_client.py's store lock can share it
# instead of hand-rolling a second copy.
_pid_alive = pid_alive


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
        # wipe() is idempotent in intent but the disk-walk + os.remove loop
        # was not concurrency-safe — two threads wiping at once could each
        # iterate the same files and trigger ``FileNotFoundError`` on the
        # loser. Serialize wipes per-cache.
        self._wipe_lock = threading.Lock()

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
        # Defer the import of the seal policy to avoid a cache → seal →
        # cache cycle at module load. The policy is a pure function over
        # the relative path so calling it inside the loop adds no I/O.
        from .seal import _should_seal  # noqa: PLC0415

        try:
            # PID sentinel — used by cleanup_orphans on next boot.
            (cache_dir / PID_SENTINEL_FILENAME).write_text(str(os.getpid()), encoding="utf-8")
            # Walk sealed_dir, decrypt sealed files, copy non-sealed.
            for src in sealed_dir.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(sealed_dir)
                # Skip the .security/ subtree — it holds key wraps and
                # rotation receipts that backends never read at query time
                # and that, if mirrored into the plaintext cache, would
                # leak wrap material into the OS temp dir on every mount.
                rel_parts = rel.parts
                if rel_parts and rel_parts[0] == ".security":
                    continue
                dst = cache_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if is_sealed_file(src):
                    aad = make_aad(key_id, str(rel).replace("\\", "/"))
                    plaintext = SealedFile.read(src, dek, aad=aad)
                    dst.write_bytes(plaintext)
                else:
                    # The seal policy decides which files MUST be encrypted;
                    # anything in the must-seal set that lacks the AXSL
                    # header here is either tampering or a corrupted seal.
                    # Refuse rather than silently materialise attacker-
                    # controlled plaintext into the cache where backends
                    # would consume it as authoritative — AAD binding
                    # prevents ciphertext forgery, but does NOT prevent
                    # an attacker with write access from REPLACING an
                    # encrypted file with plaintext.
                    if _should_seal(rel):
                        raise SealedFileTamperError(
                            f"Project file {rel} is in the must-seal set but "
                            "lacks an AXSL header. The sealed project may "
                            "have been tampered with on the synced filesystem "
                            "(an attacker may have replaced an encrypted file "
                            "with plaintext to bypass authenticated encryption). "
                            "Refusing to materialise."
                        )
                    # Plaintext passthrough file (deliberately unsealed —
                    # version.json, store_meta.json, etc.). Copy as-is so
                    # the cache is a faithful view of the project.
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
        Thread-safe — concurrent wipes serialise on ``_wipe_lock`` so the
        on-disk walk doesn't race itself into a ``FileNotFoundError``.
        """
        with self._wipe_lock:
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
