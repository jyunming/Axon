"""Sealed-project mount glue (Phase 2 of #SEALED).

Translates "open this sealed project for reading" into a concrete
plaintext directory the existing TurboQuantDB / LanceDB / BM25
backends can mmap. The actual mechanics live elsewhere:

- :mod:`axon.security.master` — supplies the per-project DEK, gated
  on the store being unlocked.
- :mod:`axon.security.seal` — supplies the on-disk sealed marker
  (``.security/.sealed``) which carries the ``seal_id`` used as the
  GCM AAD's key_id position.
- :mod:`axon.security.cache` — does the actual decrypt-into-tempdir
  + secure wipe.

Why the indirection: ``AxonBrain.switch_project`` shouldn't need to
know about scrypt, AES-GCM, or PID sentinels. It just asks
:func:`materialize_for_read` for a path, points its backends at it,
and calls :func:`release_cache` on close. Everything else stays
inside ``axon.security``.

This module is imported lazily by ``main.py`` so installs without
the ``[sealed]`` extra continue to import cleanly.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import SecurityError
from .cache import SealedCache
from .master import get_project_dek
from .seal import is_project_sealed, read_sealed_marker

logger = logging.getLogger("Axon")

__all__ = [
    "materialize_for_read",
    "release_cache",
]


def materialize_for_read(
    project_dir: Path | str,
    user_dir: Path | str,
    *,
    cache_root: Path | str | None = None,
    dek: bytes | None = None,
) -> SealedCache:
    """Decrypt a sealed project into an ephemeral cache + return the handle.
    The returned :class:`SealedCache` exposes ``cache.path`` — point
    every backend constructor at that path instead of *project_dir*.
    Call :func:`release_cache` (or use the cache as a context manager)
    when the brain closes the project to wipe the plaintext.
    Args:
        project_dir: The on-disk project directory to materialise (the
            owner's path; for grantees this is the path under their
            mount descriptor's ``target_project_dir``).
        user_dir: AxonStore user directory — needed to locate the
            keyring service and cached master when *dek* is None
            (owner path). For grantees with a pre-supplied DEK this
            arg is unused but kept for signature stability.
        cache_root: Override the OS temp dir for the cache (mostly for
            tests; production should leave it as None).
        dek: Pre-fetched 32-byte project DEK. When supplied, skips
            the master-key + ``get_project_dek`` lookup. Used by the
            grantee path to pass in a DEK fetched from the OS keyring
            via :func:`axon.security.share.get_grantee_dek`.
    Returns:
        A live :class:`SealedCache` whose ``path`` contains plaintext
        copies of every sealed content file plus the project's
        passthrough plaintext files (``version.json``, ``store_meta.json``,
        ``.security/`` contents).
    Raises:
        SecurityError: project is not sealed, or the store is locked,
            or the sealed marker is missing/malformed, or any underlying
            decryption fails (caller sees one error type rather than
            three different exception classes from three modules).
    """
    project_dir = Path(project_dir)
    user_dir = Path(user_dir)
    if not is_project_sealed(project_dir):
        raise SecurityError(
            f"materialize_for_read called on a non-sealed project at "
            f"{project_dir}. Call this only after is_project_sealed "
            "returns True."
        )
    marker = read_sealed_marker(project_dir)
    if marker is None:
        # Defensive — is_project_sealed was True so the marker file
        # exists, but it disappeared in the gap (race with a wipe?).
        raise SecurityError(f"Sealed marker disappeared between probe and read at {project_dir}.")
    seal_id = marker.get("seal_id", "")
    if not isinstance(seal_id, str) or not seal_id:
        raise SecurityError(
            f"Sealed marker at {project_dir} has no seal_id; refusing to mount. "
            "The project may have been sealed by an older incompatible version."
        )
    if dek is not None:
        # Grantee path: caller supplied the DEK from the OS keyring.
        # Skip the read-only get_project_dek lookup — grantees never
        # have access to dek.wrapped (the master-wrapped form).
        from .crypto import DEK_LEN as _DEK_LEN

        if len(dek) != _DEK_LEN:
            raise SecurityError(f"Supplied DEK is {len(dek)} bytes (expected {_DEK_LEN}).")
    else:
        # Owner path: read-only DEK lookup. Refuses to mint a fresh
        # DEK if dek.wrapped is missing — silently creating a new DEK
        # on the read path would make the existing ciphertext
        # permanently undecryptable. Raises SecurityError on locked
        # store OR on missing DEK file.
        dek = get_project_dek(user_dir, project_dir)
    try:
        cache = SealedCache.create(
            project_dir,
            dek,
            key_id=seal_id,
            cache_root=cache_root,
        )
    except Exception as exc:
        # Unify the exception surface for callers — they already
        # handle SecurityError; everything else (InvalidTag,
        # CacheCapacityError, OSError) gets wrapped.
        raise SecurityError(
            f"Failed to materialise sealed project {project_dir}: " f"{type(exc).__name__}: {exc}"
        ) from exc
    logger.info(
        "materialize_for_read: %s mounted at %s (seal_id=%s)",
        project_dir,
        cache.path,
        seal_id,
    )
    return cache


def release_cache(cache: SealedCache | None) -> None:
    """Securely wipe *cache* if it exists; safe on ``None``.
    Convenience wrapper so ``AxonBrain.close`` and similar callers can
    do ``release_cache(self._sealed_cache)`` without a None-guard.
    """
    if cache is None:
        return
    cache.wipe()
