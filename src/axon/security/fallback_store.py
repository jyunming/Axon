"""Passphrase-fallback store for headless / no-keyring environments (Phase 6).

When the OS keyring is unavailable (a Linux server without DBus /
``gnome-keyring`` / ``secret-tool``, a stripped Docker container, a CI
runner), the sealed-store master record cannot live in DPAPI / Keychain /
Secret Service. This module persists it as a plaintext-but-passphrase-
wrapped JSON file at ``<user_dir>/.security/master.enc`` instead.

Why "plaintext-but-passphrase-wrapped" works without keyring:

- The keyring entry already stores the master in an
  **AES-KW-wrapped-under-a-passphrase-derived-KEK** form (``{v, salt,
  wrapped}``), not as a raw key. The file fallback stores the **exact
  same JSON shape** — no plaintext key ever touches disk, just the
  same wrapped form the keyring would store. The user's passphrase
  is still required at every unlock to derive the KEK and unwrap.
- Treat the keyring as a convenience, not a security boundary: it
  shields the wrap from another user account on the same machine,
  but it does not add cryptographic protection beyond what AES-KW
  already provides. (DPAPI's user-tied protection IS additional
  defence in depth on Windows; we keep that path as the default and
  only fall back when DPAPI / Keychain / Secret Service is missing.)

Format on disk: same JSON as the keyring secret (UTF-8, sorted keys),
plus a ``"v"`` outer schema versioning bump if/when the record format
ever changes. File is locked down to ``0600`` on POSIX (no-op on
Windows where ACL inheritance protects).

Phase 6 deliverable. master.py routes through this module when the
keyring is unavailable; share.py (grantee DEK storage) is **not**
file-fallback in v1 (documented gap — grantees on headless boxes
can't redeem until the keyring works or the gap is addressed).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("Axon")

__all__ = [
    "FALLBACK_MASTER_FILENAME",
    "fallback_master_path",
    "read_master_record",
    "write_master_record",
    "delete_master_record",
    "is_present",
]

FALLBACK_MASTER_FILENAME: str = "master.enc"


def fallback_master_path(user_dir: Path | str) -> Path:
    """``<user_dir>/.security/master.enc`` — the file-fallback location."""
    return Path(user_dir) / ".security" / FALLBACK_MASTER_FILENAME


def is_present(user_dir: Path | str) -> bool:
    """Cheap probe: True iff the fallback file exists on disk."""
    return fallback_master_path(user_dir).is_file()


def read_master_record(user_dir: Path | str) -> str | None:
    """Return the raw JSON string from the fallback file, or None.

    Mirrors :func:`axon.security.keyring.get_secret` semantics — the
    caller (``master.py``) parses the JSON the same way it parses
    the keyring blob.
    """
    path = fallback_master_path(user_dir)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        # Surface as None so the caller's "not bootstrapped" path
        # fires; an explicit raise here would crash any operation
        # that probes ``is_bootstrapped`` defensively.
        logger.warning("Could not read fallback master file %s: %s", path, exc)
        return None


def write_master_record(user_dir: Path | str, payload: str) -> None:
    """Persist *payload* atomically to the fallback file.

    Validates that *payload* parses as JSON before writing — catches
    obvious caller bugs without involving the master keyring layer.
    """
    if not isinstance(payload, str) or not payload:
        raise ValueError("payload must be a non-empty string (JSON)")
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload must be valid JSON: {exc}") from exc

    path = fallback_master_path(user_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".write")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    # Lock down — only this user needs to read it. No-op on Windows
    # where ACL inheritance protects.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def delete_master_record(user_dir: Path | str) -> bool:
    """Remove the fallback file. True if it existed; False if absent."""
    path = fallback_master_path(user_dir)
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("Could not delete fallback master file %s: %s", path, exc)
        return False
