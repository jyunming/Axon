"""Owner master key + per-project DEK lifecycle (Phase 2 of #SEALED).

Two key tiers (envelope encryption, the standard KMS pattern):

- **Master key** — random 32 bytes per owner. NEVER stored in
  plaintext on disk and NEVER in the user's keyring directly. The
  keyring stores the master *encrypted under a passphrase-derived
  KEK*: ``salt + wrap(master, scrypt(passphrase, salt))``. Unlock =
  re-derive the KEK from the user's passphrase and unwrap.
- **Project DEK** — random 32 bytes per project. Wrapped with the
  master and persisted at ``<project>/.security/dek.wrapped`` (40
  bytes). At unlock time, the master is in process memory so the DEK
  can be re-derived on demand.

Why this layering:

- Passphrase rotation re-derives a new KEK and re-wraps the master,
  but does NOT touch any project's wrapped DEK — the master is
  unchanged across passphrase changes.
- Loss of the master would force re-encrypt of every project. So we
  never persist the master in plaintext anywhere; recovery is via
  passphrase, full stop.
- Recovery: lose the passphrase, lose access to all sealed projects.
  Documented; no escrow in v1.

Wire-level keyring secret (JSON): ``{"v": 1, "salt": "<b64>", "wrapped":
"<b64>"}`` stored at ``axon.master.<owner>`` / ``master``. Bumping
``v`` is the breaking-change signal for future format migrations.

Phase 2 deliverable. Replaces the same-named stubs in
``axon/security/__init__.py`` (which currently raise
``SecurityError("not configured")``).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.keywrap import (
        InvalidUnwrap,
        aes_key_unwrap,
        aes_key_wrap,
    )
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.master requires the 'cryptography' package. "
        "Install with: pip install axon-rag[sealed]"
    ) from exc

from . import SecurityError
from . import keyring as _kr
from .crypto import DEK_LEN, generate_dek, wrap_key

logger = logging.getLogger("Axon")

__all__ = [
    "BadPassphraseError",
    "MASTER_USERNAME",
    "PROJECT_DEK_FILENAME",
    "SCHEMA_VERSION",
    "bootstrap_store",
    "unlock_store",
    "lock_store",
    "is_unlocked",
    "change_passphrase",
    "get_master_key",
    "get_or_create_project_dek",
    "get_project_dek",
    "is_bootstrapped",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER_USERNAME: str = "master"
PROJECT_DEK_FILENAME: str = ".security/dek.wrapped"
SCHEMA_VERSION: int = 1

# scrypt parameters (RFC 7914). N=2^15 keeps unlock under ~150 ms on
# modern hardware while making brute-force expensive.
_SCRYPT_N: int = 2**15
_SCRYPT_R: int = 8
_SCRYPT_P: int = 1
_SCRYPT_LEN: int = 32  # 256-bit KEK
_SALT_LEN: int = 32  # 256 bits of salt entropy


class BadPassphraseError(SecurityError):
    """Raised when ``unlock_store`` or ``change_passphrase`` is given a
    passphrase that does not unwrap the stored master key.

    Distinct from generic ``SecurityError`` so callers (REPL, REST,
    MCP) can show a focused "wrong passphrase" message without
    swallowing actual configuration / I/O errors.
    """


# ---------------------------------------------------------------------------
# Process-local unlocked-master cache
# ---------------------------------------------------------------------------

_unlock_lock = threading.Lock()
# Per-owner cache so a multi-tenant test or future multi-user process can
# unlock several stores independently. Keyed by the keyring "service" string.
_unlocked_masters: dict[str, bytes] = {}


def _owner_from_user_dir(user_dir: Path) -> str:
    """The "owner" identity in the keyring service-name.

    Uses the user_dir basename — matches the AxonStore convention
    (``~/.axon/AxonStore/<owner>/...``). Tests can pass any user_dir.
    """
    name = Path(user_dir).name
    if not name:
        raise ValueError(f"user_dir has no basename: {user_dir!r}")
    return name


def _service(user_dir: Path) -> str:
    return _kr.master_service(_owner_from_user_dir(user_dir))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _derive_kek(passphrase: str, salt: bytes) -> bytes:
    """scrypt(passphrase, salt) → 32-byte KEK."""
    if not passphrase:
        raise ValueError("passphrase must be a non-empty string")
    if len(salt) != _SALT_LEN:
        raise ValueError(f"salt must be {_SALT_LEN} bytes, got {len(salt)}")
    return Scrypt(
        salt=salt,
        length=_SCRYPT_LEN,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    ).derive(passphrase.encode("utf-8"))


def _read_keyring_record(user_dir: Path) -> dict[str, Any] | None:
    """Read the JSON master record from the keyring or fallback file.

    Resolution order:
    1. OS keyring (DPAPI / Keychain / Secret Service).
    2. ``<user_dir>/.security/master.enc`` fallback (Phase 6) — used
       when the keyring is unavailable (headless Linux servers,
       stripped containers).

    Raises ``SecurityError`` if the record exists but is malformed —
    a defence-in-depth signal that the storage layer may have been
    tampered with.
    """
    from . import fallback_store as _fb

    raw: str | None = None
    source: str = ""
    try:
        raw = _kr.get_secret(_service(user_dir), MASTER_USERNAME)
        if raw is not None:
            source = f"keyring service {_service(user_dir)!r}"
    except _kr.KeyringUnavailableError:
        raw = None  # fall through to file fallback

    if raw is None:
        # File fallback. Read only when the keyring is unavailable OR
        # when the keyring returned no record (the keyring is preferred
        # whenever it has the answer — so the typical Windows/macOS
        # box never reads the file). Re-raise OSError as SecurityError
        # so the caller doesn't mistake "file unreadable" for "not
        # bootstrapped" and offer to re-bootstrap (which would overwrite).
        try:
            raw = _fb.read_master_record(user_dir)
        except OSError as exc:
            raise SecurityError(str(exc)) from exc
        if raw is not None:
            source = f"fallback file {_fb.fallback_master_path(user_dir)}"

    if raw is None:
        return None
    try:
        record: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SecurityError(
            f"Master record at {source} is unparseable JSON ({exc}). "
            "Manual recovery may be needed."
        ) from exc
    if not isinstance(record, dict) or record.get("v") != SCHEMA_VERSION:
        raise SecurityError(
            f"Master record at {source} has schema_version mismatch "
            f"(expected {SCHEMA_VERSION}, got "
            f"{record.get('v') if isinstance(record, dict) else type(record).__name__})."
        )
    return record


def _write_keyring_record(user_dir: Path, *, salt: bytes, wrapped_master: bytes) -> None:
    """Persist the master record to the keyring or fallback file.

    Routes to the file fallback when the OS keyring is unavailable.
    The wrapped master is identical in both stores — only the location
    differs.
    """
    from . import fallback_store as _fb

    payload = json.dumps(
        {
            "v": SCHEMA_VERSION,
            "salt": _b64(salt),
            "wrapped": _b64(wrapped_master),
        },
        separators=(",", ":"),
    )
    try:
        _kr.store_secret(_service(user_dir), MASTER_USERNAME, payload)
    except _kr.KeyringUnavailableError:
        _fb.write_master_record(user_dir, payload)
        logger.info(
            "Sealed-store master persisted to file fallback at %s " "(OS keyring unavailable)",
            _fb.fallback_master_path(user_dir),
        )


# ---------------------------------------------------------------------------
# Bootstrap / unlock / lock / change_passphrase
# ---------------------------------------------------------------------------


def is_bootstrapped(user_dir: Path) -> bool:
    """Return True if a master record already exists in the keyring."""
    try:
        return _read_keyring_record(user_dir) is not None
    except SecurityError:
        # Malformed record → still "bootstrapped" but in a bad state;
        # caller will get the same SecurityError on unlock.
        return True


def bootstrap_store(user_dir: Path, passphrase: str) -> dict[str, Any]:
    """Initial setup — generate the master + persist its passphrase-wrapped form.

    Raises:
        SecurityError: the store is already bootstrapped (use
            :func:`change_passphrase` to rotate, or wipe the keyring
            entry first if you really mean to start over).
        BadPassphraseError: the supplied passphrase is empty.

    Returns a status dict with ``{"initialized": True, "owner": ...}``.
    """
    if not passphrase:
        raise BadPassphraseError("passphrase must be a non-empty string")
    if is_bootstrapped(user_dir):
        raise SecurityError(
            f"Store at {_service(user_dir)} is already bootstrapped. "
            "Use change_passphrase to rotate, or delete the keyring entry "
            "manually if you really mean to discard the master key."
        )

    master = generate_dek()  # 32 random bytes — never persisted in plaintext
    salt = os.urandom(_SALT_LEN)
    kek = _derive_kek(passphrase, salt)
    wrapped = aes_key_wrap(kek, master)
    _write_keyring_record(user_dir, salt=salt, wrapped_master=wrapped)

    # Cache the master so the bootstrapping process is immediately
    # unlocked — no need for the caller to re-supply the passphrase.
    with _unlock_lock:
        _unlocked_masters[_service(user_dir)] = master

    logger.info(
        "Sealed-store bootstrapped for owner=%s (keyring service=%s)",
        _owner_from_user_dir(user_dir),
        _service(user_dir),
    )
    return {
        "initialized": True,
        "owner": _owner_from_user_dir(user_dir),
        "service": _service(user_dir),
    }


def unlock_store(user_dir: Path, passphrase: str) -> dict[str, Any]:
    """Verify the passphrase, decrypt the master, cache it in this process."""
    record = _read_keyring_record(user_dir)
    if record is None:
        raise SecurityError(
            f"Store at {_service(user_dir)} is not bootstrapped. "
            "Run bootstrap_store(user_dir, passphrase) first."
        )
    salt = _unb64(record["salt"])
    wrapped = _unb64(record["wrapped"])
    kek = _derive_kek(passphrase, salt)
    try:
        master = aes_key_unwrap(kek, wrapped)
    except InvalidUnwrap as exc:
        raise BadPassphraseError(f"Wrong passphrase for store {_service(user_dir)}.") from exc

    with _unlock_lock:
        _unlocked_masters[_service(user_dir)] = master

    return {
        "unlocked": True,
        "owner": _owner_from_user_dir(user_dir),
        "service": _service(user_dir),
    }


def lock_store(user_dir: Path) -> dict[str, Any]:
    """Clear the in-memory cached master for this owner.

    Subsequent DEK lookups will raise ``SecurityError("locked")`` until
    :func:`unlock_store` is called again.
    """
    with _unlock_lock:
        existed = _unlocked_masters.pop(_service(user_dir), None) is not None
    return {
        "locked": True,
        "was_unlocked": existed,
        "owner": _owner_from_user_dir(user_dir),
    }


def is_unlocked(user_dir: Path) -> bool:
    """Return True if the master is cached in this process for *user_dir*."""
    with _unlock_lock:
        return _service(user_dir) in _unlocked_masters


def change_passphrase(user_dir: Path, old_passphrase: str, new_passphrase: str) -> dict[str, Any]:
    """Re-wrap the master under a fresh KEK derived from *new_passphrase*.

    Project DEKs are NOT touched — they're wrapped under the master,
    not the passphrase, so passphrase rotation costs O(1) regardless
    of how many sealed projects the owner has.
    """
    if not new_passphrase:
        raise BadPassphraseError("new_passphrase must be a non-empty string")
    record = _read_keyring_record(user_dir)
    if record is None:
        raise SecurityError(f"Store at {_service(user_dir)} is not bootstrapped.")
    salt = _unb64(record["salt"])
    wrapped = _unb64(record["wrapped"])
    kek_old = _derive_kek(old_passphrase, salt)
    try:
        master = aes_key_unwrap(kek_old, wrapped)
    except InvalidUnwrap as exc:
        raise BadPassphraseError(
            f"Old passphrase did not unlock store {_service(user_dir)}."
        ) from exc

    new_salt = os.urandom(_SALT_LEN)
    kek_new = _derive_kek(new_passphrase, new_salt)
    new_wrapped = aes_key_wrap(kek_new, master)
    _write_keyring_record(user_dir, salt=new_salt, wrapped_master=new_wrapped)

    # Refresh in-memory cache (master is unchanged but the cache may
    # have been cleared between unlock and change).
    with _unlock_lock:
        _unlocked_masters[_service(user_dir)] = master

    return {
        "rotated": True,
        "owner": _owner_from_user_dir(user_dir),
    }


def get_master_key(user_dir: Path) -> bytes:
    """Return the cached unlocked master key.

    Raises:
        SecurityError: the store is locked; call :func:`unlock_store`
            first.
    """
    with _unlock_lock:
        master = _unlocked_masters.get(_service(user_dir))
    if master is None:
        raise SecurityError(f"Store {_service(user_dir)} is locked. Call unlock_store first.")
    return master


# ---------------------------------------------------------------------------
# Project DEK lifecycle
# ---------------------------------------------------------------------------


def _project_dek_path(project_dir: Path) -> Path:
    return Path(project_dir) / PROJECT_DEK_FILENAME


def get_or_create_project_dek(user_dir: Path, project_dir: Path) -> bytes:
    """Return the per-project DEK; create + persist on first call.

    Reads ``<project_dir>/.security/dek.wrapped`` if present; unwraps
    with the cached master and returns the 32-byte DEK. If absent,
    generates a fresh DEK, wraps it under the master, writes the wrap
    to disk atomically, and returns the new DEK.

    Raises:
        SecurityError: store is locked, or wrapped DEK exists but won't
            unwrap (likely keyring/passphrase mismatch — the project
            was sealed under a different master).
    """
    master = get_master_key(user_dir)
    wrap_path = _project_dek_path(project_dir)

    if wrap_path.is_file():
        try:
            dek: bytes = aes_key_unwrap(master, wrap_path.read_bytes())
        except InvalidUnwrap as exc:
            raise SecurityError(
                f"Wrapped DEK at {wrap_path} won't unwrap with the current "
                "master. The project was likely sealed under a different "
                "master / owner; recovery requires the original passphrase."
            ) from exc
        if len(dek) != DEK_LEN:
            raise SecurityError(
                f"Wrapped DEK at {wrap_path} unwrapped to {len(dek)} bytes "
                f"(expected {DEK_LEN}); manifest may be corrupted."
            )
        return dek

    # No DEK yet — mint one, persist, return.
    dek = generate_dek()
    wrapped = wrap_key(dek, master)
    wrap_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wrap_path.with_suffix(wrap_path.suffix + ".sealing")
    tmp.write_bytes(wrapped)
    os.replace(tmp, wrap_path)
    # Lock down the wrap file — only the owner needs to read it.
    try:
        os.chmod(wrap_path, 0o600)
    except OSError:
        # Windows: chmod is largely a no-op; ACL inheritance protects.
        pass
    logger.info(
        "Generated new project DEK at %s (wrapped, %d bytes)",
        wrap_path,
        len(wrapped),
    )
    return dek


def get_project_dek(user_dir: Path, project_dir: Path) -> bytes:
    """Read the existing wrapped DEK; raise if it doesn't exist.

    Use this on the read path (mount / open) where creating a new DEK
    silently would be a bug — if the project file is gone the caller
    needs to know.
    """
    wrap_path = _project_dek_path(project_dir)
    if not wrap_path.is_file():
        raise SecurityError(
            f"Project DEK file missing at {wrap_path}. The project may not "
            "be sealed, or the .security directory was deleted."
        )
    master = get_master_key(user_dir)
    try:
        dek: bytes = aes_key_unwrap(master, wrap_path.read_bytes())
    except InvalidUnwrap as exc:
        raise SecurityError(
            f"Wrapped DEK at {wrap_path} won't unwrap with the current " "master."
        ) from exc
    return dek


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check(user_dir: Path | None = None) -> dict[str, Any]:
    """Diagnostic round-trip used by future ``axon doctor`` output.

    Uses an isolated temp dir + a synthetic owner so the user's real
    keyring is not touched. Never raises.
    """
    import tempfile

    out: dict[str, Any] = {"ok": False, "details": ""}
    try:
        with tempfile.TemporaryDirectory() as td:
            ud = Path(td) / "selfcheck-owner"
            ud.mkdir()
            # Use a unique service to avoid stomping a real user's keyring
            # entry — same code path but different keyring slot.
            unique_service = f"axon.master.__selfcheck_{os.getpid()}"

            # Stash the original master_service to restore later.
            original = _kr.master_service
            _kr.master_service = lambda owner: unique_service
            try:
                bootstrap_store(ud, "selfcheck-passphrase")
                if not is_unlocked(ud):
                    out["details"] = "is_unlocked False after bootstrap"
                    return out
                lock_store(ud)
                if is_unlocked(ud):
                    out["details"] = "is_unlocked True after lock"
                    return out
                unlock_store(ud, "selfcheck-passphrase")
                # Per-project DEK round-trip.
                proj = ud / "proj"
                proj.mkdir()
                dek1 = get_or_create_project_dek(ud, proj)
                dek2 = get_or_create_project_dek(ud, proj)  # idempotent
                if dek1 != dek2:
                    out["details"] = "DEK changed across two get_or_create calls"
                    return out
                # Bad-passphrase rejection.
                lock_store(ud)
                try:
                    unlock_store(ud, "wrong-passphrase")
                    out["details"] = "wrong passphrase did not raise"
                    return out
                except BadPassphraseError:
                    pass
            finally:
                _kr.master_service = original
                # Best-effort cleanup of the test keyring entry.
                try:
                    _kr.delete_secret(unique_service, MASTER_USERNAME)
                except Exception:
                    pass
        out["ok"] = True
        out["details"] = "bootstrap / unlock / lock / DEK round-trip OK"
    except Exception as exc:  # pragma: no cover — defensive
        out["details"] = f"self-check raised: {type(exc).__name__}: {exc}"
    return out
