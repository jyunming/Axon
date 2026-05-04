"""Keyring helpers — owner master key + grantee per-share DEK storage.

Phase 1 ships a thin wrapper around the cross-platform :mod:`keyring`
package (DPAPI on Windows, Keychain on macOS, Secret Service on Linux).
The wrapper:

- Detects when no usable backend exists and raises
  :class:`KeyringUnavailableError` with a clear hint, instead of
  silently falling back to an unencrypted file.
- Centralises the service-name convention so callers don't sprinkle
  string literals across the codebase: ``axon.master.<owner>`` for the
  owner's master key, ``axon.share.<key_id>`` for grantee-side wrapped
  DEKs.

The **passphrase fallback** for headless / no-keyring environments is
out of scope for Phase 1 (per ``docs/architecture/SEALED_SHARING_DESIGN.md`` §6 phase
list — keyring + passphrase fallback together are a Phase 2
deliverable). For now, callers on a no-keyring machine will see a
clear ``KeyringUnavailableError`` and can choose between installing a
backend or waiting for Phase 2.

Dependency: ``keyring`` (PyPI) — installed via the ``sealed`` extra
(``pip install axon-rag[sealed]``).
"""
from __future__ import annotations

import logging
import threading
from typing import Any

try:
    import keyring as _keyring
    from keyring.errors import KeyringError as _BackendKeyringError
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.keyring requires the 'keyring' package. "
        "Install with: pip install axon-rag[sealed]"
    ) from exc

logger = logging.getLogger("Axon")

__all__ = [
    "KeyringUnavailableError",
    "MASTER_SERVICE_PREFIX",
    "SHARE_SERVICE_PREFIX",
    "master_service",
    "share_service",
    "is_available",
    "store_secret",
    "get_secret",
    "delete_secret",
    "SessionDEKCache",
    "set_keyring_mode",
    "get_keyring_mode",
    "session_cache",
]


class KeyringUnavailableError(RuntimeError):
    """Raised when the OS keyring backend is missing or unusable.
    Common causes:
      - Linux server with no D-Bus / Secret Service running (gnome-keyring).
      - WSL distro that hasn't been configured for Linux Secret Service.
      - A ``keyring`` install that defaulted to the no-op ``keyring.backends.fail``
        backend.
    The Phase 2 work in ``docs/architecture/SEALED_SHARING_DESIGN.md`` will add a
    passphrase-protected file fallback for these cases. Until then,
    install one of:
      - Linux: ``apt install gnome-keyring`` (requires a D-Bus session)
              or ``pip install keyrings.cryptfile`` for a file-based
              alternative.
      - macOS / Windows: built-in Keychain / DPAPI is always available.
    """


MASTER_SERVICE_PREFIX: str = "axon.master."
SHARE_SERVICE_PREFIX: str = "axon.share."


def master_service(owner: str) -> str:
    """Service name under which the owner's master key is stored."""
    if not owner:
        raise ValueError("owner must be non-empty")
    return f"{MASTER_SERVICE_PREFIX}{owner}"


def share_service(key_id: str) -> str:
    """Service name under which a grantee stores an unwrapped DEK."""
    if not key_id:
        raise ValueError("key_id must be non-empty")
    return f"{SHARE_SERVICE_PREFIX}{key_id}"


def _active_backend() -> Any:
    """Return the active keyring backend; raise KeyringUnavailableError when unusable.
    Detects the no-op ``keyring.backends.fail.Keyring`` stub by module
    name (the class name itself is just ``Keyring`` — same as the base
    class — so we cannot disambiguate on class name alone).
    """
    backend = _keyring.get_keyring()
    module = type(backend).__module__ or ""
    if module.endswith(".fail"):
        raise KeyringUnavailableError(
            f"Active keyring backend ({module}.{type(backend).__name__}) is a "
            "no-op fail-stub. Install a real backend (gnome-keyring on Linux, "
            "Keychain on macOS, DPAPI on Windows) or wait for Phase 2 "
            "passphrase fallback."
        )
    return backend


def is_available() -> bool:
    """Return True if the OS keyring is usable.
    Performs a lightweight write/read/delete round-trip on a sentinel
    service name. Catches any exception and returns False rather than
    propagating, so callers can branch on availability without a
    try/except.
    """
    sentinel_service = "axon.__availability_check__"
    sentinel_user = "probe"
    wrote_sentinel = False
    try:
        backend = _active_backend()
        backend.set_password(sentinel_service, sentinel_user, "ok")
        wrote_sentinel = True
        recovered = backend.get_password(sentinel_service, sentinel_user)
        return bool(recovered == "ok")
    except Exception as exc:
        logger.debug("keyring is_available() check failed: %s", exc)
        return False
    finally:
        # Best-effort cleanup of the sentinel even when the get/return
        # path above raised — otherwise a failed probe would leak the
        # sentinel into the user's keyring.
        if wrote_sentinel:
            try:
                backend.delete_password(sentinel_service, sentinel_user)
            except Exception as cleanup_exc:
                logger.debug("keyring is_available() sentinel cleanup failed: %s", cleanup_exc)


def store_secret(service: str, username: str, secret: str) -> None:
    """Store *secret* under (*service*, *username*).

    Routes through the active keyring mode:

    - ``persistent`` (default): writes to the OS keyring as before.
    - ``session``: writes to a process-local in-memory dict; OS keyring
      is never touched. Lost when the process exits.
    - ``never``: silently no-ops. Callers must re-derive the secret on
      every read; ``get_secret`` will return ``None``.

    Raises:
        KeyringUnavailableError: backend unavailable in ``persistent`` mode.
        RuntimeError: backend present but operation failed (rare).
    """
    mode = get_keyring_mode()
    if mode == "session":
        _SESSION_CACHE.set(service, username, secret)
        return
    if mode == "never":
        # Intentionally drop on the floor. Callers receive None on get.
        logger.debug(
            "keyring_mode=never: store_secret no-op for service=%s user=%s", service, username
        )
        return
    backend = _active_backend()
    try:
        backend.set_password(service, username, secret)
    except _BackendKeyringError as exc:
        raise RuntimeError(f"keyring set_password failed: {exc}") from exc


def get_secret(service: str, username: str) -> str | None:
    """Return the secret stored at (*service*, *username*) or ``None``.

    Mode dispatch matches :func:`store_secret`:

    - ``persistent``: read from OS keyring.
    - ``session``: read from in-memory dict only.
    - ``never``: always returns ``None`` — the secret was never stored.

    Returns ``None`` for "not found"; raises only on backend
    unavailability or unexpected backend errors (``persistent`` only).
    """
    mode = get_keyring_mode()
    if mode == "session":
        return _SESSION_CACHE.get(service, username)
    if mode == "never":
        return None
    backend = _active_backend()
    try:
        secret = backend.get_password(service, username)
    except _BackendKeyringError as exc:
        raise RuntimeError(f"keyring get_password failed: {exc}") from exc
    # Backends return Any; narrow to str | None for mypy and runtime sanity.
    if secret is None:
        return None
    return str(secret)


def delete_secret(service: str, username: str) -> None:
    """Delete the secret at (*service*, *username*).

    Silently no-ops on "not found" — the desired post-condition (secret
    absent) is satisfied. Mode dispatch:

    - ``persistent``: delete from OS keyring.
    - ``session``: drop from in-memory dict.
    - ``never``: nothing to delete; pure no-op.
    """
    mode = get_keyring_mode()
    if mode == "session":
        _SESSION_CACHE.delete(service, username)
        return
    if mode == "never":
        return
    from keyring.errors import PasswordDeleteError

    backend = _active_backend()
    try:
        backend.delete_password(service, username)
    except PasswordDeleteError:
        # Canonical "secret not found" signal — desired post-condition
        # ("secret is absent") is already satisfied.
        pass
    except _BackendKeyringError as exc:
        raise RuntimeError(f"keyring delete_password failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Mode dispatch (v0.4.0 Item 2)
# ---------------------------------------------------------------------------

_KeyringMode = str  # Literal["persistent", "session", "never"] — kept loose for runtime
_VALID_MODES = ("persistent", "session", "never")


class SessionDEKCache:
    """Thread-safe in-process keyring substitute for ``keyring_mode='session'``.

    Stores ``(service, username) -> secret`` in a plain dict guarded by an
    :class:`threading.Lock`. No persistence, no encryption — the lifetime
    is bounded by the Python process. Death of the process wipes every
    DEK.

    Trade-offs vs the OS keyring:

    - **Pro**: zero filesystem footprint, no OS API surface; useful in
      Docker / CI / headless servers where ``persistent`` would fail
      with ``KeyringUnavailableError``.
    - **Con**: process restart loses every DEK — grantees must
      re-redeem after Axon restart. Memory dumps could expose the DEK
      while running. Acceptable for short-lived server processes.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}
        self._lock: threading.Lock = threading.Lock()

    def set(self, service: str, username: str, secret: str) -> None:
        with self._lock:
            self._store[(service, username)] = secret

    def get(self, service: str, username: str) -> str | None:
        with self._lock:
            return self._store.get((service, username))

    def delete(self, service: str, username: str) -> None:
        with self._lock:
            self._store.pop((service, username), None)

    def clear(self) -> None:
        """Wipe the entire cache — call on shutdown / process exit if
        defence-in-depth wiping is needed (``del`` already drops the
        bytes when the dict goes out of scope)."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


_SESSION_CACHE = SessionDEKCache()
_current_mode: _KeyringMode = "persistent"
_mode_lock = threading.Lock()


def set_keyring_mode(mode: str) -> None:
    """Set the active keyring mode for this process.

    Called by ``AxonBrain`` at startup with the value from
    ``AxonConfig.keyring_mode``. Switching modes mid-flight is allowed
    but does NOT migrate already-stored secrets — a switch from
    ``persistent`` to ``session`` will leave OS keyring entries untouched
    and start writing to the in-process cache instead.

    :raises ValueError: If *mode* is not one of ``persistent | session | never``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"keyring_mode must be one of {_VALID_MODES}, got {mode!r}")
    global _current_mode
    with _mode_lock:
        _current_mode = mode
    logger.debug("keyring_mode set to %s", mode)


def get_keyring_mode() -> str:
    """Return the active keyring mode."""
    with _mode_lock:
        return _current_mode


def session_cache() -> SessionDEKCache:
    """Return the process-singleton :class:`SessionDEKCache` instance.

    Exposed mainly so tests can inspect / clear the cache between cases.
    Production code should go through :func:`store_secret` / :func:`get_secret`.
    """
    return _SESSION_CACHE


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _self_check() -> dict[str, Any]:
    """Return a diagnostic dict — used by future ``axon doctor`` output."""
    backend_name = type(_keyring.get_keyring()).__name__
    return {
        "available": is_available(),
        "backend": backend_name,
    }
