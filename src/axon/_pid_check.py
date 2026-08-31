"""Shared cross-platform PID-liveness check.

Extracted from ``axon.security.cache._pid_alive`` (used there to decide
whether a sealed-project plaintext cache is orphaned) so
``server_client.py``'s single-instance store lock can reuse the same
already-verified logic instead of a second hand-rolled implementation.
"""
from __future__ import annotations

import errno
import os


def pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is still running.

    Cross-platform best-effort check:
    - POSIX: ``os.kill(pid, 0)`` succeeds for live PIDs, raises
      ``ProcessLookupError`` (errno ``ESRCH``) for dead ones, and
      ``PermissionError`` for live PIDs we don't own.
    - Windows: ``os.kill(pid, 0)`` raises ``OSError`` with ``errno``
      ``EINVAL`` for invalid handle (PID never existed / out of range)
      and ``ESRCH`` for "no such process". ``PermissionError`` again
      means a live PID we can't signal.

    On unrecognised errors we treat the PID as **alive** (return True) so
    a caller doesn't act on the assumption a process is gone when it
    can't actually tell — the next check will re-evaluate; wrongly
    treating a live process as dead is the more dangerous failure mode
    for both of this function's callers (wiping a cache still in use,
    or letting a second server start against a store the first one still
    owns).
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # live PID owned by another user
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno in (errno.ESRCH, errno.EINVAL):
            return False
        return True  # unknown failure mode — be conservative
    return True
