"""Security routes for Axon sealed-store operations."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from axon.api_schemas import (
    SecurityBootstrapRequest,
    SecurityChangePassphraseRequest,
    SecurityUnlockRequest,
)

logger = logging.getLogger("AxonAPI")

_unlock_failures: dict[str, list[float]] = {}  # ip → list of failure timestamps
_UNLOCK_MAX_ATTEMPTS = 5
_UNLOCK_WINDOW_SECS = 300
# Hard cap on the failures dict so a steady stream of attacker IPs
# cannot grow the dict without bound (audit P1).
_UNLOCK_FAILURES_MAX_IPS = 10000


router = APIRouter()


def _evict_stale_unlock_failures(now: float) -> None:
    """Drop IPs whose newest failure is outside the lockout window.
    Called from the unlock path so eviction amortises naturally with
    real traffic — no background timer needed.
    """
    stale = [
        ip
        for ip, ts_list in _unlock_failures.items()
        if not ts_list or now - ts_list[-1] >= _UNLOCK_WINDOW_SECS
    ]
    for ip in stale:
        _unlock_failures.pop(ip, None)
    if len(_unlock_failures) > _UNLOCK_FAILURES_MAX_IPS:
        # Last-resort cap: drop oldest-attempted IPs first.
        sorted_ips = sorted(
            _unlock_failures.items(),
            key=lambda kv: kv[1][-1] if kv[1] else 0,
        )
        for ip, _ in sorted_ips[: len(_unlock_failures) - _UNLOCK_FAILURES_MAX_IPS]:
            _unlock_failures.pop(ip, None)


def _current_user_dir() -> Path:
    from axon import api as _api
    from axon.main import AxonConfig

    if _api.brain is not None:
        return Path(_api.brain.config.projects_root).expanduser().resolve()
    config = AxonConfig.load(None)
    return Path(config.projects_root).expanduser().resolve()


@router.get("/security/status")
async def security_status():
    from axon import security as _security

    try:
        return _security.store_status(_current_user_dir())
    except Exception as exc:
        logger.error("Security status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/security/bootstrap")
async def security_bootstrap(request: SecurityBootstrapRequest):
    from axon import security as _security

    try:
        return _security.bootstrap_store(_current_user_dir(), request.passphrase.get_secret_value())
    except _security.SecurityError as exc:
        message = str(exc)
        status = 409 if "already bootstrapped" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message)


@router.post("/security/unlock")
async def security_unlock(request: SecurityUnlockRequest, req: Request):
    from axon import security as _security

    client_ip = req.client.host if req.client else "unknown"
    now = time.time()
    _evict_stale_unlock_failures(now)
    # Clean up timestamps outside the lockout window
    failures = _unlock_failures.get(client_ip, [])
    failures = [t for t in failures if now - t < _UNLOCK_WINDOW_SECS]
    _unlock_failures[client_ip] = failures
    if len(failures) >= _UNLOCK_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")
    try:
        result = _security.unlock_store(_current_user_dir(), request.passphrase.get_secret_value())
        _unlock_failures.pop(client_ip, None)
        return result
    except _security.SecurityError as exc:
        message = str(exc)
        # Only credential failures count toward rate-limiting. A "store
        # not bootstrapped" or "keyring unavailable" error must not
        # tick down the user's attempt budget — those are environment
        # bugs, not credential brute force (audit P1).
        msg_lower = message.lower()
        is_auth_failure = "wrong passphrase" in msg_lower or "unlock failed" in msg_lower
        if is_auth_failure:
            _unlock_failures[client_ip].append(time.time())
        status = 401 if is_auth_failure else 400
        raise HTTPException(status_code=status, detail=message)


@router.post("/security/lock")
async def security_lock():
    from axon import security as _security

    try:
        return _security.lock_store(_current_user_dir())
    except _security.SecurityError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/security/change-passphrase")
async def security_change_passphrase(request: SecurityChangePassphraseRequest):
    from axon import security as _security

    try:
        return _security.change_passphrase(
            _current_user_dir(),
            request.old_passphrase.get_secret_value(),
            request.new_passphrase.get_secret_value(),
        )
    except _security.SecurityError as exc:
        message = str(exc)
        status = 401 if "passphrase" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message)
