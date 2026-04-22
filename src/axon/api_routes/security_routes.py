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


router = APIRouter()


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
        return _security.bootstrap_store(_current_user_dir(), request.passphrase)
    except _security.SecurityError as exc:
        message = str(exc)
        status = 409 if "already bootstrapped" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message)


@router.post("/security/unlock")
async def security_unlock(request: SecurityUnlockRequest, req: Request):
    from axon import security as _security

    client_ip = req.client.host if req.client else "unknown"
    now = time.time()

    # Clean up timestamps outside the lockout window
    failures = _unlock_failures.get(client_ip, [])
    failures = [t for t in failures if now - t < _UNLOCK_WINDOW_SECS]
    _unlock_failures[client_ip] = failures

    if len(failures) >= _UNLOCK_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")

    try:
        result = _security.unlock_store(_current_user_dir(), request.passphrase)
        _unlock_failures.pop(client_ip, None)
        return result
    except _security.SecurityError as exc:
        message = str(exc)
        _unlock_failures[client_ip].append(time.time())
        status = 401 if "unlock failed" in message.lower() else 400
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
            request.old_passphrase,
            request.new_passphrase,
        )
    except _security.SecurityError as exc:
        message = str(exc)
        status = 401 if "passphrase" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message)
