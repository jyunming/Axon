"""Security routes for Axon sealed-store operations."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from axon.api_routes._rate_limit import enforce_rate_limit
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
        result = dict(_security.store_status(_current_user_dir()))
    except Exception as exc:
        logger.error("Security status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    # v0.4.0 Item 2: surface the active keyring mode + session cache size
    # so operators can verify config.yaml took effect without poking REPL.
    # The keyring submodule is part of the optional [sealed] extra — on a
    # minimal install we silently omit these fields so the endpoint still
    # works for non-sealed users.
    try:
        from axon.security.keyring import get_keyring_mode, session_cache

        result["keyring_mode"] = get_keyring_mode()
        result["session_cache_size"] = len(session_cache())
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("keyring mode lookup failed: %s", exc)
    return result


@router.post("/security/keyring-mode")
async def security_set_keyring_mode(request: Request):
    """Change the in-process keyring mode for the running API server.

    Body: ``{"mode": "persistent"|"session"|"never"}``. Same caveat as
    REPL: previously stored secrets are NOT migrated; the new mode
    applies only to subsequent ``store_secret`` / ``get_secret`` calls.

    Subject to the global ``X-API-Key`` middleware. Suitable for one-off
    rotation during a planned maintenance window — for permanent change,
    update ``security.keyring_mode`` in ``config.yaml`` and restart.
    """
    # The keyring helper lives in the optional [sealed] extra. On a
    # minimal install, return a controlled 400 with an install hint
    # instead of letting ImportError become a 500 — matches the pattern
    # used by other sealed-store endpoints.
    try:
        from axon.security.keyring import set_keyring_mode
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail=(
                "security.keyring is unavailable. Install the sealed extra: "
                "pip install axon-rag[sealed]"
            ),
        )

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=422,
            detail="Body must be a JSON object with a 'mode' field.",
        )
    mode = body.get("mode")
    if not isinstance(mode, str):
        raise HTTPException(status_code=422, detail="Body must contain 'mode' as a string.")
    try:
        set_keyring_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Best-effort: also poke the running brain config so a subsequent
    # status read mirrors the change. ``set_keyring_mode`` already
    # validated ``mode`` against the same literal set; mypy can't see
    # that at the assignment site, hence the targeted ignore.
    try:
        from axon import api as _api

        if _api.brain is not None:
            _api.brain.config.keyring_mode = mode  # type: ignore[assignment]
    except Exception as exc:
        logger.debug("brain config keyring_mode mirror failed: %s", exc)
    return {"status": "ok", "keyring_mode": mode}


@router.post("/security/wipe-sealed-cache")
async def security_wipe_sealed_cache():
    """Wipe the active sealed-project plaintext cache.

    v0.4.0 Item 3 manual companion to ``security.seal_cache_ephemeral``.
    No-op when no sealed cache is mounted. Always returns 200; the
    response carries ``wiped: bool`` so callers can branch without
    parsing a status string.

    Subject to the global ``X-API-Key`` middleware. Idempotent — safe
    to call defensively (e.g. before logging out a session).
    """
    from axon import api as _api

    if _api.brain is None:
        return {"wiped": False, "reason": "no active brain"}
    try:
        wiped = bool(_api.brain.wipe_sealed_cache())
    except Exception as exc:
        logger.error("wipe_sealed_cache failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"wiped": wiped}


@router.get("/suggestions/passphrase")
async def suggest_passphrase(words: int = 6, separator: str = "-"):
    """Generate a Diceware passphrase from the bundled EFF wordlist.

    Pure helper — does not touch the sealed store. Subject to the global
    ``X-API-Key`` middleware when ``RAG_API_KEY`` is configured (no
    endpoint-specific bypass). Useful for setup wizards / onboarding UIs
    that want to suggest a strong default passphrase before the user
    runs ``/security/bootstrap``.

    Returns ``{passphrase, n_words, entropy_bits, separator, source}``.
    """
    from axon.security.wordlist import (
        estimate_entropy_bits,
        generate_passphrase,
    )

    try:
        phrase = generate_passphrase(n_words=words, separator=separator)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "passphrase": phrase,
        "n_words": words,
        "entropy_bits": estimate_entropy_bits(words),
        "separator": separator,
        "source": "eff_large_wordlist",
    }


@router.post("/security/bootstrap")
async def security_bootstrap(request: SecurityBootstrapRequest, req: Request):
    from axon import security as _security

    enforce_rate_limit(req, bucket="security_bootstrap", max_hits=10, window_seconds=60.0)
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
async def security_change_passphrase(request: SecurityChangePassphraseRequest, req: Request):
    from axon import security as _security

    enforce_rate_limit(req, bucket="security_change_passphrase", max_hits=10, window_seconds=60.0)
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
