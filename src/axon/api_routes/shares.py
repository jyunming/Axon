"""Share key and AxonStore routes."""


from __future__ import annotations

import base64
import getpass
import json
import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from axon.api_routes._rate_limit import enforce_rate_limit
from axon.api_schemas import (
    _VALID_PROJECT_NAME_RE,
    ShareExtendRequest,
    ShareGenerateRequest,
    ShareRedeemRequest,
    ShareRevokeRequest,
    StoreInitRequest,
)

logger = logging.getLogger("AxonAPI")


router = APIRouter()


@router.post("/store/init")
async def store_init(request: StoreInitRequest):
    """Initialise AxonStore at the given base path and update config.yaml."""
    from axon import api as _api
    from axon.main import AxonBrain, AxonConfig
    from axon.projects import ensure_user_project

    brain = _api.brain
    base = Path(request.base_path).expanduser().resolve()
    # Validate: reject null bytes and ensure the resolved path is absolute
    if "\x00" in str(base) or not base.is_absolute():
        raise HTTPException(
            status_code=400, detail="Invalid path: must be an absolute path with no null bytes."
        )
    username = getpass.getuser()
    store_root = base / "AxonStore"
    user_dir = store_root / username
    # Enumerate projects that will become unreachable after the path change
    unreachable: list[str] = []
    if brain:
        try:
            from axon.projects import list_projects as _list_projects

            old_projects = _list_projects()
            new_root = str(user_dir)
            unreachable = [
                p["name"] for p in old_projects if not p.get("path", "").startswith(new_root)
            ]
        except Exception:
            pass
    ensure_user_project(user_dir)
    config = brain.config if brain else AxonConfig()
    config.axon_store_base = str(base)
    # Explicitly update in-memory derived paths; __post_init__ only runs at
    # construction time so we must keep these assignments here.
    # config.save() will NOT write these to config.yaml (axon_store_base branch
    # strips vector_store/bm25 paths so they cannot become stale after a store move).
    config.projects_root = str(user_dir)
    config.vector_store_path = str(user_dir / "default" / "vector_store_data")
    config.bm25_path = str(user_dir / "default" / "bm25_index")
    if request.persist:
        try:
            config.save()
        except Exception as e:
            logger.warning(f"Could not save config after store init: {e}")
    else:
        logger.info("store_init: persist=False — in-memory only, config.yaml not modified")
    if brain:
        brain.close()
    _api.brain = AxonBrain(config)
    _api._source_hashes.clear()
    _api._jobs.clear()
    result: dict = {
        "status": "ok",
        "store_path": str(store_root),
        "user_dir": str(user_dir),
        "username": username,
    }
    if unreachable:
        result["warning"] = (
            f"{len(unreachable)} project(s) will be unreachable until the previous "
            f"store path is restored: {', '.join(unreachable)}"
        )
        result["unreachable_projects"] = unreachable
    return result


@router.get("/store/status")
async def store_status():
    """Return AxonStore initialisation status and metadata.
    Safe to call before brain is ready — reads ``store_meta.json`` directly.
    Clients should poll this on startup to decide whether to show a first-run
    setup UI.
    """
    import json

    from axon import api as _api

    brain = _api.brain
    import getpass as _getpass

    username = _getpass.getuser()
    if brain:
        user_dir = Path(brain.config.projects_root)
    else:
        # Best-effort: reconstruct default path without a live brain
        from axon.main import AxonConfig

        try:
            config = AxonConfig.load(None)
            user_dir = Path(config.projects_root)
        except Exception:
            return {"initialized": False, "path": None, "store_version": None}
    store_meta_path = user_dir / "store_meta.json"
    if not store_meta_path.exists():
        return {
            "initialized": False,
            "path": str(user_dir),
            "store_version": None,
            "username": username,
        }
    try:
        meta = json.loads(store_meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    return {
        "initialized": True,
        "path": str(user_dir),
        "store_version": meta.get("store_version"),
        "store_id": meta.get("store_id"),
        "created_at": meta.get("created_at"),
        "username": username,
    }


@router.get("/store/whoami")
async def store_whoami():
    """Return current user identity and AxonStore status."""
    from axon import api as _api

    brain = _api.brain
    username = getpass.getuser()
    if brain:
        projects_root = Path(brain.config.projects_root)
        return {
            "username": username,
            "store_path": str(projects_root.parent),
            "workspace": str(projects_root.parent),
            "user_dir": str(projects_root),
            "active_project": getattr(brain, "_active_project", "default"),
        }
    return {"username": username}


@router.post("/share/generate")
async def share_generate(request: ShareGenerateRequest, req: Request):
    """Generate a share key allowing another user to access one of your projects."""
    from axon import api as _api
    from axon import governance as gov
    from axon import security as _security
    from axon import shares as _shares

    enforce_rate_limit(req, bucket="share_generate", max_hits=10, window_seconds=60.0)
    if not _VALID_PROJECT_NAME_RE.match(request.project):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid project name: '{request.project}'",
        )
    user_dir = _api._get_user_dir()
    sealed = _security.get_sealed_project_record(request.project, user_dir)
    if sealed is not None:
        if not _security.is_unlocked(user_dir):
            raise HTTPException(
                status_code=409,
                detail="Sealed-store security must be unlocked before generating sealed shares.",
            )
        key_id = f"ssk_{secrets.token_hex(4)}"
        try:
            result = _security.generate_sealed_share(
                owner_user_dir=user_dir,
                project=request.project,
                grantee=request.grantee,
                key_id=key_id,
            )
        except _security.SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        result = {
            **result,
            "key_id": key_id,
            "security_mode": "sealed_v1",
        }
        rid = getattr(req.state, "request_id", "")
        surface = getattr(req.state, "surface", "api")
        gov.emit(
            "share_generated",
            "share",
            key_id,
            project=request.project,
            details={"grantee": request.grantee, "security_mode": "sealed_v1"},
            surface=surface,
            request_id=rid,
        )
        return result
    # Resolve nested projects via subs/ layout (e.g. research/papers → research/subs/papers)
    _segments = request.project.split("/")
    project_path = user_dir / _segments[0]
    for _seg in _segments[1:]:
        project_path = project_path / "subs" / _seg
    if not project_path.exists() or not (project_path / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project}' not found.")
    try:
        result = _shares.generate_share_key(
            owner_user_dir=user_dir,
            project=request.project,
            grantee=request.grantee,
            ttl_days=request.ttl_days,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    gov.emit(
        "share_generated",
        "share",
        result.get("key_id", ""),
        project=request.project,
        details={"grantee": request.grantee, "expires_at": result.get("expires_at")},
        surface=surface,
        request_id=rid,
    )
    return result


@router.post("/share/redeem")
async def share_redeem(request: ShareRedeemRequest, req: Request):
    """Redeem a share string, creating a mount descriptor in your mounts/ directory."""
    from axon import api as _api
    from axon import governance as gov
    from axon import security as _security
    from axon import shares as _shares

    enforce_rate_limit(req, bucket="share_redeem", max_hits=10, window_seconds=60.0)
    user_dir = _api._get_user_dir()
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    # Detect sealed share by the ``SEALED1:`` or ``SEALED2:`` prefix in
    # the decoded envelope (v0.4.0 added SEALED2 with embedded signing
    # pubkey; SEALED1 is still accepted for backward compat). The legacy
    # JSON-with-security_mode detection is kept as a fallback in case
    # any older share_strings using that earlier shape are still in flight.
    is_sealed_share = False
    try:
        from axon.security.share import is_sealed_share_envelope

        decoded = base64.urlsafe_b64decode(request.share_string.encode("ascii")).decode(
            "utf-8", errors="replace"
        )
        if is_sealed_share_envelope(decoded):
            is_sealed_share = True
        else:
            # Fallback: try the legacy JSON-envelope shape.
            try:
                payload = json.loads(decoded)
                if isinstance(payload, dict) and payload.get("security_mode") == "sealed_v1":
                    is_sealed_share = True
            except (json.JSONDecodeError, ValueError):
                pass
    except Exception:
        pass
    if is_sealed_share:
        try:
            result = _security.redeem_sealed_share(user_dir, request.share_string)
        except _security.SecurityError as e:
            raise HTTPException(status_code=400, detail=str(e))
        gov.emit(
            "share_redeemed",
            "share",
            result.get("key_id", ""),
            project=result.get("project", ""),
            details={"owner": result.get("owner", ""), "security_mode": "sealed_v1"},
            surface=surface,
            request_id=rid,
        )
        return result
    try:
        result = _shares.redeem_share_key(
            grantee_user_dir=user_dir,
            share_string=request.share_string,
        )
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    gov.emit(
        "share_redeemed",
        "share",
        result.get("key_id", ""),
        project=result.get("project", ""),
        details={"owner": result.get("owner", "")},
        surface=surface,
        request_id=rid,
    )
    return result


@router.post("/share/revoke")
async def share_revoke(request: ShareRevokeRequest, req: Request):
    """Revoke a share key.
    Routing:
    - ``key_id`` starting with ``ssk_`` → sealed-share revoke (Phase 4).
      Soft (default) deletes the wrap; hard (``rotate=true``) rotates
      the project DEK and invalidates ALL shares — the body's
      ``project`` field is required so the wrap file can be located.
    - Any other ``key_id`` → legacy plaintext-mount share revoke.
    """
    from axon import api as _api
    from axon import governance as gov
    from axon import security as _security
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    # Sealed-share revoke is keyed off the ``ssk_`` key_id prefix
    # produced by ``api_routes/shares.py::share_generate`` for sealed
    # projects.
    if request.key_id.startswith("ssk_"):
        if not request.project:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Sealed-share revoke requires the body's ``project`` "
                    "field so the wrap file can be located."
                ),
            )
        try:
            result = _security.revoke_sealed_share(
                owner_user_dir=user_dir,
                project=request.project,
                key_id=request.key_id,
                rotate=request.rotate,
            )
        except _security.SecurityError as exc:
            message = str(exc)
            # The most common "404"-shaped errors from
            # security.revoke_sealed_share: project doesn't exist,
            # project not sealed, no share wrap for that key_id.
            # The substring set covers _soft_revoke ("No sealed-share
            # wrap exists ..."), _resolve_owned_project_dir error
            # ("does not exist"), and is_project_sealed error
            # ("is not sealed"). Other SecurityError instances (locked
            # store, stage I/O failure) map to 400.
            msg_lower = message.lower()
            is_not_found = (
                "no sealed-share wrap" in msg_lower
                or "does not exist" in msg_lower
                or "is not sealed" in msg_lower
            )
            status = 404 if is_not_found else 400
            raise HTTPException(status_code=status, detail=message)
        gov.emit(
            "share_revoked",
            "share",
            request.key_id,
            project=request.project,
            details={
                "security_mode": "sealed_v1",
                "rotate": request.rotate,
                "files_resealed": result.get("files_resealed", 0),
                "invalidated_share_key_ids": result.get("invalidated_share_key_ids", []),
            },
            surface=surface,
            request_id=rid,
        )
        return result
    try:
        result = _shares.revoke_share_key(owner_user_dir=user_dir, key_id=request.key_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    gov.emit(
        "share_revoked",
        "share",
        request.key_id,
        project=result.get("project", ""),
        surface=surface,
        request_id=rid,
    )
    return result


@router.post("/share/extend")
async def share_extend(request: ShareExtendRequest, req: Request):
    """Renew a share key's expiry — or clear it (``ttl_days=null``).
    Pairs with ``POST /share/generate``'s ``ttl_days`` to give owners a
    hard cutoff for forgotten shares while still letting them keep an
    in-use share alive on demand.
    """
    from axon import api as _api
    from axon import governance as gov
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    try:
        result = _shares.extend_share_key(
            owner_user_dir=user_dir,
            key_id=request.key_id,
            ttl_days=request.ttl_days,
        )
    except ValueError as e:
        msg = str(e)
        # 404 when the key is unknown; 409 when it is revoked (irreversible);
        # 422 for malformed ttl. Keep the existing 422-on-validation pattern.
        if "not found" in msg.lower():
            status = 404
        elif "revoked" in msg.lower():
            status = 409
        else:
            status = 422
        raise HTTPException(status_code=status, detail=msg)
    gov.emit(
        "share_extended",
        "share",
        request.key_id,
        project=result.get("project", ""),
        details={"new_expires_at": result.get("expires_at")},
        surface=surface,
        request_id=rid,
    )
    return result


@router.get("/share/list")
async def share_list():
    """List shares for the current user: both issued (sharing) and received (shared)."""
    from axon import api as _api
    from axon import security as _security
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    removed_open = _shares.validate_received_shares(user_dir)
    removed_sealed = _security.validate_received_sealed_shares(user_dir)
    open_result = _shares.list_shares(user_dir)
    sealed_result = _security.list_sealed_shares(user_dir)

    def _tag(records: list[dict], security_mode: str) -> list[dict]:
        return [{**record, "security_mode": security_mode} for record in records]

    result = {
        "sharing": _tag(open_result.get("sharing", []), "open")
        + _tag(sealed_result.get("sharing", []), "sealed_v1"),
        "shared": _tag(open_result.get("shared", []), "open")
        + _tag(sealed_result.get("shared", []), "sealed_v1"),
    }
    removed = [*removed_open, *removed_sealed]
    if removed:
        result["removed_stale"] = removed
    return result
