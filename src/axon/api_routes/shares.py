"""Share key and AxonStore routes."""
from __future__ import annotations

import getpass
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from axon.api_schemas import (
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
    from axon.projects import ensure_user_namespace

    brain = _api.brain
    base = Path(request.base_path).expanduser().resolve()
    username = getpass.getuser()
    store_root = base / "AxonStore"
    user_dir = store_root / username

    ensure_user_namespace(user_dir)

    config = brain.config if brain else AxonConfig()
    config.axon_store_base = str(base)
    config.axon_store_mode = True
    config.projects_root = str(user_dir)
    config.vector_store_path = str(user_dir / "default" / "chroma_data")
    config.bm25_path = str(user_dir / "default" / "bm25_index")

    try:
        config.save()
    except Exception as e:
        logger.warning(f"Could not save config after store init: {e}")

    if brain:
        brain.close()
    _api.brain = AxonBrain(config)

    return {
        "status": "ok",
        "store_path": str(store_root),
        "user_dir": str(user_dir),
        "username": username,
    }


@router.get("/store/whoami")
async def store_whoami():
    """Return current user identity and AxonStore status."""
    from axon import api as _api

    brain = _api.brain
    username = getpass.getuser()
    if brain and brain.config.axon_store_mode:
        return {
            "username": username,
            "store_path": str(Path(brain.config.projects_root).parent.parent),
            "user_dir": brain.config.projects_root,
            "active_project": getattr(brain, "_active_project", "default"),
            "store_mode": True,
        }
    return {"username": username, "store_mode": False}


@router.post("/share/generate")
async def share_generate(request: ShareGenerateRequest, req: Request):
    """Generate a share key allowing another user to access one of your projects."""
    from axon import api as _api
    from axon import governance as gov
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    # Resolve nested projects via subs/ layout (e.g. research/papers → research/subs/papers)
    _segments = request.project.split("/")
    project_path = user_dir / _segments[0]
    for _seg in _segments[1:]:
        project_path = project_path / "subs" / _seg
    if not project_path.exists() or not (project_path / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project}' not found.")
    result = _shares.generate_share_key(
        owner_user_dir=user_dir,
        project=request.project,
        grantee=request.grantee,
    )
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    gov.emit(
        "share_generated",
        "share",
        result.get("key_id", ""),
        project=request.project,
        details={"grantee": request.grantee},
        surface=surface,
        request_id=rid,
    )
    return result


@router.post("/share/redeem")
async def share_redeem(request: ShareRedeemRequest, req: Request):
    """Redeem a share string, creating a mount descriptor in your mounts/ directory."""
    from axon import api as _api
    from axon import governance as gov
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
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
    """Revoke a share key."""
    from axon import api as _api
    from axon import governance as gov
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
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


@router.get("/share/list")
async def share_list():
    """List shares for the current user: both issued (sharing) and received (shared)."""
    from axon import api as _api
    from axon import shares as _shares

    user_dir = _api._get_user_dir()
    removed = _shares.validate_received_shares(user_dir)
    result = _shares.list_shares(user_dir)
    if removed:
        result["removed_stale"] = removed
    return result
