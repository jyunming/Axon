"""Share key and AxonStore routes."""


from __future__ import annotations

import getpass
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from axon.api_schemas import (
    _VALID_PROJECT_NAME_RE,
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

    config.vector_store_path = str(user_dir / "default" / "vector_data")

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
        return {
            "username": username,
            "store_path": str(Path(brain.config.projects_root).parent),
            "user_dir": brain.config.projects_root,
            "active_project": getattr(brain, "_active_project", "default"),
        }

    return {"username": username}


@router.post("/share/generate")
async def share_generate(request: ShareGenerateRequest, req: Request):
    """Generate a share key allowing another user to access one of your projects."""

    from axon import api as _api
    from axon import governance as gov
    from axon import shares as _shares

    if not _VALID_PROJECT_NAME_RE.match(request.project):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid project name: '{request.project}'",
        )

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
