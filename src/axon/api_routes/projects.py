"""Project management, sessions, config, and health routes."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from axon.api_schemas import (
    _VALID_PROJECT_NAME_RE,
    ConfigUpdateRequest,
    ProjectCreateRequest,
    ProjectSwitchRequest,
)

logger = logging.getLogger("AxonAPI")
router = APIRouter()


@router.get("/health")
async def health_check():
    """Return 200 with status 'ok' when the brain is ready; 503 when not yet available."""
    from axon import api as _api

    brain = _api.brain
    if brain is None:
        return JSONResponse({"status": "initializing"}, status_code=503)
    return {"status": "ok", "project": getattr(brain, "_active_project", "default")}


@router.get("/config")
async def get_config():
    """Return the current active configuration."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return brain.config


@router.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """Update global configuration settings."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    update_data = request.dict(exclude_unset=True)
    persist = update_data.pop("persist", False)

    reinit_llm = "llm_provider" in update_data or "llm_model" in update_data
    reinit_embed = "embedding_provider" in update_data or "embedding_model" in update_data
    reinit_rerank = "reranker_model" in update_data

    for k, v in update_data.items():
        if hasattr(brain.config, k):
            setattr(brain.config, k, v)

    if reinit_llm:
        from axon.main import OpenLLM

        brain.llm = OpenLLM(brain.config)
    if reinit_embed:
        from axon.main import OpenEmbedding

        brain.embedding = OpenEmbedding(brain.config)
    if reinit_rerank and brain.config.rerank:
        from axon.main import OpenReranker

        brain.reranker = OpenReranker(brain.config)

    if persist:
        brain.config.save()

    return {"status": "success", "config": brain.config, "persisted": persist}


@router.get("/projects")
async def get_projects():
    """List all projects known to the Axon system."""
    from axon import api as _api
    from axon import shares as _shares
    from axon.projects import list_projects as _list_projects

    brain = _api.brain

    if brain and brain.config.axon_store_mode:
        try:
            user_dir = Path(brain.config.projects_root)
            _shares.validate_received_shares(user_dir)
        except Exception as exc:
            logger.warning(f"Could not validate received shares: {exc}")

    try:
        on_disk = _list_projects()
    except Exception as exc:
        logger.warning(f"Could not enumerate on-disk projects: {exc}")
        on_disk = []

    in_memory = list(_api._source_hashes.keys())
    on_disk_names = {p["name"] for p in on_disk}
    memory_only = [
        {"name": n, "source": "memory_only"}
        for n in in_memory
        if n not in on_disk_names and n != "_global"
    ]

    shared_mounts = []
    if brain and brain.config.axon_store_mode:
        try:
            from axon.projects import list_share_mounts

            user_dir = Path(brain.config.projects_root)
            mounts = list_share_mounts(user_dir)
            shared_mounts = [
                {
                    "name": f"mounts/{m['name']}",
                    "owner": m["owner"],
                    "project": m["project"],
                    "is_broken": m["is_broken"],
                    "is_shared": True,
                }
                for m in mounts
                if not m["is_broken"]
            ]
        except Exception as exc:
            logger.warning(f"Could not enumerate share mounts: {exc}")

    return {
        "projects": on_disk,
        "memory_only": memory_only,
        "shared_mounts": shared_mounts,
        "total": len(on_disk) + len(memory_only) + len(shared_mounts),
    }


@router.post("/project/new")
async def create_project(request: ProjectCreateRequest):
    """Create a new project directory and metadata."""
    from axon.projects import ensure_project

    if not request.name or not _VALID_PROJECT_NAME_RE.match(request.name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid project name. Use 1-5 slash-separated segments of "
                "1-64 alphanumeric characters, hyphens, or underscores "
                "(e.g. 'research/papers/2024')."
            ),
        )
    try:
        ensure_project(request.name, description=request.description)
        return {"status": "success", "project": request.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/switch")
async def switch_project(request: ProjectSwitchRequest):
    """Switch the active project, reinitializing vector store and BM25."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        project_name = request.final_name

        # For mounted projects, validate revocation before switching so a revoked
        # share is caught at switch time, not only at project-list time.
        if project_name.startswith("mounts/") and brain.config.axon_store_mode:
            from axon import shares as _shares

            user_dir = Path(brain.config.projects_root)
            mount_name = project_name[len("mounts/") :]
            try:
                removed = _shares.validate_received_shares(user_dir)
                if mount_name in removed:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Mounted project '{project_name}' has been revoked by the owner.",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning(f"Could not validate share before switch: {exc}")

        brain.switch_project(project_name)
        return {"status": "success", "active_project": project_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Project switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/delete/{name}")
async def delete_project_endpoint(name: str):
    """Delete a project and all its data."""
    from axon import api as _api
    from axon import shares as _shares
    from axon.projects import ProjectHasChildrenError, delete_project

    brain = _api.brain

    if name.startswith("mounts/") or name == "mounts":
        raise HTTPException(
            status_code=400,
            detail="Mounted share entries are read-only and cannot be deleted via this endpoint.",
        )

    if brain and brain.config.axon_store_mode:
        user_dir = Path(brain.config.projects_root)
        shares_info = _shares.list_shares(user_dir)
        active_grantees = [
            s["grantee"]
            for s in shares_info.get("sharing", [])
            if s["project"] == name and not s.get("revoked", False)
        ]
        if active_grantees:
            raise HTTPException(
                status_code=409,
                detail=f"Project '{name}' has active shares with: {', '.join(active_grantees)}. Revoke shares before deleting.",
            )

    if brain and brain._active_project == name:
        brain.switch_project("default")

    try:
        delete_project(name)
        return {"status": "success", "message": f"Project '{name}' deleted."}
    except ProjectHasChildrenError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions")
async def list_sessions():
    """List all saved chat sessions for the active project."""
    from axon import api as _api
    from axon.main import _list_sessions

    brain = _api.brain
    project = getattr(brain, "_active_project", None) if brain else None
    return {"sessions": _list_sessions(project=project)}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve a specific session by ID."""
    from axon import api as _api
    from axon.main import _load_session

    brain = _api.brain
    project = getattr(brain, "_active_project", None) if brain else None
    session = _load_session(session_id, project=project)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
