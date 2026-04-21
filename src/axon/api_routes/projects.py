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
    ProjectRotateKeysRequest,
    ProjectSealRequest,
    ProjectSwitchRequest,
)

logger = logging.getLogger("AxonAPI")


router = APIRouter()


def _require_local_turboquantdb(brain, action: str = "This action") -> None:
    """Raise HTTPException 400 if brain is not using local turboquantdb."""
    if getattr(brain.config, "vector_store", "turboquantdb") != "turboquantdb":
        raise HTTPException(
            status_code=400,
            detail=f"{action} requires turboquantdb as the vector store.",
        )
    if getattr(brain.config, "qdrant_url", ""):
        raise HTTPException(
            status_code=400,
            detail=f"{action} cannot be used with a remote qdrant_url.",
        )


@router.get("/health")
async def health_check():
    """Return 200 with status 'ok' when the brain is ready; 503 when not yet available."""

    from axon import api as _api

    brain = _api.brain

    if brain is None:
        return JSONResponse({"status": "initializing"}, status_code=503)

    return {"status": "ok", "project": getattr(brain, "_active_project", "default")}


_SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "gemini_api_key",
        "ollama_cloud_key",
        "copilot_pat",
        "brave_api_key",
        "qdrant_api_key",
    }
)


@router.get("/config")
async def get_config():
    """Return the current active configuration with sensitive fields masked."""

    from dataclasses import asdict

    from axon import api as _api

    brain = _api.brain

    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    try:
        data = asdict(brain.config)

    except TypeError:
        return brain.config

    for field in _SENSITIVE_FIELDS:
        if field in data and data[field]:
            data[field] = "***"

    return data


@router.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """Update global configuration settings."""

    from axon import api as _api

    brain = _api.brain

    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    update_data = request.model_dump(exclude_unset=True)

    persist = update_data.pop("persist", False)

    reinit_llm = "llm_provider" in update_data or "llm_model" in update_data

    reinit_embed = "embedding_provider" in update_data or "embedding_model" in update_data

    reinit_rerank = "reranker_model" in update_data

    applied_keys = []

    for k, v in update_data.items():
        if hasattr(brain.config, k):
            setattr(brain.config, k, v)

            applied_keys.append(k)

    if reinit_llm:
        from axon.llm import OpenLLM

        brain.llm = OpenLLM(brain.config)

    if reinit_embed:
        from axon.embeddings import OpenEmbedding

        brain.embedding = OpenEmbedding(brain.config)

    if reinit_rerank and brain.config.rerank:
        from axon.rerank import OpenReranker

        brain.reranker = OpenReranker(brain.config)

    if persist:
        brain.config.save()

    return {
        "status": "success",
        "config": brain.config,
        "persisted": persist,
        "applied": applied_keys,
    }


@router.get("/projects")
async def get_projects():
    """List all projects known to the Axon system."""

    from axon import api as _api
    from axon import security as _security
    from axon import shares as _shares

    brain = _api.brain

    from axon.projects import is_reserved_top_level_name as _is_reserved_top_level_name
    from axon.projects import list_projects as _list_projects

    user_dir: Path | None = None
    if brain:
        user_dir = Path(brain.config.projects_root)
        try:
            _shares.validate_received_shares(user_dir)

        except Exception as exc:
            logger.warning(f"Could not validate received shares: {exc}")

        try:
            _security.validate_received_sealed_shares(user_dir)

        except Exception as exc:
            logger.warning(f"Could not validate received sealed shares: {exc}")

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
        if n not in on_disk_names and n != "_global" and not _is_reserved_top_level_name(n)
    ]

    shared_mounts = []

    if brain:
        try:
            from axon.projects import list_share_mounts

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

    result: dict = {
        "projects": on_disk,
        "memory_only": memory_only,
        "shared_mounts": shared_mounts,
        "total": len(on_disk) + len(memory_only) + len(shared_mounts),
    }

    # Merge security-store status into the response.
    if user_dir is not None:
        try:
            sec_status = _security.store_status(user_dir)
            unlocked = sec_status.get("unlocked", False)
            result["locked"] = not unlocked
            result["sealed_hidden_count"] = sec_status.get("sealed_hidden_count", 0)
            result["public_key_fingerprint"] = sec_status.get("public_key_fingerprint", "")
            result["cipher_suite"] = sec_status.get("cipher_suite", "")
        except Exception as exc:
            logger.warning(f"Could not retrieve security store status: {exc}")

    return result


@router.post("/project/new")
async def create_project(request: ProjectCreateRequest):
    """Create a new project directory and metadata."""

    from axon import api as _api
    from axon.projects import ensure_project

    if not request.name or not _VALID_PROJECT_NAME_RE.match(request.name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid project name. Use 1-5 slash-separated segments of "
                "1-50 alphanumeric characters, hyphens, or underscores "
                "(e.g. 'research/papers/2024')."
            ),
        )

    if request.security_mode == "sealed_v1":
        brain = _api.brain
        if brain is not None:
            _require_local_turboquantdb(brain, "Sealed projects")

    try:
        ensure_project(
            request.name,
            description=request.description,
            security_mode=request.security_mode,
        )

        result: dict = {"status": "success", "project": request.name}
        if request.security_mode is not None:
            result["security_mode"] = request.security_mode
        return result

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

        if project_name.startswith("mounts/"):
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

    except HTTPException:
        raise

    except ValueError as e:
        detail = str(e)

        lowered = detail.lower()

        if "must be provided" in lowered:
            status = 400

        elif "reserved" in lowered:
            status = 400

        else:
            status = 404

        raise HTTPException(status_code=status, detail=detail)

    except Exception as e:
        logger.error(f"Project switch failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/delete/{name}")
async def delete_project_endpoint(name: str):
    """Delete a project and all its data."""

    from axon import api as _api
    from axon import shares as _shares

    brain = _api.brain

    from axon.projects import ProjectHasChildrenError, delete_project

    if name.startswith("mounts/") or name == "mounts":
        raise HTTPException(
            status_code=400,
            detail="Mounted share entries are read-only and cannot be deleted via this endpoint.",
        )

    if brain:
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

        to_remove = [k for k in _api._source_hashes if k == name or k.startswith(f"{name}/")]

        for key in to_remove:
            _api._source_hashes.pop(key, None)

        return {"status": "success", "message": f"Project '{name}' deleted."}

    except ProjectHasChildrenError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/project/rotate-keys")
async def rotate_project_keys(request: ProjectRotateKeysRequest):
    """Rotate the cryptographic keys for a sealed project."""

    from axon import api as _api
    from axon import security as _security

    brain = _api.brain

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    user_dir = Path(brain.config.projects_root).expanduser().resolve()

    try:
        project_root = _security.resolve_owned_sealed_project_path(request.project_name, user_dir)
    except _security.SecurityError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        result = _security.project_rotate_keys(project_root)
        return result
    except _security.SecurityError as exc:
        message = str(exc)
        status = 409 if "pending" in message.lower() else 400
        raise HTTPException(status_code=status, detail=message)


@router.post("/project/seal")
async def seal_project(request: ProjectSealRequest):
    """Convert an existing open project to sealed_v1 mode."""

    from axon import api as _api
    from axon import security as _security

    brain = _api.brain

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    _require_local_turboquantdb(brain, "Sealing a project")

    user_dir = Path(brain.config.projects_root).expanduser().resolve()

    previously_active = getattr(brain, "_active_project", None)
    needs_switch_back = previously_active == request.project_name
    if needs_switch_back:
        brain.switch_project("default")

    try:
        _security.project_seal(
            request.project_name,
            user_dir,
            migration_mode=request.migration_mode,
            config=brain.config,
            embedding=brain.embedding,
        )
    except _security.SecurityError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if needs_switch_back:
            brain.switch_project(previously_active)

    return {
        "status": "success",
        "project": request.project_name,
        "security_mode": "sealed_v1",
        "migration_mode": request.migration_mode,
    }


@router.get("/sessions")
async def list_sessions():
    """List all saved chat sessions for the active project."""

    from axon import api as _api
    from axon.sessions import _list_sessions

    brain = _api.brain

    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    project = getattr(brain, "_active_project", None)

    return {"sessions": _list_sessions(project=project)}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve a specific session by ID."""

    from axon import api as _api
    from axon.sessions import _load_session

    brain = _api.brain

    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    project = getattr(brain, "_active_project", None)

    session = _load_session(session_id, project=project)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session
