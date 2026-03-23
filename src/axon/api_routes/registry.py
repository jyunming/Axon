"""Registry and lease routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/registry/leases")
async def get_registry_leases():
    """Return active write-lease counts for all tracked projects."""
    from axon.runtime import get_registry as _get_registry

    reg = _get_registry()
    snapshots = reg.snapshot_all()
    return {"leases": snapshots, "total_projects_tracked": len(snapshots)}
