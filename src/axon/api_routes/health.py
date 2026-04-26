"""Liveness and readiness probes for the Axon API.

Split from the legacy single `/health` endpoint (audit AUDIT_2026_04_26.md,
"Missing features by effort, Small tier"). Operators can now point Kubernetes
livenessProbe at `/health/live` (always 200 if the process is up) and
readinessProbe at `/health/ready` (200 only when the brain is initialised).

`/health` is preserved as a backward-compatible alias for `/health/ready` so
existing dashboards and uptime checkers do not break.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger("AxonAPI")

router = APIRouter()


def _readiness_payload() -> tuple[dict, int]:
    """Compute the readiness body and status code.

    Returns (payload, status_code). 200 when the brain is fully initialised,
    503 while the lifespan handler is still wiring it up (or has failed).
    """
    from axon import api as _api

    brain = _api.brain
    if brain is None:
        return {"status": "initializing"}, 503
    return {
        "status": "ok",
        "project": getattr(brain, "_active_project", "default"),
    }, 200


@router.get("/health/live")
async def health_live():
    """Liveness probe — returns 200 as long as the ASGI process is responding.

    Does NOT check the brain. Kubernetes uses this to decide whether to
    restart the container; we only want a restart on a fully wedged event
    loop, not a slow brain warm-up.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def health_ready():
    """Readiness probe — returns 200 only when ``axon.api.brain`` is ready.

    Kubernetes uses this to decide whether to route traffic to the pod.
    During cold start the brain may take several seconds to load embedding
    models / open the vector store; we want to keep the pod out of the
    load-balancer rotation until that completes.
    """
    payload, status_code = _readiness_payload()
    if status_code == 200:
        return payload
    return JSONResponse(payload, status_code=status_code)


@router.get("/health")
async def health_check():
    """Backward-compatible alias for ``/health/ready``.

    Preserved verbatim from the original ``axon.api_routes.projects.health_check``
    so existing uptime checkers, VS Code extension probes, and dashboards
    keep working unchanged.
    """
    payload, status_code = _readiness_payload()
    if status_code == 200:
        return payload
    return JSONResponse(payload, status_code=status_code)
