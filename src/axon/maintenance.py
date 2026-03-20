"""Operator-facing maintenance state management.

Combines project-metadata persistence (``projects.py``) with lease-registry
coordination (``runtime.py``) into single, consistent operations so that API
handlers and other callers do not need to orchestrate both themselves.
"""

from __future__ import annotations

__all__ = ["apply_maintenance_state", "get_maintenance_status"]


def apply_maintenance_state(name: str, state: str) -> dict:
    """Persist *state* on *name* and synchronise the lease registry.

    When *state* is ``"draining"`` the registry drain is started so that
    in-flight writes can complete before the project goes fully read-only.
    When *state* is ``"normal"`` any active drain is stopped.

    Returns:
        A dict with keys ``status``, ``project``, ``maintenance_state``,
        ``active_leases``, and ``epoch``.

    Raises:
        ValueError: If *state* is invalid or *name* does not exist.
    """
    from axon.projects import set_maintenance_state
    from axon.runtime import get_registry as _get_registry

    set_maintenance_state(name, state)
    reg = _get_registry()
    if state == "draining":
        reg.start_drain(name)
    elif state == "normal":
        reg.stop_drain(name)
    snap = reg.snapshot(name)
    return {
        "status": "ok",
        "project": name,
        "maintenance_state": state,
        "active_leases": snap["active_leases"],
        "epoch": snap["epoch"],
    }


def get_maintenance_status(name: str) -> dict:
    """Return the current maintenance state and registry snapshot for *name*.

    Returns:
        A dict with keys ``project``, ``maintenance_state``, ``active_leases``,
        ``epoch``, and ``draining``.
    """
    from axon.projects import get_maintenance_state
    from axon.runtime import get_registry as _get_registry

    state = get_maintenance_state(name)
    snap = _get_registry().snapshot(name)
    return {
        "project": name,
        "maintenance_state": state,
        "active_leases": snap["active_leases"],
        "epoch": snap["epoch"],
        "draining": snap["draining"],
    }
