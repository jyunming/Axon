"""Access control policy for Axon projects.

Centralises readonly-scope, mounted-share, and maintenance-state write guards
that were previously ad-hoc methods on ``AxonBrain``.
"""

from __future__ import annotations

__all__ = ["check_write_allowed", "is_mounted_share_path"]


def is_mounted_share_path(project_name: str) -> bool:
    """Return True when *project_name* refers to a received share mount.

    Detects both the canonical ``mounts/<name>`` format and the legacy
    ``ShareMount/<name>`` format for backwards compatibility.
    """
    if project_name == "default":
        return False
    parts = project_name.split("/")
    return parts[0] in ("mounts", "ShareMount")


def check_write_allowed(
    operation: str,
    active_project: str,
    read_only_scope: bool,
    is_mounted: bool,
) -> None:
    """Raise ``PermissionError`` if a write operation is not permitted.

    Checks (in order):

    1. Read-only merged scope (``@projects`` / ``@mounts`` / ``@store``)
    2. Mounted share — always read-only regardless of ``write_access`` flag
    3. Project maintenance state (``readonly``, ``offline``, ``draining``)

    Args:
        operation:      Human-readable name of the attempted operation (e.g. "ingest").
        active_project: Name of the currently active project.
        read_only_scope: True when the session is in a merged read-only view.
        is_mounted:      True when the active project is a received share mount.
    """
    if read_only_scope:
        raise PermissionError(
            f"Cannot {operation}: active scope is read-only (@projects / @mounts / @store)."
        )
    if is_mounted:
        raise PermissionError(
            f"Cannot {operation} on mounted share '{active_project}'. "
            "Mounted projects are always read-only. Use your own project for writes."
        )
    if active_project != "default":
        try:
            from axon.projects import get_maintenance_state

            _state = get_maintenance_state(active_project)
            if _state in ("readonly", "offline", "draining"):
                raise PermissionError(
                    f"Cannot {operation}: project '{active_project}' is in "
                    f"'{_state}' maintenance state."
                )
        except PermissionError:
            raise
        except Exception:
            pass
