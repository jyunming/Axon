"""
axon/runtime.py — Per-project write-lease registry.

Tracks in-flight write operations to support drain mode (Phase 3) and
epoch-based fencing for stale writers.

Design
------
Each named project has a ``_ProjectLeaseState`` that records:
  - ``active``   — number of write operations currently in flight
  - ``epoch``    — monotonic counter bumped on ``switch_project()`` to fence
                   writes that started under a previous project activation
  - ``draining`` — True when the project is in maintenance drain mode;
                   new ``acquire()`` calls are rejected while active > 0

Typical flow
------------
1. ``POST /project/maintenance {"state": "draining"}``
   → ``set_maintenance_state(name, "draining")``  (meta.json, persistent)
   → ``registry.start_drain(name)``               (runtime, process-local)

2. All new ingest/finalize calls raise PermissionError via
   ``_assert_write_allowed()`` (reads meta.json) **and** via
   ``registry.acquire()`` (in-memory, defense-in-depth).

3. In-flight ingest calls complete; each holds a ``_WriteLease`` that
   decrements the active counter on ``close()`` or GC.

4. ``GET /project/maintenance?name=...`` → ``active_leases`` reaches 0.

5. Admin transitions to ``readonly`` or ``offline``.
"""

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


@dataclass
class _ProjectLeaseState:
    epoch: int = 0
    active: int = 0
    draining: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _drain_event: threading.Event = field(default_factory=threading.Event, repr=False)


# ---------------------------------------------------------------------------
# Lease token returned by acquire()
# ---------------------------------------------------------------------------


class _WriteLease:
    """Represents a single acquired write lease.

    Call ``close()`` when the write operation completes.  ``__del__`` provides
    a safety net so the active counter is decremented even if the caller
    forgets or an exception propagates.
    """

    __slots__ = ("_registry", "_project", "_epoch", "_released")

    def __init__(self, registry: "LeaseRegistry", project: str, epoch: int) -> None:
        self._registry = registry
        self._project = project
        self._epoch = epoch
        self._released = False

    def close(self) -> None:
        """Decrement the active-write counter for the project."""
        if not self._released:
            self._released = True
            self._registry._release(self._project, self._epoch)

    def is_stale(self) -> bool:
        """Return True if the project epoch advanced since this lease was acquired.

        A stale lease means a project switch happened after ingest started.
        Writers should abort their commit phase and discard results rather than
        committing data to the wrong project's stores.
        """
        if self._epoch == -1:  # default project — no epoch tracking
            return False
        state = self._registry._state(self._project)
        with state._lock:
            return state.epoch != self._epoch

    def __del__(self) -> None:  # CPython: called immediately when refcount → 0
        self.close()

    # Make it usable as a context manager too
    def __enter__(self) -> "_WriteLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LeaseRegistry:
    """Thread-safe per-project lease registry.

    All public methods are safe to call from multiple threads concurrently.
    """

    def __init__(self) -> None:
        self._states: dict[str, _ProjectLeaseState] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state(self, project: str) -> _ProjectLeaseState:
        """Return (creating if absent) the state for *project*."""
        with self._global_lock:
            if project not in self._states:
                self._states[project] = _ProjectLeaseState()
            return self._states[project]

    def _release(self, project: str, epoch: int) -> None:
        """Decrement active count.  Called by ``_WriteLease.close()``."""
        if epoch == -1:
            return  # sentinel: default project — no tracking
        state = self._state(project)
        with state._lock:
            state.active = max(0, state.active - 1)
            if state.epoch != epoch:
                logger.warning(
                    "Stale write released for '%s': write epoch=%d, current epoch=%d. "
                    "Data may have been written after a project switch.",
                    project,
                    epoch,
                    state.epoch,
                )
            if state.draining and state.active == 0:
                state._drain_event.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, project: str) -> _WriteLease:
        """Acquire a write lease for *project*.

        Returns a :class:`_WriteLease` token.  Call ``close()`` on it (or
        use it as a context manager) when the write is done.

        Args:
            project: Project name. ``"default"`` is always allowed and is not
                     tracked (no drain semantics on the default project).

        Raises:
            PermissionError: If the project is in drain mode.
        """
        if project == "default":
            return _WriteLease(self, project, -1)
        state = self._state(project)
        with state._lock:
            if state.draining:
                raise PermissionError(
                    f"Cannot write to '{project}': project is draining for maintenance. "
                    "All new writes are blocked until drain completes."
                )
            state.active += 1
            return _WriteLease(self, project, state.epoch)

    def start_drain(self, project: str) -> int:
        """Enter drain mode: block new ``acquire()`` calls.

        Args:
            project: Project name.

        Returns:
            Current number of active (in-flight) write leases.
        """
        if project == "default":
            return 0
        state = self._state(project)
        with state._lock:
            state.draining = True
            state._drain_event.clear()
            if state.active == 0:
                state._drain_event.set()
            return state.active

    def stop_drain(self, project: str) -> None:
        """Exit drain mode (e.g., when returning to ``"normal"`` state)."""
        if project == "default":
            return
        state = self._state(project)
        with state._lock:
            state.draining = False
            state._drain_event.clear()

    def bump_epoch(self, project: str) -> int:
        """Increment the epoch counter to fence stale in-flight writers.

        Called by ``switch_project()`` when the active project changes, so any
        write that started on the previous activation is identifiable as stale
        at release time.

        Args:
            project: Project name.

        Returns:
            The new epoch value.
        """
        if project == "default":
            return 0
        state = self._state(project)
        with state._lock:
            state.epoch += 1
            return state.epoch

    def wait_for_drain(self, project: str, timeout: float = 30.0) -> bool:
        """Block until all active leases are released or *timeout* seconds pass.

        Args:
            project: Project name.
            timeout: Maximum seconds to wait.

        Returns:
            ``True`` if all leases drained within the timeout, ``False`` otherwise.
        """
        if project == "default":
            return True
        state = self._state(project)
        return state._drain_event.wait(timeout=timeout)

    def active_lease_count(self, project: str) -> int:
        """Return the number of currently in-flight write leases."""
        if project == "default":
            return 0
        state = self._state(project)
        with state._lock:
            return state.active

    def snapshot(self, project: str) -> dict:
        """Return a status dict suitable for serialisation.

        Keys: ``project``, ``epoch``, ``active_leases``, ``draining``.
        """
        if project == "default":
            return {"project": "default", "epoch": 0, "active_leases": 0, "draining": False}
        state = self._state(project)
        with state._lock:
            return {
                "project": project,
                "epoch": state.epoch,
                "active_leases": state.active,
                "draining": state.draining,
            }

    def snapshot_all(self) -> list[dict]:
        """Return snapshots for every tracked project that has non-zero leases or is draining.

        Returns:
            List of snapshot dicts (same shape as ``snapshot()``), sorted by project name.
            Projects with ``active_leases == 0`` and ``draining == False`` are omitted
            unless they have a non-zero epoch, to keep the output concise for operators.
        """
        with self._global_lock:
            projects = list(self._states.keys())
        return [self.snapshot(p) for p in sorted(projects) if p != "default"]

    def reset(self, project: str) -> None:
        """Remove tracking state for *project* (used in tests and teardown)."""
        with self._global_lock:
            self._states.pop(project, None)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_registry = LeaseRegistry()


def get_registry() -> LeaseRegistry:
    """Return the process-wide :class:`LeaseRegistry` singleton."""
    return _registry
