"""Factory for creating GraphBackend instances.

Usage::

    from axon.graph_backends.factory import get_graph_backend
    backend = get_graph_backend(brain)

The backend type is determined by ``brain.config.graph_backend``:
  - ``"graphrag"`` (default) → :class:`GraphRagBackend`
  - ``"dynamic_graph"`` → :class:`DynamicGraphBackend` (SQLite, DELETE journal mode, v0.3)

Adding a new backend type: register it in ``_BACKEND_REGISTRY`` below.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axon.graph_backends.base import GraphBackend

if TYPE_CHECKING:
    pass

# Registry maps config string → callable(brain) -> GraphBackend
# Populated lazily to avoid import-time circular deps.
_BACKEND_REGISTRY: dict[str, type] = {}


def _registry() -> dict[str, type]:
    global _BACKEND_REGISTRY
    if not _BACKEND_REGISTRY:
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend
        from axon.graph_backends.federated_backend import FederatedGraphBackend
        from axon.graph_backends.graphrag_backend import GraphRagBackend

        _BACKEND_REGISTRY = {
            "graphrag": GraphRagBackend,
            "dynamic_graph": DynamicGraphBackend,
            "federated": FederatedGraphBackend,
        }
    return _BACKEND_REGISTRY


def get_graph_backend(brain: Any) -> GraphBackend:
    """Return a GraphBackend instance appropriate for *brain*'s configuration.
    Args:
        brain: An ``AxonBrain`` instance.  ``brain.config.graph_backend``
               selects the backend type (default: ``"graphrag"``).
    Returns:
        A :class:`GraphBackend`-conforming instance wrapping *brain*.
    Raises:
        ValueError: If ``brain.config.graph_backend`` names an unknown backend.
    """
    backend_id: str = (
        getattr(getattr(brain, "config", None), "graph_backend", "graphrag") or "graphrag"
    )
    reg = _registry()
    cls = reg.get(backend_id)
    if cls is None:
        valid = sorted(reg)
        raise ValueError(f"Unknown graph_backend '{backend_id}'. Valid options: {valid}")
    return cls(brain)
