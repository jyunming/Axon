"""Unit tests for axon.graph_render.GraphRenderMixin._resolve_graph_payload."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

from axon.graph_render import GraphRenderMixin


class _FakeBrain(GraphRenderMixin):
    def __init__(self, backend):
        self._graph_backend = backend


def test_resolve_graph_payload_returns_backend_dict_payload():
    backend = MagicMock()
    backend.graph_data.return_value = {"nodes": [{"id": "a"}], "links": []}
    brain = _FakeBrain(backend)
    assert brain._resolve_graph_payload() == {"nodes": [{"id": "a"}], "links": []}


def test_resolve_graph_payload_logs_and_returns_empty_on_backend_error(caplog):
    """A real bug in graph payload construction (corrupted community-levels
    cache, a vector-store lookup error, etc.) must leave a diagnostic trail
    instead of silently rendering as an empty graph across every export
    surface (CLI --graph-export, REPL /graph export, REST GET /graph/export)."""
    backend = MagicMock()
    backend.graph_data.side_effect = RuntimeError("corrupted community_levels.json")
    brain = _FakeBrain(backend)
    with caplog.at_level(logging.WARNING, logger="Axon"):
        result = brain._resolve_graph_payload()
    assert result == {"nodes": [], "links": []}
    assert any("graph_data() failed" in r.message for r in caplog.records)


def test_resolve_graph_payload_no_backend_returns_empty():
    brain = _FakeBrain(None)
    assert brain._resolve_graph_payload() == {"nodes": [], "links": []}


def test_resolve_graph_payload_unpacks_to_dict_object():
    payload_obj = MagicMock()
    payload_obj.to_dict.return_value = {"nodes": [], "links": [{"source": "a", "target": "b"}]}
    backend = MagicMock()
    backend.graph_data.return_value = payload_obj
    brain = _FakeBrain(backend)
    assert brain._resolve_graph_payload() == {
        "nodes": [],
        "links": [{"source": "a", "target": "b"}],
    }
