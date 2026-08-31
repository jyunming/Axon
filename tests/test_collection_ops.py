"""Unit tests for axon.collection_ops.clear_active_project.

clear_active_project is a brain-agnostic helper (duck-typed via getattr/
hasattr) shared between the REPL's local /clear path and the REST /clear
route, so it's tested directly against a minimal fake rather than a full
AxonBrain.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from axon.collection_ops import clear_active_project


class _FakeVectorStore:
    provider = "other"
    client = None


class _FakeBrain:
    """The narrowest object clear_active_project actually touches."""

    def __init__(self):
        self.vector_store = _FakeVectorStore()
        self.bm25 = None
        self._ingested_hashes = {"h1"}
        self._doc_versions = {"a.md": {}}
        self._graph_backend = None
        self._code_graph = {"nodes": {"x": 1}, "edges": []}
        self._raptor_summary_cache = {"stale": "value"}
        self._embedding_meta_path = None


def test_clears_ingested_hashes_and_doc_versions():
    brain = _FakeBrain()
    clear_active_project(brain)
    assert brain._ingested_hashes == set()
    assert brain._doc_versions == {}


def test_clears_code_graph():
    brain = _FakeBrain()
    clear_active_project(brain)
    assert brain._code_graph == {"nodes": {}, "edges": []}


def test_raptor_cache_reset_holds_the_cache_lock_when_present():
    """A lock-free reset here can race a background summarization thread
    (main.py's _summarise_window, dispatched via brain._executor) and
    silently drop or reintroduce an entry right after a user-initiated wipe
    -- must use the same lock every other access to this cache uses."""
    brain = _FakeBrain()
    fake_lock = MagicMock()
    brain._raptor_cache_lock = fake_lock
    clear_active_project(brain)
    assert brain._raptor_summary_cache == {}
    fake_lock.__enter__.assert_called_once()
    fake_lock.__exit__.assert_called_once()


def test_raptor_cache_reset_without_a_lock_attribute_still_works():
    """Callers that never set _raptor_cache_lock (RAPTOR disabled entirely)
    must not crash — the lock is optional, looked up via getattr(..., None)."""
    brain = _FakeBrain()
    assert not hasattr(brain, "_raptor_cache_lock")
    clear_active_project(brain)
    assert brain._raptor_summary_cache == {}


def test_no_raptor_cache_attribute_is_a_noop_for_that_step():
    brain = _FakeBrain()
    del brain._raptor_summary_cache
    clear_active_project(brain)
    assert not hasattr(brain, "_raptor_summary_cache")
