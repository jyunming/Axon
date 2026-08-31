"""Tests for axon._lru_ttl_cache — the shared LRU+TTL algorithm extracted
from query_router.py's _query_cache and graph_rag.py/graphrag_engine.py's
_traversal_cache (independently duplicated OrderedDict + monotonic-time +
TTL + move_to_end + popitem(last=False) implementations).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from axon._lru_ttl_cache import lru_ttl_get, lru_ttl_put


def _store():
    return OrderedDict(), threading.Lock()


def test_put_then_get_returns_stored_value():
    store, lock = _store()
    lru_ttl_put(store, lock, "k1", "v1", maxsize=10)
    cached = lru_ttl_get(store, lock, "k1", ttl=60)
    assert cached is not None
    assert cached[1] == "v1"


def test_get_miss_returns_none():
    store, lock = _store()
    assert lru_ttl_get(store, lock, "missing", ttl=60) is None


def test_get_expired_entry_returns_none_and_removes_it():
    store, lock = _store()
    store["k1"] = (time.monotonic() - 100, "stale")
    assert lru_ttl_get(store, lock, "k1", ttl=1) is None
    assert "k1" not in store


def test_ttl_le_zero_never_expires():
    store, lock = _store()
    store["k1"] = (time.monotonic() - 10_000, "ancient")
    cached = lru_ttl_get(store, lock, "k1", ttl=0)
    assert cached is not None
    assert cached[1] == "ancient"


def test_get_hit_moves_entry_to_end():
    store, lock = _store()
    lru_ttl_put(store, lock, "k1", "v1", maxsize=10)
    lru_ttl_put(store, lock, "k2", "v2", maxsize=10)
    assert list(store.keys()) == ["k1", "k2"]
    lru_ttl_get(store, lock, "k1", ttl=60)
    assert list(store.keys()) == ["k2", "k1"]


def test_put_evicts_lru_when_at_capacity():
    store, lock = _store()
    lru_ttl_put(store, lock, "k1", "v1", maxsize=2)
    lru_ttl_put(store, lock, "k2", "v2", maxsize=2)
    lru_ttl_put(store, lock, "k3", "v3", maxsize=2)
    assert "k1" not in store
    assert set(store.keys()) == {"k2", "k3"}


def test_put_updating_existing_key_at_capacity_does_not_evict_anything():
    """Overwriting an already-present key doesn't grow the store, so the
    at-capacity check must not fire for it -- otherwise refreshing an
    existing entry while full spuriously evicts an unrelated LRU entry,
    shrinking effective capacity by one every time."""
    store, lock = _store()
    lru_ttl_put(store, lock, "k1", "v1", maxsize=2)
    lru_ttl_put(store, lock, "k2", "v2", maxsize=2)
    lru_ttl_put(store, lock, "k1", "v1-updated", maxsize=2)  # update, not a new key
    assert set(store.keys()) == {"k1", "k2"}
    cached = lru_ttl_get(store, lock, "k1", ttl=60)
    assert cached[1] == "v1-updated"


def test_put_preserves_multi_value_tuple_shape():
    store, lock = _store()
    lru_ttl_put(store, lock, "k1", "response", {"a": 1}, {"b": 2}, maxsize=10)
    _stored_time, response, citations, provenance = store["k1"]
    assert response == "response"
    assert citations == {"a": 1}
    assert provenance == {"b": 2}


def test_get_supports_variable_length_stored_tuples():
    """Regression: query_router.py's read path destructures
    `stored_time, stored_response, *rest = cached` to stay backwards
    compatible with shorter externally-injected tuples (tests). The shared
    helper must return the raw tuple unchanged, not repackage it."""
    store, lock = _store()
    store["k1"] = (time.monotonic(), "short answer")
    cached = lru_ttl_get(store, lock, "k1", ttl=60)
    _stored_time, stored_response, *rest = cached
    assert stored_response == "short answer"
    assert rest == []


def test_independent_stores_do_not_interfere():
    store_a, lock_a = _store()
    store_b, lock_b = _store()
    lru_ttl_put(store_a, lock_a, "k1", "a-value", maxsize=10)
    lru_ttl_put(store_b, lock_b, "k1", "b-value", maxsize=10)
    assert lru_ttl_get(store_a, lock_a, "k1", ttl=60)[1] == "a-value"
    assert lru_ttl_get(store_b, lock_b, "k1", ttl=60)[1] == "b-value"
