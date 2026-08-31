"""Shared LRU+TTL cache algorithm.

Extracted from two independently-written, structurally-identical
implementations: ``query_router.py``'s ``_query_cache`` (response cache) and
``graph_rag.py``/``graphrag_engine.py``'s ``_traversal_cache`` (BFS
entity-expansion cache). Both hand-rolled the same
OrderedDict + monotonic-time-in-tuple + TTL-check + move_to_end-on-hit +
popitem(last=False)-on-overflow dance.

Deliberately NOT a wrapping class: callers keep their own plain
``OrderedDict``/``threading.Lock`` attributes (several tests construct or
inject into those directly), and these functions just operate on them. The
stored tuple shape is caller-defined and preserved verbatim — these
functions only own the leading ``time.monotonic()`` slot and the
LRU/TTL bookkeeping around it.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


def lru_ttl_get(
    store: OrderedDict[Any, tuple],
    lock: threading.Lock,
    key: Any,
    ttl: float,
) -> tuple | None:
    """Return the cached entry for *key* (the full stored tuple, including
    its leading ``time.monotonic()`` timestamp) if present and not expired,
    else ``None``. ``ttl <= 0`` means entries never expire. Moves the entry
    to the most-recently-used position on a hit; deletes it on a TTL miss.
    """
    with lock:
        cached = store.get(key)
        if cached is None:
            return None
        stored_time = cached[0]
        if ttl > 0 and time.monotonic() - stored_time >= ttl:
            del store[key]
            return None
        store.move_to_end(key)
        return cached


def lru_ttl_put(
    store: OrderedDict[Any, tuple],
    lock: threading.Lock,
    key: Any,
    *value_parts: Any,
    maxsize: int,
) -> None:
    """Store ``(time.monotonic(), *value_parts)`` under *key*, evicting the
    least-recently-used entry first if *store* is already at *maxsize*.

    The at-capacity check only fires for a genuinely new key: overwriting an
    already-present key doesn't grow the store, so evicting first would
    shrink it by one for no reason (a same-key update racing with the
    at-capacity check, or simply refreshing an existing entry while full,
    would otherwise evict an unrelated LRU entry every time).
    """
    with lock:
        if key not in store and len(store) >= maxsize and store:
            store.popitem(last=False)  # evict LRU (front of OrderedDict)
        store[key] = (time.monotonic(), *value_parts)
        store.move_to_end(key)  # mark as most-recently-used
