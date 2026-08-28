"""Shared atomic JSON persistence helper.

Extracted from ``GraphRagMixin._gr_write_json_if_changed`` so
``CodeGraphMixin._save_code_graph`` can share the same digest-cache-gated
atomic-write behavior instead of its own divergent (always-write)
implementation — the two had drifted into different write-frequency
semantics despite persisting conceptually identical data.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from axon.version_marker import _atomic_replace


def write_json_if_changed(
    path: str | pathlib.Path,
    payload: Any,
    cache: dict[str, str],
    *,
    sort_keys: bool = False,
) -> bool:
    """Atomically write *payload* as JSON to *path*, skipping unchanged content.

    *cache* is a caller-owned ``{path_str: sha1_digest}`` dict used to skip
    re-writing unchanged content across repeated calls without re-reading
    the file from disk each time; it's populated from the on-disk file's
    digest the first time a given path is seen with no cache entry yet.
    Returns ``True`` if the file was written, ``False`` if content was
    unchanged and the write was skipped.

    Uses :func:`axon.version_marker._atomic_replace` so the rename survives
    Windows / OneDrive / cloud-sync transient locks (a crash or sync-client
    race mid-write must never leave the target file truncated or absent).
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=sort_keys, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    p_key = str(p)
    if cache.get(p_key) == digest and p.exists():
        return False
    if p.exists() and cache.get(p_key) is None:
        try:
            existing = p.read_text(encoding="utf-8")
            existing_digest = hashlib.sha1(existing.encode("utf-8", errors="replace")).hexdigest()
            cache[p_key] = existing_digest
            if existing_digest == digest:
                return False
        except Exception:
            pass
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _atomic_replace(tmp, p)
    cache[p_key] = digest
    return True


def write_bytes_if_changed(
    path: str | pathlib.Path,
    payload: bytes,
    cache: dict[str, str],
) -> bool:
    """Atomically write raw *payload* bytes to *path*, skipping unchanged content.

    Same digest-cache / skip-if-unchanged / Windows-safe-replace contract as
    :func:`write_json_if_changed`, for callers whose payload isn't JSON
    (msgpack, YAML text already encoded to bytes, key material, etc.).
    Pass a throwaway ``{}`` for *cache* for one-shot writers that don't
    need cross-call digest reuse.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(payload).hexdigest()
    p_key = str(p)
    if cache.get(p_key) == digest and p.exists():
        return False
    if p.exists() and cache.get(p_key) is None:
        try:
            existing_digest = hashlib.sha1(p.read_bytes()).hexdigest()
            cache[p_key] = existing_digest
            if existing_digest == digest:
                return False
        except Exception:
            pass
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(payload)
    _atomic_replace(tmp, p)
    cache[p_key] = digest
    return True


def write_text_if_changed(
    path: str | pathlib.Path,
    text: str,
    cache: dict[str, str],
    *,
    encoding: str = "utf-8",
) -> bool:
    """Atomically write *text* to *path*, skipping unchanged content.

    Thin wrapper over :func:`write_bytes_if_changed` for plain-text content
    (YAML, ``.env``-style key=value files, newline-joined lists) that isn't
    JSON-serializable via :func:`write_json_if_changed`.
    """
    return write_bytes_if_changed(path, text.encode(encoding, errors="replace"), cache)
