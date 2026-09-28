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
import os
import pathlib
import re
import secrets
from typing import Any

from axon.version_marker import _atomic_replace


def _tmp_path(p: pathlib.Path) -> pathlib.Path:
    """Per-writer temp name next to *p*.

    A fixed ``<name>.tmp`` is shared by every process writing *p*: two
    writers (e.g. ``axon-api`` and a one-off ``axon`` CLI both touching a
    project's ``meta.json``) would clobber each other's temp file, and the
    loser's rename would fail on a temp that is already gone. The pid plus a
    random suffix keeps each writer's temp private; the ``.tmp`` suffix is
    kept so directory scans for ``*.json`` never pick it up.
    """
    return p.with_name(f"{p.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


_TMP_NAME_RE = re.compile(r"\.\d+\.[0-9a-f]{8}\.tmp$")


def is_atomic_tmp(path: str | pathlib.Path) -> bool:
    """True if *path* is a temp file left by this module's writers.

    Normally they never outlive the write, but a process killed between
    writing the temp file and renaming it leaves one behind. Code that
    copies a whole directory (e.g. ``project_pack``) should skip them.
    """
    return bool(_TMP_NAME_RE.search(pathlib.Path(path).name))


def _write_tmp_then_replace(p: pathlib.Path, payload: bytes) -> None:
    """Write *payload* to a private temp file, then rename it over *p*.

    A failed temp write (disk full, I/O error) removes its partial temp file
    instead of leaving an orphan; a failed rename is cleaned up by
    ``_atomic_replace`` itself.
    """
    tmp = _tmp_path(p)
    try:
        tmp.write_bytes(payload)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _atomic_replace(tmp, p)


def write_json_if_changed(
    path: str | pathlib.Path,
    payload: Any,
    cache: dict[str, str],
    *,
    sort_keys: bool = False,
    indent: int | None = None,
    ensure_ascii: bool = True,
) -> bool:
    """Atomically write *payload* as JSON to *path*, skipping unchanged content.

    *cache* is a caller-owned ``{path_str: sha1_digest}`` dict used to skip
    re-writing unchanged content across repeated calls without re-reading
    the file from disk each time; it's populated from the on-disk file's
    digest the first time a given path is seen with no cache entry yet.
    Returns ``True`` if the file was written, ``False`` if content was
    unchanged and the write was skipped.

    Output is compact by default. Pass ``indent`` (and ``ensure_ascii``) for
    human-readable files, which then serialize exactly like
    ``json.dumps(payload, indent=..., ensure_ascii=...)``.

    Uses :func:`axon.version_marker._atomic_replace` so the rename survives
    Windows / OneDrive / cloud-sync transient locks (a crash or sync-client
    race mid-write must never leave the target file truncated or absent).
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        sort_keys=sort_keys,
        indent=indent,
        ensure_ascii=ensure_ascii,
        separators=None if indent is not None else (",", ":"),
    )
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
    _write_tmp_then_replace(p, text.encode("utf-8"))
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
    _write_tmp_then_replace(p, payload)
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
