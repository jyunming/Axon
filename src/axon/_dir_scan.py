"""Shared "walk the child directories and read one JSON file each" helper.

Several listers (projects, sub-projects, mount descriptors) used to repeat
the same loop — ``sorted(root.iterdir())`` → skip non-dirs → read
``<child>/<filename>`` → ``json.loads`` → swallow errors — each with a
slightly different set of caught exceptions. :func:`iter_json_children`
is the single implementation.

Errors are caught narrowly as ``(OSError, ValueError)``: ``OSError``
covers unreadable files (permissions, OneDrive/Dropbox placeholders that
fail on read), and ``ValueError`` covers ``json.JSONDecodeError`` and
``UnicodeDecodeError`` — the latter matters because a sealed project's
``meta.json`` is AES-GCM ciphertext, not UTF-8. A file that parses to
something other than a JSON object is treated the same as an unparseable
one, so callers can always call ``.get()`` on the yielded dict.
"""

from __future__ import annotations

import json
from collections.abc import Container, Iterator
from pathlib import Path
from typing import Any, Literal

__all__ = ["iter_json_children"]


def iter_json_children(
    root: Path,
    filename: str,
    *,
    on_error: Literal["skip", "empty"] = "skip",
    exclude: Container[str] = (),
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(child_dir, data)`` for each child of *root* holding *filename*.

    Children are visited in sorted order. A child is skipped when it is
    not a directory, its name is in *exclude*, or it has no *filename*.

    Args:
        root: Directory whose immediate children are scanned. A missing or
            unreadable *root* yields nothing.
        filename: JSON file to read inside each child (e.g. ``"meta.json"``).
        on_error: What to do when the file exists but cannot be read or is
            not a JSON object — ``"skip"`` drops the child, ``"empty"``
            yields it with ``{}`` (used where the directory's existence is
            meaningful even if its metadata is opaque, e.g. sealed projects).
        exclude: Child names to ignore (e.g. reserved top-level names).
    """
    if on_error not in ("skip", "empty"):
        raise ValueError(f"on_error must be 'skip' or 'empty', got {on_error!r}")
    try:
        children = sorted(Path(root).iterdir())
    except OSError:
        return
    for child in children:
        try:
            if not child.is_dir() or child.name in exclude:
                continue
            path = child / filename
            if not path.is_file():
                continue
        except OSError:
            continue
        data: Any
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            if on_error == "skip":
                continue
            data = {}
        yield child, data
