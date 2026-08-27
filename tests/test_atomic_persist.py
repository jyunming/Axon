"""Tests for axon._atomic_persist.write_json_if_changed — the shared
digest-cache-gated atomic-write helper extracted from
GraphRagMixin._gr_write_json_if_changed so CodeGraphMixin can share it.
"""
from __future__ import annotations

import json

from axon._atomic_persist import write_json_if_changed


def test_writes_new_file(tmp_path):
    path = tmp_path / "out.json"
    cache: dict[str, str] = {}
    wrote = write_json_if_changed(path, {"a": 1}, cache)
    assert wrote is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert str(path) in cache


def test_skips_write_when_cache_hit_and_unchanged(tmp_path):
    path = tmp_path / "out.json"
    cache: dict[str, str] = {}
    write_json_if_changed(path, {"a": 1}, cache)
    mtime_before = path.stat().st_mtime_ns
    wrote = write_json_if_changed(path, {"a": 1}, cache)
    assert wrote is False
    assert path.stat().st_mtime_ns == mtime_before


def test_writes_when_content_changes(tmp_path):
    path = tmp_path / "out.json"
    cache: dict[str, str] = {}
    write_json_if_changed(path, {"a": 1}, cache)
    wrote = write_json_if_changed(path, {"a": 2}, cache)
    assert wrote is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}


def test_skips_write_when_no_cache_entry_but_disk_content_matches(tmp_path):
    """First call with an empty cache but a file already on disk with the
    same content (e.g. after a fresh process restart) should read the
    on-disk digest and skip the write rather than assuming it must write.
    """
    path = tmp_path / "out.json"
    path.write_text(json.dumps({"a": 1}, sort_keys=False, separators=(",", ":")), encoding="utf-8")
    mtime_before = path.stat().st_mtime_ns
    cache: dict[str, str] = {}
    wrote = write_json_if_changed(path, {"a": 1}, cache)
    assert wrote is False
    assert path.stat().st_mtime_ns == mtime_before
    assert str(path) in cache


def test_writes_when_no_cache_entry_and_disk_content_differs(tmp_path):
    path = tmp_path / "out.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    cache: dict[str, str] = {}
    wrote = write_json_if_changed(path, {"a": 2}, cache)
    assert wrote is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.json"
    cache: dict[str, str] = {}
    wrote = write_json_if_changed(path, {"a": 1}, cache)
    assert wrote is True
    assert path.exists()


def test_unreadable_existing_file_falls_through_to_write(tmp_path):
    """A corrupt/unreadable existing file at the target path must not raise
    — falls through and overwrites rather than crashing the caller.
    """
    path = tmp_path / "out.json"
    path.write_bytes(b"\xff\xfe not valid utf-8 as latin garbage \x00\x01")
    cache: dict[str, str] = {}
    wrote = write_json_if_changed(path, {"a": 1}, cache)
    assert wrote is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_sort_keys_respected(tmp_path):
    path = tmp_path / "out.json"
    cache: dict[str, str] = {}
    write_json_if_changed(path, {"b": 2, "a": 1}, cache, sort_keys=True)
    assert path.read_text(encoding="utf-8") == json.dumps(
        {"b": 2, "a": 1}, sort_keys=True, separators=(",", ":")
    )


def test_independent_caches_do_not_cross_contaminate(tmp_path):
    """Two separate caller-owned cache dicts (e.g. GraphRagMixin's
    _gr_persist_hashes vs. CodeGraphMixin's _code_graph_persist_hashes)
    must not interfere with each other's skip-write decisions.
    """
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    cache_a: dict[str, str] = {}
    cache_b: dict[str, str] = {}
    write_json_if_changed(path_a, {"x": 1}, cache_a)
    write_json_if_changed(path_b, {"x": 1}, cache_b)
    assert str(path_a) in cache_a
    assert str(path_b) not in cache_a
    assert str(path_b) in cache_b
