"""Tests for axon._dir_scan.iter_json_children and its adopters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axon._dir_scan import iter_json_children


def _child(root: Path, name: str, content: bytes | str | None, filename: str = "meta.json"):
    d = root / name
    d.mkdir(parents=True)
    if content is not None:
        p = d / filename
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    return d


class TestIterJsonChildren:
    def test_missing_root_yields_nothing(self, tmp_path):
        assert list(iter_json_children(tmp_path / "nope", "meta.json")) == []

    def test_sorted_and_skips_non_dirs_and_missing_files(self, tmp_path):
        _child(tmp_path, "b", '{"n": 2}')
        _child(tmp_path, "a", '{"n": 1}')
        _child(tmp_path, "c_nofile", None)
        (tmp_path / "file.json").write_text("{}", encoding="utf-8")
        got = [(p.name, d) for p, d in iter_json_children(tmp_path, "meta.json")]
        assert got == [("a", {"n": 1}), ("b", {"n": 2})]

    @pytest.mark.parametrize(
        "content",
        [
            "{ not json",
            "[1, 2]",
            "null",
            b"\x8f\x00\xffciphertext",  # sealed meta.json is AES-GCM ciphertext
        ],
    )
    def test_bad_content_skip_vs_empty(self, tmp_path, content):
        _child(tmp_path, "bad", content)
        _child(tmp_path, "good", '{"ok": true}')
        skipped = [p.name for p, _ in iter_json_children(tmp_path, "meta.json", on_error="skip")]
        assert skipped == ["good"]
        kept = {p.name: d for p, d in iter_json_children(tmp_path, "meta.json", on_error="empty")}
        assert kept == {"bad": {}, "good": {"ok": True}}

    def test_exclude(self, tmp_path):
        _child(tmp_path, "mounts", "{}")
        _child(tmp_path, "proj", "{}")
        got = [p.name for p, _ in iter_json_children(tmp_path, "meta.json", exclude={"mounts"})]
        assert got == ["proj"]

    def test_invalid_on_error_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            list(iter_json_children(tmp_path, "meta.json", on_error="raise"))  # type: ignore[arg-type]


class TestListProjectsAdoption:
    def _root(self, tmp_path, monkeypatch) -> Path:
        import axon.projects as projects

        root = tmp_path / "user"
        root.mkdir()
        monkeypatch.setattr(projects, "PROJECTS_ROOT", root)
        return root

    def test_non_dict_meta_does_not_crash(self, tmp_path, monkeypatch):
        from axon.projects import list_projects

        root = self._root(tmp_path, monkeypatch)
        _child(root, "weird", "[1, 2, 3]")
        _child(root, "ok", json.dumps({"created_at": "2026-01-01", "description": "d"}))
        names = [p["name"] for p in list_projects()]
        assert names == ["ok", "weird"]  # newest-first; unparseable meta sorts last

    def test_sealed_ciphertext_meta_still_listed(self, tmp_path, monkeypatch):
        from axon.projects import list_projects

        root = self._root(tmp_path, monkeypatch)
        _child(root, "sealed", b"\x8f\x00\xff\xfe")
        [entry] = list_projects()
        assert entry["name"] == "sealed"
        assert entry["description"] == ""
        assert entry["maintenance_state"] == "normal"

    def test_reserved_names_and_bad_maintenance_state(self, tmp_path, monkeypatch):
        from axon.projects import list_projects

        root = self._root(tmp_path, monkeypatch)
        _child(root, "mounts", "{}")
        _child(root, "p", json.dumps({"maintenance_state": ["not", "hashable"]}))
        [entry] = list_projects()
        assert entry["name"] == "p"
        assert entry["maintenance_state"] == "normal"

    def test_sub_projects_newest_first_with_bad_meta(self, tmp_path, monkeypatch):
        from axon.projects import list_projects

        root = self._root(tmp_path, monkeypatch)
        parent = _child(root, "parent", json.dumps({"created_at": "2026-01-01"}))
        _child(parent / "subs", "old", json.dumps({"created_at": "2025-01-01"}))
        _child(parent / "subs", "new", json.dumps({"created_at": "2026-06-01"}))
        _child(parent / "subs", "broken", "{ nope")
        [entry] = list_projects()
        assert [c["name"] for c in entry["children"]] == [
            "parent/new",
            "parent/old",
            "parent/broken",
        ]


class TestListMountDescriptorsAdoption:
    def test_skips_corrupt_inactive_and_non_dict(self, tmp_path):
        from axon.mounts import list_mount_descriptors

        root = tmp_path / "mounts"
        _child(root, "b_ok", json.dumps({"mount_name": "b_ok", "state": "active"}), "mount.json")
        _child(root, "a_ok", json.dumps({"mount_name": "a_ok", "state": "active"}), "mount.json")
        _child(root, "corrupt", "{ nope", "mount.json")
        _child(root, "list", "[]", "mount.json")
        _child(
            root,
            "revoked",
            json.dumps({"mount_name": "revoked", "state": "active", "revoked": True}),
            "mount.json",
        )
        _child(root, "inactive", json.dumps({"mount_name": "x", "state": "gone"}), "mount.json")
        assert [d["mount_name"] for d in list_mount_descriptors(tmp_path)] == ["a_ok", "b_ok"]
