"""Project/share/mount/session metadata is written atomically.

meta.json, store_meta.json, share manifests, mount descriptors and session
files used to be written with a bare ``write_text()`` / ``open("w")`` straight
over the live file, so a crash (or a cloud-sync lock) mid-write left a
truncated, unparsable file. They now go through ``axon._atomic_persist``:
write a private temp file, then rename it over the target.

Each module test simulates a failure at the rename step: the original file
must survive intact. With the old direct writes the rename is never reached,
the target is overwritten, and these tests fail.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from axon._atomic_persist import write_json_if_changed, write_text_if_changed

_REPLACE = "axon._atomic_persist._atomic_replace"


def _fail_replace(src, dst):
    Path(src).unlink(missing_ok=True)
    raise OSError("simulated crash before rename")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class TestHelper:
    def test_indent_matches_json_dumps(self, tmp_path):
        path = tmp_path / "meta.json"
        data = {"name": "p", "nested": {"a": [1, 2]}}
        write_json_if_changed(path, data, {}, indent=2)
        assert path.read_text(encoding="utf-8") == json.dumps(data, indent=2)

    def test_default_output_stays_compact(self, tmp_path):
        path = tmp_path / "graph.json"
        write_json_if_changed(path, {"a": 1, "b": [1, 2]}, {})
        assert path.read_text(encoding="utf-8") == '{"a":1,"b":[1,2]}'

    def test_ensure_ascii_false_keeps_non_ascii(self, tmp_path):
        path = tmp_path / "s.json"
        write_json_if_changed(path, {"q": "文件"}, {}, indent=2, ensure_ascii=False)
        text = path.read_text(encoding="utf-8")
        assert "文件" in text and "\\u" not in text

    def test_each_write_uses_its_own_temp_file(self, tmp_path):
        """Two writers must not share a temp name — the loser's rename would
        fail on a temp file the winner already moved away."""
        path = tmp_path / "meta.json"
        seen: list[Path] = []

        def _record(src, dst):
            seen.append(Path(src))
            os.replace(src, dst)

        with patch(_REPLACE, side_effect=_record):
            write_json_if_changed(path, {"v": 1}, {}, indent=2)
            write_json_if_changed(path, {"v": 2}, {}, indent=2)
        assert len(seen) == 2 and seen[0] != seen[1]
        for tmp in seen:
            assert tmp.parent == tmp_path
            assert tmp.name.startswith("meta.json.") and tmp.suffix == ".tmp"
        assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}

    def test_failed_rename_leaves_original_and_no_temp(self, tmp_path):
        path = tmp_path / "meta.json"
        write_json_if_changed(path, {"v": 1}, {}, indent=2)
        with patch(_REPLACE, side_effect=_fail_replace):
            with pytest.raises(OSError):
                write_json_if_changed(path, {"v": 2}, {}, indent=2)
        assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------


class TestProjectsMeta:
    def test_maintenance_state_write_is_atomic(self, tmp_path):
        from axon.projects import set_maintenance_state

        meta = tmp_path / "proj" / "meta.json"
        meta.parent.mkdir()
        original = {"name": "proj", "maintenance_state": "normal"}
        meta.write_text(json.dumps(original, indent=2), encoding="utf-8")

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            with patch(_REPLACE, side_effect=_fail_replace):
                with pytest.raises(OSError):
                    set_maintenance_state("proj", "readonly")
            assert json.loads(meta.read_text(encoding="utf-8")) == original

            set_maintenance_state("proj", "readonly")
        assert json.loads(meta.read_text(encoding="utf-8"))["maintenance_state"] == "readonly"

    def test_new_project_meta_is_human_readable(self, tmp_path):
        from axon.projects import _ensure_single_project_at

        _ensure_single_project_at(tmp_path / "p", "p", "desc")
        text = (tmp_path / "p" / "meta.json").read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2)

    def test_set_active_project_failure_is_non_fatal(self, tmp_path):
        from axon.projects import set_active_project

        active = tmp_path / ".active_project"
        write_text_if_changed(active, "old", {})
        with patch("axon.projects._ACTIVE_FILE", active):
            with patch(_REPLACE, side_effect=_fail_replace):
                set_active_project("new")  # swallows OSError, as before
            assert active.read_text(encoding="utf-8") == "old"
            set_active_project("new")
            assert active.read_text(encoding="utf-8") == "new"


class TestSessions:
    def _session(self, text: str) -> dict:
        return {
            "id": "20260928T120000000",
            "started_at": "2026-09-28T12:00:00Z",
            "provider": "ollama",
            "model": "llama3",
            "project": "default",
            "history": [{"role": "user", "content": text}],
        }

    def test_failed_save_keeps_previous_session(self, tmp_path, monkeypatch):
        import axon.sessions as s

        monkeypatch.setattr(s, "_SESSIONS_DIR", str(tmp_path))
        s._save_session(self._session("first"))
        with patch(_REPLACE, side_effect=_fail_replace):
            s._save_session(self._session("second"))  # errors are swallowed
        loaded = s._load_session("20260928T120000000")
        assert loaded is not None
        assert loaded["history"][0]["content"] == "first"

    def test_non_ascii_history_is_stored_readably(self, tmp_path, monkeypatch):
        import axon.sessions as s

        monkeypatch.setattr(s, "_SESSIONS_DIR", str(tmp_path))
        s._save_session(self._session("刪除後重新匯入"))
        raw = (tmp_path / "session_20260928T120000000.json").read_text(encoding="utf-8")
        assert "刪除後重新匯入" in raw
        assert s._load_session("20260928T120000000")["history"][0]["content"] == "刪除後重新匯入"


class TestMountDescriptor:
    def test_rewrite_is_atomic(self, tmp_path):
        from axon.mounts import create_mount_descriptor, mount_descriptor_path

        grantee, owner = tmp_path / "bob", tmp_path / "alice"
        target = owner / "research"
        target.mkdir(parents=True)
        grantee.mkdir()
        kwargs = {
            "grantee_user_dir": grantee,
            "mount_name": "alice_research",
            "owner": "alice",
            "project": "research",
            "owner_user_dir": owner,
            "target_project_dir": target,
            "share_key_id": "sk_1",
        }
        create_mount_descriptor(**kwargs)
        path = mount_descriptor_path(grantee, "alice_research")
        before = path.read_text(encoding="utf-8")

        with patch(_REPLACE, side_effect=_fail_replace):
            with pytest.raises(OSError):
                create_mount_descriptor(**{**kwargs, "share_key_id": "sk_2"})
        assert path.read_text(encoding="utf-8") == before
        assert json.loads(before)["share_key_id"] == "sk_1"


class TestSharesJson:
    def test_write_is_atomic(self, tmp_path):
        from axon.shares import _write_json

        path = tmp_path / ".share_manifest.json"
        _write_json(path, {"sharing": [1]})
        with patch(_REPLACE, side_effect=_fail_replace):
            with pytest.raises(OSError):
                _write_json(path, {"sharing": [1, 2]})
        assert json.loads(path.read_text(encoding="utf-8")) == {"sharing": [1]}

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_keys_file_stays_owner_only(self, tmp_path):
        from axon.shares import _write_json

        path = tmp_path / ".share_keys.json"
        _write_json(path, {"k": "v"})
        _write_json(path, {"k": "v2"})
        assert (path.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------------------
# Leftover temp files (review follow-ups)
# ---------------------------------------------------------------------------


class TestTempFileHygiene:
    def test_failed_temp_write_leaves_no_orphan(self, tmp_path):
        """Each write mints a fresh temp name, so a temp left by a failed
        write would never be overwritten — it must be removed on failure."""
        path = tmp_path / "meta.json"
        write_json_if_changed(path, {"v": 1}, {}, indent=2)
        real_write_bytes = Path.write_bytes

        def _disk_full(self, data):
            real_write_bytes(self, data[:3])  # partial write, then fail
            raise OSError(28, "No space left on device")

        with patch.object(Path, "write_bytes", _disk_full):
            with pytest.raises(OSError):
                write_json_if_changed(path, {"v": 2}, {}, indent=2)
        assert not list(tmp_path.glob("*.tmp"))
        assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}

    def test_is_atomic_tmp_matches_only_helper_temp_names(self, tmp_path):
        from axon._atomic_persist import _tmp_path, is_atomic_tmp

        assert is_atomic_tmp(_tmp_path(tmp_path / "meta.json"))
        for name in ("meta.json", "notes.tmp", "meta.json.tmp", "a.123.tmp", "x.1.ZZZZZZZZ.tmp"):
            assert not is_atomic_tmp(tmp_path / name), name

    def test_pack_skips_leftover_temp_files(self, tmp_path, monkeypatch):
        import zipfile

        from axon._atomic_persist import _tmp_path
        from axon.project_pack import pack_project

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "_home")
        user_dir = tmp_path / "alice"
        proj = user_dir / "research"
        proj.mkdir(parents=True)
        write_json_if_changed(proj / "meta.json", {"name": "research"}, {}, indent=2)
        orphan = _tmp_path(proj / "meta.json")
        orphan.write_text("{partial", encoding="utf-8")
        (proj / "notes.tmp").write_text("user file", encoding="utf-8")

        result = pack_project("research", user_dir, out_path=tmp_path / "p.zip")
        with zipfile.ZipFile(result["out_path"]) as zf:
            names = set(zf.namelist())
        assert "meta.json" in names
        assert "notes.tmp" in names  # only the helper's own temp names are skipped
        assert orphan.name not in names

    def test_unpack_ancestor_skeleton_meta_is_readable(self, tmp_path):
        from axon.project_pack import _ensure_ancestor_skeleton

        _ensure_ancestor_skeleton(tmp_path / "parent")
        text = (tmp_path / "parent" / "meta.json").read_text(encoding="utf-8")
        assert json.loads(text)["name"] == "parent"
        assert text == json.dumps(json.loads(text), indent=2)
