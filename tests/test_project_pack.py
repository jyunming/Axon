"""Tests for axon.project_pack — pack_project() / unpack_project().

Sealed-project tests skip when cryptography/keyring aren't installed
(the sealed extra hasn't been pulled in), matching test_project_seal.py.
"""
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from axon.project_pack import ProjectPackError, pack_project, unpack_project

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Prevent any test that omits out_path from writing to the real
    ~/.axon/packs/ — pack_project()'s default out_path is derived from
    Path.home(). Individual tests may still override this with their own
    monkeypatch.setattr(Path, "home", ...) to test the default explicitly."""
    fake_home = tmp_path / "_fake_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)


@pytest.fixture
def user_dir(tmp_path):
    ud = tmp_path / "alice"
    ud.mkdir()
    return ud


def _populate_plaintext_project(user_dir: Path, project: str = "research") -> Path:
    """Write a representative project layout, including sessions/ and a
    nested subs/child/ project, to exercise full-fidelity packing."""
    proj = user_dir / project
    (proj / "bm25_index").mkdir(parents=True)
    (proj / "vector_store_data").mkdir(parents=True)
    (proj / "sessions").mkdir(parents=True)

    proj_id = "p1"
    (proj / "meta.json").write_text(
        json.dumps({"project_id": proj_id, "name": project}), encoding="utf-8"
    )
    (proj / "bm25_index" / ".bm25_log.jsonl").write_text('{"id":"d1"}\n', encoding="utf-8")
    (proj / "bm25_index" / ".doc_versions.json").write_text("{}", encoding="utf-8")
    (proj / "vector_store_data" / "manifest.json").write_text('{"d":768}', encoding="utf-8")
    (proj / "vector_store_data" / "seg-00000001.bin").write_bytes(b"\x01" * 256)
    (proj / "sessions" / "session1.json").write_text('{"turns": []}', encoding="utf-8")

    child = proj / "subs" / "child"
    (child / "bm25_index").mkdir(parents=True)
    (child / "vector_store_data").mkdir(parents=True)
    (child / "meta.json").write_text(json.dumps({"name": "child"}), encoding="utf-8")
    return proj


# ---------------------------------------------------------------------------
# Pack contents
# ---------------------------------------------------------------------------


class TestPackContents:
    def test_manifest_present_and_correct(self, user_dir):
        _populate_plaintext_project(user_dir)
        result = pack_project("research", user_dir)
        assert result["status"] == "packed"
        assert Path(result["out_path"]).exists()
        with zipfile.ZipFile(result["out_path"]) as zf:
            manifest = json.loads(zf.read("_axon_pack_manifest.json"))
        assert manifest["pack_format_version"] == 1
        assert manifest["original_project"] == "research"
        assert manifest["sealed"] is False
        assert "axon_rag_version" in manifest
        assert "packed_at" in manifest

    def test_all_files_present_with_forward_slash_arcnames(self, user_dir):
        _populate_plaintext_project(user_dir)
        result = pack_project("research", user_dir)
        with zipfile.ZipFile(result["out_path"]) as zf:
            names = set(zf.namelist())
        assert "meta.json" in names
        assert "bm25_index/.bm25_log.jsonl" in names
        assert "bm25_index/.doc_versions.json" in names
        assert "vector_store_data/manifest.json" in names
        assert "vector_store_data/seg-00000001.bin" in names
        assert "sessions/session1.json" in names
        assert "subs/child/meta.json" in names
        assert not any("\\" in n for n in names)

    def test_missing_project_raises(self, user_dir):
        user_dir.mkdir(exist_ok=True)
        with pytest.raises(ProjectPackError, match="does not exist"):
            pack_project("does-not-exist", user_dir)

    def test_default_out_path_under_home_axon_packs(self, user_dir, tmp_path, monkeypatch):
        _populate_plaintext_project(user_dir)
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        result = pack_project("research", user_dir)
        assert str(fake_home / ".axon" / "packs") in result["out_path"]

    def test_explicit_out_path_used_verbatim(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        out = tmp_path / "custom" / "myarchive.zip"
        result = pack_project("research", user_dir, out_path=out)
        assert result["out_path"] == str(out)
        assert out.exists()


# ---------------------------------------------------------------------------
# External .dynamic_graph.db relocation handling
# ---------------------------------------------------------------------------


class TestExternalDynamicGraphDb:
    @pytest.fixture(autouse=True)
    def _fake_graphs_root(self, tmp_path, monkeypatch):
        root = tmp_path / "fake_graphs_root"
        monkeypatch.setattr("axon.project_pack._graphs_root", lambda: root)
        return root

    def test_external_only_is_packed(self, user_dir, _fake_graphs_root):
        proj = _populate_plaintext_project(user_dir, "graphproj")
        proj_id = json.loads((proj / "meta.json").read_text())["project_id"]
        ext_dir = _fake_graphs_root / proj_id
        ext_dir.mkdir(parents=True)
        (ext_dir / ".dynamic_graph.db").write_bytes(b"EXTERNAL_DB_BYTES")

        result = pack_project("graphproj", user_dir)
        with zipfile.ZipFile(result["out_path"]) as zf:
            assert zf.read("bm25_index/.dynamic_graph.db") == b"EXTERNAL_DB_BYTES"

    def test_external_wins_over_stale_local(self, user_dir, _fake_graphs_root):
        proj = _populate_plaintext_project(user_dir, "graphproj")
        proj_id = json.loads((proj / "meta.json").read_text())["project_id"]
        (proj / "bm25_index" / ".dynamic_graph.db").write_bytes(b"STALE_LOCAL_BYTES")
        ext_dir = _fake_graphs_root / proj_id
        ext_dir.mkdir(parents=True)
        (ext_dir / ".dynamic_graph.db").write_bytes(b"FRESH_EXTERNAL_BYTES")

        result = pack_project("graphproj", user_dir)
        with zipfile.ZipFile(result["out_path"]) as zf:
            names = zf.namelist()
            assert names.count("bm25_index/.dynamic_graph.db") == 1
            assert zf.read("bm25_index/.dynamic_graph.db") == b"FRESH_EXTERNAL_BYTES"

    def test_sealed_project_found_via_sha256_fallback(self, user_dir, _fake_graphs_root):
        proj = user_dir / "sealedgraph"
        (proj / "bm25_index").mkdir(parents=True)
        (proj / "vector_store_data").mkdir(parents=True)
        # meta.json unreadable as JSON, simulating sealed ciphertext.
        (proj / "meta.json").write_bytes(b"AXSL_NOT_VALID_JSON")
        resolved = str((proj / "bm25_index").resolve())
        fallback_id = hashlib.sha256(resolved.encode()).hexdigest()[:16]
        ext_dir = _fake_graphs_root / fallback_id
        ext_dir.mkdir(parents=True)
        (ext_dir / ".dynamic_graph.db").write_bytes(b"SEALED_FALLBACK_BYTES")

        result = pack_project("sealedgraph", user_dir)
        with zipfile.ZipFile(result["out_path"]) as zf:
            assert zf.read("bm25_index/.dynamic_graph.db") == b"SEALED_FALLBACK_BYTES"

    def test_no_external_db_uses_local_copy_normally(self, user_dir):
        proj = _populate_plaintext_project(user_dir, "localonly")
        (proj / "bm25_index" / ".dynamic_graph.db").write_bytes(b"LOCAL_ONLY_BYTES")
        result = pack_project("localonly", user_dir)
        with zipfile.ZipFile(result["out_path"]) as zf:
            assert zf.read("bm25_index/.dynamic_graph.db") == b"LOCAL_ONLY_BYTES"


# ---------------------------------------------------------------------------
# Zip-slip rejection
# ---------------------------------------------------------------------------


class TestZipSlip:
    def _write_zip(self, path: Path, entries: dict, manifest: dict | None = None):
        with zipfile.ZipFile(path, "w") as zf:
            if manifest is not None:
                zf.writestr("_axon_pack_manifest.json", json.dumps(manifest))
            for name, payload in entries.items():
                zf.writestr(name, payload)
        return path

    def test_path_traversal_rejected(self, user_dir, tmp_path):
        evil = self._write_zip(
            tmp_path / "evil.zip",
            {"../../evil.txt": "pwned"},
            manifest={"original_project": "traversal"},
        )
        with pytest.raises(ProjectPackError, match="Unsafe path"):
            unpack_project(evil, user_dir)
        assert not (user_dir / "traversal").exists()

    def test_absolute_path_rejected(self, user_dir, tmp_path):
        import os

        absname = os.path.abspath(os.sep + "evil.txt")
        evil = self._write_zip(
            tmp_path / "evil2.zip",
            {absname: "pwned"},
            manifest={"original_project": "absolute"},
        )
        with pytest.raises(ProjectPackError):
            unpack_project(evil, user_dir)
        assert not (user_dir / "absolute").exists()

    def test_symlink_entry_rejected(self, user_dir, tmp_path):
        evil = tmp_path / "evil3.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("_axon_pack_manifest.json", json.dumps({"original_project": "symlink"}))
            info = zipfile.ZipInfo("legit.txt")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
        with pytest.raises(ProjectPackError, match="symlink"):
            unpack_project(evil, user_dir)
        assert not (user_dir / "symlink").exists()

    def test_no_leftover_staging_dir_after_rejection(self, user_dir, tmp_path):
        evil = self._write_zip(
            tmp_path / "evil4.zip",
            {"../escape.txt": "pwned"},
            manifest={"original_project": "cleanup"},
        )
        with pytest.raises(ProjectPackError):
            unpack_project(evil, user_dir)
        assert list(user_dir.glob(".*unpacking*")) == []


# ---------------------------------------------------------------------------
# Unpack collision / force
# ---------------------------------------------------------------------------


class TestUnpackCollision:
    def test_fresh_unpack_succeeds(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        packed = pack_project("research", user_dir)
        target = tmp_path / "bob"
        target.mkdir()
        result = unpack_project(packed["out_path"], target)
        assert result["status"] == "unpacked"
        assert (target / "research" / "meta.json").exists()

    def test_collision_without_force_raises_and_preserves_original(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        packed = pack_project("research", user_dir)
        target = tmp_path / "bob"
        target.mkdir()
        unpack_project(packed["out_path"], target)
        marker = target / "research" / "meta.json"
        original_bytes = marker.read_bytes()
        with pytest.raises(ProjectPackError, match="already exists"):
            unpack_project(packed["out_path"], target)
        assert marker.read_bytes() == original_bytes

    def test_force_replaces_fully_not_merge(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        packed = pack_project("research", user_dir)
        target = tmp_path / "bob"
        target.mkdir()
        unpack_project(packed["out_path"], target)
        stray = target / "research" / "stray_file.txt"
        stray.write_text("should be gone after force replace")
        unpack_project(packed["out_path"], target, force=True)
        assert not stray.exists()
        assert (target / "research" / "meta.json").exists()


# ---------------------------------------------------------------------------
# Unpack naming
# ---------------------------------------------------------------------------


class TestUnpackNaming:
    def test_no_manifest_no_as_name_raises(self, user_dir, tmp_path):
        zip_path = tmp_path / "nomanifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("meta.json", "{}")
        with pytest.raises(ProjectPackError, match="No target project name"):
            unpack_project(zip_path, user_dir)

    def test_as_name_overrides_manifest(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        packed = pack_project("research", user_dir)
        target = tmp_path / "bob"
        target.mkdir()
        result = unpack_project(packed["out_path"], target, as_name="renamed")
        assert result["project"] == "renamed"
        assert (target / "renamed" / "meta.json").exists()

    def test_hostile_manifest_name_rejected_before_any_write(self, user_dir, tmp_path):
        evil = tmp_path / "hostile.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("_axon_pack_manifest.json", json.dumps({"original_project": "../escape"}))
            zf.writestr("meta.json", "{}")
        with pytest.raises(ProjectPackError):
            unpack_project(evil, user_dir)
        assert list(user_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Nested --as parent/child targets
# ---------------------------------------------------------------------------


class TestNestedAsName:
    def test_ancestor_skeleton_created_anchored_at_user_dir(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        packed = pack_project("research", user_dir)
        target = tmp_path / "bob"
        target.mkdir()
        result = unpack_project(packed["out_path"], target, as_name="newparent/newchild")
        assert result["project"] == "newparent/newchild"
        assert (target / "newparent" / "meta.json").exists()
        assert (target / "newparent" / "subs" / "newchild" / "meta.json").exists()
        # Ancestor got the standard skeleton dirs too.
        assert (target / "newparent" / "bm25_index").is_dir()
        assert (target / "newparent" / "vector_store_data").is_dir()
        assert (target / "newparent" / "sessions").is_dir()


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestNameValidation:
    def test_pack_invalid_name_raises(self, user_dir):
        with pytest.raises(ProjectPackError):
            pack_project("../escape", user_dir)

    def test_unpack_invalid_as_name_raises(self, user_dir, tmp_path):
        _populate_plaintext_project(user_dir)
        packed = pack_project("research", user_dir)
        target = tmp_path / "bob"
        target.mkdir()
        with pytest.raises(ProjectPackError):
            unpack_project(packed["out_path"], target, as_name="../escape")


# ---------------------------------------------------------------------------
# Sealed round-trip (real crypto)
# ---------------------------------------------------------------------------


class _InMemoryKeyring:
    priority = 1

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, secret):
        self._store[(service, username)] = secret

    def get_password(self, service, username):
        return self._store.get((service, username))

    def delete_password(self, service, username):
        import keyring.errors

        if (service, username) not in self._store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


@pytest.fixture
def kr_backend():
    pytest.importorskip("cryptography")
    pytest.importorskip("keyring")
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


class TestSealedRoundTrip:
    def test_sealed_project_still_sealed_and_unlockable_after_restore(
        self, kr_backend, user_dir, tmp_path
    ):
        import shutil

        from axon.security.master import bootstrap_store, unlock_store
        from axon.security.seal import project_seal

        bootstrap_store(user_dir, "test-pass-ok")
        _populate_plaintext_project(user_dir)
        seal_result = project_seal("research", user_dir)
        assert seal_result["status"] == "sealed"

        packed = pack_project("research", user_dir)
        assert packed["sealed"] is True

        shutil.rmtree(user_dir / "research")
        restored = unpack_project(packed["out_path"], user_dir)
        assert restored["sealed"] is True
        assert (user_dir / "research" / ".security" / ".sealed").is_file()
        meta_bytes = (user_dir / "research" / "meta.json").read_bytes()
        assert meta_bytes.startswith(b"AXSL")

        # Original passphrase still unlocks the restored, still-sealed project —
        # unlock_store() raises on a wrong/broken passphrase rather than
        # returning a falsy result, so success alone is the assertion.
        result = unlock_store(user_dir, "test-pass-ok")
        assert result["unlocked"] is True
