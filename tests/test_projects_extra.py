"""Extra tests for axon.projects to push coverage above 90%."""
import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# _resolve_projects_root (line 49)
# ---------------------------------------------------------------------------


class TestResolveProjectsRoot:
    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        """AXON_PROJECTS_ROOT env var is honored (line 49)."""
        monkeypatch.setenv("AXON_PROJECTS_ROOT", str(tmp_path))

        import axon.projects as proj_mod

        result = proj_mod._resolve_projects_root()
        assert result == tmp_path


# ---------------------------------------------------------------------------
# project_sessions_path (line 211)
# ---------------------------------------------------------------------------


class TestProjectSessionsPath:
    def test_returns_sessions_subpath(self, tmp_path):
        from axon.projects import project_sessions_path

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            path = project_sessions_path("myproject")
        assert path.endswith("sessions")
        assert "myproject" in path


# ---------------------------------------------------------------------------
# get_project_namespace_id (lines 283-284)
# ---------------------------------------------------------------------------


class TestGetProjectNamespaceId:
    def test_exception_in_read_returns_none(self, tmp_path):
        """json.JSONDecodeError or OSError in meta.json returns None (lines 283-284)."""
        from axon.projects import get_project_namespace_id

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            # Create a project dir with malformed meta.json
            proj_dir = tmp_path / "testproj"
            proj_dir.mkdir()
            (proj_dir / "meta.json").write_text("NOT_VALID_JSON")
            result = get_project_namespace_id("testproj")
        assert result is None


# ---------------------------------------------------------------------------
# get_store_namespace_id (lines 294, 299-300)
# ---------------------------------------------------------------------------


class TestGetStoreNamespaceId:
    def test_returns_none_if_no_file(self, tmp_path):
        """Returns None when store_meta.json does not exist (line 294)."""
        from axon.projects import get_store_namespace_id

        result = get_store_namespace_id(tmp_path)
        assert result is None

    def test_returns_none_on_json_error(self, tmp_path):
        """Returns None when store_meta.json has malformed JSON (lines 299-300)."""
        from axon.projects import get_store_namespace_id

        (tmp_path / "store_meta.json").write_text("INVALID_JSON_DATA")
        result = get_store_namespace_id(tmp_path)
        assert result is None

    def test_returns_id_when_present(self, tmp_path):
        """Returns the store_namespace_id when valid."""
        from axon.projects import get_store_namespace_id

        (tmp_path / "store_meta.json").write_text(
            json.dumps({"store_namespace_id": "store_abc123"})
        )
        result = get_store_namespace_id(tmp_path)
        assert result == "store_abc123"


# ---------------------------------------------------------------------------
# list_descendants (lines 322, 333, 335, 362, 365, 368-369)
# ---------------------------------------------------------------------------


class TestListDescendants:
    def test_cycle_detection_stops_recursion(self, tmp_path):
        """Cycle detection: already-visited resolved path returns [] (line 322)."""
        from axon.projects import list_descendants

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            proj_dir = tmp_path / "parent"
            proj_dir.mkdir()
            (proj_dir / "meta.json").write_text(json.dumps({"name": "parent"}))
            # Pre-populate visited with this project's resolved path
            visited = {str(proj_dir.resolve())}
            result = list_descendants("parent", visited=visited)
        assert result == []

    def test_no_subs_dir_returns_empty(self, tmp_path):
        """Project with no subs/ directory returns [] (lines 326-327)."""
        from axon.projects import list_descendants

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            proj_dir = tmp_path / "myproj"
            proj_dir.mkdir()
            (proj_dir / "meta.json").write_text(json.dumps({"name": "myproj"}))
            result = list_descendants("myproj")
        assert result == []

    def test_symlinks_in_subs_are_skipped(self, tmp_path):
        """Symlinks in subs/ are skipped (line 333)."""
        from axon.projects import list_descendants

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            proj_dir = tmp_path / "parent"
            subs_dir = proj_dir / "subs"
            subs_dir.mkdir(parents=True)
            (proj_dir / "meta.json").write_text(json.dumps({"name": "parent"}))

            # Create a regular subproject
            real_sub = subs_dir / "real_sub"
            real_sub.mkdir()
            (real_sub / "meta.json").write_text(json.dumps({"name": "parent/real_sub"}))

            # Create a symlink (skip on Windows if not supported)
            try:
                link = subs_dir / "linked_sub"
                link.symlink_to(real_sub)
            except (OSError, NotImplementedError):
                pass  # skip symlink on systems without support

            result = list_descendants("parent")
        # real_sub should be found; symlink should be skipped
        assert "parent/real_sub" in result

    def test_non_dir_entry_in_subs_skipped(self, tmp_path):
        """Non-directory entries in subs/ are skipped (line 335)."""
        from axon.projects import list_descendants

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            proj_dir = tmp_path / "parent"
            subs_dir = proj_dir / "subs"
            subs_dir.mkdir(parents=True)
            (proj_dir / "meta.json").write_text(json.dumps({"name": "parent"}))
            # Create a file (not a dir) in subs/
            (subs_dir / "not_a_dir.txt").write_text("file")
            # Create a dir without meta.json
            (subs_dir / "no_meta").mkdir()
            # Create a valid subproject
            real_sub = subs_dir / "valid_sub"
            real_sub.mkdir()
            (real_sub / "meta.json").write_text(json.dumps({"name": "parent/valid_sub"}))

            result = list_descendants("parent")
        assert result == ["parent/valid_sub"]


# ---------------------------------------------------------------------------
# _list_sub_projects (lines 362, 365, 368-369)
# ---------------------------------------------------------------------------


class TestListSubProjects:
    def test_non_dir_entries_skipped(self, tmp_path):
        """Non-directory entries in subs/ are skipped (line 362)."""
        from axon.projects import _list_sub_projects

        subs_dir = tmp_path / "subs"
        subs_dir.mkdir()
        # Create a file entry (not dir)
        (subs_dir / "file_entry.txt").write_text("not a dir")
        # Create valid subproject
        sub = subs_dir / "valid"
        sub.mkdir()
        (sub / "meta.json").write_text(json.dumps({"name": "parent/valid"}))

        result = _list_sub_projects(tmp_path, "parent")
        assert len(result) == 1
        assert result[0]["name"] == "parent/valid"

    def test_dir_without_meta_skipped(self, tmp_path):
        """Directory without meta.json is skipped (line 365)."""
        from axon.projects import _list_sub_projects

        subs_dir = tmp_path / "subs"
        subs_dir.mkdir()
        (subs_dir / "no_meta").mkdir()

        result = _list_sub_projects(tmp_path, "parent")
        assert result == []

    def test_corrupt_meta_json_uses_empty_dict(self, tmp_path):
        """Corrupt meta.json uses empty dict fallback (lines 368-369)."""
        from axon.projects import _list_sub_projects

        subs_dir = tmp_path / "subs"
        subs_dir.mkdir()
        sub = subs_dir / "corrupt_sub"
        sub.mkdir()
        (sub / "meta.json").write_text("INVALID JSON")

        result = _list_sub_projects(tmp_path, "parent")
        assert len(result) == 1
        assert result[0]["name"] == "parent/corrupt_sub"


# ---------------------------------------------------------------------------
# get_maintenance_state (lines 405-406)
# ---------------------------------------------------------------------------


class TestGetMaintenanceState:
    def test_exception_reading_meta_returns_normal(self, tmp_path):
        """Exception reading meta.json returns 'normal' (lines 405-406)."""
        from axon.projects import get_maintenance_state

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            proj_dir = tmp_path / "testproj"
            proj_dir.mkdir()
            (proj_dir / "meta.json").write_text("NOT_VALID_JSON")
            state = get_maintenance_state("testproj")
        assert state == "normal"


# ---------------------------------------------------------------------------
# set_maintenance_state (lines 429-430)
# ---------------------------------------------------------------------------


class TestSetMaintenanceState:
    def test_corrupt_meta_raises_value_error(self, tmp_path):
        """Corrupt meta.json raises ValueError (lines 429-430)."""
        from axon.projects import set_maintenance_state

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            proj_dir = tmp_path / "testproj"
            proj_dir.mkdir()
            (proj_dir / "meta.json").write_text("NOT_VALID_JSON")
            with pytest.raises(ValueError, match="Could not read meta.json"):
                set_maintenance_state("testproj", "readonly")


# ---------------------------------------------------------------------------
# list_projects (lines 453, 456, 459-460)
# ---------------------------------------------------------------------------


class TestListProjects:
    def test_non_dir_entries_skipped(self, tmp_path):
        """Non-directory entries in PROJECTS_ROOT are skipped (line 453)."""
        from axon.projects import list_projects

        # Create a file in PROJECTS_ROOT
        (tmp_path / "some_file.txt").write_text("not a project")
        # Create a valid project
        proj = tmp_path / "myproj"
        proj.mkdir()
        (proj / "meta.json").write_text(json.dumps({"name": "myproj", "created_at": "2024-01-01"}))

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            result = list_projects()
        names = [p["name"] for p in result]
        assert "myproj" in names
        assert "some_file.txt" not in names

    def test_dir_without_meta_skipped(self, tmp_path):
        """Directory without meta.json is skipped (line 456)."""
        from axon.projects import list_projects

        # Create dir without meta.json
        (tmp_path / "no_meta_dir").mkdir()

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            result = list_projects()
        assert result == []

    def test_corrupt_meta_json_uses_empty_fallback(self, tmp_path):
        """Corrupt meta.json uses empty dict fallback (lines 459-460)."""
        from axon.projects import list_projects

        proj = tmp_path / "corrupt_proj"
        proj.mkdir()
        (proj / "meta.json").write_text("NOT_VALID_JSON")

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            result = list_projects()
        assert len(result) == 1
        assert result[0]["name"] == "corrupt_proj"


# ---------------------------------------------------------------------------
# _ensure_single_project_at — backfill namespace_id (lines 594-595)
# ---------------------------------------------------------------------------


class TestEnsureSingleProjectAt:
    def test_backfills_missing_namespace_id(self, tmp_path):
        """If meta.json exists without project_namespace_id, it's added (lines 594-595)."""
        from axon.projects import _ensure_single_project_at

        proj_root = tmp_path / "myproj"
        proj_root.mkdir()
        (proj_root / "chroma_data").mkdir()
        (proj_root / "bm25_index").mkdir()
        (proj_root / "sessions").mkdir()
        # Write meta without namespace_id
        meta_file = proj_root / "meta.json"
        meta_file.write_text(json.dumps({"name": "myproj", "description": "test"}))

        _ensure_single_project_at(proj_root, "myproj", "test")

        updated = json.loads(meta_file.read_text())
        assert "project_namespace_id" in updated


# ---------------------------------------------------------------------------
# _remove_share_link (lines 605-608)
# ---------------------------------------------------------------------------


class TestRemoveShareLink:
    def test_symlink_is_removed(self, tmp_path):
        """Symlink is unlinked and returns True (lines 605-607)."""
        from axon.projects import _remove_share_link

        target = tmp_path / "target.txt"
        target.write_text("target")
        link = tmp_path / "link"
        try:
            link.symlink_to(target)
            result = _remove_share_link(link)
            assert result is True
            assert not link.exists()
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported")

    def test_non_symlink_returns_false(self, tmp_path):
        """Regular file returns False (line 608)."""
        from axon.projects import _remove_share_link

        regular_file = tmp_path / "regular.txt"
        regular_file.write_text("not a symlink")
        result = _remove_share_link(regular_file)
        assert result is False

    def test_nonexistent_returns_false(self, tmp_path):
        """Non-existent path returns False."""
        from axon.projects import _remove_share_link

        result = _remove_share_link(tmp_path / "nonexistent")
        assert result is False
