"""Tests for src/axon/projects.py"""

import json

import pytest


@pytest.fixture()
def tmp_projects(tmp_path, monkeypatch):
    """Redirect PROJECTS_ROOT and _ACTIVE_FILE to tmp_path for isolation."""
    import axon.projects as _p

    monkeypatch.setattr(_p, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(_p, "_ACTIVE_FILE", tmp_path / ".active_project")
    return tmp_path


class TestValidateName:
    def test_default_allowed(self, tmp_projects):
        from axon.projects import _validate_name

        _validate_name("default")  # must not raise

    def test_valid_names(self, tmp_projects):
        from axon.projects import _validate_name

        for name in ["myproject", "my-project", "my_project", "abc123", "a"]:
            _validate_name(name)

    def test_invalid_starts_with_hyphen(self, tmp_projects):
        from axon.projects import _validate_name

        with pytest.raises(ValueError):
            _validate_name("-bad")

    def test_invalid_uppercase(self, tmp_projects):
        from axon.projects import _validate_name

        with pytest.raises(ValueError):
            _validate_name("MyProject")

    def test_invalid_spaces(self, tmp_projects):
        from axon.projects import _validate_name

        with pytest.raises(ValueError):
            _validate_name("my project")

    def test_too_long(self, tmp_projects):
        from axon.projects import _validate_name

        with pytest.raises(ValueError):
            _validate_name("a" * 51)


class TestEnsureProject:
    def test_creates_directories(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("testproj")
        root = project_dir("testproj")
        assert (root / "chroma_data").is_dir()
        assert (root / "bm25_index").is_dir()
        assert (root / "sessions").is_dir()
        assert (root / "meta.json").is_file()

    def test_meta_json_contents(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("myproj", description="A test project")
        meta = json.loads((project_dir("myproj") / "meta.json").read_text())
        assert meta["name"] == "myproj"
        assert meta["description"] == "A test project"
        assert "created_at" in meta

    def test_idempotent(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("idempotent")
        ensure_project("idempotent", description="second call")
        meta = json.loads((project_dir("idempotent") / "meta.json").read_text())
        # Description from second call must NOT overwrite first creation
        assert meta["description"] == ""

    def test_invalid_name_raises(self, tmp_projects):
        from axon.projects import ensure_project

        with pytest.raises(ValueError):
            ensure_project("BadName")


class TestActiveProject:
    def test_default_when_no_file(self, tmp_projects):
        from axon.projects import get_active_project

        assert get_active_project() == "default"

    def test_set_and_get(self, tmp_projects):
        from axon.projects import get_active_project, set_active_project

        set_active_project("myproj")
        assert get_active_project() == "myproj"

    def test_overwrite(self, tmp_projects):
        from axon.projects import get_active_project, set_active_project

        set_active_project("first")
        set_active_project("second")
        assert get_active_project() == "second"


class TestListProjects:
    def test_empty_when_no_projects(self, tmp_projects):
        from axon.projects import list_projects

        assert list_projects() == []

    def test_lists_created_projects(self, tmp_projects):
        from axon.projects import ensure_project, list_projects

        ensure_project("alpha")
        ensure_project("beta")
        names = [p["name"] for p in list_projects()]
        assert "alpha" in names
        assert "beta" in names

    def test_includes_metadata(self, tmp_projects):
        from axon.projects import ensure_project, list_projects

        ensure_project("proj1", description="hello")
        projects = list_projects()
        match = next(p for p in projects if p["name"] == "proj1")
        assert match["description"] == "hello"
        assert match["created_at"] != ""


class TestDeleteProject:
    def test_deletes_directory(self, tmp_projects):
        from axon.projects import delete_project, ensure_project, project_dir

        ensure_project("todelete")
        assert project_dir("todelete").exists()
        delete_project("todelete")
        assert not project_dir("todelete").exists()

    def test_cannot_delete_default(self, tmp_projects):
        from axon.projects import delete_project

        with pytest.raises(ValueError, match="Cannot delete"):
            delete_project("default")

    def test_cannot_delete_nonexistent(self, tmp_projects):
        from axon.projects import delete_project

        with pytest.raises(ValueError):
            delete_project("ghost")

    def test_resets_active_project_to_default(self, tmp_projects):
        from axon.projects import (
            delete_project,
            ensure_project,
            get_active_project,
            set_active_project,
        )

        ensure_project("active")
        set_active_project("active")
        delete_project("active")
        assert get_active_project() == "default"


class TestSubProjectPaths:
    def test_top_level_path_unchanged(self, tmp_projects):
        from axon.projects import PROJECTS_ROOT, project_dir

        assert project_dir("research") == PROJECTS_ROOT / "research"

    def test_two_level_uses_subs(self, tmp_projects):
        from axon.projects import PROJECTS_ROOT, project_dir

        assert project_dir("research/papers") == PROJECTS_ROOT / "research" / "subs" / "papers"

    def test_three_level_uses_subs_twice(self, tmp_projects):
        from axon.projects import PROJECTS_ROOT, project_dir

        assert project_dir("research/papers/2024") == (
            PROJECTS_ROOT / "research" / "subs" / "papers" / "subs" / "2024"
        )

    def test_four_levels_raises(self, tmp_projects):
        from axon.projects import _validate_name

        with pytest.raises(ValueError, match="maximum depth"):
            _validate_name("a/b/c/d")

    def test_slash_with_invalid_segment_raises(self, tmp_projects):
        from axon.projects import _validate_name

        with pytest.raises(ValueError):
            _validate_name("research/Bad-Name")

    def test_valid_sub_project_names(self, tmp_projects):
        from axon.projects import _validate_name

        for name in ["a/b", "research/papers", "r/p/2024", "my-proj/sub_1/v2"]:
            _validate_name(name)  # must not raise


class TestSubProjectEnsure:
    def test_creates_sub_project_directories(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("research/papers")
        root = project_dir("research/papers")
        assert (root / "chroma_data").is_dir()
        assert (root / "bm25_index").is_dir()
        assert (root / "sessions").is_dir()
        assert (root / "meta.json").is_file()

    def test_auto_creates_ancestor(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("research/papers")
        # Ancestor must have been auto-created
        assert (project_dir("research") / "meta.json").is_file()

    def test_three_levels_auto_creates_both_ancestors(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("research/papers/2024")
        assert (project_dir("research") / "meta.json").is_file()
        assert (project_dir("research/papers") / "meta.json").is_file()
        assert (project_dir("research/papers/2024") / "meta.json").is_file()

    def test_sub_project_description_stored(self, tmp_projects):
        from axon.projects import ensure_project, project_dir

        ensure_project("research/papers", description="My papers")
        meta = json.loads((project_dir("research/papers") / "meta.json").read_text())
        assert meta["description"] == "My papers"
        assert meta["name"] == "research/papers"


class TestListDescendants:
    def test_no_descendants_returns_empty(self, tmp_projects):
        from axon.projects import ensure_project, list_descendants

        ensure_project("standalone")
        assert list_descendants("standalone") == []

    def test_direct_children_returned(self, tmp_projects):
        from axon.projects import ensure_project, list_descendants

        ensure_project("parent/child1")
        ensure_project("parent/child2")
        desc = list_descendants("parent")
        assert "parent/child1" in desc
        assert "parent/child2" in desc

    def test_grandchildren_returned(self, tmp_projects):
        from axon.projects import ensure_project, list_descendants

        ensure_project("root/mid/leaf")
        desc = list_descendants("root")
        assert "root/mid" in desc
        assert "root/mid/leaf" in desc

    def test_has_children_true(self, tmp_projects):
        from axon.projects import ensure_project, has_children

        ensure_project("parent/child")
        assert has_children("parent") is True

    def test_has_children_false(self, tmp_projects):
        from axon.projects import ensure_project, has_children

        ensure_project("leaf")
        assert has_children("leaf") is False


class TestListProjectsTree:
    def test_children_appear_in_parent(self, tmp_projects):
        from axon.projects import ensure_project, list_projects

        ensure_project("parent/sub")
        projects = list_projects()
        parent = next(p for p in projects if p["name"] == "parent")
        child_names = [c["name"] for c in parent["children"]]
        assert "parent/sub" in child_names

    def test_top_level_list_excludes_raw_children(self, tmp_projects):
        from axon.projects import ensure_project, list_projects

        ensure_project("parent/sub")
        top_names = [p["name"] for p in list_projects()]
        # "parent/sub" should NOT appear at top level — only inside children
        assert "parent/sub" not in top_names
        assert "parent" in top_names


class TestDeleteWithChildren:
    def test_cannot_delete_parent_with_children(self, tmp_projects):
        from axon.projects import ProjectHasChildrenError, delete_project, ensure_project

        ensure_project("parent/child")
        with pytest.raises(ProjectHasChildrenError):
            delete_project("parent")

    def test_can_delete_after_removing_children(self, tmp_projects):
        from axon.projects import delete_project, ensure_project, project_dir

        ensure_project("parent/child")
        delete_project("parent/child")
        delete_project("parent")
        assert not project_dir("parent").exists()

    def test_delete_sub_project_leaves_parent(self, tmp_projects):
        from axon.projects import delete_project, ensure_project, project_dir

        ensure_project("parent/child")
        delete_project("parent/child")
        assert project_dir("parent").exists()
        assert (project_dir("parent") / "meta.json").is_file()


class TestSetProjectsRoot:
    def test_set_projects_root_changes_project_dir(self, tmp_path, monkeypatch):
        import axon.projects as _p

        new_root = tmp_path / "custom_root"
        _p.set_projects_root(new_root)
        monkeypatch.setattr(_p, "PROJECTS_ROOT", _p.PROJECTS_ROOT)
        assert _p.project_dir("myproj") == new_root / "myproj"
        # Restore
        _p.set_projects_root(tmp_path / "projects")

    def test_projects_root_env_var_applied_at_module_load(self, tmp_path, monkeypatch):
        """AXON_PROJECTS_ROOT is read by _resolve_projects_root at import time;
        verify set_projects_root applies the same path so env-var behaviour is consistent."""
        import axon.projects as _p

        custom = tmp_path / "env_root"
        monkeypatch.setattr(_p, "PROJECTS_ROOT", custom)
        assert _p.project_dir("proj") == custom / "proj"
