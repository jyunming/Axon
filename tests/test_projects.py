"""Tests for src/rag_brain/projects.py"""
import json
import pytest


@pytest.fixture()
def tmp_projects(tmp_path, monkeypatch):
    """Redirect PROJECTS_ROOT and _ACTIVE_FILE to tmp_path for isolation."""
    import rag_brain.projects as _p
    monkeypatch.setattr(_p, "PROJECTS_ROOT", tmp_path / "projects")
    monkeypatch.setattr(_p, "_ACTIVE_FILE", tmp_path / ".active_project")
    return tmp_path


class TestValidateName:
    def test_default_allowed(self, tmp_projects):
        from rag_brain.projects import _validate_name
        _validate_name("default")   # must not raise

    def test_valid_names(self, tmp_projects):
        from rag_brain.projects import _validate_name
        for name in ["myproject", "my-project", "my_project", "abc123", "a"]:
            _validate_name(name)

    def test_invalid_starts_with_hyphen(self, tmp_projects):
        from rag_brain.projects import _validate_name
        with pytest.raises(ValueError):
            _validate_name("-bad")

    def test_invalid_uppercase(self, tmp_projects):
        from rag_brain.projects import _validate_name
        with pytest.raises(ValueError):
            _validate_name("MyProject")

    def test_invalid_spaces(self, tmp_projects):
        from rag_brain.projects import _validate_name
        with pytest.raises(ValueError):
            _validate_name("my project")

    def test_too_long(self, tmp_projects):
        from rag_brain.projects import _validate_name
        with pytest.raises(ValueError):
            _validate_name("a" * 51)


class TestEnsureProject:
    def test_creates_directories(self, tmp_projects):
        from rag_brain.projects import ensure_project, project_dir
        ensure_project("testproj")
        root = project_dir("testproj")
        assert (root / "chroma_data").is_dir()
        assert (root / "bm25_index").is_dir()
        assert (root / "sessions").is_dir()
        assert (root / "meta.json").is_file()

    def test_meta_json_contents(self, tmp_projects):
        from rag_brain.projects import ensure_project, project_dir
        ensure_project("myproj", description="A test project")
        meta = json.loads((project_dir("myproj") / "meta.json").read_text())
        assert meta["name"] == "myproj"
        assert meta["description"] == "A test project"
        assert "created_at" in meta

    def test_idempotent(self, tmp_projects):
        from rag_brain.projects import ensure_project, project_dir
        ensure_project("idempotent")
        ensure_project("idempotent", description="second call")
        meta = json.loads((project_dir("idempotent") / "meta.json").read_text())
        # Description from second call must NOT overwrite first creation
        assert meta["description"] == ""

    def test_invalid_name_raises(self, tmp_projects):
        from rag_brain.projects import ensure_project
        with pytest.raises(ValueError):
            ensure_project("BadName")


class TestActiveProject:
    def test_default_when_no_file(self, tmp_projects):
        from rag_brain.projects import get_active_project
        assert get_active_project() == "default"

    def test_set_and_get(self, tmp_projects):
        from rag_brain.projects import get_active_project, set_active_project
        set_active_project("myproj")
        assert get_active_project() == "myproj"

    def test_overwrite(self, tmp_projects):
        from rag_brain.projects import get_active_project, set_active_project
        set_active_project("first")
        set_active_project("second")
        assert get_active_project() == "second"


class TestListProjects:
    def test_empty_when_no_projects(self, tmp_projects):
        from rag_brain.projects import list_projects
        assert list_projects() == []

    def test_lists_created_projects(self, tmp_projects):
        from rag_brain.projects import ensure_project, list_projects
        ensure_project("alpha")
        ensure_project("beta")
        names = [p["name"] for p in list_projects()]
        assert "alpha" in names
        assert "beta" in names

    def test_includes_metadata(self, tmp_projects):
        from rag_brain.projects import ensure_project, list_projects
        ensure_project("proj1", description="hello")
        projects = list_projects()
        match = next(p for p in projects if p["name"] == "proj1")
        assert match["description"] == "hello"
        assert match["created_at"] != ""


class TestDeleteProject:
    def test_deletes_directory(self, tmp_projects):
        from rag_brain.projects import delete_project, ensure_project, project_dir
        ensure_project("todelete")
        assert project_dir("todelete").exists()
        delete_project("todelete")
        assert not project_dir("todelete").exists()

    def test_cannot_delete_default(self, tmp_projects):
        from rag_brain.projects import delete_project
        with pytest.raises(ValueError, match="Cannot delete"):
            delete_project("default")

    def test_cannot_delete_nonexistent(self, tmp_projects):
        from rag_brain.projects import delete_project
        with pytest.raises(ValueError):
            delete_project("ghost")

    def test_resets_active_project_to_default(self, tmp_projects):
        from rag_brain.projects import (
            delete_project, ensure_project, get_active_project, set_active_project,
        )
        ensure_project("active")
        set_active_project("active")
        delete_project("active")
        assert get_active_project() == "default"
