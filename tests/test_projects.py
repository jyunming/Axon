"""Tests for src/axon/projects.py"""

import json
from unittest.mock import patch

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

    def test_ignores_reserved_projects_dir_even_with_meta(self, tmp_projects):
        from axon.projects import list_projects

        reserved = tmp_projects / "projects"
        reserved.mkdir()
        (reserved / "meta.json").write_text(
            json.dumps({"name": "projects", "created_at": "2026-01-01"}),
            encoding="utf-8",
        )
        assert list_projects() == []


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

    def test_delete_project_retries_on_permission_error(self, tmp_projects):
        from axon.projects import delete_project, ensure_project

        ensure_project("retry-test")

        # Mock shutil.rmtree to fail once with PermissionError and then succeed
        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = [PermissionError("Locked"), None]
            with patch("time.sleep") as mock_sleep:  # avoid actual waiting
                delete_project("retry-test")
                assert mock_rmtree.call_count == 2
                mock_sleep.assert_called_with(0.5)


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

        # _MAX_DEPTH is now 5, so 4 levels are valid; 6 levels should raise
        _validate_name("a/b/c/d")  # should not raise (4 levels <= 5)
        with pytest.raises(ValueError, match="maximum depth"):
            _validate_name("a/b/c/d/e/f")  # 6 levels > 5

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


# ---------------------------------------------------------------------------
# AxonStore: ensure_user_project
# ---------------------------------------------------------------------------


class TestEnsureUserNamespace:
    def test_creates_expected_directories(self, tmp_path):
        """ensure_user_project creates default/, projects/, mounts/, .shares/."""
        from axon.projects import ensure_user_project

        user_dir = tmp_path / "AxonStore" / "alice"
        ensure_user_project(user_dir)

        assert (user_dir / "default").is_dir()
        assert (user_dir / "projects").is_dir()
        assert (user_dir / "mounts").is_dir()
        assert (user_dir / ".shares").is_dir()
        # Legacy _default and ShareMount/ must NOT be created
        assert not (user_dir / "_default").exists()
        assert not (user_dir / "ShareMount").exists()

    def test_creates_store_meta_json(self, tmp_path):
        """ensure_user_project creates store_meta.json with store_id."""
        import json

        from axon.projects import ensure_user_project

        user_dir = tmp_path / "AxonStore" / "alice"
        ensure_user_project(user_dir)

        store_meta_path = user_dir / "store_meta.json"
        assert store_meta_path.exists()
        meta = json.loads(store_meta_path.read_text())
        assert meta["store_version"] == 2
        assert meta["store_id"].startswith("store_")

    def test_idempotent(self, tmp_path):
        """Calling ensure_user_project twice does not raise."""
        from axon.projects import ensure_user_project

        user_dir = tmp_path / "AxonStore" / "alice"
        ensure_user_project(user_dir)
        ensure_user_project(user_dir)  # should not raise

    def test_creates_default_meta_json(self, tmp_path):
        """ensure_user_project creates meta.json in default/ with project_id."""
        import json

        from axon.projects import ensure_user_project

        user_dir = tmp_path / "AxonStore" / "alice"
        ensure_user_project(user_dir)

        meta_path = user_dir / "default" / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["name"] == "default"
        assert meta["project_id"].startswith("proj_")


# ---------------------------------------------------------------------------
# Phase 1: namespace IDs
# ---------------------------------------------------------------------------


class TestNamespaceIds:
    def test_ensure_project_adds_namespace_id(self, tmp_projects):
        """ensure_project writes project_id to meta.json."""
        import json

        from axon.projects import ensure_project, project_dir

        ensure_project("nstest")
        meta = json.loads((project_dir("nstest") / "meta.json").read_text())
        assert "project_id" in meta
        assert meta["project_id"].startswith("proj_")

    def test_namespace_id_stable_on_second_call(self, tmp_projects):
        """ensure_project called twice keeps the same project_id."""
        import json

        from axon.projects import ensure_project, project_dir

        ensure_project("stable-ns")
        meta1 = json.loads((project_dir("stable-ns") / "meta.json").read_text())
        ensure_project("stable-ns")
        meta2 = json.loads((project_dir("stable-ns") / "meta.json").read_text())
        assert meta1["project_id"] == meta2["project_id"]

    def test_backfills_missing_namespace_id(self, tmp_projects):
        """ensure_project backfills project_id when meta.json exists without it."""
        import json

        from axon.projects import ensure_project, project_dir

        ensure_project("backfill-ns")
        meta_path = project_dir("backfill-ns") / "meta.json"
        # Remove the namespace ID to simulate an old project
        meta = json.loads(meta_path.read_text())
        del meta["project_id"]
        meta_path.write_text(json.dumps(meta))

        ensure_project("backfill-ns")  # should backfill
        updated = json.loads(meta_path.read_text())
        assert "project_id" in updated
        assert updated["project_id"].startswith("proj_")

    def test_get_project_id(self, tmp_projects):
        """get_project_id returns the ID for an existing project."""
        from axon.projects import ensure_project, get_project_id

        ensure_project("getter-ns")
        ns_id = get_project_id("getter-ns")
        assert ns_id is not None
        assert ns_id.startswith("proj_")

    def test_get_project_id_missing_project(self, tmp_projects):
        """get_project_id returns None for a non-existent project."""
        from axon.projects import get_project_id

        assert get_project_id("does-not-exist") is None

    def test_get_store_id(self, tmp_path):
        """get_store_id reads the ID from store_meta.json."""
        from axon.projects import ensure_user_project, get_store_id

        user_dir = tmp_path / "AxonStore" / "alice"
        ensure_user_project(user_dir)
        store_id = get_store_id(user_dir)
        assert store_id is not None
        assert store_id.startswith("store_")

    def test_build_project_id_format(self):
        """build_project_id returns a prefixed hex string."""
        from axon.projects import build_project_id

        ns = build_project_id("proj")
        assert ns.startswith("proj_")
        assert len(ns) == len("proj_") + 32  # 32 hex chars in uuid4.hex

    def test_namespace_ids_are_unique(self):
        """Two calls to build_project_id never return the same value."""
        from axon.projects import build_project_id

        assert build_project_id("proj") != build_project_id("proj")


# ---------------------------------------------------------------------------
# Phase 2: canonical ID builders
# ---------------------------------------------------------------------------


class TestIdBuilder:
    def test_build_source_id_format(self):
        from axon.projects import build_source_id

        sid = build_source_id("proj_abc", "file", "/docs/overview.md")
        assert sid.startswith("src_")
        assert len(sid) == 28  # "src_" + 24 hex chars

    def test_build_source_id_deterministic(self):
        from axon.projects import build_source_id

        sid1 = build_source_id("proj_abc", "file", "/docs/overview.md")
        sid2 = build_source_id("proj_abc", "file", "/docs/overview.md")
        assert sid1 == sid2

    def test_build_source_id_different_namespaces(self):
        from axon.projects import build_source_id

        sid1 = build_source_id("proj_aaa", "file", "/docs/overview.md")
        sid2 = build_source_id("proj_bbb", "file", "/docs/overview.md")
        assert sid1 != sid2

    def test_build_chunk_id_format(self):
        from axon.projects import build_chunk_id

        cid = build_chunk_id("proj_abc", "src_xyz", "root", 0)
        assert cid.startswith("chk_")
        assert len(cid) == 28

    def test_build_chunk_id_deterministic(self):
        from axon.projects import build_chunk_id

        cid1 = build_chunk_id("proj_abc", "src_xyz", "root", 0, "leaf")
        cid2 = build_chunk_id("proj_abc", "src_xyz", "root", 0, "leaf")
        assert cid1 == cid2

    def test_build_chunk_id_unique_per_index(self):
        from axon.projects import build_chunk_id

        cid1 = build_chunk_id("proj_abc", "src_xyz", "root", 0)
        cid2 = build_chunk_id("proj_abc", "src_xyz", "root", 1)
        assert cid1 != cid2

    def test_build_chunk_id_unique_per_kind(self):
        from axon.projects import build_chunk_id

        cid1 = build_chunk_id("proj_abc", "src_xyz", "root", 0, "leaf")
        cid2 = build_chunk_id("proj_abc", "src_xyz", "root", 0, "raptor_l1")
        assert cid1 != cid2


# ---------------------------------------------------------------------------
# AxonStore: reserved names
# ---------------------------------------------------------------------------


class TestReservedNames:
    def test_projects_dir_is_reserved(self, tmp_projects):
        """'projects' is reserved for AxonStore compatibility layout."""
        from axon.projects import ensure_project

        with pytest.raises(ValueError, match="reserved"):
            ensure_project("projects")

    def test_mounts_is_reserved(self, tmp_projects):
        """'mounts' cannot be used as a local project name."""
        from axon.projects import ensure_project

        with pytest.raises(ValueError, match="reserved"):
            ensure_project("mounts")

    def test_sharemount_is_reserved(self, tmp_projects):
        """'sharemount' cannot be used as a project name."""
        from axon.projects import ensure_project

        with pytest.raises(ValueError, match="reserved"):
            ensure_project("sharemount")

    def test_default_internal_is_reserved(self, tmp_projects):
        """'_default' starts with '_' so it fails the segment regex."""
        from axon.projects import ensure_project

        with pytest.raises(ValueError):
            ensure_project("_default")

    def test_shares_dir_is_reserved(self, tmp_projects):
        """'.shares' starts with '.' so it fails the segment regex."""
        from axon.projects import ensure_project

        with pytest.raises(ValueError):
            ensure_project(".shares")


# ---------------------------------------------------------------------------
# AxonStore: list_share_mounts
# ---------------------------------------------------------------------------


class TestListShareMounts:
    def test_empty_when_no_mounts(self, tmp_path):
        """list_share_mounts returns [] when no mount descriptors exist."""
        from axon.projects import list_share_mounts

        user_dir = tmp_path / "AxonStore" / "alice"
        user_dir.mkdir(parents=True)

        result = list_share_mounts(user_dir)
        assert result == []

    def test_returns_mount_entries_from_descriptors(self, tmp_path):
        """list_share_mounts returns dicts from mounts/ descriptors."""
        import json

        from axon.mounts import mount_descriptor_path
        from axon.projects import list_share_mounts

        user_dir = tmp_path / "AxonStore" / "alice"
        target = tmp_path / "AxonStore" / "bob" / "research"
        target.mkdir(parents=True)

        desc_path = mount_descriptor_path(user_dir, "bob_research")
        desc_path.parent.mkdir(parents=True)
        desc_path.write_text(
            json.dumps(
                {
                    "mount_name": "bob_research",
                    "owner": "bob",
                    "project": "research",
                    "target_project_dir": str(target),
                    "state": "active",
                    "revoked": False,
                    "descriptor_version": 1,
                }
            ),
            encoding="utf-8",
        )

        result = list_share_mounts(user_dir)
        names = [entry["name"] for entry in result]
        assert "bob_research" in names


class TestSetActiveProjectPermissionError:
    """Regression: set_active_project must be non-fatal when home dir is not writable."""

    def test_permission_error_is_swallowed(self, tmp_path, monkeypatch):
        import axon.projects as _p

        # Point _ACTIVE_FILE to a read-only location
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        fake_file = ro_dir / ".active_project"

        # Make mkdir and write_text raise PermissionError

        def boom(*a, **kw):
            raise PermissionError("permission denied")

        monkeypatch.setattr(_p, "_ACTIVE_FILE", fake_file)
        monkeypatch.setattr(type(fake_file.parent), "mkdir", lambda *a, **kw: boom())

        # Must not raise
        _p.set_active_project("myproject")
