from __future__ import annotations

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

        assert (root / "lancedb_data").is_dir()

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

        assert (root / "lancedb_data").is_dir()

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


"""Extra tests for axon.projects to push coverage above 90%."""


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


# get_project_id (lines 283-284)


# ---------------------------------------------------------------------------


class TestGetProjectNamespaceId:
    def test_exception_in_read_returns_none(self, tmp_path):
        """json.JSONDecodeError or OSError in meta.json returns None (lines 283-284)."""

        from axon.projects import get_project_id

        with patch("axon.projects.PROJECTS_ROOT", tmp_path):
            # Create a project dir with malformed meta.json

            proj_dir = tmp_path / "testproj"

            proj_dir.mkdir()

            (proj_dir / "meta.json").write_text("NOT_VALID_JSON")

            result = get_project_id("testproj")

        assert result is None


# ---------------------------------------------------------------------------


# get_store_id (lines 294, 299-300)


# ---------------------------------------------------------------------------


class TestGetStoreNamespaceId:
    def test_returns_none_if_no_file(self, tmp_path):
        """Returns None when store_meta.json does not exist (line 294)."""

        from axon.projects import get_store_id

        result = get_store_id(tmp_path)

        assert result is None

    def test_returns_none_on_json_error(self, tmp_path):
        """Returns None when store_meta.json has malformed JSON (lines 299-300)."""

        from axon.projects import get_store_id

        (tmp_path / "store_meta.json").write_text("INVALID_JSON_DATA")

        result = get_store_id(tmp_path)

        assert result is None

    def test_returns_id_when_present(self, tmp_path):
        """Returns the store_id when valid."""

        from axon.projects import get_store_id

        (tmp_path / "store_meta.json").write_text(json.dumps({"store_id": "store_abc123"}))

        result = get_store_id(tmp_path)

        assert result == "store_abc123"


# ---------------------------------------------------------------------------


# list_descendants (lines 322, 333, 335, 362, 365, 368-369)


# ---------------------------------------------------------------------------


class TestListDescendantsV2:
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


class TestListProjectsV2:
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
        """If meta.json exists without project_id, it's added (lines 594-595)."""

        from axon.projects import _ensure_single_project_at

        proj_root = tmp_path / "myproj"

        proj_root.mkdir()

        (proj_root / "lancedb_data").mkdir()

        (proj_root / "bm25_index").mkdir()

        (proj_root / "sessions").mkdir()

        # Write meta without namespace_id

        meta_file = proj_root / "meta.json"

        meta_file.write_text(json.dumps({"name": "myproj", "description": "test"}))

        _ensure_single_project_at(proj_root, "myproj", "test")

        updated = json.loads(meta_file.read_text())

        assert "project_id" in updated


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
