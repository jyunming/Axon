"""Tests for axon.mounts — descriptor-backed mount model."""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_dir(tmp_path: Path, username: str) -> Path:
    user_dir = tmp_path / "AxonStore" / username
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / ".shares").mkdir(parents=True, exist_ok=True)
    return user_dir


def _make_project_dir(user_dir: Path, project: str) -> Path:
    proj = user_dir / project
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "meta.json").write_text(
        json.dumps({"name": project, "project_id": f"ns_{project}"}),
        encoding="utf-8",
    )
    return proj


# ---------------------------------------------------------------------------
# mounts_root / mount_descriptor_dir / mount_descriptor_path
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_mounts_root(self, tmp_path):
        from axon.mounts import mounts_root

        user_dir = tmp_path / "alice"
        assert mounts_root(user_dir) == user_dir / "mounts"

    def test_mount_descriptor_dir(self, tmp_path):
        from axon.mounts import mount_descriptor_dir

        user_dir = tmp_path / "alice"
        assert mount_descriptor_dir(user_dir, "bob_proj") == user_dir / "mounts" / "bob_proj"

    def test_mount_descriptor_path(self, tmp_path):
        from axon.mounts import mount_descriptor_path

        user_dir = tmp_path / "alice"
        assert (
            mount_descriptor_path(user_dir, "bob_proj")
            == user_dir / "mounts" / "bob_proj" / "mount.json"
        )


# ---------------------------------------------------------------------------
# create_mount_descriptor
# ---------------------------------------------------------------------------


class TestCreateMountDescriptor:
    def test_file_is_written(self, tmp_path):
        from axon.mounts import create_mount_descriptor, mount_descriptor_path

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        path = mount_descriptor_path(grantee, "alice_research")
        assert path.exists()

    def test_descriptor_fields(self, tmp_path):
        from axon.mounts import create_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        desc = create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        assert desc["mount_name"] == "alice_research"
        assert desc["owner"] == "alice"
        assert desc["project"] == "research"
        assert desc["share_key_id"] == "sk_abc123"
        assert desc["state"] == "active"
        assert desc["revoked"] is False
        assert desc["revoked_at"] is None
        assert desc["readonly"] is True
        assert desc["descriptor_version"] == 1
        assert "redeemed_at" in desc
        assert desc["target_project_dir"] == str(target)
        assert desc["owner_user_dir"] == str(owner)

    def test_reads_project_id_from_meta(self, tmp_path):
        from axon.mounts import create_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")
        # meta.json already has project_id from _make_project_dir

        desc = create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )
        assert desc["project_id"] == "ns_research"

    def test_reads_store_id_from_store_meta(self, tmp_path):
        from axon.mounts import create_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")
        (owner / "store_meta.json").write_text(
            json.dumps({"store_id": "store_ns_alice"}), encoding="utf-8"
        )

        desc = create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )
        assert desc["store_id"] == "store_ns_alice"

    def test_graceful_when_meta_missing(self, tmp_path):
        from axon.mounts import create_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = owner / "ghostproject"
        target.mkdir()
        # No meta.json, no store_meta.json

        desc = create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_ghostproject",
            owner="alice",
            project="ghostproject",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )
        assert desc["project_id"] == ""
        assert desc["store_id"] == ""

    def test_returns_same_as_written(self, tmp_path):
        from axon.mounts import create_mount_descriptor, mount_descriptor_path

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        returned = create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )
        written = json.loads(mount_descriptor_path(grantee, "alice_research").read_text())
        assert returned == written


# ---------------------------------------------------------------------------
# load_mount_descriptor
# ---------------------------------------------------------------------------


class TestLoadMountDescriptor:
    def test_loads_existing_descriptor(self, tmp_path):
        from axon.mounts import create_mount_descriptor, load_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        loaded = load_mount_descriptor(grantee, "alice_research")
        assert loaded is not None
        assert loaded["mount_name"] == "alice_research"

    def test_returns_none_when_missing(self, tmp_path):
        from axon.mounts import load_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        assert load_mount_descriptor(grantee, "nonexistent") is None

    def test_returns_none_on_corrupt_json(self, tmp_path):
        from axon.mounts import load_mount_descriptor, mount_descriptor_path

        grantee = _make_user_dir(tmp_path, "bob")
        p = mount_descriptor_path(grantee, "badmount")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ not json !!!", encoding="utf-8")

        assert load_mount_descriptor(grantee, "badmount") is None


# ---------------------------------------------------------------------------
# list_mount_descriptors
# ---------------------------------------------------------------------------


class TestListMountDescriptors:
    def test_empty_when_no_mounts_dir(self, tmp_path):
        from axon.mounts import list_mount_descriptors

        grantee = _make_user_dir(tmp_path, "bob")
        assert list_mount_descriptors(grantee) == []

    def test_lists_active_descriptor(self, tmp_path):
        from axon.mounts import create_mount_descriptor, list_mount_descriptors

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        results = list_mount_descriptors(grantee)
        assert len(results) == 1
        assert results[0]["mount_name"] == "alice_research"

    def test_excludes_revoked_descriptor(self, tmp_path):
        from axon.mounts import (
            create_mount_descriptor,
            list_mount_descriptors,
            mount_descriptor_path,
        )

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        # Mark it revoked
        p = mount_descriptor_path(grantee, "alice_research")
        desc = json.loads(p.read_text())
        desc["revoked"] = True
        p.write_text(json.dumps(desc), encoding="utf-8")

        assert list_mount_descriptors(grantee) == []

    def test_excludes_non_active_descriptor(self, tmp_path):
        from axon.mounts import (
            create_mount_descriptor,
            list_mount_descriptors,
            mount_descriptor_path,
        )

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        p = mount_descriptor_path(grantee, "alice_research")
        desc = json.loads(p.read_text())
        desc["state"] = "removed"
        p.write_text(json.dumps(desc), encoding="utf-8")

        assert list_mount_descriptors(grantee) == []

    def test_skips_corrupt_descriptor(self, tmp_path):
        from axon.mounts import list_mount_descriptors, mounts_root

        grantee = _make_user_dir(tmp_path, "bob")
        bad_dir = mounts_root(grantee) / "bad_mount"
        bad_dir.mkdir(parents=True)
        (bad_dir / "mount.json").write_text("{ broken", encoding="utf-8")

        assert list_mount_descriptors(grantee) == []

    def test_multiple_mounts_sorted(self, tmp_path):
        from axon.mounts import create_mount_descriptor, list_mount_descriptors

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        for proj in ["alpha", "beta", "gamma"]:
            target = _make_project_dir(owner, proj)
            create_mount_descriptor(
                grantee_user_dir=grantee,
                mount_name=f"alice_{proj}",
                owner="alice",
                project=proj,
                owner_user_dir=owner,
                target_project_dir=target,
                share_key_id="sk_abc123",
            )

        results = list_mount_descriptors(grantee)
        assert len(results) == 3
        names = [r["mount_name"] for r in results]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# remove_mount_descriptor
# ---------------------------------------------------------------------------


class TestRemoveMountDescriptor:
    def test_removes_existing_descriptor(self, tmp_path):
        from axon.mounts import (
            create_mount_descriptor,
            mount_descriptor_dir,
            remove_mount_descriptor,
        )

        grantee = _make_user_dir(tmp_path, "bob")
        owner = _make_user_dir(tmp_path, "alice")
        target = _make_project_dir(owner, "research")

        create_mount_descriptor(
            grantee_user_dir=grantee,
            mount_name="alice_research",
            owner="alice",
            project="research",
            owner_user_dir=owner,
            target_project_dir=target,
            share_key_id="sk_abc123",
        )

        result = remove_mount_descriptor(grantee, "alice_research")
        assert result is True
        assert not mount_descriptor_dir(grantee, "alice_research").exists()

    def test_returns_false_when_not_found(self, tmp_path):
        from axon.mounts import remove_mount_descriptor

        grantee = _make_user_dir(tmp_path, "bob")
        assert remove_mount_descriptor(grantee, "nonexistent") is False


# ---------------------------------------------------------------------------
# validate_mount_descriptor
# ---------------------------------------------------------------------------


class TestValidateMountDescriptor:
    def _base_descriptor(self, target_dir: Path) -> dict:
        return {
            "mount_name": "alice_research",
            "revoked": False,
            "state": "active",
            "target_project_dir": str(target_dir),
        }

    def test_valid_descriptor_returns_true(self, tmp_path):
        from axon.mounts import validate_mount_descriptor

        target = tmp_path / "research"
        target.mkdir()
        ok, reason = validate_mount_descriptor(self._base_descriptor(target))
        assert ok is True
        assert reason == ""

    def test_revoked_returns_false(self, tmp_path):
        from axon.mounts import validate_mount_descriptor

        target = tmp_path / "research"
        target.mkdir()
        desc = self._base_descriptor(target)
        desc["revoked"] = True
        ok, reason = validate_mount_descriptor(desc)
        assert ok is False
        assert "revoked" in reason

    def test_non_active_state_returns_false(self, tmp_path):
        from axon.mounts import validate_mount_descriptor

        target = tmp_path / "research"
        target.mkdir()
        desc = self._base_descriptor(target)
        desc["state"] = "removed"
        ok, reason = validate_mount_descriptor(desc)
        assert ok is False
        assert "state" in reason

    def test_missing_target_dir_returns_false(self, tmp_path):
        from axon.mounts import validate_mount_descriptor

        desc = self._base_descriptor(tmp_path / "nonexistent")
        ok, reason = validate_mount_descriptor(desc)
        assert ok is False
        assert "does not exist" in reason

    def test_empty_target_returns_false(self, tmp_path):
        from axon.mounts import validate_mount_descriptor

        desc = self._base_descriptor(tmp_path / "research")
        desc["target_project_dir"] = ""
        ok, reason = validate_mount_descriptor(desc)
        assert ok is False


# ---------------------------------------------------------------------------
# redeem_share_key descriptor integration (platform-independent)
# ---------------------------------------------------------------------------


class TestRedeemShareKeyDescriptor:
    """Verify descriptor creation works on all platforms (Windows, Linux, macOS)."""

    def _make_user_dir(self, tmp_path: Path, username: str) -> Path:
        user_dir = tmp_path / "AxonStore" / username
        (user_dir / ".shares").mkdir(parents=True, exist_ok=True)
        (user_dir / "myproject" / "chroma_data").mkdir(parents=True, exist_ok=True)
        (user_dir / "myproject" / "meta.json").write_text(
            json.dumps({"name": "myproject", "project_id": "ns_myproject"}),
            encoding="utf-8",
        )
        return user_dir

    def test_descriptor_created_on_redeem(self, tmp_path):
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = self._make_user_dir(tmp_path, "alice")
        grantee_dir = self._make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert "descriptor" in result
        desc = result["descriptor"]
        assert desc["mount_name"] == "alice_myproject"
        assert desc["owner"] == "alice"
        assert desc["project"] == "myproject"
        assert desc["readonly"] is True
        assert desc["state"] == "active"

        # Descriptor should be loadable from disk
        loaded = load_mount_descriptor(grantee_dir, "alice_myproject")
        assert loaded is not None
        assert loaded["share_key_id"] == gen["key_id"]

    def test_descriptor_includes_namespace_id(self, tmp_path):
        from axon import shares

        owner_dir = self._make_user_dir(tmp_path, "alice")
        grantee_dir = self._make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert result["descriptor"]["project_id"] == "ns_myproject"

    def test_redeem_twice_overwrites_descriptor(self, tmp_path):
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = self._make_user_dir(tmp_path, "alice")
        grantee_dir = self._make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])
        # Redeem again — should not error; descriptor overwritten
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        loaded = load_mount_descriptor(grantee_dir, "alice_myproject")
        assert loaded is not None

    def test_result_contains_mount_name_and_owner(self, tmp_path):
        from axon import shares

        owner_dir = self._make_user_dir(tmp_path, "alice")
        grantee_dir = self._make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert result["mount_name"] == "alice_myproject"
        assert result["owner"] == "alice"
        assert result["project"] == "myproject"
