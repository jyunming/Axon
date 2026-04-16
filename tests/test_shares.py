"""Tests for AxonStore share key management (axon/shares.py)."""


import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------


# Helpers


# ---------------------------------------------------------------------------


def _make_user_dir(tmp_path: Path, username: str) -> Path:
    """Create a minimal AxonStore root directory for testing (new Workspace/ layout).

    Each simulated user gets their own AxonStore under tmp_path/{username}/AxonStore/
    so owner and grantee do not collide.  Returns the AxonStore root (store_dir).
    Projects live under store_dir/Workspace/, and .shares/ lives at store_dir/.shares/.
    """

    store_dir = tmp_path / username / "AxonStore"

    workspace = store_dir / "Workspace"

    (store_dir / ".shares").mkdir(parents=True, exist_ok=True)

    # Create a dummy project so redeem can find it
    (workspace / "myproject" / "vector_data").mkdir(parents=True, exist_ok=True)

    (workspace / "myproject" / "meta.json").write_text(
        json.dumps({"name": "myproject", "description": "test"})
    )

    return store_dir


# ---------------------------------------------------------------------------


# Phase 3 tests


# ---------------------------------------------------------------------------


class TestGenerateShareKey:
    def test_returns_expected_fields(self, tmp_path, monkeypatch):
        import getpass as _gp

        from axon import shares

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        owner_dir = _make_user_dir(tmp_path, "alice")

        result = shares.generate_share_key(
            owner_user_dir=owner_dir,
            project="myproject",
            grantee="bob",
        )

        assert result["key_id"].startswith("sk_")

        assert result["project"] == "myproject"

        assert result["grantee"] == "bob"

        assert result["owner"] == "alice"

        assert "share_string" in result

        assert isinstance(result["share_string"], str)

        assert len(result["share_string"]) > 20

    def test_key_id_has_expected_prefix(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        result = shares.generate_share_key(owner_dir, "myproject", "bob")

        assert result["key_id"].startswith("sk_")

    def test_records_in_keys_file(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        result = shares.generate_share_key(owner_dir, "myproject", "bob")

        keys_path = owner_dir / ".shares" / ".share_keys.json"

        assert keys_path.exists()

        keys = json.loads(keys_path.read_text())

        assert len(keys["issued"]) == 1

        assert keys["issued"][0]["key_id"] == result["key_id"]

        assert keys["issued"][0]["token"]  # token is stored in private keys

    def test_records_in_manifest(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        shares.generate_share_key(owner_dir, "myproject", "bob")

        manifest_path = owner_dir / ".shares" / ".share_manifest.json"

        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())

        assert len(manifest["issued"]) == 1

        # Manifest should NOT contain the raw token

        assert "token" not in manifest["issued"][0]


class TestRevokeShareKey:
    def test_marks_revoked_in_manifest(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        key_id = gen["key_id"]

        result = shares.revoke_share_key(owner_dir, key_id)

        assert result["key_id"] == key_id

        assert result["revoked_at"] is not None

        assert result["grantee"] == "bob"

        assert result["project"] == "myproject"

        # Verify manifest is updated

        manifest = json.loads((owner_dir / ".shares" / ".share_manifest.json").read_text())

        record = next(r for r in manifest["issued"] if r["key_id"] == key_id)

        assert record["revoked"] is True

        assert record["revoked_at"] is not None

    def test_raises_if_key_not_found(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        with pytest.raises(ValueError, match="not found"):
            shares.revoke_share_key(owner_dir, "sk_nonexistent")

    def test_raises_if_already_revoked(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        shares.revoke_share_key(owner_dir, gen["key_id"])

        with pytest.raises(ValueError, match="already revoked"):
            shares.revoke_share_key(owner_dir, gen["key_id"])


class TestListShares:
    def test_returns_both_directions(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        shares.generate_share_key(owner_dir, "myproject", "bob")

        result = shares.list_shares(owner_dir)

        assert "sharing" in result

        assert "shared" in result

        assert len(result["sharing"]) == 1

        assert result["sharing"][0]["project"] == "myproject"

        assert result["sharing"][0]["grantee"] == "bob"

        assert len(result["shared"]) == 0

    def test_empty_for_new_user(self, tmp_path):
        from axon import shares

        user_dir = _make_user_dir(tmp_path, "newuser")

        result = shares.list_shares(user_dir)

        assert result["sharing"] == []

        assert result["shared"] == []

    def test_revoked_flag_reflected(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        shares.revoke_share_key(owner_dir, gen["key_id"])

        result = shares.list_shares(owner_dir)

        assert result["sharing"][0]["revoked"] is True


class TestHmacTamper:
    def test_wrong_project_raises_on_redemption(self, tmp_path, monkeypatch):
        """A share string tampered to claim a different project should fail HMAC."""

        import base64
        import getpass as _gp

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        # Create another project for the tamper attempt

        (owner_dir / "Workspace" / "otherproject" / "vector_data").mkdir(
            parents=True, exist_ok=True
        )

        (owner_dir / "Workspace" / "otherproject" / "meta.json").write_text(
            json.dumps({"name": "otherproject", "description": "other"})
        )

        grantee_dir = _make_user_dir(tmp_path, "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        # Decode the share string and tamper the project field

        raw = base64.urlsafe_b64decode(gen["share_string"].encode()).decode()

        parts = raw.split(":", 4)

        # parts: [key_id, token, owner, project, owner_store_path]

        parts[3] = "otherproject"  # tamper the project name

        tampered = base64.urlsafe_b64encode(":".join(parts).encode()).decode()

        monkeypatch.setattr(_gp, "getuser", lambda: "bob")

        with pytest.raises(ValueError):
            shares.redeem_share_key(grantee_dir, tampered)


class TestReadWriteJsonInternals:

    """Cover _read_json corrupt-file and _write_json chmod-failure branches."""

    def test_read_json_corrupt_returns_empty(self, tmp_path):
        from axon.shares import _read_json

        bad = tmp_path / "bad.json"

        bad.write_text("{ not valid json !!!", encoding="utf-8")

        result = _read_json(bad)

        assert result == {}

    def test_write_json_chmod_oserror_does_not_raise(self, tmp_path, monkeypatch):
        """_write_json must not propagate OSError from os.chmod."""

        import axon.shares as sh

        def _failing_chmod(path, mode):
            raise OSError("permission denied (test)")

        monkeypatch.setattr(sh.os, "chmod", _failing_chmod)

        p = tmp_path / ".share_manifest.json"

        # Should not raise even if chmod fails

        sh._write_json(p, {"issued": []})

        assert p.exists()

    def test_write_json_chmod_oserror_keys_file(self, tmp_path, monkeypatch):
        """Same for .share_keys.json (600 branch)."""

        import axon.shares as sh

        def _failing_chmod(path, mode):
            raise OSError("permission denied (test)")

        monkeypatch.setattr(sh.os, "chmod", _failing_chmod)

        p = tmp_path / ".share_keys.json"

        sh._write_json(p, {"issued": []})

        assert p.exists()


class TestRedeemShareKeyPlatformIndependentErrors:

    """Test error paths in redeem_share_key that do not require Linux symlinks."""

    def test_invalid_format_raises(self, tmp_path):
        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")

        with pytest.raises(ValueError, match="Invalid share_string"):
            shares.redeem_share_key(grantee_dir, "definitely-not-valid-base64!!!")

    def test_missing_project_dir_raises(self, tmp_path):
        """Project directory referenced in the share_string does not exist."""

        import base64

        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")

        owner_store = str(_make_user_dir(tmp_path, "alice"))

        raw = f"sk_test:faketoken:alice:nonexistent_project:{owner_store}"

        share_string = base64.urlsafe_b64encode(raw.encode()).decode()

        with pytest.raises(ValueError, match="does not exist"):
            shares.redeem_share_key(grantee_dir, share_string)

    def test_key_not_in_manifest_raises(self, tmp_path):
        """Key ID in share_string is not found in owner's manifest."""

        import base64
        import json

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        # Create an empty manifest (no issued keys)

        manifest_path = owner_dir / ".shares" / ".share_manifest.json"

        manifest_path.write_text(json.dumps({"issued": []}), encoding="utf-8")

        # Build a share_string pointing to alice's store but with an unknown key_id

        raw = f"sk_unknown:faketoken:alice:myproject:{str(owner_dir)}"

        share_string = base64.urlsafe_b64encode(raw.encode()).decode()

        with pytest.raises(ValueError, match="not found in owner"):
            shares.redeem_share_key(grantee_dir, share_string)

    def test_revoked_key_in_manifest_raises(self, tmp_path):
        """Key exists in manifest but is marked revoked — should raise before symlink."""

        import base64
        import json

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        key_id = "sk_revoked"

        manifest = {
            "issued": [
                {
                    "key_id": key_id,
                    "project": "myproject",
                    "grantee": "bob",
                    "revoked": True,
                    "revoked_at": "2026-01-01T00:00:00+00:00",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        }

        manifest_path = owner_dir / ".shares" / ".share_manifest.json"

        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        raw = f"{key_id}:faketoken:alice:myproject:{str(owner_dir)}"

        share_string = base64.urlsafe_b64encode(raw.encode()).decode()

        with pytest.raises(ValueError, match="revoked"):
            shares.redeem_share_key(grantee_dir, share_string)

    def test_key_not_in_key_store_raises(self, tmp_path):
        """Key is in manifest (not revoked) but absent from private key store."""

        import base64
        import json

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        key_id = "sk_missing_ks"

        manifest = {
            "issued": [
                {
                    "key_id": key_id,
                    "project": "myproject",
                    "grantee": "bob",
                    "revoked": False,
                    "revoked_at": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        }

        manifest_path = owner_dir / ".shares" / ".share_manifest.json"

        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Write an empty key store (no issued records)

        keys_path = owner_dir / ".shares" / ".share_keys.json"

        keys_path.write_text(json.dumps({"issued": []}), encoding="utf-8")

        raw = f"{key_id}:faketoken:alice:myproject:{str(owner_dir)}"

        share_string = base64.urlsafe_b64encode(raw.encode()).decode()

        with pytest.raises(ValueError, match="not found in owner"):
            shares.redeem_share_key(grantee_dir, share_string)


class TestRedeemShareKey:
    def test_creates_descriptor(self, tmp_path, monkeypatch):
        import getpass as _gp

        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "bob")

        result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert result["mount_name"] == "alice_myproject"

        assert result["owner"] == "alice"

        assert result["project"] == "myproject"

        assert "descriptor" in result

        desc = load_mount_descriptor(grantee_dir, "alice_myproject")

        assert desc is not None

        assert desc["state"] == "active"

    def test_invalid_share_string_raises(self, tmp_path):
        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")

        with pytest.raises(ValueError, match="Invalid share_string"):
            shares.redeem_share_key(grantee_dir, "not_a_valid_base64_string!!!")

    def test_revoked_key_raises_on_redemption(self, tmp_path, monkeypatch):
        import getpass as _gp

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        shares.revoke_share_key(owner_dir, gen["key_id"])

        monkeypatch.setattr(_gp, "getuser", lambda: "bob")

        with pytest.raises(ValueError, match="revoked"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])


class TestValidateReceivedShares:
    def test_removes_stale_descriptor_on_revocation(self, tmp_path, monkeypatch):
        import getpass as _gp

        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "bob")

        shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Verify descriptor exists

        assert load_mount_descriptor(grantee_dir, "alice_myproject") is not None

        # Revoke the key

        shares.revoke_share_key(owner_dir, gen["key_id"])

        # Validate should remove the descriptor

        removed = shares.validate_received_shares(grantee_dir)

        assert "alice_myproject" in removed

        assert load_mount_descriptor(grantee_dir, "alice_myproject") is None

    def test_nothing_removed_if_not_revoked(self, tmp_path, monkeypatch):
        import getpass as _gp

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        monkeypatch.setattr(_gp, "getuser", lambda: "bob")

        shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Don't revoke — validate should return empty list

        removed = shares.validate_received_shares(grantee_dir)

        assert removed == []

    def test_malformed_received_record_missing_key_id_is_skipped(self, tmp_path):
        """validate_received_shares must not crash when a received record has no key_id."""

        import json as _json

        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")

        keys_path = grantee_dir / ".shares" / ".share_keys.json"

        # Inject a malformed record (missing key_id)

        keys_path.write_text(
            _json.dumps({"received": [{"owner_manifest_path": "/nonexistent/manifest.json"}]}),
            encoding="utf-8",
        )

        # Should not raise KeyError

        removed = shares.validate_received_shares(grantee_dir)

        assert removed == []


class TestRevokeShareKeyManifestOutOfSync:

    """revoke_share_key must still mark the key revoked when manifest is out of sync."""

    def test_tombstone_added_when_key_absent_from_manifest(self, tmp_path, monkeypatch):
        """If the manifest is missing the key_id record, a tombstone is written so


        redeem_share_key() will correctly reject the key."""

        import getpass as _gp
        import json as _json

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        monkeypatch.setattr(_gp, "getuser", lambda: "alice")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        key_id = gen["key_id"]

        # Artificially empty the manifest's issued list to simulate out-of-sync state

        manifest_path = owner_dir / ".shares" / ".share_manifest.json"

        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))

        manifest["issued"] = []

        manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")

        # Revocation must not raise, and must add a tombstone to the manifest

        result = shares.revoke_share_key(owner_dir, key_id)

        assert result["key_id"] == key_id

        updated = _json.loads(manifest_path.read_text(encoding="utf-8"))

        tombstone = next((r for r in updated.get("issued", []) if r["key_id"] == key_id), None)

        assert tombstone is not None

        assert tombstone["revoked"] is True


# ---------------------------------------------------------------------------


# Phase 1: write-enforcement tests


# ---------------------------------------------------------------------------


class TestMountedShareWriteEnforcement:

    """Verify that AxonBrain refuses mutation when the active project is a mounted share."""

    def _make_brain(self):
        """Return a minimal AxonBrain-like mock with write-guard methods."""

        from unittest.mock import MagicMock

        brain = MagicMock()

        # Wire the real methods from AxonBrain onto the mock

        from axon.main import AxonBrain

        brain._is_mounted_share = AxonBrain._is_mounted_share.__get__(brain, type(brain))

        brain._assert_write_allowed = AxonBrain._assert_write_allowed.__get__(brain, type(brain))

        brain._active_project = "mounts/alice_myproject"

        brain._read_only_scope = False

        brain._mounted_share = True

        brain._active_project_kind = "mounted"

        return brain

    def test_ingest_on_mounted_share_raises_permission_error(self):
        brain = self._make_brain()

        with pytest.raises(PermissionError, match="mounted share"):
            brain._assert_write_allowed("ingest")

    def test_delete_on_mounted_share_raises_permission_error(self):
        brain = self._make_brain()

        with pytest.raises(PermissionError, match="mounted share"):
            brain._assert_write_allowed("delete")

    def test_finalize_on_mounted_share_raises_permission_error(self):
        brain = self._make_brain()

        with pytest.raises(PermissionError, match="mounted share"):
            brain._assert_write_allowed("finalize_ingest")

    def test_read_only_scope_raises_permission_error(self):
        brain = self._make_brain()

        brain._mounted_share = False

        brain._active_project_kind = "local"

        brain._read_only_scope = True

        with pytest.raises(PermissionError, match="read-only"):
            brain._assert_write_allowed("write")

    def test_own_project_allows_write(self):
        brain = self._make_brain()

        brain._mounted_share = False

        brain._active_project_kind = "local"

        brain._read_only_scope = False

        # Should not raise

        brain._assert_write_allowed("ingest")

    def test_is_mounted_share_returns_true_when_flag_set(self):
        brain = self._make_brain()

        assert brain._is_mounted_share() is True

    def test_is_mounted_share_returns_false_when_flag_clear(self):
        brain = self._make_brain()

        brain._mounted_share = False

        brain._active_project_kind = "local"

        assert brain._is_mounted_share() is False
