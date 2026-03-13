"""Tests for AxonStore share key management (axon/shares.py)."""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_dir(tmp_path: Path, username: str) -> Path:
    """Create a minimal AxonStore user directory for testing."""
    user_dir = tmp_path / "AxonStore" / username
    (user_dir / "ShareMount").mkdir(parents=True, exist_ok=True)
    (user_dir / ".shares").mkdir(parents=True, exist_ok=True)
    # Create a dummy project so redeem can find it
    (user_dir / "myproject" / "chroma_data").mkdir(parents=True, exist_ok=True)
    (user_dir / "myproject" / "meta.json").write_text(
        json.dumps({"name": "myproject", "description": "test"})
    )
    return user_dir


# ---------------------------------------------------------------------------
# Phase 3 tests
# ---------------------------------------------------------------------------


class TestGenerateShareKey:
    def test_returns_expected_fields(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        result = shares.generate_share_key(
            owner_user_dir=owner_dir,
            project="myproject",
            grantee="bob",
            write_access=False,
        )
        assert result["key_id"].startswith("sk_")
        assert result["project"] == "myproject"
        assert result["grantee"] == "bob"
        assert result["write_access"] is False
        assert result["owner"] == "alice"
        assert "share_string" in result
        assert isinstance(result["share_string"], str)
        assert len(result["share_string"]) > 20

    def test_key_id_has_expected_prefix(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        result = shares.generate_share_key(owner_dir, "myproject", "bob")
        assert result["key_id"].startswith("sk_")

    def test_write_access_true(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        result = shares.generate_share_key(owner_dir, "myproject", "bob", write_access=True)
        assert result["write_access"] is True

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
    def test_wrong_project_raises_on_redemption(self, tmp_path):
        """A share string tampered to claim a different project should fail HMAC."""
        import base64

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        # Create another project for the tamper attempt
        (owner_dir / "otherproject" / "chroma_data").mkdir(parents=True, exist_ok=True)
        (owner_dir / "otherproject" / "meta.json").write_text(
            json.dumps({"name": "otherproject", "description": "other"})
        )

        grantee_dir = _make_user_dir(tmp_path, "bob")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        # Decode the share string and tamper the project field
        raw = base64.urlsafe_b64decode(gen["share_string"].encode()).decode()
        parts = raw.split(":", 4)
        # parts: [key_id, token, owner, project, owner_store_path]
        parts[3] = "otherproject"  # tamper the project name
        tampered = base64.urlsafe_b64encode(":".join(parts).encode()).decode()

        with pytest.raises(ValueError):
            shares.redeem_share_key(grantee_dir, tampered)


@pytest.mark.skipif(sys.platform != "linux", reason="AxonStore sharing is Linux-only")
class TestRedeemShareKey:
    def test_creates_symlink(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert result["mount_name"] == "alice_myproject"
        assert result["owner"] == "alice"
        assert result["project"] == "myproject"

        link_path = Path(result["mount_path"])
        assert link_path.is_symlink()
        assert link_path.resolve() == (owner_dir / "myproject").resolve()

    def test_invalid_share_string_raises(self, tmp_path):
        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")
        with pytest.raises(ValueError, match="Invalid share_string"):
            shares.redeem_share_key(grantee_dir, "not_a_valid_base64_string!!!")

    def test_revoked_key_raises_on_redemption(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.revoke_share_key(owner_dir, gen["key_id"])

        with pytest.raises(ValueError, match="revoked"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])


@pytest.mark.skipif(sys.platform != "linux", reason="AxonStore sharing is Linux-only")
class TestValidateReceivedShares:
    def test_removes_stale_symlinks(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        redeem_result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Verify symlink exists
        link = Path(redeem_result["mount_path"])
        assert link.is_symlink()

        # Revoke the key
        shares.revoke_share_key(owner_dir, gen["key_id"])

        # Validate should remove the symlink
        removed = shares.validate_received_shares(grantee_dir)
        assert "alice_myproject" in removed
        assert not link.exists()

    def test_nothing_removed_if_not_revoked(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Don't revoke — validate should return empty list
        removed = shares.validate_received_shares(grantee_dir)
        assert removed == []
