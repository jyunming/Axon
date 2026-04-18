"""Cross-team knowledge sharing integration tests for axon/shares.py.

Covers scenarios NOT already in test_shares.py:
- Full sharing round-trip (generate → redeem → list_mount_descriptors)
- Received shares appearing in list_shares()
- Revocation: validate_received_shares clears the received key record
- Tampered key: corrupted base64, tampered token, wrong grantee, tampered key_id
- No TTL/expiry (shares have no expiry field)
- Re-redemption idempotency (single descriptor, single record)
- Share key for non-existent project (generate succeeds, redeem fails)
- Hierarchical project share (nested subs/ layout)
- validate_received_shares edge cases (missing manifest, multiple keys, empty)
- List shares: both-direction coverage including received entries
- Permission levels: descriptor readonly flag, write block, read allowed
"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_dir(tmp_path: Path, username: str) -> Path:
    """Create a minimal AxonStore user directory under tmp_path/AxonStore/<username>."""
    user_dir = tmp_path / "AxonStore" / username
    (user_dir / ".shares").mkdir(parents=True, exist_ok=True)
    (user_dir / "myproject" / "vector_store_data").mkdir(parents=True, exist_ok=True)
    (user_dir / "myproject" / "meta.json").write_text(
        json.dumps({"name": "myproject", "project_id": "proj-abc123"}),
        encoding="utf-8",
    )
    return user_dir


def _make_nested_project(owner_dir: Path, project_path: str) -> None:
    """Create a nested project directory using the subs/ layout.

    'research/papers' → owner_dir/research/subs/papers/
    """
    segments = project_path.split("/")
    project_dir = owner_dir / segments[0]
    for seg in segments[1:]:
        project_dir = project_dir / "subs" / seg
    (project_dir / "vector_store_data").mkdir(parents=True, exist_ok=True)
    (project_dir / "meta.json").write_text(
        json.dumps({"name": segments[-1], "project_id": f"proj-{segments[-1]}"}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Full sharing round-trip
# ---------------------------------------------------------------------------


class TestFullSharingRoundTrip:
    def test_redeem_descriptor_is_readonly(self, tmp_path):
        """Redeemed descriptor always has readonly=True."""
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        desc = load_mount_descriptor(grantee_dir, "alice_myproject")
        assert desc is not None
        assert desc["readonly"] is True

    def test_redeem_descriptor_appears_in_list_mount_descriptors(self, tmp_path):
        """After redemption, list_mount_descriptors returns the active mount."""
        from axon import shares
        from axon.mounts import list_mount_descriptors

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        mounts = list_mount_descriptors(grantee_dir)
        assert len(mounts) == 1
        assert mounts[0]["mount_name"] == "alice_myproject"
        assert mounts[0]["state"] == "active"
        assert mounts[0]["revoked"] is False

    def test_list_shares_received_populated_after_redeem(self, tmp_path):
        """After redemption, grantee's list_shares()['shared'] contains the entry."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        result = shares.list_shares(grantee_dir)
        assert len(result["shared"]) == 1
        entry = result["shared"][0]
        assert entry["owner"] == "alice"
        assert entry["project"] == "myproject"
        assert entry["mount"] == "alice_myproject"
        assert entry["redeemed_at"] is not None
        assert entry["key_id"] == gen["key_id"]

    def test_redeem_sets_correct_target_project_dir(self, tmp_path):
        """Descriptor target_project_dir points to owner's project directory."""
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        desc = load_mount_descriptor(grantee_dir, "alice_myproject")
        assert desc["target_project_dir"] == str(owner_dir / "myproject")

    def test_redeem_stores_share_key_id_in_descriptor(self, tmp_path):
        """Descriptor's share_key_id matches the issued key_id."""
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        desc = load_mount_descriptor(grantee_dir, "alice_myproject")
        assert desc["share_key_id"] == gen["key_id"]


# ---------------------------------------------------------------------------
# Revocation round-trip
# ---------------------------------------------------------------------------


class TestRevocationRoundTrip:
    def test_validate_clears_received_key_record_from_keys_file(self, tmp_path):
        """After revocation + validate, the received record is removed from grantee's keys file."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        keys = json.loads((grantee_dir / ".shares" / ".share_keys.json").read_text())
        assert len(keys["received"]) == 1

        shares.revoke_share_key(owner_dir, gen["key_id"])
        shares.validate_received_shares(grantee_dir)

        keys = json.loads((grantee_dir / ".shares" / ".share_keys.json").read_text())
        assert len(keys["received"]) == 0

    def test_list_shares_shared_empty_after_revoke_and_validate(self, tmp_path):
        """After revocation + validate, grantee's list_shares()['shared'] is empty."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])
        shares.revoke_share_key(owner_dir, gen["key_id"])
        shares.validate_received_shares(grantee_dir)

        result = shares.list_shares(grantee_dir)
        assert result["shared"] == []

    def test_received_entry_persists_until_validate_called(self, tmp_path):
        """Grantee's received entry is still present before validate is called, even after owner revokes."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])
        shares.revoke_share_key(owner_dir, gen["key_id"])

        # Before validate — received record still in keys file
        keys = json.loads((grantee_dir / ".shares" / ".share_keys.json").read_text())
        assert len(keys["received"]) == 1


# ---------------------------------------------------------------------------
# Tampered key tests
# ---------------------------------------------------------------------------


class TestTamperedKey:
    def test_corrupted_base64_raises_invalid_format(self, tmp_path):
        """Completely invalid base64 raises 'Invalid share_string' ValueError."""
        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")
        with pytest.raises(ValueError, match="Invalid share_string"):
            shares.redeem_share_key(grantee_dir, "not!!valid!!base64???")

    def test_tampered_token_raises_hmac_error(self, tmp_path):
        """Replacing the token with arbitrary bytes fails HMAC verification."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        raw = base64.urlsafe_b64decode(gen["share_string"].encode()).decode()
        parts = raw.split(":", 4)
        parts[1] = "a" * 64  # replace token with garbage
        tampered = base64.urlsafe_b64encode(":".join(parts).encode()).decode()

        with pytest.raises(ValueError, match="HMAC"):
            shares.redeem_share_key(grantee_dir, tampered)

    def test_wrong_grantee_raises_hmac_error(self, tmp_path):
        """Share generated for 'bob' cannot be redeemed by 'charlie' — HMAC mismatch."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        # Share issued to "bob", but grantee_dir.name = "charlie"
        grantee_dir = _make_user_dir(tmp_path, "charlie")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        with pytest.raises(ValueError, match="HMAC"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])

    def test_tampered_key_id_raises_not_found(self, tmp_path):
        """Changing key_id in share_string causes 'not found in owner' error."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        raw = base64.urlsafe_b64decode(gen["share_string"].encode()).decode()
        parts = raw.split(":", 4)
        parts[0] = "sk_ffffffff"  # replace key_id with a fake one
        tampered = base64.urlsafe_b64encode(":".join(parts).encode()).decode()

        with pytest.raises(ValueError, match="not found in owner"):
            shares.redeem_share_key(grantee_dir, tampered)

    def test_truncated_share_string_raises_invalid_format(self, tmp_path):
        """A base64 string that decodes fine but has too few fields raises ValueError."""
        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")
        # base64 of a valid-looking string with only 3 colon-separated fields
        raw = "sk_abc:sometoken:alice"
        short = base64.urlsafe_b64encode(raw.encode()).decode()

        with pytest.raises(ValueError, match="Invalid share_string"):
            shares.redeem_share_key(grantee_dir, short)


# ---------------------------------------------------------------------------
# No TTL / expiry — document that shares have no expiry field
# ---------------------------------------------------------------------------


class TestNoExpiry:
    def test_generated_key_has_no_expires_at(self, tmp_path):
        """Shares currently have no expiry — generated result has no expires_at field."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        assert "expires_at" not in gen

    def test_key_store_record_has_no_expires_at(self, tmp_path):
        """The key store record also has no expires_at field."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        shares.generate_share_key(owner_dir, "myproject", "bob")

        keys = json.loads((owner_dir / ".shares" / ".share_keys.json").read_text())
        assert "expires_at" not in keys["issued"][0]


# ---------------------------------------------------------------------------
# Re-redemption idempotency
# ---------------------------------------------------------------------------


class TestReRedemption:
    def test_redeem_twice_creates_single_descriptor(self, tmp_path):
        """Second redemption of the same key does not create a duplicate mount descriptor."""
        from axon import shares
        from axon.mounts import list_mount_descriptors

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        mounts = list_mount_descriptors(grantee_dir)
        assert len(mounts) == 1

    def test_redeem_twice_creates_single_received_record(self, tmp_path):
        """Second redemption replaces the existing received record (no duplicates)."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        keys = json.loads((grantee_dir / ".shares" / ".share_keys.json").read_text())
        assert len(keys["received"]) == 1

    def test_redeem_twice_returns_same_mount_name(self, tmp_path):
        """Both redemptions return the same mount_name."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        result1 = shares.redeem_share_key(grantee_dir, gen["share_string"])
        result2 = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert result1["mount_name"] == result2["mount_name"]


# ---------------------------------------------------------------------------
# Share key for non-existent project
# ---------------------------------------------------------------------------


class TestShareNonExistentProject:
    def test_generate_share_key_for_missing_project_succeeds(self, tmp_path):
        """generate_share_key does not validate project existence at generation time."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        # "ghostproject" does not exist as a directory
        result = shares.generate_share_key(owner_dir, "ghostproject", "bob")
        assert result["key_id"].startswith("sk_")
        assert result["project"] == "ghostproject"

    def test_redeem_for_missing_project_raises(self, tmp_path):
        """Redemption fails with 'does not exist' when the project dir is absent."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "ghostproject", "bob")
        with pytest.raises(ValueError, match="does not exist"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])

    def test_generate_for_missing_project_key_is_recorded_in_store(self, tmp_path):
        """Even for a missing project, the key is recorded in the owner's key store."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "ghostproject", "bob")

        keys = json.loads((owner_dir / ".shares" / ".share_keys.json").read_text())
        assert any(r["key_id"] == gen["key_id"] for r in keys["issued"])


# ---------------------------------------------------------------------------
# Hierarchical project share
# ---------------------------------------------------------------------------


class TestHierarchicalProjectShare:
    def test_redeem_nested_project_creates_valid_descriptor(self, tmp_path):
        """Sharing a nested project (research/papers) creates a correct descriptor."""
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        _make_nested_project(owner_dir, "research/papers")

        gen = shares.generate_share_key(owner_dir, "research/papers", "bob")
        result = shares.redeem_share_key(grantee_dir, gen["share_string"])

        assert result["mount_name"] == "alice_research_papers"
        desc = load_mount_descriptor(grantee_dir, "alice_research_papers")
        assert desc is not None
        assert desc["project"] == "research/papers"
        assert desc["state"] == "active"
        assert desc["readonly"] is True

    def test_redeem_nested_project_target_dir_uses_subs_layout(self, tmp_path):
        """Descriptor target_project_dir follows the subs/ directory layout."""
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        _make_nested_project(owner_dir, "research/papers")

        gen = shares.generate_share_key(owner_dir, "research/papers", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        desc = load_mount_descriptor(grantee_dir, "alice_research_papers")
        expected = str(owner_dir / "research" / "subs" / "papers")
        assert desc["target_project_dir"] == expected

    def test_redeem_nested_project_missing_dir_raises(self, tmp_path):
        """Nested project redemption fails if subs/ layout path doesn't exist."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        # Don't create the nested directory

        gen = shares.generate_share_key(owner_dir, "research/papers", "bob")
        with pytest.raises(ValueError, match="does not exist"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])


# ---------------------------------------------------------------------------
# validate_received_shares edge cases
# ---------------------------------------------------------------------------


class TestValidateReceivedSharesEdgeCases:
    def test_validate_skips_missing_manifest_path(self, tmp_path):
        """When owner_manifest_path doesn't exist, validate silently skips and preserves record."""
        from axon import shares

        grantee_dir = _make_user_dir(tmp_path, "bob")
        keys_path = grantee_dir / ".shares" / ".share_keys.json"
        keys_path.write_text(
            json.dumps(
                {
                    "received": [
                        {
                            "key_id": "sk_gone",
                            "owner": "alice",
                            "project": "myproject",
                            "mount_name": "alice_myproject",
                            "owner_manifest_path": str(tmp_path / "nonexistent" / "manifest.json"),
                            "redeemed_at": "2026-01-01T00:00:00+00:00",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        removed = shares.validate_received_shares(grantee_dir)
        assert removed == []
        # Record preserved — can't confirm revocation without manifest
        keys = json.loads(keys_path.read_text())
        assert len(keys["received"]) == 1

    def test_validate_multiple_keys_removes_only_revoked(self, tmp_path):
        """With two received shares, only the revoked one's mount is removed."""
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        charlie_dir = _make_user_dir(tmp_path, "charlie")
        # Give charlie a second project to share
        (charlie_dir / "otherproj" / "vector_store_data").mkdir(parents=True, exist_ok=True)
        (charlie_dir / "otherproj" / "meta.json").write_text(
            json.dumps({"name": "otherproj"}), encoding="utf-8"
        )
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen1 = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen1["share_string"])

        gen2 = shares.generate_share_key(charlie_dir, "otherproj", "bob")
        shares.redeem_share_key(grantee_dir, gen2["share_string"])

        # Revoke only alice's key
        shares.revoke_share_key(owner_dir, gen1["key_id"])
        removed = shares.validate_received_shares(grantee_dir)

        assert removed == ["alice_myproject"]
        assert load_mount_descriptor(grantee_dir, "alice_myproject") is None
        assert load_mount_descriptor(grantee_dir, "charlie_otherproj") is not None

    def test_validate_no_received_records_returns_empty(self, tmp_path):
        """validate_received_shares on a user with no received keys returns []."""
        from axon import shares

        user_dir = _make_user_dir(tmp_path, "alice")
        removed = shares.validate_received_shares(user_dir)
        assert removed == []

    def test_validate_corrupt_manifest_is_silently_skipped(self, tmp_path):
        """Corrupt manifest JSON causes the received record to be silently skipped (no raise)."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Corrupt the owner's manifest file
        manifest_path = owner_dir / ".shares" / ".share_manifest.json"
        manifest_path.write_text("{ invalid json !!!", encoding="utf-8")

        removed = shares.validate_received_shares(grantee_dir)
        assert removed == []

    def test_validate_returns_removed_mount_names(self, tmp_path):
        """validate_received_shares returns the list of removed mount names."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])
        shares.revoke_share_key(owner_dir, gen["key_id"])

        removed = shares.validate_received_shares(grantee_dir)
        assert "alice_myproject" in removed


# ---------------------------------------------------------------------------
# List shares — full both-direction coverage
# ---------------------------------------------------------------------------


class TestListSharesBothDirections:
    def test_owner_sees_grantee_in_sharing_list(self, tmp_path):
        """Owner's list_shares()['sharing'] contains the issued key with correct grantee."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        shares.generate_share_key(owner_dir, "myproject", "bob")

        result = shares.list_shares(owner_dir)
        assert result["sharing"][0]["grantee"] == "bob"
        assert result["sharing"][0]["revoked"] is False
        assert result["sharing"][0]["project"] == "myproject"

    def test_grantee_sees_owner_in_shared_list(self, tmp_path):
        """Grantee's list_shares()['shared'] contains the received entry with correct owner."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        result = shares.list_shares(grantee_dir)
        assert len(result["shared"]) == 1
        assert result["shared"][0]["owner"] == "alice"
        assert result["shared"][0]["project"] == "myproject"
        assert result["shared"][0]["key_id"] == gen["key_id"]

    def test_multiple_grantees_all_appear_in_sharing(self, tmp_path):
        """Owner who shared with two grantees sees both entries in sharing list."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.generate_share_key(owner_dir, "myproject", "charlie")

        result = shares.list_shares(owner_dir)
        assert len(result["sharing"]) == 2
        grantees = {r["grantee"] for r in result["sharing"]}
        assert grantees == {"bob", "charlie"}

    def test_revoked_key_still_in_sharing_with_revoked_true(self, tmp_path):
        """Revoked key still appears in sharing list with revoked=True (not purged)."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.revoke_share_key(owner_dir, gen["key_id"])

        result = shares.list_shares(owner_dir)
        assert result["sharing"][0]["revoked"] is True

    def test_sharing_list_empty_shared_list_nonempty_for_grantee(self, tmp_path):
        """Grantee who received but never issued has empty sharing and nonempty shared."""
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        result = shares.list_shares(grantee_dir)
        assert result["sharing"] == []
        assert len(result["shared"]) == 1


# ---------------------------------------------------------------------------
# Permission levels
# ---------------------------------------------------------------------------


class TestPermissionLevels:
    def test_mounted_descriptor_always_readonly_flag(self, tmp_path):
        """create_mount_descriptor always sets readonly=True."""
        from axon.mounts import create_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        project_dir = owner_dir / "myproject"

        desc = create_mount_descriptor(
            grantee_user_dir=grantee_dir,
            mount_name="alice_myproject",
            owner="alice",
            project="myproject",
            owner_user_dir=owner_dir,
            target_project_dir=project_dir,
            share_key_id="sk_test",
        )
        assert desc["readonly"] is True

    def test_write_blocked_on_mounted_share(self):
        """AxonBrain._assert_write_allowed raises PermissionError when project is a mounted share."""
        from axon.main import AxonBrain

        brain = MagicMock()
        brain._is_mounted_share = AxonBrain._is_mounted_share.__get__(brain, type(brain))
        brain._assert_write_allowed = AxonBrain._assert_write_allowed.__get__(brain, type(brain))
        brain._active_project = "mounts/alice_myproject"
        brain._read_only_scope = False
        brain._mounted_share = True
        brain._active_project_kind = "mounted"

        with pytest.raises(PermissionError, match="mounted share"):
            brain._assert_write_allowed("ingest")

    def test_grantee_mount_passes_validate_mount_descriptor(self, tmp_path):
        """After redeem, validate_mount_descriptor returns (True, '') for the active mount."""
        from axon import shares
        from axon.mounts import list_mount_descriptors, validate_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        mounts = list_mount_descriptors(grantee_dir)
        assert len(mounts) == 1
        valid, reason = validate_mount_descriptor(mounts[0])
        assert valid is True
        assert reason == ""

    def test_revoked_mount_fails_validate_mount_descriptor(self, tmp_path):
        """After revocation and validate, the removed descriptor can't be found — validates False."""
        from axon import shares
        from axon.mounts import validate_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = _make_user_dir(tmp_path, "bob")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")
        shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Revoke, then patch the descriptor manually to simulate a stale descriptor
        shares.revoke_share_key(owner_dir, gen["key_id"])
        # Simulate a descriptor that hasn't been cleaned up yet but is marked revoked
        stale_desc = {
            "mount_name": "alice_myproject",
            "state": "active",
            "revoked": True,
            "target_project_dir": str(owner_dir / "myproject"),
        }
        valid, reason = validate_mount_descriptor(stale_desc)
        assert valid is False
        assert "revoked" in reason
