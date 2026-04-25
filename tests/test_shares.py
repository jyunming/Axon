"""Tests for AxonStore share key management (axon/shares.py)."""


import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------


# Helpers


# ---------------------------------------------------------------------------


def _make_user_dir(tmp_path: Path, username: str) -> Path:
    """Create a minimal AxonStore user directory for testing."""

    user_dir = tmp_path / "AxonStore" / username

    (user_dir / ".shares").mkdir(parents=True, exist_ok=True)

    # Create a dummy project so redeem can find it

    (user_dir / "myproject" / "vector_store_data").mkdir(parents=True, exist_ok=True)

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
    def test_wrong_project_raises_on_redemption(self, tmp_path):
        """A share string tampered to claim a different project should fail HMAC."""

        import base64

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        # Create another project for the tamper attempt

        (owner_dir / "otherproject" / "vector_store_data").mkdir(parents=True, exist_ok=True)

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

        owner_store = str(tmp_path / "AxonStore")

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

        raw = f"sk_unknown:faketoken:alice:myproject:{str(tmp_path / 'AxonStore')}"

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

        raw = f"{key_id}:faketoken:alice:myproject:{str(tmp_path / 'AxonStore')}"

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

        raw = f"{key_id}:faketoken:alice:myproject:{str(tmp_path / 'AxonStore')}"

        share_string = base64.urlsafe_b64encode(raw.encode()).decode()

        with pytest.raises(ValueError, match="not found in owner"):
            shares.redeem_share_key(grantee_dir, share_string)


class TestRedeemShareKey:
    def test_creates_descriptor(self, tmp_path):
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

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

    def test_revoked_key_raises_on_redemption(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        shares.revoke_share_key(owner_dir, gen["key_id"])

        with pytest.raises(ValueError, match="revoked"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])


class TestValidateReceivedShares:
    def test_removes_stale_descriptor_on_revocation(self, tmp_path):
        from axon import shares
        from axon.mounts import load_mount_descriptor

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        shares.redeem_share_key(grantee_dir, gen["share_string"])

        # Verify descriptor exists

        assert load_mount_descriptor(grantee_dir, "alice_myproject") is not None

        # Revoke the key

        shares.revoke_share_key(owner_dir, gen["key_id"])

        # Validate should remove the descriptor

        removed = shares.validate_received_shares(grantee_dir)

        assert "alice_myproject" in removed

        assert load_mount_descriptor(grantee_dir, "alice_myproject") is None

    def test_nothing_removed_if_not_revoked(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

        grantee_dir = _make_user_dir(tmp_path, "bob")

        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

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

    def test_tombstone_added_when_key_absent_from_manifest(self, tmp_path):
        """If the manifest is missing the key_id record, a tombstone is written so


        redeem_share_key() will correctly reject the key."""

        import json as _json

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")

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


# ---------------------------------------------------------------------------
# Issue #54 — share-key expiry & extend
# ---------------------------------------------------------------------------


class TestShareKeyExpiry:
    """Coverage for the optional ``ttl_days`` / ``expires_at`` lifecycle
    added in #54: generation, redemption, query-path enforcement, and
    grantee-side cleanup via validate_received_shares."""

    def test_no_ttl_means_no_expiry_field(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        result = shares.generate_share_key(owner_dir, "myproject", "bob")
        assert result["expires_at"] is None
        # Manifest record also has expires_at = None for forward-compat.
        manifest = json.loads((owner_dir / ".shares" / ".share_manifest.json").read_text())
        rec = manifest["issued"][0]
        assert "expires_at" in rec and rec["expires_at"] is None

    def test_ttl_days_sets_future_expiry(self, tmp_path):
        from datetime import datetime, timezone

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        result = shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=7)
        assert result["expires_at"]
        exp = datetime.fromisoformat(result["expires_at"])
        # Roughly 7 days from now (allow a small clock-skew window).
        delta = (exp - datetime.now(timezone.utc)).total_seconds()
        assert 6 * 86400 < delta <= 7 * 86400 + 60

    def test_ttl_days_invalid_value_rejected(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        with pytest.raises(ValueError, match="positive integer"):
            shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=0)
        with pytest.raises(ValueError, match="positive integer"):
            shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=-3)

    def test_redeem_rejects_expired_key(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = tmp_path / "AxonStore" / "bob"
        grantee_dir.mkdir(parents=True)

        gen = shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=7)

        # Force the manifest record into the past.
        manifest_path = owner_dir / ".shares" / ".share_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["issued"][0]["expires_at"] = "2020-01-01T00:00:00+00:00"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="expired"):
            shares.redeem_share_key(grantee_dir, gen["share_string"])

    def test_validate_received_shares_removes_expired(self, tmp_path):
        from axon import shares
        from axon.mounts import mount_descriptor_path

        owner_dir = _make_user_dir(tmp_path, "alice")
        grantee_dir = tmp_path / "AxonStore" / "bob"
        grantee_dir.mkdir(parents=True)

        gen = shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=30)
        red = shares.redeem_share_key(grantee_dir, gen["share_string"])
        mount_name = red["mount_name"]
        # Sanity: descriptor exists.
        assert mount_descriptor_path(grantee_dir, mount_name).exists()

        # Force expiry in owner manifest.
        manifest_path = owner_dir / ".shares" / ".share_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["issued"][0]["expires_at"] = "2020-01-01T00:00:00+00:00"
        manifest_path.write_text(json.dumps(manifest))

        removed = shares.validate_received_shares(grantee_dir)
        assert mount_name in removed
        assert not mount_descriptor_path(grantee_dir, mount_name).exists()

    def test_is_expired_helper_handles_edge_cases(self):
        from axon.shares import _is_expired

        # None / empty: never expired.
        assert _is_expired(None) is False
        assert _is_expired("") is False
        # Garbage timestamp: never expired (defensive — don't break access on malformed data).
        assert _is_expired("not-a-timestamp") is False
        # Past: expired.
        assert _is_expired("2020-01-01T00:00:00+00:00") is True
        # Far future: not expired.
        assert _is_expired("2099-01-01T00:00:00+00:00") is False
        # Naive timestamp (no tz) is treated as UTC.
        assert _is_expired("2020-01-01T00:00:00") is True


class TestExtendShareKey:
    def test_extend_updates_expiry_in_both_files(self, tmp_path):
        from datetime import datetime, timezone

        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=1)

        result = shares.extend_share_key(owner_dir, gen["key_id"], ttl_days=14)
        assert result["expires_at"]
        new_exp = datetime.fromisoformat(result["expires_at"])
        delta = (new_exp - datetime.now(timezone.utc)).total_seconds()
        assert 13 * 86400 < delta <= 14 * 86400 + 60

        # Both files updated.
        keys = json.loads((owner_dir / ".shares" / ".share_keys.json").read_text())
        manifest = json.loads((owner_dir / ".shares" / ".share_manifest.json").read_text())
        assert keys["issued"][0]["expires_at"] == result["expires_at"]
        assert manifest["issued"][0]["expires_at"] == result["expires_at"]

    def test_extend_with_none_clears_expiry(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=1)

        result = shares.extend_share_key(owner_dir, gen["key_id"], ttl_days=None)
        assert result["expires_at"] is None
        manifest = json.loads((owner_dir / ".shares" / ".share_manifest.json").read_text())
        assert manifest["issued"][0]["expires_at"] is None

    def test_extend_unknown_key_raises_value_error(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        with pytest.raises(ValueError, match="not found"):
            shares.extend_share_key(owner_dir, "sk_deadbeef", ttl_days=7)

    def test_extend_revoked_key_refused(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob", ttl_days=7)
        shares.revoke_share_key(owner_dir, gen["key_id"])

        with pytest.raises(ValueError, match="revoked"):
            shares.extend_share_key(owner_dir, gen["key_id"], ttl_days=14)

    def test_extend_invalid_ttl_rejected(self, tmp_path):
        from axon import shares

        owner_dir = _make_user_dir(tmp_path, "alice")
        gen = shares.generate_share_key(owner_dir, "myproject", "bob")

        with pytest.raises(ValueError, match="positive integer"):
            shares.extend_share_key(owner_dir, gen["key_id"], ttl_days=0)
        with pytest.raises(ValueError, match="positive integer"):
            shares.extend_share_key(owner_dir, gen["key_id"], ttl_days=-1)
