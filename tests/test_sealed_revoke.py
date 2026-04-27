"""Phase 4: sealed-share revocation tests (soft + hard).

Covers ``axon.security.share.revoke_sealed_share`` and the wired
``revoke_sealed_share`` entry point in ``axon.security``.

Skips when ``cryptography`` / ``keyring`` aren't installed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon import security as _security  # noqa: E402
from axon.security.master import bootstrap_store, lock_store  # noqa: E402
from axon.security.seal import project_seal, read_sealed_marker  # noqa: E402
from axon.security.share import (  # noqa: E402
    generate_sealed_share,
    get_grantee_dek,
    list_sealed_share_key_ids,
    redeem_sealed_share,
    revoke_sealed_share,
    share_kek_path,
    share_wrap_path,
)

# ---------------------------------------------------------------------------
# In-memory keyring fixture
# ---------------------------------------------------------------------------


class _InMemoryKeyring:
    priority = 1

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, secret):
        self._store[(service, username)] = secret

    def get_password(self, service, username):
        return self._store.get((service, username))

    def delete_password(self, service, username):
        import keyring.errors

        if (service, username) not in self._store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


@pytest.fixture
def kr_backend():
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


@pytest.fixture
def owner_user_dir(tmp_path):
    store = tmp_path / "AxonStore" / "alice"
    store.mkdir(parents=True)
    return store


@pytest.fixture
def grantee_user_dir(tmp_path):
    store = tmp_path / "AxonStore" / "bob"
    store.mkdir(parents=True)
    return store


def _populate_and_seal(owner_user_dir: Path, project: str = "research") -> Path:
    proj = owner_user_dir / project
    (proj / "bm25_index").mkdir(parents=True)
    (proj / "vector_store_data").mkdir(parents=True)
    (proj / ".security").mkdir(parents=True)
    (proj / "meta.json").write_text('{"project_id":"p1","name":"research"}', encoding="utf-8")
    (proj / "version.json").write_text('{"seq":1}', encoding="utf-8")
    (proj / "bm25_index" / ".bm25_log.jsonl").write_text('{"id":"d1"}\n', encoding="utf-8")
    (proj / "vector_store_data" / "manifest.json").write_text('{"d":768}', encoding="utf-8")
    (proj / "vector_store_data" / "seg-00000001.bin").write_bytes(b"\xab" * 4096)

    bootstrap_store(owner_user_dir, "owner-pw")
    project_seal(project, owner_user_dir)
    return proj


# ---------------------------------------------------------------------------
# list_sealed_share_key_ids
# ---------------------------------------------------------------------------


class TestListSealedShareKeyIds:
    def test_empty_when_no_shares(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        assert list_sealed_share_key_ids(proj) == []

    def test_returns_all_active_share_key_ids(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        for kid in ("ssk_alpha", "ssk_beta", "ssk_gamma"):
            generate_sealed_share(owner_user_dir, "research", kid, key_id=kid)
        ids = list_sealed_share_key_ids(proj)
        assert sorted(ids) == ["ssk_alpha", "ssk_beta", "ssk_gamma"]

    def test_returns_empty_for_missing_dir(self, tmp_path):
        # No .security/shares/ at all.
        assert list_sealed_share_key_ids(tmp_path / "no_proj") == []


# ---------------------------------------------------------------------------
# Soft revoke
# ---------------------------------------------------------------------------


class TestSoftRevoke:
    def test_deletes_wrap_file(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_soft01")
        assert share_wrap_path(proj, "ssk_soft01").is_file()
        result = revoke_sealed_share(owner_user_dir, "research", "ssk_soft01")
        assert result["status"] == "soft_revoked"
        assert result["rotate"] is False
        assert result["wrap_deleted"] is True
        assert not share_wrap_path(proj, "ssk_soft01").is_file()

    def test_other_shares_remain(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_keep01")
        generate_sealed_share(owner_user_dir, "research", "carol", key_id="ssk_drop01")
        revoke_sealed_share(owner_user_dir, "research", "ssk_drop01")
        assert share_wrap_path(proj, "ssk_keep01").is_file()
        assert not share_wrap_path(proj, "ssk_drop01").is_file()

    def test_fresh_redeem_after_soft_revoke_fails(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_revfresh")
        revoke_sealed_share(owner_user_dir, "research", "ssk_revfresh")
        with pytest.raises(_security.SecurityError, match="missing at"):
            redeem_sealed_share(grantee_user_dir, share["share_string"])

    def test_cached_grantee_dek_still_valid_after_soft_revoke(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """Document the soft-revoke trade-off: a grantee who already
        cached the DEK in their keyring keeps it after soft revoke.
        Hard rotate is needed to invalidate cached DEKs.
        """
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_keep_cached")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        cached_before = get_grantee_dek("ssk_keep_cached")

        revoke_sealed_share(owner_user_dir, "research", "ssk_keep_cached")

        # Still in keyring — soft revoke does NOT touch the grantee's keyring.
        cached_after = get_grantee_dek("ssk_keep_cached")
        assert cached_after == cached_before

    def test_revoke_missing_key_raises(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="No sealed-share wrap"):
            revoke_sealed_share(owner_user_dir, "research", "ssk_never")

    def test_revoke_unsealed_project_raises(self, kr_backend, owner_user_dir):
        bootstrap_store(owner_user_dir, "test-pass-ok")
        (owner_user_dir / "open").mkdir()
        (owner_user_dir / "open" / "meta.json").write_text("{}", encoding="utf-8")
        with pytest.raises(_security.SecurityError, match="not sealed"):
            revoke_sealed_share(owner_user_dir, "open", "ssk_x")

    def test_revoke_invalid_key_id_rejected(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="Invalid key_id"):
            revoke_sealed_share(owner_user_dir, "research", "../escape")


# ---------------------------------------------------------------------------
# Hard revoke (DEK rotate + re-encrypt + invalidate all shares)
# ---------------------------------------------------------------------------


class TestHardRevoke:
    def test_rotates_dek_and_reseals_files(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        marker_before = read_sealed_marker(proj)
        seal_id_before = marker_before["seal_id"]
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_hard01")

        result = revoke_sealed_share(owner_user_dir, "research", "ssk_hard01", rotate=True)
        assert result["status"] == "hard_revoked"
        assert result["rotate"] is True
        assert result["files_resealed"] >= 4
        assert result["new_seal_id"] != seal_id_before

        marker_after = read_sealed_marker(proj)
        assert marker_after["seal_id"] == result["new_seal_id"]

    def test_invalidates_only_revoked_share_survivors_rewrapped(self, kr_backend, owner_user_dir):
        """Hard revoke only invalidates the trigger share (+ legacy
        survivors without .kek files). Surviving shares that have .kek
        files are selectively re-wrapped: their wraps survive under the
        new DEK, so grantees do NOT need to re-redeem."""
        proj = _populate_and_seal(owner_user_dir)
        for kid in ("ssk_h_a", "ssk_h_b", "ssk_h_c"):
            generate_sealed_share(owner_user_dir, "research", kid, key_id=kid)
        result = revoke_sealed_share(owner_user_dir, "research", "ssk_h_a", rotate=True)
        # Only the revoked key_id needs re-issuing (others were re-wrapped).
        assert result["invalidated_share_key_ids"] == ["ssk_h_a"]
        # The revoked wrap is gone; survivors' wraps are still present.
        assert not share_wrap_path(proj, "ssk_h_a").is_file()
        assert share_wrap_path(proj, "ssk_h_b").is_file()
        assert share_wrap_path(proj, "ssk_h_c").is_file()
        assert sorted(list_sealed_share_key_ids(proj)) == ["ssk_h_b", "ssk_h_c"]

    def test_existing_share_string_no_longer_redeems(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """After hard rotate, the share_string the grantee was given is
        useless — no wrap file at the expected key_id path."""
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_hard_inv")
        revoke_sealed_share(owner_user_dir, "research", "ssk_hard_inv", rotate=True)
        with pytest.raises(_security.SecurityError, match="missing at"):
            redeem_sealed_share(grantee_user_dir, share["share_string"])

    def test_owner_can_still_open_after_rotate(self, kr_backend, owner_user_dir):
        """Owner's master is unchanged → owner's get_project_dek still works
        and decrypts the new ciphertext (because we updated dek.wrapped
        AND re-encrypted everything atomically).
        """
        from axon.security.master import get_project_dek
        from axon.security.mount import materialize_for_read, release_cache

        proj = _populate_and_seal(owner_user_dir)
        revoke_sealed_share(owner_user_dir, "research", "ssk_anything", rotate=True)
        # No share existed — that's fine for this test, the wrap-deleted
        # check is just informational. Verify owner can still open.
        new_dek = get_project_dek(owner_user_dir, proj)
        assert len(new_dek) == 32
        cache = materialize_for_read(proj, owner_user_dir)
        try:
            # Plaintext bytes still match the original.
            assert (
                cache.path / "vector_store_data" / "seg-00000001.bin"
            ).read_bytes() == b"\xab" * 4096
        finally:
            release_cache(cache)

    def test_old_dek_no_longer_decrypts_after_rotate(self, kr_backend, owner_user_dir):
        """The cached DEK a grantee held BEFORE rotate becomes useless
        for new files (AAD seal_id changes, ciphertext is under new DEK).
        Verifies via direct SealedFile.read with the old DEK.
        """
        from cryptography.exceptions import InvalidTag

        from axon.security.crypto import SealedFile, make_aad
        from axon.security.master import get_project_dek

        proj = _populate_and_seal(owner_user_dir)
        old_dek = get_project_dek(owner_user_dir, proj)
        old_marker = read_sealed_marker(proj)
        old_seal_id = old_marker["seal_id"]

        revoke_sealed_share(owner_user_dir, "research", "ssk_anything", rotate=True)

        # Try to read a content file with the OLD dek + OLD seal_id.
        target = proj / "vector_store_data" / "seg-00000001.bin"
        old_aad = make_aad(old_seal_id, "vector_store_data/seg-00000001.bin")
        with pytest.raises(InvalidTag):
            SealedFile.read(target, old_dek, aad=old_aad)

    def test_rotate_locked_store_raises(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        lock_store(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="locked"):
            revoke_sealed_share(owner_user_dir, "research", "ssk_x", rotate=True)

    def test_wrap_deleted_false_when_key_id_never_existed(self, kr_backend, owner_user_dir):
        """For hard revoke the trigger key_id may not correspond to any
        actual share; status dict should reflect that accurately."""
        _populate_and_seal(owner_user_dir)
        result = revoke_sealed_share(owner_user_dir, "research", "ssk_phantom", rotate=True)
        assert result["status"] == "hard_revoked"
        assert result["wrap_deleted"] is False
        assert result["invalidated_share_key_ids"] == []

    def test_wrap_deleted_true_when_key_id_existed(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_real_one")
        result = revoke_sealed_share(owner_user_dir, "research", "ssk_real_one", rotate=True)
        assert result["wrap_deleted"] is True
        assert "ssk_real_one" in result["invalidated_share_key_ids"]

    def test_grantee_can_re_redeem_after_re_issue(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """Documented recovery flow: after hard rotate, owner re-issues
        a share, grantee re-redeems with the new share_string."""
        _populate_and_seal(owner_user_dir)
        share1 = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_orig")
        redeem_sealed_share(grantee_user_dir, share1["share_string"])

        # Hard rotate.
        revoke_sealed_share(owner_user_dir, "research", "ssk_orig", rotate=True)

        # Owner re-issues with a NEW key_id (the old key_id's wrap was
        # nuked along with everything else).
        share2 = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_replacement")
        result = redeem_sealed_share(grantee_user_dir, share2["share_string"])
        assert result["sealed"] is True
        assert result["key_id"] == "ssk_replacement"
        # Grantee can now decrypt with the new DEK.
        new_dek = get_grantee_dek("ssk_replacement")
        assert len(new_dek) == 32


# ---------------------------------------------------------------------------
# Per-share KEK persistence (audit C1)
# ---------------------------------------------------------------------------


class TestShareKekPersistence:
    """Verify that generate_sealed_share writes a .kek sidecar and that
    hard revoke uses it to selectively re-wrap surviving shares."""

    def test_kek_file_created_at_generate_time(self, kr_backend, owner_user_dir):
        """generate_sealed_share must write <key_id>.kek alongside the wrap."""
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_kek01")
        kek_path = share_kek_path(proj, "ssk_kek01")
        assert kek_path.is_file(), ".kek sidecar must exist after generate_sealed_share"
        # The wrapped KEK is 40 bytes (AES-KW of a 32-byte key).
        assert kek_path.stat().st_size == 40

    def test_surviving_share_can_decrypt_after_hard_revoke(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """After hard-revoking one share, a surviving share's wrap is
        re-encrypted under the new DEK so the grantee can still redeem
        without re-issuing the share.
        This is the core guarantee of per-share KEK persistence."""
        _populate_and_seal(owner_user_dir)
        # Bob is the victim; Carol is a surviving grantee.
        share_carol = generate_sealed_share(owner_user_dir, "research", "carol", key_id="ssk_carol")
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_bob")

        # Carol redeems before the revocation.
        redeem_sealed_share(grantee_user_dir, share_carol["share_string"])
        carol_dek_before = get_grantee_dek("ssk_carol")

        # Bob's share is hard-revoked.
        result = revoke_sealed_share(owner_user_dir, "research", "ssk_bob", rotate=True)
        assert result["status"] == "hard_revoked"
        # Carol is NOT in invalidated_share_key_ids (she was re-wrapped).
        assert "ssk_carol" not in result["invalidated_share_key_ids"]
        # Bob is invalidated.
        assert "ssk_bob" in result["invalidated_share_key_ids"]

        # Carol's wrap file still exists (replaced with the re-wrapped version).
        proj = owner_user_dir / "research"
        assert share_wrap_path(proj, "ssk_carol").is_file()
        # Bob's wrap file is gone.
        assert not share_wrap_path(proj, "ssk_bob").is_file()

        # Carol's old share_string still redeems (same key_id, new wrapped DEK).
        # We need to clear her cached DEK first to force a fresh unwrap.
        from axon.security.share import delete_grantee_dek

        delete_grantee_dek("ssk_carol")
        redeem_result = redeem_sealed_share(grantee_user_dir, share_carol["share_string"])
        assert redeem_result["sealed"] is True
        carol_dek_after = get_grantee_dek("ssk_carol")
        # The DEK changed (DEK was rotated).
        assert carol_dek_after != carol_dek_before
        assert len(carol_dek_after) == 32

    def test_kek_file_deleted_on_soft_revoke(self, kr_backend, owner_user_dir):
        """Soft revoke must clean up the .kek sidecar alongside the wrap."""
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_soft_kek")
        kek_path = share_kek_path(proj, "ssk_soft_kek")
        assert kek_path.is_file()
        revoke_sealed_share(owner_user_dir, "research", "ssk_soft_kek")
        assert not kek_path.is_file(), ".kek sidecar must be deleted on soft revoke"

    def test_missing_kek_fallback_invalidates_survivor(self, kr_backend, owner_user_dir):
        """If a surviving share has no .kek file (legacy project), the
        surviving share is invalidated (listed in invalidated_share_key_ids)
        rather than aborting the whole revoke operation."""
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_revoke_me")
        generate_sealed_share(owner_user_dir, "research", "carol", key_id="ssk_legacy")

        # Simulate a legacy share by deleting Carol's .kek sidecar.
        kek_path = share_kek_path(proj, "ssk_legacy")
        kek_path.unlink()
        assert not kek_path.is_file()

        result = revoke_sealed_share(owner_user_dir, "research", "ssk_revoke_me", rotate=True)
        assert result["status"] == "hard_revoked"
        # Both the revoked share AND the legacy-fallback share appear in
        # invalidated_share_key_ids.
        assert "ssk_revoke_me" in result["invalidated_share_key_ids"]
        assert "ssk_legacy" in result["invalidated_share_key_ids"]
        # Both wraps are gone (legacy share was invalidated, not re-wrapped).
        assert not share_wrap_path(proj, "ssk_revoke_me").is_file()
        assert not share_wrap_path(proj, "ssk_legacy").is_file()

    def test_multiple_survivors_all_rewrapped(self, kr_backend, owner_user_dir, grantee_user_dir):
        """All surviving shares with .kek files are re-wrapped in a single
        hard-revoke call — no limit on how many can survive."""
        proj = _populate_and_seal(owner_user_dir)
        survivors = ["ssk_s1", "ssk_s2", "ssk_s3"]
        for kid in survivors:
            generate_sealed_share(owner_user_dir, "research", kid, key_id=kid)
        generate_sealed_share(owner_user_dir, "research", "victim", key_id="ssk_victim")

        result = revoke_sealed_share(owner_user_dir, "research", "ssk_victim", rotate=True)
        assert result["status"] == "hard_revoked"
        assert result["invalidated_share_key_ids"] == ["ssk_victim"]
        # All survivors are still present.
        for kid in survivors:
            assert share_wrap_path(proj, kid).is_file(), f"Wrap missing for survivor {kid}"
        assert sorted(list_sealed_share_key_ids(proj)) == sorted(survivors)


# ---------------------------------------------------------------------------
# Crash-safe hard revoke: stage + resume on crash
# ---------------------------------------------------------------------------


class TestHardRevokeCrashSafety:
    def test_stages_dek_and_marker_before_rotating_files(
        self, kr_backend, owner_user_dir, monkeypatch
    ):
        """If the rotation crashes mid-loop, the staged DEK + marker
        must already be on disk so the next attempt can resume.
        Asserted by intercepting SealedFile.write and raising on the
        very first file the loop tries to overwrite."""
        from axon.security.crypto import SealedFile

        proj = _populate_and_seal(owner_user_dir)
        original_write = SealedFile.write
        call_count = {"n": 0}

        def first_write_fails(path, plaintext, dek, *, aad):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated crash mid-rotation")
            return original_write(path, plaintext, dek, aad=aad)

        monkeypatch.setattr(SealedFile, "write", first_write_fails)

        with pytest.raises(_security.SecurityError, match="simulated crash"):
            revoke_sealed_share(owner_user_dir, "research", "ssk_x", rotate=True)

        # Staged sidecars survived the crash.
        assert (proj / ".security" / "dek.wrapped.rotating").is_file()
        assert (proj / ".security" / ".sealing.rotation").is_file()

    def test_resume_completes_partial_rotation(self, kr_backend, owner_user_dir, monkeypatch):
        """After a crashed rotation, re-running revoke(rotate=True) must
        re-use the staged DEK + new_seal_id (not generate fresh ones),
        so files already rotated by the crashed run are recognised and
        the final state is consistent.
        """
        from axon.security.crypto import SealedFile

        proj = _populate_and_seal(owner_user_dir)
        marker_before = read_sealed_marker(proj)
        old_seal_id = marker_before["seal_id"]

        # Crash on the second SealedFile.write (lets one file rotate).
        original_write = SealedFile.write
        call_count = {"n": 0}

        def crash_after_one(path, plaintext, dek, *, aad):
            call_count["n"] += 1
            result = original_write(path, plaintext, dek, aad=aad)
            if call_count["n"] >= 2:
                raise OSError("simulated crash after one file rotated")
            return result

        monkeypatch.setattr(SealedFile, "write", crash_after_one)

        with pytest.raises(_security.SecurityError, match="simulated crash"):
            revoke_sealed_share(owner_user_dir, "research", "ssk_x", rotate=True)

        # Staged sidecars present.
        rotation_marker = proj / ".security" / ".sealing.rotation"
        assert rotation_marker.is_file()

        # Resume: restore the original SealedFile.write and re-run.
        monkeypatch.setattr(SealedFile, "write", original_write)
        result = revoke_sealed_share(owner_user_dir, "research", "ssk_x", rotate=True)
        assert result["status"] == "hard_revoked"
        # Final marker carries the SAME new_seal_id from the rotation
        # marker (not a fresh one).
        marker_after = read_sealed_marker(proj)
        assert marker_after["seal_id"] == result["new_seal_id"]
        assert marker_after["seal_id"] != old_seal_id
        # Staging files cleaned up after success.
        assert not rotation_marker.is_file()
        assert not (proj / ".security" / "dek.wrapped.rotating").is_file()
        # files_already_rotated > 0 because one file survived the crash.
        assert result["files_already_rotated"] >= 1


# ---------------------------------------------------------------------------
# Routing through axon.security entry point
# ---------------------------------------------------------------------------


class TestRevokeWiredThroughSecurity:
    def test_security_module_revoke_soft(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_wired_soft")
        result = _security.revoke_sealed_share(owner_user_dir, "research", "ssk_wired_soft")
        assert result["status"] == "soft_revoked"
        assert not share_wrap_path(proj, "ssk_wired_soft").is_file()

    def test_security_module_revoke_hard(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_wired_hard")
        result = _security.revoke_sealed_share(
            owner_user_dir, "research", "ssk_wired_hard", rotate=True
        )
        assert result["status"] == "hard_revoked"
        assert result["files_resealed"] >= 4
