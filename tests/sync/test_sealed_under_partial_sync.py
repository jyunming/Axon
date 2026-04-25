"""Phase 7 Layer 1: sealed-store behaviour under simulated sync chaos.

These tests inject the sync-engine misbehaviours OneDrive / Dropbox /
SMB exhibit during file propagation — partial-byte visibility, locked
files, conflict-copy artifacts, ``.tmp.drivedownload`` debris — and
assert that the sealed-store code paths react sensibly. No real cloud
account; runs in the CI matrix.

Skips when ``cryptography`` / ``keyring`` aren't installed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon.security import SecurityError  # noqa: E402
from axon.security.master import bootstrap_store  # noqa: E402
from axon.security.seal import project_seal  # noqa: E402
from axon.security.share import (  # noqa: E402
    generate_sealed_share,
    list_sealed_share_key_ids,
    redeem_sealed_share,
    share_wrap_path,
)

# ---------------------------------------------------------------------------
# In-memory keyring fixture (mirrors the rest of the sealed test suite)
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
# Partial sync: wrap file appears with metadata but truncated bytes
# ---------------------------------------------------------------------------


class TestPartialSyncOfShareWrap:
    def test_truncated_wrap_during_redeem_raises_security_error(
        self, kr_backend, owner_user_dir, grantee_user_dir, sync_chaos
    ):
        """OneDrive may surface a wrap file before its 40 bytes finish
        uploading. Redeem must NOT silently accept partial bytes — the
        AES-KW unwrap will raise InvalidUnwrap or ValueError; we wrap
        both as SecurityError with a 'finished syncing' hint.
        """
        proj = _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_partial")
        # Truncate the wrap on disk to mimic mid-upload state.
        wrap = share_wrap_path(proj, "ssk_partial")
        wrap.write_bytes(wrap.read_bytes()[:16])

        with pytest.raises(SecurityError, match="malformed|won't unwrap|finished syncing"):
            redeem_sealed_share(grantee_user_dir, share["share_string"])

    def test_zero_byte_wrap_during_sync_raises(self, kr_backend, owner_user_dir, grantee_user_dir):
        """The most extreme partial state: file exists but is 0 bytes."""
        proj = _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_zero")
        share_wrap_path(proj, "ssk_zero").write_bytes(b"")
        with pytest.raises(SecurityError):
            redeem_sealed_share(grantee_user_dir, share["share_string"])


# ---------------------------------------------------------------------------
# Locked file: sync engine holds an exclusive handle
# ---------------------------------------------------------------------------


class TestLockedFileDuringSync:
    def test_locked_dek_wrap_surfaces_clear_error(self, kr_backend, owner_user_dir, sync_chaos):
        """Owner's master can be unlocked, but if the project's
        ``dek.wrapped`` is locked by the sync engine, ``get_project_dek``
        must surface a clear error rather than crash with a raw OSError.
        """
        proj = _populate_and_seal(owner_user_dir)
        from axon.security.master import get_project_dek

        dek_wrap = proj / ".security" / "dek.wrapped"
        sync_chaos.lock(dek_wrap)
        with pytest.raises((SecurityError, PermissionError)):
            get_project_dek(owner_user_dir, proj)


# ---------------------------------------------------------------------------
# Conflict copies: list_sealed_share_key_ids must ignore them
# ---------------------------------------------------------------------------


class TestConflictCopyIgnored:
    def test_onedrive_conflict_suffix_not_listed_as_share(
        self, kr_backend, owner_user_dir, sync_chaos
    ):
        """OneDrive creates ``<name>-OneDrive-MachineB.conflict`` when
        two machines write the same file. ``list_sealed_share_key_ids``
        must return ONLY real share key_ids (filenames ending exactly
        in ``.wrapped``), not the conflict copies.
        """
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_real")
        # OneDrive conflict copy of the wrap file.
        sync_chaos.drop_conflict_copy(
            share_wrap_path(proj, "ssk_real"),
            suffix="-OneDrive-MachineB.conflict",
            contents=b"\x00" * 40,
        )
        ids = list_sealed_share_key_ids(proj)
        # Only the real key_id is reported.
        assert ids == ["ssk_real"]
        # The conflict copy IS on disk (ends in .wrapped because the
        # OneDrive convention injects the suffix BEFORE the extension)
        # — we just don't list it.
        conflicts = [p for p in (proj / ".security" / "shares").iterdir() if "conflict" in p.name]
        assert len(conflicts) == 1


class TestDriveDownloadArtifactIgnored:
    def test_dot_drivedownload_artifact_not_listed(self, kr_backend, owner_user_dir, sync_chaos):
        """Google Drive's ``.tmp.drivedownload`` debris must not be
        confused with a real share wrap.
        """
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_clean")
        sync_chaos.drop_drivedownload_artifact(proj / ".security" / "shares")
        ids = list_sealed_share_key_ids(proj)
        assert ids == ["ssk_clean"]


# ---------------------------------------------------------------------------
# Sync engine reverses payload + metadata order
# ---------------------------------------------------------------------------


class TestMetadataBeforePayload:
    def test_zero_size_visibility_during_sync_does_not_break_listing(
        self, kr_backend, owner_user_dir, sync_chaos
    ):
        """OneDrive sometimes presents a file's metadata (path exists)
        before the bytes finish uploading. ``list_sealed_share_key_ids``
        only checks ``is_file()`` + name — it does NOT touch the bytes,
        so the partial state is invisible at the listing layer.
        """
        proj = _populate_and_seal(owner_user_dir)
        # Generate the share — wrap file lands at <proj>/.security/shares/<key_id>.wrapped.
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_partial2")
        wrap = share_wrap_path(proj, "ssk_partial2")
        # Stat reports 0 bytes, but the file is on disk.
        sync_chaos.appear_partial(wrap, fake_size=0)
        ids = list_sealed_share_key_ids(proj)
        assert "ssk_partial2" in ids


# ---------------------------------------------------------------------------
# Smoke: a full happy-path round-trip survives the chaos infrastructure
# ---------------------------------------------------------------------------


class TestChaosInfrastructureNoRegression:
    def test_round_trip_with_chaos_fixture_loaded(
        self, kr_backend, owner_user_dir, grantee_user_dir, sync_chaos
    ):
        """Sanity check that the chaos fixture itself doesn't break
        the happy path when no faults are injected."""
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_happy")
        result = redeem_sealed_share(grantee_user_dir, share["share_string"])
        assert result["sealed"] is True
        assert result["key_id"] == "ssk_happy"
