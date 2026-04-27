"""Phase 2: project_seal end-to-end tests.

Covers ``axon.security.seal`` and the wired ``project_seal`` /
``get_sealed_project_record`` entry points in ``axon.security``.

Skips when ``cryptography`` / ``keyring`` aren't installed (the
``sealed`` extra hasn't been pulled in).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon import security as _security  # noqa: E402
from axon.security.cache import is_sealed_file  # noqa: E402
from axon.security.master import bootstrap_store, lock_store  # noqa: E402
from axon.security.seal import (  # noqa: E402
    SEALED_MARKER_PATH,
    is_project_sealed,
    project_seal,
    read_sealed_marker,
)

# ---------------------------------------------------------------------------
# Shared fixtures: in-memory keyring + a populated plaintext project
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
def user_dir(tmp_path):
    ud = tmp_path / "alice"
    ud.mkdir()
    return ud


def _populate_plaintext_project(user_dir: Path, project: str = "research") -> Path:
    """Write a representative project layout with both content + non-content files."""
    proj = user_dir / project
    (proj / "bm25_index").mkdir(parents=True)
    (proj / "vector_store_data").mkdir(parents=True)
    (proj / ".security").mkdir(parents=True)  # mirror real layout

    (proj / "meta.json").write_text('{"project_id": "p1", "name": "research"}', encoding="utf-8")
    (proj / "version.json").write_text('{"seq": 1}', encoding="utf-8")  # plaintext
    (proj / "bm25_index" / ".bm25_log.jsonl").write_text('{"id":"d1"}\n', encoding="utf-8")
    (proj / "vector_store_data" / "manifest.json").write_text('{"d":768}', encoding="utf-8")
    (proj / "vector_store_data" / "seg-00000001.bin").write_bytes(b"\x01" * 4096)
    return proj


# ---------------------------------------------------------------------------
# Idempotency + missing project + locked store
# ---------------------------------------------------------------------------


class TestProjectSealEntryPath:
    def test_missing_project_raises(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        with pytest.raises(_security.SecurityError, match="does not exist"):
            project_seal("does-not-exist", user_dir)

    def test_locked_store_raises(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        _populate_plaintext_project(user_dir)
        lock_store(user_dir)
        with pytest.raises(_security.SecurityError, match="locked"):
            project_seal("research", user_dir)

    def test_already_sealed_is_no_op(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        first = project_seal("research", user_dir)
        assert first["status"] == "sealed"
        assert first["files_sealed"] >= 4

        # Second call must be a no-op.
        second = project_seal("research", user_dir)
        assert second["status"] == "already_sealed"
        # Files still sealed (not double-sealed → not corrupted).
        assert is_sealed_file(proj / "meta.json")


# ---------------------------------------------------------------------------
# What gets sealed vs left plaintext
# ---------------------------------------------------------------------------


class TestProjectSealCoverage:
    def test_meta_json_is_sealed(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        assert is_sealed_file(proj / "meta.json")

    def test_bm25_files_sealed(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        assert is_sealed_file(proj / "bm25_index" / ".bm25_log.jsonl")

    def test_vector_store_files_sealed(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        assert is_sealed_file(proj / "vector_store_data" / "manifest.json")
        assert is_sealed_file(proj / "vector_store_data" / "seg-00000001.bin")

    def test_version_json_stays_plaintext(self, kr_backend, user_dir):
        """Grantees need to detect changes without the DEK — so version.json
        is deliberately NOT sealed (per docs/SHARE_MOUNT_SEALED.md §4.6)."""
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        assert not is_sealed_file(proj / "version.json")
        assert (proj / "version.json").read_text(encoding="utf-8") == '{"seq": 1}'

    def test_security_dir_not_sealed(self, kr_backend, user_dir):
        """The wrap files in .security/ must stay plaintext — they contain
        no plaintext secrets themselves (DEK is wrapped) but sealing them
        would cause a chicken-and-egg unwrap loop."""
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        # The dek.wrapped file is binary AES-KW output (40 bytes), NOT
        # AXSL-headered. Verify the project_seal didn't accidentally
        # AXSL-wrap it.
        assert not is_sealed_file(proj / ".security" / "dek.wrapped")


# ---------------------------------------------------------------------------
# Sealed marker + get_sealed_project_record
# ---------------------------------------------------------------------------


class TestSealedMarker:
    def test_marker_written_after_seal(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        marker = proj / SEALED_MARKER_PATH
        assert marker.is_file()

    def test_is_project_sealed_true_after_seal(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        assert is_project_sealed(proj) is False
        project_seal("research", user_dir)
        assert is_project_sealed(proj) is True

    def test_read_sealed_marker_has_required_fields(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        result = project_seal("research", user_dir)
        marker = read_sealed_marker(proj)
        assert marker is not None
        assert marker["v"] == 1
        assert marker["cipher_suite"] == "AES-256-GCM-v1"
        assert marker["seal_id"] == result["seal_id"]
        assert "sealed_at" in marker
        assert marker["files_sealed"] == result["files_sealed"]

    def test_read_sealed_marker_returns_none_when_unsealed(self, tmp_path):
        assert read_sealed_marker(tmp_path) is None

    def test_read_sealed_marker_raises_on_malformed(self, kr_backend, user_dir):
        proj = _populate_plaintext_project(user_dir)
        marker = proj / SEALED_MARKER_PATH
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{not json", encoding="utf-8")
        with pytest.raises(_security.SecurityError, match="malformed"):
            read_sealed_marker(proj)

    def test_get_sealed_project_record_returns_marker(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        record = _security.get_sealed_project_record("research", user_dir)
        assert record is not None
        assert record["cipher_suite"] == "AES-256-GCM-v1"

    def test_get_sealed_project_record_returns_none_when_unsealed(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        _populate_plaintext_project(user_dir)
        assert _security.get_sealed_project_record("research", user_dir) is None


# ---------------------------------------------------------------------------
# Atomicity + crash-recovery
# ---------------------------------------------------------------------------


class TestSealAtomicity:
    def test_no_sealing_tempfiles_left_after_success(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        leftover = list(proj.rglob("*.sealing"))
        assert leftover == []

    def test_orphan_sealing_files_removed_on_resume(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        # Simulate a crashed prior attempt: leave a .sealing orphan.
        orphan = proj / "vector_store_data" / "seg-00000001.bin.sealing"
        orphan.write_bytes(b"junk-from-crashed-seal")

        result = project_seal("research", user_dir)
        assert result["orphans_removed"] == 1
        assert not orphan.exists()

    def test_partial_seal_resumes_with_persisted_seal_id(self, kr_backend, user_dir):
        """A real resume case: prior run persisted ``.security/.sealing``
        before encrypting one file then crashed. The next run reads the
        persisted seal_id, re-uses it, and skips the already-sealed file
        with a matching AAD so a subsequent mount can decrypt it.
        """
        from axon.security.crypto import SealedFile, make_aad
        from axon.security.master import get_or_create_project_dek
        from axon.security.seal import (
            SEALING_INPROGRESS_PATH,
            _read_inprogress_seal_id,
            _write_inprogress_seal_id,
        )

        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        dek = get_or_create_project_dek(user_dir, proj)

        # Simulate the prior run: persist a seal_id then encrypt one file
        # under that seal_id and crash before completing.
        prior_seal_id = "seal_resume_test_0001"
        _write_inprogress_seal_id(proj, prior_seal_id)
        rel = "bm25_index/.bm25_log.jsonl"
        target = proj / rel
        SealedFile.write(target, b"prior-payload", dek, aad=make_aad(prior_seal_id, rel))

        # Resume: project_seal reads the persisted seal_id and re-uses it.
        result = project_seal("research", user_dir)
        assert result["status"] == "sealed"
        assert result["files_already_sealed"] >= 1
        # The marker carries the resumed seal_id, NOT a fresh one.
        marker = read_sealed_marker(proj)
        assert marker is not None
        assert marker["seal_id"] == prior_seal_id
        # Resume context is cleaned up after success.
        assert not (proj / SEALING_INPROGRESS_PATH).is_file()
        assert _read_inprogress_seal_id(proj) is None

    def test_fresh_seal_persists_inprogress_marker_then_cleans_it_up(self, kr_backend, user_dir):
        """The .security/.sealing resume context must exist DURING seal but
        be removed after the marker is written.
        """
        from axon.security.seal import SEALING_INPROGRESS_PATH

        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        # Pre-condition: no resume context.
        assert not (proj / SEALING_INPROGRESS_PATH).is_file()
        result = project_seal("research", user_dir)
        # Post-condition: resume context cleaned up.
        assert result["status"] == "sealed"
        assert not (proj / SEALING_INPROGRESS_PATH).is_file()


# ---------------------------------------------------------------------------
# Files actually contain ciphertext (smoke check)
# ---------------------------------------------------------------------------


class TestSealedContentIsCiphertext:
    def test_meta_json_no_longer_readable_as_plaintext(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        # On-disk bytes of meta.json must NOT contain the original
        # plaintext substring "project_id".
        sealed_bytes = (proj / "meta.json").read_bytes()
        assert b"project_id" not in sealed_bytes
        # And starts with the AXSL magic.
        assert sealed_bytes[:4] == b"AXSL"

    def test_vector_store_segment_no_longer_readable(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        proj = _populate_plaintext_project(user_dir)
        project_seal("research", user_dir)
        sealed_bytes = (proj / "vector_store_data" / "seg-00000001.bin").read_bytes()
        # Original was 4096 bytes of 0x01 — sealed must not start with 0x01.
        assert sealed_bytes[:4] == b"AXSL"
        assert b"\x01" * 16 not in sealed_bytes  # no run of original plaintext


# ---------------------------------------------------------------------------
# Nested project layout (parent/child via subs/)
# ---------------------------------------------------------------------------


class TestNestedProjectSeal:
    def test_seal_nested_project_via_subs_layout(self, kr_backend, user_dir):
        bootstrap_store(user_dir, "test-pass-ok")
        # research/papers → user_dir/research/subs/papers
        nested = user_dir / "research" / "subs" / "papers"
        (nested / "bm25_index").mkdir(parents=True)
        (nested / "vector_store_data").mkdir(parents=True)
        (nested / "meta.json").write_text(
            '{"project_id": "papers", "name": "papers"}', encoding="utf-8"
        )

        result = project_seal("research/papers", user_dir)
        assert result["status"] == "sealed"
        assert is_project_sealed(nested) is True
        assert is_sealed_file(nested / "meta.json") is True
