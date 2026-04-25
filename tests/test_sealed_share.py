"""Phase 3: sealed-share generate + redeem tests.

Covers ``axon.security.share`` and the wired
``generate_sealed_share`` / ``redeem_sealed_share`` entry points in
``axon.security``.

Skips when ``cryptography`` / ``keyring`` aren't installed (the
``sealed`` extra hasn't been pulled in).
"""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon import security as _security  # noqa: E402
from axon.security.master import bootstrap_store, lock_store  # noqa: E402
from axon.security.seal import project_seal  # noqa: E402
from axon.security.share import (  # noqa: E402
    SEALED_SHARE_PREFIX,
    delete_grantee_dek,
    generate_sealed_share,
    get_grantee_dek,
    redeem_sealed_share,
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
    """Owner's AxonStore user directory."""
    store = tmp_path / "AxonStore" / "alice"
    store.mkdir(parents=True)
    return store


@pytest.fixture
def grantee_user_dir(tmp_path):
    """Grantee's AxonStore user directory (separate from owner)."""
    store = tmp_path / "AxonStore" / "bob"
    store.mkdir(parents=True)
    return store


def _populate_and_seal(owner_user_dir: Path, project: str = "research") -> Path:
    """Create a representative project, seal it, return its directory."""
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
# generate_sealed_share — happy path
# ---------------------------------------------------------------------------


class TestGenerateSealedShareHappyPath:
    def test_returns_share_string_with_sealed_prefix(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        result = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_test01")
        assert result["key_id"] == "ssk_test01"
        assert result["sealed"] is True
        assert result["owner"] == "alice"
        assert result["project"] == "research"
        assert result["grantee"] == "bob"
        # Decoded share_string starts with the prefix.
        decoded = base64.urlsafe_b64decode(result["share_string"]).decode("utf-8")
        assert decoded.startswith(f"{SEALED_SHARE_PREFIX}:")
        assert "ssk_test01" in decoded
        assert "alice" in decoded
        assert "research" in decoded

    def test_writes_wrap_file_with_correct_size(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_test02")
        wrap = share_wrap_path(proj, "ssk_test02")
        assert wrap.is_file()
        # AES-KW of a 32-byte key produces exactly 40 bytes.
        assert wrap.stat().st_size == 40

    def test_distinct_key_ids_produce_distinct_wraps(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_a")
        generate_sealed_share(owner_user_dir, "research", "carol", key_id="ssk_b")
        wrap_a = share_wrap_path(proj, "ssk_a").read_bytes()
        wrap_b = share_wrap_path(proj, "ssk_b").read_bytes()
        # Same DEK, different KEK → different ciphertext.
        assert wrap_a != wrap_b


# ---------------------------------------------------------------------------
# generate_sealed_share — error paths
# ---------------------------------------------------------------------------


class TestGenerateSealedShareErrors:
    def test_unsealed_project_raises(self, kr_backend, owner_user_dir):
        bootstrap_store(owner_user_dir, "owner-pw")
        # Project exists but never sealed.
        proj = owner_user_dir / "research"
        proj.mkdir()
        (proj / "meta.json").write_text("{}", encoding="utf-8")
        with pytest.raises(_security.SecurityError, match="not sealed"):
            generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_x")

    def test_missing_project_raises(self, kr_backend, owner_user_dir):
        bootstrap_store(owner_user_dir, "owner-pw")
        with pytest.raises(_security.SecurityError, match="does not exist"):
            generate_sealed_share(owner_user_dir, "nope", "bob", key_id="ssk_x")

    def test_locked_store_raises(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        lock_store(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="locked"):
            generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_x")

    def test_existing_key_id_raises(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_dup")
        with pytest.raises(_security.SecurityError, match="already exists"):
            generate_sealed_share(owner_user_dir, "research", "carol", key_id="ssk_dup")

    def test_empty_key_id_raises(self, kr_backend, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="non-empty"):
            generate_sealed_share(owner_user_dir, "research", "bob", key_id="")

    @pytest.mark.parametrize(
        "bad_key_id",
        [
            "../escape",
            "..\\escape",
            "ssk:colon",
            "ssk space",
            "ssk/slash",
            "ssk\\back",
            "a" * 65,  # over 64-char cap
        ],
    )
    def test_unsafe_key_id_rejected(self, kr_backend, owner_user_dir, bad_key_id):
        """Path-separators / colons / oversized key_ids must be refused."""
        _populate_and_seal(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="Invalid key_id"):
            generate_sealed_share(owner_user_dir, "research", "bob", key_id=bad_key_id)


# ---------------------------------------------------------------------------
# redeem_sealed_share — happy path round-trip
# ---------------------------------------------------------------------------


class TestRedeemSealedShareRoundTrip:
    def test_grantee_unwraps_and_caches_dek(self, kr_backend, owner_user_dir, grantee_user_dir):
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_rt01")
        result = redeem_sealed_share(grantee_user_dir, share["share_string"])
        assert result["sealed"] is True
        assert result["key_id"] == "ssk_rt01"
        assert result["owner"] == "alice"
        assert result["project"] == "research"
        assert result["mount_name"] == "alice_research"

        # DEK is now fetchable from the grantee's keyring.
        dek = get_grantee_dek("ssk_rt01")
        assert isinstance(dek, bytes)
        assert len(dek) == 32

    def test_mount_descriptor_carries_sealed_flag(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_rt02")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        mount_json = grantee_user_dir / "mounts" / "alice_research" / "mount.json"
        import json as _json

        desc = _json.loads(mount_json.read_text(encoding="utf-8"))
        assert desc["mount_type"] == "sealed"
        assert desc["share_key_id"] == "ssk_rt02"
        # Canonical mount schema uses ``readonly`` (no underscore).
        assert desc["readonly"] is True
        assert desc["state"] == "active"
        assert desc["revoked"] is False
        assert desc["owner"] == "alice"
        assert desc["project"] == "research"

    def test_descriptor_passes_canonical_validate(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """The sealed mount.json must satisfy axon.mounts.validate_mount_descriptor
        + appear in axon.mounts.list_mount_descriptors. Without this,
        AxonBrain.switch_project would refuse to open the sealed mount.
        """
        from axon.mounts import (
            list_mount_descriptors,
            load_mount_descriptor,
            validate_mount_descriptor,
        )

        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_compat")
        redeem_sealed_share(grantee_user_dir, share["share_string"])

        desc = load_mount_descriptor(grantee_user_dir, "alice_research")
        assert desc is not None
        valid, reason = validate_mount_descriptor(desc)
        assert valid, f"Sealed mount descriptor failed canonical validation: {reason}"
        # And it shows up in the user's active-mount listing.
        listed = list_mount_descriptors(grantee_user_dir)
        names = [d.get("mount_name") for d in listed]
        assert "alice_research" in names
        # Sealed-specific fields are preserved on top of the canonical ones.
        seal_desc = next(d for d in listed if d.get("mount_name") == "alice_research")
        assert seal_desc["mount_type"] == "sealed"
        assert seal_desc["share_key_id"] == "ssk_compat"

    def test_dek_matches_owners_dek(self, kr_backend, owner_user_dir, grantee_user_dir):
        """The DEK the grantee unwraps must equal the DEK the owner has —
        otherwise mounting would fail with InvalidTag on every file.
        """
        from axon.security.master import get_project_dek

        _populate_and_seal(owner_user_dir)
        owner_dek = get_project_dek(owner_user_dir, owner_user_dir / "research")

        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_rt03")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        grantee_dek = get_grantee_dek("ssk_rt03")
        assert owner_dek == grantee_dek


# ---------------------------------------------------------------------------
# redeem_sealed_share — error paths
# ---------------------------------------------------------------------------


class TestRedeemSealedShareErrors:
    def test_malformed_base64_raises(self, kr_backend, grantee_user_dir):
        with pytest.raises(_security.SecurityError, match="Invalid sealed share_string"):
            redeem_sealed_share(grantee_user_dir, "not-base64!!")

    def test_legacy_non_sealed_share_raises_value_error(self, kr_backend, grantee_user_dir):
        # A legacy share_string lacks the SEALED1: prefix.
        legacy = base64.urlsafe_b64encode(b"sk_legacy:abc:alice:research:/store").decode("ascii")
        with pytest.raises(ValueError, match="not a sealed share"):
            redeem_sealed_share(grantee_user_dir, legacy)

    def test_missing_wrap_file_raises(self, kr_backend, owner_user_dir, grantee_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_missing")
        # Owner deletes the wrap file (soft revocation simulation).
        share_wrap_path(proj, "ssk_missing").unlink()
        with pytest.raises(_security.SecurityError, match="missing at"):
            redeem_sealed_share(grantee_user_dir, share["share_string"])

    def test_corrupted_wrap_file_raises(self, kr_backend, owner_user_dir, grantee_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_corrupt")
        # Replace wrap with garbage.
        share_wrap_path(proj, "ssk_corrupt").write_bytes(b"\x00" * 40)
        with pytest.raises(_security.SecurityError, match="won't unwrap"):
            redeem_sealed_share(grantee_user_dir, share["share_string"])

    def test_truncated_wrap_raises_security_error(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """aes_key_unwrap raises ValueError (not InvalidUnwrap) on a
        truncated wrap file — our wrapper must catch both."""
        proj = _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_trunc")
        # Truncate the wrap file to 16 bytes (less than 1 AES block).
        share_wrap_path(proj, "ssk_trunc").write_bytes(b"\x00" * 16)
        with pytest.raises(_security.SecurityError, match="malformed|won't unwrap"):
            redeem_sealed_share(grantee_user_dir, share["share_string"])

    def test_tampered_key_id_in_envelope_rejected(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        """A share_string whose key_id has been tampered with to contain
        a path separator must be refused before path construction."""
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_safe")
        # Decode, swap key_id for a traversal, re-encode.
        decoded = base64.urlsafe_b64decode(share["share_string"]).decode("utf-8")
        parts = decoded.split(":")
        parts[1] = "../escape"  # tampered key_id
        bad = base64.urlsafe_b64encode(":".join(parts).encode()).decode("ascii")
        with pytest.raises(_security.SecurityError, match="Invalid key_id"):
            redeem_sealed_share(grantee_user_dir, bad)

    def test_owner_project_dir_missing_raises(
        self, kr_backend, owner_user_dir, grantee_user_dir, tmp_path
    ):
        """If the owner's synced folder hasn't appeared yet, redeem should
        say so rather than silently mounting a broken descriptor.
        """
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_unsync")
        # Simulate "owner not synced" by editing the share_string to
        # point at a non-existent owner_store_path.
        decoded = base64.urlsafe_b64decode(share["share_string"]).decode("utf-8")
        parts = decoded.split(":")
        parts[-1] = str(tmp_path / "no_such_store")  # owner_store_path
        broken = base64.urlsafe_b64encode(":".join(parts).encode()).decode("ascii")
        with pytest.raises(_security.SecurityError, match="not yet synced|does not exist"):
            redeem_sealed_share(grantee_user_dir, broken)


# ---------------------------------------------------------------------------
# get_grantee_dek + delete_grantee_dek
# ---------------------------------------------------------------------------


class TestGranteeDekAccess:
    def test_get_returns_dek_after_redeem(self, kr_backend, owner_user_dir, grantee_user_dir):
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_get01")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        dek = get_grantee_dek("ssk_get01")
        assert len(dek) == 32

    def test_get_missing_raises(self, kr_backend):
        with pytest.raises(_security.SecurityError, match="No sealed-share DEK"):
            get_grantee_dek("ssk_never_redeemed")

    def test_get_empty_key_id_raises(self, kr_backend):
        with pytest.raises(_security.SecurityError, match="non-empty"):
            get_grantee_dek("")

    def test_delete_returns_true_after_existing(self, kr_backend, owner_user_dir, grantee_user_dir):
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_del01")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        assert delete_grantee_dek("ssk_del01") is True
        with pytest.raises(_security.SecurityError):
            get_grantee_dek("ssk_del01")

    def test_delete_missing_returns_false(self, kr_backend):
        assert delete_grantee_dek("ssk_never") is False


# ---------------------------------------------------------------------------
# Mount-side: materialize_for_read accepts a pre-supplied DEK
# ---------------------------------------------------------------------------


class TestMaterializeWithSuppliedDek:
    def test_grantee_dek_path_decrypts_files(
        self, kr_backend, owner_user_dir, grantee_user_dir, tmp_path
    ):
        """End-to-end: owner seals + generates share; grantee redeems +
        fetches DEK from keyring + materialises cache from owner's
        synced project dir using ONLY the keyring DEK (no master).
        """
        from axon.security.mount import materialize_for_read, release_cache

        proj = _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", key_id="ssk_e2e01")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        dek = get_grantee_dek("ssk_e2e01")

        # Lock the owner's master to PROVE the grantee path doesn't
        # need it. Without lock, the owner-side master cache is shared
        # across the test process so the test wouldn't really verify
        # the grantee-only path.
        lock_store(owner_user_dir)

        cache_root = tmp_path / "cr"
        cache_root.mkdir()
        cache = materialize_for_read(proj, owner_user_dir, dek=dek, cache_root=cache_root)
        try:
            assert (
                cache.path / "vector_store_data" / "seg-00000001.bin"
            ).read_bytes() == b"\xab" * 4096
            assert (cache.path / "meta.json").read_text(
                encoding="utf-8"
            ) == '{"project_id":"p1","name":"research"}'
        finally:
            release_cache(cache)

    def test_supplied_dek_wrong_size_raises(self, kr_backend, owner_user_dir, tmp_path):
        from axon.security.mount import materialize_for_read

        proj = _populate_and_seal(owner_user_dir)
        with pytest.raises(_security.SecurityError, match="32"):
            materialize_for_read(proj, owner_user_dir, dek=b"too short", cache_root=tmp_path / "cr")
