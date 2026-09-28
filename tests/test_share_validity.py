"""One authoritative answer to "is this received share still valid?" (PR6).

Two groups:

* ``TestRegressions`` (R1–R11) — each test reproduces a place where two
  surfaces used to disagree and asserts the unified behaviour. They only
  touch pre-existing symbols (``AxonBrain.switch_project``,
  ``QueryRouterMixin._check_mount_revocation``,
  ``shares.validate_received_shares``, ``list_share_mounts``, the REST
  client) so they fail — rather than error at import — on a checkout
  without :mod:`axon.share_validity`.
* ``TestOneAnswerMatrix`` — for every scenario, ``share_status``,
  ``brain.switch_project``, the per-query guard, ``list_share_mounts``,
  REST ``/projects`` and the reconcile pass must all agree.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MethodType
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon import shares
from axon.api import app
from axon.main import AxonBrain
from axon.mounts import load_mount_descriptor, mount_descriptor_path
from axon.query_router import QueryRouterMixin

client = TestClient(app, raise_server_exceptions=False)

_HAS_SEALED = True
try:  # the [sealed] extra
    import cryptography  # noqa: F401
    import keyring  # noqa: F401
except ImportError:  # pragma: no cover - exercised on minimal installs only
    _HAS_SEALED = False

needs_sealed = pytest.mark.skipif(not _HAS_SEALED, reason="requires axon-rag[sealed]")


# ---------------------------------------------------------------------------
# Fixtures / helpers
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
    if not _HAS_SEALED:
        yield None
        return
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


@pytest.fixture(autouse=True)
def _restore_api_brain():
    saved = api_module.brain
    yield
    api_module.brain = saved


def _store(tmp_path: Path) -> tuple[Path, Path]:
    owner = tmp_path / "AxonStore" / "alice"
    grantee = tmp_path / "AxonStore" / "bob"
    (owner / ".shares").mkdir(parents=True)
    (grantee / ".shares").mkdir(parents=True)
    proj = owner / "research"
    (proj / "bm25_index").mkdir(parents=True)
    (proj / "vector_store_data").mkdir(parents=True)
    (proj / "meta.json").write_text(
        json.dumps({"project_id": "p1", "name": "research"}), encoding="utf-8"
    )
    return owner, grantee


def _plain_share(tmp_path: Path, **gen_kw) -> dict:
    owner, grantee = _store(tmp_path)
    gen = shares.generate_share_key(owner, "research", "bob", **gen_kw)
    red = shares.redeem_share_key(grantee, gen["share_string"])
    return {
        "owner": owner,
        "grantee": grantee,
        "key_id": gen["key_id"],
        "share_string": gen["share_string"],
        "mount": red["mount_name"],
        "project_dir": owner / "research",
        "kind": "plain",
    }


def _seal(owner: Path, project: str = "research") -> Path:
    from axon.security.master import bootstrap_store
    from axon.security.seal import project_seal

    proj = owner / project
    (proj / ".security").mkdir(parents=True, exist_ok=True)
    (proj / "version.json").write_text('{"seq":1}', encoding="utf-8")
    (proj / "bm25_index" / ".bm25_log.jsonl").write_text('{"id":"d1"}\n', encoding="utf-8")
    (proj / "vector_store_data" / "manifest.json").write_text('{"d":768}', encoding="utf-8")
    bootstrap_store(owner, "owner-pw")
    project_seal(project, owner)
    return proj


def _sealed_share(tmp_path: Path, key_id: str = "ssk_pr6", expires_at=None) -> dict:
    from axon.security.share import generate_sealed_share, redeem_sealed_share

    owner, grantee = _store(tmp_path)
    proj = _seal(owner)
    gen = generate_sealed_share(owner, "research", "bob", key_id, expires_at=expires_at)
    red = redeem_sealed_share(grantee, gen["share_string"])
    return {
        "owner": owner,
        "grantee": grantee,
        "key_id": key_id,
        "share_string": gen["share_string"],
        "mount": red["mount_name"],
        "project_dir": proj,
        "kind": "sealed",
    }


def _manifest_path(owner: Path) -> Path:
    return owner / ".shares" / ".share_manifest.json"


def _edit_manifest(owner: Path, fn) -> None:
    path = _manifest_path(owner)
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(json.dumps(data), encoding="utf-8")


class _Reached(Exception):
    """Raised by the mocked close(): the validity gate let the switch through."""


def _mock_brain(user_dir: Path) -> MagicMock:
    brain = MagicMock()
    brain.config = MagicMock()
    brain.config.projects_root = str(user_dir)
    brain._pending_seal_mount = None
    brain._sealed_cache = None
    brain.close = MagicMock(side_effect=_Reached())
    return brain


def _switch(user_dir: Path, mount: str) -> tuple[str, str, MagicMock]:
    """Return ("allowed"|"denied", message, brain)."""
    brain = _mock_brain(user_dir)
    try:
        AxonBrain.switch_project(brain, f"mounts/{mount}")
    except _Reached:
        return "allowed", "", brain
    except ValueError as exc:
        return "denied", str(exc), brain
    return "allowed", "", brain


def _per_query(user_dir: Path, mount: str, descriptor: dict | None = None):
    """Run the real per-query guard against the on-disk descriptor.

    Returns (raised_exception_or_None, brain). ``_auto_destroy_expired_share``
    is a mock so the caller can assert whether it was triggered.
    """
    brain = MagicMock()
    brain._active_project_kind = "mounted"
    brain._active_mount_descriptor = descriptor or load_mount_descriptor(user_dir, mount)
    brain._check_mount_revocation = MethodType(QueryRouterMixin._check_mount_revocation, brain)
    try:
        brain._check_mount_revocation()
    except PermissionError as exc:
        return exc, brain
    return None, brain


def _rest_projects(user_dir: Path) -> dict:
    brain = MagicMock()
    brain.config.projects_root = str(user_dir)
    api_module.brain = brain
    with patch("axon.projects.list_projects", return_value=[]):
        resp = client.get("/projects")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _dek_present(key_id: str) -> bool:
    from axon.security import keyring as _kr
    from axon.security.share import _share_keyring_service

    return _kr.get_secret(_share_keyring_service(key_id), "dek") is not None


# ---------------------------------------------------------------------------
# R1–R11 — regressions (must fail on main)
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_r1_switch_refuses_revoked_plain_share(self, tmp_path):
        s = _plain_share(tmp_path)
        shares.revoke_share_key(s["owner"], s["key_id"])
        verdict, msg, _ = _switch(s["grantee"], s["mount"])
        assert verdict == "denied"
        assert "revoked" in msg

    def test_r1_switch_refuses_expired_plain_share(self, tmp_path):
        s = _plain_share(tmp_path, ttl_days=7)
        _edit_manifest(
            s["owner"],
            lambda m: m["issued"][0].update(expires_at="2020-01-01T00:00:00+00:00"),
        )
        verdict, msg, _ = _switch(s["grantee"], s["mount"])
        assert verdict == "denied"
        assert "expired" in msg

    @needs_sealed
    def test_r2_soft_revoked_sealed_denied_at_switch_and_query(self, tmp_path, kr_backend):
        from axon.security.share import revoke_sealed_share

        s = _sealed_share(tmp_path)
        desc = load_mount_descriptor(s["grantee"], s["mount"])
        revoke_sealed_share(s["owner"], "research", s["key_id"])
        verdict, msg, _ = _switch(s["grantee"], s["mount"])
        assert verdict == "denied"
        assert "revoked" in msg
        exc, _ = _per_query(s["grantee"], s["mount"], desc)
        assert isinstance(exc, PermissionError)
        assert "revoked" in str(exc)

    @needs_sealed
    def test_r3_cli_share_list_path_clears_soft_revoked_sealed_mount(self, tmp_path, kr_backend):
        from axon.security.share import revoke_sealed_share

        s = _sealed_share(tmp_path)
        revoke_sealed_share(s["owner"], "research", s["key_id"])
        # CLI --share-list / REPL /share list call shares.validate_received_shares.
        assert shares.validate_received_shares(s["grantee"]) == [s["mount"]]
        assert load_mount_descriptor(s["grantee"], s["mount"]) is None
        # Soft revoke never promised to wipe the grantee's cached DEK.
        assert _dek_present(s["key_id"])

    @needs_sealed
    def test_r4_expired_sealed_share_removed_by_share_list(self, tmp_path, kr_backend):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        s = _sealed_share(tmp_path, expires_at=past)
        assert _dek_present(s["key_id"])
        api_module.brain = MagicMock()
        with patch("axon.api._get_user_dir", return_value=s["grantee"]):
            resp = client.get("/share/list")
        assert resp.status_code == 200, resp.text
        assert s["mount"] in resp.json().get("removed_stale", [])
        assert load_mount_descriptor(s["grantee"], s["mount"]) is None
        assert not _dek_present(s["key_id"])

    @needs_sealed
    def test_r4_expired_sealed_share_denied_per_query_with_auto_destroy(self, tmp_path, kr_backend):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        s = _sealed_share(tmp_path, expires_at=past)
        exc, brain = _per_query(s["grantee"], s["mount"])
        assert isinstance(exc, PermissionError)
        assert "expired" in str(exc)
        brain._auto_destroy_expired_share.assert_called_once()
        assert brain._auto_destroy_expired_share.call_args[0][1] == s["key_id"]

    @needs_sealed
    def test_r5_unreachable_sealed_target_is_kept_not_deleted(self, tmp_path, kr_backend):
        from axon import security

        s = _sealed_share(tmp_path)
        moved = s["project_dir"].with_name("research_offline")
        s["project_dir"].rename(moved)
        assert security.validate_received_sealed_shares(s["grantee"]) == []
        assert load_mount_descriptor(s["grantee"], s["mount"]) is not None
        verdict, _, _ = _switch(s["grantee"], s["mount"])
        assert verdict == "denied"

    @pytest.mark.parametrize("variant", ["manifest_deleted", "key_absent"])
    def test_r6_unverifiable_plain_share_denied_per_query(self, tmp_path, variant):
        s = _plain_share(tmp_path)
        if variant == "manifest_deleted":
            _manifest_path(s["owner"]).unlink()
        else:
            _edit_manifest(s["owner"], lambda m: m.update(issued=[]))
        exc, _ = _per_query(s["grantee"], s["mount"])
        assert isinstance(exc, PermissionError)
        # ...but the mount is kept (UNVERIFIABLE is never destructive).
        assert shares.validate_received_shares(s["grantee"]) == []
        assert load_mount_descriptor(s["grantee"], s["mount"]) is not None

    @needs_sealed
    def test_r7_revoking_old_plain_key_keeps_newer_sealed_mount(self, tmp_path, kr_backend):
        from axon.security.share import generate_sealed_share, redeem_sealed_share

        owner, grantee = _store(tmp_path)
        plain = shares.generate_share_key(owner, "research", "bob")
        red = shares.redeem_share_key(grantee, plain["share_string"])
        _seal(owner)
        gen = generate_sealed_share(owner, "research", "bob", "ssk_newer")
        red2 = redeem_sealed_share(grantee, gen["share_string"])
        assert red2["mount_name"] == red["mount_name"]  # same mount name reused
        shares.revoke_share_key(owner, plain["key_id"])

        assert shares.validate_received_shares(grantee) == []
        desc = load_mount_descriptor(grantee, red["mount_name"])
        assert desc is not None and desc["share_key_id"] == "ssk_newer"
        keys = json.loads((grantee / ".shares" / ".share_keys.json").read_text())
        assert all(r.get("key_id") != plain["key_id"] for r in keys.get("received", []))

    def test_r8_corrupt_manifest_is_never_rewritten(self, tmp_path):
        owner, _ = _store(tmp_path)
        a = shares.generate_share_key(owner, "research", "bob")
        shares.revoke_share_key(owner, a["key_id"])
        b = shares.generate_share_key(owner, "research", "carol")
        path = _manifest_path(owner)
        good = path.read_bytes()
        path.write_bytes(good[: len(good) // 2])  # truncated by a bad sync
        corrupt = path.read_bytes()
        keys_before = (owner / ".shares" / ".share_keys.json").read_bytes()

        for call in (
            lambda: shares.revoke_share_key(owner, b["key_id"]),
            lambda: shares.extend_share_key(owner, b["key_id"], ttl_days=3),
            lambda: shares.generate_share_key(owner, "research", "dave"),
        ):
            with pytest.raises(RuntimeError, match="refusing to overwrite"):
                call()
            assert path.read_bytes() == corrupt
            assert (owner / ".shares" / ".share_keys.json").read_bytes() == keys_before

        path.write_bytes(good)  # restore from backup: sk_a's tombstone survived
        rec = next(r for r in json.loads(good)["issued"] if r["key_id"] == a["key_id"])
        assert rec["revoked"] is True

    def test_r9_mounts_scope_excludes_revoked_mount(self, tmp_path):
        s = _plain_share(tmp_path)
        shares.revoke_share_key(s["owner"], s["key_id"])
        brain = _mock_brain(s["grantee"])
        with patch("axon.projects.PROJECTS_ROOT", s["grantee"]):
            with pytest.raises(ValueError, match="No authoritative projects"):
                AxonBrain._switch_to_scope(brain, "@mounts")

    def _delete(self, owner: Path, name: str):
        brain = MagicMock()
        brain.config.projects_root = str(owner)
        brain._active_project = "default"
        api_module.brain = brain
        with (
            patch("axon.projects.delete_project") as dp,
            patch.object(api_module, "_save_source_hashes"),
        ):
            resp = client.post(f"/project/delete/{name}")
        return resp, dp

    def test_r10_expired_plain_share_does_not_block_delete(self, tmp_path):
        owner, _ = _store(tmp_path)
        shares.generate_share_key(owner, "research", "bob", ttl_days=1)
        _edit_manifest(
            owner, lambda m: m["issued"][0].update(expires_at="2020-01-01T00:00:00+00:00")
        )
        resp, dp = self._delete(owner, "research")
        assert resp.status_code == 200, resp.text
        dp.assert_called_once_with("research")

    @needs_sealed
    def test_r10_active_sealed_share_blocks_delete(self, tmp_path, kr_backend):
        from axon.security.share import generate_sealed_share

        owner, _ = _store(tmp_path)
        _seal(owner)
        generate_sealed_share(owner, "research", "bob", "ssk_blocks")
        resp, dp = self._delete(owner, "research")
        assert resp.status_code == 409, resp.text
        assert "ssk_blocks" in resp.json()["detail"]
        dp.assert_not_called()

    def test_r11_descriptor_without_received_record_is_still_validated(self, tmp_path):
        s = _plain_share(tmp_path)
        (s["grantee"] / ".shares" / ".share_keys.json").unlink()
        shares.revoke_share_key(s["owner"], s["key_id"])
        body = _rest_projects(s["grantee"])
        assert all(m["name"] != f"mounts/{s['mount']}" for m in body["shared_mounts"])
        assert load_mount_descriptor(s["grantee"], s["mount"]) is None


# ---------------------------------------------------------------------------
# The "one answer" matrix
# ---------------------------------------------------------------------------


def _sc_plain_valid(tmp_path):
    return _plain_share(tmp_path)


def _sc_plain_revoked(tmp_path):
    s = _plain_share(tmp_path)
    shares.revoke_share_key(s["owner"], s["key_id"])
    return s


def _sc_plain_expired(tmp_path):
    s = _plain_share(tmp_path, ttl_days=3)
    _edit_manifest(
        s["owner"], lambda m: m["issued"][0].update(expires_at="2020-01-01T00:00:00+00:00")
    )
    return s


def _sc_plain_manifest_missing(tmp_path):
    s = _plain_share(tmp_path)
    _manifest_path(s["owner"]).unlink()
    return s


def _sc_plain_record_missing(tmp_path):
    s = _plain_share(tmp_path)
    _edit_manifest(s["owner"], lambda m: m.update(issued=[]))
    return s


def _sc_sealed_valid(tmp_path):
    return _sealed_share(tmp_path, expires_at=datetime.now(timezone.utc) + timedelta(days=3))


def _sc_sealed_wrap_deleted(tmp_path):
    from axon.security.share import revoke_sealed_share

    s = _sealed_share(tmp_path)
    revoke_sealed_share(s["owner"], "research", s["key_id"])
    return s


def _sc_sealed_expired(tmp_path):
    return _sealed_share(tmp_path, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))


def _sc_sealed_tampered(tmp_path):
    from axon.security.share import share_expiry_path

    s = _sealed_share(tmp_path, expires_at=datetime.now(timezone.utc) + timedelta(hours=1))
    path = share_expiry_path(s["project_dir"], s["key_id"])
    side = json.loads(path.read_text(encoding="utf-8"))
    side["expires_at"] = "2099-01-01T00:00:00Z"  # extend without the owner's key
    path.write_text(json.dumps(side), encoding="utf-8")
    return s


def _sc_sealed_target_missing(tmp_path):
    s = _sealed_share(tmp_path)
    s["project_dir"].rename(s["project_dir"].with_name("research_moved"))
    return s


def _sc_sealed_shares_dir_missing(tmp_path):
    s = _sealed_share(tmp_path)
    shutil.rmtree(s["project_dir"] / ".security" / "shares")
    return s


# (scenario, expected state, expected reason, kept after reconcile)
_MATRIX = [
    ("plain_valid", _sc_plain_valid, "valid", "ok", True),
    ("plain_revoked", _sc_plain_revoked, "revoked", "revoked", False),
    ("plain_expired", _sc_plain_expired, "expired", "expired", False),
    ("plain_manifest_missing", _sc_plain_manifest_missing, "unverifiable", None, True),
    ("plain_record_missing", _sc_plain_record_missing, "unverifiable", "record_absent", True),
    ("sealed_valid", _sc_sealed_valid, "valid", "ok", True),
    ("sealed_wrap_deleted", _sc_sealed_wrap_deleted, "revoked", "wrap_absent", False),
    ("sealed_expired", _sc_sealed_expired, "expired", "expired", False),
    ("sealed_tampered_sidecar", _sc_sealed_tampered, "expired", "expiry_unverified", False),
    ("sealed_target_missing", _sc_sealed_target_missing, "unverifiable", "target_missing", True),
    (
        "sealed_shares_dir_missing",
        _sc_sealed_shares_dir_missing,
        "unverifiable",
        "shares_dir_missing",
        True,
    ),
]


class TestOneAnswerMatrix:
    @pytest.mark.parametrize(
        "name,scenario,state,reason,kept", _MATRIX, ids=[m[0] for m in _MATRIX]
    )
    def test_every_surface_agrees(self, tmp_path, kr_backend, name, scenario, state, reason, kept):
        from axon.projects import list_share_mounts
        from axon.share_validity import ShareState, share_status

        if name.startswith("sealed") and not _HAS_SEALED:
            pytest.skip("requires axon-rag[sealed]")
        s = scenario(tmp_path)
        grantee, mount = s["grantee"], s["mount"]
        desc = load_mount_descriptor(grantee, mount)
        assert desc is not None

        # 1. share_status
        st = share_status(desc)
        assert st.state is ShareState(state)
        if reason is not None:
            assert st.reason == reason
        assert st.kind == s["kind"]
        valid = st.ok

        # 2. list_share_mounts
        [entry] = [m for m in list_share_mounts(grantee) if m["name"] == mount]
        assert entry["is_broken"] is (not valid)
        assert entry["state"] == state

        # 3. brain.switch_project
        verdict, _msg, brain = _switch(grantee, mount)
        assert verdict == ("allowed" if valid else "denied")
        # sealed EXPIRED at switch triggers the auto-destroy hygiene
        expect_destroy = s["kind"] == "sealed" and state == "expired"
        assert brain._auto_destroy_expired_share.called is expect_destroy

        # 4. per-query guard (auto-destroy is mocked here)
        exc, qbrain = _per_query(grantee, mount, desc)
        assert (exc is None) is valid
        assert qbrain._auto_destroy_expired_share.called is expect_destroy

        # 5. REST /projects (runs the one reconcile pass itself)
        body = _rest_projects(grantee)
        listed = [m for m in body["shared_mounts"] if m["name"] == f"mounts/{mount}"]
        assert bool(listed) is valid
        if listed:
            assert listed[0]["state"] == "valid"

        # 6. reconcile keep/remove
        assert (load_mount_descriptor(grantee, mount) is not None) is kept
        assert shares.validate_received_shares(grantee) == []  # idempotent
        if s["kind"] == "sealed" and state in ("revoked", "expired"):
            # soft revoke keeps the cached DEK; expiry deletes it
            assert _dek_present(s["key_id"]) is (state == "revoked")


# ---------------------------------------------------------------------------
# share_status / reconcile details
# ---------------------------------------------------------------------------


class TestShareStatusUnits:
    def test_malformed_descriptors_are_invalid_and_kept(self, tmp_path):
        from axon.share_validity import ShareState, share_status

        assert share_status(None).state is ShareState.INVALID
        assert share_status({"target_project_dir": str(tmp_path)}).reason == "key_id_missing"
        assert share_status({"share_key_id": "sk_1"}).reason == "target_unset"
        assert (
            share_status(
                {"share_key_id": "sk_1", "target_project_dir": str(tmp_path), "state": "gone"}
            ).reason
            == "descriptor_inactive"
        )
        assert (
            share_status({"share_key_id": "sk_1", "target_project_dir": str(tmp_path)}).reason
            == "owner_unset"
        )

    def test_share_status_is_read_only(self, tmp_path):
        """No directories are created in the owner's store by a check."""
        from axon.share_validity import share_status

        owner = tmp_path / "owner"
        target = owner / "proj"
        target.mkdir(parents=True)
        st = share_status(
            {
                "share_key_id": "sk_1",
                "target_project_dir": str(target),
                "owner_user_dir": str(owner),
            }
        )
        assert st.reason == "manifest_unreadable"
        assert not (owner / ".shares").exists()

    def test_plain_leeway_kept(self, tmp_path):
        from axon.share_validity import ShareState, share_status

        s = _plain_share(tmp_path, ttl_days=1)
        now = datetime.now(timezone.utc)
        _edit_manifest(
            s["owner"],
            lambda m: m["issued"][0].update(expires_at=(now - timedelta(seconds=30)).isoformat()),
        )
        desc = load_mount_descriptor(s["grantee"], s["mount"])
        assert share_status(desc, now=now).state is ShareState.VALID
        later = now + timedelta(minutes=10)
        assert share_status(desc, now=later).state is ShareState.EXPIRED

    def test_require_valid_raises_permission_error_with_status(self, tmp_path):
        from axon.share_validity import ShareInvalidError, ShareState, require_valid

        s = _sc_plain_revoked(tmp_path)
        with pytest.raises(PermissionError) as ei:
            require_valid(load_mount_descriptor(s["grantee"], s["mount"]))
        assert isinstance(ei.value, ShareInvalidError)
        assert ei.value.status.state is ShareState.REVOKED

    @needs_sealed
    def test_unreadable_expiry_sidecar_is_unverifiable_never_destroyed(
        self, tmp_path, kr_backend, monkeypatch
    ):
        """An OSError reading ``.expiry`` (cloud placeholder) must map to
        UNVERIFIABLE — never to the malformed → expired → auto-destroy path."""
        from axon.share_validity import ShareState, share_status

        s = _sealed_share(tmp_path, expires_at=datetime.now(timezone.utc) + timedelta(days=1))
        orig_read_bytes = Path.read_bytes
        orig_read_text = Path.read_text

        def _placeholder_bytes(self, *a, **kw):
            if self.suffix == ".expiry":
                raise OSError(22, "cloud file provider is not running")
            return orig_read_bytes(self, *a, **kw)

        def _placeholder_text(self, *a, **kw):
            if self.suffix == ".expiry":
                raise OSError(22, "cloud file provider is not running")
            return orig_read_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_bytes", _placeholder_bytes)
        monkeypatch.setattr(Path, "read_text", _placeholder_text)
        desc = load_mount_descriptor(s["grantee"], s["mount"])
        st = share_status(desc)
        assert st.state is ShareState.UNVERIFIABLE
        assert st.reason == "expiry_unreadable"
        exc, qbrain = _per_query(s["grantee"], s["mount"], desc)
        assert isinstance(exc, PermissionError)
        qbrain._auto_destroy_expired_share.assert_not_called()
        assert shares.validate_received_shares(s["grantee"]) == []
        assert mount_descriptor_path(s["grantee"], s["mount"]).exists()
        assert _dek_present(s["key_id"])

    @needs_sealed
    def test_sidecar_race_oserror_inside_check_is_unverifiable(self, tmp_path, kr_backend):
        """If the sidecar becomes unreadable between our read and
        ``_check_expiry_or_raise``'s read, still UNVERIFIABLE."""
        from axon.security import ShareExpiredError
        from axon.share_validity import ShareState, share_status

        s = _sealed_share(tmp_path, expires_at=datetime.now(timezone.utc) + timedelta(days=1))

        def _raise(*_a, **_kw):
            try:
                raise OSError("gone")
            except OSError as inner:
                raise ShareExpiredError("malformed") from inner

        with patch("axon.security.share._check_expiry_or_raise", _raise):
            st = share_status(load_mount_descriptor(s["grantee"], s["mount"]))
        assert st.state is ShareState.UNVERIFIABLE

    def test_reconcile_prunes_superseded_records_without_touching_descriptor(self, tmp_path):
        from axon.share_validity import reconcile_received_mounts

        s = _plain_share(tmp_path)
        keys_path = s["grantee"] / ".shares" / ".share_keys.json"
        keys = json.loads(keys_path.read_text())
        keys["received"].append({"key_id": "sk_old", "mount_name": s["mount"]})
        keys["received"].append({"key_id": "sk_orphan", "mount_name": "nobody_nothing"})
        keys_path.write_text(json.dumps(keys))
        assert reconcile_received_mounts(s["grantee"]) == []
        kept = json.loads(keys_path.read_text())["received"]
        assert [r["key_id"] for r in kept] == [s["key_id"]]
        assert load_mount_descriptor(s["grantee"], s["mount"]) is not None

    def test_reconcile_leaves_corrupt_keys_file_alone(self, tmp_path):
        from axon.share_validity import reconcile_received_mounts

        s = _sc_plain_revoked(tmp_path)
        keys_path = s["grantee"] / ".shares" / ".share_keys.json"
        keys_path.write_text("{ not json", encoding="utf-8")
        assert reconcile_received_mounts(s["grantee"]) == [s["mount"]]
        assert keys_path.read_text(encoding="utf-8") == "{ not json"

    def test_owner_share_status_plain(self, tmp_path):
        from axon.share_validity import ShareState, owner_share_status

        owner, _ = _store(tmp_path)
        a = shares.generate_share_key(owner, "research", "bob")
        assert owner_share_status(owner, "research", a["key_id"], "plain").ok
        shares.revoke_share_key(owner, a["key_id"])
        assert (
            owner_share_status(owner, "research", a["key_id"], "plain").state is ShareState.REVOKED
        )

    @needs_sealed
    def test_owner_share_status_sealed(self, tmp_path, kr_backend):
        from axon.security.share import generate_sealed_share, revoke_sealed_share
        from axon.share_validity import ShareState, owner_share_status

        owner, _ = _store(tmp_path)
        _seal(owner)
        generate_sealed_share(owner, "research", "bob", "ssk_live")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        generate_sealed_share(owner, "research", "bob", "ssk_old", expires_at=past)
        assert owner_share_status(owner, "research", "ssk_live", "sealed").ok
        assert (
            owner_share_status(owner, "research", "ssk_old", "sealed").state is ShareState.EXPIRED
        )
        revoke_sealed_share(owner, "research", "ssk_live")
        assert (
            owner_share_status(owner, "research", "ssk_live", "sealed").state is ShareState.REVOKED
        )

    def test_owner_status_ignores_a_forged_expiry(self, tmp_path, kr_backend):
        """The .expiry sidecar sits in the shared folder, which grantees can
        write. A grantee back-dating it must not make the owner's view say
        EXPIRED (that would unblock deleting a project they still hold)."""
        from axon.security.share import generate_sealed_share, share_expiry_path
        from axon.share_validity import ShareState, owner_share_status

        owner, _ = _store(tmp_path)
        proj = _seal(owner)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        generate_sealed_share(owner, "research", "bob", "ssk_live", expires_at=future)
        path = share_expiry_path(proj, "ssk_live")
        side = json.loads(path.read_text(encoding="utf-8"))
        side["expires_at"] = "2000-01-01T00:00:00Z"  # forged, signature no longer matches
        path.write_text(json.dumps(side), encoding="utf-8")

        st = owner_share_status(owner, "research", "ssk_live", "sealed")
        assert st.state is ShareState.VALID
        assert st.reason == "expiry_unverified"

    def test_owner_status_locked_store_keeps_share_active(self, tmp_path, kr_backend):
        from axon.security.master import lock_store
        from axon.security.share import generate_sealed_share
        from axon.share_validity import ShareState, owner_share_status

        owner, _ = _store(tmp_path)
        _seal(owner)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        generate_sealed_share(owner, "research", "bob", "ssk_old", expires_at=past)
        lock_store(owner)
        # Can't verify the signature -> conservative: still counts as active.
        assert owner_share_status(owner, "research", "ssk_old", "sealed").state is ShareState.VALID


class TestEmptyMountNameNeverWipesAllMounts:
    def test_per_query_auto_destroy_skipped_without_a_mount_name(self, tmp_path, kr_backend):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        s = _sealed_share(tmp_path, expires_at=past)
        desc = dict(load_mount_descriptor(s["grantee"], s["mount"]))
        desc["mount_name"] = ""
        exc, brain = _per_query(s["grantee"], s["mount"], descriptor=desc)
        assert isinstance(exc, PermissionError)
        brain._auto_destroy_expired_share.assert_not_called()

    def test_mount_descriptor_dir_rejects_the_mounts_root(self, tmp_path):
        from axon.mounts import mount_descriptor_dir, mounts_root, remove_mount_descriptor

        root = mounts_root(tmp_path)
        (root / "keep_me").mkdir(parents=True)
        for bad in ("", "."):
            with pytest.raises(ValueError):
                mount_descriptor_dir(tmp_path, bad)
            with pytest.raises(ValueError):
                remove_mount_descriptor(tmp_path, bad)
        assert (root / "keep_me").is_dir()


# ---------------------------------------------------------------------------
# No key material in logs, details or exception messages
# ---------------------------------------------------------------------------


class TestNoSecretsLeak:
    def _secrets_plain(self, s: dict) -> list[str]:
        raw = base64.urlsafe_b64decode(s["share_string"]).decode()
        token = raw.split(":")[1]
        keys = json.loads((s["owner"] / ".shares" / ".share_keys.json").read_text())
        hmacs = [r["token_hmac"] for r in keys["issued"]]
        return [s["share_string"], token, *hmacs]

    @pytest.mark.parametrize(
        "scenario",
        [_sc_plain_valid, _sc_plain_revoked, _sc_plain_expired, _sc_plain_manifest_missing],
    )
    def test_plain_paths_never_emit_secrets(self, tmp_path, caplog, scenario):
        from axon.share_validity import share_status

        s = scenario(tmp_path)
        secrets = self._secrets_plain(s) if _manifest_path(s["owner"]).exists() else []
        if not secrets:
            secrets = [s["share_string"], base64.urlsafe_b64decode(s["share_string"]).decode()]
        caplog.set_level(logging.DEBUG)
        texts = [share_status(load_mount_descriptor(s["grantee"], s["mount"])).detail]
        _verdict, msg, _ = _switch(s["grantee"], s["mount"])
        texts.append(msg)
        exc, _ = _per_query(s["grantee"], s["mount"])
        texts.append(str(exc or ""))
        body = _rest_projects(s["grantee"])
        texts.append(json.dumps(body))
        shares.validate_received_shares(s["grantee"])
        texts.append(caplog.text)
        blob = "\n".join(texts)
        for secret in secrets:
            assert secret not in blob

    def test_corrupt_store_error_names_file_not_contents(self, tmp_path):
        owner, _ = _store(tmp_path)
        a = shares.generate_share_key(owner, "research", "bob")
        keys_path = owner / ".shares" / ".share_keys.json"
        token_hmac = json.loads(keys_path.read_text())["issued"][0]["token_hmac"]
        keys_path.write_text(keys_path.read_text()[:-5], encoding="utf-8")  # truncate
        with pytest.raises(RuntimeError) as ei:
            shares.revoke_share_key(owner, a["key_id"])
        assert ".share_keys.json" in str(ei.value)
        assert token_hmac not in str(ei.value)
        assert a["share_string"] not in str(ei.value)
        # ``from None`` — the JSONDecodeError (which can quote file content)
        # is not chained into tracebacks.
        assert ei.value.__cause__ is None and ei.value.__suppress_context__

    @needs_sealed
    def test_sealed_paths_never_emit_secrets(self, tmp_path, kr_backend, caplog):
        from axon.share_validity import share_status

        s = _sc_sealed_expired(tmp_path)
        from axon.security import keyring as _kr
        from axon.security.share import _share_keyring_service

        dek_b64 = _kr.get_secret(_share_keyring_service(s["key_id"]), "dek")
        raw = base64.urlsafe_b64decode(s["share_string"]).decode()
        # The envelope's secret token is its long hex field; the owner's
        # signing pubkey is also hex but public (it is stored in mount.json).
        pubkey = load_mount_descriptor(s["grantee"], s["mount"]).get("owner_pubkey_hex")
        tokens = [
            f
            for f in raw.split(":")
            if len(f) >= 32 and f != pubkey and all(c in "0123456789abcdefABCDEF" for c in f)
        ]
        assert tokens, "expected to find the share token in the envelope"
        secrets = [s["share_string"], dek_b64, *tokens]
        caplog.set_level(logging.DEBUG)
        texts = [share_status(load_mount_descriptor(s["grantee"], s["mount"])).detail]
        _verdict, msg, _ = _switch(s["grantee"], s["mount"])
        texts.append(msg)
        # Real auto-destroy this time (bound method) so its logging is covered.
        brain = MagicMock()
        brain.config.projects_root = str(s["grantee"])
        brain._sealed_cache = None
        brain._active_project_kind = "mounted"
        brain._active_mount_descriptor = load_mount_descriptor(s["grantee"], s["mount"])
        brain._auto_destroy_expired_share = MethodType(AxonBrain._auto_destroy_expired_share, brain)
        with pytest.raises(PermissionError) as ei:
            QueryRouterMixin._check_mount_revocation(brain)
        texts.append(str(ei.value))
        texts.append(caplog.text)
        blob = "\n".join(texts)
        for secret in secrets:
            if secret:
                assert secret not in blob
        assert load_mount_descriptor(s["grantee"], s["mount"]) is None
        assert not _dek_present(s["key_id"])
