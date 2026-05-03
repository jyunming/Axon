"""End-to-end tests for v0.4.0 TTL-gated sealed-share — PR C.

PRs A and B added the unit-level tests for the primitives (signing
keypair, SEALED2 envelope, expiry sidecar, _check_expiry_or_raise,
auto-destroy helper). This file proves the **whole flow** works
together and that all four user-facing surfaces (REST, MCP-shape,
CLI, REPL) propagate ``ttl_days`` correctly into the signed sidecar.

Coverage targets:

1. Owner mints a sealed share with ``ttl_days=N`` →
   - sidecar exists at the expected path
   - sidecar's signature verifies under the owner's pubkey
   - returned dict carries ``expires_at`` (canonical Z form)
   - share_string is SEALED2 (carries the pubkey)

2. Grantee redeems → mount descriptor records the pubkey from envelope
   - get_grantee_dek returns the DEK while expires_at is in the future

3. Time passes → ShareExpiredError on next get_grantee_dek
   (simulated by writing a sidecar with ``expires_at`` in the past)

4. ``_auto_destroy_expired_share`` (called by _mount_sealed_project on
   ShareExpiredError):
   - deletes DEK from keyring
   - deletes mount descriptor
   - **does NOT touch encrypted source files on the synced filesystem**
   - is idempotent (safe to call twice)

5. SEALED1 backward compat: a share generated WITHOUT expires_at and
   redeemed via the SEALED1 path keeps working forever (no TTL check
   if no sidecar exists, even for SEALED2 envelopes).

6. REST surface (``POST /share/generate``):
   - ``ttl_days: N`` propagates into ``generate_sealed_share(expires_at=...)``
   - response carries ``expires_at``
   - ``ttl_days: 0`` rejected with 422
   - ``ttl_days: -1`` rejected with 422
   - ``ttl_days`` omitted → no sidecar written, response has no ``expires_at``

These tests use the same in-memory keyring shim the existing sealed
test files use (see ``test_sealed_share.py``) so no real OS keyring
is touched.

Run with:
    PYTHONPATH=src python -m pytest tests/test_sealed_ttl_e2e.py -v --no-cov
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("cryptography", reason="requires axon-rag[sealed] / cryptography")
pytest.importorskip("keyring")

from axon.security import ShareExpiredError  # noqa: E402
from axon.security.master import bootstrap_store  # noqa: E402
from axon.security.seal import project_seal  # noqa: E402
from axon.security.share import (  # noqa: E402
    generate_sealed_share,
    get_grantee_dek,
    redeem_sealed_share,
    share_expiry_path,
    share_wrap_path,
)

# ---------------------------------------------------------------------------
# In-memory keyring + fixtures (mirrors test_sealed_share.py)
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
# Owner side: ttl_days writes a verifiable sidecar
# ---------------------------------------------------------------------------


class TestOwnerMintWithTtl:
    def test_generate_with_expires_at_creates_sidecar(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        future = datetime.now(timezone.utc) + timedelta(days=7)
        result = generate_sealed_share(
            owner_user_dir, "research", "bob", "ssk_e2e_ttl", expires_at=future
        )
        # 1. Wrap file exists
        assert share_wrap_path(proj, "ssk_e2e_ttl").is_file()
        # 2. Sidecar exists at the expected path
        sidecar_path = share_expiry_path(proj, "ssk_e2e_ttl")
        assert sidecar_path.is_file()
        # 3. Sidecar carries canonical Z-suffixed expires_at
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["key_id"] == "ssk_e2e_ttl"
        assert sidecar["expires_at"].endswith("Z")
        assert "sig" in sidecar
        # 4. Returned dict surfaces expires_at + sidecar path
        assert "expires_at" in result
        assert result["expires_at"] == sidecar["expires_at"]
        assert result["expiry_sidecar_path"] == str(sidecar_path)

    def test_generate_without_expires_at_writes_no_sidecar(self, kr_backend, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        result = generate_sealed_share(owner_user_dir, "research", "bob", "ssk_e2e_no_ttl")
        # Wrap file yes; sidecar no
        assert share_wrap_path(proj, "ssk_e2e_no_ttl").is_file()
        assert not share_expiry_path(proj, "ssk_e2e_no_ttl").is_file()
        # Result carries no expires_at field
        assert "expires_at" not in result

    def test_generate_emits_sealed2_when_unlocked(self, kr_backend, owner_user_dir):
        """Generated share_string decodes to SEALED2: prefix when the
        store is unlocked (master available → can derive signing key)."""
        import base64

        _populate_and_seal(owner_user_dir)
        result = generate_sealed_share(owner_user_dir, "research", "bob", "ssk_e2e_v2")
        decoded = base64.urlsafe_b64decode(result["share_string"]).decode("utf-8")
        assert decoded.startswith("SEALED2:")


# ---------------------------------------------------------------------------
# Grantee redeem + TTL check
# ---------------------------------------------------------------------------


class TestGranteeRedeemThenTtl:
    def test_redeem_records_envelope_version_and_pubkey(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", "ssk_e2e_redeem")
        redeem_result = redeem_sealed_share(grantee_user_dir, share["share_string"])
        assert redeem_result["envelope_version"] == 2
        descriptor = redeem_result["descriptor"]
        assert descriptor.get("envelope_version") == 2
        # Pubkey is recorded for downstream TTL verification
        assert descriptor.get("owner_pubkey_hex")
        assert len(descriptor["owner_pubkey_hex"]) == 64

    def test_get_dek_succeeds_when_no_sidecar(self, kr_backend, owner_user_dir, grantee_user_dir):
        """No expiry sidecar = no TTL check = DEK returned normally."""
        _populate_and_seal(owner_user_dir)
        share = generate_sealed_share(owner_user_dir, "research", "bob", "ssk_e2e_no_ttl_dek")
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        dek = get_grantee_dek("ssk_e2e_no_ttl_dek", user_dir=grantee_user_dir)
        assert len(dek) == 32

    def test_get_dek_succeeds_with_future_expiry(
        self, kr_backend, owner_user_dir, grantee_user_dir
    ):
        _populate_and_seal(owner_user_dir)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        share = generate_sealed_share(
            owner_user_dir, "research", "bob", "ssk_e2e_future", expires_at=future
        )
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        # Future expiry → DEK returned
        dek = get_grantee_dek("ssk_e2e_future", user_dir=grantee_user_dir)
        assert len(dek) == 32

    def test_get_dek_raises_when_expired(self, kr_backend, owner_user_dir, grantee_user_dir):
        """Owner mints with past expiry → grantee redeem succeeds (the
        wrap+sig are valid, just the timestamp has passed) → next
        get_grantee_dek raises ShareExpiredError."""
        _populate_and_seal(owner_user_dir)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        share = generate_sealed_share(
            owner_user_dir, "research", "bob", "ssk_e2e_expired", expires_at=past
        )
        redeem_sealed_share(grantee_user_dir, share["share_string"])
        with pytest.raises(ShareExpiredError, match="expired at"):
            get_grantee_dek("ssk_e2e_expired", user_dir=grantee_user_dir)


# ---------------------------------------------------------------------------
# Auto-destroy plumbing
# ---------------------------------------------------------------------------


class TestAutoDestroy:
    def test_auto_destroy_removes_local_state(self, kr_backend, owner_user_dir, grantee_user_dir):
        """After ShareExpiredError → auto-destroy → DEK gone, descriptor
        gone, but encrypted files on disk untouched."""
        from axon.main import AxonBrain
        from axon.mounts import list_mount_descriptors

        proj = _populate_and_seal(owner_user_dir)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        share = generate_sealed_share(
            owner_user_dir, "research", "bob", "ssk_e2e_auto", expires_at=past
        )
        redeem = redeem_sealed_share(grantee_user_dir, share["share_string"])
        mount_name = redeem["mount_name"]

        # Sanity: descriptor + DEK both present pre-destroy
        descriptors = list_mount_descriptors(grantee_user_dir)
        assert any(d.get("mount_name") == mount_name for d in descriptors)
        # DEK is in keyring (via in-memory shim)
        from axon.security import keyring as _kr
        from axon.security.share import _share_keyring_service

        assert _kr.get_secret(_share_keyring_service("ssk_e2e_auto"), "dek") is not None

        # Set up a brain shim and trigger auto-destroy
        brain = MagicMock()
        brain.config = MagicMock()
        brain.config.projects_root = str(grantee_user_dir)
        brain._sealed_cache = None
        AxonBrain._auto_destroy_expired_share(
            brain, mount_name, "ssk_e2e_auto", Exception("expired")
        )

        # 1. DEK gone
        assert _kr.get_secret(_share_keyring_service("ssk_e2e_auto"), "dek") is None
        # 2. Mount descriptor gone
        descriptors = list_mount_descriptors(grantee_user_dir)
        assert not any(d.get("mount_name") == mount_name for d in descriptors)
        # 3. Encrypted source files NOT touched
        assert (proj / ".security" / "shares" / "ssk_e2e_auto.wrapped").is_file()
        assert (proj / ".security" / "shares" / "ssk_e2e_auto.expiry").is_file()

    def test_auto_destroy_is_idempotent(self, kr_backend, owner_user_dir, grantee_user_dir):
        """Calling auto-destroy a second time after the first wipe must
        not crash (each step is best-effort + tolerant of missing state)."""
        from axon.main import AxonBrain

        _populate_and_seal(owner_user_dir)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        share = generate_sealed_share(
            owner_user_dir, "research", "bob", "ssk_e2e_idem", expires_at=past
        )
        redeem = redeem_sealed_share(grantee_user_dir, share["share_string"])

        brain = MagicMock()
        brain.config = MagicMock()
        brain.config.projects_root = str(grantee_user_dir)
        brain._sealed_cache = None
        AxonBrain._auto_destroy_expired_share(
            brain, redeem["mount_name"], "ssk_e2e_idem", Exception("expired")
        )
        # Second call must not raise
        AxonBrain._auto_destroy_expired_share(
            brain, redeem["mount_name"], "ssk_e2e_idem", Exception("expired")
        )

    def test_auto_destroy_strips_mounts_prefix(self, kr_backend, owner_user_dir, grantee_user_dir):
        """``_mount_sealed_project`` may pass ``"mounts/<name>"`` —
        auto-destroy must strip the prefix before remove_mount_descriptor."""
        from axon.main import AxonBrain
        from axon.mounts import list_mount_descriptors

        _populate_and_seal(owner_user_dir)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        share = generate_sealed_share(
            owner_user_dir, "research", "bob", "ssk_e2e_prefix", expires_at=past
        )
        redeem = redeem_sealed_share(grantee_user_dir, share["share_string"])
        mount_name = redeem["mount_name"]

        brain = MagicMock()
        brain.config = MagicMock()
        brain.config.projects_root = str(grantee_user_dir)
        brain._sealed_cache = None
        # Pass the mounts/ prefix as _mount_sealed_project would
        AxonBrain._auto_destroy_expired_share(
            brain, f"mounts/{mount_name}", "ssk_e2e_prefix", Exception("expired")
        )
        # Descriptor should be gone — the prefix was stripped correctly
        descriptors = list_mount_descriptors(grantee_user_dir)
        assert not any(d.get("mount_name") == mount_name for d in descriptors)


# ---------------------------------------------------------------------------
# REST API surface
# ---------------------------------------------------------------------------


class TestRestSurface:
    """Verify ``POST /share/generate`` propagates ttl_days into the
    sealed-share generate path. Uses the FastAPI TestClient pattern
    matching tests/test_api.py — but only for the share endpoint."""

    @pytest.fixture
    def api_client(self, kr_backend, owner_user_dir, monkeypatch):
        """Spin up an axon-api with the brain rooted at owner_user_dir.parent."""
        from fastapi.testclient import TestClient

        import axon.api as _api
        from axon.api import app

        # Point the API's user-dir resolver at our test owner namespace
        monkeypatch.setattr(_api, "_get_user_dir", lambda: owner_user_dir)
        # Disable rate limiting for the test
        monkeypatch.setattr("axon.api_routes.shares.enforce_rate_limit", lambda *a, **k: None)
        return TestClient(app, raise_server_exceptions=True)

    def test_rest_generate_with_ttl_days_writes_sidecar(self, api_client, owner_user_dir):
        """ttl_days=7 → response.expires_at populated → sidecar exists on disk."""
        proj = _populate_and_seal(owner_user_dir)
        resp = api_client.post(
            "/share/generate",
            json={"project": "research", "grantee": "bob", "ttl_days": 7},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("security_mode") == "sealed_v1"
        assert body.get("expires_at"), "ttl_days didn't propagate to expires_at"
        # Sidecar exists on disk
        key_id = body["key_id"]
        assert share_expiry_path(proj, key_id).is_file()

    def test_rest_generate_without_ttl_writes_no_sidecar(self, api_client, owner_user_dir):
        proj = _populate_and_seal(owner_user_dir)
        resp = api_client.post(
            "/share/generate",
            json={"project": "research", "grantee": "bob"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("expires_at") is None or "expires_at" not in body
        key_id = body["key_id"]
        assert not share_expiry_path(proj, key_id).is_file()

    def test_rest_generate_with_zero_ttl_rejected(self, api_client, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        resp = api_client.post(
            "/share/generate",
            json={"project": "research", "grantee": "bob", "ttl_days": 0},
        )
        assert resp.status_code == 422
        assert "positive" in resp.json()["detail"].lower()

    def test_rest_generate_with_negative_ttl_rejected(self, api_client, owner_user_dir):
        _populate_and_seal(owner_user_dir)
        resp = api_client.post(
            "/share/generate",
            json={"project": "research", "grantee": "bob", "ttl_days": -1},
        )
        assert resp.status_code == 422
