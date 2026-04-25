"""Phase 5: cross-interface tests for sealed share surfaces.

Verifies the REST routes (``/share/redeem``, ``/share/revoke``) and the
MCP wrappers (``revoke_share``, ``share_project``, ``redeem_share``)
correctly route sealed shares vs legacy plaintext shares.

Skips when ``cryptography`` / ``keyring`` aren't installed.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")


@pytest.fixture
def stub_brain(tmp_path):
    """Install a minimal brain on axon.api so /share/* routes' user_dir
    lookup returns a real tmp Path instead of 503."""
    import axon.api as api_module

    brain = MagicMock()
    brain.config.projects_root = str(tmp_path / "AxonStore" / "tester")
    Path(brain.config.projects_root).mkdir(parents=True, exist_ok=True)
    api_module.brain = brain
    yield brain
    api_module.brain = None


# ---------------------------------------------------------------------------
# REST: /share/redeem auto-detects SEALED1: prefix
# ---------------------------------------------------------------------------


class TestRestRedeemSealedDetection:
    def test_sealed_envelope_routes_to_security_redeem(self, stub_brain):
        """A SEALED1: envelope must be routed to redeem_sealed_share,
        NOT the legacy redeem_share_key.
        """
        from fastapi.testclient import TestClient

        from axon.api import app
        from axon.api_routes.shares import (  # noqa: F401 — ensures route registered
            share_redeem,
        )

        # SEALED1: envelope (decoded form). Doesn't need to be valid —
        # we only verify the routing dispatches to the sealed branch.
        sealed_payload = "SEALED1:ssk_test:abc:alice:research:/store"
        share_str = base64.urlsafe_b64encode(sealed_payload.encode()).decode("ascii")

        with patch(
            "axon.security.redeem_sealed_share",
            return_value={
                "key_id": "ssk_test",
                "mount_name": "alice_research",
                "owner": "alice",
                "project": "research",
                "sealed": True,
            },
        ) as sealed_mock:
            client = TestClient(app)
            resp = client.post("/share/redeem", json={"share_string": share_str})

        assert resp.status_code == 200
        body = resp.json()
        assert body["sealed"] is True
        assert body["key_id"] == "ssk_test"
        sealed_mock.assert_called_once()
        # share_string was passed through unchanged.
        _, kwargs = sealed_mock.call_args
        called_args = sealed_mock.call_args[0]
        assert share_str in called_args

    def test_legacy_envelope_routes_to_legacy_redeem(self, stub_brain):
        """A non-SEALED1 envelope must use the legacy redeem path."""
        from fastapi.testclient import TestClient

        from axon.api import app

        legacy_str = base64.urlsafe_b64encode(b"sk_legacy:tok:alice:research:/store").decode(
            "ascii"
        )

        with patch(
            "axon.shares.redeem_share_key",
            return_value={
                "key_id": "sk_legacy",
                "mount_name": "alice_research",
                "owner": "alice",
                "project": "research",
            },
        ) as legacy_mock:
            client = TestClient(app)
            resp = client.post("/share/redeem", json={"share_string": legacy_str})

        assert resp.status_code == 200
        legacy_mock.assert_called_once()


# ---------------------------------------------------------------------------
# REST: /share/revoke routes ssk_ key_ids to sealed revoke
# ---------------------------------------------------------------------------


class TestRestRevokeSealedRouting:
    def test_ssk_key_id_routes_to_sealed_revoke(self, stub_brain):
        from fastapi.testclient import TestClient

        from axon.api import app

        with patch(
            "axon.security.revoke_sealed_share",
            return_value={
                "status": "soft_revoked",
                "key_id": "ssk_abc",
                "rotate": False,
                "wrap_deleted": True,
            },
        ) as sealed_mock:
            client = TestClient(app)
            resp = client.post(
                "/share/revoke",
                json={"key_id": "ssk_abc", "project": "research"},
            )

        assert resp.status_code == 200
        sealed_mock.assert_called_once()
        _, kwargs = sealed_mock.call_args
        assert kwargs["project"] == "research"
        assert kwargs["key_id"] == "ssk_abc"
        assert kwargs["rotate"] is False

    def test_ssk_key_id_with_rotate_passes_through(self, stub_brain):
        from fastapi.testclient import TestClient

        from axon.api import app

        with patch(
            "axon.security.revoke_sealed_share",
            return_value={
                "status": "hard_revoked",
                "key_id": "ssk_hard",
                "rotate": True,
                "wrap_deleted": True,
                "files_resealed": 5,
                "invalidated_share_key_ids": ["ssk_hard"],
            },
        ) as sealed_mock:
            client = TestClient(app)
            resp = client.post(
                "/share/revoke",
                json={"key_id": "ssk_hard", "project": "research", "rotate": True},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "hard_revoked"
        assert body["rotate"] is True
        _, kwargs = sealed_mock.call_args
        assert kwargs["rotate"] is True

    def test_ssk_without_project_returns_422(self, stub_brain):
        from fastapi.testclient import TestClient

        from axon.api import app

        client = TestClient(app)
        resp = client.post("/share/revoke", json={"key_id": "ssk_no_project"})
        assert resp.status_code == 422
        assert "project" in resp.json()["detail"].lower()

    def test_missing_wrap_returns_404(self, stub_brain):
        """A revoke for a key_id whose wrap file doesn't exist must
        return 404, not 400 — matches the standard REST shape."""
        from fastapi.testclient import TestClient

        from axon.api import app
        from axon.security import SecurityError

        with patch(
            "axon.security.revoke_sealed_share",
            side_effect=SecurityError(
                "No sealed-share wrap exists for key_id='ssk_phantom' at "
                "/store/research/.security/shares/ssk_phantom.wrapped. "
                "Either it was already revoked or never generated."
            ),
        ):
            client = TestClient(app)
            resp = client.post(
                "/share/revoke",
                json={"key_id": "ssk_phantom", "project": "research"},
            )
        assert resp.status_code == 404
        assert "no sealed-share wrap" in resp.json()["detail"].lower()

    def test_unsealed_project_returns_404(self, stub_brain):
        """Project not sealed → 404 (not 400) so REST clients can
        distinguish 'project doesn't accept this operation' from
        'invalid request body'."""
        from fastapi.testclient import TestClient

        from axon.api import app
        from axon.security import SecurityError

        with patch(
            "axon.security.revoke_sealed_share",
            side_effect=SecurityError("Project 'open' is not sealed; nothing to revoke."),
        ):
            client = TestClient(app)
            resp = client.post(
                "/share/revoke",
                json={"key_id": "ssk_x", "project": "open"},
            )
        assert resp.status_code == 404

    def test_locked_store_returns_400(self, stub_brain):
        """Store-locked errors are 400 (transient, fix and retry), not 404."""
        from fastapi.testclient import TestClient

        from axon.api import app
        from axon.security import SecurityError

        with patch(
            "axon.security.revoke_sealed_share",
            side_effect=SecurityError(
                "Store axon.master.alice is locked. Call unlock_store first."
            ),
        ):
            client = TestClient(app)
            resp = client.post(
                "/share/revoke",
                json={"key_id": "ssk_x", "project": "research", "rotate": True},
            )
        assert resp.status_code == 400

    def test_legacy_key_id_routes_to_legacy_revoke(self, stub_brain):
        from fastapi.testclient import TestClient

        from axon.api import app

        with patch(
            "axon.shares.revoke_share_key",
            return_value={"key_id": "sk_legacy", "project": "research"},
        ) as legacy_mock:
            client = TestClient(app)
            resp = client.post("/share/revoke", json={"key_id": "sk_legacy"})

        assert resp.status_code == 200
        legacy_mock.assert_called_once()


# ---------------------------------------------------------------------------
# MCP: revoke_share forwards rotate + project flags
# ---------------------------------------------------------------------------


class TestMcpRevokeShare:
    def test_legacy_revoke_only_sends_key_id(self):
        from axon.mcp_server import revoke_share

        async def _run():
            mock = AsyncMock(return_value={"key_id": "sk_x"})
            with patch("axon.mcp_server._post", mock):
                await revoke_share("sk_x")
            args, _ = mock.call_args
            assert args[0] == "/share/revoke"
            assert args[1] == {"key_id": "sk_x"}

        asyncio.run(_run())

    def test_sealed_revoke_with_project_and_rotate(self):
        from axon.mcp_server import revoke_share

        async def _run():
            mock = AsyncMock(return_value={"status": "hard_revoked"})
            with patch("axon.mcp_server._post", mock):
                await revoke_share(key_id="ssk_x", project="research", rotate=True)
            args, _ = mock.call_args
            body = args[1]
            assert body == {
                "key_id": "ssk_x",
                "project": "research",
                "rotate": True,
            }

        asyncio.run(_run())

    def test_sealed_revoke_default_rotate_false_omits_field(self):
        """When rotate=False (default), we omit it from the body so the
        wire format stays minimal — matches the legacy revoke_share
        omits-when-default convention used elsewhere."""
        from axon.mcp_server import revoke_share

        async def _run():
            mock = AsyncMock(return_value={"status": "soft_revoked"})
            with patch("axon.mcp_server._post", mock):
                await revoke_share(key_id="ssk_x", project="research")
            args, _ = mock.call_args
            body = args[1]
            assert body == {"key_id": "ssk_x", "project": "research"}

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Surface registration: revoke_share is still in EXPECTED_MCP_TOOL_NAMES
# (the existing test_mcp_server.py contract test)
# ---------------------------------------------------------------------------


class TestMcpToolRegistration:
    def test_revoke_share_still_registered(self):
        from axon.mcp_server import mcp

        async def _run():
            tools_resp = await mcp.list_tools()
            tools = tools_resp.tools if hasattr(tools_resp, "tools") else tools_resp
            names = [t.name for t in tools]
            assert "revoke_share" in names

        asyncio.run(_run())
