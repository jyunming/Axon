"""REST changes behind the 0.5.0 agent-surface consolidation.

* ``POST /config/set`` accepts a ``settings`` batch (the MCP / VS Code
  ``set_config`` tool): all keys resolved first, unknown keys reject the whole
  batch, runtime components reinitialised once, config saved once.
* ``POST /ingest`` and ``POST /ingest/refresh`` accept a ``project``
  *assertion* (409 on mismatch). The old MCP ``refresh_ingest(project=...)``
  silently switched projects instead; ``ingest_knowledge`` now sends the
  assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


@dataclass
class FakeConfig:
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    hybrid_search: bool = True
    top_k: int = 5
    rerank: bool = False
    openai_api_key: str = "sk-old-secret"
    chunk_strategy: str = "recursive"

    def save(self):
        return None


@pytest.fixture(autouse=True)
def _reset_brain():
    original = api_module.brain
    yield
    api_module.brain = original


@pytest.fixture
def brain():
    b = MagicMock()
    b.config = FakeConfig()
    b.config.save = MagicMock()
    b._active_project = "alpha"
    api_module.brain = b
    return b


# ---------------------------------------------------------------------------
# POST /config/set — batch form
# ---------------------------------------------------------------------------


class TestConfigSetBatch:
    def test_batch_applies_every_key_and_reports_each(self, brain):
        resp = client.post(
            "/config/set",
            json={"settings": {"rag.top_k": 9, "hybrid_search": False}, "persist": False},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "success"
        assert data["persisted"] is False
        assert brain.config.top_k == 9
        assert brain.config.hybrid_search is False
        applied = {a["key"]: a for a in data["applied"]}
        assert applied["rag.top_k"] == {
            "key": "rag.top_k",
            "flat_key": "top_k",
            "old_value": 5,
            "new_value": 9,
        }
        assert applied["hybrid_search"]["flat_key"] == "hybrid_search"
        brain.config.save.assert_not_called()

    def test_unknown_key_rejects_the_whole_batch(self, brain):
        resp = client.post(
            "/config/set",
            json={"settings": {"top_k": 9, "nope_one": 1, "rag.nope_two": 2}, "persist": True},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "nope_one" in detail and "rag.nope_two" in detail
        assert "nothing was applied" in detail
        assert brain.config.top_k == 5  # the valid key was NOT applied
        brain.config.save.assert_not_called()

    def test_persist_saves_exactly_once(self, brain):
        resp = client.post(
            "/config/set",
            json={"settings": {"top_k": 7, "chunk.strategy": "markdown"}, "persist": True},
        )
        assert resp.status_code == 200
        assert resp.json()["persisted"] is True
        brain.config.save.assert_called_once_with()

    def test_runtime_components_reinitialised_once_for_all_keys(self, brain):
        with patch("axon.api_routes.config_routes._reinitialize_runtime_components") as reinit:
            resp = client.post(
                "/config/set",
                json={"settings": {"llm.model": "m2", "llm.provider": "openai"}, "persist": False},
            )
        assert resp.status_code == 200
        reinit.assert_called_once()
        assert reinit.call_args.args[1] == {"llm_model", "llm_provider"}

    def test_secrets_are_masked_in_applied(self, brain):
        resp = client.post(
            "/config/set",
            json={"settings": {"llm.openai_api_key": "sk-new-secret"}, "persist": False},
        )
        assert resp.status_code == 200
        (entry,) = resp.json()["applied"]
        assert "sk-old-secret" not in resp.text and "sk-new-secret" not in resp.text
        assert entry["old_value"] == "***" and entry["new_value"] == "***"
        assert brain.config.openai_api_key == "sk-new-secret"

    def test_none_is_a_legitimate_value(self, brain):
        resp = client.post("/config/set", json={"settings": {"llm_model": None}})
        assert resp.status_code == 200
        assert brain.config.llm_model is None

    def test_empty_settings_is_400(self, brain):
        resp = client.post("/config/set", json={"settings": {}})
        assert resp.status_code == 400

    def test_key_and_settings_together_is_400(self, brain):
        resp = client.post(
            "/config/set", json={"key": "top_k", "value": 3, "settings": {"top_k": 4}}
        )
        assert resp.status_code == 400
        assert brain.config.top_k == 5

    def test_neither_key_nor_settings_is_400(self, brain):
        resp = client.post("/config/set", json={"persist": False})
        assert resp.status_code == 400

    def test_batch_persist_defaults_to_true_like_single_key(self, brain):
        """REST keeps its historical default; only the agent tools default False."""
        resp = client.post("/config/set", json={"settings": {"top_k": 6}})
        assert resp.status_code == 200
        assert resp.json()["persisted"] is True
        brain.config.save.assert_called_once_with()


class TestConfigSetSingleKeyBackCompat:
    def test_single_key_shape_unchanged(self, brain):
        resp = client.post("/config/set", json={"key": "rag.top_k", "value": 11, "persist": False})
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "success",
            "key": "rag.top_k",
            "flat_key": "top_k",
            "old_value": 5,
            "new_value": 11,
            "persisted": False,
        }

    def test_single_key_null_value_is_applied(self, brain):
        resp = client.post(
            "/config/set", json={"key": "llm_model", "value": None, "persist": False}
        )
        assert resp.status_code == 200
        assert brain.config.llm_model is None

    def test_single_key_without_value_is_400(self, brain):
        resp = client.post("/config/set", json={"key": "top_k"})
        assert resp.status_code == 400
        assert brain.config.top_k == 5


# ---------------------------------------------------------------------------
# POST /ingest and /ingest/refresh — project assertion
# ---------------------------------------------------------------------------


class TestIngestProjectAssertion:
    def test_ingest_mismatched_project_is_409_before_touching_the_path(self, brain):
        resp = client.post("/ingest", json={"path": "/does/not/matter", "project": "beta"})
        assert resp.status_code == 409
        assert "alpha" in resp.json()["detail"]
        brain._assert_write_allowed.assert_not_called()

    def test_ingest_matching_project_proceeds(self, brain, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_BASE", str(tmp_path))
        (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
        resp = client.post("/ingest", json={"path": str(tmp_path / "doc.txt"), "project": "alpha"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["job_id"]

    def test_refresh_without_body_still_works(self, brain):
        brain.get_doc_versions.return_value = {}
        resp = client.post("/ingest/refresh")
        assert resp.status_code == 200
        assert resp.json()["job_id"]

    def test_refresh_with_empty_body_still_works(self, brain):
        brain.get_doc_versions.return_value = {}
        resp = client.post("/ingest/refresh", json={})
        assert resp.status_code == 200
        assert resp.json()["job_id"]

    def test_refresh_matching_project_proceeds(self, brain):
        brain.get_doc_versions.return_value = {}
        resp = client.post("/ingest/refresh", json={"project": "alpha"})
        assert resp.status_code == 200

    def test_refresh_mismatched_project_is_409_and_never_switches(self, brain):
        resp = client.post("/ingest/refresh", json={"project": "beta"})
        assert resp.status_code == 409
        brain.switch_project.assert_not_called()
        brain._assert_write_allowed.assert_not_called()
        brain.get_doc_versions.assert_not_called()


# ---------------------------------------------------------------------------
# /config/set — all-or-nothing when reinitialising a component fails
# ---------------------------------------------------------------------------


class TestConfigSetRollback:
    def test_batch_reinit_failure_restores_every_field(self, brain):
        before = dict(vars(brain.config))
        calls = []

        def _reinit(b, keys):
            calls.append(dict(vars(b.config)))
            if len(calls) == 1:
                raise ImportError("sentence-transformers is not installed")

        with patch("axon.api_routes.config_routes._reinitialize_runtime_components", _reinit):
            resp = client.post(
                "/config/set",
                json={
                    "settings": {"top_k": 9, "llm_provider": "openai"},
                    "persist": True,
                },
            )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "ImportError" in detail and "Nothing was applied" in detail
        assert dict(vars(brain.config)) == before
        brain.config.save.assert_not_called()
        # The first (failed) attempt saw the new values; the rebuild saw the old ones.
        assert calls[0]["top_k"] == 9
        assert calls[1]["top_k"] == 5

    def test_single_key_reinit_failure_restores_the_field(self, brain):
        with patch(
            "axon.api_routes.config_routes._reinitialize_runtime_components",
            side_effect=[RuntimeError("boom"), None],
        ):
            resp = client.post(
                "/config/set", json={"key": "llm.model", "value": "m2", "persist": True}
            )
        assert resp.status_code == 400
        assert brain.config.llm_model == "llama3"
        brain.config.save.assert_not_called()


# ---------------------------------------------------------------------------
# POST /clear — optional project assertion
# ---------------------------------------------------------------------------


class TestClearProjectAssertion:
    def _clear(self, brain, **kwargs):
        with (
            patch("axon.api_routes.query.clear_active_project") as clear,
            patch.object(api_module, "_save_source_hashes"),
        ):
            resp = client.post("/clear", **kwargs)
        return resp, clear

    def test_mismatched_project_is_409_and_nothing_cleared(self, brain):
        resp, clear = self._clear(brain, json={"project": "beta"})
        assert resp.status_code == 409
        assert "alpha" in resp.json()["detail"]
        clear.assert_not_called()
        brain._assert_write_allowed.assert_not_called()

    def test_matching_project_clears(self, brain):
        resp, clear = self._clear(brain, json={"project": "alpha"})
        assert resp.status_code == 200
        clear.assert_called_once_with(brain)

    def test_bodiless_post_still_clears(self, brain):
        resp, clear = self._clear(brain)
        assert resp.status_code == 200
        clear.assert_called_once_with(brain)

    def test_empty_body_still_clears(self, brain):
        resp, clear = self._clear(brain, json={})
        assert resp.status_code == 200
        clear.assert_called_once_with(brain)


# ---------------------------------------------------------------------------
# Clients that switch-then-write send the project they switched to
# ---------------------------------------------------------------------------


class TestClientsAssertTheirProject:
    def test_remote_clear_sends_project(self):
        from axon import server_client as sc

        with patch.object(sc, "_request", return_value={"status": "success"}) as req:
            sc.remote_clear("http://h", {}, project="research")
        assert req.call_args.args[:2] == ("POST", "http://h/clear")
        assert req.call_args.args[3] == {"project": "research"}

    def test_remote_clear_without_project_sends_empty_body(self):
        from axon import server_client as sc

        with patch.object(sc, "_request", return_value={}) as req:
            sc.remote_clear("http://h", {})
        assert req.call_args.args[3] == {}

    def test_remote_ingest_sends_project(self, tmp_path):
        from axon import server_client as sc

        with patch.object(sc, "_request", return_value={"status": "ok"}) as req:
            sc.remote_ingest("http://h", str(tmp_path), {}, project="research")
        assert req.call_args.args[:2] == ("POST", "http://h/ingest")
        assert req.call_args.args[3]["project"] == "research"

    def test_remote_brain_clear_asserts_its_project(self):
        from axon.remote_brain import RemoteBrain

        rb = RemoteBrain(MagicMock(), {"project": "alpha", "_api_base": "http://h"})
        with patch.object(rb, "_request", return_value={"status": "success"}) as req:
            rb.clear()
        assert req.call_args.args == ("POST", "/clear", {"project": "alpha"})
