"""Tests for axon.api_routes.config_routes."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)

from dataclasses import dataclass


@dataclass
class FakeConfig:
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    hybrid_search: bool = True
    api_key: str = "secret"

    def save(self):
        return None


@pytest.fixture(autouse=True)
def _reset_brain():
    """Restore api_module.brain after every test to prevent state leakage."""
    original = api_module.brain
    yield
    api_module.brain = original


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.config = FakeConfig()
    brain._active_project = "default"
    return brain


def test_get_config(mock_brain):
    api_module.brain = mock_brain
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert data["llm_model"] == "llama3"
    # Verify masking
    assert "api_key" in data
    assert data["api_key"] == "***"


def test_update_config(mock_brain):
    api_module.brain = mock_brain
    # The endpoint is /config/update in projects.py
    response = client.post("/config/update", json={"llm_model": "new-model"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_set_config_field_updates_dot_notation_key(mock_brain):
    api_module.brain = mock_brain
    response = client.post(
        "/config/set",
        json={"key": "rag.hybrid_search", "value": False, "persist": False},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["flat_key"] == "hybrid_search"
    assert mock_brain.config.hybrid_search is False


def test_set_config_field_reinitializes_llm(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.llm.OpenLLM") as mock_open_llm:
        response = client.post(
            "/config/set",
            json={"key": "llm.model", "value": "new-model", "persist": False},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_open_llm.assert_called_once_with(mock_brain.config)
    assert mock_brain.llm == mock_open_llm.return_value


def test_get_config_no_brain():
    api_module.brain = None
    response = client.get("/config")
    assert response.status_code == 503
