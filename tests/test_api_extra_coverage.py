"""Tests for axon.api — increasing coverage."""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_brain():
    brain = MagicMock()
    brain.config.llm_provider = "ollama"
    brain.config.llm_model = "llama3"
    brain._active_project = "default"
    # Mock other attributes used by API
    brain.vector_store = MagicMock()
    brain.bm25 = MagicMock()
    return brain


def test_health_no_brain():
    api_module.brain = None
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "initializing"


def test_health_with_brain(mock_brain):
    api_module.brain = mock_brain
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_config_no_brain():
    api_module.brain = None
    response = client.get("/config")
    assert response.status_code == 503


def test_get_config_success(mock_brain):
    api_module.brain = mock_brain
    mock_brain._apply_overrides.return_value = mock_brain.config
    response = client.get("/config")
    assert response.status_code == 200


def test_update_config_no_brain():
    api_module.brain = None
    response = client.post("/config/update", json={"top_k": 10})
    assert response.status_code == 503


def test_update_config_success(mock_brain):
    api_module.brain = mock_brain
    response = client.post("/config/update", json={"top_k": 10})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_list_sessions_no_brain():
    api_module.brain = None
    response = client.get("/sessions")
    assert response.status_code == 503


def test_list_sessions_success(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.sessions._list_sessions", return_value=[]):
        response = client.get("/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json()


def test_get_session_not_found(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.sessions._load_session", return_value=None):
        response = client.get("/session/missing")
    assert response.status_code == 404


def test_get_session_success(mock_brain):
    api_module.brain = mock_brain
    with patch("axon.sessions._load_session", return_value={"id": "s1"}):
        response = client.get("/session/s1")
    assert response.status_code == 200
    assert response.json()["id"] == "s1"
