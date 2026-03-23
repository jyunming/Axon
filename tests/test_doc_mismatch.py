from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app)


def _make_brain():
    brain = MagicMock()
    brain._active_project = "default"
    # Mock return values for success paths
    brain.vector_store.get_by_ids.return_value = []
    return brain


def test_add_texts_documented_payload_fails():
    """
    API_REFERENCE.md claims /add_texts uses:
    {"texts": [...], "metadata": [...]}
    But code requires:
    {"docs": [{"text": "...", "metadata": {...}}]}
    """
    api_module.brain = _make_brain()

    # Payload as documented in API_REFERENCE.md L70
    bad_payload = {
        "texts": ["Doc 1", "Doc 2"],
        "metadata": [{"source": "a.txt"}, {"source": "b.txt"}],
    }

    resp = client.post("/add_texts", json=bad_payload)

    # Should fail with 422 Unprocessable Entity because 'docs' is missing
    assert resp.status_code == 422
    assert "docs" in str(resp.json()["detail"])
    print("\n[QA] Confirmed: Documented /add_texts payload causes 422.")


def test_delete_documented_payload_fails():
    """
    QUICKREF.md claims /delete uses:
    {"sources": ["file.txt"]}
    But code requires:
    {"doc_ids": ["uuid-1"]}
    """
    api_module.brain = _make_brain()

    # Payload as documented in QUICKREF.md L443
    bad_payload = {"sources": ["path/to/file.txt"]}

    resp = client.post("/delete", json=bad_payload)

    # Should fail with 422 because 'doc_ids' is missing
    assert resp.status_code == 422
    assert "doc_ids" in str(resp.json()["detail"])
    print("[QA] Confirmed: Documented /delete payload causes 422.")
