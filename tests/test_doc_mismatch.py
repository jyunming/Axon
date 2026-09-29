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
    The pre-0.5.0 API reference doc (now docs/REFERENCE.md) claimed /add_texts used:
    {"texts": [...], "metadata": [...]}
    But code requires:
    {"docs": [{"text": "...", "metadata": {...}}]}
    """
    api_module.brain = _make_brain()

    # Payload as the old API reference doc documented it
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
    The pre-0.5.0 quick-reference doc (now docs/REFERENCE.md) claimed /delete used:
    {"sources": ["file.txt"]}
    But code requires:
    {"doc_ids": ["uuid-1"]}
    """
    api_module.brain = _make_brain()

    # Payload as the old quick-reference doc documented it
    bad_payload = {"sources": ["path/to/file.txt"]}

    resp = client.post("/delete", json=bad_payload)

    # Should fail with 422 because 'doc_ids' is missing
    assert resp.status_code == 422
    assert "doc_ids" in str(resp.json()["detail"])
    print("[QA] Confirmed: Documented /delete payload causes 422.")
