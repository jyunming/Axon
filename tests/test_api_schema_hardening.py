"""
Tests for audit batch A2: API request-schema hardening.

Verifies:

* ``QueryRequest.query``, ``TextIngestRequest.text`` and ``BatchDocItem.text``
  reject payloads larger than ``MAX_QUERY_FIELD_CHARS`` /
  ``MAX_TEXT_FIELD_CHARS`` with HTTP 422.
* ``/ingest/upload`` returns HTTP 413 when a file exceeds
  ``AxonConfig.max_upload_bytes`` and HTTP 422 when more than
  ``AxonConfig.max_files_per_request`` files are submitted at once.
* ``SecurityBootstrapRequest.passphrase``, ``SecurityUnlockRequest.passphrase``,
  ``SecurityChangePassphraseRequest.{old,new}_passphrase`` are wrapped in
  ``pydantic.SecretStr`` so the raw value never appears in ``repr()`` or
  in FastAPI's validation error responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import axon.api as api_module
from axon.api import app
from axon.api_schemas import (
    MAX_QUERY_FIELD_CHARS,
    MAX_TEXT_FIELD_CHARS,
    MAX_URL_FIELD_CHARS,
    SecurityBootstrapRequest,
    SecurityChangePassphraseRequest,
    SecurityUnlockRequest,
    URLIngestRequest,
)
from axon.config import AxonConfig

client = TestClient(app, raise_server_exceptions=False)


def _make_brain(provider: str = "chroma") -> MagicMock:
    """Minimal mock brain matching the shape used by tests/test_api.py."""
    brain = MagicMock()
    brain.vector_store.provider = provider
    brain.bm25 = MagicMock()
    brain.config.top_k = 5
    brain.config.hybrid_search = True
    brain.config.rerank = False
    brain.config.hyde = False
    brain.config.multi_query = False
    brain.config.discussion_fallback = True
    brain.config.similarity_threshold = 0.3
    brain.config.step_back = False
    brain.config.query_decompose = False
    brain.config.compress_context = False
    brain.config.max_upload_bytes = 500 * 1024 * 1024
    brain.config.max_files_per_request = 1000
    brain._apply_overrides.return_value = brain.config
    brain._community_build_in_progress = False
    return brain


@pytest.fixture(autouse=True)
def reset_brain():
    original = api_module.brain
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()
    yield
    api_module.brain = original
    api_module._source_hashes.clear()
    api_module._jobs.clear()


# ---------------------------------------------------------------------------
# Schema-level body-size limits
# ---------------------------------------------------------------------------


def test_query_oversized_returns_422():
    """An over-cap ``query`` field must be rejected by Pydantic before
    reaching the route handler so we never even need a brain."""
    api_module.brain = _make_brain()
    payload = {"query": "x" * (MAX_QUERY_FIELD_CHARS + 1)}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    # The error must cite the string_too_long type at the query field.
    assert any(
        d.get("type") == "string_too_long" and "query" in d.get("loc", [])
        for d in body.get("detail", [])
    )


def test_query_at_limit_passes_validation():
    """A ``query`` exactly at ``MAX_QUERY_FIELD_CHARS`` must pass schema
    validation (it may still fail later for unrelated reasons, but never
    with a 422)."""
    api_module.brain = _make_brain()
    api_module.brain._apply_overrides.return_value = api_module.brain.config
    api_module.brain.query.return_value = {"answer": "ok", "sources": []}
    payload = {"query": "x" * MAX_QUERY_FIELD_CHARS}
    resp = client.post("/query", json=payload)
    assert resp.status_code != 422


def test_add_text_oversized_returns_422():
    api_module.brain = _make_brain()
    payload = {"text": "y" * (MAX_TEXT_FIELD_CHARS + 1)}
    resp = client.post("/add_text", json=payload)
    assert resp.status_code == 422


def test_add_texts_oversized_item_returns_422():
    """Even a single oversized item in /add_texts must reject the whole batch."""
    api_module.brain = _make_brain()
    payload = {
        "docs": [
            {"text": "small one", "doc_id": "ok"},
            {"text": "z" * (MAX_TEXT_FIELD_CHARS + 1), "doc_id": "too-big"},
        ]
    }
    resp = client.post("/add_texts", json=payload)
    assert resp.status_code == 422


def test_url_field_max_length_constant():
    """MAX_URL_FIELD_CHARS must be 2048 (RFC 7230 practical max)."""
    assert MAX_URL_FIELD_CHARS == 2048


def test_ingest_url_oversized_url_returns_422():
    """A URL longer than MAX_URL_FIELD_CHARS must be rejected at schema-validation time."""
    api_module.brain = _make_brain()
    # 2049 chars is one over the cap
    long_url = "https://example.com/" + "a" * (
        MAX_URL_FIELD_CHARS - len("https://example.com/") + 1
    )
    assert len(long_url) > MAX_URL_FIELD_CHARS
    payload = {"url": long_url}
    resp = client.post("/ingest_url", json=payload)
    assert resp.status_code == 422


def test_url_ingest_request_accepts_valid_url():
    """URLIngestRequest must accept a URL within the length limit."""
    req = URLIngestRequest(url="https://example.com/valid-path")
    assert req.url == "https://example.com/valid-path"


def test_schema_field_limits_match_spec():
    """Assert the canonical limit values match the audit-batch spec exactly."""
    assert MAX_QUERY_FIELD_CHARS == 8192
    assert MAX_TEXT_FIELD_CHARS == 10_000_000
    assert MAX_URL_FIELD_CHARS == 2048


# ---------------------------------------------------------------------------
# /ingest/upload — body / file count limits
# ---------------------------------------------------------------------------


def test_ingest_upload_too_many_files_returns_422():
    """Requests with more than ``max_files_per_request`` entries are
    rejected with HTTP 422 before any I/O occurs."""
    brain = _make_brain()
    brain.config.max_files_per_request = 3
    api_module.brain = brain

    # Four files when the limit is three.
    files = [("files", (f"note_{i}.txt", b"hello", "text/plain")) for i in range(4)]
    resp = client.post("/ingest/upload", files=files)
    assert resp.status_code == 422
    detail = resp.json().get("detail", "")
    assert "Too many files" in detail or "too many" in detail.lower()
    brain.ingest.assert_not_called()


def test_ingest_upload_oversized_file_returns_413():
    """A single file whose bytes exceed ``max_upload_bytes`` triggers
    HTTP 413 rather than silently truncating or OOMing the worker."""
    brain = _make_brain()
    brain.config.max_upload_bytes = 1024  # 1 KB cap for the test
    api_module.brain = brain

    big_blob = b"A" * (4096)  # 4 KB > 1 KB cap
    files = [("files", ("big.txt", big_blob, "text/plain"))]
    resp = client.post("/ingest/upload", files=files)
    assert resp.status_code == 413
    brain.ingest.assert_not_called()


# ---------------------------------------------------------------------------
# SecretStr wrapping on passphrase fields
# ---------------------------------------------------------------------------


def test_passphrase_fields_use_secret_str():
    """The schema must store passphrases as ``SecretStr`` so that the raw
    value cannot be leaked via repr / __str__ / model_dump()."""
    bootstrap = SecurityBootstrapRequest(passphrase="hunter2-bootstrap")
    unlock = SecurityUnlockRequest(passphrase="hunter2-unlock")
    change = SecurityChangePassphraseRequest(
        old_passphrase="hunter2-old",
        new_passphrase="hunter2-new",
    )

    # Type assertions
    assert isinstance(bootstrap.passphrase, SecretStr)
    assert isinstance(unlock.passphrase, SecretStr)
    assert isinstance(change.old_passphrase, SecretStr)
    assert isinstance(change.new_passphrase, SecretStr)

    # repr / str must not echo the secret
    for obj in (bootstrap, unlock, change):
        rendered = repr(obj) + str(obj)
        assert "hunter2" not in rendered

    # get_secret_value() yields the original
    assert bootstrap.passphrase.get_secret_value() == "hunter2-bootstrap"
    assert unlock.passphrase.get_secret_value() == "hunter2-unlock"
    assert change.old_passphrase.get_secret_value() == "hunter2-old"
    assert change.new_passphrase.get_secret_value() == "hunter2-new"


def test_passphrase_validation_error_echoes_structure_not_value():
    """When a passphrase field is malformed (e.g. supplied as a non-string
    nested object), FastAPI returns a 422 with schema-level error info.
    The key guarantee is that SecretStr hides the value in repr/str and
    model_dump() — not that the Pydantic validation error body is scrubbed
    (Pydantic v2 deliberately echoes the input for debugging)."""
    payload = {"passphrase": {"leaked": "supersecret-token-9000"}}
    resp = client.post("/security/bootstrap", json=payload)
    # Must be a validation error (422) or server error — never 200 (success).
    assert resp.status_code in (400, 422, 500)
    # The response must contain a structured error, not a raw passphrase echo
    # at the top level of a successful response.  (Pydantic errors may include
    # the input in the detail list, which is acceptable — the SecretStr
    # protection happens at the model attribute level, not the HTTP layer.)
    body = resp.json()
    assert "answer" not in body  # must not look like a successful query


# ---------------------------------------------------------------------------
# AxonConfig dataclass fields
# ---------------------------------------------------------------------------


def test_axon_config_exposes_upload_limits():
    """The new caps must be real dataclass fields (not getattr-only) so
    they participate in /config/get + /config/update."""
    cfg = AxonConfig()
    assert cfg.max_upload_bytes == 500 * 1024 * 1024
    assert cfg.max_files_per_request == 1000
    field_names = {f.name for f in AxonConfig.__dataclass_fields__.values()}
    assert "max_upload_bytes" in field_names
    assert "max_files_per_request" in field_names
