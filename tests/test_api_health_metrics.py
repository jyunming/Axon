"""Tests for the split liveness/readiness probes and the /metrics endpoint.

Added in audit batch A1 (audit AUDIT_2026_04_26.md, "Missing features by
effort, Small tier") together with `axon.api_routes.health` and
`axon.api_routes.metrics`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
from axon.api import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_brain():
    """Snapshot/restore the global brain so cross-test ordering is stable."""
    original = api_module.brain
    api_module.brain = None
    yield
    api_module.brain = original


def _make_brain(active_project: str = "default"):
    brain = MagicMock()
    brain._active_project = active_project
    return brain


# ---------------------------------------------------------------------------
# /health/live — always 200
# ---------------------------------------------------------------------------


def test_health_live_returns_200_when_brain_missing():
    """Liveness probe must succeed even when the brain is not initialised."""
    api_module.brain = None
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_health_live_returns_200_when_brain_ready():
    """Liveness probe still 200 once the brain is up."""
    api_module.brain = _make_brain()
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


# ---------------------------------------------------------------------------
# /health/ready — 503 until brain is initialised
# ---------------------------------------------------------------------------


def test_health_ready_returns_503_when_brain_missing():
    """Readiness probe returns 503 while the brain is None."""
    api_module.brain = None
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "initializing"}


def test_health_ready_returns_200_when_brain_ready():
    """Readiness probe returns 200 with active project when brain is set."""
    api_module.brain = _make_brain(active_project="research/atlas")
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["project"] == "research/atlas"


# ---------------------------------------------------------------------------
# /health alias — same shape as /health/ready
# ---------------------------------------------------------------------------


def test_health_alias_503_when_brain_missing():
    api_module.brain = None
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "initializing"


def test_health_alias_200_when_brain_ready():
    api_module.brain = _make_brain(active_project="default")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "project": "default"}


def test_health_alias_v1_prefix_works():
    """The /v1 mirror must still serve the alias."""
    api_module.brain = _make_brain()
    resp = client.get("/v1/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /metrics — Prometheus exposition
# ---------------------------------------------------------------------------


def test_metrics_returns_prometheus_content_type():
    """/metrics must use the text/plain version=0.0.4 exposition format."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    ctype = resp.headers["content-type"]
    assert "text/plain" in ctype
    assert "version=0.0.4" in ctype


def test_metrics_contains_axon_requests_total_line():
    """At least one axon_requests_total sample must appear after a request.

    We hit /health/live first to guarantee the middleware records a sample,
    then scrape /metrics and look for the metric name (HELP/TYPE lines also
    contain it but a counter sample line is the strongest signal).
    """
    # Generate at least one observation so the counter has a sample.
    client.get("/health/live")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "axon_requests_total" in body
    # The HELP line is always present once the metric exists.
    assert "# HELP axon_requests_total" in body


def test_metrics_exposes_request_duration_histogram():
    client.get("/health/live")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "axon_request_duration_seconds" in resp.text


def test_metrics_exposes_axon_brain_ready_gauge():
    """/metrics must expose the axon_brain_ready gauge."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "axon_brain_ready" in resp.text


def test_metrics_brain_ready_gauge_reflects_state():
    """Gauge must read 1 when brain is set, 0 otherwise (refreshed per-scrape)."""
    api_module.brain = None
    resp = client.get("/metrics")
    body = resp.text
    assert "axon_brain_ready 0.0" in body or "axon_brain_ready 0" in body

    api_module.brain = _make_brain()
    resp = client.get("/metrics")
    body = resp.text
    assert "axon_brain_ready 1.0" in body or "axon_brain_ready 1" in body


def test_metrics_exposes_axon_query_total():
    """/metrics must declare the axon_query_total counter."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "axon_query_total" in resp.text


def test_metrics_exposes_axon_ingest_total():
    """/metrics must declare the axon_ingest_total counter."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "axon_ingest_total" in resp.text
