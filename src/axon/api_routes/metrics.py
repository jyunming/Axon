"""Prometheus exposition endpoint for the Axon API.

Added per audit AUDIT_2026_04_26.md, "Missing features by effort, Small tier"
(`/metrics endpoint with prometheus-client`). Operators can scrape this with
a standard Prometheus job; the metric set intentionally starts minimal:

* ``axon_query_total`` — Counter labelled by project / surface. Lets
  operators alert on query volume per project and calling surface (api, mcp,
  repl, etc.).
* ``axon_ingest_total`` — Counter labelled by project / surface. Tracks
  ingest throughput per project and surface.
* ``axon_brain_ready`` — Gauge that is 1 when ``axon.api.brain`` is
  initialised and 0 otherwise. Pairs naturally with the ``/health/ready``
  probe.
* ``axon_requests_total`` — Counter labelled by path / method / status. Lets
  operators alert on 5xx spikes per route without enabling per-route logging.
* ``axon_request_duration_seconds`` — Histogram of wall-clock request latency.
  Enables p50/p95 dashboards out of the box.

The middleware that updates the per-request counters lives in ``axon.api`` so
it can sit alongside the existing request-id middleware; this module owns the
metric definitions and the exposition endpoint.

If ``prometheus-client`` is missing (unusual — it ships in base deps as of
PR #68 (audit batch A1)), we fall back to no-op shims so the API still
boots. Operators just see ``/metrics`` return 503 with a helpful message.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

logger = logging.getLogger("AxonAPI")

router = APIRouter()


# ---------------------------------------------------------------------------
# Metric definitions — guarded so a missing prometheus_client install does
# not crash import. Base deps include prometheus-client>=0.20 so the
# fallback path should never run in practice.
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover — base deps include prometheus-client
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    generate_latest = None  # type: ignore[assignment]


# Use a dedicated registry so test isolation works (the default global
# registry persists across pytest runs and trips Counter("…already registered")
# errors when tests reload axon.api).
REGISTRY: object | None = None
REQUEST_COUNTER: object | None = None
REQUEST_DURATION: object | None = None
QUERY_COUNTER: object | None = None
INGEST_COUNTER: object | None = None
BRAIN_READY: object | None = None


if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry()
    # Per-request observability (middleware-driven).
    REQUEST_COUNTER = Counter(
        "axon_requests_total",
        "Total HTTP requests handled by the Axon API.",
        labelnames=("path", "method", "status"),
        registry=REGISTRY,
    )
    REQUEST_DURATION = Histogram(
        "axon_request_duration_seconds",
        "Wall-clock duration of HTTP requests handled by the Axon API.",
        labelnames=("path", "method"),
        registry=REGISTRY,
    )
    # Domain-level counters (incremented from query/ingest handlers).
    QUERY_COUNTER = Counter(
        "axon_query_total",
        "Total RAG queries handled by the Axon API.",
        labelnames=("project", "surface"),
        registry=REGISTRY,
    )
    INGEST_COUNTER = Counter(
        "axon_ingest_total",
        "Total ingest requests handled by the Axon API.",
        labelnames=("project", "surface"),
        registry=REGISTRY,
    )
    # Brain readiness gauge (set in lifespan + refreshed per scrape).
    BRAIN_READY = Gauge(
        "axon_brain_ready",
        "1 when axon.api.brain is initialised, 0 otherwise.",
        registry=REGISTRY,
    )


# Pin the content-type to the legacy 0.0.4 exposition format so existing
# scrapers (and the tests in test_api_health_metrics.py) see a stable value
# regardless of which prometheus-client release is installed. The text
# format is forward-compatible.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def record_request(path: str, method: str, status: int, duration_seconds: float) -> None:
    """Update the request counter + latency histogram for one served request.

    Called from the middleware in ``axon.api``. Safe to call when
    prometheus_client is not installed — becomes a no-op.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        REQUEST_COUNTER.labels(path=path, method=method, status=str(status)).inc()  # type: ignore[union-attr]
        REQUEST_DURATION.labels(path=path, method=method).observe(duration_seconds)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Failed to record metrics for %s %s: %s", method, path, exc)


def record_query(project: str = "_global", surface: str = "api") -> None:
    """Increment the ``axon_query_total`` counter.

    Called from ``axon.api_routes.query`` at the start of each query handler
    so the counter reflects accepted requests rather than successful ones.
    Safe to call when prometheus_client is not installed — becomes a no-op.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        QUERY_COUNTER.labels(project=project, surface=surface).inc()  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Failed to record query metric: %s", exc)


def record_ingest(project: str = "_global", surface: str = "api") -> None:
    """Increment the ``axon_ingest_total`` counter.

    Called from ``axon.api_routes.ingest`` at the start of each ingest handler
    so the counter reflects accepted requests rather than successful ones.
    Safe to call when prometheus_client is not installed — becomes a no-op.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        INGEST_COUNTER.labels(project=project, surface=surface).inc()  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Failed to record ingest metric: %s", exc)


def update_brain_ready(is_ready: bool) -> None:
    """Set the axon_brain_ready gauge to 1 (ready) or 0 (not ready)."""
    if not PROMETHEUS_AVAILABLE:
        return
    try:
        BRAIN_READY.set(1 if is_ready else 0)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Failed to update axon_brain_ready gauge: %s", exc)


@router.get("/metrics")
async def metrics_endpoint():
    """Expose Prometheus metrics in the 0.0.4 text exposition format.

    Returns 503 with a plain message if prometheus-client is not installed
    so scrapers see a clear failure mode rather than an empty 200.
    """
    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse(
            "prometheus-client not installed; install axon-rag base deps to enable /metrics.",
            status_code=503,
        )
    # Refresh the brain gauge on every scrape so it reflects the current
    # lifespan state without needing a separate background task.
    from axon import api as _api

    update_brain_ready(_api.brain is not None)
    assert REGISTRY is not None  # guaranteed by PROMETHEUS_AVAILABLE guard above
    payload = generate_latest(REGISTRY)  # type: ignore[arg-type]
    return Response(content=payload, media_type=CONTENT_TYPE)
