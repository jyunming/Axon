"""Shared per-IP sliding-window rate limiter for Axon API routes.

Each logical endpoint uses a named *bucket* so independent rate-limit
counters don't bleed across routes.  The implementation intentionally
avoids external dependencies — it is a pure-Python in-process store
suitable for single-process deployments.

Usage::

    from axon.api_routes._rate_limit import enforce_rate_limit

    @router.post("/my/endpoint")
    async def my_endpoint(request: Request, ...):
        enforce_rate_limit(request, bucket="my_endpoint")
        ...
"""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request

# Top-level dict: bucket_name → {ip → ([hit_timestamps], per-ip Lock)}
_buckets: dict[str, dict[str, tuple[list[float], threading.Lock]]] = {}
_global_lock = threading.Lock()

# Hard cap: evict oldest IP when the per-bucket dict grows beyond this size
# to prevent unbounded memory growth under a steady stream of unique IPs.
_MAX_IPS = 10_000


def _get_ip(request: Request) -> str:
    """Return the best-effort client IP, honouring X-Forwarded-For."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(
    request: Request,
    *,
    bucket: str,
    max_hits: int = 10,
    window_seconds: float = 60.0,
) -> None:
    """Raise HTTP 429 if the caller exceeds *max_hits* within *window_seconds*.

    Parameters
    ----------
    request:
        The incoming FastAPI ``Request`` object (used to extract the IP).
    bucket:
        Logical group name — distinct endpoints should use distinct bucket
        names so their counters are independent.
    max_hits:
        Maximum number of requests allowed per IP within *window_seconds*.
    window_seconds:
        Length of the sliding time window in seconds.
    """
    ip = _get_ip(request)
    now = time.monotonic()

    with _global_lock:
        if bucket not in _buckets:
            _buckets[bucket] = {}
        b = _buckets[bucket]

        # Evict oldest IP when cap is reached so the dict stays bounded.
        if len(b) >= _MAX_IPS:
            oldest = min(
                b,
                key=lambda k: b[k][0][0] if b[k][0] else 0,
            )
            del b[oldest]

        if ip not in b:
            b[ip] = ([], threading.Lock())

        hits, lock = b[ip]

    # Per-IP lock: only one goroutine-like-path updates this IP's timestamps.
    with lock:
        # Prune timestamps that have fallen outside the sliding window.
        cutoff = now - window_seconds
        while hits and hits[0] < cutoff:
            hits.pop(0)

        if len(hits) >= max_hits:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Try again later.",
            )
        hits.append(now)
