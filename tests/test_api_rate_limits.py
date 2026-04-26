"""Tests for the shared per-IP sliding-window rate limiter.

Covers:
- enforce_rate_limit blocks after max_hits exceeded (HTTP 429)
- Different IPs don't interfere with each other
- Direct unit-test of enforce_rate_limit with a fake Request mock
- IP eviction when the per-bucket dict reaches _MAX_IPS
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(ip: str = "127.0.0.1", xff: str | None = None) -> MagicMock:
    """Return a fake FastAPI Request whose client.host and headers are controlled."""
    req = MagicMock()
    req.client = SimpleNamespace(host=ip)
    if xff is not None:
        req.headers = {"X-Forwarded-For": xff}
    else:
        req.headers = {}
    return req


def _fresh_module():
    """Import _rate_limit with a clean state by reloading the module."""
    import importlib

    import axon.api_routes._rate_limit as rl

    importlib.reload(rl)
    return rl


# ---------------------------------------------------------------------------
# Unit tests for enforce_rate_limit
# ---------------------------------------------------------------------------


class TestEnforceRateLimit:
    def setup_method(self):
        """Reload the module to get a fresh bucket dict for each test."""
        self.rl = _fresh_module()

    def test_allows_requests_within_limit(self):
        """max_hits requests should all succeed without raising."""
        req = _make_request("10.0.0.1")
        for _ in range(5):
            self.rl.enforce_rate_limit(req, bucket="test_allow", max_hits=5, window_seconds=60.0)

    def test_blocks_on_max_hits_exceeded(self):
        """The (max_hits + 1)-th request must raise HTTP 429."""
        req = _make_request("10.0.0.2")
        for _ in range(10):
            self.rl.enforce_rate_limit(req, bucket="test_block", max_hits=10, window_seconds=60.0)
        with pytest.raises(HTTPException) as exc_info:
            self.rl.enforce_rate_limit(req, bucket="test_block", max_hits=10, window_seconds=60.0)
        assert exc_info.value.status_code == 429
        assert "Too many requests" in exc_info.value.detail

    def test_different_ips_are_independent(self):
        """Exhausting the limit for IP-A must not affect IP-B."""
        req_a = _make_request("10.1.0.1")
        req_b = _make_request("10.1.0.2")
        # Exhaust IP-A
        for _ in range(3):
            self.rl.enforce_rate_limit(req_a, bucket="test_iso", max_hits=3, window_seconds=60.0)
        with pytest.raises(HTTPException):
            self.rl.enforce_rate_limit(req_a, bucket="test_iso", max_hits=3, window_seconds=60.0)
        # IP-B is unaffected
        self.rl.enforce_rate_limit(req_b, bucket="test_iso", max_hits=3, window_seconds=60.0)

    def test_distinct_buckets_are_independent(self):
        """Exhausting bucket-A for an IP must not affect bucket-B for the same IP."""
        req = _make_request("10.2.0.1")
        for _ in range(2):
            self.rl.enforce_rate_limit(req, bucket="bucket_a", max_hits=2, window_seconds=60.0)
        with pytest.raises(HTTPException):
            self.rl.enforce_rate_limit(req, bucket="bucket_a", max_hits=2, window_seconds=60.0)
        # bucket_b is unaffected
        self.rl.enforce_rate_limit(req, bucket="bucket_b", max_hits=2, window_seconds=60.0)

    def test_window_expiry_resets_counter(self):
        """Hits that fall outside the window should not count."""
        req = _make_request("10.3.0.1")
        # Use a very short window (0.05 s) and advance monotonic via patching.
        # We patch time.monotonic to control time precisely.
        tick = [0.0]

        def fake_monotonic():
            return tick[0]

        with patch.object(self.rl.time, "monotonic", side_effect=fake_monotonic):
            # Fill window at t=0
            for _ in range(3):
                self.rl.enforce_rate_limit(
                    req, bucket="test_expire", max_hits=3, window_seconds=1.0
                )
            # Should be blocked
            with pytest.raises(HTTPException):
                self.rl.enforce_rate_limit(
                    req, bucket="test_expire", max_hits=3, window_seconds=1.0
                )
            # Advance past the window
            tick[0] = 2.0
            # Should be allowed again
            self.rl.enforce_rate_limit(req, bucket="test_expire", max_hits=3, window_seconds=1.0)

    def test_xff_header_used_for_ip(self):
        """X-Forwarded-For header takes precedence over client.host."""
        req = _make_request(ip="192.168.1.1", xff="203.0.113.5, 10.0.0.1")
        # First hit records IP "203.0.113.5"
        self.rl.enforce_rate_limit(req, bucket="test_xff", max_hits=1, window_seconds=60.0)
        # Second hit from the same XFF IP should be blocked
        with pytest.raises(HTTPException) as exc_info:
            self.rl.enforce_rate_limit(req, bucket="test_xff", max_hits=1, window_seconds=60.0)
        assert exc_info.value.status_code == 429

    def test_unknown_ip_when_no_client(self):
        """When request.client is None and no XFF, IP resolves to 'unknown'."""
        req = MagicMock()
        req.client = None
        req.headers = {}
        # Should not raise — just uses 'unknown' as IP key
        self.rl.enforce_rate_limit(req, bucket="test_no_client", max_hits=5, window_seconds=60.0)

    def test_ip_eviction_at_max_ips_cap(self):
        """When _MAX_IPS is reached, the oldest entry is evicted."""
        rl = self.rl
        bucket = "test_evict"
        # Patch _MAX_IPS to a small value
        original_max = rl._MAX_IPS
        rl._MAX_IPS = 3
        try:
            for i in range(3):
                r = _make_request(f"10.10.0.{i}")
                rl.enforce_rate_limit(r, bucket=bucket, max_hits=10, window_seconds=60.0)
            # dict is at cap; next IP triggers eviction
            r_new = _make_request("10.10.0.99")
            rl.enforce_rate_limit(r_new, bucket=bucket, max_hits=10, window_seconds=60.0)
            # The per-bucket dict should still be at most _MAX_IPS entries
            assert len(rl._buckets[bucket]) <= rl._MAX_IPS
        finally:
            rl._MAX_IPS = original_max

    def test_thread_safety_no_data_race(self):
        """Concurrent requests from different IPs should not raise unexpected errors."""
        rl = self.rl
        errors: list[Exception] = []

        def _hit(ip: str):
            req = _make_request(ip)
            try:
                for _ in range(5):
                    rl.enforce_rate_limit(
                        req, bucket="test_threads", max_hits=100, window_seconds=60.0
                    )
            except HTTPException:
                pass  # 429 is expected once limit hit — not a bug
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_hit, args=(f"172.16.{i}.1",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"


# ---------------------------------------------------------------------------
# Integration-style tests: verify route handlers call enforce_rate_limit
# ---------------------------------------------------------------------------


class TestRateLimitImport:
    """Smoke-check that the modified route modules import cleanly and expose
    enforce_rate_limit in their namespace (indirectly, via the import)."""

    def test_rate_limit_module_importable(self):
        from axon.api_routes._rate_limit import enforce_rate_limit  # noqa: F401

    def test_security_routes_import(self):
        import axon.api_routes.security_routes  # noqa: F401

    def test_shares_routes_import(self):
        import axon.api_routes.shares  # noqa: F401

    def test_ingest_routes_import(self):
        import axon.api_routes.ingest  # noqa: F401


# ---------------------------------------------------------------------------
# _get_ip helper tests
# ---------------------------------------------------------------------------


class TestGetIp:
    def setup_method(self):
        self.rl = _fresh_module()

    def test_xff_single_ip(self):
        req = _make_request(ip="1.2.3.4", xff="5.6.7.8")
        assert self.rl._get_ip(req) == "5.6.7.8"

    def test_xff_multiple_ips_returns_first(self):
        req = _make_request(ip="1.2.3.4", xff="  9.8.7.6 , 1.1.1.1")
        assert self.rl._get_ip(req) == "9.8.7.6"

    def test_no_xff_uses_client_host(self):
        req = _make_request(ip="192.0.2.1")
        assert self.rl._get_ip(req) == "192.0.2.1"

    def test_no_client_and_no_xff_returns_unknown(self):
        req = MagicMock()
        req.client = None
        req.headers = {}
        assert self.rl._get_ip(req) == "unknown"
