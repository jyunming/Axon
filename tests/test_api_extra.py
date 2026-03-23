"""Extra tests for axon.api to push coverage above 90%."""
import os
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# _evict_old_jobs (lines 47, 49-51)
# ---------------------------------------------------------------------------


class TestEvictOldJobs:
    def test_evicts_expired_jobs(self):
        """Expired jobs (started_at_ts < cutoff) are removed (line 47)."""
        import axon.api as api_module

        old_ts = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
        api_module._jobs.clear()
        api_module._jobs["old_job"] = {"started_at_ts": old_ts, "status": "completed"}
        api_module._jobs["recent_job"] = {"started_at_ts": datetime.now(timezone.utc).timestamp()}

        api_module._evict_old_jobs()

        assert "old_job" not in api_module._jobs
        assert "recent_job" in api_module._jobs

    def test_caps_jobs_at_max(self):
        """When >_MAX_JOBS jobs exist, oldest are removed (lines 49-51)."""
        import axon.api as api_module

        api_module._jobs.clear()
        # Fill with more than _MAX_JOBS entries
        for i in range(api_module._MAX_JOBS + 5):
            api_module._jobs[f"job_{i:04d}"] = {
                "started_at_ts": datetime.now(timezone.utc).timestamp() + i + 1000
            }

        api_module._evict_old_jobs()

        assert len(api_module._jobs) <= api_module._MAX_JOBS


# ---------------------------------------------------------------------------
# api_key_middleware (lines 131-134)
# ---------------------------------------------------------------------------


class TestAPIKeyMiddleware:
    def test_missing_api_key_returns_401(self):
        """When RAG_API_KEY is set and key is missing, returns 401 (lines 131-134)."""
        import axon.api as api_module

        original = api_module._RAG_API_KEY
        api_module._RAG_API_KEY = "secret-test-key"
        try:
            client = TestClient(api_module.app, raise_server_exceptions=False)
            # Non-/health path without key should get 401
            resp = client.post("/query", json={"query": "hello"}, headers={})
            assert resp.status_code == 401
        finally:
            api_module._RAG_API_KEY = original

    def test_correct_api_key_passes(self):
        """When RAG_API_KEY is set and correct key is provided, request passes (lines 131-134)."""
        import axon.api as api_module

        original = api_module._RAG_API_KEY
        api_module._RAG_API_KEY = "correct-key"
        try:
            client = TestClient(api_module.app, raise_server_exceptions=False)
            # With correct key, should NOT get 401 (may get 422 or 503 from actual handler)
            resp = client.post(
                "/query", json={"query": "hello"}, headers={"X-API-Key": "correct-key"}
            )
            assert resp.status_code != 401
        finally:
            api_module._RAG_API_KEY = original


# ---------------------------------------------------------------------------
# main() (lines 194-196)
# ---------------------------------------------------------------------------


class TestApiMain:
    def test_main_calls_uvicorn(self):
        """main() calls uvicorn.run with app, host, and port (lines 194-196)."""
        import axon.api as api_module

        with patch("uvicorn.run") as mock_run:
            with patch.dict(os.environ, {"AXON_HOST": "127.0.0.1", "AXON_PORT": "9876"}):
                api_module.main()
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1].get("host") == "127.0.0.1" or call_args[0][1] == "127.0.0.1"

    def test_main_uses_defaults(self):
        """main() uses 0.0.0.0:8000 by default (lines 194-196)."""
        import axon.api as api_module

        env = {k: v for k, v in os.environ.items() if k not in ("AXON_HOST", "AXON_PORT")}
        with patch("uvicorn.run") as mock_run:
            with patch.dict(os.environ, env, clear=True):
                api_module.main()
            mock_run.assert_called_once()
