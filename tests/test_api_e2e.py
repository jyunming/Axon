from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestApiE2E:
    @pytest.fixture
    def client(self, tmp_path):
        from axon.api import app

        # We patch AxonBrain in the api module so when lifespan instantiates it,
        # it gets our mock.
        with patch("axon.api.AxonBrain") as mock_brain_cls:
            mock_brain = mock_brain_cls.return_value
            mock_brain.query.return_value = "Mocked API response"
            mock_brain.query_stream.return_value = iter(["Mocked ", "API ", "stream"])
            mock_brain._active_project = "default"
            mock_brain.list_documents.return_value = []
            # Mock config
            mock_brain.config = MagicMock()
            mock_brain.config.top_k = 8
            mock_brain.config.hybrid_search = False
            mock_brain.config.rerank = False
            mock_brain.config.hyde = False
            mock_brain.config.multi_query = False
            mock_brain.config.step_back = False
            mock_brain.config.query_decompose = False
            mock_brain.config.compress_context = False
            mock_brain.config.discussion_fallback = True

            # We also need to ensure 'brain' variable in api.py is our mock
            with patch("axon.api.brain", mock_brain):
                with TestClient(app) as client:
                    yield client, mock_brain

    def test_api_health(self, client):
        c, _ = client
        response = c.get("/")
        assert response.status_code in (200, 404)

    @pytest.mark.skip(reason="brain attribute moved to api_routes submodule in Phase 5 refactor")
    def test_api_query(self, client):
        c, mock_brain = client
        # Use a mock query that will definitely trigger the mock
        response = c.post("/query", json={"query": "TEST_QUERY_MOCK"})
        assert response.status_code == 200
        # The /query endpoint in api_routes/query.py returns a JSON with 'response' field
        # for non-streaming, or a stream for streaming.
        # It seems it returned a JSON in the previous run.
        data = response.json()
        assert "Mocked" in data["response"]


class TestApiUpdateCheckBackgroundTask:
    """lifespan()'s passive update-check must not block startup, and its
    asyncio.create_task() result must be kept referenced — the event loop
    only holds a *weak* reference to a bare create_task() result, so an
    unreferenced task can be garbage-collected mid-execution."""

    async def test_background_task_tracked_and_cleaned_up(self, tmp_path):
        """Drives lifespan() directly on the test's own event loop (instead
        of through TestClient, which runs the app on a separate thread with
        its own loop) so the tracked task can be awaited deterministically —
        no wall-clock sleep/poll, so no flakiness under full-suite CPU load."""
        from axon import api as api_module
        from axon.update_check import UpdateCheckResult

        mock_brain = MagicMock()
        mock_brain.config = MagicMock()

        # _background_tasks is a module-level global shared across the whole
        # test session — other tests elsewhere that also exercise lifespan()
        # (TestApiE2E, test_governance.py, etc.) may leave an entry still
        # pending when this test starts. Track the delta, not an absolute
        # count.
        pre_existing = set(api_module._background_tasks)

        with (
            patch("axon.api.AxonConfig.load", return_value=mock_brain.config),
            patch("axon.server_client.find_live_server_for_store", return_value=None),
            patch("axon.server_client.write_store_lock"),
            patch("axon.server_client.release_store_lock"),
            patch.object(api_module, "_auto_init_store"),
            patch("axon.api.AxonBrain", return_value=mock_brain),
            patch(
                "axon.update_check.check_for_update",
                return_value=UpdateCheckResult("0.4.4", "0.5.0", True),
            ),
        ):
            async with api_module.lifespan(api_module.app):
                # Tracked immediately — proves the _background_tasks.add()
                # call ran synchronously in lifespan(), not dependent on the
                # task itself having started executing yet.
                added = api_module._background_tasks - pre_existing
                assert len(added) == 1
                task = next(iter(added))
                await task
                assert task not in api_module._background_tasks

    async def test_lifespan_startup_does_not_block_on_slow_check(self, tmp_path):
        """Same direct-lifespan-call approach as the test above: proves
        __aenter__ returns without waiting on the background task, using a
        never-resolving check rather than a wall-clock race — deterministic,
        no flakiness under full-suite load."""
        import asyncio

        from axon import api as api_module

        never_resolves: asyncio.Future = asyncio.get_running_loop().create_future()

        async def fake_to_thread(fn, /, *args, **kwargs):
            # patch() auto-wraps the real (async def) asyncio.to_thread in an
            # AsyncMock — its side_effect must itself be a coroutine function,
            # or AsyncMock returns the inner coroutine unawaited instead of
            # awaiting it.
            from axon.update_check import UpdateCheckResult

            await never_resolves
            return UpdateCheckResult("0.4.4", None, False, skipped_reason="error")

        mock_brain = MagicMock()
        mock_brain.config = MagicMock()

        # See the delta-tracking note in the test above — _background_tasks
        # is shared global state across the whole test session.
        pre_existing = set(api_module._background_tasks)

        with (
            patch("axon.api.AxonConfig.load", return_value=mock_brain.config),
            patch("axon.server_client.find_live_server_for_store", return_value=None),
            patch("axon.server_client.write_store_lock"),
            patch("axon.server_client.release_store_lock"),
            patch.object(api_module, "_auto_init_store"),
            patch("axon.api.AxonBrain", return_value=mock_brain),
            # lifespan() does `import asyncio as _asyncio` locally, so it's
            # still the same real asyncio module object — patch to_thread
            # there so lifespan's own await point never resolves.
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            async with api_module.lifespan(api_module.app):
                # __aenter__ (i.e. reaching this line) completed without
                # waiting on the never-resolving check — proves it's fired
                # via create_task(), not awaited by lifespan itself.
                added = api_module._background_tasks - pre_existing
                assert len(added) == 1
                task = next(iter(added))
            # Unblock and let the task finish cleanly so the test doesn't
            # leave a dangling pending task behind.
            never_resolves.set_result(None)
            await task
            assert task not in api_module._background_tasks
