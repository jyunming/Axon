"""Tests for axon.logging_setup — structured logging + request_id contextvar."""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_logging_setup_module() -> None:
    """Reset the module-level _configured flag so each test gets a clean slate."""
    import axon.logging_setup as _ls

    _ls._configured = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def setup_method(self):
        _reset_logging_setup_module()
        # Also clear root handlers so we start clean.
        root = logging.getLogger()
        root.handlers.clear()

    def teardown_method(self):
        _reset_logging_setup_module()
        root = logging.getLogger()
        root.handlers.clear()

    def test_configure_adds_one_handler(self):
        from axon.logging_setup import configure_logging

        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_configure_is_idempotent(self):
        """Calling configure_logging() twice must not duplicate handlers."""
        from axon.logging_setup import configure_logging

        configure_logging()
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_configure_sets_level(self):
        from axon.logging_setup import configure_logging

        configure_logging(level=logging.WARNING)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_handler_has_request_id_filter(self):
        from axon.logging_setup import RequestIdFilter, configure_logging

        configure_logging()
        root = logging.getLogger()
        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert RequestIdFilter in filter_types


class TestRequestIdFilter:
    def test_filter_adds_default_request_id(self):
        """RequestIdFilter should add request_id='-' when the contextvar is unset."""
        from axon.logging_setup import RequestIdFilter

        f = RequestIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        result = f.filter(record)
        assert result is True
        assert record.request_id == "-"  # type: ignore[attr-defined]

    def test_filter_adds_set_request_id(self):
        """RequestIdFilter should use the contextvar value when set."""
        from axon.logging_setup import RequestIdFilter, reset_request_id, set_request_id

        token = set_request_id("test-rid-123")
        try:
            f = RequestIdFilter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="hello",
                args=(),
                exc_info=None,
            )
            f.filter(record)
            assert record.request_id == "test-rid-123"  # type: ignore[attr-defined]
        finally:
            reset_request_id(token)


class TestSetResetRequestId:
    def test_set_and_get_request_id(self):
        from axon.logging_setup import get_request_id, reset_request_id, set_request_id

        assert get_request_id() == "-"
        token = set_request_id("abc-123")
        assert get_request_id() == "abc-123"
        reset_request_id(token)
        assert get_request_id() == "-"

    def test_nested_set_reset(self):
        """Nested set/reset should restore intermediate values correctly."""
        from axon.logging_setup import get_request_id, reset_request_id, set_request_id

        token1 = set_request_id("outer")
        token2 = set_request_id("inner")
        assert get_request_id() == "inner"
        reset_request_id(token2)
        assert get_request_id() == "outer"
        reset_request_id(token1)
        assert get_request_id() == "-"

    def test_set_request_id_returns_token(self):
        """set_request_id must return a contextvars.Token."""
        from axon.logging_setup import reset_request_id, set_request_id

        token = set_request_id("tok-test")
        assert isinstance(token, contextvars.Token)
        reset_request_id(token)


class TestThreadContextPropagation:
    """Test that contextvar propagates to thread pools via copy_context()."""

    def test_context_propagates_via_copy_context(self):
        """Contextvar values should be visible in threads when using copy_context().run()."""
        from axon.logging_setup import get_request_id, reset_request_id, set_request_id

        token = set_request_id("thread-test-rid")
        try:
            ctx = contextvars.copy_context()
            captured: list[str] = []

            def _worker():
                captured.append(get_request_id())

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(ctx.run, _worker)
                future.result(timeout=5)

            assert captured == ["thread-test-rid"]
        finally:
            reset_request_id(token)

    def test_context_not_propagated_without_copy_context(self):
        """Without copy_context, new threads see the default value, not the caller's."""
        from axon.logging_setup import get_request_id, reset_request_id, set_request_id

        token = set_request_id("should-not-propagate")
        try:
            captured: list[str] = []

            def _worker():
                captured.append(get_request_id())

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_worker)
                future.result(timeout=5)

            # Without copy_context, threads get the default value ("-")
            assert captured == ["-"]
        finally:
            reset_request_id(token)

    def test_multiple_concurrent_contexts(self):
        """Different contexts in different threads should be isolated."""
        from axon.logging_setup import get_request_id, reset_request_id, set_request_id

        results: list[str] = []
        errors: list[Exception] = []

        def _task(rid: str, delay: float) -> str:
            import time

            token = set_request_id(rid)
            time.sleep(delay)
            val = get_request_id()
            reset_request_id(token)
            return val

        ctxs = []
        for _i in range(3):
            ctx = contextvars.copy_context()
            ctxs.append(ctx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(ctxs[i].run, _task, f"rid-{i}", 0.01) for i in range(3)]
            for f in concurrent.futures.as_completed(futs):
                try:
                    results.append(f.result(timeout=5))
                except Exception as e:
                    errors.append(e)

        assert not errors
        assert sorted(results) == ["rid-0", "rid-1", "rid-2"]
