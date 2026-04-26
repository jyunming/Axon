"""
tests/test_mount_refresh_modes.py

Focused tests for mount_refresh_mode=per_query / switch TTL logic wired into
the query path (QueryRouterMixin._maybe_refresh_mount) and the companion
mount_refresh_ttl_s AxonConfig field.

Complements the broader tests in tests/test_version_marker.py.
"""
from __future__ import annotations

import time
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub(
    mode: str,
    *,
    ttl: int = 300,
    refresh_return: bool = False,
    refresh_side_effect=None,
):
    """Return a lightweight SimpleNamespace that binds the real
    ``QueryRouterMixin._maybe_refresh_mount`` so we can exercise the
    method without standing up a full ``AxonBrain``.
    """
    from axon.query_router import QueryRouterMixin

    cfg = SimpleNamespace(mount_refresh_mode=mode, mount_refresh_ttl_s=ttl)
    refresh = MagicMock(return_value=refresh_return, side_effect=refresh_side_effect)
    stub = SimpleNamespace(config=cfg, refresh_mount=refresh)
    stub._maybe_refresh_mount = MethodType(QueryRouterMixin._maybe_refresh_mount, stub)
    return stub, refresh


# ---------------------------------------------------------------------------
# mount_refresh_ttl_s must be a declared dataclass field on AxonConfig
# ---------------------------------------------------------------------------


class TestMountRefreshTtlConfigField:
    def test_field_declared_on_axon_config(self):
        """``mount_refresh_ttl_s`` must be a proper dataclass field, not a
        silent ``getattr`` default — the CLAUDE.md rule forbids the latter."""
        import dataclasses

        from axon.config import AxonConfig

        field_names = {f.name for f in dataclasses.fields(AxonConfig)}
        assert "mount_refresh_ttl_s" in field_names, (
            "mount_refresh_ttl_s must be declared as a dataclass field on AxonConfig; "
            "using getattr(..., default) silently bypasses YAML loading and validation."
        )

    def test_mount_refresh_mode_declared_on_axon_config(self):
        """``mount_refresh_mode`` must also be a proper dataclass field."""
        import dataclasses

        from axon.config import AxonConfig

        field_names = {f.name for f in dataclasses.fields(AxonConfig)}
        assert "mount_refresh_mode" in field_names

    def test_default_ttl_value_is_300(self):
        from axon.config import AxonConfig

        cfg = AxonConfig()
        assert cfg.mount_refresh_ttl_s == 300

    def test_default_mode_is_switch(self):
        from axon.config import AxonConfig

        cfg = AxonConfig()
        assert cfg.mount_refresh_mode == "switch"

    def test_yaml_override_ttl_is_respected(self, tmp_path):
        """A user-supplied YAML value must propagate into the config object."""
        from axon.config import AxonConfig

        config_file = tmp_path / "config.yaml"
        config_file.write_text("security:\n  mount_refresh_ttl_s: 60\n", encoding="utf-8")
        cfg = AxonConfig.load(str(config_file))
        assert cfg.mount_refresh_ttl_s == 60

    def test_yaml_override_mode_is_respected(self, tmp_path):
        """YAML security.mount_refresh_mode must be loaded into the config."""
        from axon.config import AxonConfig

        config_file = tmp_path / "config.yaml"
        config_file.write_text("security:\n  mount_refresh_mode: per_query\n", encoding="utf-8")
        cfg = AxonConfig.load(str(config_file))
        assert cfg.mount_refresh_mode == "per_query"


# ---------------------------------------------------------------------------
# per_query mode
# ---------------------------------------------------------------------------


class TestPerQueryMode:
    def test_calls_refresh_on_every_invocation(self):
        """In per_query mode, _maybe_refresh_mount must delegate to
        refresh_mount on every single call."""
        stub, refresh = _make_stub("per_query")
        stub._maybe_refresh_mount()
        stub._maybe_refresh_mount()
        stub._maybe_refresh_mount()
        assert refresh.call_count == 3

    def test_refresh_called_even_when_marker_unchanged(self):
        """refresh_mount returning False (no change) is fine -- we still call
        it each query so we never serve stale results silently."""
        stub, refresh = _make_stub("per_query", refresh_return=False)
        stub._maybe_refresh_mount()
        assert refresh.call_count == 1

    def test_sync_pending_error_propagates(self):
        """MountSyncPendingError must not be swallowed -- callers need to
        surface it as a transient signal."""
        from axon.version_marker import MountSyncPendingError

        stub, _ = _make_stub(
            "per_query", refresh_side_effect=MountSyncPendingError("indices in flight")
        )
        with pytest.raises(MountSyncPendingError, match="indices in flight"):
            stub._maybe_refresh_mount()

    def test_non_fatal_exceptions_are_swallowed(self):
        """Any other exception from refresh_mount must be caught so a
        refresh-machinery bug never prevents retrieval from completing."""
        stub, _ = _make_stub("per_query", refresh_side_effect=OSError("disk busy"))
        # Must NOT raise.
        stub._maybe_refresh_mount()

    def test_no_timestamp_tracking_in_per_query(self):
        """per_query mode does not use TTL tracking -- no _last_mount_check_ts
        should be set by _maybe_refresh_mount in this mode."""
        stub, _ = _make_stub("per_query")
        stub._maybe_refresh_mount()
        # The attribute should NOT be set by per_query path.
        assert not hasattr(stub, "_last_mount_check_ts")


# ---------------------------------------------------------------------------
# switch mode -- TTL-based auto-refresh
# ---------------------------------------------------------------------------


class TestSwitchModeTtl:
    def test_ttl_zero_never_auto_refreshes(self):
        """When mount_refresh_ttl_s=0, auto-refresh is disabled entirely
        in switch mode regardless of how long since the last check."""
        stub, refresh = _make_stub("switch", ttl=0)
        stub._maybe_refresh_mount()
        stub._maybe_refresh_mount()
        assert refresh.call_count == 0

    def test_within_ttl_does_not_refresh(self):
        """If the TTL has not yet elapsed, switch mode should skip the
        disk I/O check entirely."""
        stub, refresh = _make_stub("switch", ttl=300)
        # Seed _last_mount_check_ts to "just now" so TTL has not elapsed.
        stub._last_mount_check_ts = time.monotonic()
        stub._maybe_refresh_mount()
        assert refresh.call_count == 0

    def test_after_ttl_expires_calls_refresh(self):
        """When the TTL has elapsed, switch mode must call refresh_mount."""
        stub, refresh = _make_stub("switch", ttl=1)
        # Set _last_mount_check_ts to a point far enough in the past that
        # the TTL (1 s) has definitely elapsed.
        stub._last_mount_check_ts = time.monotonic() - 10.0
        stub._maybe_refresh_mount()
        assert refresh.call_count == 1

    def test_no_prior_check_ts_triggers_refresh(self):
        """First call with no _last_mount_check_ts (getattr default=0.0)
        should always fire when TTL > 0."""
        stub, refresh = _make_stub("switch", ttl=1)
        # Do NOT set _last_mount_check_ts -- default getattr fallback is 0.0.
        stub._maybe_refresh_mount()
        assert refresh.call_count == 1

    def test_timestamp_updated_after_switch_mode_check(self):
        """After a TTL-triggered check, _last_mount_check_ts must be
        refreshed so the next call starts a new TTL window."""
        stub, _ = _make_stub("switch", ttl=1)
        stub._last_mount_check_ts = time.monotonic() - 10.0
        before = stub._last_mount_check_ts
        stub._maybe_refresh_mount()
        assert hasattr(stub, "_last_mount_check_ts")
        assert stub._last_mount_check_ts > before

    def test_timestamp_updated_even_on_non_fatal_exception(self):
        """A non-fatal refresh_mount error must still reset the timestamp
        so we do not busy-poll on a permanently-broken path."""
        stub, _ = _make_stub("switch", ttl=1, refresh_side_effect=OSError("path gone"))
        stub._last_mount_check_ts = time.monotonic() - 10.0
        before = stub._last_mount_check_ts
        stub._maybe_refresh_mount()  # must not raise
        assert stub._last_mount_check_ts > before

    def test_second_call_within_ttl_skips_refresh(self):
        """After a TTL check resets the timestamp, the very next call
        must be within the new TTL window and must NOT call refresh_mount."""
        stub, refresh = _make_stub("switch", ttl=300)
        # Force the first call to trigger (expired TTL).
        stub._last_mount_check_ts = time.monotonic() - 400.0
        stub._maybe_refresh_mount()
        first_call_count = refresh.call_count  # should be 1

        # Immediately call again -- should be within the new 300 s TTL.
        stub._maybe_refresh_mount()
        assert refresh.call_count == first_call_count  # no extra call

    def test_switch_mode_does_not_refresh_multiple_within_ttl(self):
        stub, refresh = _make_stub("switch", ttl=3600)
        stub._last_mount_check_ts = time.monotonic()
        for _ in range(5):
            stub._maybe_refresh_mount()
        assert refresh.call_count == 0


# ---------------------------------------------------------------------------
# off mode
# ---------------------------------------------------------------------------


class TestOffMode:
    def test_never_calls_refresh(self):
        stub, refresh = _make_stub("off")
        stub._maybe_refresh_mount()
        stub._maybe_refresh_mount()
        assert refresh.call_count == 0

    def test_off_mode_ignores_ttl_setting(self):
        stub, refresh = _make_stub("off", ttl=0)
        stub._maybe_refresh_mount()
        assert refresh.call_count == 0

    def test_off_mode_no_timestamp_set(self):
        stub, _ = _make_stub("off")
        stub._maybe_refresh_mount()
        assert not hasattr(stub, "_last_mount_check_ts")


# ---------------------------------------------------------------------------
# Integration: _execute_retrieval wires _maybe_refresh_mount correctly
# ---------------------------------------------------------------------------


class TestExecuteRetrievalWiresRefresh:
    """Verify that _execute_retrieval actually calls _maybe_refresh_mount
    before any retrieval work, using a minimal stub of AxonBrain."""

    def test_per_query_mode_triggers_before_retrieval(self):
        """_execute_retrieval must call _maybe_refresh_mount exactly once
        per invocation when mode is per_query."""
        from axon.config import AxonConfig
        from axon.query_router import QueryRouterMixin

        cfg = AxonConfig()
        cfg.mount_refresh_mode = "per_query"
        cfg.mount_refresh_ttl_s = 300
        cfg.hyde = False
        cfg.multi_query = False
        cfg.step_back = False
        cfg.query_decompose = False

        stub = MagicMock(spec=QueryRouterMixin)
        stub.config = cfg
        stub._check_mount_revocation = MagicMock()
        stub._maybe_refresh_mount = MagicMock()

        stub._execute_retrieval = MethodType(QueryRouterMixin._execute_retrieval, stub)

        with pytest.raises(Exception):  # noqa: B017
            # Will fail deep in the pipeline -- that's fine.
            stub._execute_retrieval("test query")

        stub._check_mount_revocation.assert_called_once()
        stub._maybe_refresh_mount.assert_called_once()

    def test_source_confirms_wiring(self):
        """Source-level check: _execute_retrieval source must reference
        both _check_mount_revocation and _maybe_refresh_mount."""
        import inspect

        from axon.query_router import QueryRouterMixin

        source = inspect.getsource(QueryRouterMixin._execute_retrieval)
        assert (
            "_maybe_refresh_mount" in source
        ), "_execute_retrieval must call self._maybe_refresh_mount() before retrieval"
        assert (
            "_check_mount_revocation" in source
        ), "_execute_retrieval must call self._check_mount_revocation() before retrieval"
