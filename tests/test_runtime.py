"""tests/test_runtime.py — Unit tests for axon.runtime.LeaseRegistry (Phase 3)."""

import threading
import time

import pytest

from axon.runtime import LeaseRegistry, _WriteLease


@pytest.fixture()
def reg():
    """Fresh LeaseRegistry for each test — no shared singleton state."""
    return LeaseRegistry()


# ---------------------------------------------------------------------------
# Basic acquire / release
# ---------------------------------------------------------------------------


class TestAcquireRelease:
    def test_acquire_returns_write_lease(self, reg):
        lease = reg.acquire("myproj")
        assert isinstance(lease, _WriteLease)
        lease.close()

    def test_acquire_increments_active_count(self, reg):
        lease = reg.acquire("myproj")
        assert reg.active_lease_count("myproj") == 1
        lease.close()
        assert reg.active_lease_count("myproj") == 0

    def test_multiple_acquire_stacks(self, reg):
        l1 = reg.acquire("myproj")
        l2 = reg.acquire("myproj")
        assert reg.active_lease_count("myproj") == 2
        l1.close()
        assert reg.active_lease_count("myproj") == 1
        l2.close()
        assert reg.active_lease_count("myproj") == 0

    def test_close_is_idempotent(self, reg):
        lease = reg.acquire("myproj")
        lease.close()
        lease.close()  # second close must not go negative
        assert reg.active_lease_count("myproj") == 0

    def test_default_project_always_allowed(self, reg):
        """Lease for 'default' is never tracked — drain is a no-op."""
        lease = reg.acquire("default")
        assert isinstance(lease, _WriteLease)
        assert reg.active_lease_count("default") == 0
        lease.close()

    def test_context_manager_releases_on_exit(self, reg):
        with reg.acquire("myproj"):
            assert reg.active_lease_count("myproj") == 1
        assert reg.active_lease_count("myproj") == 0

    def test_context_manager_releases_on_exception(self, reg):
        with pytest.raises(RuntimeError):
            with reg.acquire("myproj"):
                raise RuntimeError("boom")
        assert reg.active_lease_count("myproj") == 0


# ---------------------------------------------------------------------------
# Drain mode
# ---------------------------------------------------------------------------


class TestDrainMode:
    def test_start_drain_returns_active_count(self, reg):
        l1 = reg.acquire("myproj")
        l2 = reg.acquire("myproj")
        active = reg.start_drain("myproj")
        assert active == 2
        l1.close()
        l2.close()

    def test_acquire_blocked_when_draining(self, reg):
        reg.start_drain("myproj")
        with pytest.raises(PermissionError, match="draining"):
            reg.acquire("myproj")

    def test_stop_drain_re_enables_acquire(self, reg):
        reg.start_drain("myproj")
        reg.stop_drain("myproj")
        lease = reg.acquire("myproj")
        assert reg.active_lease_count("myproj") == 1
        lease.close()

    def test_drain_event_set_immediately_when_no_active_leases(self, reg):
        reg.start_drain("myproj")
        drained = reg.wait_for_drain("myproj", timeout=0.1)
        assert drained is True

    def test_drain_event_set_when_last_lease_released(self, reg):
        lease = reg.acquire("myproj")
        reg.start_drain("myproj")
        # not yet drained
        assert reg.wait_for_drain("myproj", timeout=0.05) is False
        # release the lease
        lease.close()
        assert reg.wait_for_drain("myproj", timeout=1.0) is True

    def test_drain_event_fires_on_background_release(self, reg):
        """Drain completes when a background thread releases its lease."""
        lease = reg.acquire("myproj")
        reg.start_drain("myproj")

        def _delayed_close():
            time.sleep(0.05)
            lease.close()

        t = threading.Thread(target=_delayed_close)
        t.start()
        drained = reg.wait_for_drain("myproj", timeout=2.0)
        t.join()
        assert drained is True

    def test_wait_for_drain_timeout(self, reg):
        lease = reg.acquire("myproj")
        reg.start_drain("myproj")
        result = reg.wait_for_drain("myproj", timeout=0.05)
        assert result is False
        lease.close()  # cleanup

    def test_default_project_drain_is_noop(self, reg):
        """Drain operations on 'default' are safe no-ops."""
        active = reg.start_drain("default")
        assert active == 0
        assert reg.wait_for_drain("default", timeout=0.1) is True
        reg.stop_drain("default")


# ---------------------------------------------------------------------------
# Epoch / stale-writer fencing
# ---------------------------------------------------------------------------


class TestEpochFencing:
    def test_initial_epoch_is_zero(self, reg):
        snap = reg.snapshot("myproj")
        assert snap["epoch"] == 0

    def test_bump_epoch_increments(self, reg):
        e1 = reg.bump_epoch("myproj")
        e2 = reg.bump_epoch("myproj")
        assert e1 == 1
        assert e2 == 2

    def test_stale_write_logs_warning_on_release(self, reg, caplog):
        import logging

        lease = reg.acquire("myproj")
        reg.bump_epoch("myproj")  # epoch advances after acquire
        with caplog.at_level(logging.WARNING, logger="axon.runtime"):
            lease.close()
        assert any("Stale write" in r.message for r in caplog.records)

    def test_fresh_write_no_warning(self, reg, caplog):
        import logging

        lease = reg.acquire("myproj")
        # No epoch bump — release should be silent
        with caplog.at_level(logging.WARNING, logger="axon.runtime"):
            lease.close()
        assert not any("Stale write" in r.message for r in caplog.records)

    def test_default_epoch_bump_returns_zero(self, reg):
        assert reg.bump_epoch("default") == 0

    def test_reset_removes_state(self, reg):
        reg.acquire("myproj").close()
        reg.bump_epoch("myproj")
        reg.reset("myproj")
        snap = reg.snapshot("myproj")
        # After reset, snapshot creates fresh state with epoch 0
        assert snap["epoch"] == 0
        assert snap["active_leases"] == 0


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_fields(self, reg):
        snap = reg.snapshot("myproj")
        assert set(snap.keys()) == {"project", "epoch", "active_leases", "draining"}

    def test_snapshot_reflects_state(self, reg):
        lease = reg.acquire("myproj")
        reg.start_drain("myproj")
        snap = reg.snapshot("myproj")
        assert snap["active_leases"] == 1
        assert snap["draining"] is True
        lease.close()

    def test_default_snapshot(self, reg):
        snap = reg.snapshot("default")
        assert snap == {"project": "default", "epoch": 0, "active_leases": 0, "draining": False}
