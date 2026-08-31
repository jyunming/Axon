"""Tests for axon._pid_check.pid_alive — extracted from
axon.security.cache._pid_alive so server_client.py's store lock can reuse
the same cross-platform liveness check.
"""
from __future__ import annotations

import os

from axon._pid_check import pid_alive


def test_current_process_is_alive():
    assert pid_alive(os.getpid()) is True


def test_zero_or_negative_pid_is_not_alive():
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_very_unlikely_pid_is_not_alive():
    # A PID far above any realistic live process on any platform tested.
    assert pid_alive(2**30) is False
