"""Sync-chaos fixtures for sealed-share tests (Phase 7).

These fixtures simulate the behaviours OneDrive / Dropbox / SMB sync
exhibit during file propagation **without** requiring a real cloud
account. The two-layer test strategy (research summary in PR thread):

- **Layer 1 (this directory):** fault-injection on top of the local
  filesystem. Cheap (≤ 15 s per suite), runs every PR, catches
  partial-byte visibility, conflict-copy artifacts, locked files,
  and ``.tmp.drivedownload``-style debris.
- **Layer 2 (deferred):** multi-process race harness on top of
  WsgiDAV. Tracked as future work; covers true two-writer races and
  the eventual-consistency settle behaviour Layer 1 can't model.

The fixtures are deliberately small & explicit — no new dependency
(no pyfakefs / no fault-injection FS) so the harness runs anywhere
the existing pytest config does.

What CANNOT be simulated here (documented gaps, see
``docs/SHARE_MOUNT_SEALED_SMOKE.md`` for the manual recipe):

- Real Microsoft Graph throttling / 429 patterns.
- Windows Explorer preview-pane file lock (kernel-level).
- OneDrive Files-On-Demand placeholder semantics (filter driver).
- Multi-region propagation lag — Layer 1 picks one number; reality
  is a distribution.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest


class SyncChaos:
    """Inject sync-engine misbehaviour into a single tmp directory.

    Usage::

        chaos.appear_partial(file_path, real_size=4096)
        chaos.lock(file_path)
        chaos.drop_conflict_copy(file_path, suffix="-OneDrive-MachineB.conflict")
        chaos.drop_drivedownload_artifact(parent_dir)

    Each method is self-contained — call to inject, call ``restore``
    to undo. The fixture itself wraps everything in ``__enter__`` /
    ``__exit__`` so test bodies don't have to track state.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._patches: list = []
        self._created: list[Path] = []

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Partial sync — stat returns smaller size than the real file
    # ------------------------------------------------------------------

    def appear_partial(self, path: Path, *, fake_size: int = 0) -> None:
        """Make ``path.stat().st_size`` return *fake_size* (default 0).

        Models OneDrive's "file appears with metadata before the
        payload finishes uploading" window. Uses an unconditional
        patch — no auto-restore; pair with the fixture's ``restore``
        or rely on context-manager teardown.

        Path comparison uses ``os.fspath`` rather than ``Path.resolve``
        to avoid infinite recursion (resolve() calls stat() internally).
        """
        import os as _os

        target_str = _os.fspath(Path(path))
        original_stat = Path.stat

        def patched_stat(self_path, *args, **kwargs):
            if _os.fspath(self_path) == target_str:
                real = original_stat(self_path, *args, **kwargs)
                return _StatProxy(real, fake_size)
            return original_stat(self_path, *args, **kwargs)

        p = patch.object(Path, "stat", patched_stat)
        p.start()
        self._patches.append(p)

    # ------------------------------------------------------------------
    # Locked file — open() raises PermissionError
    # ------------------------------------------------------------------

    def lock(self, path: Path) -> None:
        """Make every ``open(path, ...)`` raise PermissionError.

        Models a sync engine holding an exclusive handle while it
        finishes writing the local cache. Patches both ``Path.open``
        and ``Path.read_bytes`` since the sealed-store code uses both.
        """
        import os as _os

        target_str = _os.fspath(Path(path))
        original_open = Path.open
        original_read_bytes = Path.read_bytes

        def patched_open(self_path, *args, **kwargs):
            if _os.fspath(self_path) == target_str:
                raise PermissionError(f"Simulated sync-engine lock on {self_path}")
            return original_open(self_path, *args, **kwargs)

        def patched_read_bytes(self_path, *args, **kwargs):
            if _os.fspath(self_path) == target_str:
                raise PermissionError(f"Simulated sync-engine lock on {self_path}")
            return original_read_bytes(self_path, *args, **kwargs)

        for attr, fn in (("open", patched_open), ("read_bytes", patched_read_bytes)):
            p = patch.object(Path, attr, fn)
            p.start()
            self._patches.append(p)

    # ------------------------------------------------------------------
    # Conflict copy — sync engine appends a suffix to a duplicate
    # ------------------------------------------------------------------

    def drop_conflict_copy(
        self,
        path: Path,
        *,
        suffix: str = "-OneDrive-MachineB.conflict",
        contents: bytes = b"",
    ) -> Path:
        """Create a sibling conflict-copy file alongside *path*.

        Models the artifact OneDrive / Dropbox creates when two
        machines write the same file simultaneously. The actual file
        at *path* is left intact; only the sibling is added.
        """
        path = Path(path)
        conflict = path.with_name(path.stem + suffix + path.suffix)
        conflict.write_bytes(contents)
        self._created.append(conflict)
        return conflict

    # ------------------------------------------------------------------
    # .tmp.drivedownload-style debris
    # ------------------------------------------------------------------

    def drop_drivedownload_artifact(
        self, parent_dir: Path, *, name: str = "ssk_phantom.wrapped.tmp.drivedownload"
    ) -> Path:
        """Leave a sync-engine temp file in *parent_dir*.

        Models Google Drive's ``.tmp.drivedownload`` and OneDrive's
        ``~$``-prefixed temp files. Tests verify that
        ``list_sealed_share_key_ids`` (and similar walkers) ignore
        these — they should only count ``.wrapped`` files.
        """
        parent_dir = Path(parent_dir)
        parent_dir.mkdir(parents=True, exist_ok=True)
        artifact = parent_dir / name
        artifact.write_bytes(b"sync-engine-temp-noise")
        self._created.append(artifact)
        return artifact

    # ------------------------------------------------------------------
    # Cleanup — restore patches and remove sentinel files
    # ------------------------------------------------------------------

    def restore(self) -> None:
        for p in reversed(self._patches):
            try:
                p.stop()
            except RuntimeError:
                pass  # already stopped
        self._patches.clear()
        for f in self._created:
            try:
                if f.is_file():
                    f.unlink()
            except OSError:
                pass
        self._created.clear()


class _StatProxy:
    """Proxy for ``os.stat_result`` that overrides ``st_size`` only."""

    def __init__(self, real_stat, fake_size: int) -> None:
        self._real = real_stat
        self._fake_size = fake_size

    @property
    def st_size(self) -> int:
        return self._fake_size

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def sync_chaos(tmp_path) -> Iterator[SyncChaos]:
    """Per-test ``SyncChaos`` rooted at ``tmp_path``."""
    chaos = SyncChaos(tmp_path)
    try:
        yield chaos
    finally:
        chaos.restore()


@contextmanager
def chaos_scope(root: Path) -> Iterator[SyncChaos]:
    """Standalone context-manager form (when not using the fixture)."""
    chaos = SyncChaos(root)
    try:
        yield chaos
    finally:
        chaos.restore()
