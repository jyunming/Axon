"""Synthetic cloud-sync ordering hazard tests for share-mount (#51–#53).

Stands in for the real-OneDrive verification that this branch could not
perform from CI. Simulates the mid-sync race that breaks naive
filesystem-shared SQLite + binary-index setups: tiny files (the version
marker, snapshot manifests) replicate before the larger payload (vector
store, BM25 corpus). A correct grantee read path either reflects
internally consistent state or raises ``MountSyncPendingError`` —
**never** serves a torn read.

These tests are NOT a substitute for end-to-end testing on real cloud
sync; they exercise the mechanism (out-of-order file arrival, partial
file visibility) without needing a Microsoft account, two synced
machines, or the wall-clock waits a real test would require.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from axon.version_marker import (
    MountSyncPendingError,
    artifacts_match,
    bump,
    is_newer_than,
    read,
)

# ---------------------------------------------------------------------------
# A tiny "sync engine" that mirrors files between two tmpdirs in a
# user-controlled order, simulating the way OneDrive / Dropbox / GDrive
# replicate one file at a time and may visibly land a smaller file before
# its sibling has finished.
# ---------------------------------------------------------------------------


def _seed_owner(root: Path, *, tag: str) -> None:
    """Lay down a project the grantee will see via 'sync'."""
    (root / "bm25_index").mkdir(parents=True, exist_ok=True)
    (root / "vector_store_data").mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(f'{{"project_id": "{tag}"}}', encoding="utf-8")
    (root / "bm25_index" / ".bm25_log.jsonl").write_text(f'{{"tag":"{tag}"}}\n', encoding="utf-8")
    (root / "vector_store_data" / "manifest.json").write_text(
        f'{{"tag":"{tag}"}}', encoding="utf-8"
    )


def _sync(owner: Path, grantee: Path, *files: str) -> None:
    """Copy *files* (relative paths) from owner → grantee. Order matters:
    earlier args land first, simulating a sync client picking files in
    its own opaque order."""
    for rel in files:
        src = owner / rel
        dst = grantee / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            # Real sync clients also propagate deletes; mirror that here.
            if dst.exists():
                dst.unlink()
            continue
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Mid-sync race: the marker has the new state but artifacts are stale
# ---------------------------------------------------------------------------


class TestMidSyncRace:
    def test_marker_first_then_artifacts_is_detected(self, tmp_path):
        """Owner bumps and the version.json (tiny) arrives at the grantee
        before the larger artifact files. ``artifacts_match`` must catch
        the mismatch so the grantee can avoid serving a torn read."""
        owner = tmp_path / "owner"
        grantee = tmp_path / "grantee"

        # Initial state: both sides at v1.
        _seed_owner(owner, tag="v1")
        bump(owner)
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
            "version.json",
        )
        cached = read(grantee)
        assert cached["seq"] == 1
        assert artifacts_match(grantee, cached) is True

        # Owner re-ingests; bumps to v2.
        _seed_owner(owner, tag="v2")
        bump(owner)

        # Cloud sync: marker arrives first (it's the smallest file), but
        # the larger artifact files haven't replicated yet.
        _sync(owner, grantee, "version.json")
        current = read(grantee)
        assert is_newer_than(current, cached)  # marker says v2
        # But on disk the grantee still has v1 bytes.
        assert artifacts_match(grantee, current) is False

        # Once the rest of the files arrive, artifacts_match flips True.
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
        )
        assert artifacts_match(grantee, current) is True

    def test_artifacts_first_then_marker_is_safe(self, tmp_path):
        """Reverse order: artifacts arrive first, marker arrives last
        (the order the owner *intended*, since it writes the marker
        last). Until the marker lands, the grantee still sees the old
        marker — which matches the OLD bytes, which haven't been
        overwritten yet on the grantee. Safe."""
        owner = tmp_path / "owner"
        grantee = tmp_path / "grantee"

        _seed_owner(owner, tag="v1")
        bump(owner)
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
            "version.json",
        )
        cached_v1 = read(grantee)

        # Owner re-ingests to v2.
        _seed_owner(owner, tag="v2")
        bump(owner)

        # Sync delivers the artifact files first (the marker hasn't arrived).
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
        )
        # Grantee's cached marker (v1) does NOT match the on-disk bytes
        # anymore — the grantee would see this if it re-hashed. The right
        # behaviour is to re-read the marker, find no advance, and not
        # reopen handles. Once the marker lands, the standard refresh
        # path kicks in.
        latest_marker_on_grantee = read(grantee)
        assert latest_marker_on_grantee["seq"] == cached_v1["seq"]
        # Artifacts on disk are now v2; marker is still v1; this is the
        # interim invisible-to-grantee state. Not strictly an error.
        assert artifacts_match(grantee, cached_v1) is False

        # Marker arrives → grantee can detect the advance.
        _sync(owner, grantee, "version.json")
        current = read(grantee)
        assert is_newer_than(current, cached_v1)
        # And artifacts now match the new marker.
        assert artifacts_match(grantee, current) is True


# ---------------------------------------------------------------------------
# refresh_mount end-to-end: simulated grantee that uses the synthetic sync
# ---------------------------------------------------------------------------


class TestRefreshMountUnderSyncReorder:
    def _stub_brain(self, owner_dir: Path, *, retry_max: int = 3, backoff: float = 0.0):
        from types import MethodType, SimpleNamespace

        from axon.main import AxonBrain

        cfg = SimpleNamespace(
            mount_sync_retry_max=retry_max,
            mount_sync_retry_backoff_s=backoff,
        )
        switch_calls: list[str] = []
        stub = SimpleNamespace(
            config=cfg,
            _active_project="mounts/shared",
            _active_project_kind="mounted",
            _active_mount_descriptor={"target_project_dir": str(owner_dir)},
            _mount_version_marker=None,
            switch_project=lambda name: switch_calls.append(name),
        )
        stub._is_mounted_share = MethodType(AxonBrain._is_mounted_share, stub)
        stub.refresh_mount = MethodType(AxonBrain.refresh_mount, stub)
        return stub, switch_calls

    def test_refresh_when_sync_completes_in_order(self, tmp_path):
        owner = tmp_path / "owner"
        grantee = tmp_path / "grantee"
        _seed_owner(owner, tag="v1")
        bump(owner)
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
            "version.json",
        )

        stub, switch_calls = self._stub_brain(grantee)
        stub._mount_version_marker = read(grantee)

        # Owner re-ingests; sync delivers everything in good order.
        _seed_owner(owner, tag="v2")
        bump(owner)
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
            "version.json",
        )

        assert stub.refresh_mount() is True
        assert switch_calls == ["mounts/shared"]

    def test_refresh_raises_sync_pending_when_marker_outraces_artifacts(self, tmp_path):
        owner = tmp_path / "owner"
        grantee = tmp_path / "grantee"
        _seed_owner(owner, tag="v1")
        bump(owner)
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
            "version.json",
        )

        stub, switch_calls = self._stub_brain(grantee, retry_max=2, backoff=0.0)
        stub._mount_version_marker = read(grantee)

        # Owner re-ingests; sync delivers ONLY the marker. The artifacts
        # never catch up before retry budget exhausts.
        _seed_owner(owner, tag="v2")
        bump(owner)
        _sync(owner, grantee, "version.json")

        with pytest.raises(MountSyncPendingError):
            stub.refresh_mount()
        # And we did NOT call switch_project — handles stay on the old state.
        assert switch_calls == []

    def test_refresh_succeeds_when_artifacts_arrive_during_retry(self, tmp_path):
        """The retry loop has to actually re-hash on each attempt; if it
        doesn't, slow but eventually-consistent sync would always raise
        MountSyncPendingError. We patch in a tiny "advance" between retries
        to prove the loop re-checks instead of caching the first result."""
        owner = tmp_path / "owner"
        grantee = tmp_path / "grantee"
        _seed_owner(owner, tag="v1")
        bump(owner)
        _sync(
            owner,
            grantee,
            "meta.json",
            "bm25_index/.bm25_log.jsonl",
            "vector_store_data/manifest.json",
            "version.json",
        )

        stub, switch_calls = self._stub_brain(grantee, retry_max=4, backoff=0.0)
        stub._mount_version_marker = read(grantee)

        _seed_owner(owner, tag="v2")
        bump(owner)
        # Marker first…
        _sync(owner, grantee, "version.json")

        # Patch time.sleep so the retry "tick" actually runs the
        # delayed-arrival sync between attempts. This mimics
        # eventually-consistent sync: the artifacts land partway through
        # the grantee's polling.
        from unittest.mock import patch

        sync_steps = iter(
            [
                lambda: _sync(owner, grantee, "meta.json"),
                lambda: _sync(owner, grantee, "bm25_index/.bm25_log.jsonl"),
                lambda: _sync(owner, grantee, "vector_store_data/manifest.json"),
            ]
        )

        def _fake_sleep(_):
            try:
                next(sync_steps)()
            except StopIteration:
                pass

        with patch("time.sleep", side_effect=_fake_sleep):
            assert stub.refresh_mount() is True
        assert switch_calls == ["mounts/shared"]


# ---------------------------------------------------------------------------
# DELETE-mode SQLite vs synthetic out-of-order sync
# ---------------------------------------------------------------------------


class TestDeleteJournalSurvivesOutOfOrderSync:
    """The branch's WAL→DELETE switch is grounded in SQLite docs (the
    -wal/-shm sidecar can't be replicated atomically). This test makes
    that concrete: a SQLite DB written in DELETE mode and then
    'sync-replicated' in random order is still openable; a WAL-mode DB
    in the same conditions can lose committed rows when the journal
    sidecar arrives out of sync. We assert only the DELETE-mode side —
    that's what we actually shipped — but the negative case is
    documented in the test docstring for posterity."""

    def test_delete_mode_db_round_trips_via_synthetic_sync(self, tmp_path):
        import sqlite3

        owner = tmp_path / "owner"
        owner.mkdir()
        db_path = owner / "audit.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
            conn.executemany("INSERT INTO t (v) VALUES (?)", [("a",), ("b",), ("c",)])
            conn.commit()

        # No -wal / -shm sidecars exist in DELETE mode — so there's only
        # one file for "sync" to move, and ordering is trivially safe.
        sidecars = list(owner.glob("audit.db-*"))
        assert sidecars == []

        grantee = tmp_path / "grantee"
        grantee.mkdir()
        shutil.copy2(db_path, grantee / "audit.db")

        with sqlite3.connect(str(grantee / "audit.db")) as conn:
            rows = conn.execute("SELECT v FROM t ORDER BY id").fetchall()
        assert [r[0] for r in rows] == ["a", "b", "c"]
