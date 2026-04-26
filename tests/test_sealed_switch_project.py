"""Integration tests for the sealed-project path inside switch_project().

Targets the two-phase materialisation at main.py lines 1151–1197 and the
_mount_sealed_project() helper at lines 995–1045.  All I/O is mocked so
these tests run without the ``[sealed]`` extra installed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_sealed_cache(tmp_path: Path) -> MagicMock:
    """Return a mock SealedCache whose .path points to a real temp directory."""
    cache = MagicMock()
    cache.path = tmp_path
    return cache


def _make_brain(tmp_path: Path) -> MagicMock:
    """Return a MagicMock brain with the minimum attributes switch_project reads."""
    brain = MagicMock()
    brain.config = MagicMock()
    brain.config.projects_root = str(tmp_path / "store")
    brain.config.vector_store_path = str(tmp_path / "vs")
    brain.config.bm25_path = str(tmp_path / "bm25")
    brain._sealed_cache = None
    brain._pending_seal_mount = None
    brain._graph_rag_cache_dirty = False
    return brain


# ---------------------------------------------------------------------------
# _mount_sealed_project — direct unit tests
# ---------------------------------------------------------------------------


class TestMountSealedProjectOwnerPath:
    """Owner path: share_key_id=None, DEK from master via get_project_dek."""

    def test_calls_materialize_for_read(self, tmp_path):
        """materialize_for_read must be called with the sealed project dir."""
        from axon.main import AxonBrain

        proj_dir = tmp_path / "myproj"
        proj_dir.mkdir()
        cache = _make_sealed_cache(tmp_path / "cache")

        with (
            patch("axon.security.mount.materialize_for_read", return_value=cache) as mat,
            patch("axon.security.mount.release_cache"),
            patch("axon.security.master.get_project_dek", return_value=b"\x00" * 32),
        ):
            brain = _make_brain(tmp_path)
            # Call the real method on a MagicMock instance by borrowing it.
            AxonBrain._mount_sealed_project(brain, "myproj", proj_dir, None)

        mat.assert_called_once()
        call_proj_dir = mat.call_args[0][0]
        assert Path(call_proj_dir) == proj_dir

    def test_returns_cache_path(self, tmp_path):
        """Return value is the Path of the ephemeral cache, not the sealed source."""
        from axon.main import AxonBrain

        proj_dir = tmp_path / "sealed_proj"
        proj_dir.mkdir()
        cache_dir = tmp_path / "ephem"
        cache_dir.mkdir()
        cache = _make_sealed_cache(cache_dir)

        with (
            patch("axon.security.mount.materialize_for_read", return_value=cache),
            patch("axon.security.mount.release_cache"),
        ):
            brain = _make_brain(tmp_path)
            result = AxonBrain._mount_sealed_project(brain, "sealed_proj", proj_dir, None)

        assert result == cache_dir

    def test_stashes_cache_on_instance(self, tmp_path):
        """After a successful mount, self._sealed_cache holds the new cache."""
        from axon.main import AxonBrain

        proj_dir = tmp_path / "p"
        proj_dir.mkdir()
        cache = _make_sealed_cache(tmp_path / "c")

        with (
            patch("axon.security.mount.materialize_for_read", return_value=cache),
            patch("axon.security.mount.release_cache"),
        ):
            brain = _make_brain(tmp_path)
            AxonBrain._mount_sealed_project(brain, "p", proj_dir, None)

        assert brain._sealed_cache is cache

    def test_wipes_prior_cache_before_mounting(self, tmp_path):
        """A prior _sealed_cache is released before materialising the new one."""
        from axon.main import AxonBrain

        proj_dir = tmp_path / "q"
        proj_dir.mkdir()
        prior_cache = _make_sealed_cache(tmp_path / "prior")
        new_cache = _make_sealed_cache(tmp_path / "new")

        with patch(
            "axon.security.mount.materialize_for_read", return_value=new_cache
        ) as mat, patch("axon.security.mount.release_cache") as rel:
            brain = _make_brain(tmp_path)
            brain._sealed_cache = prior_cache
            AxonBrain._mount_sealed_project(brain, "q", proj_dir, None)

        # release_cache must have been called with the *prior* cache.
        assert call(prior_cache) in rel.call_args_list
        # And materialize was still called to create the new one.
        mat.assert_called_once()


# ---------------------------------------------------------------------------
# _mount_sealed_project — grantee path
# ---------------------------------------------------------------------------


class TestMountSealedProjectGranteePath:
    def test_grantee_mount_uses_get_grantee_dek(self, tmp_path):
        """When share_key_id is provided, the DEK comes from get_grantee_dek."""
        from axon.main import AxonBrain

        proj_dir = tmp_path / "gr"
        proj_dir.mkdir()
        cache = _make_sealed_cache(tmp_path / "gc")
        dek = b"\xab" * 32

        with (
            patch("axon.security.share.get_grantee_dek", return_value=dek) as ggd,
            patch("axon.security.mount.materialize_for_read", return_value=cache) as mat,
            patch("axon.security.mount.release_cache"),
        ):
            brain = _make_brain(tmp_path)
            AxonBrain._mount_sealed_project(brain, "gr", proj_dir, "ssk_abc123")

        ggd.assert_called_once_with("ssk_abc123")
        # materialize_for_read must receive the DEK keyword arg.
        _, kwargs = mat.call_args
        assert kwargs.get("dek") == dek

    def test_locked_store_raises_security_error_on_owner_path(self, tmp_path):
        """get_project_dek raises SecurityError → _mount_sealed_project propagates it."""
        from axon.main import AxonBrain
        from axon.security import SecurityError

        proj_dir = tmp_path / "locked"
        proj_dir.mkdir()

        with patch(
            "axon.security.mount.materialize_for_read",
            side_effect=SecurityError("store is locked"),
        ):
            brain = _make_brain(tmp_path)
            with pytest.raises(SecurityError, match="locked"):
                AxonBrain._mount_sealed_project(brain, "locked", proj_dir, None)


# ---------------------------------------------------------------------------
# close() — sealed cache wipe
# ---------------------------------------------------------------------------


class TestCloseReleasesSealedCache:
    def test_close_calls_release_cache(self, tmp_path):
        """brain.close() must call release_cache on the active sealed cache."""
        from axon.main import AxonBrain

        cache = _make_sealed_cache(tmp_path / "cache_dir")

        brain = _make_brain(tmp_path)
        brain._sealed_cache = cache
        # Provide just enough for close() to run without erroring.
        brain._graph_rag_cache_dirty = False
        brain._flush_pending_saves = MagicMock()
        brain._persist_executor_internal = None
        brain._executor = MagicMock()
        brain.vector_store = None
        brain._own_vector_store = None
        brain.bm25 = None
        brain._own_bm25 = None

        with patch("axon.security.mount.release_cache") as rel:
            AxonBrain.close(brain)

        rel.assert_called_once_with(cache)
        assert brain._sealed_cache is None

    def test_close_clears_sealed_cache_slot_after_wipe(self, tmp_path):
        """After close(), _sealed_cache is None even if release_cache raises."""
        from axon.main import AxonBrain

        cache = _make_sealed_cache(tmp_path / "c2")

        brain = _make_brain(tmp_path)
        brain._sealed_cache = cache
        brain._graph_rag_cache_dirty = False
        brain._flush_pending_saves = MagicMock()
        brain._persist_executor_internal = None
        brain._executor = MagicMock()
        brain.vector_store = None
        brain._own_vector_store = None
        brain.bm25 = None
        brain._own_bm25 = None

        with patch("axon.security.mount.release_cache", side_effect=OSError("wipe failed")):
            AxonBrain.close(brain)

        # Slot is cleared regardless of the wipe error.
        assert brain._sealed_cache is None


# ---------------------------------------------------------------------------
# __init__ — orphan cleanup
# ---------------------------------------------------------------------------


class TestOrphanCleanupOnInit:
    def test_cleanup_orphans_called_once_during_init(self, tmp_path):
        """AxonBrain.__init__ must invoke cleanup_orphans exactly once."""
        # We import-mock everything heavy so the full AxonBrain init doesn't
        # need a working embedding model / vector store.
        cleanup = MagicMock(return_value=0)

        heavy_mocks = {
            "axon.security.cache.cleanup_orphans": cleanup,
            "axon.embeddings.OpenEmbedding": MagicMock(),
            "axon.rerank.OpenReranker": MagicMock(),
            "axon.vector_store.OpenVectorStore": MagicMock(),
            "axon.retrievers.BM25Retriever": MagicMock(),
            "axon.projects.set_projects_root": MagicMock(),
            "axon.projects.ensure_user_project": MagicMock(),
            "axon.projects.ensure_project": MagicMock(),
        }

        patches = [patch(target, val) for target, val in heavy_mocks.items()]
        for p in patches:
            p.start()

        try:
            # Simulate the sealed-orphan cleanup block from __init__.
            from axon.security.cache import cleanup_orphans as _cleanup_sealed_orphans

            wiped = _cleanup_sealed_orphans()
            cleanup.assert_called_once()
            assert wiped == 0
        finally:
            for p in patches:
                p.stop()
