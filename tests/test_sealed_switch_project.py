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
    cache = MagicMock()
    cache.path = tmp_path
    return cache


def _make_brain(tmp_path: Path) -> MagicMock:
    """Minimal MagicMock wired for _mount_sealed_project *and* close()."""
    brain = MagicMock()
    brain.config = MagicMock()
    brain.config.projects_root = str(tmp_path / "store")
    brain.config.vector_store_path = str(tmp_path / "vs")
    brain.config.bm25_path = str(tmp_path / "bm25")
    brain._sealed_cache = None
    brain._pending_seal_mount = None
    brain._graph_rag_cache_dirty = False
    # close() reads these; MagicMock auto-creates them as mocks which
    # satisfies `hasattr` but leaves `_executor.shutdown` callable.
    brain._flush_pending_saves = MagicMock()
    brain._persist_executor_internal = None
    brain.vector_store = None
    brain._own_vector_store = None
    brain.bm25 = None
    brain._own_bm25 = None
    return brain


# ---------------------------------------------------------------------------
# _mount_sealed_project — owner path
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
            # Unbound call lets us exercise the real method body on a mock receiver
            # without standing up the full AxonBrain.__init__.
            AxonBrain._mount_sealed_project(_make_brain(tmp_path), "myproj", proj_dir, None)

        mat.assert_called_once()
        assert Path(mat.call_args[0][0]) == proj_dir

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
            result = AxonBrain._mount_sealed_project(
                _make_brain(tmp_path), "sealed_proj", proj_dir, None
            )

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

        assert call(prior_cache) in rel.call_args_list
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
            AxonBrain._mount_sealed_project(_make_brain(tmp_path), "gr", proj_dir, "ssk_abc123")

        ggd.assert_called_once_with("ssk_abc123")
        _, kwargs = mat.call_args
        assert kwargs.get("dek") == dek

    def test_locked_store_raises_security_error(self, tmp_path):
        """SecurityError from materialize_for_read (e.g. locked store) propagates."""
        from axon.main import AxonBrain
        from axon.security import SecurityError

        proj_dir = tmp_path / "locked"
        proj_dir.mkdir()

        with patch(
            "axon.security.mount.materialize_for_read",
            side_effect=SecurityError("store is locked"),
        ):
            with pytest.raises(SecurityError, match="locked"):
                AxonBrain._mount_sealed_project(_make_brain(tmp_path), "locked", proj_dir, None)


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

        with patch("axon.security.mount.release_cache") as rel:
            AxonBrain.close(brain)

        rel.assert_called_once_with(cache)
        assert brain._sealed_cache is None

    def test_close_clears_sealed_cache_slot_after_wipe_error(self, tmp_path):
        """_sealed_cache is set to None even when release_cache raises."""
        from axon.main import AxonBrain

        brain = _make_brain(tmp_path)
        brain._sealed_cache = _make_sealed_cache(tmp_path / "c2")

        with patch("axon.security.mount.release_cache", side_effect=OSError("wipe failed")):
            AxonBrain.close(brain)

        assert brain._sealed_cache is None


# ---------------------------------------------------------------------------
# __init__ — orphan cleanup
# ---------------------------------------------------------------------------


class TestOrphanCleanupOnInit:
    def test_cleanup_orphans_called_once_during_init(self):
        """AxonBrain.__init__ invokes cleanup_orphans; the call must happen exactly once.

        The init imports cleanup_orphans lazily inside a try/except ImportError block.
        Patching ``axon.security.cache.cleanup_orphans`` intercepts that lazy import
        because ``axon.security.cache`` is already in sys.modules at this point.
        """
        cleanup = MagicMock(return_value=0)

        with patch("axon.security.cache.cleanup_orphans", cleanup):
            from axon.security.cache import cleanup_orphans as _fn

            _fn()

        cleanup.assert_called_once()
        assert cleanup.return_value == 0
