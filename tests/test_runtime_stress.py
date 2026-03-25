from unittest.mock import patch

import pytest

from axon.main import AxonBrain, AxonConfig
from axon.runtime import get_registry


def _make_brain(tmp_path):
    config = AxonConfig(
        vector_store_path=str(tmp_path / "vs"),
        bm25_path=str(tmp_path / "bm25"),
        projects_root=str(tmp_path / "projects"),
    )
    # Mock expensive components
    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenReranker"
    ), patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"):
        brain = AxonBrain(config)
    return brain


@pytest.mark.stress
def test_concurrent_switch_and_ingest_fencing(tmp_path):
    """
    Simulate a 'stale writer' race condition.
    Ingest starts for Project A, acquires lease with Epoch 0.
    Switch happens to Project B, Epoch for A bumps to 1.
    Ingest for A finishes and tries to release/save.
    """
    brain = _make_brain(tmp_path)
    reg = get_registry()

    # Setup projects
    (tmp_path / "projects" / "proj_a").mkdir(parents=True)
    (tmp_path / "projects" / "proj_a" / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "projects" / "proj_b").mkdir(parents=True)
    (tmp_path / "projects" / "proj_b" / "meta.json").write_text("{}", encoding="utf-8")

    # 1. Start on proj_a
    with patch("axon.projects.project_dir", return_value=tmp_path / "projects" / "proj_a"), patch(
        "axon.projects.project_vector_path", return_value="v"
    ), patch("axon.projects.project_bm25_path", return_value="b"), patch(
        "axon.projects.set_active_project"
    ):
        brain.switch_project("proj_a")

    assert brain._active_project == "proj_a"
    initial_epoch = reg._state("proj_a").epoch

    # 2. Mock a slow ingest that acquires lease then hangs
    # We will manually simulate what happens inside brain.ingest
    # to control the timing precisely.

    # a) Ingest starts, acquires lease
    lease = reg.acquire("proj_a")
    assert lease._epoch == initial_epoch

    # b) Concurrent switch happens while lease is held
    with patch("axon.projects.project_dir", return_value=tmp_path / "projects" / "proj_b"), patch(
        "axon.projects.project_vector_path", return_value="v2"
    ), patch("axon.projects.project_bm25_path", return_value="b2"), patch(
        "axon.projects.set_active_project"
    ):
        brain.switch_project("proj_b")

    assert brain._active_project == "proj_b"
    # c) Verify epoch for proj_a was bumped
    assert reg._state("proj_a").epoch > initial_epoch

    # d) The stale lease is released. In the current code, _WriteLease.__del__
    # just decrements 'active'. It DOES NOT check if the epoch is still valid
    # for the project at the time of commit if the dev didn't use it.

    # Let's check if brain.ingest actually checks the lease epoch before final commit.
    # Looking at main.py:1710: _ingest_lease = _get_registry().acquire(self._active_project)
    # The lease is held during the whole function.

    # The CRITICAL BUG would be if brain.ingest saves to the CURRENT vector store
    # (which is now Proj B) using data that was intended for Proj A.

    # Let's verify if 'brain.vector_store' changes immediately on switch.
    # Yes, switch_project re-initializes self.vector_store.

    print(
        f"\n[QA] Switch test: Proj A Epoch={initial_epoch}, After Switch Proj A Epoch={reg._state('proj_a').epoch}"
    )
    lease.close()
    brain.close()


def test_epoch_mismatch_detection():
    """Verify that _WriteLease detects epoch mismatch via is_stale()."""
    reg = get_registry()
    state = reg._state("test_epoch_det")
    state.epoch = 5

    from axon.runtime import _WriteLease

    lease = _WriteLease(reg, "test_epoch_det", 5)

    # Epoch advances — lease should now be stale
    with state._lock:
        state.epoch = 6

    assert lease.is_stale() is True

    # Close is the public release method
    lease.close()
