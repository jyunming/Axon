from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig
from axon.vector_store import OpenVectorStore


def test_add_unbalanced_inputs_raises():
    """Verify behavior with mismatched input lengths."""
    cfg = AxonConfig()
    # Prevent store initialization (avoid heavy deps)
    with patch.object(OpenVectorStore, "_init_store", lambda self: None):
        store = OpenVectorStore(cfg)
        store.provider = "chroma"
        store.collection = MagicMock()
        with pytest.raises(ValueError, match="length mismatch"):
            store.add(["id1", "id2"], ["text1"], [[0.1]])


def test_add_metadatas_mismatch_raises():
    """Verify metadatas length mismatch raises."""
    cfg = AxonConfig()
    with patch.object(OpenVectorStore, "_init_store", lambda self: None):
        store = OpenVectorStore(cfg)
        store.provider = "chroma"
        store.collection = MagicMock()
        with pytest.raises(ValueError, match="metadatas length mismatch"):
            store.add(["id1"], ["text1"], [[0.1]], metadatas=[{}, {}])


class TestUnreadableStoreDegradation:
    """An existing-but-unopenable store must not take the process down.

    Regression for #165: `tqdb.Database.open` raised out of `_init_store`, so
    `AxonBrain.__init__` raised, so an application embedding Axon could not
    start at all — even for work that never touches retrieval. The trigger was
    a TurboQuantDB format change that shipped in a patch release, but the
    fragility is independent of the cause: any unopenable store did this.
    """

    def _store_with_broken_manifest(self, tmp_path, monkeypatch):
        import json

        from axon.config import AxonConfig
        from axon.vector_store import OpenVectorStore

        (tmp_path / "manifest.json").write_text(
            json.dumps({"version": 2, "d": 384, "b": 4, "seed": 42}), encoding="utf-8"
        )

        class _Boom:
            @staticmethod
            def open(*a, **k):
                raise RuntimeError("io error: unexpected end of file")

        import sys
        import types

        fake = types.ModuleType("tqdb")
        fake.Database = _Boom
        monkeypatch.setitem(sys.modules, "tqdb", fake)
        cfg = AxonConfig(vector_store_path=str(tmp_path), bm25_path=str(tmp_path))
        return OpenVectorStore(cfg)

    def test_construction_survives_an_unopenable_store(self, tmp_path, monkeypatch):
        vs = self._store_with_broken_manifest(tmp_path, monkeypatch)
        assert vs.is_unreadable
        assert "unexpected end of file" in vs.unreadable_reason

    def test_search_returns_empty_instead_of_raising(self, tmp_path, monkeypatch):
        vs = self._store_with_broken_manifest(tmp_path, monkeypatch)
        assert vs.search([0.1] * 384, top_k=3) == []

    def test_add_is_refused_rather_than_overwriting(self, tmp_path, monkeypatch):
        """The dangerous case: `client is None` also means "create on first add".

        Without an explicit unreadable state, degrading would let the next
        ingest start a fresh store on top of files that were merely unreadable
        — turning a recoverable situation into real data loss.
        """
        import pytest

        vs = self._store_with_broken_manifest(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="Refusing to write"):
            vs.add(ids=["a"], texts=["t"], embeddings=[[0.1] * 384])

    def test_a_healthy_store_is_not_marked_unreadable(self, tmp_path):
        from axon.config import AxonConfig
        from axon.vector_store import OpenVectorStore

        cfg = AxonConfig(vector_store_path=str(tmp_path), bm25_path=str(tmp_path))
        assert OpenVectorStore(cfg).is_unreadable is False


class TestRebuildVectorStore:
    """The recovery path itself — the riskiest code in the degradation work.

    It moves the user's store aside before writing a new one, so the cases
    that matter are the ones where the write does not finish.
    """

    def _brain(self, tmp_path, chunks=3):
        import json
        from unittest.mock import MagicMock

        from axon.config import AxonConfig
        from axon.main import AxonBrain

        vsd, bm = tmp_path / "vector_store_data", tmp_path / "bm25_index"
        vsd.mkdir(parents=True)
        bm.mkdir(parents=True)
        (vsd / "manifest.json").write_text(
            json.dumps({"version": 2, "d": 4, "b": 4, "seed": 42}), encoding="utf-8"
        )
        (vsd / "marker").write_text("original", encoding="utf-8")
        (bm / "bm25_corpus.json").write_text(
            json.dumps(
                [
                    {"id": f"c{i}", "text": f"chunk {i}", "metadata": {"source": "s"}}
                    for i in range(chunks)
                ]
            ),
            encoding="utf-8",
        )
        cfg = AxonConfig(vector_store_path=str(vsd), bm25_path=str(bm))
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg
        brain._active_project = "default"
        brain.bm25 = MagicMock()
        brain.bm25.corpus = json.loads((bm / "bm25_corpus.json").read_text(encoding="utf-8"))
        brain.embedding = MagicMock()
        brain.embedding.embed = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        brain.vector_store = MagicMock()
        brain.vector_store.close = MagicMock()
        return brain, vsd, bm

    def test_dry_run_touches_nothing(self, tmp_path):
        brain, vsd, _ = self._brain(tmp_path)
        r = brain.rebuild_vector_store(dry_run=True)
        assert r["chunks"] == 3 and r["dry_run"] is True
        assert (vsd / "marker").read_text(encoding="utf-8") == "original"
        assert not list(vsd.parent.glob("vector_store_data.old-*"))

    def test_refuses_when_there_is_no_text_to_rebuild_from(self, tmp_path):
        import pytest

        brain, vsd, _ = self._brain(tmp_path, chunks=0)
        brain.bm25.corpus = []
        with pytest.raises(RuntimeError, match="Nothing to rebuild from"):
            brain.rebuild_vector_store()
        # the guard must fire before anything is moved
        assert (vsd / "marker").read_text(encoding="utf-8") == "original"

    def test_a_failed_rebuild_puts_the_original_back(self, tmp_path):
        """The case that matters: embedding dies part-way through.

        Leaving a fresh empty directory would be worse than the original
        breakage — an empty store is not `is_unreadable`, so the next run
        would treat it as "never ingested" and say nothing.
        """
        import pytest

        brain, vsd, _ = self._brain(tmp_path)

        def _boom(_texts):
            raise TimeoutError("embedding provider went away")

        brain.embedding.embed = _boom
        with pytest.raises(RuntimeError, match="put back"):
            brain.rebuild_vector_store()
        assert (vsd / "marker").read_text(encoding="utf-8") == "original"
        assert not list(vsd.parent.glob("vector_store_data.old-*")), "backup left orphaned"

    def test_keyboard_interrupt_also_restores(self, tmp_path):
        """KeyboardInterrupt is a BaseException — a plain `except Exception`
        would let it through with the store still moved aside, and Ctrl+C
        during a long re-embed is the likeliest interruption there is."""
        import pytest

        brain, vsd, _ = self._brain(tmp_path)

        def _interrupt(_texts):
            raise KeyboardInterrupt()

        brain.embedding.embed = _interrupt
        with pytest.raises(RuntimeError, match="put back"):
            brain.rebuild_vector_store()
        assert (vsd / "marker").read_text(encoding="utf-8") == "original"

    def test_duplicate_ids_are_dropped_and_reported(self, tmp_path):
        brain, _, _ = self._brain(tmp_path)
        brain.bm25.corpus = [
            {"id": "same", "text": "first", "metadata": {}},
            {"id": "same", "text": "second — different content", "metadata": {}},
            {"id": "other", "text": "third", "metadata": {}},
        ]
        r = brain.rebuild_vector_store(dry_run=True)
        assert r["chunks"] == 2
        assert r["duplicate_ids"] == 1
        assert r["rows_dropped_as_duplicates"] == 1
