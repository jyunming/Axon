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
