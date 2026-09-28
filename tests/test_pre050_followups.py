"""Follow-ups from the 2026-09-28 capabilities audit.

- BM25Retriever.ensure_corpus_loaded(): code outside the retriever that reads
  ``.corpus`` directly (document deletion, the code-symbol index) must see the
  full corpus even before the lazily-decoded on-disk payload is first used.
- RemoteBrain.ingest() returns a count like AxonBrain.ingest().
- _atomic_persist.remove_orphaned_tmps(): crashed writers' temp files are
  removed, a live writer's are not.
- The API's source-level dedup records (_source_hashes, behind
  /collection/stale) survive a server restart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# BM25Retriever.ensure_corpus_loaded
# ---------------------------------------------------------------------------


class TestEnsureCorpusLoaded:
    def test_loads_a_lazily_decoded_corpus(self):
        from axon.retrievers import BM25Retriever

        r = BM25Retriever.__new__(BM25Retriever)
        r.corpus = []
        r._dedup_payload = {"payload": True}
        r._dedup_doc_count = 1
        r._corpus_dedup_fast_load_enabled = False
        r._text_intern_mode = "off"
        docs = [{"id": "a", "text": "alpha", "metadata": {}}]
        with patch.object(BM25Retriever, "_decode_loaded_corpus", return_value=docs):
            r.ensure_corpus_loaded()
        assert r.corpus == docs
        assert r._dedup_payload is None

    def test_noop_when_already_loaded(self):
        from axon.retrievers import BM25Retriever

        r = BM25Retriever.__new__(BM25Retriever)
        r.corpus = [{"id": "a"}]
        r._dedup_payload = None
        r.ensure_corpus_loaded()
        assert r.corpus == [{"id": "a"}]

    def test_symbol_search_loads_each_corpus_first(self):
        """code_retrieval used to read ``.corpus`` without loading it, so a
        symbol lookup straight after a restart saw an empty corpus."""
        from axon.code_retrieval import CodeRetrievalMixin

        loaded: list[str] = []

        class _Lazy:
            def __init__(self, name):
                self.name, self.corpus = name, []

            def ensure_corpus_loaded(self):
                loaded.append(self.name)

        host = MagicMock()
        host.bm25 = MagicMock(_retrievers=[_Lazy("p1"), _Lazy("p2")])
        host.config.symbol_index_engine = "python"
        host._get_symbol_cache.return_value = ([], [], {})
        try:
            CodeRetrievalMixin._symbol_channel_search(host, frozenset({"bm25retriever"}), top_k=5)
        except Exception:
            pass  # only the loading order matters here
        assert loaded == ["p1", "p2"]


# ---------------------------------------------------------------------------
# RemoteBrain.ingest
# ---------------------------------------------------------------------------


class TestRemoteIngestCount:
    def _brain(self, response):
        from axon.remote_brain import RemoteBrain

        rb = RemoteBrain.__new__(RemoteBrain)
        rb._active_project = "default"
        rb._request = MagicMock(return_value=response)
        return rb

    def test_counts_created_chunks(self):
        rb = self._brain(
            [
                {"id": "a", "status": "created", "chunks": 2},
                {"id": "b", "status": "skipped", "error": None},
                {"id": "c", "status": "created", "chunks": 1},
            ]
        )
        assert rb.ingest([{"id": "a", "text": "x"}]) == 3

    def test_zero_when_everything_was_a_duplicate(self):
        rb = self._brain([{"id": "a", "status": "skipped", "error": None}])
        assert rb.ingest([{"id": "a", "text": "x"}]) == 0

    def test_empty_input(self):
        rb = self._brain([])
        assert rb.ingest([]) == 0
        rb._request.assert_not_called()


# ---------------------------------------------------------------------------
# remove_orphaned_tmps
# ---------------------------------------------------------------------------


def _age(path: Path, *, hours: float) -> None:
    import time

    t = time.time() - hours * 3600
    os.utime(path, (t, t))


class TestRemoveOrphanedTmps:
    def test_removes_dead_writers_temps_only(self, tmp_path):
        from axon._atomic_persist import remove_orphaned_tmps

        dead = tmp_path / "meta.json.999999.0a1b2c3d.tmp"
        live = tmp_path / f"meta.json.{os.getpid()}.0a1b2c3d.tmp"
        user = tmp_path / "notes.tmp"
        nested = tmp_path / "sub" / "mount.json.999998.deadbeef.tmp"
        nested.parent.mkdir()
        for p in (dead, live, user, nested):
            p.write_text("x", encoding="utf-8")
            _age(p, hours=2)

        with patch("axon._pid_check.pid_alive", side_effect=lambda pid: pid == os.getpid()):
            assert remove_orphaned_tmps(tmp_path) == 2
        assert not dead.exists() and not nested.exists()
        assert live.exists() and user.exists()

    def test_keeps_recent_temp_even_if_pid_unknown_here(self, tmp_path):
        """On a synced store another machine may be mid-write; its pid means
        nothing on this machine, so only age can prove a temp is abandoned."""
        from axon._atomic_persist import remove_orphaned_tmps

        in_flight = tmp_path / "meta.json.424242.0a1b2c3d.tmp"
        in_flight.write_text("x", encoding="utf-8")
        with patch("axon._pid_check.pid_alive", return_value=False):
            assert remove_orphaned_tmps(tmp_path) == 0
        assert in_flight.exists()


# ---------------------------------------------------------------------------
# _source_hashes persistence
# ---------------------------------------------------------------------------


@pytest.fixture
def api_state():
    import axon.api as api

    saved = (dict(api._source_hashes), api._source_hashes_file)
    api._source_hashes.clear()
    api._source_hashes_file = None
    yield api
    api._source_hashes.clear()
    api._source_hashes.update(saved[0])
    api._source_hashes_file = saved[1]


class TestSourceHashesPersistence:
    def test_survive_a_restart(self, tmp_path, api_state):
        api = api_state
        api._load_source_hashes(tmp_path)
        api._record_dedup("some text", "doc-1", "research")

        api._source_hashes.clear()  # "restart"
        api._load_source_hashes(tmp_path)
        (record,) = api._source_hashes["research"].values()
        assert record["doc_id"] == "doc-1"

    def test_purge_is_persisted(self, tmp_path, api_state):
        api = api_state
        api._load_source_hashes(tmp_path)
        api._record_dedup("some text", "doc-1", "research")
        api._purge_dedup(["doc-1"], "research")
        data = json.loads((tmp_path / ".source_hashes.json").read_text(encoding="utf-8"))
        assert "research" not in data

    def test_process_that_never_loaded_does_not_write(self, tmp_path, api_state):
        """A CLI/library brain never loads the file; its clear() must not
        replace the server's records with an empty dict."""
        api = api_state
        path = tmp_path / ".source_hashes.json"
        path.write_text(json.dumps({"research": {"h": {"doc_id": "d"}}}), encoding="utf-8")
        api._record_dedup("other", "doc-2", "research")
        api._save_source_hashes()
        assert json.loads(path.read_text(encoding="utf-8")) == {"research": {"h": {"doc_id": "d"}}}

    def test_batch_records_write_once(self, tmp_path, api_state):
        api = api_state
        api._load_source_hashes(tmp_path)
        with patch("axon._atomic_persist.write_json_if_changed") as write:
            for i in range(5):
                api._record_dedup(f"text {i}", f"d{i}", "p", save=False)
            api._save_source_hashes()
        assert write.call_count == 1

    def test_corrupt_file_loads_empty(self, tmp_path, api_state):
        api = api_state
        (tmp_path / ".source_hashes.json").write_text("{not json", encoding="utf-8")
        api._load_source_hashes(tmp_path)
        assert api._source_hashes == {}
        assert api._source_hashes_file == Path(tmp_path) / ".source_hashes.json"
