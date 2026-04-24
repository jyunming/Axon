"""Unit tests for DynamicGraphBackend (Phase 3 — SQLite-WAL temporal graph).

All LLM calls are mocked so tests run offline without an LLM configured.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_brain(tmp_path):
    """Return a minimal fake AxonBrain with a tmp bm25_path."""
    cfg = SimpleNamespace(bm25_path=str(tmp_path), graph_backend="dynamic_graph")
    llm = MagicMock()
    llm.complete.return_value = ""
    return SimpleNamespace(config=cfg, llm=llm)


def _make_backend(tmp_path, llm_responses: dict | None = None):
    """Instantiate DynamicGraphBackend with a mocked LLM."""
    from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

    brain = _make_brain(tmp_path)
    if llm_responses:

        def _complete(prompt, **kwargs):
            for key, resp in llm_responses.items():
                if key in prompt:
                    return resp
            return ""

        brain.llm.complete.side_effect = _complete
    return DynamicGraphBackend(brain)


def _chunk(text: str, chunk_id: str = "c1") -> dict:
    return {"id": chunk_id, "text": text, "metadata": {"source": "test"}}


# ---------------------------------------------------------------------------
# Schema / init tests
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendInit:
    def test_db_file_created(self, tmp_path):
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        brain = _make_brain(tmp_path)
        DynamicGraphBackend(brain)
        assert (tmp_path / ".dynamic_graph.db").exists()

    def test_status_empty_db(self, tmp_path):
        backend = _make_backend(tmp_path)
        s = backend.status()
        assert s["backend"] == "dynamic_graph"
        assert s["episodes"] == 0
        assert s["entities"] == 0
        assert s["active_facts"] == 0

    def test_protocol_satisfied(self, tmp_path):
        from axon.graph_backends.base import GraphBackend
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        brain = _make_brain(tmp_path)
        backend = DynamicGraphBackend(brain)
        assert isinstance(backend, GraphBackend)


# ---------------------------------------------------------------------------
# ingest() tests
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendIngest:
    def test_ingest_no_llm_response(self, tmp_path):
        """With empty LLM responses, ingest stores episodes but no entities/facts."""
        backend = _make_backend(tmp_path)
        result = backend.ingest([_chunk("Alice works at Acme Corp.")])
        assert result.chunks_processed == 1
        # No extraction without LLM response
        assert result.entities_added == 0
        s = backend.status()
        assert s["episodes"] == 1

    def test_ingest_with_entity_extraction(self, tmp_path):
        """Entity extraction populates entities table."""
        backend = _make_backend(
            tmp_path,
            llm_responses={
                "Extract the key named entities": "Alice | PERSON | A software engineer\nAcme Corp | ORGANIZATION | A technology company",
                "Extract key relationships": "",
            },
        )
        result = backend.ingest([_chunk("Alice works at Acme Corp.", "c1")])
        assert result.entities_added == 2
        s = backend.status()
        assert s["entities"] == 2

    def test_ingest_with_fact_extraction(self, tmp_path):
        """Fact extraction populates facts table."""
        backend = _make_backend(
            tmp_path,
            llm_responses={
                "Extract the key named entities": "Alice | PERSON | Engineer\nAcme Corp | ORGANIZATION | Tech company",
                "Extract key relationships": "Alice | WORKS_FOR | Acme Corp | Alice is employed at Acme Corp | 9",
            },
        )
        result = backend.ingest([_chunk("Alice works at Acme Corp.", "c1")])
        assert result.relations_added == 1
        s = backend.status()
        assert s["active_facts"] == 1

    def test_ingest_entity_dedup(self, tmp_path):
        """Same entity ingested twice stays as one row in entities table."""
        backend = _make_backend(
            tmp_path,
            llm_responses={
                "Extract the key named entities": "Alice | PERSON | Engineer",
                "Extract key relationships": "",
            },
        )
        backend.ingest([_chunk("Alice runs.", "c1")])
        backend.ingest([_chunk("Alice codes.", "c2")])
        s = backend.status()
        assert s["entities"] == 1  # Alice deduplicated
        assert s["episodes"] == 2

    def test_ingest_exclusive_fact_supersedes(self, tmp_path):
        """A second exclusive fact for the same subject supersedes the first."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"

        # Insert two facts with exclusive relation IS_CEO_OF
        backend._upsert_entity("alice", "PERSON", "CEO", now)
        backend._upsert_entity("acme", "ORGANIZATION", "Tech", now)
        backend._upsert_entity("globex", "ORGANIZATION", "Corp", now)

        backend._upsert_fact("alice", "IS_CEO_OF", "acme", "Alice is CEO", 1.0, "c1", "ep1", now)
        later = "2026-06-01T00:00:00+00:00"
        backend._upsert_fact(
            "alice", "IS_CEO_OF", "globex", "Alice moved to Globex", 1.0, "c2", "ep2", later
        )

        s = backend.status()
        assert s["active_facts"] == 1
        assert s["superseded_facts"] == 1

    def test_ingest_non_exclusive_appends(self, tmp_path):
        """Non-exclusive facts are appended without superseding prior ones."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"

        backend._upsert_fact("alice", "KNOWS", "bob", "", 1.0, "c1", "ep1", now)
        backend._upsert_fact("alice", "KNOWS", "carol", "", 1.0, "c2", "ep2", now)

        s = backend.status()
        assert s["active_facts"] == 2
        assert s["superseded_facts"] == 0

    def test_ingest_empty_chunks(self, tmp_path):
        """Empty chunks list is a no-op."""
        backend = _make_backend(tmp_path)
        result = backend.ingest([])
        assert result.chunks_processed == 0
        assert backend.status()["episodes"] == 0


# ---------------------------------------------------------------------------
# retrieve() tests
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendRetrieve:
    def _seed_facts(self, backend, tmp_path):
        """Insert some facts directly for retrieval testing."""
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_entity("alice", "PERSON", "", now)
        backend._upsert_entity("acme corp", "ORGANIZATION", "", now)
        backend._upsert_entity("bob", "PERSON", "", now)
        backend._upsert_fact(
            "alice", "WORKS_FOR", "acme corp", "Alice at Acme", 0.9, "c1", "ep1", now
        )
        backend._upsert_fact(
            "bob", "FRIENDS_WITH", "alice", "Bob knows Alice", 0.8, "c2", "ep2", now
        )

    def test_retrieve_matching_subject(self, tmp_path):
        """Query matching subject returns facts."""
        from axon.graph_backends.base import RetrievalConfig

        backend = _make_backend(tmp_path)
        self._seed_facts(backend, tmp_path)
        results = backend.retrieve("alice", cfg=RetrievalConfig(top_k=10))
        assert len(results) >= 1
        all_matched = [name for r in results for name in r.matched_entity_names]
        assert "alice" in all_matched

    def test_retrieve_empty_graph(self, tmp_path):
        """Empty graph returns empty list."""
        backend = _make_backend(tmp_path)
        results = backend.retrieve("any query")
        assert results == []

    def test_retrieve_respects_top_k(self, tmp_path):
        """top_k limits results."""
        from axon.graph_backends.base import RetrievalConfig

        backend = _make_backend(tmp_path)
        self._seed_facts(backend, tmp_path)
        results = backend.retrieve("alice", cfg=RetrievalConfig(top_k=1))
        assert len(results) <= 1

    def test_retrieve_excludes_superseded(self, tmp_path):
        """Superseded facts are not returned."""
        from axon.graph_backends.base import RetrievalConfig

        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        later = "2026-06-01T00:00:00+00:00"
        backend._upsert_fact("alice", "IS_CEO_OF", "acme", "CEO 1", 1.0, "c1", "ep1", now)
        backend._upsert_fact("alice", "IS_CEO_OF", "globex", "CEO 2", 1.0, "c2", "ep2", later)

        results = backend.retrieve("alice", cfg=RetrievalConfig(top_k=10))
        # Only the second (active) fact should be returned
        assert len(results) == 1
        assert "globex" in results[0].text

    def test_retrieve_dedup_existing_results(self, tmp_path):
        """existing_results exclusion: already-returned context_ids are skipped."""
        from axon.graph_backends.base import RetrievalConfig

        backend = _make_backend(tmp_path)
        self._seed_facts(backend, tmp_path)
        all_results = backend.retrieve("alice", cfg=RetrievalConfig(top_k=10))
        assert all_results  # sanity

        # Pass first result as existing — should be excluded in second call
        existing = [{"id": all_results[0].context_id}]
        filtered = backend.retrieve(
            "alice", cfg=RetrievalConfig(top_k=10), existing_results=existing
        )
        ids_in_filtered = {r.context_id for r in filtered}
        assert all_results[0].context_id not in ids_in_filtered

    def test_retrieve_context_type_is_fact(self, tmp_path):
        """All returned contexts have context_type='fact'."""
        from axon.graph_backends.base import RetrievalConfig

        backend = _make_backend(tmp_path)
        self._seed_facts(backend, tmp_path)
        results = backend.retrieve("alice", cfg=RetrievalConfig(top_k=10))
        assert all(r.context_type == "fact" for r in results)
        assert all(r.backend_id == "dynamic_graph" for r in results)


# ---------------------------------------------------------------------------
# clear() / delete_documents() tests
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendClear:
    def _seed(self, backend):
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_entity("alice", "PERSON", "", now)
        backend._upsert_fact("alice", "KNOWS", "bob", "", 1.0, "c1", "ep1", now)

    def test_clear_empties_all_tables(self, tmp_path):
        backend = _make_backend(tmp_path)
        self._seed(backend)
        backend.clear()
        s = backend.status()
        assert s["entities"] == 0
        assert s["active_facts"] == 0
        assert s["episodes"] == 0

    def test_delete_documents_orphans_facts(self, tmp_path):
        """Deleting the only chunk supporting a fact supersedes that fact."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_entity("alice", "PERSON", "", now)
        backend._upsert_fact("alice", "KNOWS", "bob", "", 1.0, "c1", "ep1", now)

        assert backend.status()["active_facts"] == 1
        backend.delete_documents(["c1"])
        s = backend.status()
        assert s["active_facts"] == 0
        assert s["superseded_facts"] == 1

    def test_delete_documents_partial(self, tmp_path):
        """Deleting one chunk leaves facts from other chunks intact."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_entity("alice", "PERSON", "", now)
        backend._upsert_fact("alice", "KNOWS", "bob", "", 1.0, "c1", "ep1", now)
        backend._upsert_fact("alice", "KNOWS", "carol", "", 1.0, "c2", "ep2", now)

        backend.delete_documents(["c1"])
        s = backend.status()
        assert s["active_facts"] == 1  # c2 fact survives
        assert s["superseded_facts"] == 1


# ---------------------------------------------------------------------------
# graph_data() tests
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendGraphData:
    def test_graph_data_empty(self, tmp_path):
        from axon.graph_backends.base import GraphPayload

        backend = _make_backend(tmp_path)
        payload = backend.graph_data()
        assert isinstance(payload, GraphPayload)
        assert payload.nodes == []
        assert payload.links == []

    def test_graph_data_has_nodes_and_links(self, tmp_path):
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_fact("alice", "WORKS_FOR", "acme corp", "", 1.0, "c1", "ep1", now)
        payload = backend.graph_data()
        assert len(payload.nodes) == 2
        assert len(payload.links) == 1
        link = payload.links[0]
        assert link["source"] == "alice"
        assert link["target"] == "acme corp"


# ---------------------------------------------------------------------------
# finalize() tests
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendFinalize:
    def test_finalize_returns_result(self, tmp_path):
        from axon.graph_backends.base import FinalizationResult

        backend = _make_backend(tmp_path)
        result = backend.finalize()
        assert isinstance(result, FinalizationResult)
        assert result.backend_id == "dynamic_graph"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendFactory:
    def test_factory_creates_dynamic_backend(self, tmp_path):
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend
        from axon.graph_backends.factory import get_graph_backend

        brain = _make_brain(tmp_path)
        backend = get_graph_backend(brain)
        assert isinstance(backend, DynamicGraphBackend)

    def test_factory_unknown_backend_raises(self, tmp_path):
        import pytest

        from axon.graph_backends.factory import get_graph_backend

        brain = _make_brain(tmp_path)
        brain.config.graph_backend = "nonexistent"
        with pytest.raises(ValueError, match="Unknown graph_backend"):
            get_graph_backend(brain)


# ---------------------------------------------------------------------------
# Share-mount safety: journal mode + snapshot export/load
# ---------------------------------------------------------------------------


class TestDynamicGraphShareMountSafety:
    def test_journal_mode_is_delete_not_wal(self, tmp_path):
        """Owner DB uses DELETE journal — no -wal/-shm sidecars (cloud-sync-safe)."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_entity("alice", "PERSON", "", now)
        # PRAGMA should report 'delete' mode.
        mode = backend._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"
        # And no -wal/-shm sidecar files should have been created.
        assert not (tmp_path / ".dynamic_graph.db-wal").exists()
        assert not (tmp_path / ".dynamic_graph.db-shm").exists()

    def test_ingest_writes_snapshot_file(self, tmp_path):
        """Owner ingest emits a JSON snapshot for grantees to read."""
        from axon.graph_backends.dynamic_graph_backend import SNAPSHOT_FILENAME

        backend = _make_backend(
            tmp_path,
            llm_responses={
                "Extract the key named entities": "Alice | PERSON | Engineer",
                "Extract key relationships": "",
            },
        )
        backend.ingest([_chunk("Alice runs.", "c1")])
        snap = tmp_path / SNAPSHOT_FILENAME
        assert snap.exists()
        import json as _json

        data = _json.loads(snap.read_text(encoding="utf-8"))
        assert data["snapshot_version"] >= 1
        assert any(e["canonical_name"] == "alice" for e in data["entities"])

    def test_snapshot_written_atomically_no_tempfile_left(self, tmp_path):
        """Snapshot export uses tmp+replace — no ``.json.tmp`` remains afterwards."""
        backend = _make_backend(
            tmp_path,
            llm_responses={"Extract the key named entities": "Alice | PERSON | x"},
        )
        backend.ingest([_chunk("Alice.", "c1")])
        tmp_files = list(tmp_path.glob("*.json.tmp"))
        assert tmp_files == []

    def test_grantee_uses_in_memory_db_not_owner_file(self, tmp_path, monkeypatch):
        """A mounted (grantee) brain never opens the owner's on-disk SQLite."""
        from axon.graph_backends.dynamic_graph_backend import (
            SNAPSHOT_FILENAME,
            DynamicGraphBackend,
        )

        owner_brain = _make_brain(tmp_path)
        owner_brain._active_project = "research"  # owner — not a mount
        owner = DynamicGraphBackend(owner_brain)
        assert not owner._is_mounted
        # Seed an entity and a fact so the snapshot has content.
        now = "2026-01-01T00:00:00+00:00"
        owner._upsert_entity("alice", "PERSON", "engineer", now)
        owner._upsert_fact("alice", "WORKS_FOR", "acme", "", 0.9, "c1", "ep1", now)
        owner._export_snapshot()
        assert (tmp_path / SNAPSHOT_FILENAME).exists()

        # Now simulate a grantee pointing at the same path — they should load
        # the snapshot, not open the owner's .dynamic_graph.db.
        grantee_brain = _make_brain(tmp_path)
        grantee_brain._active_project = "mounts/shared"
        grantee = DynamicGraphBackend(grantee_brain)
        assert grantee._is_mounted
        # Grantee sees the snapshot data via an in-memory DB.
        s = grantee.status()
        assert s["entities"] == 1
        assert s["active_facts"] == 1
        # Grantee did NOT create or touch an on-disk DB in the mount.
        # (The owner's file still exists, but grantee's connection is :memory:.)
        rows = grantee._conn.execute("PRAGMA database_list").fetchall()
        main_file = [r["file"] for r in rows if r["name"] == "main"][0]
        assert main_file == ""  # in-memory DB reports empty file path

    def test_grantee_with_missing_snapshot_is_empty_not_error(self, tmp_path):
        """No snapshot file yet: grantee gets an empty graph, no exception."""
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        grantee_brain = _make_brain(tmp_path)
        grantee_brain._active_project = "mounts/nothing_yet"
        grantee = DynamicGraphBackend(grantee_brain)
        s = grantee.status()
        assert s["entities"] == 0
        assert s["active_facts"] == 0

    def test_grantee_with_corrupt_snapshot_logs_and_stays_empty(self, tmp_path):
        """A malformed snapshot is logged and treated as empty, not re-raised."""
        from axon.graph_backends.dynamic_graph_backend import (
            SNAPSHOT_FILENAME,
            DynamicGraphBackend,
        )

        (tmp_path / SNAPSHOT_FILENAME).write_text("not json {[", encoding="utf-8")
        grantee_brain = _make_brain(tmp_path)
        grantee_brain._active_project = "mounts/shared"
        grantee = DynamicGraphBackend(grantee_brain)
        s = grantee.status()
        assert s["entities"] == 0
