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
        """Deleting the only chunk supporting a fact retracts that fact."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_entity("alice", "PERSON", "", now)
        backend._upsert_fact("alice", "KNOWS", "bob", "", 1.0, "c1", "ep1", now)

        assert backend.status()["active_facts"] == 1
        backend.delete_documents(["c1"])
        s = backend.status()
        assert s["active_facts"] == 0
        assert s["superseded_facts"] == 0
        assert s["retracted_facts"] == 1

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
        assert s["retracted_facts"] == 1


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
# close() — needed by main.py's switch_project()/close() wiring, which calls
# it whenever the active project's backend is torn down or replaced.
# ---------------------------------------------------------------------------


class TestDynamicGraphBackendClose:
    def test_close_does_not_raise(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.close()  # should not raise

    def test_close_is_idempotent(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.close()
        backend.close()  # calling twice must not raise


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

    @staticmethod
    def _make_grantee_with_snapshot(tmp_path, snapshot: dict):
        """Write *snapshot* to bm25_path and return a freshly-built grantee."""
        import json as _json

        from axon.graph_backends.dynamic_graph_backend import (
            SNAPSHOT_FILENAME,
            DynamicGraphBackend,
        )

        (tmp_path / SNAPSHOT_FILENAME).write_text(_json.dumps(snapshot), encoding="utf-8")
        grantee_brain = _make_brain(tmp_path)
        grantee_brain._active_project = "mounts/shared"
        return DynamicGraphBackend(grantee_brain)

    @staticmethod
    def _ent(eid: str, name: str, description: str = "") -> dict:
        return {
            "entity_id": eid,
            "canonical_name": name,
            "entity_type": "PERSON",
            "description": description,
            "first_seen_at": "2026-01-01T00:00:00+00:00",
            "last_seen_at": "2026-01-01T00:00:00+00:00",
        }

    def test_grantee_refuses_future_snapshot_version(self, tmp_path):
        """Audit P1: a snapshot from a NEWER Axon (snapshot_version > current)
        must NOT be silently replayed against the v1 schema. The grantee
        sees an empty graph instead of partially-populated junk.
        """
        from axon.graph_backends.dynamic_graph_backend import SNAPSHOT_VERSION

        grantee = self._make_grantee_with_snapshot(
            tmp_path,
            {
                "snapshot_version": SNAPSHOT_VERSION + 1,
                "entities": [self._ent("e1", "alice", "future-only field shape")],
                "facts": [],
                "v2_only_field": {"some": "thing"},
            },
        )
        s = grantee.status()
        assert s["entities"] == 0
        assert s["active_facts"] == 0

    def test_grantee_skips_non_dict_rows_in_snapshot(self, tmp_path):
        """Audit P1: a snapshot row that isn't a dict must be skipped, not
        raise AttributeError and leave the DB half-populated.
        """
        grantee = self._make_grantee_with_snapshot(
            tmp_path,
            {
                "snapshot_version": 1,
                "entities": [
                    self._ent("e1", "alice", "ok"),
                    "not-a-dict",
                    self._ent("e2", "bob", "also ok"),
                ],
                "facts": [],
            },
        )
        assert grantee.status()["entities"] == 2

    def test_grantee_refuses_non_list_entities(self, tmp_path):
        """Audit P1: entities/facts that aren't lists are refused before iteration."""
        grantee = self._make_grantee_with_snapshot(
            tmp_path,
            {
                "snapshot_version": 1,
                "entities": "not a list",
                "facts": [],
            },
        )
        assert grantee.status()["entities"] == 0


class TestDynamicGraphDbRelocation:
    """When bm25_path is on a cloud-sync / network path, the owner DB is
    redirected to ``~/.axon/graphs/<id>/`` so a sync client never observes
    a torn mid-write file."""

    def test_safe_path_keeps_db_inline(self, tmp_path):
        """Local path: DB stays at bm25_path (no relocation, no migration)."""
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        backend = DynamicGraphBackend(_make_brain(tmp_path))
        assert not backend._db_relocated
        assert backend._db_path == tmp_path / ".dynamic_graph.db"
        assert backend._db_path.exists()

    def test_cloud_sync_path_redirects_to_local_root(self, tmp_path, monkeypatch):
        """Synthetic OneDrive path: DB lands under ~/.axon/graphs/<id>/."""
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        # Pin Path.home() so the test is deterministic and writes only into tmp.
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

        # Build a project layout under a synthetic OneDrive subtree.
        synced_root = tmp_path / "OneDrive" / "AxonStore" / "alice" / "research"
        bm25_dir = synced_root / "bm25_index"
        bm25_dir.mkdir(parents=True)
        (synced_root / "meta.json").write_text('{"project_id": "proj-deadbeef"}', encoding="utf-8")

        brain = _make_brain(bm25_dir)
        backend = DynamicGraphBackend(brain)

        assert backend._db_relocated
        # DB is under ~/.axon/graphs/<project_id>/
        local_root = tmp_path / "home" / ".axon" / "graphs" / "proj-deadbeef"
        assert backend._db_path == local_root / ".dynamic_graph.db"
        assert backend._db_path.exists()
        # The synced bm25_path holds NO .dynamic_graph.db (only the snapshot
        # path is OK to live there once an ingest runs).
        assert not (bm25_dir / ".dynamic_graph.db").exists()

    def test_legacy_db_migrated_on_first_open(self, tmp_path, monkeypatch):
        """An existing DB at the old (synced) path is copied into the local root."""
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

        synced_root = tmp_path / "Dropbox" / "AxonStore" / "alice" / "research"
        bm25_dir = synced_root / "bm25_index"
        bm25_dir.mkdir(parents=True)
        (synced_root / "meta.json").write_text('{"project_id": "legacy-proj"}', encoding="utf-8")
        # Seed a "legacy" DB at the old location with a recognisable row.
        legacy_db = bm25_dir / ".dynamic_graph.db"
        import sqlite3 as _sqlite

        with _sqlite.connect(legacy_db) as _c:
            _c.executescript(
                "CREATE TABLE entities (entity_id TEXT PRIMARY KEY, "
                "canonical_name TEXT NOT NULL UNIQUE, entity_type TEXT NOT NULL DEFAULT 'X', "
                "description TEXT NOT NULL DEFAULT '', first_seen_at TEXT NOT NULL, "
                "last_seen_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}');"
                "INSERT INTO entities VALUES ('e1','legacy-marker','X','','t','t','{}');"
            )

        backend = DynamicGraphBackend(_make_brain(bm25_dir))
        assert backend._db_relocated
        assert backend._db_path.exists()
        assert backend._db_path != legacy_db

        rows = backend._conn.execute(
            "SELECT canonical_name FROM entities WHERE entity_id = 'e1'"
        ).fetchall()
        assert rows and rows[0]["canonical_name"] == "legacy-marker"


class TestConfigShareMountValidation:
    """AxonConfig.validate() flags axon_store_base / vector_store / bm25 paths
    that sit on cloud-sync, UNC, or WSL Windows mount filesystems."""

    def test_safe_path_emits_no_share_mount_warnings(self, tmp_path, monkeypatch):
        from axon.config import AxonConfig

        # Point everything inside tmp_path so __post_init__ derives safe paths.
        monkeypatch.setenv("AXON_STORE_BASE", str(tmp_path / "axon"))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("rag:\n  top_k: 10\n", encoding="utf-8")
        issues = AxonConfig.validate(str(cfg_path))
        share_mount_warns = [
            i
            for i in issues
            if "unsafe filesystem" in i.message and i.section in {"store", "vector_store", "bm25"}
        ]
        assert share_mount_warns == []

    def test_onedrive_store_base_emits_warning(self, tmp_path, monkeypatch):
        from axon.config import AxonConfig

        synced = tmp_path / "OneDrive" / "axon"
        synced.mkdir(parents=True)
        monkeypatch.setenv("AXON_STORE_BASE", str(synced))
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("rag:\n  top_k: 10\n", encoding="utf-8")
        issues = AxonConfig.validate(str(cfg_path))
        msgs = [i.message for i in issues if i.section == "store"]
        assert any("unsafe filesystem" in m and "cloud-sync" in m for m in msgs)


# ---------------------------------------------------------------------------
# upsert_fact() — explicit (agent / user) fact writes (PR5b)
# ---------------------------------------------------------------------------


class _Clock:
    """Deterministic replacement for dynamic_graph_backend._now_iso.

    Each call returns the current time, then advances it by ``step`` so that
    every timestamp is distinct but all writes stay well inside the ±1 s
    window the ingest path treats as "same-time".
    """

    def __init__(self, start, step_ms: int = 1):
        from datetime import timedelta

        self.t = start
        self.step = timedelta(milliseconds=step_ms)

    def __call__(self) -> str:
        cur = self.t
        self.t = self.t + self.step
        return cur.isoformat(timespec="microseconds")


def _iso_to_dt(iso: str):
    from datetime import datetime

    return datetime.fromisoformat(iso)


class TestUpsertFact:
    @staticmethod
    def _clock(monkeypatch):
        from datetime import datetime, timezone

        from axon.graph_backends import dynamic_graph_backend as dgb

        clock = _Clock(datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc))
        monkeypatch.setattr(dgb, "_now_iso", clock)
        return clock

    @staticmethod
    def _facts(backend, subject="alice"):
        return [
            dict(r)
            for r in backend._conn.execute(
                "SELECT fact_id, subject, relation, object, status, valid_at, invalid_at, "
                "scope_key, confidence, metadata FROM facts WHERE subject = ? ORDER BY valid_at",
                (subject,),
            ).fetchall()
        ]

    @staticmethod
    def _entity_types(backend):
        return {
            r["canonical_name"]: r["entity_type"]
            for r in backend._conn.execute("SELECT canonical_name, entity_type FROM entities")
        }

    def test_created(self, tmp_path):
        import json as _json

        backend = _make_backend(tmp_path)
        res = backend.upsert_fact(
            "Alice", "WORKS_FOR", "Acme", description="since 2024", confidence=0.8
        )
        assert res.status == "created"
        assert res.backend_id == "dynamic_graph"
        assert res.fact_id
        assert res.superseded_ids == [] and res.conflicted_ids == []
        rows = self._facts(backend)
        assert len(rows) == 1
        row = rows[0]
        assert (row["subject"], row["relation"], row["object"]) == ("alice", "WORKS_FOR", "acme")
        assert row["status"] == "active"
        assert row["confidence"] == 0.8
        assert _json.loads(row["metadata"]) == {"description": "since 2024", "provenance": "agent"}
        # Both entities were upserted (UNKNOWN type when new).
        assert self._entity_types(backend) == {"alice": "UNKNOWN", "acme": "UNKNOWN"}
        # Sentinel evidence + episode rows carry the provenance.
        ev = backend._conn.execute(
            "SELECT chunk_id, episode_id FROM fact_evidence WHERE fact_id = ?", (res.fact_id,)
        ).fetchall()
        assert [r["chunk_id"] for r in ev] == [f"agent:{res.fact_id}"]
        ep = backend._conn.execute(
            "SELECT content, metadata FROM episodes WHERE episode_id = ?", (ev[0]["episode_id"],)
        ).fetchone()
        assert ep["content"] == "Alice WORKS_FOR Acme: since 2024"
        assert _json.loads(ep["metadata"]) == {"provenance": "agent"}

    def test_unchanged_on_repeat(self, tmp_path):
        backend = _make_backend(tmp_path)
        first = backend.upsert_fact("Alice", "WORKS_FOR", "Acme")
        second = backend.upsert_fact("alice ", "works for", " ACME")
        assert second.status == "unchanged"
        assert second.fact_id == first.fact_id
        assert len(self._facts(backend)) == 1
        assert backend.status()["episodes"] == 1

    def test_replace_supersedes_ingest_extracted_non_exclusive_fact(self, tmp_path):
        backend = _make_backend(
            tmp_path,
            llm_responses={
                "Extract the key named entities": "Alice | PERSON | Engineer",
                "Extract key relationships": "Alice | WORKS_FOR | Acme | employed | 9",
            },
        )
        backend.ingest([_chunk("Alice works for Acme.", "c1")])
        (extracted,) = self._facts(backend)
        assert extracted["scope_key"] is None  # non-exclusive: no scope_key
        res = backend.upsert_fact("Alice", "WORKS_FOR", "Globex", replace=True)
        assert res.status == "superseded"
        assert res.superseded_ids == [extracted["fact_id"]]
        by_obj = {r["object"]: r for r in self._facts(backend)}
        assert by_obj["acme"]["status"] == "superseded"
        assert by_obj["acme"]["invalid_at"] is not None
        assert by_obj["globex"]["status"] == "active"
        # The extracted entity keeps its type; the new one is UNKNOWN.
        types = self._entity_types(backend)
        assert types["alice"] == "PERSON"
        assert types["globex"] == "UNKNOWN"

    def test_add_appends(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "KNOWS", "Bob")
        res = backend.upsert_fact("Alice", "KNOWS", "Carol", replace=False)
        assert res.status == "created"
        rows = self._facts(backend)
        assert {r["object"] for r in rows if r["status"] == "active"} == {"bob", "carol"}

    def test_add_mode_leaves_exclusive_relation_alone(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "IS_CEO_OF", "Acme")
        res = backend.upsert_fact("Alice", "IS_CEO_OF", "Globex", replace=False)
        assert res.status == "created"
        assert backend.status()["active_facts"] == 2

    def test_default_replaces_for_exclusive_relation(self, tmp_path):
        backend = _make_backend(tmp_path)
        first = backend.upsert_fact("Alice", "IS_CEO_OF", "Acme")
        res = backend.upsert_fact("Alice", "is ceo of", "Globex")
        assert res.status == "superseded"
        assert res.superseded_ids == [first.fact_id]
        s = backend.status()
        assert s["active_facts"] == 1 and s["superseded_facts"] == 1

    def test_default_adds_for_non_exclusive_relation(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "KNOWS", "Bob")
        res = backend.upsert_fact("Alice", "KNOWS", "Carol")
        assert res.status == "created"
        assert backend.status()["active_facts"] == 2

    def test_replace_with_identical_active_fact_supersedes_the_others(self, tmp_path):
        backend = _make_backend(tmp_path)
        keep = backend.upsert_fact("Alice", "KNOWS", "Bob")
        other = backend.upsert_fact("Alice", "KNOWS", "Carol")
        res = backend.upsert_fact("Alice", "KNOWS", "Bob", replace=True)
        assert res.status == "superseded"
        assert res.fact_id == keep.fact_id  # kept, no new row
        assert res.superseded_ids == [other.fact_id]
        assert len(self._facts(backend)) == 2
        assert backend.status()["active_facts"] == 1

    def test_two_writes_under_one_second_supersede_without_conflict(self, tmp_path, monkeypatch):
        self._clock(monkeypatch)
        backend = _make_backend(tmp_path)
        first = backend.upsert_fact("Alice", "IS_CEO_OF", "Acme")
        second = backend.upsert_fact("Alice", "IS_CEO_OF", "Globex")
        rows = {r["object"]: r for r in self._facts(backend)}
        # Both writes happened within a few ms — the ingest path would mark
        # them both 'conflicted'; explicit writes must not.
        gap = _iso_to_dt(rows["globex"]["valid_at"]) - _iso_to_dt(rows["acme"]["valid_at"])
        assert gap.total_seconds() < 1.0
        assert second.status == "superseded"
        assert second.superseded_ids == [first.fact_id]
        assert second.conflicted_ids == []
        assert rows["acme"]["status"] == "superseded"
        assert rows["globex"]["status"] == "active"
        assert backend.status()["conflicted_facts"] == 0
        assert backend.list_conflicts() == []

    def test_ingest_path_still_conflicts_within_one_second(self, tmp_path):
        """Regression guard: the explicit flag must not change extraction."""
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_fact("alice", "IS_CEO_OF", "acme", "", 1.0, "c1", "ep1", now)
        backend._upsert_fact("alice", "IS_CEO_OF", "globex", "", 1.0, "c2", "ep2", now)
        assert backend.status()["conflicted_facts"] == 2

    def test_survives_delete_documents_of_unrelated_chunks(self, tmp_path):
        backend = _make_backend(
            tmp_path,
            llm_responses={"Extract key relationships": "Bob | KNOWS | Carol | x | 9"},
        )
        backend.ingest([_chunk("Bob knows Carol.", "c1")])
        res = backend.upsert_fact("Alice", "WORKS_FOR", "Acme")
        backend.delete_documents(["c1"])
        (agent_fact,) = self._facts(backend, "alice")
        assert agent_fact["status"] == "active"
        (bob_fact,) = self._facts(backend, "bob")
        assert bob_fact["status"] == "retracted"
        # Deleting the sentinel chunk id is how an agent fact is retracted.
        backend.delete_documents([f"agent:{res.fact_id}"])
        assert self._facts(backend, "alice")[0]["status"] == "retracted"

    def test_snapshot_contains_fact(self, tmp_path):
        import json as _json

        from axon.graph_backends.dynamic_graph_backend import SNAPSHOT_FILENAME

        backend = _make_backend(tmp_path)
        res = backend.upsert_fact("Alice", "WORKS_FOR", "Acme")
        data = _json.loads((tmp_path / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
        assert any(f["fact_id"] == res.fact_id for f in data["facts"])
        assert {"alice", "acme"} <= {e["canonical_name"] for e in data["entities"]}

    def test_invalidates_cached_nx_graph(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "KNOWS", "Bob")
        assert backend._build_nx_graph_from_db().has_edge("alice", "bob")
        backend.upsert_fact("Alice", "KNOWS", "Carol")
        assert backend._cached_nx_graph is None
        assert backend._build_nx_graph_from_db().has_edge("alice", "carol")

    def test_mounted_instance_raises_permission_error(self, tmp_path):
        import pytest

        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        brain = _make_brain(tmp_path)
        brain._active_project = "mounts/shared"
        grantee = DynamicGraphBackend(brain)
        with pytest.raises(PermissionError):
            grantee.upsert_fact("Alice", "WORKS_FOR", "Acme")
        assert grantee.status()["active_facts"] == 0

    def test_relation_normalization(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "  works for ", "Acme")
        assert self._facts(backend)[0]["relation"] == "WORKS_FOR"

    def test_invalid_input_rejected(self, tmp_path):
        import pytest

        backend = _make_backend(tmp_path)
        for bad_rel in ("1ST_PLACE", "works-for", "", "   ", "A" * 65, "KNOWS!"):
            with pytest.raises(ValueError):
                backend.upsert_fact("Alice", bad_rel, "Acme")
        with pytest.raises(ValueError):
            backend.upsert_fact("   ", "KNOWS", "Acme")
        with pytest.raises(ValueError):
            backend.upsert_fact("Alice", "KNOWS", "")
        with pytest.raises(ValueError):
            backend.upsert_fact("Alice", "KNOWS", "Bob", confidence=1.5)
        assert backend.status()["active_facts"] == 0
        assert backend.status()["entities"] == 0

    def test_point_in_time_returns_old_then_new_object(self, tmp_path, monkeypatch):
        from datetime import timedelta

        from axon.graph_backends.base import RetrievalConfig

        clock = self._clock(monkeypatch)
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "IS_CEO_OF", "Acme")
        between = clock.t
        clock.t = clock.t + timedelta(days=30)
        backend.upsert_fact("Alice", "IS_CEO_OF", "Globex")
        after = clock.t

        def texts(pit):
            ctxs = backend.retrieve("alice", RetrievalConfig(top_k=10, point_in_time=pit))
            return {c.text for c in ctxs}

        # Before the update: only the old object — the new fact must not leak
        # in through the multi-hop expansion either.
        assert texts(between) == {"alice is ceo of acme"}
        assert texts(after) == {"alice is ceo of globex"}
        # Current view (no point_in_time) is the new object only.
        assert {c.text for c in backend.retrieve("alice")} == {"alice is ceo of globex"}

    def test_point_in_time_excludes_facts_retracted_by_delete(self, tmp_path, monkeypatch):
        from datetime import timedelta

        from axon.graph_backends.base import RetrievalConfig

        clock = self._clock(monkeypatch)
        backend = _make_backend(tmp_path)
        res = backend.upsert_fact("Alice", "KNOWS", "Bob")
        between = clock.t
        clock.t = clock.t + timedelta(days=1)
        backend.delete_documents([f"agent:{res.fact_id}"])
        ctxs = backend.retrieve("alice", RetrievalConfig(top_k=10, point_in_time=between))
        assert ctxs == []

    def test_unchanged_confirmation_protects_extracted_fact_from_delete(self, tmp_path):
        """An agent confirming an extracted fact (status 'unchanged') adds the
        sentinel evidence, so deleting the source document no longer
        retracts it."""
        backend = _make_backend(
            tmp_path,
            llm_responses={"Extract key relationships": "Alice | WORKS_FOR | Acme | x | 9"},
        )
        backend.ingest([_chunk("Alice works for Acme.", "doc1")])
        (extracted,) = self._facts(backend)
        res = backend.upsert_fact("Alice", "WORKS_FOR", "Acme")
        assert res.status == "unchanged"
        assert res.fact_id == extracted["fact_id"]
        chunks = {
            r["chunk_id"]
            for r in backend._conn.execute(
                "SELECT chunk_id FROM fact_evidence WHERE fact_id = ?", (res.fact_id,)
            )
        }
        assert chunks == {"doc1", f"agent:{res.fact_id}"}
        backend.delete_documents(["doc1"])
        assert self._facts(backend)[0]["status"] == "active"

    def test_repeated_confirmation_does_not_duplicate_evidence_or_episodes(self, tmp_path):
        backend = _make_backend(tmp_path)
        res = backend.upsert_fact("Alice", "KNOWS", "Bob")
        for _ in range(3):
            assert backend.upsert_fact("Alice", "KNOWS", "Bob").status == "unchanged"
        n_ev = backend._conn.execute(
            "SELECT COUNT(*) FROM fact_evidence WHERE fact_id = ?", (res.fact_id,)
        ).fetchone()[0]
        assert n_ev == 1
        assert backend.status()["episodes"] == 1

    def test_kept_fact_in_replace_mode_gets_sentinel_evidence(self, tmp_path):
        backend = _make_backend(
            tmp_path,
            llm_responses={
                "Extract key relationships": "Alice | KNOWS | Bob | x | 9\nAlice | KNOWS | Carol | y | 9"
            },
        )
        backend.ingest([_chunk("Alice knows Bob and Carol.", "doc1")])
        res = backend.upsert_fact("Alice", "KNOWS", "Bob", replace=True)
        assert res.status == "superseded"
        backend.delete_documents(["doc1"])
        by_obj = {r["object"]: r["status"] for r in self._facts(backend)}
        # bob: kept + sentinel, survives. carol: superseded by the replace,
        # stays superseded (history) even though its source is gone.
        assert by_obj == {"bob": "active", "carol": "superseded"}

    def test_retracted_fact_is_not_reactivated_by_identical_upsert(self, tmp_path):
        """A retracted fact stays retracted; asserting the same triple again
        creates a fresh active fact with a new id."""
        backend = _make_backend(tmp_path)
        first = backend.upsert_fact("Alice", "KNOWS", "Bob")
        backend.delete_documents([f"agent:{first.fact_id}"])
        assert self._facts(backend)[0]["status"] == "retracted"
        again = backend.upsert_fact("Alice", "KNOWS", "Bob", replace=True)
        assert again.status == "created"
        assert again.fact_id != first.fact_id
        assert again.superseded_ids == []
        by_id = {r["fact_id"]: r["status"] for r in self._facts(backend)}
        assert by_id == {first.fact_id: "retracted", again.fact_id: "active"}

    def test_clear_removes_it(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "WORKS_FOR", "Acme")
        backend.clear()
        s = backend.status()
        assert s["active_facts"] == 0 and s["episodes"] == 0 and s["entities"] == 0
        assert backend._conn.execute("SELECT COUNT(*) FROM fact_evidence").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Explicit retraction: delete_documents() marks facts 'retracted' (PR5b)
# ---------------------------------------------------------------------------


class TestRetraction:
    DAY_MS = 86_400_000

    @classmethod
    def _daily_clock(cls, monkeypatch):
        from datetime import datetime, timezone

        from axon.graph_backends import dynamic_graph_backend as dgb

        clock = _Clock(datetime(2026, 1, 1, tzinfo=timezone.utc), step_ms=cls.DAY_MS)
        monkeypatch.setattr(dgb, "_now_iso", clock)
        return clock

    @staticmethod
    def _texts_at(backend, query, pit):
        from axon.graph_backends.base import RetrievalConfig

        ctxs = backend.retrieve(query, RetrievalConfig(top_k=10, point_in_time=pit))
        return {c.text.split(":")[0] for c in ctxs}  # drop ": description"

    @staticmethod
    def _hq_backend(tmp_path):
        return _make_backend(
            tmp_path,
            llm_responses={
                "HQ is Seattle": "Acme | HEADQUARTERS_IN | Seattle | x | 9",
                "HQ moved to Portland": "Acme | HEADQUARTERS_IN | Portland | y | 9",
            },
        )

    def test_superseded_by_ingest_survives_deleting_its_stale_source(self, tmp_path, monkeypatch):
        """Repro (a): C1 says Seattle, C2 supersedes with Portland; deleting the
        stale C1 later must not erase Seattle from point-in-time history."""
        from datetime import timedelta

        clock = self._daily_clock(monkeypatch)
        backend = self._hq_backend(tmp_path)
        start = clock.t
        backend.ingest([_chunk("Acme HQ is Seattle.", "C1")])
        backend.ingest([_chunk("Acme HQ moved to Portland.", "C2")])
        between = start + timedelta(hours=12)
        assert self._texts_at(backend, "acme", between) == {"acme headquarters in seattle"}
        backend.delete_documents(["C1"])
        assert self._texts_at(backend, "acme", between) == {"acme headquarters in seattle"}
        statuses = {
            r["object"]: r["status"]
            for r in backend._conn.execute("SELECT object, status FROM facts")
        }
        assert statuses == {"seattle": "superseded", "portland": "active"}

    def test_superseded_by_agent_survives_deleting_its_source(self, tmp_path, monkeypatch):
        """Repro (b): extracted fact superseded by upsert_fact(replace=True),
        then its source is deleted — history must still show it."""
        from datetime import timedelta

        clock = self._daily_clock(monkeypatch)
        backend = _make_backend(
            tmp_path,
            llm_responses={"Extract key relationships": "Alice | IS_CEO_OF | Acme | x | 9"},
        )
        start = clock.t
        backend.ingest([_chunk("Alice is CEO of Acme.", "c1")])
        backend.upsert_fact("Alice", "IS_CEO_OF", "Globex", replace=True)
        before = start + timedelta(hours=12)
        backend.delete_documents(["c1"])
        assert self._texts_at(backend, "alice", before) == {"alice is ceo of acme"}
        assert {c.text.split(":")[0] for c in backend.retrieve("alice")} == {
            "alice is ceo of globex"
        }

    def test_retracting_the_current_fact_hides_it_but_keeps_superseded_history(
        self, tmp_path, monkeypatch
    ):
        from datetime import timedelta

        clock = self._daily_clock(monkeypatch)
        backend = self._hq_backend(tmp_path)
        start = clock.t
        backend.ingest([_chunk("Acme HQ is Seattle.", "C1")])
        backend.ingest([_chunk("Acme HQ moved to Portland.", "C2")])
        seattle_invalid = backend._conn.execute(
            "SELECT invalid_at FROM facts WHERE object = 'seattle'"
        ).fetchone()[0]
        backend.delete_documents(["C2"])  # retract the current fact
        backend.delete_documents(["C1"])  # the superseded one's source goes too
        rows = {
            r["object"]: (r["status"], r["invalid_at"])
            for r in backend._conn.execute("SELECT object, status, invalid_at FROM facts")
        }
        # Superseded history is kept, with its original end boundary.
        assert rows["seattle"] == ("superseded", seattle_invalid)
        assert rows["portland"][0] == "retracted" and rows["portland"][1] is not None
        assert self._texts_at(backend, "acme", start + timedelta(hours=12)) == {
            "acme headquarters in seattle"
        }
        assert self._texts_at(backend, "acme", clock.t) == set()  # Portland is gone
        assert backend.retrieve("acme") == []
        s = backend.status()
        assert (s["active_facts"], s["superseded_facts"], s["retracted_facts"]) == (0, 1, 1)

    def test_retracted_conflicted_fact_leaves_list_conflicts(self, tmp_path):
        backend = _make_backend(tmp_path)
        now = "2026-01-01T00:00:00+00:00"
        backend._upsert_fact("alice", "IS_CEO_OF", "acme", "", 1.0, "c1", "ep1", now)
        backend._upsert_fact("alice", "IS_CEO_OF", "globex", "", 1.0, "c2", "ep2", now)
        assert len(backend.list_conflicts()) == 2
        backend.delete_documents(["c1"])
        assert [r["object"] for r in backend.list_conflicts()] == ["globex"]

    def test_retracted_facts_are_not_exported_or_drawn(self, tmp_path):
        import json as _json

        from axon.graph_backends.dynamic_graph_backend import SNAPSHOT_FILENAME

        backend = _make_backend(tmp_path)
        res = backend.upsert_fact("Alice", "KNOWS", "Bob")
        backend.delete_documents([f"agent:{res.fact_id}"])
        data = _json.loads((tmp_path / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
        assert data["facts"] == []
        assert backend.graph_data().links == []

    def test_migration_marks_legacy_orphaned_superseded_facts_retracted(self, tmp_path):
        """A pre-PR DB (user_version 0) where delete_documents() left
        superseded facts with no evidence: those become 'retracted' on open;
        superseded facts that still have evidence are untouched."""
        from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

        backend = _make_backend(tmp_path)
        rows = [
            ("orphan", "superseded", "2026-02-01T00:00:00+00:00", None),
            ("kept", "superseded", "2026-02-01T00:00:00+00:00", "c2"),
            ("live", "active", None, "c3"),
        ]
        with backend._write_lock:
            for fid, status, inv, chunk in rows:
                backend._conn.execute(
                    "INSERT INTO facts (fact_id, subject, relation, object, valid_at, "
                    "invalid_at, status, confidence, metadata) VALUES "
                    "(?, 'alice', 'KNOWS', ?, '2026-01-01T00:00:00+00:00', ?, ?, 1.0, '{}')",
                    (fid, fid, inv, status),
                )
                if chunk:
                    backend._conn.execute(
                        "INSERT INTO fact_evidence (fact_id, chunk_id) VALUES (?, ?)",
                        (fid, chunk),
                    )
            backend._conn.execute("PRAGMA user_version = 0")
            backend._conn.commit()
        backend.close()

        reopened = DynamicGraphBackend(_make_brain(tmp_path))
        got = {
            r["fact_id"]: (r["status"], r["invalid_at"])
            for r in reopened._conn.execute("SELECT fact_id, status, invalid_at FROM facts")
        }
        assert got == {
            "orphan": ("retracted", "2026-02-01T00:00:00+00:00"),
            "kept": ("superseded", "2026-02-01T00:00:00+00:00"),
            "live": ("active", None),
        }
        assert reopened._conn.execute("PRAGMA user_version").fetchone()[0] >= 1
        reopened.close()
        # Idempotent: a second open changes nothing.
        again = DynamicGraphBackend(_make_brain(tmp_path))
        s = again.status()
        assert (s["retracted_facts"], s["superseded_facts"], s["active_facts"]) == (1, 1, 1)
        again.close()

    def test_clear_resets_nx_cache(self, tmp_path):
        backend = _make_backend(tmp_path)
        backend.upsert_fact("Alice", "KNOWS", "Bob")
        backend._build_nx_graph_from_db()
        assert backend._cached_nx_graph is not None
        backend.clear()
        assert backend._cached_nx_graph is None
        assert backend._cached_nx_time == 0.0
        assert backend._build_nx_graph_from_db().number_of_edges() == 0
