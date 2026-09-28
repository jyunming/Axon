"""AxonBrain.delete_documents — the one delete path behind POST /delete, the
CLI and the agent tool (issue #168).

Deleting a document must forget everything that would stop the same text from
being ingested again: the per-chunk dedup hashes, the source-level dedup
records and the _doc_versions entry.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

import axon.api as api_module
from axon.main import AxonBrain


class FakeVectorStore:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    def add(self, ids, texts, embeddings, metadatas=None):
        for i, (doc_id, text) in enumerate(zip(ids, texts)):
            meta = (metadatas or [{}] * len(ids))[i]
            self.docs[doc_id] = {"id": doc_id, "text": text, "metadata": dict(meta)}

    def get_by_ids(self, ids):
        return [dict(self.docs[i]) for i in ids if i in self.docs]

    def delete_by_ids(self, ids):
        for i in ids:
            self.docs.pop(i, None)


class FakeBM25:
    def __init__(self):
        self.corpus: list[dict] = []

    def add_documents(self, docs, save_deferred=False):
        self.corpus.extend(
            {"id": d["id"], "text": d["text"], "metadata": dict(d.get("metadata", {}))}
            for d in docs
        )

    def delete_documents(self, ids):
        drop = set(ids)
        self.corpus = [c for c in self.corpus if c["id"] not in drop]


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _bare_brain(project: str = "default") -> AxonBrain:
    brain = AxonBrain.__new__(AxonBrain)
    brain._own_vector_store = FakeVectorStore()
    brain._own_bm25 = FakeBM25()
    brain._graph_backend = MagicMock()
    brain._active_project = project
    brain._ingested_hashes = set()
    brain._doc_versions = {}
    brain._save_hash_store = MagicMock()
    brain._save_doc_versions = MagicMock()
    brain._assert_write_allowed = MagicMock()
    brain._doc_hash = lambda doc: _md5(doc.get("text", ""))
    return brain


def _store(brain: AxonBrain, chunk_id: str, text: str, **meta) -> None:
    """Put one chunk in both stores the way ingest() leaves it."""
    meta.setdefault("dedup_hash", _md5(text))
    brain._own_vector_store.add([chunk_id], [text], [[0.0]], [meta])
    brain._own_bm25.add_documents([{"id": chunk_id, "text": text, "metadata": meta}])
    brain._ingested_hashes.add(meta["dedup_hash"])


@pytest.fixture(autouse=True)
def _clean_source_hashes():
    api_module._source_hashes.clear()
    yield
    api_module._source_hashes.clear()


class TestDeleteByDocumentId:
    def test_forgets_chunk_hashes_versions_and_source_dedup(self):
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "alpha", source="A")
        _store(brain, "A_chunk_1", "beta", source="A")
        _store(brain, "B_chunk_0", "gamma", source="B")
        brain._doc_versions = {"A": {"content_hash": "x"}, "B": {"content_hash": "y"}}
        api_module._source_hashes["default"] = {"h": {"doc_id": "A"}}

        result = brain.delete_documents(["A"])

        assert result == {
            "status": "success",
            "deleted": 2,
            "doc_ids": ["A_chunk_0", "A_chunk_1"],
            "not_found": [],
        }
        assert set(brain._own_vector_store.docs) == {"B_chunk_0"}
        assert [c["id"] for c in brain._own_bm25.corpus] == ["B_chunk_0"]
        assert brain._ingested_hashes == {_md5("gamma")}
        assert brain._doc_versions == {"B": {"content_hash": "y"}}
        assert "default" not in api_module._source_hashes
        brain._graph_backend.delete_documents.assert_called_once_with(["A_chunk_0", "A_chunk_1"])
        brain._save_hash_store.assert_called_once()
        brain._save_doc_versions.assert_called_once()

    def test_parent_split_chunks_without_source_are_found(self):
        """In-process ingest with parent splitting gives chunks X_p<n>_chunk_<i>
        with source_id X_p<n> and no metadata.source (issue #168, problem 3)."""
        brain = _bare_brain()
        _store(brain, "X_p0_chunk_0", "one", source_id="X_p0", parent_text="one")
        _store(brain, "X_p1_chunk_0", "two", source_id="X_p1", parent_text="two")
        _store(brain, "X2_p0_chunk_0", "other", source_id="X2_p0", parent_text="other")

        result = brain.delete_documents(["X"])

        assert result["doc_ids"] == ["X_p0_chunk_0", "X_p1_chunk_0"]
        assert result["not_found"] == []
        assert set(brain._own_vector_store.docs) == {"X2_p0_chunk_0"}

    def test_unsplit_document_ending_in_p_digits_is_not_a_child(self):
        """An unsplit document whose own id is "report_p1" must not be swept
        up when "report" is deleted — only parent-split chunks (which carry
        parent_text) are matched on the stripped _p<n> suffix."""
        brain = _bare_brain()
        _store(brain, "report_p0_chunk_0", "part A", source_id="report_p0", parent_text="A")
        _store(brain, "report_p1_chunk_0", "unrelated", source_id="report_p1")

        result = brain.delete_documents(["report"])

        assert result["doc_ids"] == ["report_p0_chunk_0"]
        assert "report_p1_chunk_0" in brain._own_vector_store.docs

    def test_lazy_bm25_corpus_is_materialized_before_expansion(self):
        """BM25Retriever can leave .corpus empty until first use; delete must
        materialize it like the retriever's own methods do."""
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "alpha", source="A")
        bm25 = brain._own_bm25
        pending, bm25.corpus = bm25.corpus, []

        def _materialize():
            bm25.corpus = pending

        bm25.ensure_corpus_loaded = _materialize

        assert brain.delete_documents(["A"])["doc_ids"] == ["A_chunk_0"]

    def test_plain_split_chunks_matched_on_source_id(self):
        brain = _bare_brain()
        _store(brain, "Y_chunk_0", "one", source_id="Y")

        assert brain.delete_documents(["Y"])["deleted"] == 1

    def test_unknown_id_reported_not_found(self):
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "alpha", source="A")

        result = brain.delete_documents(["nope"])

        assert result == {"status": "success", "deleted": 0, "doc_ids": [], "not_found": ["nope"]}
        assert "A_chunk_0" in brain._own_vector_store.docs
        brain._graph_backend.delete_documents.assert_not_called()

    def test_namespaced_project_accepts_unprefixed_ids(self):
        """Non-default projects prefix chunk ids and source_ids with
        '<project_id>::'; callers pass the id they ingested with."""
        brain = _bare_brain(project="research")
        _store(brain, "pid::A_p0_chunk_0", "alpha", source_id="pid::A_p0", parent_text="alpha")
        _store(brain, "pid::C_chunk_0", "gamma", source_id="pid::C")

        with patch("axon.projects.get_project_id", return_value="pid"):
            result = brain.delete_documents(["A", "C_chunk_0"])

        assert sorted(result["doc_ids"]) == ["pid::A_p0_chunk_0", "pid::C_chunk_0"]
        assert result["not_found"] == []


class TestDedupHashes:
    def test_uses_recorded_hash_when_stored_text_was_rewritten(self):
        """Contextual retrieval prepends a sentence after the dedup hash is
        taken, so the stored text no longer hashes to the recorded value."""
        brain = _bare_brain()
        _store(
            brain,
            "A_chunk_0",
            "Situating sentence.\noriginal",
            source="A",
            dedup_hash=_md5("original"),
        )

        brain.delete_documents(["A"])

        assert brain._ingested_hashes == set()

    def test_falls_back_to_hashing_stored_text(self):
        """Chunks stored before dedup_hash existed have no recorded hash."""
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "legacy", source="A")
        del brain._own_vector_store.docs["A_chunk_0"]["metadata"]["dedup_hash"]

        brain.delete_documents(["A_chunk_0"])

        assert brain._ingested_hashes == set()


class TestDocVersions:
    def test_partial_delete_keeps_version_record(self):
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "alpha", source="A")
        _store(brain, "A_chunk_1", "beta", source="A")
        brain._doc_versions = {"A": {"content_hash": "x"}}

        brain.delete_documents(["A_chunk_0"])

        assert "A" in brain._doc_versions
        brain._save_doc_versions.assert_not_called()

    def test_chunk_keyed_version_record_is_dropped(self):
        """ingest() keys _doc_versions by the chunk id when there is no source."""
        brain = _bare_brain()
        _store(brain, "Z_chunk_0", "zeta", source_id="Z")
        brain._doc_versions = {"Z_chunk_0": {"content_hash": "x"}}

        brain.delete_documents(["Z"])

        assert brain._doc_versions == {}


class TestSafety:
    def test_write_denied_raises_before_touching_stores(self):
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "alpha", source="A")
        brain._assert_write_allowed.side_effect = PermissionError("mounted share")

        with pytest.raises(PermissionError):
            brain.delete_documents(["A"])
        assert "A_chunk_0" in brain._own_vector_store.docs

    def test_parent_project_uses_own_stores_not_merged_views(self):
        """On a parent project brain.bm25 is a read-only MultiBM25Retriever
        with no .corpus (issue #168, problem 4) — delete must not touch it."""
        from axon.vector_store import MultiBM25Retriever

        brain = _bare_brain()
        brain.bm25 = MultiBM25Retriever([])
        brain.vector_store = MagicMock()
        brain.vector_store.delete_by_ids.side_effect = RuntimeError("read-only view")
        _store(brain, "A_chunk_0", "alpha", source="A")

        assert brain.delete_documents(["A"])["deleted"] == 1

    def test_no_bm25_deletes_direct_chunk_ids(self):
        brain = _bare_brain()
        _store(brain, "A_chunk_0", "alpha", source="A")
        brain._own_bm25 = None
        brain._doc_versions = {"A": {"content_hash": "x"}}

        result = brain.delete_documents(["A_chunk_0", "A"])

        assert result["doc_ids"] == ["A_chunk_0"]
        assert result["not_found"] == ["A"]
        assert brain._ingested_hashes == set()
        # Without a corpus there's no way to tell whether other chunks of "A"
        # remain, so its version record is kept.
        assert "A" in brain._doc_versions


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestIngestDeleteReingest:
    def test_same_text_is_ingested_again_after_delete(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """The repro from issue #168: ingest A, delete A, ingest B with the same
        text — B must land instead of being dropped as already seen."""
        from axon.main import AxonConfig

        config = AxonConfig(
            dedup_on_ingest=True,
            graph_rag=False,
            raptor=False,
            bm25_path=str(tmp_path / "bm25"),
            vector_store_path=str(tmp_path / "vs"),
        )
        brain = AxonBrain(config)
        vs, bm25 = FakeVectorStore(), FakeBM25()
        brain.vector_store = brain._own_vector_store = vs
        brain.bm25 = brain._own_bm25 = bm25
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_doc_versions = MagicMock()
        brain._graph_backend = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1]] * len(texts))
        text = "The same paper text, stored twice under different ids."

        brain.ingest([{"id": "A", "text": text, "metadata": {"source": "A"}}])
        assert vs.docs, "first ingest wrote nothing"
        assert all(d["metadata"].get("dedup_hash") for d in vs.docs.values())

        result = brain.delete_documents(["A"])
        assert result["deleted"] >= 1 and not vs.docs
        assert brain._ingested_hashes == set()

        written = brain.ingest([{"id": "B", "text": text, "metadata": {"source": "B"}}])
        assert written >= 1
        assert {d["metadata"]["source"] for d in vs.docs.values()} == {"B"}
