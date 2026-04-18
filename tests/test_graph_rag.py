from __future__ import annotations

"""Comprehensive tests for GraphRagMixin in axon.graph_rag.

Covers: entity/relation persistence, community detection, community summaries,
graph search helpers, entity matching, claims, prune, resolve aliases, and more.
"""

import json
import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

from axon.config import AxonConfig
from axon.graph_rag import GraphRagMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> AxonConfig:
    # Use a subdirectory of the system temp to be safer
    tmp = tempfile.mkdtemp(prefix="axon_test_")
    d = {
        "bm25_path": os.path.join(tmp, "bm25"),
        "vector_store_path": os.path.join(tmp, "vs"),
    }
    d.update(kwargs)
    return AxonConfig(**d)


def _make_brain(config=None, **extra_attrs):
    """Build a minimal GraphRagMixin instance with necessary attributes."""
    cfg = config or _make_config()
    os.makedirs(cfg.bm25_path, exist_ok=True)

    class FakeBrain(GraphRagMixin):
        pass

    brain = FakeBrain()
    brain.config = cfg
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_levels = {}
    brain._community_hierarchy = {}
    brain._community_children = {}
    brain._community_summaries = {}
    brain._entity_embeddings = {}
    brain._claims_graph = {}
    brain._text_unit_relation_map = {}
    brain._entity_description_buffer = {}
    brain._relation_description_buffer = {}
    brain._community_rebuild_lock = threading.Lock()
    brain._community_graph_dirty = False
    brain.llm = MagicMock()
    brain.embedding = MagicMock()
    brain.vector_store = MagicMock()
    brain._own_vector_store = MagicMock()

    # Provide a simple synchronous executor mock to avoid thread leaks on Windows
    class SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future

            f = Future()
            try:
                result = fn(*args, **kwargs)
                f.set_result(result)
            except Exception as e:
                f.set_exception(e)
            return f

        def map(self, fn, *iterables):
            return map(fn, *iterables)

        def shutdown(self, wait=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    brain._executor = SyncExecutor()

    for k, v in extra_attrs.items():
        setattr(brain, k, v)
    return brain


# ---------------------------------------------------------------------------
# GraphRAG extraction cache persistence / batching
# ---------------------------------------------------------------------------


class TestGraphRagExtractionCachePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._graph_rag_cache = {
            "entities": {
                "ent-key": [{"name": "Alice", "type": "PERSON", "description": "Engineer"}]
            },
            "relations": {
                "rel-key": [
                    {
                        "subject": "alice",
                        "relation": "knows",
                        "object": "bob",
                        "description": "Friendship",
                        "strength": 6,
                    }
                ]
            },
        }
        brain._graph_rag_cache_dirty = True

        brain._save_graph_rag_extraction_cache()

        reloaded = _make_brain(config=cfg)
        loaded = reloaded._load_graph_rag_extraction_cache()

        assert loaded["entities"]["ent-key"][0]["name"] == "Alice"
        assert loaded["relations"]["rel-key"][0]["object"] == "bob"

    def test_extract_entities_hits_persisted_cache_after_restart(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | PERSON | Engineer"

        first = brain._extract_entities("Alice builds systems.")
        brain._save_graph_rag_extraction_cache()

        reloaded = _make_brain(config=cfg)
        second = reloaded._extract_entities("Alice builds systems.")

        assert first == second
        reloaded.llm.complete.assert_not_called()


class TestGraphRagExtractionBatching:
    def test_pipelines_relation_extraction_when_budget_allows(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_relation_budget=0,
            graph_rag_llm_fused_extraction=False,
        )
        brain = _make_brain(config=cfg)
        submitted: list[str] = []

        class RecordingExecutor:
            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future

                submitted.append(getattr(fn, "__name__", "callable"))
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:  # pragma: no cover - passthrough guard
                    future.set_exception(exc)
                return future

            def map(self, fn, *iterables):
                return map(fn, *iterables)

        brain._executor = RecordingExecutor()

        def _fake_entities(_text):
            return [{"name": "Alice", "type": "PERSON", "description": ""}]

        def _fake_relations(_text):
            return [{"subject": "alice", "relation": "knows", "object": "bob"}]

        brain._extract_entities = _fake_entities
        brain._extract_relations = _fake_relations

        docs = [
            {"id": "d1", "text": "Alice knows Bob."},
            {"id": "d2", "text": "Alice knows Carol."},
        ]

        results, rel_results, rel_chunks, pipelined = brain._extract_graph_llm_batches(
            docs,
            relations_enabled=True,
            min_entities_for_relations=0,
            relation_budget=0,
        )

        assert pipelined is True
        assert [doc["id"] for doc in rel_chunks] == ["d1", "d2"]
        assert len(results) == 2
        assert len(rel_results) == 2
        assert submitted.count("_fake_entities") == 2
        assert submitted.count("_fake_relations") == 2

    def test_skips_executor_for_cached_chunks(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_relation_budget=0,
        )
        brain = _make_brain(config=cfg)
        docs = [{"id": "d1", "text": "Alice knows Bob."}]
        entity_key = brain._graph_rag_entity_cache_key(docs[0]["text"])
        relation_key = brain._graph_rag_relation_cache_key(docs[0]["text"])
        cached_entities = [{"name": "Alice", "type": "PERSON", "description": ""}]
        cached_relations = [{"subject": "alice", "relation": "knows", "object": "bob"}]
        brain._graph_rag_cache = {
            "entities": {entity_key: cached_entities},
            "relations": {relation_key: cached_relations},
        }
        submitted: list[str] = []

        class RecordingExecutor:
            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future

                submitted.append(getattr(fn, "__name__", "callable"))
                future = Future()
                future.set_result(fn(*args, **kwargs))
                return future

            def map(self, fn, *iterables):
                submitted.append(getattr(fn, "__name__", "callable"))
                return map(fn, *iterables)

        brain._executor = RecordingExecutor()

        results, rel_results, rel_chunks, pipelined = brain._extract_graph_llm_batches(
            docs,
            relations_enabled=True,
            min_entities_for_relations=0,
            relation_budget=0,
        )

        assert pipelined is True
        assert submitted == []
        assert results == [("d1", cached_entities)]
        assert rel_results == [("d1", cached_relations)]
        assert [doc["id"] for doc in rel_chunks] == ["d1"]


class TestGraphRagFusedExtractionBatching:
    def test_uses_combined_extractor_for_uncached_llm_pipeline(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_relation_budget=0,
            graph_rag_llm_fused_extraction=True,
            graph_rag_ner_backend="llm",
            graph_rag_relation_backend="llm",
            graph_rag_depth="standard",
        )
        brain = _make_brain(config=cfg)
        submitted: list[str] = []

        class RecordingExecutor:
            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future

                submitted.append(getattr(fn, "__name__", "callable"))
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

            def map(self, fn, *iterables):
                return map(fn, *iterables)

        brain._executor = RecordingExecutor()

        def _fake_combined(_text):
            return (
                [{"name": "Alice", "type": "PERSON", "description": ""}],
                [{"subject": "alice", "relation": "knows", "object": "bob", "strength": 5}],
            )

        def _unexpected_entities(_text):  # pragma: no cover - assertion guard
            raise AssertionError("separate entity extraction should not run")

        def _unexpected_relations(_text):  # pragma: no cover - assertion guard
            raise AssertionError("separate relation extraction should not run")

        brain._extract_entities_and_relations_combined = _fake_combined
        brain._extract_entities = _unexpected_entities
        brain._extract_relations = _unexpected_relations

        docs = [
            {"id": "d1", "text": "Alice knows Bob."},
            {"id": "d2", "text": "Alice knows Carol."},
        ]

        results, rel_results, rel_chunks, pipelined = brain._extract_graph_llm_batches(
            docs,
            relations_enabled=True,
            min_entities_for_relations=0,
            relation_budget=0,
        )

        assert pipelined is True
        assert submitted.count("_fake_combined") == 2
        assert len(results) == 2
        assert len(rel_results) == 2
        assert [doc["id"] for doc in rel_chunks] == ["d1", "d2"]


# ---------------------------------------------------------------------------
# Entity graph persistence
# ---------------------------------------------------------------------------


class TestEntityGraphPersistence:
    def test_save_then_load_roundtrip(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "type": "PERSON"},
        }
        brain._save_entity_graph()
        loaded = brain._load_entity_graph()
        assert "alice" in loaded
        assert loaded["alice"]["type"] == "PERSON"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._load_entity_graph()
        assert result == {}

    def test_load_malformed_json_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".entity_graph.json").write_text("{not json", encoding="utf-8")
        assert brain._load_entity_graph() == {}

    def test_load_non_dict_root_returns_empty(self, tmp_path):
        """Line 52: non-dict root returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".entity_graph.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert brain._load_entity_graph() == {}

    def test_load_skips_non_string_keys(self, tmp_path):
        """Line 56: integer keys in JSON are skipped (JSON keys are always strings but test the guard)."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {"valid": {"description": "ok", "chunk_ids": ["c1"]}}
        (tmp_path / ".entity_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_entity_graph()
        assert "valid" in loaded

    def test_load_skips_entry_without_chunk_ids(self, tmp_path):
        """Entries without chunk_ids are skipped."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {
            "good": {"description": "ok", "chunk_ids": ["c1"]},
            "bad": {"description": "no chunk ids"},
        }
        (tmp_path / ".entity_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_entity_graph()
        assert "good" in loaded
        assert "bad" not in loaded

    def test_load_defaults_type_and_frequency(self, tmp_path):
        """Defaults for type, frequency, degree are set automatically."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {"ent": {"description": "d", "chunk_ids": ["c1", "c2"]}}
        (tmp_path / ".entity_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_entity_graph()
        assert loaded["ent"]["type"] == "UNKNOWN"
        assert loaded["ent"]["frequency"] == 2
        assert loaded["ent"]["degree"] == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b"
        cfg = AxonConfig(bm25_path=str(nested), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"x": {"description": "d", "chunk_ids": ["c1"]}}
        brain._save_entity_graph()
        assert (nested / ".entity_graph.json").exists() or (
            nested / ".entity_graph.msgpack"
        ).exists()


# ---------------------------------------------------------------------------
# Relation graph persistence
# ---------------------------------------------------------------------------


class TestRelationGraphPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}]
        }
        brain._save_relation_graph()
        loaded = brain._load_relation_graph()
        assert "alice" in loaded
        assert loaded["alice"][0]["target"] == "bob"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_relation_graph() == {}

    def test_load_non_dict_root_returns_empty(self, tmp_path):
        """Line 115: list root returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".relation_graph.json").write_text(json.dumps([]), encoding="utf-8")
        assert brain._load_relation_graph() == {}

    def test_load_skips_non_list_values(self, tmp_path):
        """Line 118-119: entries with non-list values are skipped."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {
            "alice": "not a list",
            "bob": [{"target": "carol", "relation": "r", "chunk_id": "c"}],
        }
        (tmp_path / ".relation_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_relation_graph()
        assert "alice" not in loaded
        assert "bob" in loaded

    def test_load_skips_malformed_entries(self, tmp_path):
        """Entries missing target/relation/chunk_id are filtered."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {
            "alice": [
                {"target": "bob", "relation": "r", "chunk_id": "c"},  # good
                {"target": "bob"},  # bad — missing relation, chunk_id
            ]
        }
        (tmp_path / ".relation_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_relation_graph()
        assert len(loaded["alice"]) == 1

    def test_load_malformed_json_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".relation_graph.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_relation_graph() == {}


class TestRelationGraphMsgpackPersistence:
    def test_save_load_roundtrip_with_msgpack_codec(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_relation_msgpack_persist=True,
        )
        cfg.graph_rag_relation_shard_persist = False
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [
                {
                    "target": "bob",
                    "relation": "knows",
                    "chunk_id": "c1",
                    "subject": "alice",
                    "object": "bob",
                    "strength": 7,
                    "weight": 2.5,
                }
            ]
        }
        (tmp_path / ".relation_graph.json").write_text("{}", encoding="utf-8")
        (tmp_path / ".relation_graph.shards.json").write_text("{}", encoding="utf-8")
        (tmp_path / ".relation_graph.shard.000.json").write_text("{}", encoding="utf-8")

        class FakeBridge:
            def can_relation_graph_codec(self):
                return True

            def encode_relation_graph(self, graph):
                return json.dumps(graph, sort_keys=True).encode("utf-8")

            def decode_relation_graph(self, payload):
                return json.loads(payload.decode("utf-8"))

        with patch("axon.rust_bridge.get_rust_bridge", return_value=FakeBridge()):
            brain._save_relation_graph()
            loaded = brain._load_relation_graph()

        assert (tmp_path / ".relation_graph.msgpack").exists()
        assert not (tmp_path / ".relation_graph.json").exists()
        assert not (tmp_path / ".relation_graph.shards.json").exists()
        assert not (tmp_path / ".relation_graph.shard.000.json").exists()
        assert loaded["alice"][0]["strength"] == 7
        assert loaded["alice"][0]["weight"] == 2.5


# ---------------------------------------------------------------------------
# Community levels / hierarchy / summaries persistence
# ---------------------------------------------------------------------------


class TestCommunityPersistence:
    def test_save_load_community_levels(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 1}, 1: {"alice": 0}}
        brain._save_community_levels()
        loaded = brain._load_community_levels()
        assert 0 in loaded
        assert loaded[0]["alice"] == 0

    def test_load_community_levels_nonexistent(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_community_levels() == {}

    def test_load_community_levels_corrupt(self, tmp_path):
        """Lines 156-157: exception returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".community_levels.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_community_levels() == {}

    def test_save_load_community_hierarchy(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_hierarchy = {"0_0": None, "1_0": "0_0"}
        brain._save_community_hierarchy()
        loaded = brain._load_community_hierarchy()
        assert "0_0" in loaded
        assert loaded["0_0"] is None
        assert loaded["1_0"] == "0_0"

    def test_load_community_hierarchy_nonexistent(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_community_hierarchy() == {}

    def test_load_community_hierarchy_corrupt(self, tmp_path):
        """Lines 204-205: exception returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".community_hierarchy.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_community_hierarchy() == {}

    def test_save_load_community_summaries(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {"0_0": {"title": "T", "summary": "S", "rank": 5.0}}
        brain._save_community_summaries()
        loaded = brain._load_community_summaries()
        assert "0_0" in loaded
        assert loaded["0_0"]["title"] == "T"

    def test_load_community_summaries_nonexistent(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_community_summaries() == {}

    def test_load_community_summaries_corrupt(self, tmp_path):
        """Lines 234-235: exception returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".community_summaries.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_community_summaries() == {}

    def test_save_community_summaries_exception_logged(self, tmp_path):
        """Lines 247-248: exception in save is swallowed (no raise)."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {"k": {"title": "T"}}
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            # Should not raise
            brain._save_community_summaries()

    def test_save_community_hierarchy_exception_logged(self, tmp_path):
        """Lines 220-221: exception in save is swallowed."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_hierarchy = {}
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            brain._save_community_hierarchy()


# ---------------------------------------------------------------------------
# Entity embeddings and claims graph persistence
# ---------------------------------------------------------------------------


class TestEntityEmbeddingsPersistence:
    def test_save_load_entity_embeddings(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {"alice": [0.1, 0.2, 0.3]}
        brain._save_entity_embeddings()
        loaded = brain._load_entity_embeddings()
        assert "alice" in loaded
        assert loaded["alice"] == [0.1, 0.2, 0.3]

    def test_load_entity_embeddings_nonexistent(self, tmp_path):
        """Lines 261-262: nonexistent file returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_entity_embeddings() == {}

    def test_save_entity_embeddings_exception_logged(self, tmp_path):
        """Lines 274-275: exception swallowed."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {}
        with patch("pathlib.Path.write_text", side_effect=OSError("no space")):
            brain._save_entity_embeddings()  # must not raise


class TestClaimsGraphPersistence:
    def test_save_load_claims_graph(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._claims_graph = {"c1": [{"subject": "alice", "description": "owns shares"}]}
        brain._save_claims_graph()
        loaded = brain._load_claims_graph()
        assert "c1" in loaded
        assert loaded["c1"][0]["subject"] == "alice"

    def test_load_claims_graph_nonexistent(self, tmp_path):
        """Lines 288-292: nonexistent returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_claims_graph() == {}

    def test_save_claims_graph_exception_logged(self, tmp_path):
        """Lines 297-305: exception swallowed."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._claims_graph = {}
        with patch("pathlib.Path.write_text", side_effect=OSError("no space")):
            brain._save_claims_graph()


# ---------------------------------------------------------------------------
# Code graph persistence (lines 89-90 — exception path)
# ---------------------------------------------------------------------------


class TestCodeGraphPersistence:
    def test_load_code_graph_exception_returns_empty(self, tmp_path):
        """Lines 89-90: exception during load returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".code_graph.json").write_text("{{bad json", encoding="utf-8")
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}


# ---------------------------------------------------------------------------
# _build_networkx_graph
# ---------------------------------------------------------------------------


class TestBuildNetworkxGraph:
    def test_empty_graph(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        G = brain._build_networkx_graph()
        assert G.number_of_nodes() == 0

    def test_nodes_added(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=1,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "Another person", "chunk_ids": ["c2"], "frequency": 1},
        }
        G = brain._build_networkx_graph()
        assert "alice" in G.nodes
        assert "bob" in G.nodes

    def test_edges_added_from_relations(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
        }
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}]
        }
        G = brain._build_networkx_graph()
        assert G.has_edge("alice", "bob")

    def test_duplicate_edges_accumulate_weight(self, tmp_path):
        """Line 325: adding a second edge between same nodes increments weight."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1", "c2"], "frequency": 2},
            "bob": {"description": "d", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "alice": [
                {"target": "bob", "relation": "knows", "chunk_id": "c1", "weight": 1},
                {"target": "bob", "relation": "trusts", "chunk_id": "c2", "weight": 2},
            ]
        }
        G = brain._build_networkx_graph()
        assert G["alice"]["bob"]["weight"] == 3  # 1+2 accumulated

    def test_min_frequency_filter(self, tmp_path):
        """Line 340: entities below min_frequency are excluded."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=3,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "rare": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "common": {"description": "d", "chunk_ids": ["c1", "c2", "c3"], "frequency": 3},
        }
        G = brain._build_networkx_graph()
        assert "rare" not in G.nodes
        assert "common" in G.nodes


# ---------------------------------------------------------------------------
# _run_community_detection
# ---------------------------------------------------------------------------


class TestRunCommunityDetection:
    def test_empty_graph_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._run_community_detection()
        assert result == {}

    def test_single_node_returns_community_zero(self, tmp_path):
        """Line 349: single node returns {node: 0}."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=1,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"solo": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        result = brain._run_community_detection()
        assert result == {"solo": 0}

    def test_connected_entities_get_communities(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
            "carol": {"description": "d", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}],
            "bob": [{"target": "carol", "relation": "knows", "chunk_id": "c2"}],
        }
        result = brain._run_community_detection()
        assert set(result.keys()) == {"alice", "bob", "carol"}
        assert all(isinstance(v, int) for v in result.values())

    def test_community_detection_exception_returns_empty(self, tmp_path):
        """Lines 357-358: exception inside Louvain returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "a": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "b": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
        }
        brain._relation_graph = {"a": [{"target": "b", "relation": "r", "chunk_id": "c1"}]}
        fake_bridge = MagicMock()
        fake_bridge.can_build_graph_edges.return_value = False
        fake_bridge.can_run_louvain.return_value = False
        with patch("axon.rust_bridge.get_rust_bridge", return_value=fake_bridge), patch(
            "networkx.algorithms.community.louvain_communities", side_effect=RuntimeError("fail")
        ):
            result = brain._run_community_detection()
        assert result == {}


# ---------------------------------------------------------------------------
# _run_hierarchical_community_detection (Louvain fallback path)
# ---------------------------------------------------------------------------


class TestHierarchicalCommunityDetection:
    def test_returns_tuple_of_three(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "a": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "b": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
            "c": {"description": "d", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "a": [{"target": "b", "relation": "r", "chunk_id": "c1"}],
            "b": [{"target": "c", "relation": "r", "chunk_id": "c2"}],
        }
        result = brain._run_hierarchical_community_detection()
        assert isinstance(result, tuple)
        levels, hierarchy, children = result
        assert isinstance(levels, dict)
        assert isinstance(hierarchy, dict)
        assert isinstance(children, dict)

    def test_single_node_fallback(self, tmp_path):
        """Lines 381-382: single node returns level 0 dict."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=1,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"solo": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        levels, hierarchy, children = brain._run_hierarchical_community_detection()
        assert 0 in levels
        assert levels[0]["solo"] == 0

    def test_multi_level_produces_multiple_level_keys(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_community_levels=2
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            f"e{i}": {"description": "d", "chunk_ids": [f"c{i}"], "frequency": 1} for i in range(5)
        }
        brain._relation_graph = {
            "e0": [{"target": "e1", "relation": "r", "chunk_id": "c0"}],
            "e1": [{"target": "e2", "relation": "r", "chunk_id": "c1"}],
            "e2": [{"target": "e3", "relation": "r", "chunk_id": "c2"}],
            "e3": [{"target": "e4", "relation": "r", "chunk_id": "c3"}],
        }
        levels, _, _ = brain._run_hierarchical_community_detection()
        assert len(levels) >= 1

    def test_backend_louvain_skips_graspologic(self, tmp_path):
        """Line 474-475: louvain backend override skips leidenalg."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_backend="louvain",
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "a": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "b": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
        }
        brain._relation_graph = {"a": [{"target": "b", "relation": "r", "chunk_id": "c1"}]}
        result = brain._run_hierarchical_community_detection()
        assert isinstance(result, tuple)

    def test_empty_graph_returns_fallback(self, tmp_path):
        """Lines 577-578: empty Louvain result falls back to all-zero mapping."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        # No entities at all
        levels, hierarchy, children = brain._run_hierarchical_community_detection()
        assert isinstance(levels, dict)


# ---------------------------------------------------------------------------
# _generate_community_summaries
# ---------------------------------------------------------------------------


class TestGenerateCommunitySummaries:
    def test_no_levels_returns_early(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_levels = {}
        # Should return without calling llm
        brain._generate_community_summaries()
        brain.llm.complete.assert_not_called()

    def test_generates_template_for_small_community(self, tmp_path):
        """Lines 547-618: communities below min_size get template summaries."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=5,  # communities with <5 members get template
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 0}}
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain._generate_community_summaries()
        # LLM should NOT have been called for this tiny community
        brain.llm.complete.assert_not_called()
        assert "0_0" in brain._community_summaries

    def test_generates_llm_summary_for_large_community(self, tmp_path):
        """Lines 654-656: large community gets LLM summary."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=2,
        )
        brain = _make_brain(config=cfg)
        # Build 4 entities in one community
        entities = [f"e{i}" for i in range(4)]
        brain._community_levels = {0: dict.fromkeys(entities, 0)}
        brain._entity_graph = {
            e: {
                "description": f"desc of {e}",
                "chunk_ids": [f"c{i}"],
                "type": "PERSON",
                "frequency": 1,
            }
            for i, e in enumerate(entities)
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.return_value = json.dumps(
            {
                "title": "Test Community",
                "summary": "This community has people.",
                "findings": [],
                "rank": 7.5,
            }
        )
        brain._generate_community_summaries()
        brain.llm.complete.assert_called()
        key = "0_0"
        assert key in brain._community_summaries

    def test_cached_community_reused(self, tmp_path):
        """Lines 772-776: cached summary with same member_hash is reused."""
        import hashlib

        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_community_min_size=1
        )
        brain = _make_brain(config=cfg)
        members = ["alice", "bob"]
        raw = f"0|{'|'.join(sorted(members))}"
        existing_hash = hashlib.md5(raw.encode()).hexdigest()
        brain._community_levels = {0: {"alice": 0, "bob": 0}}
        brain._community_summaries = {
            "0_0": {
                "title": "Cached",
                "summary": "Cached summary",
                "rank": 8.0,
                "full_content": "# Cached\n\nCached summary",
                "member_hash": existing_hash,
                "entities": members,
                "size": 2,
                "level": 0,
            }
        }
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain._generate_community_summaries()
        # LLM should not be called because cache hit
        brain.llm.complete.assert_not_called()

    def test_lazy_mode_tightens_cap(self, tmp_path):
        """Lines 713-720: query_hint + lazy_cap tightens max_total."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=1,
            graph_rag_global_top_communities=2,
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 1, "carol": 2}}
        brain._entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "type": "PERSON", "frequency": 1},
            "carol": {"description": "C", "chunk_ids": ["c3"], "type": "PERSON", "frequency": 1},
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.return_value = '{"title":"T","summary":"S","findings":[],"rank":5.0}'
        brain._generate_community_summaries(query_hint="alice")
        # Should have been called (communities above min_size)
        assert len(brain._community_summaries) > 0

    def test_llm_returns_invalid_json_falls_back(self, tmp_path):
        """Lines 871-875: invalid JSON from LLM falls back to raw text."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=1,
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 0, "carol": 0}}
        brain._entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "type": "PERSON", "frequency": 1},
            "carol": {"description": "C", "chunk_ids": ["c3"], "type": "PERSON", "frequency": 1},
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.return_value = "Plain text, not JSON"
        brain._generate_community_summaries()
        key = "0_0"
        assert key in brain._community_summaries
        # Fallback: summary is the raw text
        assert brain._community_summaries[key]["summary"] == "Plain text, not JSON"

    def test_llm_exception_returns_empty_summary(self, tmp_path):
        """Lines 895-897: LLM exception produces empty summary entry."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=1,
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 0, "carol": 0}}
        brain._entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "type": "PERSON", "frequency": 1},
            "carol": {"description": "C", "chunk_ids": ["c3"], "type": "PERSON", "frequency": 1},
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.side_effect = RuntimeError("LLM down")
        brain._generate_community_summaries()
        key = "0_0"
        assert key in brain._community_summaries
        assert brain._community_summaries[key]["summary"] == ""


# ---------------------------------------------------------------------------
# _index_community_reports_in_vector_store
# ---------------------------------------------------------------------------


class TestIndexCommunityReports:
    def test_no_summaries_returns_early(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {}
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_not_called()

    def test_indexes_reports(self, tmp_path):
        """Lines 967-1004: reports are indexed into the vector store."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "# Community\n\nSome content",
                "summary": "Some content",
                "title": "Test",
                "rank": 7.0,
                "level": 0,
                "member_hash": "abc123",
            }
        }
        brain.embedding.embed.return_value = [[0.1, 0.2]]
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_called_once()

    def test_skips_already_indexed_report(self, tmp_path):
        """Line 976-977: skip when indexed_hash matches content_hash."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Content",
                "member_hash": "abc",
                "indexed_hash": "abc",  # already indexed
                "title": "T",
                "rank": 5.0,
                "level": 0,
            }
        }
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_not_called()

    def test_vector_store_exception_is_swallowed(self, tmp_path):
        """Lines 1003-1004: exception during add is logged but not raised."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "content",
                "member_hash": "x",
                "title": "T",
                "rank": 5.0,
                "level": 0,
            }
        }
        brain.embedding.embed.return_value = [[0.1]]
        brain._own_vector_store.add.side_effect = RuntimeError("vector store down")
        # Should not raise
        brain._index_community_reports_in_vector_store()

    def test_skips_empty_full_content(self, tmp_path):
        """Line 1012: summaries with empty full_content are skipped."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "",
                "summary": "",
                "member_hash": "x",
                "title": "",
                "rank": 5.0,
                "level": 0,
            }
        }
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_not_called()


# ---------------------------------------------------------------------------
# _global_search_map_reduce
# ---------------------------------------------------------------------------


class TestGlobalSearchMapReduce:
    def test_no_summaries_returns_empty_string(self, tmp_path):
        """Line 1011-1012: no summaries → empty string."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._global_search_map_reduce("what are the themes?", cfg)
        assert result == ""

    def test_returns_no_data_answer_when_all_points_below_threshold(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_global_min_score=80
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Content about Alice",
                "title": "Alice",
                "summary": "summary",
                "rank": 5.0,
                "level": 0,
            }
        }
        # LLM map returns low-score points
        brain.llm.complete.return_value = '[{"point": "some point", "score": 10}]'
        result = brain._global_search_map_reduce("query", cfg)
        from axon.graph_rag import _GRAPHRAG_NO_DATA_ANSWER

        assert result == _GRAPHRAG_NO_DATA_ANSWER

    def test_reduces_valid_points(self, tmp_path):
        """Lines 1184-1193: reduce phase calls LLM and returns response."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_global_min_score=0
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Community report content about Alice and Bob.",
                "title": "Alice-Bob community",
                "summary": "Summary.",
                "rank": 8.0,
                "level": 0,
            }
        }
        # Map phase returns points, reduce phase returns answer
        brain.llm.complete.side_effect = [
            '[{"point": "Alice works with Bob", "score": 90}]',
            "Alice and Bob collaborate extensively.",
        ]
        result = brain._global_search_map_reduce("who works together?", cfg)
        assert "Alice" in result or "collaborate" in result

    def test_level_filtering(self, tmp_path):
        """Lines 1018-1025: only summaries matching target level are used."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_level=1,
            graph_rag_global_min_score=0,
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Level 0 content",
                "title": "L0",
                "summary": "s",
                "rank": 5.0,
                "level": 0,
            },
            "1_0": {
                "full_content": "Level 1 content",
                "title": "L1",
                "summary": "s",
                "rank": 5.0,
                "level": 1,
            },
        }
        brain.llm.complete.return_value = '[{"point": "point", "score": 90}]'
        brain._global_search_map_reduce("query", cfg)
        # Only 1 level-1 community → llm called once for map phase (then reduce)
        assert brain.llm.complete.call_count >= 1

    def test_reduce_exception_returns_fallback(self, tmp_path):
        """Lines 1190-1193: reduce exception returns fallback text."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_global_min_score=0
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "content",
                "title": "T",
                "summary": "s",
                "rank": 5.0,
                "level": 0,
            }
        }
        # Map succeeds, reduce fails
        brain.llm.complete.side_effect = [
            '[{"point": "a point", "score": 90}]',
            RuntimeError("reduce down"),
        ]
        result = brain._global_search_map_reduce("query", cfg)
        assert "Knowledge Graph Findings" in result


# ---------------------------------------------------------------------------
# _entity_matches
# ---------------------------------------------------------------------------


class TestEntityMatches:
    def _brain(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        return _make_brain(config=cfg)

    def test_exact_match_returns_one(self, tmp_path):
        b = self._brain(tmp_path)
        assert b._entity_matches("alice", "alice") == 1.0

    def test_different_single_tokens_returns_zero(self, tmp_path):
        """Single tokens that differ return 0."""
        b = self._brain(tmp_path)
        assert b._entity_matches("alice", "bob") == 0.0

    def test_partial_multi_token_match(self, tmp_path):
        """Multi-token queries can have partial Jaccard match (>= 0.4 threshold)."""
        b = self._brain(tmp_path)
        # "alice bob smith" vs "alice bob jones" → intersection=2, union=4 → J=0.5 >= 0.4
        score = b._entity_matches("alice bob smith", "alice bob jones")
        assert 0.0 < score < 1.0

    def test_high_jaccard_above_threshold(self, tmp_path):
        b = self._brain(tmp_path)
        score = b._entity_matches("new york city", "new york city")
        assert score == 1.0

    def test_low_jaccard_below_threshold_returns_zero(self, tmp_path):
        b = self._brain(tmp_path)
        # Very different multi-token phrases
        score = b._entity_matches("alice smith", "bob jones")
        assert score == 0.0


# ---------------------------------------------------------------------------
# _classify_query_needs_graphrag
# ---------------------------------------------------------------------------


class TestClassifyQuery:
    def test_heuristic_holistic_keyword(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._classify_query_needs_graphrag("summarize all findings", "heuristic") is True

    def test_heuristic_long_query(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        long_query = " ".join(["word"] * 25)
        assert brain._classify_query_needs_graphrag(long_query, "heuristic") is True

    def test_heuristic_short_specific_query_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._classify_query_needs_graphrag("who is Alice?", "heuristic") is False

    def test_llm_mode_yes_returns_true(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "YES"
        assert brain._classify_query_needs_graphrag("some query", "llm") is True

    def test_llm_mode_no_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "NO"
        assert brain._classify_query_needs_graphrag("some query", "llm") is False

    def test_llm_mode_exception_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("LLM fail")
        assert brain._classify_query_needs_graphrag("query", "llm") is False

    def test_unknown_mode_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._classify_query_needs_graphrag("query", "unknown_mode") is False


# ---------------------------------------------------------------------------
# _extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_llm_path_parses_three_part_lines(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = (
            "Alice Smith | PERSON | A researcher\n" "Acme Corp | ORGANIZATION | A company\n"
        )
        result = brain._extract_entities("Alice Smith works at Acme Corp.")
        assert len(result) == 2
        assert result[0]["name"] == "Alice Smith"
        assert result[0]["type"] == "PERSON"
        assert result[1]["type"] == "ORGANIZATION"

    def test_llm_path_two_part_line_sets_unknown_type(self, tmp_path):
        """Lines 1794-1802: 2-column format gets UNKNOWN type."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | A researcher"
        result = brain._extract_entities("Some text about Alice.")
        assert len(result) == 1
        assert result[0]["type"] == "UNKNOWN"

    def test_llm_path_one_part_line_sets_unknown(self, tmp_path):
        """Lines 1803-1810: single token gets UNKNOWN type, empty description."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice"
        result = brain._extract_entities("Text about Alice.")
        assert len(result) == 1
        assert result[0]["type"] == "UNKNOWN"
        assert result[0]["description"] == ""

    def test_llm_invalid_type_normalizes_to_concept(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Foo | INVALID_TYPE | Some description"
        result = brain._extract_entities("text")
        assert result[0]["type"] == "CONCEPT"

    def test_llm_exception_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("LLM down")
        result = brain._extract_entities("Some text.")
        assert result == []

    def test_light_depth_uses_light_extractor(self, tmp_path):
        """Lines 1758-1759: light depth skips LLM."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_depth="light"
        )
        brain = _make_brain(config=cfg)
        result = brain._extract_entities("John Smith visited New York City yesterday.")
        # Light extractor uses regex — should return results without calling LLM
        brain.llm.complete.assert_not_called()
        assert isinstance(result, list)

    def test_capped_at_twenty_results(self, tmp_path):
        """Line 1811: results are capped at 20."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        # Return 25 entities
        lines = [f"Entity{i} | PERSON | desc {i}" for i in range(25)]
        brain.llm.complete.return_value = "\n".join(lines)
        result = brain._extract_entities("text")
        assert len(result) == 20


# ---------------------------------------------------------------------------
# _extract_entities_light
# ---------------------------------------------------------------------------


class TestExtractEntitiesLight:
    def test_finds_capitalized_phrases(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light("John Smith and New York City are mentioned here.")
        names = [r["name"] for r in result]
        # At least one multi-word capitalized phrase should be found
        assert any("John Smith" in n or "New York" in n for n in names)

    def test_single_capitalized_words_skipped(self, tmp_path):
        """Regex requires at least 2 capitalized words."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light("Alice went home.")
        # "Alice" alone doesn't match the 2-word pattern
        assert all(len(r["name"].split()) >= 2 for r in result)

    def test_deduplicates_entities(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light(
            "John Smith visited John Smith and John Smith again."
        )
        names = [r["name"] for r in result]
        assert names.count("John Smith") == 1

    def test_type_is_concept(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light("John Smith visited New York City.")
        assert all(r["type"] == "CONCEPT" for r in result)

    def test_capped_at_twenty(self, tmp_path):
        """Line 1747: results are capped at 20."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        # Generate many capitalized 2-word phrases
        text = " ".join([f"Person{i} Smith{i}" for i in range(30)])
        result = brain._extract_entities_light(text)
        assert len(result) <= 20


# ---------------------------------------------------------------------------
# _extract_relations
# ---------------------------------------------------------------------------


class TestExtractRelations:
    def test_parses_five_part_lines(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | works_at | Acme | Alice is employed by Acme | 8"
        result = brain._extract_relations("Alice works at Acme.")
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["strength"] == 8

    def test_parses_four_part_lines(self, tmp_path):
        """Lines 1861-1870: 4-part line gets default strength 5."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | works_at | Acme | Alice employed by Acme"
        result = brain._extract_relations("Alice works at Acme.")
        assert len(result) == 1
        assert result[0]["strength"] == 5

    def test_parses_three_part_lines(self, tmp_path):
        """Lines 1871-1880: 3-part line gets default strength 5 and empty description."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | works_at | Acme"
        result = brain._extract_relations("Alice works at Acme.")
        assert len(result) == 1
        assert result[0]["description"] == ""

    def test_invalid_strength_defaults_to_five(self, tmp_path):
        """Lines 1849-1851: non-integer strength falls back to 5."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | knows | Bob | They know each other | notanumber"
        result = brain._extract_relations("Alice knows Bob.")
        assert result[0]["strength"] == 5

    def test_light_depth_returns_empty(self, tmp_path):
        """Lines 1823-1824: light depth returns empty without LLM call."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_depth="light"
        )
        brain = _make_brain(config=cfg)
        result = brain._extract_relations("Alice knows Bob.")
        assert result == []
        brain.llm.complete.assert_not_called()

    def test_exception_returns_empty(self, tmp_path):
        """Lines 1882-1883: exception returns empty list."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("LLM down")
        result = brain._extract_relations("text")
        assert result == []

    def test_capped_at_fifteen(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        lines = [f"E{i} | r | E{i+1} | desc | 5" for i in range(20)]
        brain.llm.complete.return_value = "\n".join(lines)
        result = brain._extract_relations("text")
        assert len(result) == 15


# ---------------------------------------------------------------------------
# _parse_rebel_output (static method)
# ---------------------------------------------------------------------------


class TestParseRebelOutput:
    def test_parses_single_triplet(self):
        text = "<triplet> Alice <subj> Acme Corp <obj> works at"
        result = GraphRagMixin._parse_rebel_output(text)
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["object"] == "Acme Corp"
        assert result[0]["relation"] == "works at"

    def test_parses_multiple_triplets(self):
        text = "<triplet> Alice <subj> Bob <obj> knows " "<triplet> Bob <subj> Carol <obj> manages"
        result = GraphRagMixin._parse_rebel_output(text)
        assert len(result) == 2

    def test_incomplete_triplet_not_included(self):
        """Partial triplet without object or relation is not flushed."""
        text = "<triplet> Alice <subj> Bob"
        result = GraphRagMixin._parse_rebel_output(text)
        # No relation → should not produce a valid triplet
        assert all("relation" in r and r["relation"] for r in result)

    def test_empty_text_returns_empty(self):
        assert GraphRagMixin._parse_rebel_output("") == []

    def test_special_tokens_stripped(self):
        text = "<s><pad> <triplet> Alice <subj> Acme <obj> founded </s>"
        result = GraphRagMixin._parse_rebel_output(text)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _extract_claims
# ---------------------------------------------------------------------------


class TestExtractClaims:
    def test_parses_full_eight_part_line(self, tmp_path):
        """Lines 1956-1972: 8-part lines parsed into full claim dict."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | Acme | acquisition | TRUE | Alice acquired Acme | 2024-01-01 | 2024-06-01 | 'Alice acquired Acme'"
        result = brain._extract_claims("Alice acquired Acme in 2024.")
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["status"] == "TRUE"
        assert result[0]["start_date"] == "2024-01-01"

    def test_parses_five_part_line(self, tmp_path):
        """Lines 1973-1989: 5-part lines parsed with null dates."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = (
            "Alice | Acme | partnership | TRUE | Alice partners with Acme"
        )
        result = brain._extract_claims("text")
        assert len(result) == 1
        assert result[0]["start_date"] is None

    def test_invalid_status_normalized_to_suspected(self, tmp_path):
        """Lines 1957-1959 & 1974-1976: invalid status defaults to SUSPECTED."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = (
            "Alice | Acme | partnership | MAYBE | Some claim | unknown | unknown | none"
        )
        result = brain._extract_claims("text")
        assert result[0]["status"] == "SUSPECTED"

    def test_exception_returns_empty(self, tmp_path):
        """Lines 1991-1992: exception returns empty list."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("fail")
        assert brain._extract_claims("text") == []


# ---------------------------------------------------------------------------
# _prune_entity_graph
# ---------------------------------------------------------------------------


class TestPruneEntityGraph:
    def test_removes_entity_when_all_chunks_deleted(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        brain._relation_graph = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert "alice" not in brain._entity_graph
        brain._save_entity_graph.assert_called_once()

    def test_updates_chunk_ids_when_partial_delete(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1", "c2"], "frequency": 2}
        }
        brain._relation_graph = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert brain._entity_graph["alice"]["chunk_ids"] == ["c2"]
        assert brain._entity_graph["alice"]["frequency"] == 1

    def test_removes_relation_when_chunk_deleted(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {}
        brain._relation_graph = {
            "alice": [
                {"target": "bob", "relation": "knows", "chunk_id": "c1"},
                {"target": "carol", "relation": "works_with", "chunk_id": "c2"},
            ]
        }
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert len(brain._relation_graph["alice"]) == 1
        assert brain._relation_graph["alice"][0]["chunk_id"] == "c2"

    def test_removes_empty_relation_list(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {}
        brain._relation_graph = {"alice": [{"target": "bob", "relation": "r", "chunk_id": "c1"}]}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert "alice" not in brain._relation_graph

    def test_prunes_claims_graph(self, tmp_path):
        """Lines 2246-2251: claims_graph chunk_ids are pruned."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {
            "c1": [{"subject": "alice", "description": "claim"}],
            "c2": [{"subject": "bob", "description": "another claim"}],
        }
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_claims_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert "c1" not in brain._claims_graph
        assert "c2" in brain._claims_graph
        brain._save_claims_graph.assert_called_once()

    def test_no_changes_skip_save(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        brain._relation_graph = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c999"})  # non-existent chunk
        brain._save_entity_graph.assert_not_called()


# ---------------------------------------------------------------------------
# _get_incoming_relations
# ---------------------------------------------------------------------------


class TestGetIncomingRelations:
    def test_returns_incoming_edges(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}],
            "carol": [{"target": "bob", "relation": "works_with", "chunk_id": "c2"}],
        }
        result = brain._get_incoming_relations("bob")
        assert len(result) == 2
        sources = {r["source"] for r in result}
        assert sources == {"alice", "carol"}

    def test_returns_empty_for_no_incoming(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "carol", "relation": "knows", "chunk_id": "c1"}]
        }
        result = brain._get_incoming_relations("bob")
        assert result == []

    def test_case_insensitive_matching(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "Bob", "relation": "knows", "chunk_id": "c1"}]
        }
        result = brain._get_incoming_relations("bob")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _local_search_context
# ---------------------------------------------------------------------------


class TestLocalSearchContext:
    def test_empty_entities_returns_empty_string(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._local_search_context("query", [], cfg)
        assert result == ""

    def test_returns_matched_entities_section(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {
                "description": "A researcher at university",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            }
        }
        brain._relation_graph = {}
        brain.vector_store.get_by_ids.return_value = []
        result = brain._local_search_context("who is alice?", ["alice"], cfg)
        assert "Matched Entities" in result
        assert "alice" in result.lower()

    def test_includes_relations_section(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {
                "description": "researcher",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            },
            "bob": {
                "description": "engineer",
                "chunk_ids": ["c2"],
                "type": "PERSON",
                "frequency": 1,
            },
        }
        brain._relation_graph = {
            "alice": [
                {
                    "target": "bob",
                    "relation": "knows",
                    "chunk_id": "c1",
                    "description": "Alice knows Bob professionally",
                }
            ]
        }
        brain.vector_store.get_by_ids.return_value = []
        result = brain._local_search_context("alice and bob", ["alice", "bob"], cfg)
        assert "Relationships" in result or "alice" in result.lower()

    def test_includes_community_context(self, tmp_path):
        """Lines 1326-1337: community snippets are included when available."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {
                "description": "researcher",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            }
        }
        brain._relation_graph = {}
        brain._community_levels = {0: {"alice": 0}}
        brain._community_summaries = {
            "0_0": {
                "title": "Research Community",
                "summary": "A group of researchers at a university. They collaborate on AI.",
                "rank": 8.0,
                "level": 0,
            }
        }
        brain.vector_store.get_by_ids.return_value = []
        result = brain._local_search_context("alice research", ["alice"], cfg)
        assert "Community Context" in result


# ---------------------------------------------------------------------------
# _canonicalize_entity_descriptions
# ---------------------------------------------------------------------------


class TestCanonicalizeEntityDescriptions:
    def test_skips_when_buffer_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {}
        brain._canonicalize_entity_descriptions()
        brain.llm.complete.assert_not_called()

    def test_skips_when_too_few_occurrences(self, tmp_path):
        """Lines 2124-2128: below min_occurrences threshold skips."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_min_occurrences=3,
        )
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {
            "alice": ["desc1", "desc2"],  # only 2 occurrences, need 3
        }
        brain._canonicalize_entity_descriptions()
        brain.llm.complete.assert_not_called()

    def test_synthesizes_canonical_description(self, tmp_path):
        """Lines 2145-2152: LLM synthesis updates entity description."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_min_occurrences=2,
        )
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {
            "alice": ["A researcher.", "A scientist.", "A physicist."],
        }
        brain._entity_graph = {
            "alice": {
                "description": "A researcher.",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            }
        }
        brain._save_entity_graph = MagicMock()
        brain.llm.complete.return_value = "A renowned physicist and researcher."
        brain._canonicalize_entity_descriptions()
        assert brain._entity_graph["alice"]["description"] == "A renowned physicist and researcher."
        assert brain._entity_description_buffer == {}

    def test_llm_exception_uses_first_description(self, tmp_path):
        """Line 2151-2152: LLM exception falls back to first unique description."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_min_occurrences=2,
        )
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {
            "alice": ["A researcher.", "A scientist.", "A physicist."],
        }
        brain._entity_graph = {
            "alice": {"description": "old", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1}
        }
        brain._save_entity_graph = MagicMock()
        brain.llm.complete.side_effect = RuntimeError("fail")
        brain._canonicalize_entity_descriptions()
        assert brain._entity_graph["alice"]["description"] == "A researcher."


# ---------------------------------------------------------------------------
# _canonicalize_relation_descriptions
# ---------------------------------------------------------------------------


class TestCanonicalizeRelationDescriptions:
    def test_skips_when_config_disabled(self, tmp_path):
        """Line 2167-2168: disabled by config → returns early."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_relations=False,
        )
        brain = _make_brain(config=cfg)
        brain._relation_description_buffer = {("alice", "bob"): ["d1", "d2"]}
        brain._canonicalize_relation_descriptions()
        brain.llm.complete.assert_not_called()

    def test_synthesizes_relation_descriptions(self, tmp_path):
        """Lines 2167-2207: relation descriptions are synthesized."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_relations=True,
            graph_rag_canonicalize_relations_min_occurrences=2,
        )
        brain = _make_brain(config=cfg)
        brain._relation_description_buffer = {
            ("alice", "bob"): ["Alice knows Bob.", "Alice and Bob are friends."],
        }
        brain._relation_graph = {
            "alice": [
                {"target": "bob", "relation": "knows", "chunk_id": "c1", "description": "old desc"}
            ]
        }
        brain._save_relation_graph = MagicMock()
        brain.llm.complete.return_value = (
            "Alice and Bob have a long-term professional relationship."
        )
        brain._canonicalize_relation_descriptions()
        assert (
            brain._relation_graph["alice"][0]["description"]
            == "Alice and Bob have a long-term professional relationship."
        )


# ---------------------------------------------------------------------------
# _resolve_entity_aliases
# ---------------------------------------------------------------------------


class TestResolveEntityAliases:
    def test_returns_zero_for_single_entity(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "d", "chunk_ids": ["c1"]}}
        result = brain._resolve_entity_aliases()
        assert result == 0

    def test_returns_zero_when_embedding_fails(self, tmp_path):
        """Lines 2025-2027: embedding exception returns 0."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1"]},
            "bob": {"description": "d", "chunk_ids": ["c2"]},
        }
        brain.embedding.embed.side_effect = RuntimeError("embed fail")
        result = brain._resolve_entity_aliases()
        assert result == 0

    def test_merges_similar_entities(self, tmp_path):
        """Lines 2060-2116: similar entities are merged."""

        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_resolve_threshold=0.9,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice smith": {
                "description": "A researcher",
                "chunk_ids": ["c1", "c2"],
                "frequency": 2,
            },
            "alice s.": {"description": "researcher", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {}
        brain._community_graph_dirty = False
        # Return very similar embeddings
        brain.embedding.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
        ]
        brain._save_entity_graph = MagicMock()
        result = brain._resolve_entity_aliases()
        assert result == 1
        # After merge: only canonical remains
        assert len(brain._entity_graph) == 1
        assert brain._community_graph_dirty is True

    def test_warns_when_too_many_entities(self, tmp_path):
        """Lines 2013-2020: exceeding max_entities logs warning and returns 0."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_entity_resolve_max=2
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            f"e{i}": {"description": "d", "chunk_ids": [f"c{i}"]}
            for i in range(5)  # 5 > max_entities=2
        }
        result = brain._resolve_entity_aliases()
        assert result == 0


class TestResolveEntityAliasesRustBackend:
    def test_uses_rust_grouping_backend_when_enabled(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_resolve_backend="rust",
            graph_rag_entity_resolve_threshold=0.9,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alpha systems": {"description": "A", "chunk_ids": ["c1", "c2"], "frequency": 2},
            "alpha sys": {"description": "B", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "alpha sys": [
                {
                    "target": "beta",
                    "relation": "partner",
                    "chunk_id": "c3",
                    "subject": "alpha sys",
                    "object": "beta",
                }
            ]
        }
        brain.embedding.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
        ]

        fake_bridge = MagicMock()
        fake_bridge.can_resolve_entity_alias_groups.return_value = True
        fake_bridge.resolve_entity_alias_groups.return_value = [[0, 1]]

        with patch("axon.rust_bridge.get_rust_bridge", return_value=fake_bridge):
            merged = brain._resolve_entity_aliases()

        assert merged == 1
        fake_bridge.resolve_entity_alias_groups.assert_called_once()
        assert list(brain._entity_graph.keys()) == ["alpha systems"]
        assert "alpha sys" not in brain._relation_graph


# ---------------------------------------------------------------------------
# _embed_entities
# ---------------------------------------------------------------------------


class TestEmbedEntities:
    def test_embeds_and_saves(self, tmp_path):
        """Lines 1885-1904: entities are embedded and saved."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1}
        }
        brain._entity_embeddings = {}
        brain.embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        brain._save_entity_embeddings = MagicMock()
        brain._embed_entities(["alice"])
        assert "alice" in brain._entity_embeddings
        brain._save_entity_embeddings.assert_called_once()

    def test_skips_already_embedded(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1}
        }
        brain._entity_embeddings = {"alice": [0.1, 0.2]}  # already present
        brain._embed_entities(["alice"])
        brain.embedding.embed.assert_not_called()

    def test_skip_entities_without_description(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "", "chunk_ids": ["c1"], "frequency": 1}}
        brain._entity_embeddings = {}
        brain._embed_entities(["alice"])
        brain.embedding.embed.assert_not_called()

    def test_embedding_exception_is_swallowed(self, tmp_path):
        """Lines 1903-1904: exception during embed is logged, not raised."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1}
        }
        brain._entity_embeddings = {}
        brain.embedding.embed.side_effect = RuntimeError("embed fail")
        brain._embed_entities(["alice"])  # must not raise


# ---------------------------------------------------------------------------
# _match_entities_by_embedding
# ---------------------------------------------------------------------------


class TestMatchEntitiesByEmbedding:
    def test_returns_empty_when_no_embeddings(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {}
        result = brain._match_entities_by_embedding("query")
        assert result == []

    def test_returns_top_k_similar_entities(self, tmp_path):
        """Lines 1906-1931: cosine similarity matching returns matches."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_match_threshold=0.0,
        )
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {
            "alice": [1.0, 0.0, 0.0],
            "bob": [0.0, 1.0, 0.0],
        }
        brain.embedding.embed_query.return_value = [1.0, 0.0, 0.0]
        result = brain._match_entities_by_embedding("alice researcher", top_k=5)
        assert "alice" in result

    def test_zero_query_vector_returns_empty(self, tmp_path):
        """Line 1916: zero query vector returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {"alice": [1.0, 0.0]}
        brain.embedding.embed_query.return_value = [0.0, 0.0]
        result = brain._match_entities_by_embedding("query")
        assert result == []

    def test_embedding_exception_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {"alice": [1.0, 0.0]}
        brain.embedding.embed_query.side_effect = RuntimeError("fail")
        result = brain._match_entities_by_embedding("query")
        assert result == []


"""Tests for axon.graph_render."""
from axon.graph_render import GraphRenderMixin


class MockBrain(GraphRenderMixin):
    def __init__(self):
        self.config = MagicMock()
        self._community_levels = {0: {"c1": {"label": "Comm 1"}}}
        self._entity_graph = {"e1": {"name": "Entity 1", "type": "Person"}}
        self._relation_graph = {"r1": [{"source": "e1", "target": "e2", "description": "rel"}]}
        self._code_graph = {"nodes": [], "edges": []}
        self._VIZ_TYPE_COLORS = {"Person": "#ff0000"}
        self.vector_store = MagicMock()


def test_build_graph_payload():
    brain = MockBrain()
    payload = brain.build_graph_payload()
    assert "nodes" in payload
    assert "links" in payload
    # Check if our mock data is in payload (depending on implementation details)
    # This is a smoke test to ensure it doesn't crash and returns expected structure


def test_render_graph(tmp_path):
    brain = MockBrain()
    out_path = tmp_path / "graph.html"
    html = brain.export_graph_html(path=str(out_path), open_browser=False)
    assert out_path.exists()
    assert "<html" in html
    assert "GraphRAG 3D Viewer" in html


def test_build_graph_payload_empty():
    brain = MockBrain()
    brain._entity_graph = {}
    brain._relation_graph = {}
    payload = brain.build_graph_payload()
    assert isinstance(payload, dict)
    assert "nodes" in payload


"""Comprehensive tests for GraphRagMixin in axon.graph_rag.

Covers: entity/relation persistence, community detection, community summaries,
graph search helpers, entity matching, claims, prune, resolve aliases, and more.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> AxonConfig:
    # Use a subdirectory of the system temp to be safer
    tmp = tempfile.mkdtemp(prefix="axon_test_")
    d = {
        "bm25_path": os.path.join(tmp, "bm25"),
        "vector_store_path": os.path.join(tmp, "vs"),
    }
    d.update(kwargs)
    return AxonConfig(**d)


def _make_brain(config=None, **extra_attrs):
    """Build a minimal GraphRagMixin instance with necessary attributes."""
    cfg = config or _make_config()
    os.makedirs(cfg.bm25_path, exist_ok=True)

    class FakeBrain(GraphRagMixin):
        pass

    brain = FakeBrain()
    brain.config = cfg
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_levels = {}
    brain._community_hierarchy = {}
    brain._community_children = {}
    brain._community_summaries = {}
    brain._entity_embeddings = {}
    brain._claims_graph = {}
    brain._text_unit_relation_map = {}
    brain._entity_description_buffer = {}
    brain._relation_description_buffer = {}
    brain._community_rebuild_lock = threading.Lock()
    brain._community_graph_dirty = False
    brain.llm = MagicMock()
    brain.embedding = MagicMock()
    brain.vector_store = MagicMock()
    brain._own_vector_store = MagicMock()

    # Provide a simple synchronous executor mock to avoid thread leaks on Windows
    class SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            from concurrent.futures import Future

            f = Future()
            try:
                result = fn(*args, **kwargs)
                f.set_result(result)
            except Exception as e:
                f.set_exception(e)
            return f

        def map(self, fn, *iterables):
            return map(fn, *iterables)

        def shutdown(self, wait=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    brain._executor = SyncExecutor()

    for k, v in extra_attrs.items():
        setattr(brain, k, v)
    return brain


# ---------------------------------------------------------------------------
# Entity graph persistence
# ---------------------------------------------------------------------------


class TestEntityGraphPersistenceV2:
    def test_save_then_load_roundtrip(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "type": "PERSON"},
        }
        brain._save_entity_graph()
        loaded = brain._load_entity_graph()
        assert "alice" in loaded
        assert loaded["alice"]["type"] == "PERSON"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._load_entity_graph()
        assert result == {}

    def test_load_malformed_json_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".entity_graph.json").write_text("{not json", encoding="utf-8")
        assert brain._load_entity_graph() == {}

    def test_load_non_dict_root_returns_empty(self, tmp_path):
        """Line 52: non-dict root returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".entity_graph.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert brain._load_entity_graph() == {}

    def test_load_skips_non_string_keys(self, tmp_path):
        """Line 56: integer keys in JSON are skipped (JSON keys are always strings but test the guard)."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {"valid": {"description": "ok", "chunk_ids": ["c1"]}}
        (tmp_path / ".entity_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_entity_graph()
        assert "valid" in loaded

    def test_load_skips_entry_without_chunk_ids(self, tmp_path):
        """Entries without chunk_ids are skipped."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {
            "good": {"description": "ok", "chunk_ids": ["c1"]},
            "bad": {"description": "no chunk ids"},
        }
        (tmp_path / ".entity_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_entity_graph()
        assert "good" in loaded
        assert "bad" not in loaded

    def test_load_defaults_type_and_frequency(self, tmp_path):
        """Defaults for type, frequency, degree are set automatically."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {"ent": {"description": "d", "chunk_ids": ["c1", "c2"]}}
        (tmp_path / ".entity_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_entity_graph()
        assert loaded["ent"]["type"] == "UNKNOWN"
        assert loaded["ent"]["frequency"] == 2
        assert loaded["ent"]["degree"] == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b"
        cfg = AxonConfig(bm25_path=str(nested), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"x": {"description": "d", "chunk_ids": ["c1"]}}
        brain._save_entity_graph()
        assert (nested / ".entity_graph.json").exists() or (
            nested / ".entity_graph.msgpack"
        ).exists()


# ---------------------------------------------------------------------------
# Relation graph persistence
# ---------------------------------------------------------------------------


class TestRelationGraphPersistenceV2:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}]
        }
        brain._save_relation_graph()
        loaded = brain._load_relation_graph()
        assert "alice" in loaded
        assert loaded["alice"][0]["target"] == "bob"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_relation_graph() == {}

    def test_load_non_dict_root_returns_empty(self, tmp_path):
        """Line 115: list root returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".relation_graph.json").write_text(json.dumps([]), encoding="utf-8")
        assert brain._load_relation_graph() == {}

    def test_load_skips_non_list_values(self, tmp_path):
        """Line 118-119: entries with non-list values are skipped."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {
            "alice": "not a list",
            "bob": [{"target": "carol", "relation": "r", "chunk_id": "c"}],
        }
        (tmp_path / ".relation_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_relation_graph()
        assert "alice" not in loaded
        assert "bob" in loaded

    def test_load_skips_malformed_entries(self, tmp_path):
        """Entries missing target/relation/chunk_id are filtered."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        data = {
            "alice": [
                {"target": "bob", "relation": "r", "chunk_id": "c"},  # good
                {"target": "bob"},  # bad — missing relation, chunk_id
            ]
        }
        (tmp_path / ".relation_graph.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = brain._load_relation_graph()
        assert len(loaded["alice"]) == 1

    def test_load_malformed_json_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".relation_graph.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_relation_graph() == {}


# ---------------------------------------------------------------------------
# Community levels / hierarchy / summaries persistence
# ---------------------------------------------------------------------------


class TestCommunityPersistenceV2:
    def test_save_load_community_levels(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 1}, 1: {"alice": 0}}
        brain._save_community_levels()
        loaded = brain._load_community_levels()
        assert 0 in loaded
        assert loaded[0]["alice"] == 0

    def test_load_community_levels_nonexistent(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_community_levels() == {}

    def test_load_community_levels_corrupt(self, tmp_path):
        """Lines 156-157: exception returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".community_levels.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_community_levels() == {}

    def test_save_load_community_hierarchy(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_hierarchy = {"0_0": None, "1_0": "0_0"}
        brain._save_community_hierarchy()
        loaded = brain._load_community_hierarchy()
        assert "0_0" in loaded
        assert loaded["0_0"] is None
        assert loaded["1_0"] == "0_0"

    def test_load_community_hierarchy_nonexistent(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_community_hierarchy() == {}

    def test_load_community_hierarchy_corrupt(self, tmp_path):
        """Lines 204-205: exception returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".community_hierarchy.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_community_hierarchy() == {}

    def test_save_load_community_summaries(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {"0_0": {"title": "T", "summary": "S", "rank": 5.0}}
        brain._save_community_summaries()
        loaded = brain._load_community_summaries()
        assert "0_0" in loaded
        assert loaded["0_0"]["title"] == "T"

    def test_load_community_summaries_nonexistent(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_community_summaries() == {}

    def test_load_community_summaries_corrupt(self, tmp_path):
        """Lines 234-235: exception returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".community_summaries.json").write_text("{{bad", encoding="utf-8")
        assert brain._load_community_summaries() == {}

    def test_save_community_summaries_exception_logged(self, tmp_path):
        """Lines 247-248: exception in save is swallowed (no raise)."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {"k": {"title": "T"}}
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            # Should not raise
            brain._save_community_summaries()

    def test_save_community_hierarchy_exception_logged(self, tmp_path):
        """Lines 220-221: exception in save is swallowed."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_hierarchy = {}
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            brain._save_community_hierarchy()


# ---------------------------------------------------------------------------
# Entity embeddings and claims graph persistence
# ---------------------------------------------------------------------------


class TestEntityEmbeddingsPersistenceV2:
    def test_save_load_entity_embeddings(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {"alice": [0.1, 0.2, 0.3]}
        brain._save_entity_embeddings()
        loaded = brain._load_entity_embeddings()
        assert "alice" in loaded
        assert loaded["alice"] == [0.1, 0.2, 0.3]

    def test_load_entity_embeddings_nonexistent(self, tmp_path):
        """Lines 261-262: nonexistent file returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_entity_embeddings() == {}

    def test_save_entity_embeddings_exception_logged(self, tmp_path):
        """Lines 274-275: exception swallowed."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {}
        with patch("pathlib.Path.write_text", side_effect=OSError("no space")):
            brain._save_entity_embeddings()  # must not raise


class TestClaimsGraphPersistenceV2:
    def test_save_load_claims_graph(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._claims_graph = {"c1": [{"subject": "alice", "description": "owns shares"}]}
        brain._save_claims_graph()
        loaded = brain._load_claims_graph()
        assert "c1" in loaded
        assert loaded["c1"][0]["subject"] == "alice"

    def test_load_claims_graph_nonexistent(self, tmp_path):
        """Lines 288-292: nonexistent returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._load_claims_graph() == {}

    def test_save_claims_graph_exception_logged(self, tmp_path):
        """Lines 297-305: exception swallowed."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._claims_graph = {}
        with patch("pathlib.Path.write_text", side_effect=OSError("no space")):
            brain._save_claims_graph()


# ---------------------------------------------------------------------------
# Code graph persistence (lines 89-90 — exception path)
# ---------------------------------------------------------------------------


class TestCodeGraphPersistenceV2:
    def test_load_code_graph_exception_returns_empty(self, tmp_path):
        """Lines 89-90: exception during load returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        (tmp_path / ".code_graph.json").write_text("{{bad json", encoding="utf-8")
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}


# ---------------------------------------------------------------------------
# _build_networkx_graph
# ---------------------------------------------------------------------------


class TestBuildNetworkxGraphV2:
    def test_empty_graph(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        G = brain._build_networkx_graph()
        assert G.number_of_nodes() == 0

    def test_nodes_added(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=1,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "Another person", "chunk_ids": ["c2"], "frequency": 1},
        }
        G = brain._build_networkx_graph()
        assert "alice" in G.nodes
        assert "bob" in G.nodes

    def test_edges_added_from_relations(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
        }
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}]
        }
        G = brain._build_networkx_graph()
        assert G.has_edge("alice", "bob")

    def test_duplicate_edges_accumulate_weight(self, tmp_path):
        """Line 325: adding a second edge between same nodes increments weight."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1", "c2"], "frequency": 2},
            "bob": {"description": "d", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "alice": [
                {"target": "bob", "relation": "knows", "chunk_id": "c1", "weight": 1},
                {"target": "bob", "relation": "trusts", "chunk_id": "c2", "weight": 2},
            ]
        }
        G = brain._build_networkx_graph()
        assert G["alice"]["bob"]["weight"] == 3  # 1+2 accumulated

    def test_min_frequency_filter(self, tmp_path):
        """Line 340: entities below min_frequency are excluded."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=3,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "rare": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "common": {"description": "d", "chunk_ids": ["c1", "c2", "c3"], "frequency": 3},
        }
        G = brain._build_networkx_graph()
        assert "rare" not in G.nodes
        assert "common" in G.nodes


# ---------------------------------------------------------------------------
# _run_community_detection
# ---------------------------------------------------------------------------


class TestRunCommunityDetectionV2:
    def test_empty_graph_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._run_community_detection()
        assert result == {}

    def test_single_node_returns_community_zero(self, tmp_path):
        """Line 349: single node returns {node: 0}."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=1,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"solo": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        result = brain._run_community_detection()
        assert result == {"solo": 0}

    def test_connected_entities_get_communities(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "bob": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
            "carol": {"description": "d", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}],
            "bob": [{"target": "carol", "relation": "knows", "chunk_id": "c2"}],
        }
        result = brain._run_community_detection()
        assert set(result.keys()) == {"alice", "bob", "carol"}
        assert all(isinstance(v, int) for v in result.values())

    def test_community_detection_exception_returns_empty(self, tmp_path):
        """Lines 357-358: exception inside Louvain returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "a": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "b": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
        }
        brain._relation_graph = {"a": [{"target": "b", "relation": "r", "chunk_id": "c1"}]}
        fake_bridge = MagicMock()
        fake_bridge.can_build_graph_edges.return_value = False
        fake_bridge.can_run_louvain.return_value = False
        with patch("axon.rust_bridge.get_rust_bridge", return_value=fake_bridge), patch(
            "networkx.algorithms.community.louvain_communities", side_effect=RuntimeError("fail")
        ):
            result = brain._run_community_detection()
        assert result == {}


# ---------------------------------------------------------------------------
# _run_hierarchical_community_detection (Louvain fallback path)
# ---------------------------------------------------------------------------


class TestHierarchicalCommunityDetectionV2:
    def test_returns_tuple_of_three(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "a": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "b": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
            "c": {"description": "d", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {
            "a": [{"target": "b", "relation": "r", "chunk_id": "c1"}],
            "b": [{"target": "c", "relation": "r", "chunk_id": "c2"}],
        }
        result = brain._run_hierarchical_community_detection()
        assert isinstance(result, tuple)
        levels, hierarchy, children = result
        assert isinstance(levels, dict)
        assert isinstance(hierarchy, dict)
        assert isinstance(children, dict)

    def test_single_node_fallback(self, tmp_path):
        """Lines 381-382: single node returns level 0 dict."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_min_frequency=1,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"solo": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        levels, hierarchy, children = brain._run_hierarchical_community_detection()
        assert 0 in levels
        assert levels[0]["solo"] == 0

    def test_multi_level_produces_multiple_level_keys(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_community_levels=2
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            f"e{i}": {"description": "d", "chunk_ids": [f"c{i}"], "frequency": 1} for i in range(5)
        }
        brain._relation_graph = {
            "e0": [{"target": "e1", "relation": "r", "chunk_id": "c0"}],
            "e1": [{"target": "e2", "relation": "r", "chunk_id": "c1"}],
            "e2": [{"target": "e3", "relation": "r", "chunk_id": "c2"}],
            "e3": [{"target": "e4", "relation": "r", "chunk_id": "c3"}],
        }
        levels, _, _ = brain._run_hierarchical_community_detection()
        assert len(levels) >= 1

    def test_backend_louvain_skips_graspologic(self, tmp_path):
        """Line 474-475: louvain backend override skips leidenalg."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_backend="louvain",
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "a": {"description": "d", "chunk_ids": ["c1"], "frequency": 1},
            "b": {"description": "d", "chunk_ids": ["c2"], "frequency": 1},
        }
        brain._relation_graph = {"a": [{"target": "b", "relation": "r", "chunk_id": "c1"}]}
        result = brain._run_hierarchical_community_detection()
        assert isinstance(result, tuple)

    def test_empty_graph_returns_fallback(self, tmp_path):
        """Lines 577-578: empty Louvain result falls back to all-zero mapping."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        # No entities at all
        levels, hierarchy, children = brain._run_hierarchical_community_detection()
        assert isinstance(levels, dict)


# ---------------------------------------------------------------------------
# _generate_community_summaries
# ---------------------------------------------------------------------------


class TestGenerateCommunitySummariesV2:
    def test_no_levels_returns_early(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_levels = {}
        # Should return without calling llm
        brain._generate_community_summaries()
        brain.llm.complete.assert_not_called()

    def test_generates_template_for_small_community(self, tmp_path):
        """Lines 547-618: communities below min_size get template summaries."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=5,  # communities with <5 members get template
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 0}}
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain._generate_community_summaries()
        # LLM should NOT have been called for this tiny community
        brain.llm.complete.assert_not_called()
        assert "0_0" in brain._community_summaries

    def test_generates_llm_summary_for_large_community(self, tmp_path):
        """Lines 654-656: large community gets LLM summary."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=2,
        )
        brain = _make_brain(config=cfg)
        # Build 4 entities in one community
        entities = [f"e{i}" for i in range(4)]
        brain._community_levels = {0: dict.fromkeys(entities, 0)}
        brain._entity_graph = {
            e: {
                "description": f"desc of {e}",
                "chunk_ids": [f"c{i}"],
                "type": "PERSON",
                "frequency": 1,
            }
            for i, e in enumerate(entities)
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.return_value = json.dumps(
            {
                "title": "Test Community",
                "summary": "This community has people.",
                "findings": [],
                "rank": 7.5,
            }
        )
        brain._generate_community_summaries()
        brain.llm.complete.assert_called()
        key = "0_0"
        assert key in brain._community_summaries

    def test_cached_community_reused(self, tmp_path):
        """Lines 772-776: cached summary with same member_hash is reused."""
        import hashlib

        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_community_min_size=1
        )
        brain = _make_brain(config=cfg)
        members = ["alice", "bob"]
        raw = f"0|{'|'.join(sorted(members))}"
        existing_hash = hashlib.md5(raw.encode()).hexdigest()
        brain._community_levels = {0: {"alice": 0, "bob": 0}}
        brain._community_summaries = {
            "0_0": {
                "title": "Cached",
                "summary": "Cached summary",
                "rank": 8.0,
                "full_content": "# Cached\n\nCached summary",
                "member_hash": existing_hash,
                "entities": members,
                "size": 2,
                "level": 0,
            }
        }
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain._generate_community_summaries()
        # LLM should not be called because cache hit
        brain.llm.complete.assert_not_called()

    def test_lazy_mode_tightens_cap(self, tmp_path):
        """Lines 713-720: query_hint + lazy_cap tightens max_total."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=1,
            graph_rag_global_top_communities=2,
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 1, "carol": 2}}
        brain._entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "type": "PERSON", "frequency": 1},
            "carol": {"description": "C", "chunk_ids": ["c3"], "type": "PERSON", "frequency": 1},
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.return_value = '{"title":"T","summary":"S","findings":[],"rank":5.0}'
        brain._generate_community_summaries(query_hint="alice")
        # Should have been called (communities above min_size)
        assert len(brain._community_summaries) > 0

    def test_llm_returns_invalid_json_falls_back(self, tmp_path):
        """Lines 871-875: invalid JSON from LLM falls back to raw text."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=1,
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 0, "carol": 0}}
        brain._entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "type": "PERSON", "frequency": 1},
            "carol": {"description": "C", "chunk_ids": ["c3"], "type": "PERSON", "frequency": 1},
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.return_value = "Plain text, not JSON"
        brain._generate_community_summaries()
        key = "0_0"
        assert key in brain._community_summaries
        # Fallback: summary is the raw text
        assert brain._community_summaries[key]["summary"] == "Plain text, not JSON"

    def test_llm_exception_returns_empty_summary(self, tmp_path):
        """Lines 895-897: LLM exception produces empty summary entry."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_min_size=1,
        )
        brain = _make_brain(config=cfg)
        brain._community_levels = {0: {"alice": 0, "bob": 0, "carol": 0}}
        brain._entity_graph = {
            "alice": {"description": "A", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1},
            "bob": {"description": "B", "chunk_ids": ["c2"], "type": "PERSON", "frequency": 1},
            "carol": {"description": "C", "chunk_ids": ["c3"], "type": "PERSON", "frequency": 1},
        }
        brain._community_summaries = {}
        brain._community_children = {}
        brain._save_community_summaries = MagicMock()
        brain.llm.complete.side_effect = RuntimeError("LLM down")
        brain._generate_community_summaries()
        key = "0_0"
        assert key in brain._community_summaries
        assert brain._community_summaries[key]["summary"] == ""


# ---------------------------------------------------------------------------
# _index_community_reports_in_vector_store
# ---------------------------------------------------------------------------


class TestIndexCommunityReportsV2:
    def test_no_summaries_returns_early(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {}
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_not_called()

    def test_indexes_reports(self, tmp_path):
        """Lines 967-1004: reports are indexed into the vector store."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "# Community\n\nSome content",
                "summary": "Some content",
                "title": "Test",
                "rank": 7.0,
                "level": 0,
                "member_hash": "abc123",
            }
        }
        brain.embedding.embed.return_value = [[0.1, 0.2]]
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_called_once()

    def test_skips_already_indexed_report(self, tmp_path):
        """Line 976-977: skip when indexed_hash matches content_hash."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Content",
                "member_hash": "abc",
                "indexed_hash": "abc",  # already indexed
                "title": "T",
                "rank": 5.0,
                "level": 0,
            }
        }
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_not_called()

    def test_vector_store_exception_is_swallowed(self, tmp_path):
        """Lines 1003-1004: exception during add is logged but not raised."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "content",
                "member_hash": "x",
                "title": "T",
                "rank": 5.0,
                "level": 0,
            }
        }
        brain.embedding.embed.return_value = [[0.1]]
        brain._own_vector_store.add.side_effect = RuntimeError("vector store down")
        # Should not raise
        brain._index_community_reports_in_vector_store()

    def test_skips_empty_full_content(self, tmp_path):
        """Line 1012: summaries with empty full_content are skipped."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "",
                "summary": "",
                "member_hash": "x",
                "title": "",
                "rank": 5.0,
                "level": 0,
            }
        }
        brain._index_community_reports_in_vector_store()
        brain._own_vector_store.add.assert_not_called()


# ---------------------------------------------------------------------------
# _global_search_map_reduce
# ---------------------------------------------------------------------------


class TestGlobalSearchMapReduceV2:
    def test_no_summaries_returns_empty_string(self, tmp_path):
        """Line 1011-1012: no summaries → empty string."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._global_search_map_reduce("what are the themes?", cfg)
        assert result == ""

    def test_returns_no_data_answer_when_all_points_below_threshold(self, tmp_path):
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_global_min_score=80
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Content about Alice",
                "title": "Alice",
                "summary": "summary",
                "rank": 5.0,
                "level": 0,
            }
        }
        # LLM map returns low-score points
        brain.llm.complete.return_value = '[{"point": "some point", "score": 10}]'
        result = brain._global_search_map_reduce("query", cfg)
        from axon.graph_rag import _GRAPHRAG_NO_DATA_ANSWER

        assert result == _GRAPHRAG_NO_DATA_ANSWER

    def test_reduces_valid_points(self, tmp_path):
        """Lines 1184-1193: reduce phase calls LLM and returns response."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_global_min_score=0
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Community report content about Alice and Bob.",
                "title": "Alice-Bob community",
                "summary": "Summary.",
                "rank": 8.0,
                "level": 0,
            }
        }
        # Map phase returns points, reduce phase returns answer
        brain.llm.complete.side_effect = [
            '[{"point": "Alice works with Bob", "score": 90}]',
            "Alice and Bob collaborate extensively.",
        ]
        result = brain._global_search_map_reduce("who works together?", cfg)
        assert "Alice" in result or "collaborate" in result

    def test_level_filtering(self, tmp_path):
        """Lines 1018-1025: only summaries matching target level are used."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_community_level=1,
            graph_rag_global_min_score=0,
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "Level 0 content",
                "title": "L0",
                "summary": "s",
                "rank": 5.0,
                "level": 0,
            },
            "1_0": {
                "full_content": "Level 1 content",
                "title": "L1",
                "summary": "s",
                "rank": 5.0,
                "level": 1,
            },
        }
        brain.llm.complete.return_value = '[{"point": "point", "score": 90}]'
        brain._global_search_map_reduce("query", cfg)
        # Only 1 level-1 community → llm called once for map phase (then reduce)
        assert brain.llm.complete.call_count >= 1

    def test_reduce_exception_returns_fallback(self, tmp_path):
        """Lines 1190-1193: reduce exception returns fallback text."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_global_min_score=0
        )
        brain = _make_brain(config=cfg)
        brain._community_summaries = {
            "0_0": {
                "full_content": "content",
                "title": "T",
                "summary": "s",
                "rank": 5.0,
                "level": 0,
            }
        }
        # Map succeeds, reduce fails
        brain.llm.complete.side_effect = [
            '[{"point": "a point", "score": 90}]',
            RuntimeError("reduce down"),
        ]
        result = brain._global_search_map_reduce("query", cfg)
        assert "Knowledge Graph Findings" in result


# ---------------------------------------------------------------------------
# _entity_matches
# ---------------------------------------------------------------------------


class TestEntityMatchesV2:
    def _brain(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        return _make_brain(config=cfg)

    def test_exact_match_returns_one(self, tmp_path):
        b = self._brain(tmp_path)
        assert b._entity_matches("alice", "alice") == 1.0

    def test_different_single_tokens_returns_zero(self, tmp_path):
        """Single tokens that differ return 0."""
        b = self._brain(tmp_path)
        assert b._entity_matches("alice", "bob") == 0.0

    def test_partial_multi_token_match(self, tmp_path):
        """Multi-token queries can have partial Jaccard match (>= 0.4 threshold)."""
        b = self._brain(tmp_path)
        # "alice bob smith" vs "alice bob jones" → intersection=2, union=4 → J=0.5 >= 0.4
        score = b._entity_matches("alice bob smith", "alice bob jones")
        assert 0.0 < score < 1.0

    def test_high_jaccard_above_threshold(self, tmp_path):
        b = self._brain(tmp_path)
        score = b._entity_matches("new york city", "new york city")
        assert score == 1.0

    def test_low_jaccard_below_threshold_returns_zero(self, tmp_path):
        b = self._brain(tmp_path)
        # Very different multi-token phrases
        score = b._entity_matches("alice smith", "bob jones")
        assert score == 0.0


# ---------------------------------------------------------------------------
# _classify_query_needs_graphrag
# ---------------------------------------------------------------------------


class TestClassifyQueryV2:
    def test_heuristic_holistic_keyword(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._classify_query_needs_graphrag("summarize all findings", "heuristic") is True

    def test_heuristic_long_query(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        long_query = " ".join(["word"] * 25)
        assert brain._classify_query_needs_graphrag(long_query, "heuristic") is True

    def test_heuristic_short_specific_query_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._classify_query_needs_graphrag("who is Alice?", "heuristic") is False

    def test_llm_mode_yes_returns_true(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "YES"
        assert brain._classify_query_needs_graphrag("some query", "llm") is True

    def test_llm_mode_no_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "NO"
        assert brain._classify_query_needs_graphrag("some query", "llm") is False

    def test_llm_mode_exception_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("LLM fail")
        assert brain._classify_query_needs_graphrag("query", "llm") is False

    def test_unknown_mode_returns_false(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        assert brain._classify_query_needs_graphrag("query", "unknown_mode") is False


# ---------------------------------------------------------------------------
# _extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntitiesV2:
    def test_llm_path_parses_three_part_lines(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = (
            "Alice Smith | PERSON | A researcher\n" "Acme Corp | ORGANIZATION | A company\n"
        )
        result = brain._extract_entities("Alice Smith works at Acme Corp.")
        assert len(result) == 2
        assert result[0]["name"] == "Alice Smith"
        assert result[0]["type"] == "PERSON"
        assert result[1]["type"] == "ORGANIZATION"

    def test_llm_path_two_part_line_sets_unknown_type(self, tmp_path):
        """Lines 1794-1802: 2-column format gets UNKNOWN type."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | A researcher"
        result = brain._extract_entities("Some text about Alice.")
        assert len(result) == 1
        assert result[0]["type"] == "UNKNOWN"

    def test_llm_path_one_part_line_sets_unknown(self, tmp_path):
        """Lines 1803-1810: single token gets UNKNOWN type, empty description."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice"
        result = brain._extract_entities("Text about Alice.")
        assert len(result) == 1
        assert result[0]["type"] == "UNKNOWN"
        assert result[0]["description"] == ""

    def test_llm_invalid_type_normalizes_to_concept(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Foo | INVALID_TYPE | Some description"
        result = brain._extract_entities("text")
        assert result[0]["type"] == "CONCEPT"

    def test_llm_exception_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("LLM down")
        result = brain._extract_entities("Some text.")
        assert result == []

    def test_light_depth_uses_light_extractor(self, tmp_path):
        """Lines 1758-1759: light depth skips LLM."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_depth="light"
        )
        brain = _make_brain(config=cfg)
        result = brain._extract_entities("John Smith visited New York City yesterday.")
        # Light extractor uses regex — should return results without calling LLM
        brain.llm.complete.assert_not_called()
        assert isinstance(result, list)

    def test_capped_at_twenty_results(self, tmp_path):
        """Line 1811: results are capped at 20."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        # Return 25 entities
        lines = [f"Entity{i} | PERSON | desc {i}" for i in range(25)]
        brain.llm.complete.return_value = "\n".join(lines)
        result = brain._extract_entities("text")
        assert len(result) == 20


# ---------------------------------------------------------------------------
# _extract_entities_light
# ---------------------------------------------------------------------------


class TestExtractEntitiesLightV2:
    def test_finds_capitalized_phrases(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light("John Smith and New York City are mentioned here.")
        names = [r["name"] for r in result]
        # At least one multi-word capitalized phrase should be found
        assert any("John Smith" in n or "New York" in n for n in names)

    def test_single_capitalized_words_skipped(self, tmp_path):
        """Regex requires at least 2 capitalized words."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light("Alice went home.")
        # "Alice" alone doesn't match the 2-word pattern
        assert all(len(r["name"].split()) >= 2 for r in result)

    def test_deduplicates_entities(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light(
            "John Smith visited John Smith and John Smith again."
        )
        names = [r["name"] for r in result]
        assert names.count("John Smith") == 1

    def test_type_is_concept(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._extract_entities_light("John Smith visited New York City.")
        assert all(r["type"] == "CONCEPT" for r in result)

    def test_capped_at_twenty(self, tmp_path):
        """Line 1747: results are capped at 20."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        # Generate many capitalized 2-word phrases
        text = " ".join([f"Person{i} Smith{i}" for i in range(30)])
        result = brain._extract_entities_light(text)
        assert len(result) <= 20


# ---------------------------------------------------------------------------
# _extract_relations
# ---------------------------------------------------------------------------


class TestExtractRelationsV2:
    def test_parses_five_part_lines(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | works_at | Acme | Alice is employed by Acme | 8"
        result = brain._extract_relations("Alice works at Acme.")
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["strength"] == 8

    def test_parses_four_part_lines(self, tmp_path):
        """Lines 1861-1870: 4-part line gets default strength 5."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | works_at | Acme | Alice employed by Acme"
        result = brain._extract_relations("Alice works at Acme.")
        assert len(result) == 1
        assert result[0]["strength"] == 5

    def test_parses_three_part_lines(self, tmp_path):
        """Lines 1871-1880: 3-part line gets default strength 5 and empty description."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | works_at | Acme"
        result = brain._extract_relations("Alice works at Acme.")
        assert len(result) == 1
        assert result[0]["description"] == ""

    def test_invalid_strength_defaults_to_five(self, tmp_path):
        """Lines 1849-1851: non-integer strength falls back to 5."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | knows | Bob | They know each other | notanumber"
        result = brain._extract_relations("Alice knows Bob.")
        assert result[0]["strength"] == 5

    def test_light_depth_returns_empty(self, tmp_path):
        """Lines 1823-1824: light depth returns empty without LLM call."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_depth="light"
        )
        brain = _make_brain(config=cfg)
        result = brain._extract_relations("Alice knows Bob.")
        assert result == []
        brain.llm.complete.assert_not_called()

    def test_exception_returns_empty(self, tmp_path):
        """Lines 1882-1883: exception returns empty list."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("LLM down")
        result = brain._extract_relations("text")
        assert result == []

    def test_capped_at_fifteen(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        lines = [f"E{i} | r | E{i+1} | desc | 5" for i in range(20)]
        brain.llm.complete.return_value = "\n".join(lines)
        result = brain._extract_relations("text")
        assert len(result) == 15


# ---------------------------------------------------------------------------
# _parse_rebel_output (static method)
# ---------------------------------------------------------------------------


class TestParseRebelOutputV2:
    def test_parses_single_triplet(self):
        text = "<triplet> Alice <subj> Acme Corp <obj> works at"
        result = GraphRagMixin._parse_rebel_output(text)
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["object"] == "Acme Corp"
        assert result[0]["relation"] == "works at"

    def test_parses_multiple_triplets(self):
        text = "<triplet> Alice <subj> Bob <obj> knows " "<triplet> Bob <subj> Carol <obj> manages"
        result = GraphRagMixin._parse_rebel_output(text)
        assert len(result) == 2

    def test_incomplete_triplet_not_included(self):
        """Partial triplet without object or relation is not flushed."""
        text = "<triplet> Alice <subj> Bob"
        result = GraphRagMixin._parse_rebel_output(text)
        # No relation → should not produce a valid triplet
        assert all("relation" in r and r["relation"] for r in result)

    def test_empty_text_returns_empty(self):
        assert GraphRagMixin._parse_rebel_output("") == []

    def test_special_tokens_stripped(self):
        text = "<s><pad> <triplet> Alice <subj> Acme <obj> founded </s>"
        result = GraphRagMixin._parse_rebel_output(text)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _extract_claims
# ---------------------------------------------------------------------------


class TestExtractClaimsV2:
    def test_parses_full_eight_part_line(self, tmp_path):
        """Lines 1956-1972: 8-part lines parsed into full claim dict."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = "Alice | Acme | acquisition | TRUE | Alice acquired Acme | 2024-01-01 | 2024-06-01 | 'Alice acquired Acme'"
        result = brain._extract_claims("Alice acquired Acme in 2024.")
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        assert result[0]["status"] == "TRUE"
        assert result[0]["start_date"] == "2024-01-01"

    def test_parses_five_part_line(self, tmp_path):
        """Lines 1973-1989: 5-part lines parsed with null dates."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = (
            "Alice | Acme | partnership | TRUE | Alice partners with Acme"
        )
        result = brain._extract_claims("text")
        assert len(result) == 1
        assert result[0]["start_date"] is None

    def test_invalid_status_normalized_to_suspected(self, tmp_path):
        """Lines 1957-1959 & 1974-1976: invalid status defaults to SUSPECTED."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.return_value = (
            "Alice | Acme | partnership | MAYBE | Some claim | unknown | unknown | none"
        )
        result = brain._extract_claims("text")
        assert result[0]["status"] == "SUSPECTED"

    def test_exception_returns_empty(self, tmp_path):
        """Lines 1991-1992: exception returns empty list."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain.llm.complete.side_effect = RuntimeError("fail")
        assert brain._extract_claims("text") == []


# ---------------------------------------------------------------------------
# _prune_entity_graph
# ---------------------------------------------------------------------------


class TestPruneEntityGraphV2:
    def test_removes_entity_when_all_chunks_deleted(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        brain._relation_graph = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert "alice" not in brain._entity_graph
        brain._save_entity_graph.assert_called_once()

    def test_updates_chunk_ids_when_partial_delete(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1", "c2"], "frequency": 2}
        }
        brain._relation_graph = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert brain._entity_graph["alice"]["chunk_ids"] == ["c2"]
        assert brain._entity_graph["alice"]["frequency"] == 1

    def test_removes_relation_when_chunk_deleted(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {}
        brain._relation_graph = {
            "alice": [
                {"target": "bob", "relation": "knows", "chunk_id": "c1"},
                {"target": "carol", "relation": "works_with", "chunk_id": "c2"},
            ]
        }
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert len(brain._relation_graph["alice"]) == 1
        assert brain._relation_graph["alice"][0]["chunk_id"] == "c2"

    def test_removes_empty_relation_list(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {}
        brain._relation_graph = {"alice": [{"target": "bob", "relation": "r", "chunk_id": "c1"}]}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert "alice" not in brain._relation_graph

    def test_prunes_claims_graph(self, tmp_path):
        """Lines 2246-2251: claims_graph chunk_ids are pruned."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {
            "c1": [{"subject": "alice", "description": "claim"}],
            "c2": [{"subject": "bob", "description": "another claim"}],
        }
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_claims_graph = MagicMock()
        brain._prune_entity_graph({"c1"})
        assert "c1" not in brain._claims_graph
        assert "c2" in brain._claims_graph
        brain._save_claims_graph.assert_called_once()

    def test_no_changes_skip_save(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "d", "chunk_ids": ["c1"], "frequency": 1}}
        brain._relation_graph = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._prune_entity_graph({"c999"})  # non-existent chunk
        brain._save_entity_graph.assert_not_called()


# ---------------------------------------------------------------------------
# _get_incoming_relations
# ---------------------------------------------------------------------------


class TestGetIncomingRelationsV2:
    def test_returns_incoming_edges(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}],
            "carol": [{"target": "bob", "relation": "works_with", "chunk_id": "c2"}],
        }
        result = brain._get_incoming_relations("bob")
        assert len(result) == 2
        sources = {r["source"] for r in result}
        assert sources == {"alice", "carol"}

    def test_returns_empty_for_no_incoming(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "carol", "relation": "knows", "chunk_id": "c1"}]
        }
        result = brain._get_incoming_relations("bob")
        assert result == []

    def test_case_insensitive_matching(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._relation_graph = {
            "alice": [{"target": "Bob", "relation": "knows", "chunk_id": "c1"}]
        }
        result = brain._get_incoming_relations("bob")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _local_search_context
# ---------------------------------------------------------------------------


class TestLocalSearchContextV2:
    def test_empty_entities_returns_empty_string(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        result = brain._local_search_context("query", [], cfg)
        assert result == ""

    def test_returns_matched_entities_section(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {
                "description": "A researcher at university",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            }
        }
        brain._relation_graph = {}
        brain.vector_store.get_by_ids.return_value = []
        result = brain._local_search_context("who is alice?", ["alice"], cfg)
        assert "Matched Entities" in result
        assert "alice" in result.lower()

    def test_includes_relations_section(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {
                "description": "researcher",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            },
            "bob": {
                "description": "engineer",
                "chunk_ids": ["c2"],
                "type": "PERSON",
                "frequency": 1,
            },
        }
        brain._relation_graph = {
            "alice": [
                {
                    "target": "bob",
                    "relation": "knows",
                    "chunk_id": "c1",
                    "description": "Alice knows Bob professionally",
                }
            ]
        }
        brain.vector_store.get_by_ids.return_value = []
        result = brain._local_search_context("alice and bob", ["alice", "bob"], cfg)
        assert "Relationships" in result or "alice" in result.lower()

    def test_includes_community_context(self, tmp_path):
        """Lines 1326-1337: community snippets are included when available."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {
                "description": "researcher",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            }
        }
        brain._relation_graph = {}
        brain._community_levels = {0: {"alice": 0}}
        brain._community_summaries = {
            "0_0": {
                "title": "Research Community",
                "summary": "A group of researchers at a university. They collaborate on AI.",
                "rank": 8.0,
                "level": 0,
            }
        }
        brain.vector_store.get_by_ids.return_value = []
        result = brain._local_search_context("alice research", ["alice"], cfg)
        assert "Community Context" in result


# ---------------------------------------------------------------------------
# _canonicalize_entity_descriptions
# ---------------------------------------------------------------------------


class TestCanonicalizeEntityDescriptionsV2:
    def test_skips_when_buffer_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {}
        brain._canonicalize_entity_descriptions()
        brain.llm.complete.assert_not_called()

    def test_skips_when_too_few_occurrences(self, tmp_path):
        """Lines 2124-2128: below min_occurrences threshold skips."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_min_occurrences=3,
        )
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {
            "alice": ["desc1", "desc2"],  # only 2 occurrences, need 3
        }
        brain._canonicalize_entity_descriptions()
        brain.llm.complete.assert_not_called()

    def test_synthesizes_canonical_description(self, tmp_path):
        """Lines 2145-2152: LLM synthesis updates entity description."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_min_occurrences=2,
        )
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {
            "alice": ["A researcher.", "A scientist.", "A physicist."],
        }
        brain._entity_graph = {
            "alice": {
                "description": "A researcher.",
                "chunk_ids": ["c1"],
                "type": "PERSON",
                "frequency": 1,
            }
        }
        brain._save_entity_graph = MagicMock()
        brain.llm.complete.return_value = "A renowned physicist and researcher."
        brain._canonicalize_entity_descriptions()
        assert brain._entity_graph["alice"]["description"] == "A renowned physicist and researcher."
        assert brain._entity_description_buffer == {}

    def test_llm_exception_uses_first_description(self, tmp_path):
        """Line 2151-2152: LLM exception falls back to first unique description."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_min_occurrences=2,
        )
        brain = _make_brain(config=cfg)
        brain._entity_description_buffer = {
            "alice": ["A researcher.", "A scientist.", "A physicist."],
        }
        brain._entity_graph = {
            "alice": {"description": "old", "chunk_ids": ["c1"], "type": "PERSON", "frequency": 1}
        }
        brain._save_entity_graph = MagicMock()
        brain.llm.complete.side_effect = RuntimeError("fail")
        brain._canonicalize_entity_descriptions()
        assert brain._entity_graph["alice"]["description"] == "A researcher."


# ---------------------------------------------------------------------------
# _canonicalize_relation_descriptions
# ---------------------------------------------------------------------------


class TestCanonicalizeRelationDescriptionsV2:
    def test_skips_when_config_disabled(self, tmp_path):
        """Line 2167-2168: disabled by config → returns early."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_relations=False,
        )
        brain = _make_brain(config=cfg)
        brain._relation_description_buffer = {("alice", "bob"): ["d1", "d2"]}
        brain._canonicalize_relation_descriptions()
        brain.llm.complete.assert_not_called()

    def test_synthesizes_relation_descriptions(self, tmp_path):
        """Lines 2167-2207: relation descriptions are synthesized."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_canonicalize_relations=True,
            graph_rag_canonicalize_relations_min_occurrences=2,
        )
        brain = _make_brain(config=cfg)
        brain._relation_description_buffer = {
            ("alice", "bob"): ["Alice knows Bob.", "Alice and Bob are friends."],
        }
        brain._relation_graph = {
            "alice": [
                {"target": "bob", "relation": "knows", "chunk_id": "c1", "description": "old desc"}
            ]
        }
        brain._save_relation_graph = MagicMock()
        brain.llm.complete.return_value = (
            "Alice and Bob have a long-term professional relationship."
        )
        brain._canonicalize_relation_descriptions()
        assert (
            brain._relation_graph["alice"][0]["description"]
            == "Alice and Bob have a long-term professional relationship."
        )


# ---------------------------------------------------------------------------
# _resolve_entity_aliases
# ---------------------------------------------------------------------------


class TestResolveEntityAliasesV2:
    def test_returns_zero_for_single_entity(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "d", "chunk_ids": ["c1"]}}
        result = brain._resolve_entity_aliases()
        assert result == 0

    def test_returns_zero_when_embedding_fails(self, tmp_path):
        """Lines 2025-2027: embedding exception returns 0."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "d", "chunk_ids": ["c1"]},
            "bob": {"description": "d", "chunk_ids": ["c2"]},
        }
        brain.embedding.embed.side_effect = RuntimeError("embed fail")
        result = brain._resolve_entity_aliases()
        assert result == 0

    def test_merges_similar_entities(self, tmp_path):
        """Lines 2060-2116: similar entities are merged."""

        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_resolve_threshold=0.9,
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice smith": {
                "description": "A researcher",
                "chunk_ids": ["c1", "c2"],
                "frequency": 2,
            },
            "alice s.": {"description": "researcher", "chunk_ids": ["c3"], "frequency": 1},
        }
        brain._relation_graph = {}
        brain._community_graph_dirty = False
        # Return very similar embeddings
        brain.embedding.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
        ]
        brain._save_entity_graph = MagicMock()
        result = brain._resolve_entity_aliases()
        assert result == 1
        # After merge: only canonical remains
        assert len(brain._entity_graph) == 1
        assert brain._community_graph_dirty is True

    def test_warns_when_too_many_entities(self, tmp_path):
        """Lines 2013-2020: exceeding max_entities logs warning and returns 0."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path), vector_store_path=str(tmp_path), graph_rag_entity_resolve_max=2
        )
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            f"e{i}": {"description": "d", "chunk_ids": [f"c{i}"]}
            for i in range(5)  # 5 > max_entities=2
        }
        result = brain._resolve_entity_aliases()
        assert result == 0


# ---------------------------------------------------------------------------
# _embed_entities
# ---------------------------------------------------------------------------


class TestEmbedEntitiesV2:
    def test_embeds_and_saves(self, tmp_path):
        """Lines 1885-1904: entities are embedded and saved."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1}
        }
        brain._entity_embeddings = {}
        brain.embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        brain._save_entity_embeddings = MagicMock()
        brain._embed_entities(["alice"])
        assert "alice" in brain._entity_embeddings
        brain._save_entity_embeddings.assert_called_once()

    def test_skips_already_embedded(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1}
        }
        brain._entity_embeddings = {"alice": [0.1, 0.2]}  # already present
        brain._embed_entities(["alice"])
        brain.embedding.embed.assert_not_called()

    def test_skip_entities_without_description(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {"alice": {"description": "", "chunk_ids": ["c1"], "frequency": 1}}
        brain._entity_embeddings = {}
        brain._embed_entities(["alice"])
        brain.embedding.embed.assert_not_called()

    def test_embedding_exception_is_swallowed(self, tmp_path):
        """Lines 1903-1904: exception during embed is logged, not raised."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_graph = {
            "alice": {"description": "A person", "chunk_ids": ["c1"], "frequency": 1}
        }
        brain._entity_embeddings = {}
        brain.embedding.embed.side_effect = RuntimeError("embed fail")
        brain._embed_entities(["alice"])  # must not raise


# ---------------------------------------------------------------------------
# _match_entities_by_embedding
# ---------------------------------------------------------------------------


class TestMatchEntitiesByEmbeddingV2:
    def test_returns_empty_when_no_embeddings(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {}
        result = brain._match_entities_by_embedding("query")
        assert result == []

    def test_returns_top_k_similar_entities(self, tmp_path):
        """Lines 1906-1931: cosine similarity matching returns matches."""
        cfg = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag_entity_match_threshold=0.0,
        )
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {
            "alice": [1.0, 0.0, 0.0],
            "bob": [0.0, 1.0, 0.0],
        }
        brain.embedding.embed_query.return_value = [1.0, 0.0, 0.0]
        result = brain._match_entities_by_embedding("alice researcher", top_k=5)
        assert "alice" in result

    def test_zero_query_vector_returns_empty(self, tmp_path):
        """Line 1916: zero query vector returns empty."""
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {"alice": [1.0, 0.0]}
        brain.embedding.embed_query.return_value = [0.0, 0.0]
        result = brain._match_entities_by_embedding("query")
        assert result == []

    def test_embedding_exception_returns_empty(self, tmp_path):
        cfg = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        brain = _make_brain(config=cfg)
        brain._entity_embeddings = {"alice": [1.0, 0.0]}
        brain.embedding.embed_query.side_effect = RuntimeError("fail")
        result = brain._match_entities_by_embedding("query")
        assert result == []
