import json
import pickle

import pytest

from axon.graph_rag import GraphRagMixin


class MockConfig:
    def __init__(self, bm25_path: str):
        self.bm25_path = bm25_path
        # Defaults that match AxonConfig for graph fields the loader inspects.
        self.graph_rag_relation_msgpack_persist = True
        self.graph_rag_relation_pickle_cache = False
        self.graph_rag_relation_pickle_cache_protocol = 4
        self.graph_rag_relation_shard_list_manifest = True
        self.graph_rag_relation_shard_parallel_load = True
        self.graph_rag_relation_shard_load_workers = 4
        self.graph_rag_relation_shard_persist = False


class TestGraphRagIntegrity:
    @pytest.fixture
    def temp_bm25_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def brain_stub(self, temp_bm25_dir):
        # We want to test the actual methods of GraphRagMixin, so we mix it in
        class Brain(GraphRagMixin):
            def __init__(self, config):
                self.config = config
                self._entity_graph = {}

        return Brain(MockConfig(str(temp_bm25_dir)))

    def test_save_load_roundtrip(self, brain_stub, temp_bm25_dir):
        """Verify that a simple entity graph survives save/load with defaults added."""
        graph = {
            "entity1": {"description": "desc1", "chunk_ids": ["c1", "c2"], "type": "PERSON"},
            "entity2": {"description": "desc2", "chunk_ids": ["c3"], "type": "ORG"},
        }
        brain_stub._entity_graph = graph
        brain_stub._save_entity_graph()
        brain_stub._flush_pending_saves()

        # Check file existence (msgpack preferred, json fallback)
        assert (temp_bm25_dir / ".entity_graph.json").exists() or (
            temp_bm25_dir / ".entity_graph.msgpack"
        ).exists()

        # Load into a new brain instance
        new_brain = GraphRagMixin()
        new_brain.config = brain_stub.config
        loaded_graph = new_brain._load_entity_graph()

        expected = {
            "entity1": {
                "description": "desc1",
                "chunk_ids": ["c1", "c2"],
                "type": "PERSON",
                "frequency": 2,
                "degree": 0,
            },
            "entity2": {
                "description": "desc2",
                "chunk_ids": ["c3"],
                "type": "ORG",
                "frequency": 1,
                "degree": 0,
            },
        }
        assert loaded_graph == expected

    def test_load_malformed_json_returns_empty(self, brain_stub, temp_bm25_dir):
        """Malformed JSON must not raise and return empty graph."""
        path = temp_bm25_dir / ".entity_graph.json"
        path.write_text("{ this is not json }", encoding="utf-8")

        loaded = brain_stub._load_entity_graph()
        assert loaded == {}

    def test_load_nonexistent_returns_empty(self, brain_stub):
        """Non-existent file returns empty graph."""
        loaded = brain_stub._load_entity_graph()
        assert loaded == {}

    def test_load_corrupt_entries_skipped(self, brain_stub, temp_bm25_dir):
        """Entries missing 'chunk_ids' should be skipped by _load_entity_graph."""
        graph = {
            "good": {"description": "ok", "chunk_ids": ["c1"]},
            "bad": {"description": "missing chunk_ids"},
        }
        path = temp_bm25_dir / ".entity_graph.json"
        path.write_text(json.dumps(graph), encoding="utf-8")

        loaded = brain_stub._load_entity_graph()
        assert "good" in loaded
        assert "bad" not in loaded

    def test_large_graph_persistence(self, brain_stub, temp_bm25_dir):
        """Stress test serialization/deserialization with 10k entities."""
        large_graph = {
            f"entity_{i}": {
                "description": f"Extremely long description for entity {i} to increase payload size. "
                * 5,
                "chunk_ids": [f"c_{i}_{j}" for j in range(5)],
                "type": "TEST",
                "frequency": i,
                "degree": i % 10,
            }
            for i in range(10000)
        }
        brain_stub._entity_graph = large_graph
        brain_stub._save_entity_graph()

        loaded = brain_stub._load_entity_graph()
        assert len(loaded) == 10000
        assert loaded["entity_9999"]["frequency"] == 9999


class TestRelationPickleCacheHmac:
    """Audit P0: relation-graph pickle cache must verify HMAC before unpickling.

    pickle.load on attacker-controlled bytes is RCE. When bm25_path is a
    shared / cloud-synced directory, anyone with write access can drop a
    malicious pickle there. The HMAC binds the cache to this host via a
    key under ~/.axon/.
    """

    @pytest.fixture
    def brain_stub(self, tmp_path, monkeypatch):
        # Pin Path.home() so the HMAC key lands in tmp_path and the test
        # doesn't pollute the developer's real ~/.axon.
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")

        class Brain(GraphRagMixin):
            def __init__(self, config):
                self.config = config
                self._entity_graph = {}
                self._relation_graph = {}

        cfg = MockConfig(str(tmp_path))
        cfg.graph_rag_relation_pickle_cache = True
        cfg.graph_rag_relation_msgpack_persist = False
        return Brain(cfg)

    def _seed_shards_with_cache(self, brain, tmp_path, payload):
        """Write the minimum file set that triggers the pickle-cache fast-path."""
        import hashlib

        # Shards manifest + a single shard so _load_relation_graph reaches the
        # cache-checking branch.
        shard_payload = {"format": "rg_rel_v2", "g": payload}
        shard_path = tmp_path / ".relation_graph.shard.000.json"
        shard_path.write_text(json.dumps(shard_payload), encoding="utf-8")
        (tmp_path / ".relation_graph.shards.json").write_text(
            json.dumps({"format": "rg_rel_shard_v1", "shards": [".relation_graph.shard.000.json"]}),
            encoding="utf-8",
        )
        # Shard state file with a deterministic signature so cache_key is computable.
        sigs = ["deadbeef"]
        (tmp_path / ".relation_graph.shard_state.json").write_text(
            json.dumps({"format": "rg_rel_shard_state_v1", "signatures": sigs}),
            encoding="utf-8",
        )
        # Compute cache_key the same way _load_relation_graph does.
        return hashlib.sha1("|".join(sigs).encode("utf-8")).hexdigest()

    def test_unsigned_pickle_cache_is_refused(self, brain_stub, tmp_path):
        """A pickle file without a valid HMAC must NOT be unpickled.

        Without this guard, an attacker who can write to bm25_path could
        place a malicious pickle and achieve RCE on the next load.
        """
        cache_key = self._seed_shards_with_cache(
            brain_stub,
            tmp_path,
            {"alice": [{"target": "shard_value", "relation": "knows", "chunk_id": "c1"}]},
        )
        # Plant an "evil" pickle bytes — but for safety in the test we just
        # use a benign dict; the load path must not call pickle.load on it
        # because the meta file has no HMAC.
        evil = {"poisoned": [{"target": "rce", "relation": "rce", "chunk_id": "rce"}]}
        (tmp_path / ".relation_graph.cache.pkl").write_bytes(pickle.dumps(evil))
        # Meta without the "hmac" field — older format.
        (tmp_path / ".relation_graph.cache.meta.json").write_text(
            json.dumps({"key": cache_key}), encoding="utf-8"
        )
        loaded = brain_stub._load_relation_graph()
        # The pickle MUST be rejected; we should fall through to the JSON
        # shard load path and see the shard contents, not the planted dict.
        assert "poisoned" not in loaded
        assert "alice" in loaded

    def test_tampered_pickle_cache_is_refused(self, brain_stub, tmp_path):
        """A pickle file with an HMAC that doesn't match the bytes is refused."""
        from axon.graph_rag import _compute_relation_pickle_hmac

        cache_key = self._seed_shards_with_cache(
            brain_stub,
            tmp_path,
            {"bob": [{"target": "shard_value", "relation": "knows", "chunk_id": "c2"}]},
        )
        # Compute a valid HMAC for one payload, then write a different payload.
        good_payload = pickle.dumps({"legit": [{"target": "x", "relation": "y", "chunk_id": "z"}]})
        good_mac = _compute_relation_pickle_hmac(good_payload, cache_key)
        evil_payload = pickle.dumps(
            {"tampered": [{"target": "rce", "relation": "rce", "chunk_id": "rce"}]}
        )
        (tmp_path / ".relation_graph.cache.pkl").write_bytes(evil_payload)
        (tmp_path / ".relation_graph.cache.meta.json").write_text(
            json.dumps({"key": cache_key, "hmac": good_mac}), encoding="utf-8"
        )
        loaded = brain_stub._load_relation_graph()
        assert "tampered" not in loaded
        assert "bob" in loaded

    def test_signed_pickle_cache_is_loaded(self, brain_stub, tmp_path):
        """The happy path: a pickle written by ``_load_relation_graph`` itself
        carries a valid HMAC and is loaded on the next call."""
        # Seed shards, run a first load — that writes the pickle cache.
        self._seed_shards_with_cache(
            brain_stub,
            tmp_path,
            {"carol": [{"target": "shard_value", "relation": "knows", "chunk_id": "c3"}]},
        )
        first = brain_stub._load_relation_graph()
        assert "carol" in first
        # The pickle and meta files should now exist with a valid HMAC.
        assert (tmp_path / ".relation_graph.cache.pkl").exists()
        meta = json.loads(
            (tmp_path / ".relation_graph.cache.meta.json").read_text(encoding="utf-8")
        )
        assert "hmac" in meta and isinstance(meta["hmac"], str) and len(meta["hmac"]) == 64
        # Delete the shard so the JSON fallback would return empty —
        # this proves the second load came from the pickle cache.
        (tmp_path / ".relation_graph.shard.000.json").write_text(
            json.dumps({"format": "rg_rel_v2", "g": {}}), encoding="utf-8"
        )
        # Clear shard-names cache between loads
        if hasattr(brain_stub, "_rel_shard_names_sig"):
            delattr(brain_stub, "_rel_shard_names_sig")
        second = brain_stub._load_relation_graph()
        assert "carol" in second

    def test_hmac_key_file_is_persistent(self, brain_stub, tmp_path):
        """The HMAC key is created once and reused."""
        from axon.graph_rag import _get_or_create_relation_pickle_hmac_key

        key1 = _get_or_create_relation_pickle_hmac_key()
        key2 = _get_or_create_relation_pickle_hmac_key()
        assert key1 == key2
        assert len(key1) >= 32
