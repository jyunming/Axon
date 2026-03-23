import json

import pytest

from axon.graph_rag import GraphRagMixin


class MockConfig:
    def __init__(self, bm25_path: str):
        self.bm25_path = bm25_path


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

        # Check file existence
        path = temp_bm25_dir / ".entity_graph.json"
        assert path.exists()

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
