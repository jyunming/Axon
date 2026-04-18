import threading
import time
from unittest.mock import MagicMock

import pytest

from axon.config import AxonConfig
from axon.main import AxonBrain


def test_graph_concurrency_stress(tmp_path):
    """Stress test: simultaneous ingest and query should not crash with RuntimeError."""
    cfg = AxonConfig()
    cfg.bm25_path = str(tmp_path)
    cfg.vector_store_path = str(tmp_path / "vs")
    cfg.graph_rag = True

    brain = AxonBrain(cfg)
    # Mock LLM and Embedding to avoid network calls
    brain.llm = MagicMock()
    brain.llm.complete.return_value = "Extracted | Entity | Description"
    brain.embedding = MagicMock()
    brain.embedding.embed_query.return_value = [0.1] * 1536
    brain.embedding.embed.return_value = [[0.1] * 1536]

    stop_event = threading.Event()
    errors = []

    def run_ingest():
        try:
            while not stop_event.is_set():
                doc = {"id": f"doc_{time.time()}", "text": "Some text about OpenAI and LLMs."}
                # Use private method to avoid full ingest overhead if possible,
                # but we want to test the hardened block in ingest().
                brain.ingest([doc])
                time.sleep(0.01)
        except Exception as e:
            errors.append(f"Ingest error: {e}")

    def run_query():
        try:
            while not stop_event.is_set():
                # This triggers _expand_with_entity_graph which iterates over the graph
                brain.query("What is OpenAI?")
                time.sleep(0.01)
        except Exception as e:
            errors.append(f"Query error: {e}")

    threads = [
        threading.Thread(target=run_ingest),
        threading.Thread(target=run_query),
        threading.Thread(target=run_query),
        threading.Thread(target=run_query),
    ]

    for t in threads:
        t.start()

    time.sleep(2)  # Run for 2 seconds
    stop_event.set()

    for t in threads:
        t.join()

    assert not errors, f"Concurrency errors detected: {errors}"


def test_dynamic_graph_chunking(tmp_path):
    """Verify that multi-hop BFS handles large fringes by chunking queries."""
    from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend

    brain = MagicMock()
    brain.config.bm25_path = str(tmp_path)
    backend = DynamicGraphBackend(brain)

    # Ingest many facts directly via _upsert_fact
    now = "2026-04-17T12:00:00Z"
    for i in range(1500):
        backend._upsert_fact("seed", "linked_to", f"node_{i}", "desc", 1.0, "chunk_0", "ep_0", now)

    # Second hop facts
    for i in range(5):  # just a few to verify reachability
        backend._upsert_fact(
            "node_0", "linked_to", f"hop2_{i}", "desc", 1.0, "chunk_0", "ep_0", now
        )

    # Trigger retrieval with max_hops=2
    # We must set the environment variables because the backend uses them as fallback
    from axon.graph_backends.base import RetrievalConfig

    cfg = RetrievalConfig()
    cfg.top_k = 2000

    with pytest.MonkeyPatch().context() as m:
        m.setenv("AXON_GRAPH_RAG_MAX_HOPS", "2")
        # Querying "seed" should match facts (seed, linked_to, node_i) at hop 0
        # node_0 is a target, so facts containing node_0 will be reached at hop 1.
        results = backend.retrieve("seed", cfg=cfg)

    assert len(results) >= 1500
    # facts containing node_0 (reached via 1 hop from seed) should have hop_count == 1
    assert any(ctx.hop_count == 1 for ctx in results)
