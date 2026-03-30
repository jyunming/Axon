"""Stress and Performance tests for Axon."""
import os
import time
import uuid
import pytest
from unittest.mock import MagicMock, patch
from axon.main import AxonBrain, AxonConfig


@pytest.fixture
def brain_with_chroma(tmp_path):
    config = AxonConfig(
        vector_store="chroma",
        vector_store_path=str(tmp_path / "vs"),
        bm25_path=str(tmp_path / "bm25"),
        projects_root=str(tmp_path / "projects"),
        llm_provider="openai",  # Use openai to avoid ollama dependency in tests
        openai_api_key="fake-key",
    )
    # Mock LLM to avoid external calls
    with patch("axon.llm.OpenLLM") as mock_llm_cls:
        mock_llm = mock_llm_cls.return_value
        mock_llm.complete.return_value = "Mocked LLM Response"
        mock_llm.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
        brain = AxonBrain(config)
        # Ensure brain.llm is our mock
        brain.llm = mock_llm
        yield brain


def test_stress_high_volume_ingest(brain_with_chroma):
    """Stress test: ingest a large number of documents."""
    num_docs = 1000
    docs = [
        {
            "id": f"s_{i}",
            "text": f"This is document number {i} for stress testing.",
            "metadata": {"source": f"stress_{i}.txt"},
        }
        for i in range(num_docs)
    ]

    start_time = time.time()
    brain_with_chroma.ingest(docs)
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nIngested {num_docs} documents in {duration:.2f} seconds.")
    assert len(brain_with_chroma._ingested_hashes) >= num_docs


def test_perf_query_latency(brain_with_chroma):
    """Performance test: measure latency of a standard query."""
    # First ingest some data
    docs = [
        {"id": f"p_{i}", "text": f"Sample data {i}", "metadata": {"source": f"perf_{i}.txt"}}
        for i in range(100)
    ]
    brain_with_chroma.ingest(docs)

    query = "What is sample data 50?"

    # Warm up
    brain_with_chroma.query(query)

    latencies = []
    for _ in range(10):
        start = time.time()
        brain_with_chroma.query(query)
        latencies.append(time.time() - start)

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage query latency: {avg_latency:.4f} seconds.")
    assert avg_latency < 1.0  # Expect sub-second latency for simple mock setup


def test_stress_concurrent_queries(brain_with_chroma):
    """Stress test: concurrent queries (simulated)."""
    import concurrent.futures

    # Ingest data
    docs = [
        {
            "id": f"c_{i}",
            "text": f"Concurrent data {i}",
            "metadata": {"source": f"concurrent_{i}.txt"},
        }
        for i in range(50)
    ]
    brain_with_chroma.ingest(docs)

    def run_query():
        return brain_with_chroma.query("find concurrent data")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_query) for _ in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 50
    for res in results:
        assert "Mocked LLM Response" in str(res)
