"""
tests/test_stress.py

Stress tests — high volume ingestion and concurrent query load.

Run in isolation:
    pytest tests/test_stress.py -m stress -v -s
"""

import concurrent.futures
import random
import time

import pytest

from axon.main import AxonBrain, AxonConfig

pytestmark = pytest.mark.stress

TOPICS = [
    "machine learning",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "data engineering",
    "distributed systems",
    "cryptography",
    "quantum computing",
    "bioinformatics",
    "robotics",
]

SENTENCES = [
    "{topic} is a rapidly evolving field with many practical applications.",
    "Recent advances in {topic} have enabled new breakthroughs in industry.",
    "Researchers in {topic} focus on scalability, accuracy, and efficiency.",
    "The core algorithms in {topic} were developed over several decades.",
    "Modern {topic} systems rely heavily on GPU acceleration.",
    "Open-source frameworks have democratized access to {topic} tools.",
    "Industry adoption of {topic} has grown significantly since 2020.",
    "Benchmark datasets for {topic} are maintained by academic institutions.",
    "Transfer learning in {topic} reduces the need for large labeled datasets.",
    "Interpretability and fairness are growing concerns in {topic}.",
]


def _make_brain(tmp_path, **overrides) -> AxonBrain:
    config = AxonConfig(
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        vector_store="chroma",
        vector_store_path=str(tmp_path / "chroma"),
        bm25_path=str(tmp_path / "bm25"),
        hybrid_search=True,
        rerank=False,
        raptor=False,
        graph_rag=False,
        similarity_threshold=0.0,
        top_k=5,
        **overrides,
    )
    return AxonBrain(config)


def _generate_docs(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        topic = rng.choice(TOPICS)
        sentences = rng.sample(SENTENCES, k=rng.randint(2, 5))
        text = " ".join(s.format(topic=topic) for s in sentences)
        docs.append(
            {
                "id": f"doc_{i:04d}",
                "text": text,
                "metadata": {"topic": topic, "index": i},
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Stress 1: 100-document ingestion
# ---------------------------------------------------------------------------


class TestStress_BulkIngest:
    def test_ingest_100_docs_no_crash(self, tmp_path):
        brain = _make_brain(tmp_path)
        docs = _generate_docs(100)
        t0 = time.time()
        brain.ingest(docs)
        elapsed = time.time() - t0
        print(f"\n[stress] 100-doc ingest: {elapsed:.2f}s  ({100/elapsed:.1f} docs/s)")
        brain.close()

    def test_ingest_200_docs_completes(self, tmp_path):
        brain = _make_brain(tmp_path)
        docs = _generate_docs(200)
        brain.ingest(docs)
        collection_info = brain.vector_store.list_documents()
        total_chunks = sum(d["chunks"] for d in collection_info)
        assert total_chunks >= 200, f"Expected ≥200 chunks, got {total_chunks}"
        brain.close()

    def test_ingest_large_text_doc(self, tmp_path):
        """Single large document (~50 KB) should split into multiple chunks without crashing."""
        brain = _make_brain(tmp_path)
        big_text = " ".join(f"Sentence {i} about topic {i % 20}." for i in range(3000))
        docs = [{"id": "big_doc", "text": big_text, "metadata": {}}]
        brain.ingest(docs)
        results = brain._execute_retrieval("topic sentence")
        assert len(results["results"]) > 0
        brain.close()

    def test_ingest_many_small_docs(self, tmp_path):
        """100 single-sentence documents should all be retrievable."""
        brain = _make_brain(tmp_path)
        docs = [
            {"id": f"tiny_{i}", "text": f"Fact number {i}: the sky is blue.", "metadata": {}}
            for i in range(100)
        ]
        brain.ingest(docs)
        results = brain._execute_retrieval("sky blue")
        assert len(results["results"]) > 0
        brain.close()


# ---------------------------------------------------------------------------
# Stress 2: 50 concurrent retrieval requests
# ---------------------------------------------------------------------------


class TestStress_ConcurrentQueries:
    @pytest.fixture
    def loaded_brain(self, tmp_path):
        brain = _make_brain(tmp_path, max_workers=8)
        brain.ingest(_generate_docs(50))
        yield brain
        brain.close()

    def test_50_concurrent_queries_no_crash(self, loaded_brain):
        queries = [f"what is {topic}?" for topic in TOPICS * 5]  # 50 queries

        errors = []
        results_list = []

        def run_query(q):
            try:
                return loaded_brain._execute_retrieval(q)
            except Exception as e:
                errors.append(str(e))
                return None

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(run_query, q) for q in queries]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r is not None:
                    results_list.append(r)
        elapsed = time.time() - t0

        print(f"\n[stress] 50 concurrent queries: {elapsed:.2f}s  ({50/elapsed:.1f} q/s)")
        assert len(errors) == 0, f"Errors during concurrent queries: {errors}"
        assert len(results_list) == 50

    def test_concurrent_queries_all_return_results(self, loaded_brain):
        queries = [f"advances in {t}" for t in TOPICS]  # 10 queries

        results_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(loaded_brain._execute_retrieval, q) for q in queries]
            for f in concurrent.futures.as_completed(futures):
                results_list.append(f.result())

        for r in results_list:
            assert len(r["results"]) > 0, "Some concurrent query returned empty results"


# ---------------------------------------------------------------------------
# Stress 3: Repeated ingest + query interleaved
# ---------------------------------------------------------------------------


class TestStress_IngestQueryInterleaved:
    def test_ingest_query_10_rounds(self, tmp_path):
        brain = _make_brain(tmp_path)
        errors = []

        for round_num in range(10):
            batch = _generate_docs(10, seed=round_num)
            try:
                brain.ingest(batch)
                brain._execute_retrieval(f"advances in {TOPICS[round_num % len(TOPICS)]}")
            except Exception as e:
                errors.append(f"Round {round_num}: {e}")

        assert len(errors) == 0, f"Errors during interleaved stress: {errors}"
        brain.close()


# ---------------------------------------------------------------------------
# Stress 4: Dedup under load
# ---------------------------------------------------------------------------


class TestStress_DeduplicationUnderLoad:
    def test_duplicate_docs_skipped_in_bulk(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._ingested_hashes = set()

        from unittest.mock import MagicMock

        brain._save_hash_store = MagicMock()

        docs = _generate_docs(20)
        brain.ingest(docs)

        # Re-ingest identical docs — should skip all
        initial_add_count = brain.vector_store.list_documents()
        brain.ingest(docs)
        final_add_count = brain.vector_store.list_documents()

        initial_chunks = sum(d["chunks"] for d in initial_add_count)
        final_chunks = sum(d["chunks"] for d in final_add_count)
        assert (
            initial_chunks == final_chunks
        ), f"Chunk count changed after re-ingest: {initial_chunks} → {final_chunks}"
        brain.close()
