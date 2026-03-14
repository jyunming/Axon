"""
tests/test_performance.py

Performance (latency) benchmarks.

Run in isolation:
    pytest tests/test_performance.py -m perf -v -s

All thresholds are intentionally conservative to avoid flakiness on slow CI.
Adjust INGEST_BUDGET_S and QUERY_BUDGET_S to your hardware's baseline.
"""

import random
import time

import pytest

from axon.main import AxonBrain, AxonConfig

pytestmark = pytest.mark.perf

# ---- tuneable thresholds ----
INGEST_BUDGET_S = 60.0  # 50 docs must ingest in under this many seconds
QUERY_BUDGET_S = 5.0  # single retrieval must complete in under this many seconds
BATCH_QUERY_BUDGET_S = 30.0  # 20 sequential queries must finish under this budget


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


TOPICS = [
    "machine learning",
    "natural language processing",
    "computer vision",
    "quantum computing",
    "distributed systems",
    "cryptography",
    "bioinformatics",
    "robotics",
    "data engineering",
    "reinforcement learning",
]

TEMPLATES = [
    "{t} relies on statistical methods and large-scale computation.",
    "The study of {t} has accelerated due to open-source frameworks.",
    "Practitioners in {t} require strong mathematical foundations.",
    "Applications of {t} span healthcare, finance, and education.",
    "Advances in {t} are published at top-tier research venues annually.",
]


def _docs(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(TOPICS)
        text = " ".join(rng.choice(TEMPLATES).format(t=t) for _ in range(3))
        out.append({"id": f"perf_{i:04d}", "text": text, "metadata": {"topic": t}})
    return out


# ---------------------------------------------------------------------------
# Perf 1: Ingest throughput
# ---------------------------------------------------------------------------


class TestPerf_IngestThroughput:
    def test_50_docs_ingest_within_budget(self, tmp_path):
        brain = _make_brain(tmp_path)
        docs = _docs(50)

        t0 = time.perf_counter()
        brain.ingest(docs)
        elapsed = time.perf_counter() - t0

        rate = 50 / elapsed
        print(f"\n[perf] ingest 50 docs: {elapsed:.2f}s  ->  {rate:.1f} docs/s")
        assert (
            elapsed < INGEST_BUDGET_S
        ), f"Ingest too slow: {elapsed:.2f}s (budget: {INGEST_BUDGET_S}s)"
        brain.close()

    def test_ingest_rate_reported(self, tmp_path, capsys):
        brain = _make_brain(tmp_path)
        docs = _docs(20)
        t0 = time.perf_counter()
        brain.ingest(docs)
        elapsed = time.perf_counter() - t0
        print(f"\n[perf] ingest 20 docs: {elapsed:.2f}s  ->  {20/elapsed:.1f} docs/s")
        brain.close()


# ---------------------------------------------------------------------------
# Perf 2: Single-query latency
# ---------------------------------------------------------------------------


class TestPerf_QueryLatency:
    @pytest.fixture
    def loaded_brain(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.ingest(_docs(50))
        yield brain
        brain.close()

    def test_single_query_within_budget(self, loaded_brain):
        t0 = time.perf_counter()
        results = loaded_brain._execute_retrieval("what is machine learning?")
        elapsed = time.perf_counter() - t0
        print(f"\n[perf] single query latency: {elapsed*1000:.1f}ms")
        assert (
            elapsed < QUERY_BUDGET_S
        ), f"Query too slow: {elapsed:.2f}s (budget: {QUERY_BUDGET_S}s)"
        assert len(results["results"]) > 0

    def test_20_sequential_queries_within_budget(self, loaded_brain):
        queries = [f"explain {t}" for t in TOPICS * 2]

        t0 = time.perf_counter()
        for q in queries:
            loaded_brain._execute_retrieval(q)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / len(queries)) * 1000
        print(f"\n[perf] 20 sequential queries: {elapsed:.2f}s total  ->  {avg_ms:.1f}ms avg")
        assert (
            elapsed < BATCH_QUERY_BUDGET_S
        ), f"20 queries too slow: {elapsed:.2f}s (budget: {BATCH_QUERY_BUDGET_S}s)"


# ---------------------------------------------------------------------------
# Perf 3: Weighted vs RRF latency comparison
# ---------------------------------------------------------------------------


class TestPerf_FusionModeLatency:
    def test_weighted_vs_rrf_latency(self, tmp_path):
        docs = _docs(30)

        brain_w = _make_brain(tmp_path / "w", hybrid_mode="weighted")
        brain_w.ingest(docs)

        brain_r = _make_brain(tmp_path / "r", hybrid_mode="rrf")
        brain_r.ingest(docs)

        query = "neural network optimization techniques"

        # Warm-up
        brain_w._execute_retrieval(query)
        brain_r._execute_retrieval(query)

        # Measure 5 runs each
        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            brain_w._execute_retrieval(query)
        weighted_ms = (time.perf_counter() - t0) / n * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            brain_r._execute_retrieval(query)
        rrf_ms = (time.perf_counter() - t0) / n * 1000

        print(f"\n[perf] weighted avg: {weighted_ms:.1f}ms  |  rrf avg: {rrf_ms:.1f}ms")

        # Both must be well within budget
        assert weighted_ms < QUERY_BUDGET_S * 1000
        assert rrf_ms < QUERY_BUDGET_S * 1000

        brain_w.close()
        brain_r.close()


# ---------------------------------------------------------------------------
# Perf 4: Dataset type detection is fast (no significant overhead)
# ---------------------------------------------------------------------------


class TestPerf_DatasetTypeDetection:
    def test_detection_1000_docs_under_1s(self, tmp_path):
        brain = _make_brain(tmp_path)
        rng = random.Random(0)
        docs = []
        for i in range(1000):
            texts = [
                "def foo(): pass\nclass Bar:\n    x = 1\n",
                "name,age,city\nAlice,30,NYC\nBob,25,LA\n",
                "# Introduction\n\nStep 1: do this.\n**Note:** important.\n",
                "Abstract\nThis paper introduces. Introduction\nWe study. DOI: x.\n",
                '{"role": "user", "content": "hi"}',
                "The quick brown fox jumps over the lazy dog. " * 5,
            ]
            docs.append(
                {
                    "id": f"d{i}",
                    "text": rng.choice(texts),
                    "metadata": {"source": f"file_{rng.randint(0,5)}.py"},
                }
            )

        t0 = time.perf_counter()
        for doc in docs:
            brain._detect_dataset_type(doc)
        elapsed = time.perf_counter() - t0

        per_doc_us = elapsed / 1000 * 1e6
        print(
            f"\n[perf] type detection: 1000 docs in {elapsed*1000:.1f}ms  ->  {per_doc_us:.0f}us/doc"
        )
        assert elapsed < 1.0, f"Type detection too slow: {elapsed:.3f}s for 1000 docs"
        brain.close()
