import json
import time
from pathlib import Path
from unittest.mock import patch

from axon.main import AxonBrain, AxonConfig


def _make_brain(tmp_path, sentence_window=False):
    config = AxonConfig(
        vector_store_path=str(tmp_path / "vs"),
        bm25_path=str(tmp_path / "bm25"),
        projects_root=str(tmp_path / "projects"),
        sentence_window=sentence_window,
        sentence_window_size=2,
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
    )
    # Mock embeddings to be fast and deterministic
    with patch("axon.main.OpenEmbedding") as MockEmb, patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenReranker"
    ):
        mock_emb = MockEmb.return_value

        # Return random but stable embeddings
        def embed_side_effect(texts):
            return [[0.1] * 384 for _ in texts]

        mock_emb.embed.side_effect = embed_side_effect
        mock_emb.embed_query.return_value = [0.1] * 384

        brain = AxonBrain(config)
        return brain


def run_benchmark(tmp_path):
    prose_data = [
        {
            "id": f"doc_{i}",
            "text": ". ".join([f"This is sentence {j} of document {i}" for j in range(20)]) + ".",
            "metadata": {"source": f"file_{i}.txt", "source_class": "prose"},
        }
        for i in range(5)
    ]

    results = {}

    # 1. Benchmark Standard Chunk Retrieval
    brain_std = _make_brain(tmp_path / "std", sentence_window=False)
    t0 = time.time()
    brain_std.ingest(prose_data)
    ingest_std = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        brain_std._execute_retrieval("sentence 5 document 2")
    query_std = (time.time() - t0) / 10

    results["standard"] = {
        "ingest_time_sec": ingest_std,
        "query_latency_avg_sec": query_std,
        "index_size_sentences": 0,
    }
    brain_std.close()

    # 2. Benchmark Sentence-Window Retrieval
    brain_sw = _make_brain(tmp_path / "sw", sentence_window=True)
    t0 = time.time()
    brain_sw.ingest(prose_data)
    ingest_sw = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        brain_sw._execute_retrieval("sentence 5 document 2")
    query_sw = (time.time() - t0) / 10

    results["sentence_window"] = {
        "ingest_time_sec": ingest_sw,
        "query_latency_avg_sec": query_sw,
        "index_size_sentences": len(brain_sw._sw_index),
    }
    brain_sw.close()

    print("\n=== Sentence-Window Benchmark Results ===")
    print(json.dumps(results, indent=2))

    # Save results to a reproducible format
    output_path = Path("Qualification/GeminiQual/sentence_window_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nBenchmark results saved to {output_path}")


if __name__ == "__main__":
    # Simple tmp_path for standalone run
    tmp = Path(".test_tmp_bench")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir()
    run_benchmark(tmp)
